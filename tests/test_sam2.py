"""SAM 2 adapter tests run without neural dependencies, downloaded weights or accelerators."""

import csv
import io
import zipfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from fishtrack import model_setup, sam2_backend
from fishtrack.detection import DetectionConfig
from fishtrack.sam2_backend import annotate_masks, choose_device, model_spec
from fishtrack.video import ProcessingConfig, process_video


def options(**overrides):
    """Return valid selected-object experiment settings with optional overrides."""
    values = dict(backend="sam2", rois=((20, 30, 20, 20),), max_frames=4)
    values.update(overrides)
    return ProcessingConfig(**values)


@pytest.mark.parametrize(
    "settings",
    [
        dict(rois=()),
        dict(duration=None, max_frames=None),
        dict(detect_additional=True),
        dict(sam2_device="unknown"),
        dict(sam2_model="unknown"),
        dict(sam2_min_area=0),
    ],
)
def test_sam2_configuration_validation(settings):
    """Reject incomplete selections and ambiguous experiment configuration."""
    with pytest.raises(ValueError):
        options(**settings)


def test_model_config_checkpoint_pairing():
    """Model sizes map to matching SAM 2.1 configs and official checkpoint filenames."""
    assert model_spec("tiny").config == "configs/sam2.1/sam2.1_hiera_t.yaml"
    assert model_spec("base-plus").filename == "sam2.1_hiera_base_plus.pt"
    assert model_spec("large").url.startswith("https://dl.fbaipublicfiles.com/")
    with pytest.raises(ValueError):
        model_spec("other")


def test_device_selection():
    """Prefer available accelerators but never silently replace an explicit request."""
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
    )
    assert choose_device(torch, "auto") == "mps"
    assert choose_device(torch, "cpu") == "cpu"
    with pytest.raises(OSError, match="unavailable"):
        choose_device(torch, "cuda")


def test_masks_preserve_identity_and_empty_masks_are_not_drawn():
    """Render independent silhouettes while leaving the source image untouched."""
    frame = np.zeros((90, 120, 3), dtype=np.uint8)
    masks = np.zeros((2, 90, 120), dtype=bool)
    masks[0, 20:40, 25:45] = True
    annotated, observations = annotate_masks(frame, [7, 3], masks, options(track_ids=True))
    assert not frame.any() and annotated.any()
    assert observations[0].identifier == 7 and observations[0].box == (25, 20, 20, 20)
    assert observations[1].identifier == 3 and observations[1].status == "no_mask"
    masks[1, 1:3, 1:3] = True
    _, observations = annotate_masks(frame, [7, 3], masks, options())
    assert observations[1].status == "below_min_area"
    with pytest.raises(ValueError, match="dimensions"):
        annotate_masks(frame, [1], masks, options())


class ArrayTensor:
    """Minimal tensor conversion interface returned by the injected predictor."""

    def __init__(self, array):
        """Store a NumPy-backed tensor value."""
        self.array = array

    def __gt__(self, value):
        """Return thresholded mask values."""
        return ArrayTensor(self.array > value)

    def detach(self):
        """Represent detachment without autograd."""
        return self

    def cpu(self):
        """Represent a CPU tensor."""
        return self

    def numpy(self):
        """Return the backing array."""
        return self.array


class PredictableVideoPredictor:
    """Produce known masks while recording the actual video-adapter API calls."""

    def __init__(self, truncate=False, wrong_order=False):
        """Configure a normal or deliberately incomplete propagation."""
        self.truncate = truncate
        self.wrong_order = wrong_order
        self.prompts = []
        self.directory = None

    def init_state(self, video_path, **kwargs):
        """Inspect staged JPEGs and verify explicit CPU memory offloading."""
        assert kwargs["offload_video_to_cpu"] and kwargs["offload_state_to_cpu"]
        self.directory = Path(video_path)
        return {"count": len(list(self.directory.glob("*.jpg")))}

    def add_new_points_or_box(self, **kwargs):
        """Capture full-resolution first-frame XYXY prompts."""
        self.prompts.append(kwargs)

    def propagate_in_video(self, state):
        """Yield one mask per prompted object on every staged frame."""
        count = state["count"] - int(self.truncate)
        for index in range(count):
            mask = np.full((len(self.prompts), 1, 90, 120), -1.0, dtype=np.float32)
            mask[:, :, 30:50, 20 + index : 40 + index] = 1
            yield (
                index + int(self.wrong_order),
                list(range(1, len(self.prompts) + 1)),
                ArrayTensor(mask),
            )


def make_source(path):
    """Encode four short source frames for the common MP4 processing path."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12, (120, 90))
    assert writer.isOpened()
    for index in range(4):
        frame = np.full((90, 120, 3), 40 + index * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def inject_predictor(monkeypatch, predictor):
    """Inject an upstream-shaped predictor without importing optional packages."""
    monkeypatch.setattr(
        sam2_backend,
        "load_predictor",
        lambda settings: (SimpleNamespace(inference_mode=nullcontext), predictor, "cpu"),
    )


def test_sam2_video_integration_and_cleanup(tmp_path, monkeypatch):
    """Export all masks against original images with clip indexing and stable IDs."""
    source, output = tmp_path / "source.mp4", tmp_path / "output.mp4"
    make_source(source)
    original = source.read_bytes()
    predictor = PredictableVideoPredictor()
    inject_predictor(monkeypatch, predictor)
    settings = options(rois=((20, 30, 20, 20), (55, 30, 20, 20)), start=1 / 12, max_frames=2)
    assert process_video(source, output, DetectionConfig(), settings, tmp_path / "debug") == 2
    assert source.read_bytes() == original and not predictor.directory.exists()
    assert len(predictor.prompts) == 2
    assert predictor.prompts[1]["obj_id"] == 2 and predictor.prompts[1]["frame_idx"] == 0
    assert np.array_equal(predictor.prompts[0]["box"], [20, 30, 40, 50])
    with (tmp_path / "debug" / "sam2-statistics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert rows[0]["source_frame"] == "1" and rows[0]["object_id"] == "1"
    assert rows[1]["object_id"] == "2" and rows[-1]["source_frame"] == "2"
    assert len(list((tmp_path / "debug").glob("*-object-*.png"))) == 4
    capture = cv2.VideoCapture(str(output))
    decoded = 0
    while capture.read()[0]:
        decoded += 1
    capture.release()
    assert decoded == 2


@pytest.mark.parametrize("broken", ["truncated", "wrong_order"])
def test_broken_propagation_never_publishes_output(tmp_path, monkeypatch, broken):
    """Missing or out-of-order neural results cannot silently produce an incomplete video."""
    source, output = tmp_path / "source.mp4", tmp_path / "output.mp4"
    make_source(source)
    predictor = PredictableVideoPredictor(
        truncate=broken == "truncated", wrong_order=broken == "wrong_order"
    )
    inject_predictor(monkeypatch, predictor)
    with pytest.raises(OSError):
        process_video(source, output, DetectionConfig(), options())
    assert set(tmp_path.glob("*.mp4")) == {source}
    assert not predictor.directory.exists()


def test_experiment_frame_bound(tmp_path, monkeypatch):
    """Reject oversized experiments without publishing partial output or leaving staged frames."""
    source = tmp_path / "source.mp4"
    make_source(source)
    predictor = PredictableVideoPredictor()
    inject_predictor(monkeypatch, predictor)
    monkeypatch.setattr(sam2_backend, "MAX_EXPERIMENT_FRAMES", 2)
    with pytest.raises(ValueError, match="limited to 2"):
        process_video(source, tmp_path / "output.mp4", DetectionConfig(), options())
    assert set(tmp_path.glob("*.mp4")) == {source}


def test_missing_model_and_optional_dependencies(tmp_path, monkeypatch):
    """Missing setup reports actionable errors instead of importing models on normal runs."""
    import sys

    checkpoint = tmp_path / "model.pt"
    with pytest.raises(OSError, match="Missing SAM 2 checkpoint"):
        sam2_backend.load_predictor(options(sam2_checkpoint=checkpoint))
    checkpoint.write_bytes(b"placeholder")
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(OSError, match="dependencies are missing"):
        sam2_backend.load_predictor(options(sam2_checkpoint=checkpoint))


def checkpoint_bytes():
    """Build a tiny ZIP container to test transport validation without model weights."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("model/data.pkl", b"test checkpoint")
    return buffer.getvalue()


class DownloadResponse(io.BytesIO):
    """A download stream with optional inaccurate HTTP length metadata."""

    def __init__(self, content, extra_length=0):
        """Supply response bytes and length headers."""
        super().__init__(content)
        self.headers = {"Content-Length": str(len(content) + extra_length)}


def test_checkpoint_download_and_no_overwrite(tmp_path, monkeypatch):
    """Publish validated downloads once and reuse existing weights without network access."""
    content = checkpoint_bytes()
    calls = []

    def response(url, timeout):
        """Return an official-URL download stream."""
        calls.append(url)
        return DownloadResponse(content)

    monkeypatch.setattr(model_setup.urllib.request, "urlopen", response)
    target = model_setup.download_checkpoint("tiny", tmp_path)
    assert target.read_bytes() == content
    assert model_setup.download_checkpoint("tiny", tmp_path) == target
    assert len(calls) == 1 and len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize(
    "content, extra_length", [(b"not a checkpoint", 0), (checkpoint_bytes(), 3)]
)
def test_failed_download_is_cleaned_up(tmp_path, monkeypatch, content, extra_length):
    """Invalid or incomplete downloads leave no misleading checkpoint or temporary file."""
    monkeypatch.setattr(
        model_setup.urllib.request,
        "urlopen",
        lambda url, timeout: DownloadResponse(content, extra_length),
    )
    with pytest.raises(OSError):
        model_setup.download_checkpoint("tiny", tmp_path)
    assert not list(tmp_path.iterdir())


def test_reviewed_auto_proposals_seed_only_accepted_objects(tmp_path, monkeypatch):
    """Review edits, not raw detector guesses, determine the actual SAM 2 box prompts."""
    from fishtrack import grounding, video
    from fishtrack.grounding import RegionProposal

    source, output = tmp_path / "source.mp4", tmp_path / "output.mp4"
    make_source(source)
    predictor = PredictableVideoPredictor()
    inject_predictor(monkeypatch, predictor)
    proposed = [
        RegionProposal((20, 30, 20, 20), 0.9, "fish"),
        RegionProposal((55, 30, 20, 20), 0.8, "fish"),
    ]
    monkeypatch.setattr(grounding, "propose_regions", lambda frame, config: proposed)

    def review(frame, initial_regions):
        """Remove the first detector proposal before tracking starts."""
        assert initial_regions == tuple(proposal.box for proposal in proposed)
        return (initial_regions[1],)

    monkeypatch.setattr(video, "select_regions", review)
    config = options(rois=(), auto_roi=True, max_frames=2)
    assert process_video(source, output, DetectionConfig(), config, tmp_path / "debug") == 2
    assert len(predictor.prompts) == 1 and predictor.prompts[0]["obj_id"] == 1
    assert np.array_equal(predictor.prompts[0]["box"], [55, 30, 75, 50])


def test_sam2_cancellation_unwinds_staging_and_partial_output(tmp_path, monkeypatch):
    """GUI cancellation after a neural frame uses the same safe cleanup as other failures."""
    from fishtrack.control import ProcessingCancelled, ProcessingHooks

    source, output = tmp_path / "source.mp4", tmp_path / "output.mp4"
    make_source(source)
    predictor = PredictableVideoPredictor()
    inject_predictor(monkeypatch, predictor)
    cancelled = False

    def progress(completed, total, stage):
        """Cancel after propagation starts, rather than while frames are being staged."""
        nonlocal cancelled
        if completed == 1 and stage == "Tracking fish with SAM 2":
            cancelled = True

    with pytest.raises(ProcessingCancelled):
        process_video(
            source,
            output,
            DetectionConfig(),
            options(),
            hooks=ProcessingHooks(cancelled=lambda: cancelled, progress=progress),
        )
    assert not output.exists() and not predictor.directory.exists()
    assert set(tmp_path.glob("*.mp4")) == {source}
