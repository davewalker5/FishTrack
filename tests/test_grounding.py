"""Offline proposal filtering, first-frame review and SAM 2 hand-off tests."""

import json
from pathlib import Path

import numpy as np
import pytest

from fishtrack import grounding, grounding_setup, video
from fishtrack.grounding import RegionProposal, filter_proposals
from fishtrack.selection import region_at_point
from fishtrack.video import ProcessingConfig, prepare_initial_frame


def settings(**overrides):
    """Build a valid reviewed automatic-ROI SAM 2 experiment."""
    values = dict(backend="sam2", auto_roi=True, max_frames=3)
    values.update(overrides)
    return ProcessingConfig(**values)


def test_proposal_clamping_nms_and_stable_order():
    """Keep high-scoring independent boxes while rejecting duplicates and invalid regions."""
    boxes = np.array(
        [
            [10.2, 20.3, 40.1, 60.9],
            [11, 21, 41, 61],
            [-5, 70, 50, 120],
            [1, 1, 3, 3],
            [10, 20, np.nan, 50],
        ],
        dtype=float,
    )
    scores = np.array([0.9, 0.8, 0.7, 0.99, 0.95])
    proposals = filter_proposals(boxes, scores, ["fish"] * 5, 100, 100, 10)
    assert [proposal.box for proposal in proposals] == [(10, 20, 31, 41), (0, 70, 50, 30)]
    assert len(filter_proposals(boxes, scores, ["fish"] * 5, 100, 100, 1)) == 1
    with pytest.raises(ValueError):
        filter_proposals(boxes, scores, ["fish"], 100, 100, 10)


def test_review_removes_selected_overlapping_box():
    """A click removes the smallest containing region, permitting review of nested proposals."""
    boxes = [(10, 10, 60, 60), (20, 20, 20, 20), (80, 80, 10, 10)]
    assert region_at_point(boxes, (25, 25)) == 1
    assert region_at_point(boxes, (15, 15)) == 0
    assert region_at_point(boxes, (95, 95)) is None


def test_only_reviewed_regions_are_accepted(tmp_path, monkeypatch):
    """Initial proposals populate review; removed proposals never reach SAM 2 prompts."""
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    proposals = [
        RegionProposal((10, 20, 20, 30), 0.8, "fish"),
        RegionProposal((60, 40, 20, 20), 0.7, "fish"),
    ]
    monkeypatch.setattr(grounding, "propose_regions", lambda frame, config: proposals)
    seen = []

    def review(image, initial_regions):
        """Remove the first proposal and add a manually selected missed fish."""
        seen.append(initial_regions)
        return (initial_regions[1], (90, 50, 15, 20))

    monkeypatch.setattr(video, "select_regions", review)
    accepted = prepare_initial_frame(frame, settings(), tmp_path)
    assert seen == [((10, 20, 20, 30), (60, 40, 20, 20))]
    assert accepted == ((60, 40, 20, 20), (90, 50, 15, 20))
    assert json.loads((tmp_path / "selected-rois.json").read_text()) == [
        list(box) for box in accepted
    ]
    assert len(json.loads((tmp_path / "roi-proposals.json").read_text())) == 2
    assert (tmp_path / "roi-proposals.jpg").exists()


def test_explicit_headless_acceptance(monkeypatch):
    """Skipping review requires an explicit option and preserves detector box order."""
    monkeypatch.setattr(
        grounding,
        "propose_regions",
        lambda frame, config: [RegionProposal((10, 20, 20, 30), 0.8, "fish")],
    )

    def forbidden(*args, **kwargs):
        """Reject an unexpected attempt to open a window."""
        raise AssertionError("Review window should not open")

    monkeypatch.setattr(video, "select_regions", forbidden)
    assert prepare_initial_frame(
        np.zeros((100, 120, 3), np.uint8), settings(accept_auto_roi=True)
    ) == ((10, 20, 20, 30),)


def test_empty_proposals_allow_manual_selection(monkeypatch):
    """A failed detector must still let the user add fish rather than aborting reviewed mode."""
    monkeypatch.setattr(grounding, "propose_regions", lambda frame, config: [])
    monkeypatch.setattr(video, "select_regions", lambda frame: ((10, 20, 20, 30),))
    frame = np.zeros((100, 120, 3), np.uint8)
    assert prepare_initial_frame(frame, settings()) == ((10, 20, 20, 30),)
    with pytest.raises(ValueError, match="No fish regions"):
        prepare_initial_frame(frame, settings(accept_auto_roi=True))


@pytest.mark.parametrize(
    "overrides",
    [
        dict(grounding_threshold=0),
        dict(appearance=False),
        dict(grounding_threshold=float("nan")),
        dict(grounding_text_threshold=1.1),
        dict(max_auto_rois=0),
        dict(grounding_prompt=" . "),
        dict(accept_auto_roi=True, select_roi=True),
    ],
)
def test_invalid_grounding_configuration(overrides):
    """Reject invalid or contradictory automatic-acquisition settings before inference."""
    with pytest.raises(ValueError):
        settings(**overrides)


def test_acceptance_flag_requires_automatic_proposals():
    """An acceptance flag alone cannot silently bypass manual selection."""
    with pytest.raises(ValueError, match="requires --auto-roi"):
        settings(auto_roi=False, accept_auto_roi=True, rois=((10, 20, 20, 30),))


def test_missing_grounding_weights_are_actionable(tmp_path):
    """Missing weights fail locally without attempting a network download."""
    with pytest.raises(OSError, match="setup-grounding"):
        grounding.propose_regions(
            np.zeros((100, 120, 3), np.uint8), settings(grounding_model=tmp_path / "missing")
        )


def test_pinned_snapshot_download_and_reuse(tmp_path, monkeypatch):
    """Download only safe model/config files at the pinned revision and reuse them offline."""
    import sys
    from types import SimpleNamespace

    calls = []

    def snapshot_download(**kwargs):
        """Create an expected local snapshot instead of using the network."""
        calls.append(kwargs)
        directory = Path(kwargs["local_dir"])
        directory.mkdir()
        for filename in grounding_setup.REQUIRED_FILES:
            (directory / filename).write_text("test")

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download)
    )
    target = tmp_path / "model"
    assert grounding_setup.download_grounding(target) == target
    assert calls[0]["revision"] == grounding.MODEL_REVISION
    assert "pytorch_model.bin" not in calls[0]["allow_patterns"]
    assert grounding_setup.download_grounding(target) == target
    assert len(calls) == 1
    assert (
        json.loads((target / "fishtrack-model.json").read_text())["revision"]
        == grounding.MODEL_REVISION
    )


def test_failed_snapshot_never_publishes_directory(tmp_path, monkeypatch):
    """A partial download cannot be mistaken for an installed detector."""
    import sys
    from types import SimpleNamespace

    def failed(**kwargs):
        """Simulate a model-transfer failure."""
        raise OSError("transfer failed")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=failed))
    with pytest.raises(OSError):
        grounding_setup.download_grounding(tmp_path / "model")
    assert not list(tmp_path.iterdir())
