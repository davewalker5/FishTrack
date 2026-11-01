"""Explicitly download a pinned official Grounding DINO model for offline ROI inference."""

import argparse
import json
import logging
import tempfile
from pathlib import Path

from fishtrack.grounding import MODEL_ID, MODEL_REVISION

LOGGER = logging.getLogger(__name__)
REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
)


def download_grounding(directory: Path) -> Path:
    """Download a complete local snapshot without replacing an existing directory.

    :param directory: New model directory, or a complete previously downloaded snapshot.
    :return: Local model directory suitable for offline Transformers loading.
    :raises OSError: If dependencies, network transfer or snapshot validation fail.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise OSError(
            "Run scripts/make-venv.sh to install the Grounding DINO dependencies"
        ) from error

    # Reuse complete snapshots, but never overwrite an incomplete or user-owned directory.
    if directory.exists():
        if all((directory / filename).is_file() for filename in REQUIRED_FILES):
            LOGGER.info("Grounding DINO snapshot already present: %s", directory)
            return directory
        raise OSError(
            f"Existing model directory is incomplete; choose a new directory: {directory}"
        )

    # Stage downloads beside the destination so publication stays on the same filesystem.
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".grounding-download-", dir=directory.parent
    ) as temporary:
        snapshot = Path(temporary) / "model"
        LOGGER.info("Downloading %s at %s", MODEL_ID, MODEL_REVISION)
        try:

            # Fetch only the pinned model assets, excluding alternative pickle weights.
            snapshot_download(
                repo_id=MODEL_ID,
                revision=MODEL_REVISION,
                local_dir=snapshot,
                cache_dir=directory.parent / ".grounding-cache",
                allow_patterns=[*REQUIRED_FILES, "added_tokens.json", "README.md"],
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise OSError(f"Cannot download Grounding DINO: {error}") from error
        if not all((snapshot / filename).is_file() for filename in REQUIRED_FILES):
            raise OSError("Downloaded Grounding DINO snapshot is incomplete")

        # Record provenance before exposing the completed snapshot to inference.
        (snapshot / "fishtrack-model.json").write_text(
            json.dumps({"model_id": MODEL_ID, "revision": MODEL_REVISION}, indent=2) + "\n"
        )
        snapshot.rename(directory)
    LOGGER.info("Grounding DINO is ready: %s", directory)
    return directory


def main(argv: list[str] | None = None) -> int:
    """Download the pinned model and return a shell status.

    :param argv: Optional explicit command-line arguments.
    :return: Zero on success, one on setup failure.
    """

    # Keep downloads an explicit setup operation rather than a side effect of inference.
    parser = argparse.ArgumentParser(description="Download the local Grounding DINO Tiny model")
    parser.add_argument("--directory", type=Path, default=Path("models/grounding-dino-tiny"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        download_grounding(args.directory)
    except OSError as error:
        LOGGER.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
