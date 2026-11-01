"""Explicitly download an official SAM 2.1 checkpoint; inference itself stays offline."""

import argparse
import hashlib
import logging
import os
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from fishtrack.sam2_backend import MODEL_NAMES, model_spec

LOGGER = logging.getLogger(__name__)


def download_checkpoint(model: str, directory: Path) -> Path:
    """Download official model weights without overwriting an existing file.

    :param model: Supported SAM 2.1 model size.
    :param directory: Local checkpoint directory, created if necessary.
    :return: Existing or atomically published checkpoint path.
    :raises OSError: For failed, incomplete or invalid downloads.
    """
    specification = model_spec(model)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / specification.filename

    # Reuse an existing archive and refuse to replace an invalid file silently.
    if target.exists():
        if not target.is_file() or not zipfile.is_zipfile(target):
            raise OSError(f"Existing checkpoint is not a valid archive: {target}")
        LOGGER.info("Checkpoint already present: %s", target)
        return target

    # Download beside the destination so incomplete weights never appear under the final name.
    descriptor, temporary_name = tempfile.mkstemp(prefix=".download-", dir=directory)
    temporary = Path(temporary_name)
    try:
        LOGGER.info("Downloading %s", specification.url)
        with os.fdopen(descriptor, "wb") as output:
            with urllib.request.urlopen(specification.url, timeout=60) as response:
                advertised = response.headers.get("Content-Length")
                received = 0
                digest = hashlib.sha256()

                # Stream in bounded chunks while measuring the transfer and recording its digest.
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)

        # Check transfer length when supplied, then verify the expected archive format.
        if advertised is not None and received != int(advertised):
            raise OSError("Checkpoint download was incomplete")
        if not zipfile.is_zipfile(temporary):
            raise OSError("Downloaded checkpoint is not a valid PyTorch archive")

        # Publish without overwriting a checkpoint another setup process may have created.
        os.link(temporary, target)
        LOGGER.info("Downloaded %.1f MB; SHA-256 %s", received / 1_000_000, digest.hexdigest())
        return target
    finally:

        # Remove staging files on both success and failure.
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    """Download an explicitly requested checkpoint and return a shell exit status.

    :param argv: Optional explicit argument list.
    :return: Zero on success and one on a download failure.
    """

    # Resolve the requested model explicitly before performing any network transfer.
    parser = argparse.ArgumentParser(description="Download official SAM 2.1 weights")
    parser.add_argument("--model", choices=tuple(MODEL_NAMES), default="tiny")
    parser.add_argument("--directory", type=Path, default=Path("models"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        download_checkpoint(args.model, args.directory)
    except (OSError, urllib.error.URLError) as error:
        LOGGER.error("Cannot download checkpoint: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
