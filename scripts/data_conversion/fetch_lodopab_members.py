"""Fetch only the two reviewed LoDoPaB DEFLATE prefixes used by the converter.

This maintainer utility uses fixed, bounded HTTP Range requests. It never asks
Zenodo for the remaining multi-gigabyte archives and refuses servers that do
not return the exact requested ``Content-Range``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RemotePrefix:
    name: str
    url: str
    archive_size: int
    start: int
    length: int
    sha256: str


_PREFIXES = (
    RemotePrefix(
        name="ground_truth.prefix",
        url="https://zenodo.org/records/3384092/files/ground_truth_test.zip",
        archive_size=1_582_139_537,
        start=1_538_816_710,
        length=4_194_304,
        sha256="58d517b9cca643e2f9d8df3927752587998e415b9aa2a77f73ff02bf7feb9adf",
    ),
    RemotePrefix(
        name="observation.prefix",
        url="https://zenodo.org/records/3384092/files/observation_test.zip",
        archive_size=2_996_574_366,
        start=2_914_525_998,
        length=8_388_608,
        sha256="6379e14b597d244cdf10e7def44e7f14e9f0edb3c63ed325d96a819295e43c4e",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_prefix(
    prefix: RemotePrefix,
    output_directory: Path,
) -> Path:
    end = prefix.start + prefix.length - 1
    destination = output_directory / prefix.name
    if destination.exists():
        if destination.stat().st_size != prefix.length or _sha256(destination) != prefix.sha256:
            raise RuntimeError(f"existing range has the wrong size: {destination}")
        return destination
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.download")
    if temporary.exists():
        raise RuntimeError(f"stale download file must be reviewed first: {temporary}")
    expected_content_range = f"bytes {prefix.start}-{end}/{prefix.archive_size}"
    for attempt in range(1, 13):
        request = urllib.request.Request(
            prefix.url,
            headers={
                "Range": f"bytes={prefix.start}-{end}",
                "User-Agent": "OpenMedVisionX-maintainer-data-converter/1",
            },
        )
        try:
            with (
                urllib.request.urlopen(request, timeout=60) as response,
                temporary.open("xb") as output,
            ):
                status = getattr(response, "status", None)
                content_range = response.headers.get("Content-Range", "")
                if status != 206 or content_range != expected_content_range:
                    raise RuntimeError("server did not honor the exact bounded HTTP Range request")
                remaining = prefix.length
                while remaining:
                    chunk = response.read(min(1024 * 1024, remaining + 1))
                    if not chunk:
                        break
                    if len(chunk) > remaining:
                        raise RuntimeError("range response exceeded its declared byte limit")
                    output.write(chunk)
                    remaining -= len(chunk)
                if remaining:
                    raise RuntimeError(f"range response ended {remaining:,} bytes early")
            temporary.replace(destination)
            if _sha256(destination) != prefix.sha256:
                destination.unlink(missing_ok=True)
                raise RuntimeError("downloaded prefix SHA-256 does not match the review record")
            return destination
        except (OSError, RuntimeError, urllib.error.URLError):
            temporary.unlink(missing_ok=True)
            if attempt == 12:
                raise
            time.sleep(2)
    raise AssertionError("unreachable")


def fetch_prefixes(output_directory: Path) -> tuple[Path, ...]:
    output_directory.mkdir(parents=True, exist_ok=True)
    return tuple(_download_prefix(prefix, output_directory) for prefix in _PREFIXES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    arguments = parser.parse_args()
    for output in fetch_prefixes(arguments.output_directory):
        print(f"wrote {output} ({output.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
