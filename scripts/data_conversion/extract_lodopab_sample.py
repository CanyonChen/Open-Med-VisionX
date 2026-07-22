"""Build the reviewed LoDoPaB-CT test sample from two bounded ZIP ranges.

The official test archives are multi-gigabyte ZIP files.  This maintainer tool
does not accept or download those archives.  It accepts only the exact raw
DEFLATE prefixes listed below, verifies their byte counts and SHA-256 digests,
and extracts row zero from the final test members.  That row is global test
index 3456 (``27 * 128``).

The output is a deterministic, non-pickle NPZ containing numeric arrays,
canonical JSON metadata, and the exact ASTRA-CUDA filtered back-projection
used as input to the bundled DIVal FBP-U-Net.  All intermediate HDF5 data
stays in a temporary directory.

The FBP parameters come from the fixed supp.dival revision
``6117cabc55d223f5b62ea43d4a40225270fb6756``:
``filter_type=Hann`` and ``frequency_scaling=1.0``.  This is intentionally not
the separately tuned FBP baseline cutoff of ``0.641025641025641``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import tempfile
import zipfile
import zlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np

SAMPLE_INDEX = 3456
MEMBER_ROW = 0
ODL_VERSION = "0.8.3"
ASTRA_VERSION = "2.5.0"
FBP_FILTER_TYPE = "Hann"
FBP_FREQUENCY_SCALING = 1.0
FBP_SHA256 = "2fbfeb49fc11dc239c9c44226ddc2c611bf96c93b4b89f85d6d0fc61105f1b75"
_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
_HDF5_V0_EOF_OFFSET = 40


@dataclass(frozen=True, slots=True)
class SourcePrefix:
    label: str
    archive_name: str
    archive_url: str
    archive_size: int
    archive_md5: str
    member: str
    member_offset: int
    member_compressed_size: int
    member_expanded_size: int
    member_crc32: int
    compressed_data_offset: int
    prefix_size: int
    prefix_sha256: str
    dataset_shape: tuple[int, ...]
    dataset_chunks: tuple[int, ...]
    sample_shape: tuple[int, ...]
    sample_sha256: str


GROUND_TRUTH = SourcePrefix(
    label="ground truth",
    archive_name="ground_truth_test.zip",
    archive_url=("https://zenodo.org/records/3384092/files/ground_truth_test.zip?download=1"),
    archive_size=1_582_139_537,
    archive_md5="ecc655767fbe3d40908ca823921f4c7f",
    member="ground_truth_test_027.hdf5",
    member_offset=1_538_816_654,
    member_compressed_size=43_320_789,
    member_expanded_size=56_387_960,
    member_crc32=0x3EDD9669,
    compressed_data_offset=1_538_816_710,
    prefix_size=4_194_304,
    prefix_sha256="58d517b9cca643e2f9d8df3927752587998e415b9aa2a77f73ff02bf7feb9adf",
    dataset_shape=(128, 362, 362),
    dataset_chunks=(8, 46, 46),
    sample_shape=(362, 362),
    sample_sha256="24392e79235397cf3275588201ab67faf25401f5a9587c8bce8980e1e864aa0b",
)

OBSERVATION = SourcePrefix(
    label="observation",
    archive_name="observation_test.zip",
    archive_url=("https://zenodo.org/records/3384092/files/observation_test.zip?download=1"),
    archive_size=2_996_574_366,
    archive_md5="9ae6b053bb1faa94d573311af8ec67b2",
    member="observation_test_027.hdf5",
    member_offset=2_914_525_943,
    member_compressed_size=82_046_358,
    member_expanded_size=221_594_744,
    member_crc32=0xBE47200C,
    compressed_data_offset=2_914_525_998,
    prefix_size=8_388_608,
    prefix_sha256="6379e14b597d244cdf10e7def44e7f14e9f0edb3c63ed325d96a819295e43c4e",
    dataset_shape=(97, 1000, 513),
    dataset_chunks=(8, 63, 33),
    sample_shape=(1000, 513),
    sample_sha256="b4f20f395d68e9755e0a250a78206351fee1a95b8c4f4ed4e236f40eb12b0be7",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array, dtype="<f4").tobytes(order="C")).hexdigest()


def _inflate_prefix(source: Path, destination: Path, expected: SourcePrefix) -> None:
    """Inflate a verified raw-DEFLATE prefix into a deliberately partial HDF5."""

    if source.stat().st_size != expected.prefix_size:
        raise ValueError(
            f"{expected.label} prefix size changed: expected {expected.prefix_size:,} bytes"
        )
    if _sha256_file(source) != expected.prefix_sha256:
        raise ValueError(f"{expected.label} prefix SHA-256 mismatch")

    decoder = zlib.decompressobj(-zlib.MAX_WBITS)
    with source.open("rb") as compressed, destination.open("wb") as expanded:
        for chunk in iter(lambda: compressed.read(1024 * 1024), b""):
            expanded.write(decoder.decompress(chunk))
        expanded.write(decoder.flush())
    if decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError(f"{expected.label} input is not the reviewed partial DEFLATE stream")


def _patch_partial_hdf5_eof(path: Path, expected: SourcePrefix) -> None:
    """Constrain HDF5's declared EOF to the verified decompressed prefix.

    The released HDF5 files use a version-zero superblock with eight-byte
    addresses.  Only its logical EOF field is changed in the temporary copy;
    dataset metadata and chunk bytes are untouched.
    """

    with path.open("r+b") as stream:
        header = stream.read(56)
        if len(header) != 56 or header[:8] != _HDF5_SIGNATURE:
            raise ValueError(f"{expected.label} prefix is not an HDF5 file")
        if header[8] != 0 or header[13] != 8 or header[14] != 8:
            raise ValueError(f"{expected.label} HDF5 superblock contract changed")
        declared_eof = struct.unpack_from("<Q", header, _HDF5_V0_EOF_OFFSET)[0]
        if declared_eof != expected.member_expanded_size:
            raise ValueError(f"{expected.label} expanded-size metadata changed")
        actual_eof = path.stat().st_size
        stream.seek(_HDF5_V0_EOF_OFFSET)
        stream.write(struct.pack("<Q", actual_eof))


def _read_sample(prefix_path: Path, expected: SourcePrefix) -> np.ndarray:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("This maintainer conversion requires h5py") from exc

    with tempfile.TemporaryDirectory(prefix="openmedvisionx-lodopab-") as directory:
        partial_hdf5 = Path(directory) / expected.member
        _inflate_prefix(prefix_path, partial_hdf5, expected)
        _patch_partial_hdf5_eof(partial_hdf5, expected)
        with h5py.File(partial_hdf5, "r") as source:
            if set(source.keys()) != {"data"}:
                raise ValueError(f"{expected.label} HDF5 entries changed")
            dataset = source["data"]
            if tuple(dataset.shape) != expected.dataset_shape:
                raise ValueError(f"{expected.label} HDF5 shape changed")
            if tuple(dataset.chunks or ()) != expected.dataset_chunks:
                raise ValueError(f"{expected.label} HDF5 chunk layout changed")
            if dataset.dtype != np.dtype("float32") or dataset.compression is not None:
                raise ValueError(f"{expected.label} HDF5 dtype or compression changed")
            sample = np.asarray(dataset[MEMBER_ROW], dtype="<f4")

    if sample.shape != expected.sample_shape or not np.all(np.isfinite(sample)):
        raise ValueError(f"{expected.label} sample shape or values are invalid")
    if _array_sha256(sample) != expected.sample_sha256:
        raise ValueError(f"{expected.label} extracted-array SHA-256 mismatch")
    return np.ascontiguousarray(sample)


def _npy_payload(array: np.ndarray) -> bytes:
    output = BytesIO()
    np.lib.format.write_array(output, np.ascontiguousarray(array), allow_pickle=False)
    return output.getvalue()


def _compute_dival_fbp(observation: np.ndarray) -> np.ndarray:
    """Apply the reviewed DIVal FBP-U-Net preprocessing operator.

    ODL and ASTRA are maintainer-only conversion dependencies.  The generated
    FBP is bundled so application users do not need either package at runtime.
    """

    try:
        import astra
        import odl
    except ImportError as exc:
        raise RuntimeError(
            "LoDoPaB conversion requires odl==0.8.3 and astra-toolbox==2.5.0"
        ) from exc
    if odl.__version__ != ODL_VERSION or astra.__version__ != ASTRA_VERSION:
        raise RuntimeError(
            "LoDoPaB FBP conversion requires exactly "
            f"odl=={ODL_VERSION} and astra-toolbox=={ASTRA_VERSION}"
        )
    if not astra.use_cuda():
        raise RuntimeError("the reviewed LoDoPaB FBP conversion requires ASTRA CUDA")

    domain = odl.uniform_discr(
        [-0.13, -0.13],
        [0.13, 0.13],
        (362, 362),
        dtype=np.float32,
    )
    geometry = odl.tomo.parallel_beam_geometry(
        domain,
        num_angles=1000,
        det_shape=(513,),
    )
    ray_transform = odl.tomo.RayTransform(domain, geometry, impl="astra_cuda")
    fbp_operator = odl.tomo.fbp_op(
        ray_transform,
        padding=True,
        filter_type=FBP_FILTER_TYPE,
        frequency_scaling=FBP_FREQUENCY_SCALING,
    )
    fbp = np.ascontiguousarray(np.asarray(fbp_operator(observation), dtype="<f4"))
    if fbp.shape != (362, 362) or not np.all(np.isfinite(fbp)):
        raise ValueError("derived FBP shape or values are invalid")
    if _array_sha256(fbp) != FBP_SHA256:
        raise ValueError(
            "derived FBP hash changed; review ODL, ASTRA, CUDA, and hardware before release"
        )
    return fbp


def _write_deterministic_npz(
    output_path: Path,
    *,
    observation: np.ndarray,
    fbp: np.ndarray,
    ground_truth: np.ndarray,
    metadata: dict[str, object],
) -> None:
    canonical_metadata = json.dumps(
        metadata,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    entries = {
        "fbp.npy": _npy_payload(fbp),
        "ground_truth.npy": _npy_payload(ground_truth),
        "metadata_json.npy": _npy_payload(np.frombuffer(canonical_metadata, dtype=np.uint8)),
        "observation.npy": _npy_payload(observation),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            allowZip64=False,
        ) as archive:
            for name in sorted(entries):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o600 << 16
                archive.writestr(info, entries[name], compresslevel=9)
        temporary.replace(output_path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def extract_sample(
    ground_truth_prefix: Path,
    observation_prefix: Path,
    output_path: Path,
) -> str:
    ground_truth = _read_sample(ground_truth_prefix, GROUND_TRUTH)
    observation = _read_sample(observation_prefix, OBSERVATION)
    fbp = _compute_dival_fbp(observation)
    metadata: dict[str, object] = {
        "schema_version": 1,
        "case_id": "lodopab-ct-test-03456",
        "dataset": "LoDoPaB-CT",
        "dataset_version": "1.0.0",
        "doi": "10.5281/zenodo.3384092",
        "license": "CC-BY-4.0",
        "split": "test",
        "sample_index": SAMPLE_INDEX,
        "source_member_row": MEMBER_ROW,
        "attribution": (
            "Leuschner, Johannes; Schmidt, Maximilian; Otero Baguer, Daniel. "
            "LoDoPaB-CT Dataset (2019)."
        ),
        "citation_doi": "10.1038/s41597-021-00893-z",
        "contains_patient_identifiers": False,
        "contains_dicom_metadata": False,
        "intended_use": "education-and-research-only",
        "observation_semantics": "simulated low-dose parallel-beam line-integral data",
        "fbp_semantics": (
            "DIVal FBP-U-Net input; ODL 0.8.3 and ASTRA CUDA 2.5.0; "
            "Hann filter; frequency_scaling=1.0; padding enabled"
        ),
        "ground_truth_semantics": "normalized attenuation image; not clinical HU",
        "array_sha256": {
            "fbp": FBP_SHA256,
            "ground_truth": GROUND_TRUTH.sample_sha256,
            "observation": OBSERVATION.sample_sha256,
        },
        "transformations": [
            "selected test member 027 row 0 (global index 3456)",
            "cast arrays to little-endian float32",
            (
                "computed DIVal FBP-U-Net input with ODL 0.8.3, ASTRA CUDA "
                "2.5.0, Hann filter, frequency_scaling=1.0, and padding"
            ),
        ],
    }
    _write_deterministic_npz(
        output_path,
        observation=observation,
        fbp=fbp,
        ground_truth=ground_truth,
        metadata=metadata,
    )
    return _sha256_file(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth-prefix", required=True, type=Path)
    parser.add_argument("--observation-prefix", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    digest = extract_sample(
        arguments.ground_truth_prefix,
        arguments.observation_prefix,
        arguments.output,
    )
    print(f"wrote {arguments.output} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
