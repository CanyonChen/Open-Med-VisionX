"""Optional NIfTI loader that canonicalizes volumes to RAS+."""

from __future__ import annotations

import gzip
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from ..domain.images import ImageVolume, IntensitySemantics, SourceType
from ..errors import (
    DecodeError,
    FormatMismatchError,
    MissingDependencyError,
    OperationCancelled,
    ResourceLimitError,
    ValidationError,
)
from .base import CancelCheck, ImageLoader, LoadLimits, PathLike, ProbeResult, raise_if_cancelled


class NiftiVolumeSelectionRequiredError(DecodeError):
    """Raised when a 4-D NIfTI needs an explicit time-point/channel choice."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.volume_count = int(shape[3]) if len(shape) == 4 else 0
        super().__init__(
            "This NIfTI contains multiple 3-D volumes. Select one time point or channel "
            "explicitly; OpenMedVisionX will not guess."
        )


def _nifti_signature(path: Path) -> bool:
    try:
        with path.open("rb") as prefix_stream:
            compressed = prefix_stream.read(2) == b"\x1f\x8b"
        opener = gzip.open if compressed else open
        with opener(path, "rb") as stream:  # type: ignore[arg-type]
            header = stream.read(560)
    except (OSError, EOFError):
        return False
    if len(header) < 12:
        return False
    header_size_little = int.from_bytes(header[:4], "little")
    header_size_big = int.from_bytes(header[:4], "big")
    if 348 in {header_size_little, header_size_big} and len(header) >= 348:
        return header[344:347] in {b"n+1", b"ni1"}
    if 540 in {header_size_little, header_size_big} and len(header) >= 12:
        return header[4:7] in {b"n+2", b"ni2"}
    return False


def _is_gzip_stream(path: Path) -> bool:
    """Detect gzip from file contents instead of trusting the filename."""

    try:
        with path.open("rb") as stream:
            return stream.read(2) == b"\x1f\x8b"
    except OSError:
        return False


class NiftiLoader(ImageLoader):
    name = "nifti"

    def probe(self, source: PathLike) -> ProbeResult:
        path = Path(source)
        if not path.is_file():
            return ProbeResult(False)
        lower_name = path.name.lower()
        expected = lower_name.endswith(".nii") or lower_name.endswith(".nii.gz")
        signature = _nifti_signature(path)
        gzip_compressed = _is_gzip_stream(path)
        return ProbeResult(
            expected or signature,
            "NIFTI" if signature or expected else None,
            100 if expected and signature else 85 if signature else 45,
            {
                "extension_matches": expected and signature,
                "signature_valid": signature,
                "gzip_compressed": gzip_compressed,
            },
        )

    def load(
        self,
        source: PathLike,
        *,
        limits: LoadLimits | None = None,
        cancel: CancelCheck = None,
        modality: str | None = None,
        intensity_semantics: IntensitySemantics | str | None = None,
        volume_index: int | None = None,
    ) -> ImageVolume:
        """Load a NIfTI volume without treating its container as a modality.

        NIfTI headers generally do not identify whether samples are MR signal,
        CT HU, probability, or labels.  The default is therefore ``UNKNOWN``.
        A caller may provide an explicit, recorded declaration after checking
        the dataset documentation or sidecar metadata.
        """

        declared_modality = "UNKNOWN" if modality is None else str(modality).upper().strip()
        if not declared_modality:
            raise ValidationError("An explicitly declared NIfTI modality cannot be empty.")
        try:
            declared_semantics = (
                IntensitySemantics.UNKNOWN
                if intensity_semantics is None
                else IntensitySemantics(intensity_semantics)
            )
        except ValueError as exc:
            raise ValidationError(
                f"Unsupported NIfTI intensity semantics {intensity_semantics!r}."
            ) from exc
        try:
            import nibabel as nib
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise MissingDependencyError(
                "NIfTI loading requires nibabel. Install the 'nifti' optional dependency."
            ) from exc

        path = Path(source)
        active_limits = limits or LoadLimits()
        result = self.probe(path)
        if not result.accepted or not result.details.get("signature_valid"):
            raise DecodeError(f"{path.name!r} is not a valid NIfTI-1/NIfTI-2 file.")
        lower_name = path.name.lower()
        if not (lower_name.endswith(".nii") or lower_name.endswith(".nii.gz")):
            raise FormatMismatchError(
                "NIfTI signature detected, but the filename must end in .nii or .nii.gz."
            )
        raise_if_cancelled(cancel)
        try:
            with ExitStack() as stack:
                gzip_compressed = bool(result.details.get("gzip_compressed"))
                gzip_suffix = lower_name.endswith(".nii.gz")
                if gzip_suffix and not gzip_compressed:
                    # Some datasets contain a valid, uncompressed single-file NIfTI
                    # whose name incorrectly ends in .nii.gz. nibabel chooses its
                    # opener from the suffix, so give it an explicit raw stream.
                    raw_stream = stack.enter_context(path.open("rb"))
                    header_size_bytes = raw_stream.read(4)
                    raw_stream.seek(0)
                    header_sizes = {
                        int.from_bytes(header_size_bytes, "little"),
                        int.from_bytes(header_size_bytes, "big"),
                    }
                    image_class = nib.Nifti2Image if 540 in header_sizes else nib.Nifti1Image
                    file_map = image_class.make_file_map()
                    file_map["image"].fileobj = raw_stream
                    image = image_class.from_file_map(file_map, mmap=True)
                else:
                    image = nib.load(str(path), mmap=True)

                canonical = nib.as_closest_canonical(image)
                source_shape = tuple(int(item) for item in canonical.shape)
                if len(source_shape) == 4:
                    if source_shape[3] <= 0:
                        raise DecodeError("A 4-D NIfTI must contain at least one 3-D volume.")
                    if source_shape[3] > active_limits.max_frames:
                        raise ResourceLimitError(
                            "NIfTI time-point/channel count exceeds the configured frame limit."
                        )
                    if volume_index is None:
                        raise NiftiVolumeSelectionRequiredError(source_shape)
                    if isinstance(volume_index, bool):
                        raise ValidationError("NIfTI volume_index must be an integer index.")
                    selected_index = int(volume_index)
                    if selected_index != volume_index or not 0 <= selected_index < source_shape[3]:
                        raise ValidationError(
                            f"NIfTI volume_index must be between 0 and {source_shape[3] - 1}."
                        )
                    shape = source_shape[:3]
                elif len(source_shape) == 3:
                    if volume_index is not None:
                        raise ValidationError(
                            "A NIfTI volume index can be used only with a 4-D NIfTI source."
                        )
                    selected_index = None
                    shape = source_shape
                else:
                    raise DecodeError(
                        "The stable viewer supports 3-D NIfTI or explicit selection from a "
                        f"4-D NIfTI, got shape {source_shape}."
                    )
                dtype = np.dtype(canonical.get_data_dtype())
                voxel_count = int(np.prod(shape, dtype=np.int64))
                proxy = canonical.dataobj
                slope = getattr(proxy, "slope", 1.0)
                intercept = getattr(proxy, "inter", 0.0)
                scaling_active = (
                    slope is not None
                    and intercept is not None
                    and (not np.isclose(float(slope), 1.0) or not np.isclose(float(intercept), 0.0))
                )
                # nibabel may materialize scaled integer data as floating point.
                # Use a conservative estimate before touching the proxy, then
                # validate the actual allocation immediately after decoding.
                decoded_itemsize = max(dtype.itemsize, 8 if scaling_active else dtype.itemsize)
                decoded_bytes = voxel_count * decoded_itemsize
                if voxel_count > active_limits.max_pixels * active_limits.max_frames:
                    raise ResourceLimitError(
                        "NIfTI voxel count exceeds the configured safety limit."
                    )
                if decoded_bytes > active_limits.max_decoded_bytes:
                    prefix = "Scaled NIfTI data" if scaling_active else "NIfTI data"
                    raise ResourceLimitError(
                        f"{prefix} requires at least {decoded_bytes:,} decoded bytes, "
                        "above the configured limit."
                    )
                raise_if_cancelled(cancel)
                xyz_array = np.asanyarray(
                    canonical.dataobj
                    if selected_index is None
                    else canonical.dataobj[..., selected_index]
                )
                if int(xyz_array.nbytes) > active_limits.max_decoded_bytes:
                    raise ResourceLimitError(
                        "Decoded NIfTI volume exceeds the configured memory limit."
                    )
                raise_if_cancelled(cancel)
        except (DecodeError, OperationCancelled, ResourceLimitError, ValidationError):
            raise
        except Exception as exc:
            raise DecodeError(f"Could not decode NIfTI file {path.name!r}: {exc}") from exc

        affine = np.asarray(canonical.affine, dtype=np.float64)
        if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
            raise DecodeError("NIfTI affine is missing or invalid.")
        basis = affine[:3, :3]
        spacing = tuple(float(item) for item in np.linalg.norm(basis, axis=0))
        if any(item <= 0 for item in spacing):
            raise DecodeError("NIfTI affine contains a zero-length spatial axis.")
        direction = basis @ np.diag([1.0 / item for item in spacing])
        origin = tuple(float(item) for item in affine[:3, 3])
        # NIfTI/nibabel uses (x, y, z); the platform stores arrays as (z, y, x)
        # while affine coordinates remain explicitly (x, y, z).
        zyx_array = np.transpose(xyz_array, (2, 1, 0))
        actual_decoded_bytes = int(zyx_array.nbytes)
        return ImageVolume(
            array=zyx_array,
            source_type=SourceType.NIFTI,
            intensity_semantics=declared_semantics,
            runtime_metadata={
                "loader": self.name,
                "format": "NIFTI",
                "intensity_semantics_source": (
                    "user_declared" if intensity_semantics is not None else "not_declared"
                ),
                "modality_source": "user_declared" if modality is not None else "not_declared",
                "storage_dtype": str(dtype),
                "canonical_orientation": "RAS+",
                "decoded_bytes": actual_decoded_bytes,
                "storage_compression": "gzip" if gzip_compressed else "none",
                "source_shape": source_shape,
                "selected_volume_index": selected_index,
                "volume_selection": (
                    "explicit-user-selection" if selected_index is not None else "single-volume"
                ),
            },
            affine=affine,
            spacing=spacing,
            origin=origin,
            direction=direction,
            modality=declared_modality,
        )
