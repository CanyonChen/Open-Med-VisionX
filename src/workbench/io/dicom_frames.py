"""Small helpers for classic and Enhanced DICOM frame functional groups."""

from __future__ import annotations

from typing import Any


def dicom_frame_count(dataset: object) -> int:
    """Return a conservative frame count; invalid metadata remains one frame."""

    try:
        value = int(getattr(dataset, "NumberOfFrames", 1))
    except (TypeError, ValueError, OverflowError):
        return 1
    return value if value > 0 else 1


def dicom_frame_attribute(
    dataset: object,
    frame_index: int,
    *,
    sequence_name: str,
    attribute_name: str,
) -> Any | None:
    """Resolve a per-frame, shared, then top-level DICOM attribute.

    Enhanced objects place geometry and pixel transforms in functional-group
    sequences. Classic single-frame images keep the same values at top level.
    This helper preserves that precedence without mutating pydicom datasets.
    """

    per_frame = getattr(dataset, "PerFrameFunctionalGroupsSequence", None)
    if per_frame is not None and 0 <= frame_index < len(per_frame):
        value = _nested_attribute(per_frame[frame_index], sequence_name, attribute_name)
        if value is not None:
            return value

    shared = getattr(dataset, "SharedFunctionalGroupsSequence", None)
    if shared:
        value = _nested_attribute(shared[0], sequence_name, attribute_name)
        if value is not None:
            return value

    return getattr(dataset, attribute_name, None)


def _nested_attribute(
    functional_group: object,
    sequence_name: str,
    attribute_name: str,
) -> Any | None:
    sequence = getattr(functional_group, sequence_name, None)
    if not sequence:
        return None
    return getattr(sequence[0], attribute_name, None)
