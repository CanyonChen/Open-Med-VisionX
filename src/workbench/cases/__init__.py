"""Reviewed teaching cases that ship without an application-time download."""

from .brats import (
    BRATS_2021_MODALITIES,
    BRATS_2021_OFFICIAL_ACQUISITION_URL,
    BRATS_2021_SEGMENTATION_LABELS,
    BraTS2021FileRecord,
    BraTS2021Geometry,
    BraTS2021Issue,
    BraTS2021ValidationReport,
    build_brats2021_manifest,
    validate_brats2021_case,
    write_brats2021_manifest,
)
from .bundled import (
    BUNDLED_TEACHING_CASE_IDS,
    LodopabTeachingCase,
    TeachingCaseIntegrityError,
    TeachingCaseRecord,
    load_lodopab_case,
    verify_bundled_teaching_cases,
    verify_teaching_case,
)

__all__ = [
    "BUNDLED_TEACHING_CASE_IDS",
    "BRATS_2021_MODALITIES",
    "BRATS_2021_OFFICIAL_ACQUISITION_URL",
    "BRATS_2021_SEGMENTATION_LABELS",
    "BraTS2021FileRecord",
    "BraTS2021Geometry",
    "BraTS2021Issue",
    "BraTS2021ValidationReport",
    "LodopabTeachingCase",
    "TeachingCaseIntegrityError",
    "TeachingCaseRecord",
    "load_lodopab_case",
    "build_brats2021_manifest",
    "validate_brats2021_case",
    "verify_bundled_teaching_cases",
    "verify_teaching_case",
    "write_brats2021_manifest",
]
