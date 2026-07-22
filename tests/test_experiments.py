from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from workbench.domain import IntensitySemantics, RasterImage2D, SourceType
from workbench.errors import OperationCancelled, ValidationError
from workbench.services import (
    create_experiment_record,
    export_rendered_png,
    load_local_annotations,
    load_local_mask,
    save_experiment_record,
)


def test_explicit_mask_and_annotation_pairing_is_shape_checked(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask.png"
    pixels = np.zeros((8, 10), dtype=np.uint8)
    pixels[2:6, 3:7] = 5
    Image.fromarray(pixels).save(mask_path)
    mask = load_local_mask(mask_path, (8, 10))
    assert mask.dtype == np.bool_
    assert int(mask.sum()) == 16

    annotation_path = tmp_path / "annotation.json"
    annotation_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "coordinate_system": "pixel_xy_top_left",
                "image_size": [10, 8],
                "boxes": [{"x1": 1, "y1": 2, "x2": 7, "y2": 6, "label": "ROI"}],
                "points": [{"x": 4, "y": 3, "label": "landmark"}],
            }
        ),
        encoding="utf-8",
    )
    overlay = load_local_annotations(annotation_path, (8, 10))
    assert overlay.boxes == ((1.0, 2.0, 7.0, 6.0),)
    assert overlay.points == ((4.0, 3.0),)
    with pytest.raises(ValidationError, match="does not match"):
        load_local_mask(mask_path, (9, 10))


def test_experiment_record_is_pixel_free_and_export_never_overwrites(tmp_path: Path) -> None:
    image = RasterImage2D(
        np.arange(64, dtype=np.uint16).reshape(8, 8),
        SourceType.RASTER,
        IntensitySemantics.GRAYSCALE,
        bit_depth=16,
    )
    record = create_experiment_record(
        "display mapping",
        image=image,
        parameters={"lower": 3.0, "upper": 60.0},
        metrics={"mse": 0.25},
        created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    encoded = json.dumps(record)
    assert "pixels" not in encoded
    assert "source_path" not in encoded
    record_path = tmp_path / "experiment.json"
    save_experiment_record(record_path, record)
    with pytest.raises(ValidationError, match="does not overwrite"):
        save_experiment_record(record_path, record)

    png_path = tmp_path / "rendered.png"
    export_rendered_png(png_path, image.array)
    assert Image.open(png_path).size == (8, 8)
    with pytest.raises(ValidationError, match="does not overwrite"):
        export_rendered_png(png_path, image.array)


def test_cancelled_export_leaves_no_partial_file(tmp_path: Path) -> None:
    target = tmp_path / "cancelled.png"
    with pytest.raises(OperationCancelled):
        export_rendered_png(
            target,
            np.zeros((8, 8), dtype=np.uint8),
            cancel=lambda: True,
        )
    assert not target.exists()


def test_save_revalidates_record_and_rejects_pixel_or_identity_injection(
    tmp_path: Path,
) -> None:
    record = create_experiment_record(
        "safe",
        image=None,
        parameters={"display_lower": 0.0},
    )
    injected = dict(record)
    injected["raw_pixels"] = [1, 2, 3]
    with pytest.raises(ValidationError, match="unsupported fields"):
        save_experiment_record(tmp_path / "injected.json", injected)

    identity = dict(record)
    identity["parameters"] = {"patient_name": "must not be saved"}
    with pytest.raises(ValidationError, match="may expose"):
        save_experiment_record(tmp_path / "identity.json", identity)
