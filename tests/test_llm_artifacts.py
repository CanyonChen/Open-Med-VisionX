from __future__ import annotations

import binascii
import hashlib
import zlib
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import numpy as np
import pytest

import workbench.llm.artifacts as artifact_module
from workbench.domain.images import IntensitySemantics
from workbench.errors import ValidationError
from workbench.llm.artifacts import (
    ArtifactLabelDefinition,
    ArtifactReview,
    ArtifactReviewDecision,
    ArtifactValidationStatus,
    CalibrationMethod,
    ClassScore,
    ClassScoresArtifact,
    DataConsistencyMethod,
    DerivedLayerKind,
    LabelsArtifact,
    LLMArtifactResponse,
    LLMArtifactType,
    LLMInputKind,
    LLMInputReference,
    LLMTaskKind,
    LLMTaskRequest,
    Mask2DArtifact,
    Mask3DArtifact,
    ProviderResponseMetadata,
    ReconstructedImageArtifact,
    ReconstructedVolumeArtifact,
    ReconstructionEvidence,
    ScoreSemantics,
    SemanticLabel,
    SpatialArtifactReference,
    TextArtifact,
    encode_npy,
)
from workbench.llm.types import TransferItem, TransferPlan

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _input(
    *,
    kind: LLMInputKind = LLMInputKind.VOLUME_3D,
    payload_sha256: str = SHA_A,
    slice_index: int | None = None,
) -> LLMInputReference:
    return LLMInputReference(
        input_id="input-1",
        kind=kind,
        payload_sha256=payload_sha256,
        layer_id="source-layer",
        series_id="series-1",
        slice_index=slice_index,
        transform_sha256=SHA_B,
        deidentified=True,
    )


def _request(
    artifact_type: LLMArtifactType,
    *,
    task: LLMTaskKind = LLMTaskKind.SEGMENT,
    input_kind: LLMInputKind = LLMInputKind.VOLUME_3D,
) -> LLMTaskRequest:
    evidence = None
    if task is LLMTaskKind.RECONSTRUCT:
        evidence = ReconstructionEvidence(
            sampling_operator_sha256=SHA_B,
            data_consistency=DataConsistencyMethod.ITERATIVE_UPDATE,
            acquisition_parameters_sha256=SHA_C,
        )
    return LLMTaskRequest(
        request_id="request-1",
        task=task,
        inputs=(_input(kind=input_kind),),
        transfer_plan_sha256=SHA_B,
        prompt_sha256=SHA_C,
        requested_artifact_types=(artifact_type,),
        reconstruction_evidence=evidence,
        created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )


def _provider(**changes: object) -> ProviderResponseMetadata:
    values: dict[str, object] = {
        "provider_id": "provider-1",
        "model_id": "model-1",
        "response_id": "response-1",
        "authenticated": True,
        "received_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
        "latency_ms": 12.5,
        "adapter_metadata": {"api_version": "v1", "region": "local"},
    }
    values.update(changes)
    return ProviderResponseMetadata(**values)  # type: ignore[arg-type]


def _reference(*, sliced: bool) -> SpatialArtifactReference:
    return SpatialArtifactReference(
        series_id="series-1",
        layer_id="source-layer",
        shape_zyx=(2, 3, 4),
        affine_ras=np.diag([1.0, 2.0, 3.0, 1.0]),
        frame_of_reference_uid="1.2.840.10008.1",
        slice_axis=0 if sliced else None,
        slice_index=1 if sliced else None,
    )


def _labels() -> tuple[ArtifactLabelDefinition, ...]:
    return (
        ArtifactLabelDefinition(0, "background", "Background", "#000000"),
        ArtifactLabelDefinition(1, "target", "Target", "#FF375F"),
    )


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    crc = binascii.crc32(chunk_type + payload) & 0xFFFFFFFF
    return len(payload).to_bytes(4, "big") + chunk_type + payload + crc.to_bytes(4, "big")


def _grayscale_png(array: np.ndarray, *, metadata: bool = False) -> bytes:
    height, width = array.shape
    header = width.to_bytes(4, "big") + height.to_bytes(4, "big") + bytes([8, 0, 0, 0, 0])
    chunks = [_png_chunk(b"IHDR", header)]
    if metadata:
        chunks.append(_png_chunk(b"tEXt", b"unsafe\x00metadata"))
    rows = b"".join(b"\x00" + bytes(row) for row in array.astype(np.uint8))
    chunks.extend(
        [
            _png_chunk(b"IDAT", zlib.compress(rows)),
            _png_chunk(b"IEND", b""),
        ]
    )
    return b"\x89PNG\r\n\x1a\n" + b"".join(chunks)


def test_task_request_contains_only_hashes_and_opaque_references() -> None:
    request = _request(LLMArtifactType.MASK_3D)

    assert request.task is LLMTaskKind.SEGMENT
    assert request.inputs[0].deidentified is True
    assert len(request.request_sha256) == 64
    assert "prompt" not in repr(request).lower().replace("prompt_sha256", "")
    with pytest.raises(ValidationError, match="opaque identifier"):
        _input().__class__(
            input_id="../source",
            kind=LLMInputKind.VOLUME_3D,
            payload_sha256=SHA_A,
        )
    with pytest.raises(ValidationError, match="explicitly marked deidentified"):
        LLMInputReference(
            input_id="input-2",
            kind=LLMInputKind.IMAGE_2D,
            payload_sha256=SHA_A,
            deidentified=False,
        )


def test_reconstruction_requires_measurement_and_physics_evidence() -> None:
    with pytest.raises(ValidationError, match="requires k-space or sinogram"):
        _request(
            LLMArtifactType.RECONSTRUCTED_VOLUME,
            task=LLMTaskKind.RECONSTRUCT,
            input_kind=LLMInputKind.VOLUME_3D,
        )

    request = _request(
        LLMArtifactType.RECONSTRUCTED_VOLUME,
        task=LLMTaskKind.RECONSTRUCT,
        input_kind=LLMInputKind.KSPACE,
    )
    assert request.reconstruction_evidence is not None

    with pytest.raises(ValidationError, match="only for the reconstruct task"):
        LLMTaskRequest(
            request_id="request-restore",
            task=LLMTaskKind.RESTORE,
            inputs=(_input(kind=LLMInputKind.IMAGE_2D),),
            transfer_plan_sha256=SHA_B,
            prompt_sha256=SHA_C,
            requested_artifact_types=(LLMArtifactType.RECONSTRUCTED_IMAGE,),
            reconstruction_evidence=request.reconstruction_evidence,
        )


def test_task_binds_to_exact_transfer_plan() -> None:
    item = TransferItem(
        name="payload.png",
        mime_type="image/png",
        size_bytes=10,
        sha256=SHA_A,
        transform="rendered slice",
        deidentification_actions=("source metadata excluded",),
        burned_in_text_review="reviewed",
    )
    plan = TransferPlan(
        provider_id="provider-1",
        provider_name="Provider",
        endpoint="https://example.invalid/v1",
        model_id="model-1",
        task="segment",
        prompt_sha256=SHA_C,
        items=(item,),
    )
    request = LLMTaskRequest(
        request_id="request-plan",
        task=LLMTaskKind.SEGMENT,
        inputs=(_input(),),
        transfer_plan_sha256=plan.plan_id,
        prompt_sha256=SHA_C,
        requested_artifact_types=(LLMArtifactType.MASK_3D,),
    )

    request.validate_transfer_plan(plan)
    with pytest.raises(ValidationError, match="reviewed TransferPlan digest"):
        _request(LLMArtifactType.MASK_3D).validate_transfer_plan(plan)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"authenticated": False}, "authenticated provider"),
        ({"model_id": "../../model"}, "opaque identifier"),
        ({"adapter_metadata": {"callback_url": "https://bad.invalid"}}, "paths, or URLs"),
        ({"adapter_metadata": {"patient_name": "private"}}, "identify people"),
    ],
)
def test_provider_metadata_rejects_unauthenticated_or_unsafe_values(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        _provider(**changes)


def test_provider_metadata_accepts_registry_style_model_identifier() -> None:
    assert _provider(model_id="registry/model-v1").model_id == "registry/model-v1"


def test_text_artifact_is_utf8_and_cannot_create_an_image_layer() -> None:
    request = _request(LLMArtifactType.MASK_3D)
    response = LLMArtifactResponse.from_normalized_payload(
        artifact_id="artifact-text",
        request=request,
        artifact_type=LLMArtifactType.TEXT,
        payload=TextArtifact("No structured image was returned.", "en", ("citation-1",)),
        provider=_provider(),
        warnings=("The requested segmentation was not produced.",),
    )

    assert response.mime_type == "text/plain; charset=utf-8"
    assert response.produced_structured_image is False
    assert response.validation_status is ArtifactValidationStatus.UNVERIFIED
    with pytest.raises(ValidationError, match="cannot create image layers"):
        response.describe_derived_layer(request, layer_id="new-layer")


def test_class_scores_retain_score_semantics_threshold_and_calibration() -> None:
    request = _request(LLMArtifactType.CLASS_SCORES, task=LLMTaskKind.CLASSIFY)
    score = ClassScore(
        class_id="finding-a",
        class_name="Finding A",
        score=-1.2,
        semantics=ScoreSemantics.LOGIT,
        threshold=0.0,
        calibration=CalibrationMethod.UNCALIBRATED,
    )
    response = LLMArtifactResponse.from_normalized_payload(
        artifact_id="artifact-scores",
        request=request,
        artifact_type=LLMArtifactType.CLASS_SCORES,
        payload=ClassScoresArtifact((score,)),
        provider=_provider(),
    )

    assert response.payload.scores[0].score == -1.2  # type: ignore[union-attr]
    assert response.payload.scores[0].semantics is ScoreSemantics.LOGIT  # type: ignore[union-attr]
    assert response.mime_type == "application/json"
    with pytest.raises(ValidationError, match=r"probability must be in \[0, 1\]"):
        ClassScore("bad", "Bad", 2.0, ScoreSemantics.PROBABILITY)


def test_language_labels_remain_semantic_and_never_become_pixel_masks() -> None:
    request = _request(LLMArtifactType.LABELS, task=LLMTaskKind.LABELS)
    response = LLMArtifactResponse.from_normalized_payload(
        artifact_id="artifact-labels",
        request=request,
        artifact_type=LLMArtifactType.LABELS,
        payload=LabelsArtifact(
            (SemanticLabel("finding", "Possible finding", 0.4, ("region-1",)),),
            "en-US",
        ),
        provider=_provider(),
    )

    assert response.is_spatial is False
    with pytest.raises(ValidationError, match="cannot create image layers"):
        response.describe_derived_layer(request, layer_id="mask-impostor")


def test_mask_2d_png_is_decoded_and_describes_new_immutable_layer() -> None:
    request = _request(LLMArtifactType.MASK_2D)
    source = np.array([[0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.uint8)
    payload = Mask2DArtifact(source, _reference(sliced=True), _labels())
    source[:] = 0
    encoded = _grayscale_png(payload.array)
    response = LLMArtifactResponse.from_normalized_payload(
        artifact_id="artifact-mask-2d",
        request=request,
        artifact_type=LLMArtifactType.MASK_2D,
        payload=payload,
        provider=_provider(),
        mime_type="image/png",
        encoded_bytes=encoded,
    )

    assert not payload.array.flags.writeable
    assert np.count_nonzero(payload.array) == 6
    layer = response.describe_derived_layer(request, layer_id="derived-mask")
    assert layer.layer_kind is DerivedLayerKind.SEGMENTATION
    assert layer.immutable is True
    assert layer.replace_original is False
    assert layer.derived_from_layer_ids == ("source-layer",)
    assert layer.validation_status is ArtifactValidationStatus.UNVERIFIED
    assert not layer.array.flags.writeable
    with pytest.raises((ValueError, RuntimeError)):
        layer.array[0, 0] = 9
    with pytest.raises(FrozenInstanceError):
        layer.name = "overwrite"  # type: ignore[misc]


def test_png_metadata_chunk_and_pixel_mismatch_are_rejected() -> None:
    request = _request(LLMArtifactType.MASK_2D)
    array = np.zeros((3, 4), dtype=np.uint8)
    payload = Mask2DArtifact(array, _reference(sliced=True), _labels())

    with pytest.raises(ValidationError, match="unsafe metadata"):
        LLMArtifactResponse.from_normalized_payload(
            artifact_id="artifact-unsafe-png",
            request=request,
            artifact_type=LLMArtifactType.MASK_2D,
            payload=payload,
            provider=_provider(),
            mime_type="image/png",
            encoded_bytes=_grayscale_png(array, metadata=True),
        )
    with pytest.raises(ValidationError, match="pixels do not match"):
        LLMArtifactResponse.from_normalized_payload(
            artifact_id="artifact-wrong-pixels",
            request=request,
            artifact_type=LLMArtifactType.MASK_2D,
            payload=payload,
            provider=_provider(),
            mime_type="image/png",
            encoded_bytes=_grayscale_png(np.ones_like(array)),
        )


def test_mask_3d_npy_validates_shape_schema_hash_and_read_only_array() -> None:
    request = _request(LLMArtifactType.MASK_3D)
    source = np.zeros((2, 3, 4), dtype=np.uint8)
    source[:, 1, 2] = 1
    payload = Mask3DArtifact(source, _reference(sliced=False), _labels())
    encoded = encode_npy(payload.array)
    response = LLMArtifactResponse.from_normalized_payload(
        artifact_id="artifact-mask-3d",
        request=request,
        artifact_type=LLMArtifactType.MASK_3D,
        payload=payload,
        provider=_provider(),
        mime_type="application/x-npy",
        encoded_bytes=encoded,
    )

    assert response.artifact_sha256 == hashlib.sha256(encoded).hexdigest()
    assert not payload.array.flags.writeable
    with pytest.raises(ValidationError, match="absent from label_schema"):
        Mask3DArtifact(np.full((2, 3, 4), 2), _reference(sliced=False), _labels())
    with pytest.raises(ValidationError, match="complete referenced volume"):
        Mask3DArtifact(np.zeros((1, 3, 4)), _reference(sliced=False), _labels())


def test_mask_3d_nifti_validates_decoded_voxels_and_affine() -> None:
    nib = pytest.importorskip("nibabel")
    request = _request(LLMArtifactType.MASK_3D)
    array = np.zeros((2, 3, 4), dtype=np.uint8)
    array[:, 1, 2] = 1
    reference = _reference(sliced=False)
    xyz = np.transpose(array, (2, 1, 0))
    encoded = nib.Nifti1Image(xyz, reference.affine_ras).to_bytes()
    payload = Mask3DArtifact(array, reference, _labels())

    response = LLMArtifactResponse.from_normalized_payload(
        artifact_id="artifact-mask-nifti",
        request=request,
        artifact_type=LLMArtifactType.MASK_3D,
        payload=payload,
        provider=_provider(),
        mime_type="application/x-nifti",
        encoded_bytes=encoded,
    )
    assert response.artifact_sha256 == hashlib.sha256(encoded).hexdigest()

    wrong_reference = SpatialArtifactReference(
        series_id="series-1",
        layer_id="source-layer",
        shape_zyx=(2, 3, 4),
        affine_ras=np.diag([2.0, 2.0, 3.0, 1.0]),
    )
    with pytest.raises(ValidationError, match="affine does not match"):
        LLMArtifactResponse.from_normalized_payload(
            artifact_id="artifact-wrong-affine",
            request=request,
            artifact_type=LLMArtifactType.MASK_3D,
            payload=Mask3DArtifact(array, wrong_reference, _labels()),
            provider=_provider(),
            mime_type="application/x-nifti",
            encoded_bytes=encoded,
        )


def test_reconstructed_image_and_volume_preserve_intensity_and_geometry() -> None:
    image_request = _request(
        LLMArtifactType.RECONSTRUCTED_IMAGE,
        task=LLMTaskKind.RESTORE,
        input_kind=LLMInputKind.IMAGE_2D,
    )
    image_payload = ReconstructedImageArtifact(
        np.arange(12, dtype=np.uint8).reshape(3, 4),
        _reference(sliced=True),
        IntensitySemantics.GRAYSCALE,
    )
    image_response = LLMArtifactResponse.from_normalized_payload(
        artifact_id="artifact-image",
        request=image_request,
        artifact_type=LLMArtifactType.RECONSTRUCTED_IMAGE,
        payload=image_payload,
        provider=_provider(),
    )
    image_layer = image_response.describe_derived_layer(image_request, layer_id="derived-image")
    assert image_layer.task is LLMTaskKind.RESTORE
    assert image_layer.intensity_semantics is IntensitySemantics.GRAYSCALE

    volume_request = _request(
        LLMArtifactType.RECONSTRUCTED_VOLUME,
        task=LLMTaskKind.RECONSTRUCT,
        input_kind=LLMInputKind.KSPACE,
    )
    volume_payload = ReconstructedVolumeArtifact(
        np.ones((2, 3, 4), dtype=np.float32),
        _reference(sliced=False),
        IntensitySemantics.ARBITRARY_SIGNAL,
    )
    volume_response = LLMArtifactResponse.from_normalized_payload(
        artifact_id="artifact-volume",
        request=volume_request,
        artifact_type=LLMArtifactType.RECONSTRUCTED_VOLUME,
        payload=volume_payload,
        provider=_provider(),
    )
    assert (
        volume_response.describe_derived_layer(volume_request, layer_id="derived-volume").layer_kind
        is DerivedLayerKind.VOLUME
    )


def test_artifact_types_are_mutually_exclusive_and_response_is_request_bound() -> None:
    request = _request(LLMArtifactType.MASK_3D)
    text = TextArtifact("Text only", "en")
    encoded = text.canonical_bytes()
    with pytest.raises(ValidationError, match="cannot masquerade"):
        LLMArtifactResponse(
            artifact_id="artifact-impostor",
            request_id=request.request_id,
            request_sha256=request.request_sha256,
            artifact_type=LLMArtifactType.MASK_3D,
            payload=text,
            provider=_provider(),
            mime_type="application/x-npy",
            encoded_bytes=encoded,
            artifact_sha256=hashlib.sha256(encoded).hexdigest(),
        )

    other = _request(LLMArtifactType.MASK_2D)
    response = LLMArtifactResponse.from_normalized_payload(
        artifact_id="artifact-text-bound",
        request=request,
        artifact_type=LLMArtifactType.TEXT,
        payload=text,
        provider=_provider(),
    )
    with pytest.raises(ValidationError, match="exact task request"):
        response.validate_against(other)


def test_hash_mime_trailing_npy_and_size_limits_are_enforced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(LLMArtifactType.MASK_3D)
    payload = Mask3DArtifact(np.zeros((2, 3, 4)), _reference(sliced=False), _labels())
    encoded = encode_npy(payload.array)

    with pytest.raises(ValidationError, match="SHA-256"):
        LLMArtifactResponse(
            artifact_id="artifact-bad-hash",
            request_id=request.request_id,
            request_sha256=request.request_sha256,
            artifact_type=LLMArtifactType.MASK_3D,
            payload=payload,
            provider=_provider(),
            mime_type="application/x-npy",
            encoded_bytes=encoded,
            artifact_sha256=SHA_A,
        )
    with pytest.raises(ValidationError, match="MIME type"):
        LLMArtifactResponse.from_normalized_payload(
            artifact_id="artifact-bad-mime",
            request=request,
            artifact_type=LLMArtifactType.MASK_3D,
            payload=payload,
            provider=_provider(),
            mime_type="application/octet-stream",
            encoded_bytes=encoded,
        )
    with pytest.raises(ValidationError, match="trailing data"):
        LLMArtifactResponse.from_normalized_payload(
            artifact_id="artifact-trailing",
            request=request,
            artifact_type=LLMArtifactType.MASK_3D,
            payload=payload,
            provider=_provider(),
            mime_type="application/x-npy",
            encoded_bytes=encoded + b"script",
        )
    monkeypatch.setattr(artifact_module, "MAX_ARTIFACT_BYTES", 8)
    with pytest.raises(ValidationError, match="32 MiB limit"):
        LLMArtifactResponse.from_normalized_payload(
            artifact_id="artifact-too-large",
            request=request,
            artifact_type=LLMArtifactType.MASK_3D,
            payload=payload,
            provider=_provider(),
            mime_type="application/x-npy",
            encoded_bytes=encoded,
        )


def test_user_review_is_appended_without_erasing_model_provenance() -> None:
    request = _request(LLMArtifactType.MASK_3D)
    payload = Mask3DArtifact(np.zeros((2, 3, 4)), _reference(sliced=False), _labels())
    response = LLMArtifactResponse.from_normalized_payload(
        artifact_id="artifact-review",
        request=request,
        artifact_type=LLMArtifactType.MASK_3D,
        payload=payload,
        provider=_provider(),
    )
    reviewed = response.with_review(
        ArtifactReview(
            decision=ArtifactReviewDecision.CONFIRMED,
            reviewer_id="local-reviewer",
            note_sha256=SHA_A,
        )
    )

    assert response.validation_status is ArtifactValidationStatus.UNVERIFIED
    assert reviewed.validation_status is ArtifactValidationStatus.USER_CONFIRMED
    assert reviewed.provider == response.provider
    assert reviewed.artifact_sha256 == response.artifact_sha256
    with pytest.raises(ValidationError, match="cannot be overwritten"):
        reviewed.with_review(ArtifactReview(ArtifactReviewDecision.REJECTED, "reviewer-2", SHA_B))


def test_spatial_reference_affine_and_arrays_are_defensive_read_only_copies() -> None:
    affine = np.eye(4)
    reference = SpatialArtifactReference(
        series_id="series-1",
        layer_id="source-layer",
        shape_zyx=(2, 3, 4),
        affine_ras=affine,
    )
    affine[0, 0] = 99

    assert reference.affine_ras[0, 0] == 1
    assert not reference.affine_ras.flags.writeable
    with pytest.raises(ValidationError, match="independent spatial axes"):
        SpatialArtifactReference(
            series_id="series-1",
            layer_id="source-layer",
            shape_zyx=(2, 3, 4),
            affine_ras=np.diag([0.0, 0.0, 0.0, 1.0]),
        )
