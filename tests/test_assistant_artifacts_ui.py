from __future__ import annotations

import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QSignalSpy
from PyQt5.QtWidgets import QAbstractButton, QComboBox, QGroupBox, QLabel

from workbench.llm.artifacts import (
    ArtifactReview,
    ArtifactReviewDecision,
    ArtifactValidationStatus,
    LLMArtifactResponse,
    LLMArtifactType,
    LLMInputKind,
    LLMInputReference,
    LLMTaskKind,
    LLMTaskRequest,
    ProviderResponseMetadata,
    TextArtifact,
)
from workbench.ui.assistant_artifacts import (
    ARTIFACT_TASK_TEMPLATES,
    AssistantArtifactsPanel,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _request(*, request_id: str = "request-local", prompt_sha256: str = SHA_C) -> LLMTaskRequest:
    return LLMTaskRequest(
        request_id=request_id,
        task=LLMTaskKind.CLASSIFY,
        inputs=(
            LLMInputReference(
                input_id="private-case-alias",
                kind=LLMInputKind.IMAGE_2D,
                payload_sha256=SHA_A,
            ),
        ),
        transfer_plan_sha256=SHA_B,
        prompt_sha256=prompt_sha256,
        requested_artifact_types=(LLMArtifactType.CLASS_SCORES,),
    )


def _response(request: LLMTaskRequest) -> LLMArtifactResponse:
    return LLMArtifactResponse.from_normalized_payload(
        artifact_id="artifact-local",
        request=request,
        artifact_type=LLMArtifactType.TEXT,
        payload=TextArtifact(
            text=r"Private payload C:\records\case-one and PatientName=Example",
            language="en",
        ),
        provider=ProviderResponseMetadata(
            provider_id="provider-private",
            model_id="registry/private-model",
            response_id="response-private",
            authenticated=True,
        ),
        warnings=(r"Inspect C:\records\case-one before use",),
    )


def _rendered_text(panel: AssistantArtifactsPanel) -> str:
    texts: list[str] = []
    for label in panel.findChildren(QLabel):
        texts.extend(
            (
                label.text(),
                label.toolTip(),
                label.accessibleName(),
                label.accessibleDescription(),
            )
        )
    for group in panel.findChildren(QGroupBox):
        texts.extend(
            (
                group.title(),
                group.toolTip(),
                group.accessibleName(),
                group.accessibleDescription(),
            )
        )
    for button in panel.findChildren(QAbstractButton):
        texts.extend(
            (
                button.text(),
                button.toolTip(),
                button.accessibleName(),
                button.accessibleDescription(),
            )
        )
    for combo in panel.findChildren(QComboBox):
        texts.extend(
            (
                combo.toolTip(),
                combo.accessibleName(),
                combo.accessibleDescription(),
            )
        )
        for index in range(combo.count()):
            texts.extend((combo.itemText(index), str(combo.itemData(index, Qt.ToolTipRole))))
    return "\n".join(texts)


def test_template_selector_covers_all_seven_contracts_and_emits_safe_values(qtbot) -> None:
    panel = AssistantArtifactsPanel()
    qtbot.addWidget(panel)
    spy = QSignalSpy(panel.taskTemplateChanged)

    assert len(ARTIFACT_TASK_TEMPLATES) == 7
    assert panel.template_combo.count() == 7
    assert [
        panel.template_combo.itemData(index) for index in range(panel.template_combo.count())
    ] == [artifact_type.value for artifact_type in LLMArtifactType]

    panel.template_combo.setCurrentIndex(4)

    assert panel.selected_template.artifact_type is LLMArtifactType.MASK_3D
    assert panel.selected_template.compatible_tasks == (LLMTaskKind.SEGMENT,)
    assert list(spy[-1]) == ["mask_3d", "mask_3d"]

    panel.set_selected_artifact_type("reconstructed_volume")
    assert panel.selected_template.artifact_type is LLMArtifactType.RECONSTRUCTED_VOLUME


def test_language_switch_updates_visible_and_accessible_copy(qtbot) -> None:
    panel = AssistantArtifactsPanel()
    qtbot.addWidget(panel)

    assert panel.title_label.text() == "Structured artifact contracts"
    assert panel.template_combo.accessibleName() == "Structured AI artifact task template"
    assert panel.confirm_button.toolTip().startswith("Request confirmation")

    panel.set_language("zh_CN")

    assert panel.title_label.text() == "结构化产物契约"
    assert panel.template_combo.itemText(0) == "概念解释 · 文本"
    assert panel.template_combo.accessibleName() == "结构化 AI 产物任务模板"
    assert panel.confirm_button.accessibleName() == "确认产物"
    assert "不会发送" in panel.boundary_label.text()
    assert "桌面文件选择器" in panel.boundary_label.text()

    with pytest.raises(ValueError, match="Unsupported UI language"):
        panel.set_language("fr")  # type: ignore[arg-type]


def test_summary_validates_binding_without_rendering_payload_or_identifiers(qtbot) -> None:
    request = _request()
    response = _response(request)
    panel = AssistantArtifactsPanel()
    qtbot.addWidget(panel)

    panel.set_context(request, response)

    assert panel.binding_valid
    assert panel.confirm_button.isEnabled()
    assert panel.reject_button.isEnabled()
    assert panel.status_label.property("state") == "ready"
    assert panel._request_value_labels["inputs"].text() == "1 deidentified · 2-D image"
    assert "Matches the exact typed request" in panel._artifact_value_labels["request_match"].text()
    assert "text/plain; charset=utf-8" in panel._artifact_value_labels["payload"].text()
    assert "1 warning(s)" in panel._artifact_value_labels["payload"].text()

    rendered = _rendered_text(panel)
    for private_value in (
        "private-case-alias",
        "artifact-local",
        "provider-private",
        "registry/private-model",
        "response-private",
        "PatientName=Example",
        r"C:\records",
        SHA_A,
        SHA_B,
        SHA_C,
        response.artifact_sha256,
        response.request_sha256,
    ):
        assert private_value not in rendered


def test_review_buttons_emit_a_single_request_and_never_mutate_response(qtbot) -> None:
    request = _request()
    response = _response(request)
    panel = AssistantArtifactsPanel()
    qtbot.addWidget(panel)
    panel.set_context(request, response)
    spy = QSignalSpy(panel.reviewActionRequested)

    qtbot.mouseClick(panel.confirm_button, Qt.LeftButton)

    assert list(spy[-1]) == ["artifact-local", ArtifactReviewDecision.CONFIRMED.value]
    assert response.validation_status is ArtifactValidationStatus.UNVERIFIED
    assert response.reviews == ()
    assert not panel.confirm_button.isEnabled()
    assert not panel.reject_button.isEnabled()
    assert panel.status_label.property("state") == "busy"

    qtbot.mouseClick(panel.confirm_button, Qt.LeftButton)
    assert len(spy) == 1

    panel.set_artifact(response)
    qtbot.mouseClick(panel.reject_button, Qt.LeftButton)
    assert list(spy[-1]) == ["artifact-local", ArtifactReviewDecision.REJECTED.value]
    assert len(spy) == 2


def test_missing_or_mismatched_request_blocks_review_with_compact_status(qtbot) -> None:
    original_request = _request()
    response = _response(original_request)
    panel = AssistantArtifactsPanel()
    qtbot.addWidget(panel)

    panel.set_context(None, response)
    assert not panel.binding_valid
    assert not panel.confirm_button.isEnabled()
    assert panel._artifact_value_labels["request_match"].text().startswith("Load its typed")

    other_request = _request(request_id="request-other", prompt_sha256=SHA_D)
    panel.set_context(other_request, response)
    assert not panel.binding_valid
    assert not panel.reject_button.isEnabled()
    assert panel.status_label.property("state") == "blocked"
    assert "does not match" in panel.status_label.text()


def test_completed_review_is_read_only_and_localized(qtbot) -> None:
    request = _request()
    response = _response(request)
    confirmed = response.with_review(
        ArtifactReview(
            decision=ArtifactReviewDecision.CONFIRMED,
            reviewer_id="reviewer-local",
            note_sha256=SHA_D,
        )
    )
    panel = AssistantArtifactsPanel()
    qtbot.addWidget(panel)
    panel.set_context(request, confirmed)

    assert panel.binding_valid
    assert not panel.confirm_button.isEnabled()
    assert panel._artifact_value_labels["review"].text() == "User confirmed"
    assert panel.status_label.property("state") == "ready"

    panel.set_language("zh_CN")
    assert panel._artifact_value_labels["review"].text() == "用户已确认"
    assert "不可变响应" in panel.status_label.text()


def test_layout_reflows_and_review_controls_are_keyboard_reachable(qtbot) -> None:
    request = _request()
    panel = AssistantArtifactsPanel()
    qtbot.addWidget(panel)
    panel.set_context(request, _response(request))

    panel.resize(920, 620)
    panel.show()
    qtbot.wait(10)
    assert panel.summary_splitter.orientation() == Qt.Horizontal

    panel.resize(560, 720)
    qtbot.wait(10)
    assert panel.summary_splitter.orientation() == Qt.Vertical
    assert panel.minimumSizeHint().width() == 0

    panel.template_combo.setFocus()
    qtbot.keyClick(panel.template_combo, Qt.Key_Tab)
    assert panel.confirm_button.hasFocus()
    qtbot.keyClick(panel.confirm_button, Qt.Key_Tab)
    assert panel.reject_button.hasFocus()


def test_public_setters_reject_untyped_values(qtbot) -> None:
    panel = AssistantArtifactsPanel()
    qtbot.addWidget(panel)

    with pytest.raises(TypeError, match="LLMTaskRequest"):
        panel.set_request(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="LLMArtifactResponse"):
        panel.set_artifact(object())  # type: ignore[arg-type]
