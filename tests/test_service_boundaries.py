from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dicom_viewer.inference import (
    EntrypointSpec,
    PluginContractError,
    PluginNotLoadedError,
    PythonEnvironmentSpec,
    RuntimeKind,
    RuntimeSpec,
    load_manifest,
)
from dicom_viewer.runtime import CancellationToken
from dicom_viewer.services import (
    LLMProviderRegistry,
    ModelInferenceService,
    ProviderConfiguration,
    ProviderDefaults,
    TeachingAssistantService,
)


def _example_manifest_path() -> Path:
    return (
        Path(__file__).parents[1]
        / "src"
        / "dicom_viewer"
        / "inference"
        / "examples"
        / "manifest.yaml"
    )


def _factory_map(plugin: object):
    return {runtime: (lambda _manifest, _root: plugin) for runtime in RuntimeKind}


def test_model_service_owns_manifest_factory_load_predict_and_visualize() -> None:
    plugin = SimpleNamespace(
        load=Mock(),
        predict=Mock(return_value="typed-result"),
        visualize=Mock(return_value=("artifact",)),
        cancel=Mock(return_value=True),
        close=Mock(),
    )
    service = ModelInferenceService(plugin_factories=_factory_map(plugin))  # type: ignore[arg-type]

    manifest = service.inspect_manifest(_example_manifest_path())
    assert manifest.runtime.kind is RuntimeKind.ONNX
    assert not service.requires_python_consent
    service.load()
    assert service.ready
    load_context = plugin.load.call_args.args[0]
    assert load_context.plugin_root == _example_manifest_path().parent
    assert not load_context.user_consented_python_code

    result = service.predict(
        {"image": "pixels"},
        parameters={"input_context": {"image": {"color_space": "rgb"}}},
    )
    assert result == "typed-result"
    request = plugin.predict.call_args.args[0]
    assert request.inputs == {"image": "pixels"}
    assert service.visualize(result) == ("artifact",)  # type: ignore[arg-type]
    assert service.cancel()
    assert service.ready  # Standard runtimes remain reusable after cooperative cancellation.


def test_python_adapter_cancel_requires_an_explicit_reload_and_consent() -> None:
    base = load_manifest(_example_manifest_path())
    python_manifest = replace(
        base,
        runtime=RuntimeSpec(
            kind=RuntimeKind.PYTHON_ADAPTER,
            python=PythonEnvironmentSpec(python_executable="python"),
        ),
        entrypoint=EntrypointSpec("adapter.py", "Adapter"),
    )
    plugin = SimpleNamespace(
        load=Mock(),
        predict=Mock(return_value="typed-result"),
        visualize=Mock(return_value=()),
        cancel=Mock(return_value=True),
        close=Mock(),
    )
    service = ModelInferenceService(
        manifest_loader=lambda _path, *, validate_files=False: python_manifest,
        plugin_factories=_factory_map(plugin),  # type: ignore[arg-type]
    )
    service.inspect_manifest("manifest.yaml")
    assert service.requires_python_consent
    with pytest.raises(PluginContractError, match="explicit user consent"):
        service.load()

    service.load(user_consented_python_code=True)
    assert service.ready
    service.cancel()
    assert service.reload_required
    assert not service.ready
    with pytest.raises(PluginNotLoadedError, match="must be loaded again"):
        service.predict({"image": "pixels"})


def test_provider_registry_owns_defaults_transport_and_factory_selection() -> None:
    created: list[dict[str, object]] = []
    online_transport = object()
    offline_transport = object()

    def builder(**kwargs):
        created.append(kwargs)
        return SimpleNamespace()

    registry = LLMProviderRegistry(
        defaults=(ProviderDefaults("Lab", "http://localhost:8080/v1", "env:LAB_KEY"),),
        builders={"Lab": builder},
        online_transport_factory=lambda: online_transport,  # type: ignore[arg-type]
        offline_transport_factory=lambda: offline_transport,  # type: ignore[arg-type]
    )
    assert registry.names == ("Lab",)
    assert registry.defaults_for("Lab").credential_ref == "env:LAB_KEY"
    registry.create(
        ProviderConfiguration(
            "Lab",
            "model-v1",
            "http://localhost:8080/v1",
            "env:LAB_KEY",
            network_enabled=True,
        )
    )
    assert created[0]["transport"] is online_transport
    assert created[0]["model_id"] == "model-v1"


def test_kimi_provider_default_allows_longer_reasoning_requests() -> None:
    registry = LLMProviderRegistry()

    assert registry.defaults_for("Moonshot/Kimi").timeout == 120.0


def test_assistant_service_forwards_authorization_and_cancellation_token() -> None:
    token = CancellationToken()
    provider = SimpleNamespace(
        capabilities=Mock(return_value=SimpleNamespace(vision=True)),
        authorize_image_transfer=Mock(),
        chat=Mock(return_value="response"),
    )
    service = TeachingAssistantService()
    service.authorize_image_transfer(provider)  # type: ignore[arg-type]
    response = service.chat(
        provider,  # type: ignore[arg-type]
        "Explain the reconstruction.",
        cancellation_token=token,
    )
    assert response == "response"
    provider.authorize_image_transfer.assert_called_once_with()
    provider.chat.assert_called_once_with(
        "Explain the reconstruction.",
        preview=None,
        cancellation_token=token,
    )
