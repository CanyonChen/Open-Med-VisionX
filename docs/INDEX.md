# OpenMedVisionX Documentation

[简体中文](INDEX.zh-CN.md) · [Project overview](../README.md)

This is the starting point for the OpenMedVisionX user documentation. Choose
the path that matches what you want to accomplish; you do not need to read
every page in order.

> [!NOTE]
> All workflows are for learning and research. OpenMedVisionX is not a medical
> device and its outputs must not be used for patient decisions.

## New here? Follow this path

1. **Install and launch:** [Quickstart](QUICKSTART.md), sections 1–2.
2. **Get a result without using local data:** run the first synthetic CT
   experiment in [Quickstart](QUICKSTART.md), section 3.
3. **Understand the screen:** read the interface map in the [User
   guide](USER_GUIDE.md).
4. **Choose a topic:** use the workspace and task tables below.
5. **Build sound learning habits:** follow the [Learning
   curriculum](TEACHING_CURRICULUM.md).

If installation or loading fails at any point, go directly to
[Troubleshooting](TROUBLESHOOTING.md).

## Browse by workspace

| Workspace | Start here | Deeper reading |
| --- | --- | --- |
| **Images** | [User guide · Images](USER_GUIDE.md#images) | [Data and formats](DATA_FORMATS.md) |
| **CT Lab** | [User guide · CT Lab](USER_GUIDE.md#ct-lab) | [Curriculum · CT projections](TEACHING_CURRICULUM.md#3-from-object-to-sinogram-and-back) |
| **Models** | [User guide · Models](USER_GUIDE.md#models) | [Bundled models](MODEL_BUNDLES.md) · [Custom local models](CUSTOM_MODELS.md) |
| **Learn** | [User guide · Learn](USER_GUIDE.md#learn) | [Learning curriculum](TEACHING_CURRICULUM.md) · [Glossary](GLOSSARY.md) |
| **Evaluate** | [User guide · Evaluate](USER_GUIDE.md#evaluate) | [Curriculum · Evaluation](TEACHING_CURRICULUM.md#5-evaluation-without-leakage) |
| **AI Assistant** | [User guide · AI Assistant](USER_GUIDE.md#ai-assistant) | [AI Assistant, privacy, and cloud images](LLM_SECURITY.md) |

## Browse by task

| I want to… | Read… |
| --- | --- |
| Open a PNG, TIFF, DICOM series/ZIP, or NIfTI volume | [Data and formats](DATA_FORMATS.md), then [User guide · Images](USER_GUIDE.md#images) |
| Import a mask, DICOM SEG, RTSTRUCT, or label map | [Data and formats · Annotations and labels](DATA_FORMATS.md#annotations-masks-and-label-layers) |
| Complete the built-in CT experiment | [Quickstart · First session](QUICKSTART.md#3-complete-your-first-session) |
| Install PyTorch and run a reviewed model | [Bundled model guide](MODEL_BUNDLES.md) |
| Load my own ONNX, TorchScript, or Python-adapter model | [Custom model guide](CUSTOM_MODELS.md) |
| Check a dataset split or binary predictions | [User guide · Evaluate](USER_GUIDE.md#evaluate) |
| Configure an AI provider without storing the key in the project | [AI Assistant, privacy, and cloud images](LLM_SECURITY.md) |
| Understand an unfamiliar term | [Glossary](GLOSSARY.md) |
| Diagnose an error message | [Troubleshooting](TROUBLESHOOTING.md) |

## Core guides

| Guide | What it gives you |
| --- | --- |
| [Quickstart](QUICKSTART.md) | Installation, launch, a first privacy-safe session, and optional capability setup. |
| [User guide](USER_GUIDE.md) | Complete, task-oriented instructions for the six desktop workspaces. |
| [Data and formats](DATA_FORMATS.md) | Supported files, selection rules, geometry/intensity meaning, annotations, and safe preparation. |
| [Bundled model guide](MODEL_BUNDLES.md) | PyTorch setup, integrity checks, and the DIVal, DeepInverse, and MONAI workflows. |
| [Custom model guide](CUSTOM_MODELS.md) | What another local model needs before the desktop can validate, load, and run it. |
| [AI Assistant, privacy, and cloud images](LLM_SECURITY.md) | Provider setup, credential references, text requests, exact image consent, and incident steps. |
| [Learning curriculum](TEACHING_CURRICULUM.md) | A structured path from pixels and geometry to reconstruction, evaluation, and responsible AI. |
| [Troubleshooting](TROUBLESHOOTING.md) | Symptom-based fixes for setup, data, models, evaluation, exports, and AI requests. |

## Reference and boundaries

- [One-page bilingual overview](INTRODUCTION.md) is a concise project
  introduction suitable for sharing before the full guide.
- [Glossary](GLOSSARY.md) explains the terms used in the interface and guides.
- [Supported data and formats](DATA_FORMATS.md#support-matrix) is the authority
  for what the desktop can open and how it interprets each input.
- [Bundled model cards](MODEL_BUNDLES.md#understand-each-models-input) are the
  authority for model-specific inputs and limitations.
- [AI privacy guide](LLM_SECURITY.md#what-can-leave-your-computer) explains
  exactly when text or a rendered image can leave the local trust boundary.
- [Security policy](../SECURITY.md) explains how to report a vulnerability or
  suspected disclosure privately.

No training dataset, BraTS case, API credential, or patient image is included.
OpenMedVisionX never turns an unsupported input into a plausible one by silently
inventing spacing, modality, channels, geometry, or acquisition semantics.

## Language and version scope

Every page in the user journey has an English and Simplified Chinese version;
use the language link at the top of a page to switch without changing topics.
Button names are written exactly as they appear in the corresponding interface
language.

These guides describe the current alpha release in this repository. When a
workflow is intentionally unavailable—such as using the generic external-model
page for a compound k-space/mask input—the documentation says so explicitly.

Next: [install and complete your first session](QUICKSTART.md).
