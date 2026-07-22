# LoDoPaB-CT minimal teaching case / LoDoPaB-CT 最小教学案例

## English

This package contains test sample 3456 extracted from LoDoPaB-CT Dataset
version 1.0.0, DOI `10.5281/zenodo.3384092`, created by Johannes Leuschner,
Maximilian Schmidt, and Daniel Otero Baguer. The dataset is distributed under
Creative Commons Attribution 4.0 International. The accompanying article is
DOI `10.1038/s41597-021-00893-z`.

OpenMedVisionX selected one public observation/ground-truth pair, converted
both arrays to little-endian `float32`, and derived the fixed Hann-filtered
back-projection used by the bundled DIVal FBP-U-Net. The three arrays are
stored in a deterministic, non-pickle NPZ. No DICOM headers, patient
identifiers, local source paths, or executable objects are included. This
modified subset is not endorsed by the original creators. It is for education
and research only, is not a clinical acquisition, and must not be used as a
basis for diagnosis.

## 中文

本资源从 LoDoPaB-CT 数据集 1.0.0 版（DOI：
`10.5281/zenodo.3384092`）中提取测试集第 3456 个样本。原作者为 Johannes
Leuschner、Maximilian Schmidt 与 Daniel Otero Baguer，数据集采用知识共享
署名 4.0 国际许可；配套论文 DOI 为 `10.1038/s41597-021-00893-z`。

OpenMedVisionX 仅选择一对公开的 observation/ground-truth 数组，将其转换为
小端 `float32`，并派生出随包 DIVal FBP-U-Net 所需的固定 Hann 滤波反投影。
三个数组均保存于确定性、无 pickle 的 NPZ。资源不含 DICOM 头、患者标识、
本地源路径或可执行对象；上述修改不代表原作者认可。本样本仅用于教学与研究，
不是临床采集数据，也不能作为诊断依据。
