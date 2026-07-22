# Source and derivation evidence / 来源与派生证据

## English

This teaching asset is derived from **LoDoPaB-CT Dataset 1.0.0**, DOI
[`10.5281/zenodo.3384092`](https://doi.org/10.5281/zenodo.3384092), by Johannes
Leuschner, Maximilian Schmidt, and Daniel Otero Baguer. The official Zenodo
record licenses the dataset under `CC-BY-4.0`; the accompanying data descriptor
is DOI [`10.1038/s41597-021-00893-z`](https://doi.org/10.1038/s41597-021-00893-z).

The released pair is test index **3456**: row 0 of HDF5 members `027`.
OpenMedVisionX fetched only the fixed raw-DEFLATE ranges needed for that row.
It did not download either complete multi-gigabyte archive or complete member.

| Source | Official archive/member evidence | Reviewed range |
| --- | --- | --- |
| Ground truth | `ground_truth_test.zip`; 1,582,139,537 bytes; MD5 `ecc655767fbe3d40908ca823921f4c7f`; member `ground_truth_test_027.hdf5`; local header 1,538,816,654; compressed 43,320,789 bytes; expanded 56,387,960 bytes; CRC-32 `3edd9669` | bytes `1538816710-1543011013`; 4,194,304 bytes; SHA-256 `58d517b9cca643e2f9d8df3927752587998e415b9aa2a77f73ff02bf7feb9adf` |
| Observation | `observation_test.zip`; 2,996,574,366 bytes; MD5 `9ae6b053bb1faa94d573311af8ec67b2`; member `observation_test_027.hdf5`; local header 2,914,525,943; compressed 82,046,358 bytes; expanded 221,594,744 bytes; CRC-32 `be47200c` | bytes `2914525998-2922914605`; 8,388,608 bytes; SHA-256 `6379e14b597d244cdf10e7def44e7f14e9f0edb3c63ed325d96a819295e43c4e` |

The converter validates exact HTTP `Content-Range`, range size and SHA-256;
then it checks the HDF5 signature, declared size, dataset shape, chunk layout,
dtype, and derived-array hashes. Archive MD5 and complete-member CRC-32 above
are immutable metadata from Zenodo and the ZIP central directory; they were not
recomputed from complete archives during this bounded extraction.

### FBP-U-Net input derivation

The fixed DIVal repository revision
[`dd86f03733593dd6e226263aa1d3abec961c7881`](https://github.com/jleuschn/dival/tree/dd86f03733593dd6e226263aa1d3abec961c7881)
defines the LoDoPaB domain (`[-0.13, 0.13]²`), 362×362 image grid, 1,000
parallel-beam angles, and 513 detector pixels. The fixed supp.dival revision
[`6117cabc55d223f5b62ea43d4a40225270fb6756`](https://github.com/jleuschn/supp.dival/tree/6117cabc55d223f5b62ea43d4a40225270fb6756/reference_params/lodopab)
specifies the FBP-U-Net preprocessing as a Hann filter with
`frequency_scaling=1.0`. This is different from the standalone FBP baseline's
tuned cutoff of `0.641025641025641`.

The released FBP was computed offline with ODL 0.8.3, ASTRA Toolbox 2.5.0,
the `astra_cuda` backend, padding enabled, and the parameters above. Repeating
the conversion on the reviewed machine produced the same byte-level SHA-256.
ODL and ASTRA are not application runtime dependencies because the derived FBP
is bundled as a numeric array.

- Observation array SHA-256: `b4f20f395d68e9755e0a250a78206351fee1a95b8c4f4ed4e236f40eb12b0be7`
- FBP array SHA-256: `2fbfeb49fc11dc239c9c44226ddc2c611bf96c93b4b89f85d6d0fc61105f1b75`
- Ground-truth array SHA-256: `24392e79235397cf3275588201ab67faf25401f5a9587c8bce8980e1e864aa0b`
- Released NPZ: 1,912,858 bytes; SHA-256 `b323cdef2529927336069b3385605d1049117fe69e59583072861fa573493846`
- Reproduction tools: `scripts/data_conversion/fetch_lodopab_members.py` and
  `scripts/data_conversion/extract_lodopab_sample.py`

This public benchmark subset contains no DICOM header, patient identifier, or
source path. Its images are normalized attenuation arrays, not clinical HU.
It is suitable only for education and research, not diagnosis or validation of
clinical performance.

## 中文

本教学资源派生自 Johannes Leuschner、Maximilian Schmidt 与 Daniel Otero
Baguer 发布的 **LoDoPaB-CT Dataset 1.0.0**，数据集 DOI 为
[`10.5281/zenodo.3384092`](https://doi.org/10.5281/zenodo.3384092)，许可为
`CC-BY-4.0`；配套数据论文 DOI 为
[`10.1038/s41597-021-00893-z`](https://doi.org/10.1038/s41597-021-00893-z)。

发布样本固定为测试集索引 **3456**，即 HDF5 成员 `027` 的第 0 行。
OpenMedVisionX 仅获取读取这一行所需的固定原始 DEFLATE 字节范围，没有下载
完整的大型归档或完整 ZIP 成员。转换器严格验证 HTTP `Content-Range`、范围大小
和 SHA-256，并检查 HDF5 签名、声明大小、数据形状、分块布局、数据类型及派生
数组哈希。上表中的归档 MD5 与完整成员 CRC-32 来自 Zenodo 和 ZIP 中央目录；
本次受限提取没有下载完整归档重新计算它们。

DIVal 固定版本 `dd86f…` 给出了 LoDoPaB 的空间范围、362×362 图像网格、
1,000 个平行束角度和 513 个探测器像素；supp.dival 固定版本 `6117…` 明确规定
FBP-U-Net 的预处理为 Hann 滤波、`frequency_scaling=1.0`。该值不同于独立 FBP
基线使用的 `0.641025641025641`。发布的 FBP 使用 ODL 0.8.3、ASTRA Toolbox
2.5.0、`astra_cuda` 后端及 padding 离线生成；在审核机器上重复生成所得 SHA-256
完全一致。应用运行时直接读取已固化的数值数组，不需要 ODL 或 ASTRA。

- observation 数组 SHA-256：`b4f20f395d68e9755e0a250a78206351fee1a95b8c4f4ed4e236f40eb12b0be7`
- FBP 数组 SHA-256：`2fbfeb49fc11dc239c9c44226ddc2c611bf96c93b4b89f85d6d0fc61105f1b75`
- ground-truth 数组 SHA-256：`24392e79235397cf3275588201ab67faf25401f5a9587c8bce8980e1e864aa0b`
- 最终 NPZ：1,912,858 字节；SHA-256 `b323cdef2529927336069b3385605d1049117fe69e59583072861fa573493846`

该公开 benchmark 子集不含 DICOM 头、患者标识或源路径；图像为归一化衰减
数组，不是临床 HU。它仅适用于教学与研究，不能用于诊断或证明临床性能。
