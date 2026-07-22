# Source evidence / 来源证据

## English

- Reviewed model repository: <https://huggingface.co/deepinv/demo>
- Immutable revision: `35ef1ffedb3c8973721754a5f2bd1ca1a36added`
- Artifact: `demo_tour_mri.pth`, 38,274 bytes
- Original SHA-256: `e756b303917babeca8c32896597c9e4989fdd6a9792de3a79e9965f1f5cdbaff`
- Repository metadata at the reviewed revision declares `bsd-3-clause`.
- The complete BSD 3-Clause text is retained from the pinned official
  DeepInverse source revision `fa6b144999ad648b112ec4a998168771870e342f`.

The upstream pickle is not distributed. Only the reviewed tensors in a
pickle-free NPZ, the trusted source graph, conversion script, golden reference,
and exact hashes are retained. The converter compares that graph against the
official DeepInverse 0.4.1 implementation on the same deterministic input.

## 中文

- 经审查的模型仓库：<https://huggingface.co/deepinv/demo>
- 固定版本：`35ef1ffedb3c8973721754a5f2bd1ca1a36added`
- 源资产：`demo_tour_mri.pth`，38,274 字节
- 源 SHA-256：`e756b303917babeca8c32896597c9e4989fdd6a9792de3a79e9965f1f5cdbaff`
- 固定版本的仓库元数据声明采用 `bsd-3-clause`。
- 完整 BSD 3-Clause 文本取自 DeepInverse 官方源码固定版本
  `fa6b144999ad648b112ec4a998168771870e342f` 并保留在本目录。

发行包不包含上游 pickle，仅保留无 pickle NPZ 中经审查的张量、可信源码计算图、
转换脚本、golden 参考和精确哈希。转换器还会在同一确定性输入上与官方
DeepInverse 0.4.1 实现进行数值对比。
