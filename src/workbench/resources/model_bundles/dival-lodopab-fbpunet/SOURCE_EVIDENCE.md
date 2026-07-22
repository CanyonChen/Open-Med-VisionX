# Source evidence / 来源证据

## English

- Official supplementary repository: <https://github.com/jleuschn/supp.dival>
- Immutable artifact revision: `6117cabc55d223f5b62ea43d4a40225270fb6756`
- Artifact path: `reference_params/lodopab/lodopab_fbpunet.pt`
- Original size: 2,485,382 bytes
- Original SHA-256: `990b366a5053485093e2b9876f1503d92fcd786b44f8970ca6f5d9e4c1dfdf23`
- Reviewed DIVal U-Net source revision: `dd86f03733593dd6e226263aa1d3abec961c7881`

The Git LFS pointer at the immutable supplementary revision declares the same
object size and SHA-256. The release does not contain the upstream pickle.
The conversion command reads the reviewed checkpoint with
`torch.load(..., weights_only=True)`, checks every key, shape, and dtype against
the pinned DIVal graph, then writes numeric arrays only to `weights.npz`.

## 中文

- 官方补充材料仓库：<https://github.com/jleuschn/supp.dival>
- 固定资源版本：`6117cabc55d223f5b62ea43d4a40225270fb6756`
- 资源路径：`reference_params/lodopab/lodopab_fbpunet.pt`
- 原始大小：2,485,382 字节
- 原始 SHA-256：`990b366a5053485093e2b9876f1503d92fcd786b44f8970ca6f5d9e4c1dfdf23`
- 经审查的 DIVal U-Net 源码版本：`dd86f03733593dd6e226263aa1d3abec961c7881`

固定补充材料版本中的 Git LFS 指针声明了相同的对象大小与 SHA-256。发行包不包含
上游 pickle。转换命令使用 `torch.load(..., weights_only=True)` 读取经审查的
checkpoint，逐一核对固定 DIVal 计算图所需的键、形状和数据类型，随后仅将数值数组
写入 `weights.npz`。
