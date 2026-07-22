# Model conversion environment / 模型转换环境

## English

The checked-in artifacts were reviewed with Python 3.11.15 and PyTorch
2.13.0+cu130 on Windows 11; DeepInverse conversion additionally requires
DeepInverse 0.4.1. Source size and SHA-256 are fixed in every conversion script.
The scripts use `torch.load(..., weights_only=True)` only for the two reviewed
maintainer inputs. They export numeric NPZ tensors and deterministic golden
references; application runtime opens those files with `allow_pickle=False`.
The MONAI artifact remains the unmodified, hash-matched upstream TorchScript.

## 中文

仓库中的资产使用 Windows 11、Python 3.11.15 和 PyTorch 2.13.0+cu130 完成审查；
DeepInverse 转换还要求 DeepInverse 0.4.1。每个转换脚本都固定源文件的字节数与
SHA-256。脚本仅对两个经审查的维护输入使用
`torch.load(..., weights_only=True)`，随后导出数值 NPZ 与确定性 golden 参考；
应用运行时始终以 `allow_pickle=False` 打开这些文件。MONAI 资产保留为哈希完全
匹配的上游原始 TorchScript。
