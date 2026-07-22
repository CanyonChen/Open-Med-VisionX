# DIVal LoDoPaB FBP-U-Net / DIVal LoDoPaB FBP-U-Net 模型卡

## English

Purpose: compare analytic/iterative CT baselines with learned post-processing.
The input is the **fixed DIVal LoDoPaB Hann-filtered FBP result**, shaped
`1×1×362×362`; it is not a clinical HU image and not a scanner-native sinogram.
Input, output, and error maps must share one declared display/evaluation range.

The network is specific to the LoDoPaB-CT benchmark distribution and geometry.
It cannot establish diagnostic performance, must not be applied to arbitrary CT
images, and may suppress or invent structures outside its training domain.

## 中文

用途：比较解析型、迭代型 CT 基线与学习型后处理。输入必须是采用固定 DIVal
LoDoPaB Hann 滤波 FBP 流程得到的 `1×1×362×362` 图像；它既不是临床 HU
图像，也不是扫描仪原始 sinogram。输入、输出和误差图必须共用一个已声明的显示
与评价范围。

该网络只适用于 LoDoPaB-CT 基准的数据分布和几何条件，不能证明诊断性能，也不应
用于任意 CT 图像；离开训练域时可能抑制或生成不存在的结构。
