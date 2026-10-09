# 原稿已发表方法基线结果精确提取

提取已完成。原稿 Table 3 含 PASNVGA、SDNN-PPI 和旧 CaPPI 三行；前两行是已发表方法在原稿中报告的历史基线结果，CaPPI 行是原稿自身方法的历史对照行。本次没有重新运行任何方法。

## 原始表格

| Model | ACC | Sen | Spec | Prec | MCC | F1-Score |
|---|---|---|---|---|---|---|
| PASNVGA | 0.7736±0.0025 | 0.7779±0.0295 | 0.7693±0.0327 | 0.7722±0.0179 | 0.5481±0.0044 | 0.7745±0.0057 |
| SDNN-PPI | 0.8357±0.0049 | 0.8768±0.0198 | 0.7946±0.0110 | 0.8103±0.0049 | 0.6739±0.0118 | 0.8421±0.0068 |
| CaPPI | 0.8920±0.0017 | 0.9231±0.0054 | 0.8609±0.0084 | 0.8691±0.0061 | 0.7856±0.0029 | 0.8953±0.0011 |

以上 28 个单元格（含表头）的可见字符串直接从 DOCX 读取，并逐格与既有文本提取交叉检查一致。均值和标准差保留原字符串、原小数位及无空格的 ±，不修正或重新估计。原 P87 将这些结果描述为五个保留测试 fold 的均值和标准差。

## 定位与来源

- DOCX：`D:\1.肝癌\CaPPI-code\SR投稿\CaPPI Manuscript_8.29.docx`。SHA256：`8d8dc91fd2ee80d01acd0acf9b2f95a38e40685f324ba0bceae0f7d72690968e`。
- 文本交叉核对：`D:\1.肝癌\revision_20260930\manuscript_829_extracted.txt`。SHA256：`ff940972a5dcfda6e14dd974d9b70a1d19c52fc2d3b75dd9c6a895796dcb2eef`。
- 表题原文：Table 3. Performance comparison of CaPPI and some existing models。位于 P127（零基段落索引）。
- DOCX 中为第 4 个表（零基 table index 3，因为第 1 个表是 Algorithm 1）；XML 定位 `/w:document/w:body/w:tbl[4]`。
- 既有文本行号：TABLE 3 标记在第 282 行，表头第 283 行，三条数据第 284–286 行。
- 含表头计，PASNVGA 为第 2 行、SDNN-PPI 为第 3 行、CaPPI 为第 4 行。JSON 给出每个单元格的 XML XPath、行列索引、原字符串和可分离的 mean_text/sd_text。

## 原文是否说这些是作者复现结果

是，就原稿的陈述而言，§3.1.1（P82，既有文本第 83 行）明确写明 baseline 使用官方开源代码或库实现，并在与 CaPPI 相同的数据及预定义划分上评估：

> All baseline models were implemented using their official open-source code or libraries and evaluated using the same dataset of interaction pairs among HCC-associated proteins and predefined data partitions as CaPPI.

§3.6 的 P121 再次称所有方法使用同一数据集。因此这些数值应标为“原稿报告的已发表方法历史实验结果”，不能称为直接摘引 PASNVGA 或 SDNN-PPI 论文中的测试分数。以上判断是对原文含义的核实；本次未核验旧运行日志、权重、原始预测或逐 fold 成员，不据此确认旧代码确实执行了文中所有协议。

## 原文所述历史协议

- P24：582 个蛋白、23,707 个阳性，主实验配 23,707 个随机推定负例，正负 1:1；度匹配负集另作附加分析。
- P82、P121：原稿声称基线在同一数据集及预定义划分上评估；未逐方法提供精确划分成员或原始运行证据。
- P86：分层蛋白对五折；每次 20% 外测试，其余按 87.5%/12.5% 分训练/验证，即约 70%/10%/20%；只训练下游 MLP，验证用于学习率调度和模型选择。原文同时声称每折仅训练阳性重建 DRW 图。
- P87：每个外 fold 恰好测试一次，按五折报告均值和标准差；蛋白可跨分区出现，因此是已表示节点中的蛋白对预测，不能解释为未知蛋白泛化。
- 这些是原稿自述协议，并非本次已经验证的新协议或证据。JSON 内保留上述段落原文与定位。

## Table 3 之外的原文 AUC 与 AUPR

P122（既有文本第 123 行）在 Figure 7 的比较叙述中给出如下点值；它们不是 Table 3 的单元格。本段未给出相应标准差，JSON 使用 null，不补造；保留原指标名 AUC/AUPR，不自动改称本轮 Average Precision。

| 原方法名 | AUC 原字符串 | AUPR 原字符串 | 此段标准差 |
|---|---|---|---|
| PASNVGA | 0.8547 | 0.8287 | 未给出 |
| SDNN-PPI | 0.9148 | 0.9038 | 未给出 |
| CaPPI | 0.9513 | 0.9376 | 未给出 |

## 对应原文参考文献

- 原编号 63，P230：63	Sarkar, D. & Saha, S. Machine-learning techniques for the prediction of protein–protein interactions. Journal of biosciences 44, 104, doi:https://doi.org/10.1007/s12038-019-9909-z (2019).
- 原编号 64，P231：64	Luo, X., Wang, L., Hu, P. & Hu, L. Predicting protein-protein interactions using sequence and network information via variational graph autoencoder. IEEE/ACM Transactions on Computational Biology and Bioinformatics 20, 3182–3194, doi:https://doi.org/10.1109/TCBB.2023.3273567 (2023).
- 原编号 65，P232：65	Li, X. et al. SDNN-PPI: self-attention with deep neural network effect on protein-protein interaction prediction. BMC genomics 23, 474, doi:https://doi.org/10.1186/s12864-022-08687-2 (2022).

PASNVGA 对应原编号 64；SDNN-PPI 对应原编号 65。编号 63 是原文对常规机器学习方法的集合性引用，不是 Table 3 中这两种深度方法的专属引用。文献条目按原稿保留，本次未进行外部核验。

## 在返修中保留时的边界

可按作者要求保留上述原数值，但必须说明它们来自 8.29 原稿的历史实验，未在本轮重跑。旧 CaPPI 行仅作为历史对照上下文，不能改标成本轮修复模型。原文同数据/同划分的声称也不等于其旧折成员与本轮固定折相同，因此不能将这些旧基线与本轮新 CaPPI 数字伪装成完全匹配的同批重跑比较。
