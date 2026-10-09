# CaPPI 数据来源、蛋白标识符与结构分析核查

核查日期：2026-09-30（Asia/Singapore）。本报告以新提供的 `CaPPI-code/SR投稿/CaPPI Manuscript_8.29.docx` 为当前稿，段落编号引用 `manuscript_829_extracted.txt` 的 P 编号。旧稿 `Manuscript V1.docx` 仅用于版本差异说明。所有序列和正文在本地处理，外部查询只使用公开 taxid、数据库链接和公开 locus 标识符。本报告未修改原始数据或稿件。

## 结论及证据边界

**当前本地数据不能支撑“HCC-associated human proteins”这一研究对象定义。** 这不是仅凭文件名的怀疑：582 条序列的内部 STRING 标识符、CSV 的逐条序列匹配、网络边的完整映射，与 NCBI/STRING/KEGG 的 taxid 对照形成一致证据。另发现病例表把连续内部编号当成基因 locus 编号，以及新版将单体 AlphaFold 3 预测写成复合物预测的问题。应先解决研究对象来源，再确定全文修订方向；不能通过语言润色把这些证据冲突隐藏掉。

**尚不能据此判断本地之外是否另有真正的人 HCC 数据集。** 如果存在，其来源、物种、protein ID、HCC 关联依据和与已报告模型结果的映射都需要重新核实。不能把现有细菌数据的结果直接转述为该数据集的结果。

## 1. 物种：已证实与稿件研究对象冲突

当前稿 P24 称首先收集了 582 个与 HCC 发生或进展有关的蛋白；P10、P18、P20、P141 等沿用 HCC-associated protein/network 定义。

本地实际核查结果：

| 核查对象 | 结果 |
|---|---|
| `1427984.protein.sequences.v12.0.fa` | 582 条序列、582 个唯一序列；所有记录的内部 header taxid 都是 `1427984` |
| 首条 FASTA header | `>1427984.X271_00001` |
| `肝癌 sequence.csv` | 582/582 条序列与 FASTA 精确匹配；CSV 内数字 ID 与 FASTA 行序逐条一致 |
| `肝癌pair不重复.csv` 中的 identifier/sequence 映射 | 582/582 条映射与 FASTA 精确一致 |
| `肝癌 pair.csv` | 47,414 条有向记录与原始 STRING links 的端点逐行完全相同 |
| `肝癌 interact.csv` | 23,707 条数字编号边经 FASTA 行序还原后，23,707/23,707 全部匹配原始 STRING 边 |

公开来源交叉核验（2026-09-30 打开）：

- [NCBI Taxonomy 1427984](https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=1427984)：当前名称为 `Candidatus Hepatoplasma crinochetorum Av`，lineage 包含 Bacteria、Mycoplasmatota、Mollicutes。
- [STRING X271_00484 条目](https://string-db.org/network/1427984.X271_00484)：明确标注物种 `Hepatoplasma crinochetorum` 与 NCBI taxonomy ID 1427984，并展示同一 X271 locus 系统。此实时页面当前重定向至 STRING 12.5；本地数据文件标为 v12.0，不能用实时页面替代原始版本的边内容。物种对照与本地边计数分别核验。
- [KEGG hcr 基因组记录](https://www.kegg.jp/kegg-bin/show_organism?org=hcr)：同一 taxid、582 个 protein genes、组装 `GCA_000582535.1`、染色体 `CP006932`；记录其来自陆生等足类 `Armadillidium vulgare`。

由此可确定：当前本地 FASTA 和映射后的训练网络属于上述细菌，不是人蛋白网络。目录名“肝癌”没有改变数据的物种身份；“582”也与该细菌基因组记录一致，但结论主要依据内部 ID 与完整内容映射，而非仅依据数量相同。

## 2. 正边来源与阈值：新版已修正旧稿的一项错误

原始 `1427984.protein.links.v12.0.txt` 的 header 为 `protein1 protein2 combined_score`。其数据包括 47,414 条有向记录，23,707 个唯一无向边，579 个有边节点，0 个自环，combined_score 范围 150–999。

| 保留条件 | 唯一无向边数 |
|---|---:|
| 所有原文件边，score >= 150 | 23,707 |
| score >= 400 | 7,572 |
| score >= 700 | 3,775 |
| score > 700 | 3,771 |
| score >= 900 | 2,224 |

旧稿 P22 的“23,707 high-confidence positive interactions with confidence scores exceeding 0.7”与原始文件不一致。**当前 8.29 稿 P24 已改为所有 STRING-recorded interaction pairs 去重后得到 23,707 条，未保留 0.7 阈值；不能再把旧稿阈值错误列为当前稿尚未修复的问题。** 新版这一数量描述在当前细菌数据上可复现。

但是，当前稿 P20 仍称网络由 curated and experimentally validated interactions 整合而成；这里只有 combined_score，无法据此验证全部边具有实验支持。STRING 官方页面说明其完整网络边代表蛋白关联，并不必然代表物理结合。当前数据应准确称 STRING protein associations；若研究确实要限定实验物理互作，需要对应 evidence channel 和筛选记录。

## 3. 病例表：内部行号被误写为 locus ID

当前 Tables 5、6 的 40 条 partner ID、模型分值和 Evidence 标签与旧稿 Tables 3、4 相同。对两种解释进行本地比较：

| 目标 | 按稿件字面 X271 locus ID 查询原始边 | 将末尾数字解释为 CSV 内部行号，再还原真实 locus ID | 还原后 score >= 700 的数量 |
|---|---:|---:|---:|
| X271_00001 | 2/20 | 19/20 | 0/20 |
| X271_00015 | 3/20 | 19/20 | 4/20 |

两个 target 本身的行号与 locus 后缀恰好相同；大部分 partner 因原 locus 编号存在缺号而不再相同。例如：

- 表5 rank 1 的 `X271_00478` 若实际表示内部编号 478，则真实蛋白是 `1427984.X271_00500`，源 STRING score 为 230；字面 `X271_00478` 与目标无本地源边。
- 表5 N/A 候选“445”的序列对应 `1427984.X271_00463`，不是稿件写的 `X271_00445`。
- 表6 N/A 候选“67”的序列对应 `1427984.X271_00076`，不是稿件写的 `X271_00067`。

**已确认的结构输入序列错配**进一步支持此行号/locus 混用问题（见下一节）。40 行的两套 ID 与源分数已保存为 `case_study_id_mapping.csv`，供作者逐条检查。不能在缺少原始病例推理日志的情况下直接覆盖原表，建议通过显式 ID map 重导表。

新版 P141–P142 已将病例正确收窄为 transductive retrospective ranking，明确 complete-network topology 暴露并放弃独立蛋白 holdout 的主张，这是相对旧稿的实质改进。行号还原后的“每个目标 19/20 有源库边”可复现，但仅属于与原数据源的一致性：目标1对应的19条源边score范围150–380；目标2为163–817。它们不能证明实验验证，Evidence 列仍使用 `Confirmed` 容易产生歧义，建议改为具有明确版本和记录ID的 `Recorded in STRING`。

本次未在已展开的研究代码中找到病例专用推理脚本、模型 checkpoint、40条排序结果的原始导出与逐项外部数据库证据表。此为“当前检索范围内未找到”，不是断言这些材料从未存在；归档包未逐个解包审计。

## 4. AlphaFold 原始输入：四个单体，非两个 AF3 复合物

当前稿 P146 写每对蛋白提交 AF3 进行 complex-structure prediction；P147 与 P149 进一步写 AF3 generated complex models。可见原始 job_request 与 summary 不支持这一描述。

| 原始 job 名称 | 输入序列长度 | 精确匹配的真实 STRING ID | 输入实体/拷贝数 | 五个模型 pTM 范围 | ipTM |
|---|---:|---|---|---:|---|
| `fold_00001` | 455 | `1427984.X271_00001` | 1 proteinChain / count 1 | 0.60 | 全部 null |
| `fold_445` | 350 | `1427984.X271_00463` | 1 proteinChain / count 1 | 0.94 | 全部 null |
| `fold_15` | 718 | `1427984.X271_00015` | 1 proteinChain / count 1 | 0.68–0.69 | 全部 null |
| `fold_67` | 421 | `1427984.X271_00076` | 1 proteinChain / count 1 | 0.85–0.86 | 全部 null |

路径分别位于 `分子对接/第一个/fold_00001`、`分子对接/第一个/fold_445`、`分子对接/第二个/fold_15`、`分子对接/第二个/fold_67`。`*_job_request.json` 都为 alphafoldserver dialect。20份 `*_summary_confidences_*.json` 的 ipTM 均为 null；单链 pTM 无法作为蛋白对界面可信度的证据。

进一步从实际 GRAMM 输入 PDB 的 ATOM 残基重建序列，验证：

| Docking文件 | PDB 残基序列精确匹配 |
|---|---|
| `第一个/GRAMMresults-112718-No1/ligand.pdb` | X271_00001，455 aa |
| `第一个/GRAMMresults-112718-No1/receptor.pdb` | X271_00463，350 aa |
| `第二个/GRAMMresults-112719-No2/receptor.pdb` | X271_00015，718 aa |
| `第二个/GRAMMresults-112719-No2/ligand.pdb` | X271_00076，421 aa |

因此当前记录可证实的流程是 **AlphaFold 3 单体建模，随后 GRAMM 对接**，并且真实对接蛋白对为 X271_00001–X271_00463 和 X271_00015–X271_00076。若作者还有其他双链 AF3 jobs，需提供其原始 request 和 confidence 文件，才能支持稿件现有说法。

## 5. 对接分数：构象文件存在，kcal/mol 数值来源尚未验证

两组 `GRAMMresults-*/README`、`receptor-ligand.res` 和 rank 1–10 PDB 模型均存在。`.res` 文件标记 `GRAMM v2.0 prerelease`，两组采用 grid step 3.5、rotation angle interval 10、输出30000 matches；grid size 分别64、128。其首位 Energy 数值分别594、631，并写有 `Energy values are divided by 15.0`，未提供 kcal/mol 单位或将这些值变换成 −1.1 / −9.1 的计算记录。

`分子对接/分子对接2.jpg` 的人工图中文字确实显示 −1.1 与 −9.1 kcal/mol；稿件亦重复这些数值。但图片标签不等于原始打分输出。本次检索未发现所用能量/亲和力再打分程序、版本、命令参数、所选 pose ID、原始日志或单位换算说明，亦未找到 Vina/PRODIGY/MM-GBSA 等相关脚本或日志。

据此不能确认两个 kcal/mol 数值的计算来源，也不能将 GRAMM score 直接解释为物理结合自由能。新版已删除“直接实验确证”式结论，但仍依据来源未明的分数推导 comparative structural plausibility；应暂时撤下数值与对应比较结论，或补齐可复核的原始评分记录后再重写。

## 6. 建议的修订处理

1. 先由作者确认用于已报告结果的真实物种与数据来源。若仅有目前这些文件，HCC 研究对象不成立；若有人 HCC 数据，必须另行建立结果与该数据的溯源。
2. 新稿保留其已明确的随机/degree-matched负采样区分和 retrospective 解释，不回退到旧稿“实验确认”或“新蛋白泛化”的措辞。
3. 病例表经原始内部ID映射重导，明确数据库版本与证据类型。已提供映射审计表，但不代替原始推理复核。
4. 将结构流程按证据改为 AF3 单体预测 + GRAMM 对接，并更正输入蛋白ID。缺少 kcal/mol 原始评分记录时不保留相关定量结论。
5. 这些问题属于数据、标识符和方法事实，须解决后才可形成可投稿的科学性终稿。可以立即制作标注问题的审阅稿；不能把审阅稿误称为经过验证的最终稿。

## 可复核产物

- `inspect_provenance.py`：本地只读核查脚本，读取原始文件，输出审计产物。
- `provenance_numbers.json`：原始文件 SHA-256、全部计数、AF输入/置信度、PDB序列匹配与逐行病例对照。
- `case_study_id_mapping.csv`：当前 Tables 5/6 的40行原ID、按内部编号还原的ID、两种解释下的源边分值。

脚本使用 Python 标准库，没有调用训练、修改原始数据或向外部发送序列/稿件。上述数字是对当前本地快照的实算结果，不是重新训练或重新预测的结果。
