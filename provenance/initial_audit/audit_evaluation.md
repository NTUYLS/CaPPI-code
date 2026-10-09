# CaPPI 评估与结果审计（2026-09-30）

本报告以 `CaPPI-code/SR投稿/CaPPI Manuscript_8.29.docx` 为当前稿，段落号对应 `manuscript_829_extracted.txt`。旧 `Manuscript V1.docx` 仅用于辨认历史来源。只读源代码、split、预测 CSV；未训练、未修改原文件。已复算 265 个逐样本预测 CSV，审计 34 组 split（含 26 组参数实验复制件）。数值与 SHA256 见同目录 `audit_evaluation_statistics.json`、`recomputed_prediction_metrics.csv` 和 `recomputed_prediction_summary.csv`。审计代码为 `audit_evaluation.py`，仅依赖 NumPy 和标准库。

## 1. 当前论文宣称的五折、逐折 train-only 建图未被当前实现支持（必须修正）

- 新稿 P86–P87 明确称外层 stratified five-fold、每例仅测试一次、每折重建仅含训练正边的 DRW 图。
- 当前 `CaPPI-code/MLP/MLP.py:190–226` 只读取一次固定 Train/Validation/Test 文件，`:229–230` 再重复训练五次。循环内没有新划分，也没有 DRW 重训练。`:186` 从一个固定 `fused_features_7_1500.csv` 加载全部节点特征。
- 此脚本 SHA256 为 `BE667FCA062A6E16EC1DC8AE5F3F438BCA33ED816118371C999F330190D709F2`，与 `参数分析/max_length/7/1500/2. Gated only.py` 完全相同。`CaPPI-code` 代码目录没有提供任何 CSV，故其实际启动时使用哪个工作目录仍需运行记录确认。
- 新补充实验 `degree-matched negative sampling/degree_matched_negative_sampling_outputs/1. HBI training and predicting(1).py:206–250` 同样固定读取 split 后循环五次。
- `CaPPI-code/DRW/DRW.py:8–9` 从完整 `肝癌 interact.csv` 建图。该文件恰好包含全部 16,594 条训练正边、2,371 条验证正边、4,742 条测试正边；根目录 `D RW 节点特征.py:7–8` 同样如此。
- 目前可支持的表述是“在固定 70/10/20 划分上五次重复训练”，不能将重复称五折。完整图脚本存在直接的 held-out edge 泄漏路径；特征缓存没有逐折图 provenance，不能断言现有结果已排除该路径。共享节点本身属于 transductive 设置，直接保留待预测正边是额外问题，二者应区分。

## 2. 历史随机负样本包含已知正边、反向重复和跨集合重复（新稿 P24 与数据不符）

所有主 split 每类 Train/Validation/Test 数分别为 16,594 / 2,371 / 4,742，比例 1:1；没有发现 10:1 配比。正边自身无重复、无自环，三个正集合无交叠。以下是 `dataset 分开` 的**负样本**审计，按无向边比较：

| 问题 | Train | Validation | Test |
|---|---:|---:|---:|
| 与全部已知正边冲突的无向边 | 1,405 | 204 | 395 |
| 反向重复导致的多余行 | 370 | 11 | 40 |
| 自环行 | 22 | 5 | 16 |

负集跨 Train/Validation 重叠 116 条，Train/Test 重叠 237 条，Validation/Test 重叠 31 条。`消融实验` 六个 split 的 SHA256 与上述历史 split 完全相同；全部 26 个含 split 的参数实验目录也复制同一批 split。逐目录结果已输出 JSON，不能把复制目录数当作独立实验数。

两个可直接核对的例子：

- `dataset 分开/NegativeSample_Train_肝癌.csv:23` 是 `446,92`，而 `PositiveSample_Test_肝癌.csv:1838` 是 `92,446`。
- `dataset 分开/NegativeSample_Test_肝癌.csv:2` 是 `498,333`，而 `PositiveSample_Train_肝癌.csv:5033` 是 `333,498`。

原因可定位至 `dataset 分开/split2.py:25–38`：有向行列矩阵仅屏蔽已录入方向，没有无向 canonicalization、自环排除和反向排除。不能仅在论文中声称这些步骤已执行；应重建负集并重跑关联实验，或对原有结果作明确降级说明。不得只删测试错误行后把筛选出的高分当作正式新结果。

## 3. 新 degree-matched 负集质量有改进，但还不是单因素受控补充实验

该新目录的六个 split 经核查：负边无自环、无无向重复、无正负冲突、三个负集合互斥；正 split 与历史主实验完全一致。这一点可写入论文，但不能泛化到历史随机负集。

采样器 `degree_matched_negative_sampling.py:48–62,176–250,354–419` 使用全体正图度分布及候选非边，搜索 degree weighting 强度；固定 seed 20260729，41 个 alpha、每个重复 5 次，选 alpha=2.9。summary 的 endpoint-degree histogram Pearson r 为 0.920736，Wasserstein 距离 13.078458，KS 距离 0.113258，正/负端点平均度分别 140.031763 / 130.490825。该相关系数是**直方图相似性**，不是逐节点正负度一一相等，也不能单独证明消除了 degree bias。

最重要的控制问题：新增训练脚本 `1. HBI training and predicting(1).py:198–202,254–284` 使用分别加载的 DRW、GCNN 两套缓存，四路 Dense 输出 512/256/128/64 后直接拼接；当前 `CaPPI-code/MLP/MLP.py:186,233–251` 使用 fused cache、两路 64/128 后接 MLP。两者 epoch 分别 50 与 30，dropout 0.3 与 0.4/0.4/0.1。故原随机结果与 degree-matched 结果之间同时存在输入融合、分类器容量和训练配置变化；新稿 P112、P154 不能把下降全部归因于负采样。

新版 degree-matched 脚本已修正历史 HBI 脚本的 input3 错接问题：旧 `dataset 分开/1. HBI training and predicting.py:284` 将 input3 和 input4 都接到蛋白 2 序列特征，新脚本 `:297` 正确使用蛋白 1。请勿把此历史 bug 归罪于新版补充实验，也不要据此认为两次实验只有负样本变化。

## 4. 机器学习比较的测试集与标签存在可复现的严重错误

`机器学习分类器/机器学习.py:12–20,54–66` 及 `xgboost_classifier.py:13–21,61–76`：

1. `pd.read_csv` 未设置 header=None，将每个无表头 split 的第一条数据当表头。
2. 测试集合被构造成 `positive_test + negative_validation + negative_test`。
3. 根据总长度前半赋正标签、后半赋负标签。

每个 run 实际输入 4,741 个正测试行和 2,370+4,741 个负行，共 11,852 行，但前 5,926 行被标为正例，即 1,185 个负例会被误标为正例。保存的 AdaBoost/KNN/LR/RF/SVM/XGB 预测 CSV 均为 11,852 行、5,926 个标签 1，与此错误路径吻合。CaPPI 是 9,484 行。`split2.py:53–84` 每次重新随机划分并重新负采样，也不是互斥五折。该比较不支持新稿 P82/P121 的“相同预定义划分与相同评估条件”。

这些预测 CSV 没有蛋白 pair ID、split 指纹、seed、checkpoint 标识，因此不能单凭长度吻合证明每一行来源。应保留错误现象和条件性来源判断，并重做比较。不要直接改现存预测 CSV 标签冒充重跑结果。

按现存标签复算，RF 的 AUC=0.853727、AP=0.871746，XGB 的 AUC=0.851514、AP=0.866038。因此新稿 P122“XGBoost 是最强常规模型”即使在其现存文件口径下也不成立。上述数字只是审计结果，不可作为经过修复的正式性能。

## 5. 训练设置、模型选择、学习率和结果口径不一致

- 当前 `CaPPI-code/MLP/MLP.py:262–277` 使用 `optimizer='adam'`，没有显式设置 0.0005；回调仅 ReduceLROnPlateau(monitor='val_accuracy', patience=10)。没有 EarlyStopping、restore_best_weights 或 ModelCheckpoint，保存的是最终 epoch 模型。
- `scheduler` 在 `:162–168` 被定义，但没有加入回调；不能以该函数的乘数证明某次学习率实验配置。
- 实际 epochs=30；degree-matched 为 50。新稿算法写 early stopping patience=20、初始 LR=0.0005，P117 写最终 300 epochs，需要用实际运行配置/日志证明，不能靠论文文字补齐。
- 参数分析保存的输出为测试预测，未找到足以重现“以验证集选择每个超参数”的验证指标/选择日志。不能直接把参数测试分数最高等同于无偏的验证选择。

## 6. 能复算的结果与论文表的对应关系

下表报告从现存逐样本 CSV 重新计算的五次均值；只是确认文件数值，**不证明评估设计有效**。sample SD 已另存 summary。

| 来源 | 每次 n | ACC | MCC | F1 | mean AUROC | mean AP | mean trapezoid PR area |
|---|---:|---:|---:|---:|---:|---:|---:|
| 论文主结果，modelcompair/Cappi | 9,484 | 0.891987 | 0.785554 | 0.895251 | 0.951362 | 0.937558 | 0.937546 |
| 历史 dataset 分开 | 9,484 | 0.897933 | 0.796436 | 0.899539 | 0.954704 | 0.940053 | 0.940037 |
| degree-matched 新补充实验 | 9,484 | 0.783045 | 0.566343 | 0.779785 | 0.866186 | 0.878163 | 0.879189 |
| 消融 Behavior 文件 | 9,484 | 0.822522 | 0.646350 | 0.823642 | 0.900182 | 0.892914 | 0.892892 |
| 消融 GCNN 文件 | 9,484 | 0.843083 | 0.688893 | 0.848274 | 0.920812 | 0.908111 | 0.908086 |
| 消融 CAM 文件 | 9,484 | 0.741333 | 0.482938 | 0.739896 | 0.810458 | 0.805649 | 0.805576 |

对应及不一致：

- 新稿 Table 1 的 LR=0.0005 行、Table 3/4 的 CaPPI 行可追溯到 `参数分析/max_length/7/1500/0.0005/肝癌_五折的表_7_1500_0.005.csv`（末尾文件名标 0.005，但目录为 0.0005）。主结果的副本见 `机器学习分类器`、`消融实验/融合AUC值` 和 `modelcompair/Cappi`。
- 这些表先将各 run 阈值指标四舍五入至 4 位，再计算 mean/SD，故直接从全精度预测复算 SD 有极小差异，例如 F1 的 0.001163 与表列 0.0011；应选择统一的“全精度计算，最终展示舍入”规则。
- 新稿 Table 2 的各 run 阈值指标与 degree-matched CSV 可匹配。但“Fold 1–5”实际是固定划分重复，应改为 Run 1–5，除非另有真实互斥五折结果。
- 最新稿 degree-matched AUPR=0.8782 对应 **average precision**，其现有 `肝癌_五折的表.csv` AUPR=0.8792 对应梯形 PR 面积。`3. PR-5fold.py:228` 与 `2. ROC-5fold 2.py:143` 使用不同定义。两者都可计算，但必须声明并统一。
- Table 4 的 w/o DRW（ACC .8225）与 Behavior 预测数值吻合，文件名并不能证明它是 sequence-only，需提供产生该 CSV 的训练入口。
- Table 4 的 w/o GCNN（ACC .8407）吻合 `消融实验/gated_五折的表.csv`，却不吻合 `消融实验/GCNN` 当前预测（.843083）。Table 4 的 w/o fusion（ACC .7454）吻合 `消融实验/CAM/CAM_五折的表.csv`，但同目录当前预测给出 .741333。存在旧 summary 与新预测混用；在找回对应原始 run 前不能任选一组数字替换。
- `消融实验/2. Gated only.py:186,233–273` 当前加载的是 `cancer_line_feature.csv`，网络 32/16、epochs=2；它不能支持“仅删除 fusion、其余训练策略相同”的文字，也不能证明当前 CAM summary 的训练来源。
- `modelcompair/plot_three_models_auc_aupr(5).py:78–82` 图例值是 DISPLAY_VALUES 手写常量。现存 `auc_aupr_comparison/AUC_AUPR_summary.csv` 的 PASNVGA AUC=.845654，但图例/稿写 .8547；该问题应与 baseline 审计一起处理。CaPPI mean AUC=.951362 通常应舍入 .9514，稿写 .9513；不能拿硬编码图例作为数值依据。

## 7. 可直接用于当前批注稿的短批注

1. **P24 负采样**：原随机负集未实现文中描述的排除规则；审计发现训练/验证/测试分别有 1,405/204/395 条负边与已知正边冲突，另有反向重复、自环和跨集合重叠。请重建规范的无向负样本后重跑，或明确披露局限；不得仅修改文字称已排除。
2. **P86–P87 评估流程**：当前 MLP 代码只在固定 70/10/20 split 上循环五次，DRW 脚本读取完整图；不支持互斥 stratified five-fold 或逐折仅训练边建图。请提供真实逐折索引/图指纹/日志，或改写成已执行的固定 holdout 重复，并重新评估 target-edge leakage。
3. **P112 补充实验**：degree-matched 负集通过无向去重/正负排除检查，但补充训练模型与主模型在融合方式、网络容量、epoch 和 dropout 上不同，下降不能单独归因于 degree matching。
4. **P82/P121–124 分类器比较**：基线代码把负验证样本并入测试并按前后各半赋标签；每 run 1,185 个负样本被错标，且基线测试 n=11,852，CaPPI n=9,484。需要在同一正确测试集合上重新比较，现有“公平比较”“最佳模型”论断不成立。
5. **训练段落及 Algorithm 1/P117**：代码使用固定 30 epoch 和 val_accuracy 降学习率，无 early stopping、无 best-checkpoint 选择、未设置文述 0.0005。请用实际日志统一 30/50/300 epoch、学习率、回调及选择依据。
6. **Table 4/消融**：w/o GCNN 与 w/o fusion 汇总表不能由同目录当前预测复算，且缺少保持相同容量和训练预算的对应入口。请按 run manifest 找回准确结果，暂不把差异归因于所删除模块。
7. **AUPR、图例及 Table 1–3**：AP 与梯形 PR 面积混用（degree-matched .8782 vs .8792）；模型比较图例有手写数值，PASNVGA .8547 与保存 summary .845654 不符。请由统一的逐样本结果自动生成全部表图。

## 范围限制

本报告没有重训练、没有修正标签、没有声明任何修复后性能。没有把旧随机负样本的缺陷移植到已通过这些检查的 degree-matched 新负集。图缓存缺乏 provenance 时，表述为“代码泄漏路径与缓存来源未证实”，不伪称已经证明每个缓存均由该脚本生成。物种、582 蛋白的 HCC 来源以及结构案例由另一份 provenance 审计专门处理。
