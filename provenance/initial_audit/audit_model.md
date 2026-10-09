# CAppi / CaPPI 模型实现与论文一致性审查

## A. 最新权威稿与发布代码补审（优先于后面的历史发现）

用户随后明确最新稿为 `CaPPI-code/SR投稿/CaPPI Manuscript_8.29.docx`。以下以 `revision_20260930/manuscript_829_extracted.txt` 的 P 编号和 `CaPPI-code` 五个源码文件为主。后文历史报告中的 `P` 编号不能移用到新稿。

**已经修正，当前不应重复批评的内容：**新稿 P62–66 准确承认单 token softmax 恒为 1，使用 bidirectional residual cross-modal fusion 表述，列出四头、残差、FFN、归一化和拼接投影，基本符合 CAM/Cross attention 脚本；Algorithm 1 明确下游 MLP 使用固定的预计算融合特征；P141–142、P154–156 承认病例排序的传导式限制、非独立生物验证以及未评估未见蛋白。历史四输入脚本的 input3 错喂问题**不适用于发布版两输入 MLP**。

| ID / 严重性 | 最新稿位置 | 发布代码的直接证据 | 建议改法与是否需要重跑 |
|---|---|---|---|
| N01 / 关键 | P10、P24、P86–87、P112、P115、P152、P158：stratified five-fold，互斥 outer test，每折重算 train-edge 图 | `CaPPI-code/MLP/MLP.py:190–218` 只读取一次固定 train/val/test 文件；229–230 五次循环只重建分类器，未切换 fold；`DRW/DRW.py:8–9` 直接从整个肝癌 interact.csv 构图，没有 fold 或 train IDs | 若这些代码就是结果来源，须改称固定划分五次训练并重新审视图泄漏；若要保留五折与每折排除验证/测试边的结论，须提供真正实现、5套互斥ID和fold特征，或按所述协议重跑。不能靠改写说明把实验变成五折。 |
| N02 / 高 | P46–48、P57、Algorithm 1：one-hot与端到端GCNN | `GCNN/Gated Convolutional Networks.py:59–75` 为0–19 LabelEncoder、单通道float；78–90随机初始化后eval/no_grad，无train/load | 可按真实生成方式改编码/固定变换表述，但先闭合现有CSV来源；若希望保留监督学习GCNN、端到端和one-hot主张，需补代码及重跑。新稿虽Algorithm承认固定特征，P57仍保留end-to-end，文内自相矛盾。 |
| N03 / 高 | P26–32、P151：按访问频次连续惩罚，边权参与转移 | `DRW/DRW.py:12–25` 只有visited集合：存在未访邻居就均匀选未访，否则全部邻居均匀选；未累计访问次数或读边权 | 若保持既有结果，应改为实际visited-set优先规则和对应公式；若保留频次惩罚定义，必须实现并重跑嵌入、融合及下游实验。 |
| N04 / 高 | P59：64维DRW节点投影至256，256维GCNN；Algorithm融合256维 | `DRW/DRW.py:37–38` 得64维节点，但55–64只输出带两端ID的Hadamard边特征；`GCNN/...py:44–45,79` 读PBI sequence.csv并输出512维GateCNprotein_features_512.csv；`CAM/corss attention.py:45,76` 需要两个256维文件；`MLP/MLP.py:186` 又读fused_features_7_1500.csv。未发现64→256节点投影和512→256序列投影 | 补齐实际使用的脚本、ID映射、权重及文件对应，统一论文/代码的粒度与维度。不可把缺失投影写成已实现；不能直接顺序运行发布代码复现。来源不能恢复时，需要形成一致管线并重跑。 |
| N05 / 高 | P72、P78：MLP256/128/64、dropout0.3、权重L2、lr0.0005、EarlyStopping20、val_loss factor0.5 | `MLP/MLP.py:234–250` 是两端64/128分支，拼接后128/64/32，dropout0.4/0.4/0.1，activity_regularizer；262、265–273为字符串Adam默认、val_accuracy plateau patience10、30epochs，未显式lr/factor、无EarlyStopping；276–277保存最终模型。162–168的自定义scheduler未注册 | 若模型与结果来源核实就是此版本，可修改Methods/Algorithm为实际设置；若声称论文设置已执行，则必须给出匹配配置/日志或重跑。不要将函数里未被调用的lr×0.005误写成初始lr0.0005。 |
| N06 / 高 | P117：UMAP用于选择最终300epochs；P86外层测试仅最终评估 | 发布MLP为30epochs。历史可找到的 `参数分析/max_length/7/UMAP classification.py:119–124` 明确从测试集生成图，再拟合UMAP；新稿仍据图选择300 | 删除“由测试UMAP确定最佳泛化/最终epoch”的因果结论；说明图为事后描述，最终epoch必须按独立验证规则确定。若曾据测试图调参，需要独立未触碰测试集或nested重评估。 |
| N07 / 高 | Algorithm固定fusion；P45/P57/P124仍暗示生物语义学习；fusion可复现性 | `CAM/corss attention.py:76–78` 与 `Cross attention/corss attention.py:76–78` 均新建融合器直接no_grad，无load/train/seed/save weights；GCNN同理 | 固定融合并不代表预训练融合。明确权重从何处获得、是否随机初始化、是否跨fold相同及seed；现有文件来源不明时不能断言所有历史特征都是随机，亦不能声称受监督学习。复现需要权重或确定性再生与必要重跑。 |
| N08 / 中至高 | P59–66：同一蛋白两模态融合 | `CAM/corss attention.py:48–69` 与Cross attention副本按行对齐，短表补零，新ID从0生成，没有节点ID join | 提供原始蛋白ID与两侧行顺序的一致性证据；若错位或存在虚构零行则重算受影响特征和结果。当前只能确认脚本缺少身份校验，不能仅据此宣布历史数据错位。 |

### 最新代码的其他核查结果

- `CaPPI-code/GCNN/Gated Convolutional Networks.py:10,55,79` 硬编码kernel=3、MAX_LEN=2000、output_dim=512；与稿件及MLP读取文件名中的所选kernel/length配置也需统一。此目录每个模块仅有对应源码文件，没有随目录提供所需CSV或权重（本次目录清单确认）。
- CAM 与 Cross attention 两个源码逐行表现一致；均 `batch_first=True`，N是batch维，1是token维，未发现跨样本 attention。不能将此模块列为批次混合泄漏。
- `MLP/MLP.py:130–150` 静默跳过缺失节点，153–159依赖正负数量恰好各半；这些输入前提应显式校验。234–239两端分支宽度不同且无权重共享/对称池化，未规定无向对的排序或双向平均；交换端点可能改变预测，尚未做数值测试。
- 新稿对病例的传导式限制已有充分说明，不能据此推断普通cross-validation也已正确重建训练图；P86明确提出更强的每折train-edge-only协议，而发布DRW/MLP并未实现该协议。这两种场景应分别核验。
- 以上属于静态代码和现稿一致性结论。未据未知环境默认值武断指定Adam实际初始lr，也未把早期文件所有错误迁移到最新版。若已有不同真实运行版，优先用运行证据覆盖发布示例代码。

## B. 早期工作区审查记录（保留追踪，不能直接当最新稿批注）

审查日期：2026-09-30。范围：现有本地源码的只读静态审查；未修改实验代码、未导入会触发训练的模块、未训练、未重新生成实验特征或结果。读取了 `research-training-development/SKILL.md`，按实验协议和来源追踪要求审查。未检索 `venv`。以下行号均是本次检查时源文件的 1-based 行号，论文 P 编号来自 `revision_20260930/manuscript_extracted.txt`。

**结论边界：能够确认“当前脚本怎样运行”，不能据此断言所有已有 CSV 都由这些脚本的当前版本产生。特征 CSV、模型 H5、论文结果之间尚缺运行清单、权重和生成日志的闭合对应。**

## 1. 必须优先解决的问题

### M01 — 当前 GCNN 导出入口没有训练或加载权重，不能支撑端到端学习表述

- 证据：`Gated Convolutional Networks.py:78–90` 创建 `GatedConvNet`，随后立即 `model.eval()` 和 `torch.no_grad()` 前向导出；全文件 1–96 行未定义优化器、训练损失、反向传播或 checkpoint 加载。
- 实际结构：单通道 Conv1d 和门控 Conv1d，各 64 filters，kernel=7，same-like padding；`conv(x) * sigmoid(gate(x))` 后对所有长度位置取均值，再线性投影为 32 维（10–39、55、78–80 行）。主卷积分支没有额外非线性激活。
- 含义：若直接运行当前入口，它导出的是随机初始化网络的固定变换；`eval()` 本身不是训练或预训练。无法据此证明已有 `GateCNprotein_features_64.csv`、`Gated_256.csv` 等文件也属随机变换，因为名称、维度和生成历史未闭合。
- 论文受影响：P43、P52、P55 的任务自适应生物模式、去噪、端到端学习，以及 P16 的进化保守/结构感知说法均缺本实现的训练或结构监督证据。仅有序列卷积不能自行证明进化保守性或结构理解。

### M02 — 序列编码与论文 one-hot 描述直接不符

- 证据：`Gated Convolutional Networks.py:59–75` 使用 `LabelEncoder` 对 20 个氨基酸编码为 0–19，再转 float32，输入为 `(N,1,2000)`；不足长度补 0，超长截断。
- 论文 P44–46 描述 20 通道 one-hot；该输入合同与实际单通道数值编码不同。
- 其他实现限制：A 的编码和 padding 都为 0（60–66 行）；平均池化没有 padding mask（36 行）；非标准氨基酸无 unknown 路径（62 行）。应准确描述，不应把这些实现自行升级为 one-hot、mask 或 trainable embedding。

### M03 — 当前 CAM 入口未训练，且单 token 注意力退化

- 根入口 `corss attention.py:118–121` 创建融合器后立即 eval/no_grad；`cross attention/corss attention.py:76–78` 以及 `参数分析/max_length/7/1500/corss attention.py:76–78` 同样直接创建后 no_grad，均未加载权重。
- 张量合同：根文件 40–43、54–65 行，子目录版本 12–13、25–35 行均为 `batch_first=True`，把二维行特征 `(N,D)` 扩成 `(N,1,D)`；两个方向分别以 A/B 和 B/A 作为 Q/K/V。
- 数学后果：每个蛋白、每个 head 都只有一个 key。softmax 长度为 1，因此权重恒为 1；注意力输出不依赖 query/key 的相似性。注意力投影仍能变换 value，残差、LayerNorm 和 FFN 仍能融合两个输入，但这不是在多个残基/token之间自适应选择。P60、P65 中按网络相关性突出序列片段的解释不能由该实现支持。
- **未发现此 MHA 路径的跨样本注意力或批次依赖**：N 是 batch 轴，1 是 token 轴。不能把单 token 退化误报成跨蛋白注意力。
- 论文 P63–65 是双向加权聚合且含 weighting parameter；代码根文件 69–73 行和子版本 37–40 行为双向输出拼接、LayerNorm、Linear 投影，没有显式可学习标量融合权重。应按确定的实际版本重写公式和文字。

### M04 — 早期主分类脚本的第三路输入存在训练/评估不一致

- `dataset 分开/1. HBI training and predicting.py:249–254` 定义第 3 路为第一个蛋白的 sequence 特征，第 4 路为第二蛋白的 sequence 特征。
- 训练第 284 行却将 input3 和 input4 都设置为 `x_train_2_Behavior2`；验证 287–288 行和测试 298–301 行使用两个不同蛋白。训练和测试输入语义不一致。
- **版本边界**：`degree-matched negative sampling/degree_matched_negative_sampling_outputs/1. HBI training and predicting(1).py:295–301` 的 input3 已正确使用 `x_train_1_Behavior2`。不要把早期错误扩大为所有实验。已有 H5 和概率 CSV 是否由早期含错误版本生成尚未证明。

### M05 — 五次固定划分重复训练不能写成五折交叉验证

- `dataset 分开/1. HBI training and predicting.py:193–225` 在循环前一次性读取同一 train/validation/test 文件；236–237 行 `while CounterT < 5`，循环内重新初始化模型并复用这些数组。
- `参数分析/max_length/7/1500/2. Gated only.py:190–218,228–230` 和 degree-matched 脚本 206–238、248–250 行有相同结构。
- 应称固定划分上的五次重复训练；不能用输出文件名或 ROC 图的 `5fold` 名称反证。`dataset 分开/2. ROC-5fold 2.py:180,187` 的图名/表名确实使用 5fold/五折，但没有产生新折。
- 这属于协议命名问题，同时影响均值、标准差和泛化解释。论文若真采用另一套五折方案，需提供该版本的五折样本清单及运行依据。

### M06 — 训练架构、正则化、调度和轮次与论文并不一致

| 项目 | 论文表述 | 本地证据 |
|---|---|---|
| MLP 隐层 | P72：256/128/64 | `dataset 分开/1. HBI...py:241–264` 为输入分支 512/256/128/64，拼接后 hidden 128/64/32；`参数分析/max_length/7/1500/2. Gated only.py:234–250` 为两分支 64/128 后 hidden 128/64/32 |
| dropout | P72：各 hidden 后 0.3 | dataset 版本仅前两 hidden 为 0.3（261、263）；参数分析版本为 0.4/0.4/0.1（246、248、250） |
| L2 | P72/Algorithm：权重惩罚 | 两种 Keras 版本均用 `activity_regularizer=regularizers.l2(0.0001)`，正则对象是层输出激活，不是 kernel 权重 |
| early stopping | P78：patience=20 | dataset 280–290 和参数分析 265–273 只有 ReduceLROnPlateau，无 EarlyStopping、无恢复最佳权重 |
| LR 调度 | P78：val_loss plateau、factor=0.5 | 源码监控 `val_accuracy`、patience=10，未显式设置 factor；顶部自定义 scheduler 未挂入 callbacks，不能把其倍率当实际学习率 |
| 最终 epoch | P112：300 最佳 | dataset 290 行为 50；参数分析 273 行为 30；保存的都是最终模型（dataset 293–294、参数分析 276–277） |
| 训练模块 | CAM 与 GCNN 联合学习 | dataset 分类器仅读取两张特征 CSV，然后构建四分支拼接 MLP（185–189、241–277），计算图中没有 GCNN/CAM |

补充：`参数分析/max_length/7/UMAP classification.py:88–114` 是另一种两分支、两个 hidden 的模型；119–124 行从测试集抽特征并重新拟合 UMAP。测试图中簇分离情况不能作为“300 epoch泛化最优”的独立验证依据，也不是同一主模型版本的直接训练对照。

## 2. DRW 与 OpenNE 路径需要明确区分

### 2.1 根目录 Diversified Random Walk

- `D RW 节点特征.py:7–8` 从完整 `肝癌 interact.csv` 构图；11–25 行维护当前 walk 的 visited set，若存在未访问邻居则只在该集合均匀抽取，否则在全部邻居均匀抽取。
- `D RW 边特征.py:8–26` 实现同样规则。两者都没有访问次数计数、没有基于次数的连续惩罚系数、没有边权转移；故与论文 P24–29 的“按访问频次与边权动态调整概率”不完全相符。
- 节点版本 29–36 行：每节点 10 次、walk length=20、embedding=64、window=5、sg=1、workers=4、epochs=10，输出 `node.csv`（46 行）。
- 边版本 30–37 行：10 次、20 长度、64 维、window=5、sg=1、workers=4；使用旧 gensim `size=64`，未指定 epochs。40–60 行使用 Hadamard product 形成边特征，仅遍历原始已知边，输出 `cancer_link_features.csv`。
- 该读取方式没有按 train split 筛边；若直接把其特征用于对同一图中留出的 target edges 预测，则存在目标边已进入构图的泄漏风险。当前特征文件的实际来源需由数据审查和运行记录另行确认。

### 2.2 常规 OpenNE DeepWalk

- `OpenNE-master/src/openne/__main__.py:129–132` 把 `deepWalk` 交给 `Node2vec(...,dw=True)`；`node2vec.py:12–19` 使用 `BasicWalker`；`walker.py:27–33` 每步在所有邻居中随机选择，没有 diversified visited constraint。
- `node2vec.py:25–34` 的 Word2Vec 训练确实存在；不要将“GCNN/CAM 没有训练”扩大成“所有模块均未训练”。
- CLI 默认：walks=10、length=80、workers=8、dim=128、window=10（`__main__.py:29–40`）。`--epochs` 说明用于 LINE/GCN，DeepWalk 调用没有传入该参数（41–42、129–132）；不能把 CLI epochs 直接当作 DeepWalk 的训练轮数。
- CLI 在 178–181 行设 Python random 与 NumPy seed=32。root DRW 和 GCNN/CAM 导出入口没有设 seed；未发现这些步骤持久化权重或运行 manifest。

### 2.3 Behavior.py 只是重排/补零，不能作为自定义 DRW 的证据

- `OpenNE-master/Behavior.py:49,53–54,60–88` 读取 `PBI list.csv` 和 `PBi_node2vec.txt`，按节点 ID 重排预先存在的嵌入并为缺失节点补零，输出 `PBI_Behavior_node2vec64.csv`。
- HCC 专用 `OpenNE-master/data/new/Behavior.py:49,53–54,60–88` 读取 `肝癌 list.csv` 与 `cancer_deepWalk_Behavior.csv`，做同类映射，输出 `cancer_Behavior_feature.csv`。
- 二者本身不生成 walk、不训练 embedding。`OpenNE-master/命令和参数.txt:33` 还保存过 HCC 的 LINE 命令。文件名带 Behavior 不能证明所有特征均来自论文 DRW。

## 3. 数据和模型接口的未闭合项

- 子目录融合脚本按行对齐、较短文件末尾补零，然后产生 0-based `id`（`cross attention/corss attention.py:48–69`）。这里没有蛋白 ID join；相同行数不代表相同节点顺序。root 融合入口虽可选择 ID join（89–96），但缺 ID 时会截断到最短长度并按行对齐（97–106）。
- 已检查 `参数分析/max_length/7/1500/fused_features_7_1500.csv` 开头：当前文件为无 header、ID 从 1 起；同目录生成脚本 68、81–83 行则写 0-based ID 且带 `id,f1,...` header。这证明至少存在未记录的后处理或另一生成版本；不等同于证明现有映射错误。
- root GCNN 输出保留 protein_names 首列（93–94 行），而 cross attention/Gated_256.csv 当前首行是纯数值特征，无 ID。这也需要来源和 ID 映射文件。
- `dataset 分开/DNN training.py:23–30` 与 xgboost training 25–33 行把两列输入理解成 `id,label`，然后覆盖 label，仅按一个 id join 特征。它们没有显式使用两个蛋白的 pair endpoints；若输入实为蛋白对，两者会丢弃第二端。当前目录没有 `pos_train_fold1.csv` 等文件，故不能认定这一尝试入口构成已完成五折主实验。
- `dataset 分开/DNN training.py` 是明确训练的下游 128/128/2 分类器，CrossEntropyLoss，Adam lr=0.001，30 epochs，batch=64，dropout=0.3（12–14、35–44、96–102）。其特征入口与 Keras 版本不同；不应混写。
- `GenerateBehaviorFeature` 在两端未匹配到节点时静默略过，未同时过滤 pair 和 label（dataset 130–150）；`MyLabel` 只按样本总长的一半建立正负标签（153–159），依赖正负平衡且先正后负。应把这些作为输入合同，在后续修复中加入验证，不能静默改标签。
- Keras 两端分支使用不同宽度和不同参数，未显式对称化；交换 `(u,v)` 可能改变分数。对于无向 PPI，论文需说明 pair 排序约定或对称预测方案。此次未做数值交换测试。

## 4. 建议的论文处理边界

1. 可以立即修正：one-hot 与实际编码、分阶段特征与端到端训练、固定划分五次重复与五折、已核实版本的架构/激活正则/调度/轮次、DRW visited-set 规则、fusion 的拼接投影形式。
2. 暂时不能写成已证明：任务训练的 GCNN/CAM、按残基或维度选择的注意力解释、由注意力产生的可解释性、300 epoch 的最佳泛化、由当前源文件能唯一复现实验表、所有历史 CSV 均为随机初始化输出。
3. 若作者希望保留原算法主张，需要先确认实际提交版本及完整权重；如果当前源码就是唯一实验来源，则必须以可追踪的新版本补齐训练、tensor设计、split内构图和受控重跑，再更新结果。此次审查未执行这些方法变化。
