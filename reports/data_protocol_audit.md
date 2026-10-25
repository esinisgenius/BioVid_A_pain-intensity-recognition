# BioVid-A — Data Protocol & Experimental Validity Audit

审计日期：2026-10-25  
审计范围：Phase 3–8 的数据协议、标签定义、评估与解释有效性  
执行方式：只读检查代码、配置、已有实验结果、原始 metadata/信号文件与公开文献；未训练模型、未重跑大型实验、未改动既有代码或产物。

## Executive assessment

本审计没有发现 Phase 3–8 主体 subject-independent 划分存在 subject overlap，也没有发现 Phase 8 的 fold 内 imputation/scaling 泄漏。现有结果仍可作为“BioVid Part A 固定 5.5 秒片段上的刺激等级预测”结果保留。

但有四个必须进入后续论文表述的限制：

1. **Confirmed / P0：BL1 不是刺激前的中性静息基线，而是 PA1（最低刺激）之后 8–12 秒休息期中抽取的片段。** 因此 BL1/PA4 的高 AUC 是“PA1 后恢复片段 vs 高强度刺激片段”的协议内可分性，不能直接外推为一般静息状态与高疼痛的可分性。
2. **Unknown / P0：文件名末尾序号未被本地文档明确声明为采集时间。** PA1–PA4 的 `001–080` 很可能保留随机刺激序列，但仍缺少事件日志或连续记录的直接证明；BL1 的 `081–100` 则确定是导出/类别编号，不能是全实验的真实时间顺序。
3. **Confirmed / P1：Phase 8 的 within 与 cross 比较没有 preprocessing leakage，但训练信息量极不对称。** within 每 fold 只有 32 个训练 trial；cross 为 2,760 或 2,800 个训练 trial。因此 `AUC_within - AUC_cross` 不能识别纯粹的 subject-transfer effect。
4. **Likely / P2：Part A 片段从刺激触发约 3 秒后开始，覆盖高温平台的主要部分与早期降温，但没有完整的刺激前基线和晚期恢复。** 该窗口适合窗口内幅度/趋势特征，不足以完整观测 SCR onset、peak、recovery，也不适合传统 HRV 或恢复动力学结论。

综合判断：相邻 PA 等级难分、EDA 在当前片段中贡献较大、个体异质性存在等方向性结论仍成立，但必须限定为**校准热刺激等级和 Part A 观测窗下的预测关联**。最需要弱化的是 BL1 的“中性基线”解释、BL1/PA4 的一般化含义、Phase 6 baseline subtraction 的无泄漏/部署解释，以及 Phase 8 gap 对 transfer failure 与 within-subject overlap 的因果归因。

## 1. 审计方法与证据等级

证据优先级为：本地原始数据说明与 metadata → 仓库实际代码/产物 → BioVid 官方页面 → 数据集原始论文 → 后续方法论文。状态定义：

- **Confirmed**：由本地文件/代码直接确定，或由 BioVid 官方资料及原始论文明确陈述。
- **Likely**：多项证据一致，但缺少原始事件日志、连续记录或一手字段定义。
- **Unknown**：现有仓库与公开资料不足以判定；报告同时列出所缺证据。

风险等级沿用请求中的优先级：P0 为会改变核心解释，P1 为影响比较有效性或不确定性，P2 为影响特征可观测性与外推范围。

审计开始时 Git 状态为 clean，分支为 `research/phase8-within-cross-separability`，HEAD 为 `eee8197`；最近 Phase 8 提交依次为 `eee8197`、`8f85bab`、`5c6fa19`、`9e99162`、`719d4cf`。本审计未切换分支、未提交、未执行 push/merge/rebase/reset。

## 2. A — BL1 标签与基线定义（P0）

### 2.1 证据与发现

| 状态 | 发现 | 证据 |
|---|---|---|
| **Confirmed** | Part A 的四个刺激强度各重复 20 次，顺序随机；最高温保持 4 秒，刺激间休息 8–12 秒。 | BioVid [官方数据页](https://www.nit.ovgu.de/BioVid.html) “Overview”；Walter/Werner 等的原始数据集论文，[DOI 10.1109/CYBConf.2013.6617456](https://doi.org/10.1109/CYBConf.2013.6617456)。 |
| **Confirmed** | BL1/BLN 来自最低强度 PA1 后的休息期，不是每次刺激前统一抽取的 pre-stimulus baseline。 | Werner et al. 2014 明确写明 “rest periods following the lowest intensity stimuli” 被选为 baseline，实验段再次写明 BLN “extracted from the pause after PA1”：[*Automatic Pain Recognition from Video and Biomedical Signals*](https://doi.org/10.1109/ICPR.2014.784)，Sec. I-C、V；本地 `starting_point/Readme.txt:24-26` 只给出 baseline/no-pain 标签，并未给出采集来源。 |
| **Confirmed** | 每位 subject 有 20 个 PA1 与 20 个 BL1，因此 BL1 在类别层面与 PA1 后恢复期有对应关系。 | 原论文 Sec. V 说明每类 20 样本；本地 `reports/data_audit.md:43-60`、`starting_point/samples.csv`。 |
| **Unknown** | 无法确认 `BL1-081` 精确对应哪一个 PA1 trial，也无法确认 BL1 片段在该次 8–12 秒 pause 内的起点。 | 本地 `samples.csv` 没有 parent-trial/event ID；每个片段时间戳相对各自视频首帧重置（`starting_point/Readme.txt:46-53`）；仓库没有连续实验时钟或导出脚本。需要原始 event log、Part C 连续温度/信号或数据发布方说明。 |
| **Confirmed** | 本地 BL1 温度片段是稳定的约 32°C 状态；PA 片段从高温平台附近开始并在末段降温。 | 对 `temperature/*/*_temp.csv` 的只读全量核查：BL1 温度范围中位数 0.088°C（95% 分位 0.118°C）；各 PA 类片段起点中位数约 46.97/47.79/48.80/49.94°C，末点约 31.89/32.41/33.23/34.05°C。文件结构与 32 Hz、176 行见 `reports/data_audit.md:149-165`。 |

BL1 的“no pain”是**实验协议标签**：温度已回到基线且处于 PA1 后休息期。它不等价于独立、无前序刺激的静息生理状态；是否仍有 PA1 引发的自主神经反应或恢复效应，现有短片段不能排除。

### 2.2 对 Phase 3–8 的影响

- **Phase 3 基线与 Phase 4 modality ablation：可保留但需限定。** 5-class 指标仍正确描述 Part A 五种协议片段的分类；BL1 recall、EDA 对 BL1 的贡献可能同时利用刺激/恢复阶段差异，不应解释为普适“无痛生理状态”的识别能力（`reports/baseline_results.md:5,45-49`；`reports/cross_subject_stability.md:28-30`）。
- **Phase 5/5.5 attribution 与 BL1/PA4 高 AUC：需弱化。** BL1/PA4 的 Logistic ROC-AUC 0.881、XGBoost 0.902 是有效的协议内结果（`reports/adjacent_class_analysis.md:42-55`），但比较同时改变了刺激强度、热刺激阶段以及 PA1 后恢复历史。EDA delta/slope/SCR attribution 尤其可能编码这种时间窗状态。偏差方向未知：PA1 carryover 可能使 BL1 更接近疼痛类而降低 AUC，也可能由稳定 32°C/恢复阶段提供额外可分信息而提高 AUC；没有 matched pre-stimulus baseline 时不能判定。
- **Phase 6 生理轨迹：PA1–PA4 部分可保留；所有以 BL1 为端点的轨迹需弱化。** 相邻 PA 边界方向一致性不依赖 BL1 定义。`BL1→PA*`、特别是 `BL1→PA4` 的高度方向一致性（`reports/cross_subject_stability.md:48-79`）不能被称为从“中性静息”到疼痛的纯强度轨迹。
- **Phase 6 baseline subtraction：高风险，须重新命名为 oracle/transductive BL1 calibration pilot。** 实现先用每位 subject 的全部 `class_id == 0` trial 求均值，再从该 subject 的所有样本扣除（`src/models/run_cross_subject_stability.py:218-226`），并在 grouped evaluation 之前对整表执行（`:329-344`）。这使用了 held-out subject 的 BL1 标签和全部 20 个 BL1 样本；若 BL1 本身也进入测试，其均值还包含被评估样本。官方规则允许 subject-specific transform 使用全 subject 数据，但明确要求**不得使用标签**（[BioVid evaluation procedure](https://www.nit.ovgu.de/BioVid.html)）。因此它不是严格的 label-free subject-independent preprocessing。现有结果是负结果（5-class macro F1 0.304→0.296；`reports/cross_subject_stability.md:97-106`），不会解释为夸大了主模型正结果；可保留的结论仅为：**在已知 20 个 post-PA1 BL1 片段的 oracle calibration 条件下，减去静态均值仍未改善相邻边界。** 它不能证明可部署 baseline normalization 无效，也不能排除 pre-stimulus 或动态恢复校正。
- **Phase 7：模型间比较仍有效，但涉及 BL1 的累计边界和 BL1/PA1 AUC 是协议特定结果。** PA1/PA2、PA2/PA3、PA3/PA4 的相邻结论不受 BL1 来源直接影响（`reports/boundary_gated_ordinal.md:123-165`）。
- **Phase 8：BL1/PA4 control 不能作为纯“宽疼痛间隔”对照。** 该 pair 同时是强度间隔和协议阶段对照；PA1/PA3 才是更干净的宽间隔 stimulus-only control。

### 2.3 风险结论

**P0 / 高解释风险，低代码正确性风险。** 标签实现与官方 Part A 定义一致；问题不是标签错置，而是把 post-PA1 recovery baseline 解释成了独立中性基线。需要修改论文语言与重新验证基线相关机制结论，不需否定所有预测指标。

## 3. B — Trial ordering（P0）

### 3.1 本地编号结构

只读核查 `starting_point/samples.csv` 得到：每位 subject 的 80 个疼痛 trial 使用 `001–080` 且**每个序号仅出现一次**，四个 PA 类在这些序号中交错；20 个 BL1 使用 `081–100`。`src/models/run_within_cross_separability.py:57-77` 也对这一结构作了硬验证。

这里需纠正一条既有文档事实错误：`reports/data_audit.md:102-107` 写道 pain trial numbers “reused across pain classes”，但实际每位 subject 在 PA1–PA4 合计范围内 `001–080` 无重复。由于本次约束只允许新增本报告，原文件未修改。

| 状态 | 判断 | 依据与边界 |
|---|---|---|
| **Confirmed** | `001–080` 是每 subject 唯一、跨 PA 类交错的 pain sample index；`081–100` 是单独编号的 BL1 样本。 | `starting_point/samples.csv`；Phase 8 校验 `src/models/run_within_cross_separability.py:67-77`。 |
| **Likely** | PA1–PA4 的 `001–080` 保留随机刺激序列/导出序列，因其类别交错方式与官方“80 次随机刺激”协议吻合。 | 官方随机顺序协议；原论文说明 CV 随机分区用于降低 sample order 影响（Werner et al. 2014, Sec. V）。但本地 Readme 没有把 suffix 定义成 chronology。 |
| **Unknown** | suffix 是否严格等于真实 acquisition order，以及相邻 suffix 之间是否保持连续 wall-clock 时间。 | 缺少导出脚本、全局时间戳、事件日志和连续温度序列。每个短片段时间戳相对自身视频首帧，不能跨片段排序（`starting_point/Readme.txt:46-53`）。 |
| **Confirmed** | BL1 `081–100` 不可能代表相对于全部 100 片段的真实后置采集顺序。 | 原论文确认每个 BL1 来自散布于 80 次随机刺激中的 PA1 后 pause；故把全部 BL1 编在 81–100 是导出/类别整理，而非实验末尾统一采集。 |
| **Unknown** | `BL1-081…100` 与 20 个 PA1 event 的精确一对一映射。 | metadata 无 parent event ID。 |

### 3.2 Phase 8 trial-index imbalance 的含义

Phase 8 明确将 suffix 标为 `filename_index_proxy_not_documented_as_acquisition_time`（`src/models/run_within_cross_separability.py:141-154`），这一防护是正确的。报告发现 pain–pain pair 的 index separability 和 feature–index correlation（`reports/within_cross_separability.md:75-81`），当前只能支持：

- 某些 subject 的**文件/样本序号与类别分配不均衡**；在有限的 20 trial/class 随机序列中，这种不均衡可以自然出现。
- 若未来证明 `001–080` 是 chronology，且同一 feature 随时间有漂移，则它才构成 habituation、sensitization、sensor drift 或 carryover 的候选混杂。

当前不能支持：

- 把 suffix 当 elapsed time；
- 把 feature–suffix Spearman 解释为 habituation/sensitization；
- 用 BL1/PA4 的 index AUC 诊断 temporal drift。该 pair 的 index 分离由 `001–080` 与 `081–100` 的导出结构决定。

**风险判断：P0 / 中高解释风险。** Phase 8 没有把 trial index 放入模型特征，因此这不是直接的 predictor leakage；风险在于将诊断性 proxy 误写为真实时间机制。现有报告已经作出保留性说明，应进一步把“plausible temporal confound”改为“index association; temporal interpretation unverified”。

## 4. C — Within/Cross evaluation protocol（P1）

### 4.1 划分、预处理与模型

| 项目 | 审计结论 | 证据 |
|---|---|---|
| Cross folds | **Confirmed：无 subject overlap。** Phase 8 复用 Phase 3 的 5-fold `StratifiedGroupKFold` assignments；代码显式检查 train/test subject 交集。 | `src/evaluation/splits.py:19-55`；`src/models/run_within_cross_separability.py:80-92,310-353`。 |
| 官方可比性 | **Confirmed：内部 subject-independent 有效，但不是官方首选 LOSO。** | 官方建议全 87 subjects LOSO，并要求每 fold 内做 feature/model/parameter selection；5-fold grouped CV 可作项目内估计，但不能与官方 LOSO 数字直接等同（[官方说明](https://www.nit.ovgu.de/BioVid.html)）。 |
| Within split | **Confirmed：trial-level OOF 隔离正确。** 每 subject/pair 40 trial，5-fold × 10 repeats；每 fold 32 train/8 test；每 trial 每 repeat 恰有一次 OOF score。 | `src/models/run_within_cross_separability.py:248-307`。 |
| Cross split | **Confirmed：每 trial 一次 held-out-subject OOF。** | `src/models/run_within_cross_separability.py:310-375`。 |
| Pipeline | **Confirmed：imputer 与 scaler 只在相应训练 fold fit。** L2 LR，`C=1`，5 features。 | `src/models/baselines.py:16-32`；特征与配置见 `src/models/run_within_cross_separability.py:27-45`。 |
| 训练量 | **Confirmed：严重不对称。** Within 每 fold 32 trial；cross 按 69/70 个训练 subjects 为 2,760/2,800 trial。 | `experiments/within_cross_separability/cross_subject_oof_predictions.csv` 的 `n_train_trials`；代码 `src/models/run_within_cross_separability.py:347-349`。 |
| Phase 8 feature search | **Confirmed：Phase 8 本身未调特征或超参数。** | 固定 `phase6_preselected` 五特征，`src/models/run_within_cross_separability.py:31-38`。 |
| 特征选择独立性 | **Likely：存在 exploratory selection optimism，不是 fold 内预处理泄漏。** 五特征来自先前 Phase 5/6 同一 87 subjects 的全 OOF 汇总；Phase 5 先聚合所有 fold held-out attribution，再选 top features（`src/models/run_feature_contribution.py:79-124,692-713`）。 | 由于测试 subjects 影响了先前研究阶段的候选选择，Phase 8 AUC 不应作为完全独立确认性验证；但模型在 Phase 8 fold 内没有看到 held-out trial 标签。 |
| 表征范围 | **Confirmed：仅 4 个 EDA + 1 个 EMG 特征，无 ECG。** | `src/models/run_within_cross_separability.py:31-38`。因此结果不能等同于“所有可用生理信号的可分性上限”。 |

### 4.2 Repeated OOF 与 bootstrap

**OOF point estimate：基本正确。** 每个 within trial 的 10 个预测都来自未包含该 trial 的模型，随后取均值（`:262-275`）。这可降低 split 随机性，但不会把 40 个 trial 变成 400 个独立观测。10 次模型的训练集高度重叠，不应把 repeat 当独立 replication。

**个体 CI：条件性区间，覆盖范围偏窄的风险较高。** `bootstrap_metric_ci` 在固定的平均 OOF score 上，按类重采样 20+20 trial（`:225-245`）；它不重新划分/拟合模型，因而没有传播模型拟合、重复 CV split、特征预选或 cross-fold 共享模型的不确定性。如果 trial 存在时间相关/carryover，iid trial bootstrap 也不保留这种依赖。

**gap CI：配对做法正确，但仍是固定 score 的条件性 CI。** Within 与 cross 按 `sample_name` 一对一合并，并对同一 bootstrap index 计算 AUC 差（`:378-408`），保持了 trial 层配对；仍未 refit 两套模型。

**pair-level subject bootstrap：层级选择合理但不完整。** `bootstrap_location_ci` 对 87 个 subject gap 重采样（`:412-419`），主汇总至少以 subject 为人口推断单位；但同 fold subjects 共享一个 cross model，且每个 subject gap 的一阶段估计噪声没有被完整传播。因此报告中的 subject-bootstrap CI 应称为**基于既有 OOF 估计的 subject-resampling CI**，而非涵盖整个训练流程的严格置信区间。

Balanced accuracy 用固定 0.5 score threshold（`:213-218`），会受 within/cross 概率校准差异影响；以 ROC-AUC 作为主要 gap 指标是较稳妥的选择。

### 4.3 Gap 可以与不可以支持的结论

可以支持：

- 在固定五特征、固定 L2 LR 和现有训练预算下，相邻等级对同一 subject 仍难分；subject-mean within AUC 0.593 是描述性证据（`reports/within_cross_separability.md:23-37`）。
- 在这个具体 pipeline 中，少量个体数据训练的 within 模型没有平均超过大样本 population 模型；当前实验没有观察到“cross 模型普遍更差”的模式。
- 宽间隔 pair 比相邻 pair 更容易，在 within 与 cross 两种 regime 中方向一致（`:53-59`）。
- 个别 positive-gap subjects 可作为异质性候选，但多数个体 CI 很宽，只是筛查信号（`:39-51`）。

不能支持：

- 负 gap 证明不存在 cross-subject transfer failure；
- within-subject overlap 是唯一或主导因果机制；
- gap 是纯 subject effect 或可解释为 variance decomposition；
- 32-trial personalized learner 与 2,760–2,800-trial population learner 的差异代表 personalized vs generic 的公平比较；
- 五个预选特征上的 AUC 是生理信号可识别性的上限；
- Phase 6 consistency “解释”或导致 Phase 8 AUC。两者使用同一 subjects、重叠特征与标签，现有相关性是非独立描述关联（`reports/within_cross_separability.md:61-73`）。

这一有限样本警告也与数据集原始作者的观察一致：person-specific 模型在低可分场景可能因训练样本更少而过拟合，generic GSR 模型可能受益于更大的训练集（Werner et al. 2014, Sec. V-A/B，[DOI](https://doi.org/10.1109/ICPR.2014.784)）。

**风险判断：P1 / 中高比较风险。** point estimates 对当前算法问题有效；对 transfer-vs-overlap 机制的因果归因需要 budget-matched、nested、refit-aware 的补充实验。

## 5. D — Physiological observation window（P2）

### 5.1 窗口位置与可用时长

| 状态 | 发现 | 证据 |
|---|---|---|
| **Confirmed** | Part A 每个 bio clip 为 2,816 点、512 Hz、约 5.5 秒；temperature 为 176 点、32 Hz、约 5.5 秒。 | `starting_point/Readme.txt:46-56`；`reports/data_audit.md:149-165`；`src/features/extract_trial_features.py:21-22,274-304`。 |
| **Confirmed** | 原始 2014 方法把 biomedical 5.5 秒窗描述为 “during the stimulus”，窗口由刺激温度信号定位。 | Werner et al. 2014, Sec. III、V，[DOI 10.1109/ICPR.2014.784](https://doi.org/10.1109/ICPR.2014.784)。 |
| **Likely** | Part A 公布窗口相对 elicitation onset 延迟约 3 秒。 | 后续专门比较 BioVid segmentation 的方法论文明确称原始窗口 shift 为 3 秒并绘出窗口位置：Gouverneur et al., *Sensors* 2021, [DOI 10.3390/s21144838](https://doi.org/10.3390/s21144838)，Sec. 3.2.1/Fig. 4。 |
| **Confirmed** | 本地 PA 窗口起点已接近目标高温，末段明显降温；因此它不是从热刺激 ramp onset 开始的完整 5.5 秒。 | `temperature/*/*_temp.csv` 的全量只读检查，汇总值见 2.1。 |
| **Unknown** | “3 秒”相对 thermode command、温度开始上升、接触点温度变化还是另一个 trigger；以及每 trial 精确 start/end event offset。 | 本地短片段时间归零，metadata 无 onset/offset 标记。需 Part C 连续记录及 event log 验证。 |
| **Confirmed** | 当前仓库没有更长的 per-trial physiological window。官方 Part C 有相同 87 subjects 的长序列、raw GSR/ECG/trapezius EMG 与 stimulus labels。 | 当前目录只有 Part A clipped files；BioVid [官方 Data Organization](https://www.nit.ovgu.de/BioVid.html) 对 Part A/C 的说明。是否已获 Part C 授权/可取得，仓库中 **Unknown**。 |

据此最保守的窗口解释是：**约 3 秒 post-elicitation 开始的 5.5 秒裁剪，主要覆盖高温平台和早期 cooldown；没有完整 pre-stimulus baseline，也未保证覆盖晚期恢复。** 不应把 clip time 0 当刺激 onset。

### 5.2 特征与窗口匹配

- **EDA/SCL mean, range, AUC, slope, delta：窗口内描述有效，但高度依赖裁剪位置。** 代码在整段 5.5 秒上计算，delta/slope 会同时反映 conductance response、刺激 offset 后变化与窗口边界（`src/features/extract_trial_features.py:171-185`）。它们可作 discriminative features，不能自动解释为完整 tonic level 或恢复速度。
- **SCR count/amplitude：仅 borderline observable。** 代码对每个孤立窗口独立执行 `nk.eda_process` 并统计 peak（`:148-169`），没有独立 pre-stimulus segment；窗口前已开始的反应、靠近窗口末端才出现的反应及其 recovery 都可能被截断。SPR/EDA 方法建议通常把 elicited response onset latency 定义在刺激变化后 1–4 秒，rise-to-peak 还可再需约 0.5–5 秒，recovery 更晚；见 Society for Psychophysiological Research 指南，[DOI 10.1111/j.1469-8986.2012.01384.x](https://doi.org/10.1111/j.1469-8986.2012.01384.x)。项目原 feature spec 已正确把 SCR count/amplitude 标为 borderline，并排除 full recovery（`reports/feature_spec.md:117-129`）。
- **ECG：短时 rate/beat-count 与粗幅度特征可计算，传统 HRV 不充分。** 5.5 秒通常只有少量 R peaks；实现少于 3 peaks 时将 rate/RR summaries 置零（`src/features/extract_trial_features.py:125-145`）。长期自主调节和可靠 HRV 不能由该窗推断。
- **EMG：窗口内 amplitude/energy/trend 基本匹配，但不是完整 onset/recovery。** 50 ms envelope 和 window-local threshold 适合瞬时活动描述（`:189-224`）；如果肌肉反应在窗前开始或窗后持续，持续时间和恢复会被截断。
- **Phase 5–8 的 EDA 主导结果对 window placement 敏感。** Phase 8 的五个特征有四个是 EDA delta/slope/SCR（`src/models/run_within_cross_separability.py:31-38`），所以“adjacent overlap”既可能是真实反应重叠，也可能包含低强度 SCR peak 未进入窗口、反应时延个体差异及恢复截断造成的 observability loss。当前设计不能分解这些来源。

**风险判断：P2 / 中等测量有效性风险，对 EDA 机制解释可升至高。** 分类任务本身使用官方 Part A 窗口，数值没有因此失效；失效的是把短窗性能解释为完整生理响应或绝对识别上限。

## 6. 对 Phase 3–8 结论的处置

| 结论类别 | 处置 | 审计后的有效表述 |
|---|---|---|
| Phase 3 grouped OOF baseline 数值 | **保留** | 这是 87 subjects、Part A 5.5 秒片段、5-fold subject-grouped CV 下的 stimulus-class prediction；不是官方 LOSO 的直接可比数字。 |
| PA2 与相邻 PA pair 较难 | **保留并限定** | 在当前 Part A 窗口及特征/模型下稳定出现；标签是 subject-calibrated heat intensity，不是每 trial 主观疼痛评分。 |
| EDA 在 Phase 4/5 的贡献较大 | **弱化为协议/窗口特定** | EDA 对当前裁剪片段具有较高 predictive contribution；不能称普适疼痛 biomarker，且 BL1/PA4 contribution 可能包含恢复/窗口阶段信息。 |
| BL1/PA4 高 AUC | **需要重新验证外推** | 保留为 post-PA1 recovery vs PA4-stimulus 的协议内 separability；不称 neutral-rest vs high-pain separability。 |
| Phase 6 PA1–PA4 个体异质性与相邻方向不稳定 | **保留** | 是同一刺激窗口定义下的 descriptive heterogeneity；不是 causal physiology。 |
| Phase 6 BL1-based trajectories | **弱化** | 将 BL1 明示为 post-PA1 recovery reference。 |
| Phase 6 baseline subtraction 排除 baseline shift | **需改写/重新验证** | 只说明 oracle labeled post-PA1 BL1 mean subtraction 未改善；不能排除 pre-stimulus baseline、动态 recovery 或 label-free calibration。 |
| Phase 7 ordinal/gated 方法未解决相邻等级 | **保留并限定** | 当前观察窗/标签/特征下，模型结构变化未解决相邻刺激等级；不代表生理识别上限。 |
| Phase 8 within AUC 低 | **保留为描述结果** | 小样本 subject-specific LR 在固定五特征上仍难区分相邻等级。 |
| Phase 8 负 within−cross gap | **弱化机制归因** | 当前实验未显示 within learner 平均优于 large-sample cross learner；不能证明 transfer failure 不存在或 within overlap 主导。 |
| Trial-index imbalance = habituation | **不得保留** | 目前仅为 filename-index association；真实 chronology 与 temporal mechanism 未确认。 |
| Phase 6 consistency 解释 Phase 8 separability | **弱化为非独立关联** | 同数据、同特征体系中的 descriptive correlation，不是外部预测验证或因果解释。 |

## 7. 最小补充实验建议（仅建议，未执行）

1. **先恢复 temporal observability。** 获取已授权的 Part C 连续 GSR/ECG/trapezius EMG、温度与 stimulus event log；确认 Part A clip 的精确 onset/plateau/offset 对齐。预注册少量窗口：pre-stimulus、ramp/plateau、0–4 秒 post-offset、较长 recovery，保持同一 subject-independent folds。该步骤同时解决 BL1 定义、trial chronology 与 EDA latency 三个核心不确定性。
2. **建立 matched baseline/recovery 对照。** 对每个刺激 event 提取同长度 pre-stimulus baseline 与 post-stimulus recovery；至少比较 pre-PA1、post-PA1、pre-PA4、post-PA4。若原始事件日志能恢复 BL1↔PA1 pairing，使用 paired analysis；否则不得假定文件 suffix 配对。
3. **核实 trial ordering。** 用连续 temperature onset 或 event ID 建立唯一真实序列，比较 filename suffix；在确认前停止 habituation/漂移叙事。确认后采用 block-aware 或 sequence-aware 分析，并分别估计 randomized allocation imbalance 与真实 drift。
4. **做 budget-matched within/cross learning curves。** 固定同一五特征和 LR，把 cross 训练集下采样到与 within 相同的 32 trial（同时报告 subjects 数与每 subject trial 数），并对 within/cross 共同绘制训练量曲线。另可用跨 subject 预训练 + 少量 calibration，但需与纯 within 分开命名。
5. **进行 nested selection 与 refit-aware uncertainty。** 确认性版本应在外层 held-out subjects 之外完成 feature selection；重复整个 split→fit→predict 流程。人口推断以 subject bootstrap 为主；若确认 chronology，within-subject 使用 block bootstrap/cluster-robust 方法。报告 conditional OOF CI 与 full-procedure CI 的区别。
6. **补一个 BL1-free 主分析。** 用 PA1/PA3 作 wide control，并以 PA1/PA2、PA2/PA3、PA3/PA4 为核心，避免 baseline protocol 同时变化。BL1/PA4 保留为 protocol contrast。

## 8. 下一阶段方向建议

**优先级 1：temporal observability。** 当前最大的不确定性不是模型容量，而是模型实际看到了生理反应的哪个阶段。它直接影响 BL1 的含义、EDA SCR 可观测性、trial order 解释和 BL1/PA4 高 AUC。获得连续记录/事件标记后，很多“overlap”可被进一步分成真实 response overlap 与 window truncation。

**优先级 2：finite-sample limitation。** 在时程对齐被确认后，立即做 budget-matched learning curves；这是把 Phase 8 的负 gap 从描述结果推进到 transfer-vs-within 机制判断所需的最小实验。

**优先级 3：reliability-aware prediction。** 它很重要，但应建立在可靠的 event alignment 和有限样本基线之上。随后可估计 trial/retest reliability、按 subject 给出 uncertainty，并研究“低可靠度时拒绝预测/合并等级”，而不是先增加模型复杂度。

## 9. Finding → Evidence → Risk → Recommended Action

| Finding | Evidence | Risk | Recommended Action |
|---|---|---|---|
| BL1 是 PA1 后 pause 片段 | 原论文 Sec. I-C/V；每类 20 样本 | **P0 高**：中性基线解释错误 | 全部改称 post-PA1 recovery baseline；用 matched pre-stimulus baseline 重验 |
| BL1↔PA1 精确 trial pairing 未知 | metadata 无 parent ID/global clock | **P0 中**：不能做配对恢复分析 | 获取 event log/Part C；未获前不假定 suffix 配对 |
| BL1/PA4 AUC 同时含强度与阶段差 | 原论文 + 本地温度片段 | **P0 高**：外推过强 | 保留 protocol contrast；以 PA1/PA3 作为主要 wide stimulus control |
| Phase 6 subtraction 使用 held-out subject 的 labeled BL1 | `run_cross_subject_stability.py:218-226,329-344` | **P0/P1 高**：不是 label-free preprocessing | 重命名 oracle/transductive pilot；如需确认，训练 fold 内或预先独立 calibration 重做 |
| Pain suffix `001–080` 唯一且交错 | `samples.csv`；Phase 8 validator | **P0 中**：既有 audit 有错误描述 | 以本报告为更正；后续维护时修正文档，不把 suffix 自动当时间 |
| PA suffix 很可能但未确认是 chronology | 随机协议与编号结构；缺 event log | **P0 中高**：habituation 叙事未证实 | 用连续温度 onset/event IDs 验证 |
| BL1 `081–100` 确定不是全局 chronology | BL1 来自随机 PA1 后 pause | **P0 高**：BL1/PA4 order test 结构性失真 | 禁止用该 pair 的 suffix 推断 drift |
| Phase 8 fold 内 preprocessing/OOF 隔离正确 | `run_within_cross_separability.py:248-375`；pipeline | **低** | 保留 point estimates 与明确协议 |
| Within 32 vs cross 2,760–2,800 | 代码与 prediction artifacts | **P1 高**：gap 非纯 transfer effect | 做 budget-matched learning curves 与共同训练预算比较 |
| Repeated OOF 不是独立重复 | 10 次重叠训练集后平均 score | **P1 中**：不确定性易被低估 | 重复全流程，报告 split variability；不把 10 predictions 当 n×10 |
| Trial bootstrap 固定 OOF score、不 refit | `bootstrap_metric_ci`/gap code | **P1 中高**：CI 为条件性 | subject-level outer resampling + refit；若有时间依赖则 block bootstrap |
| Phase 8 特征预选使用同一 subjects 的前期结果 | Phase 5 aggregate selection → Phase 6/8 fixed set | **P1 中**：确认性 AUC 有 selection optimism | 外层 nested selection 或独立 confirmatory cohort；现阶段标 exploratory |
| 5.5 秒窗约在 onset 后 3 秒，含 plateau/early cooldown | 2014 原论文、2021 segmentation 图、本地温度 | **P2 中；EDA 解释高** | 用 Part C 多窗口对齐敏感性分析 |
| SCR onset/peak/recovery 可被截断 | EDA 指南 1–4 秒 latency、0.5–5 秒 rise；代码逐 clip 处理 | **P2 高**：SCR 机制归因不足 | 增加 pre/post 长窗，显式定义 response window 与 recovery censoring |
| Part A 本地不能提取更长窗；Part C 官方提供长序列 | 仓库文件与官方 data organization | **P2 中** | 核实许可并取得 Part C physiological/event data；不使用视频特征 |
| 标签是校准刺激等级，不是 trial-level self-report | Readme label mapping；原始实验协议 | **全阶段高解释风险** | 全文使用 stimulus-intensity classification；不声称真实主观痛感上限 |

## 10. 未解决问题与所需证据

1. `001–080` 是否由 acquisition timestamp 直接生成？需要官方导出脚本或 Part C event log。
2. 每个 `BL1-081…100` 对应哪一个 PA1 event，且 pause 内起始位置为何？需要 parent event ID/segmentation manifest。
3. 原始 3 秒 shift 的 trigger 定义为何，是否每 trial 固定？需要官方 segmentation code 或带 event markers 的连续温度记录。
4. Part C 是否已在当前数据许可内并可获得？当前仓库无法判断；需向数据持有人或本机构数据管理员确认。
5. 数据是否包含 trial-level 主观评分？当前 Part A metadata 未发现；若另有未导入问卷文件，需单独审计，不能用刺激标签替代。

## References

- BioVid Heat Pain Database, Otto-von-Guericke University Magdeburg, [official data organization and evaluation protocol](https://www.nit.ovgu.de/BioVid.html).
- Werner, P. et al. (2014). *Automatic Pain Recognition from Video and Biomedical Signals*. ICPR 2014. [DOI: 10.1109/ICPR.2014.784](https://doi.org/10.1109/ICPR.2014.784).
- Walter, S. et al. (2013). *The BioVid Heat Pain Database: Data for the Advancement and Systematic Validation of an Automated Pain Recognition System*. [DOI: 10.1109/CYBConf.2013.6617456](https://doi.org/10.1109/CYBConf.2013.6617456).
- Werner, P. et al. (2013). *Towards Pain Monitoring: Facial Expression, Head Pose, a New Database, an Automatic System and Remaining Challenges*. [DOI: 10.5244/C.27.119](https://doi.org/10.5244/C.27.119).
- Gouverneur, P. et al. (2021). *Comparison of Feature Extraction Methods for Physiological Signals for Heat-Based Pain Recognition*. *Sensors*, 21, 4838. [DOI: 10.3390/s21144838](https://doi.org/10.3390/s21144838).
- Society for Psychophysiological Research Ad Hoc Committee on Electrodermal Measures (2012). *Publication recommendations for electrodermal measurements*. [DOI: 10.1111/j.1469-8986.2012.01384.x](https://doi.org/10.1111/j.1469-8986.2012.01384.x).
