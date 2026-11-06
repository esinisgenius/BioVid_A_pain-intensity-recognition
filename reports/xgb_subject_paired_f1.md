# 五分类 XGBoost：受试者级配对 F1 分析

## 方法

直接从保存的预测概率取 argmax，不训练模型。Cross 对每人 100 条试次计算一次 F1；Within 在每次重复的完整 100 条折外预测上计算 F1，再对 10 次重复取均值。每类 F1 使用五分类 one-vs-rest 混淆计数（包括来自其他真实类别的假阳性），不是仅截取该类试次。Macro F1 为五类 F1 的算术平均。

受试者配对差值 ΔF1 = Within − Cross。10,000 次受试者级配对 bootstrap，每次有放回抽取 87 人，同一组重采样用于所有指标；95% CI 为百分位区间。重复先在受试者内平均，不视为独立样本。无受试者排除；新计算的全部受试者级分数与原指标文件一致。

## 配对结果

差值及区间以百分点表示；Cross/Within 为受试者平均 F1。

| 指标 | Cross | Within | 平均 Δ（95% CI） | 改善 | 下降 | 持平 |
|---|---:|---:|---:|---:|---:|---:|
| macro_F1 | 0.2906 | 0.3666 | +7.60 [+5.58, +9.75] | 70/87 (80.5%) | 17 | 0 |
| BL1 | 0.4496 | 0.3824 | -6.72 [-9.69, -3.63] | 21/87 (24.1%) | 66 | 0 |
| PA1 | 0.2117 | 0.2621 | +5.05 [+1.48, +8.60] | 48/87 (55.2%) | 39 | 0 |
| PA2 | 0.1131 | 0.2915 | +17.84 [+14.57, +21.07] | 73/87 (83.9%) | 14 | 0 |
| PA3 | 0.2119 | 0.3477 | +13.58 [+9.31, +18.03] | 64/87 (73.6%) | 23 | 0 |
| PA4 | 0.4667 | 0.5494 | +8.27 [+4.90, +11.71] | 56/87 (64.4%) | 31 | 0 |

## 差异分布

| 指标 | 中位数 Δ | 第25–75百分位 | 最小–最大 | SD |
|---|---:|---:|---:|---:|
| macro_F1 | +6.57 | +1.07 至 +13.13 | -12.39 至 +41.72 | 10.05 |
| BL1 | -9.58 | -15.38 至 -1.05 | -38.21 至 +44.97 | 14.46 |
| PA1 | +1.71 | -8.04 至 +16.42 | -27.75 至 +59.69 | 16.90 |
| PA2 | +16.64 | +7.27 至 +28.34 | -12.06 至 +63.55 | 15.77 |
| PA3 | +10.62 | -0.17 至 +27.39 | -22.51 至 +83.58 | 20.70 |
| PA4 | +6.17 | -4.45 至 +17.00 | -26.44 至 +67.88 | 16.37 |

## 主要发现

- 总体 macro F1 平均提高 7.60 个百分点，95% CI [5.58, 9.75]；70/87 人（80.5%）改善，17 人下降。改善比例的 bootstrap 95% CI 为 [72.4%, 88.5%]。差值中位数为 +6.57 个百分点，四分位范围为 +1.07 至 +13.13，范围为 −12.39 至 +41.72，提示存在明显个体差异。
- PA2 的平均 F1 从 0.1131 升至 0.2915（+17.84 个百分点），73/87 人改善；PA3 从 0.2119 升至 0.3477（+13.58 个百分点），64/87 人改善。两者是本次观察到的最大逐类均值收益；未正式检验类别间收益差异。
- PA1 的均值收益为 +5.05 个百分点，但只有 48/87 人改善，中位数收益仅 +1.71 个百分点；其改善并不普遍。PA4 均值提高 8.27 个百分点，56/87 人改善。
- BL1 平均下降 6.72 个百分点，66/87 人下降。因此总体提升并非五个等级一致提升，而伴随明显等级间差异。PA2/PA3 的改善支持当前设置下中间等级的个体内可预测性高于跨个体预测，但绝对 F1 仍有限。
- 五个逐类均值差值的平均恰好等于总体 macro F1 差值；这来自指标定义，不应被解释为各类之间独立的生理贡献分解。

## 解释边界

逐类结果可以回答 PA1–PA3 是否在当前个体模型下获得改善；PA2 为五等级中央类别。必须同时考察平均差值、区间和改善人数，不能用总体均值代表所有人。正差值是该固定模型与协议下的预测证据，不直接证明生理机制或特征重要性改变。

Within 与 Cross 训练量和训练人群构成不同，同一受试者随机试次分折还可能受记录条件、试次依赖及时间漂移影响。Bootstrap 区间以现有拟合模型和队列为条件，不包含重训或新队列不确定性。六个区间为逐项区间，不是同时置信区间；未进行显著性检验或挑选等级。实验热刺激等级及 BL1 恢复期的解释限制仍适用。

## 图注

Five-class fixed benchmark XGBoost; 59 features; 87 subjects. Each point is one subject: within F1 is computed over 100 OOF trials separately for each of ten 5-fold repeats, then averaged; cross F1 uses 100 subject-held-out predictions. Blue positive differences favour within; rose negative differences favour cross. Diamonds and bars show mean paired differences and pointwise 95% percentile bootstrap CIs from 10,000 resamples of 87 subjects, preserving pairing; violin widths describe the empirical distribution. Repeats and trials are not independent bootstrap units. Within uses 80 training trials per fit; cross uses thousands from other subjects. No retraining.

输出：`experiments/xgb_subject_paired_f1/`；复现：`python -m src.analysis.analyze_xgb_subject_paired_f1`。
