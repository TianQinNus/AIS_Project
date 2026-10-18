# FleetCare C4 Risk Ranking Ensemble 详细模型报告

## 1. 执行摘要

当前最终模型的正式名称为 **FleetCare C4 Risk Ranking Ensemble**，版本为 `c4-risk-ensemble-v1-2026-10`。它的首要任务不是在所有车辆上追求最高五分类准确率，而是在有限检查资源下，尽量把真正处于 C4 阶段的车辆排到风险队列前部。

C4 表示距离 Component X 维修事件不超过 6 个数据集时间单位，是项目中最紧急、样本最少、也最难识别的类别。最终系统采用四个内部组件：两个独立的 C4 二分类模型形成基础排序，一个 C0/C4 veto 模型抑制缺乏支持的高风险判断，另一个五分类 XGBoost 提供 C0–C4 状态以及 Risk12、Risk24、Risk48。四个组件已封装在一个文件 `fleetcare_c4_risk_ensemble.joblib` 中，推理脚本只进行一次模型文件加载。

在官方 Test 数据的 5,045 辆车中，有 60 辆 C4。最终模型取得：

- C4 Average Precision：0.04118
- C4 ROC-AUC：0.72049
- Top 1%：检查 51 辆，识别 3 辆 C4，Precision 5.88%，Recall 5.00%
- Top 2%：检查 101 辆，识别 7 辆 C4，Precision 6.93%，Recall 11.67%
- Top 5%：检查 253 辆，识别 11 辆 C4，Precision 4.35%，Recall 18.33%
- Top 10%：检查 505 辆，识别 17 辆 C4，Precision 3.37%，Recall 28.33%
- Top 20%：检查 1,009 辆，识别 26 辆 C4，Precision 2.58%，Recall 43.33%

与项目早期 baseline 相比，最终模型在 Top 5% 中由 4 辆 C4 提升到 11 辆，Precision 从约 1.58% 提升到 4.35%，约为原来的 2.75 倍；AP 从约 0.0205 提升到 0.0412，约为原来的 2.01 倍；ROC-AUC 从约 0.672 提升到 0.720。

当前部署决策采用混合策略：**模型负责对稀缺的立即检查名额排序，rule engine 负责结合多次历史观察扩大监控和人工复核范围。** Rule engine 不应被表述为提升了同等立即检查预算下的 C4 识别，因为冻结 Test 结果显示它在 158 辆立即检查范围内识别 9 辆 C4，而同规模纯模型排序识别 10 辆；它真正带来的提升是在更宽的 attention 范围内识别 34 辆 C4，相比同规模模型排序的 29 辆增加 5 辆。

## 2. 预测任务与标签定义

### 2.1 五个维护阶段

标签基于车辆当前观察时刻与 Component X 维修事件之间的剩余时间构造：

| 类别 | 剩余时间范围 | 项目解释 |
|---|---:|---|
| C0 | 大于 48 | 当前没有进入 48 单位风险窗口 |
| C1 | 24–48 | 中长期风险 |
| C2 | 12–24 | 风险正在接近 |
| C3 | 6–12 | 近期风险 |
| C4 | 0–6 | 最紧急风险，也是最终模型重点识别对象 |

对于训练集中未观察到维修事件的车辆，不能简单假设其剩余寿命无限。只有在某个观察点之后仍有超过 48 个时间单位的有效随访时，该观察点才可作为 C0 训练样本。研究结束前最后 48 个时间单位内但没有发生维修的记录结局不明确，不作为确定 C0 使用。

### 2.2 当前模型的实际优化目标

当前模型的主要优化目标是 **C4 风险排序**，不是五分类 Macro F1，也不是把一个固定阈值以上的车辆直接判定为 C4。

这样设计的原因是：

1. C4 在 validation 和 test 中占比都很低，固定阈值分类容易受数据分布变化影响。
2. 真实业务不可能每天检查所有车辆，更合理的问题是“在只能检查前 1%、2% 或 5% 车辆时，能找到多少 C4”。
3. 排序指标能直接映射检查容量；最终是否执行检查，再由 rule engine 和人工判断决定。

因此，`risk6_score` 应解释为 **同一预测批次中车辆的相对 C4 风险排序分数**，范围为 0–1。它不是未来 6 个单位内发生维修的校准概率，不能解释为“0.8 就代表 80% 会维修”。

## 3. 数据划分与评估协议

### 3.1 官方车辆级划分

| 数据集 | 车辆数 | C4 数量 | 用途 |
|---|---:|---:|---|
| Train | 23,550 | 训练抽样后有 3,851 条 C4 训练记录 | 训练模型和建立特征 schema |
| Validation | 5,046 | 76 | 选择特征、模型和规则参数 |
| Test | 5,045 | 60 | 最终结果报告与后期比较 |

划分以 `vehicle_id` 为单位，不把同一辆车的不同记录分散到训练和评估数据中。Validation 和 Test 的模型评估均使用每辆车的最后一次 readout，因此一辆车只贡献一个最终预测。

### 3.2 关于 Test 结果的必要说明

项目早期实验已经查看过官方 Test 结果，Test 39 的 veto 组合方式也进行了 post-hoc Test 比较。因此当前 Test 指标适合用于项目内部比较和最终系统说明，但不能再声称是完全未接触过的独立泛化估计。

更严谨的表述是：

> 模型主要通过 Validation 开发，并在官方 Test 上报告最终表现；但由于项目迭代期间已经查看过 Test 结果，Test 指标属于 post-hoc evaluation，而不是严格意义上的一次性盲测。

Rule engine 的最终 profile 是在 Validation 上选择后冻结，再应用于 Test；因此规则部分的 Test 评估相对更接近规范的 hold-out 流程。

## 4. 模型开发过程

### 4.1 初始五分类模型

早期方案使用单个多分类 XGBoost 预测 C0–C4，并使用 Accuracy、Macro F1、各类别指标以及 Risk6/12/24/48 的 AP 和 ROC-AUC 进行评估。该方案整体 Accuracy 很高，但主要来自占绝大多数的 C0。Macro F1 约为 0.20，说明模型对少数类，尤其是 C4，识别能力有限。

这个阶段带来的主要结论是：高 Accuracy 并不等于高风险车辆识别有效。项目目标因此从“总体分类准确率”调整为“在有限检查容量内提升 C4 的识别率和精度”。

### 4.2 特征组合和树参数筛选

随后测试了不同 temporal feature collections 和树模型参数，包括：

- 当前值转化后的 rate 特征
- histogram proportion 特征
- 短期和长期变化特征
- recent rate 与 lifetime rate 的 ratio
- 不同最大深度、学习率、正则化强度和树数量
- XGBoost 与 LightGBM

早期以 Macro F1 或平均 horizon AP 进行模型选择时，`F5_rate_plus_short_change` 和 `P3_medium` 等配置表现较好。但这些目标仍然不能充分反映 C4 在有限检查名额中的识别效果。

### 4.3 不平衡学习和 C4 专项实验

为提升 C4，项目继续测试：

- class weighting 和不同 weight exponent
- random oversampling
- SMOTE
- 对多数类进行 downsampling
- 每辆车不同数量的历史记录抽样
- C4 专项二分类模型
- 固定阈值与 validation cross-fitted 阈值

Test 32 直接优化 C4 F1。Validation 上 C4 F1 达到约 0.0836，但冻结阈值应用到 Test 后只识别到 1 辆 C4，F1 约 0.0198。这说明固定分类阈值对 validation/test 分布变化非常敏感，不适合作为最终决策核心。

### 4.4 转向低容量风险排序

Test 33 将目标改成 Top 1%、2%、5%、10% 容量下的 C4 recall 和 precision，并形成当前特征工程基础：

- 每辆车 C4 阶段最多抽取 10 条分散的历史记录
- 其他维修阶段每个阶段最多抽取 3 条
- 未维修车辆最多抽取 2 条
- temporal lags 改为 1、3、5 个 readouts
- 特征总数增加到 534

该模型在 Test Top 5% 中识别 14 辆 C4，Recall 23.33%，但 Top 2% 只识别 3 辆，Precision 2.97%。它提高了覆盖率，但最前部队列仍不够精确。

### 4.5 Precision 和稳定性探索

Test 34–38 继续测试了：

- XGBoost 与 LightGBM 的 rank averaging
- 多 seed ensemble
- 特征重要性筛选，如 Top 250/350 features
- 更浅或更保守的树结构
- bootstrap stability selection
- OOF logistic stacking
- cascade 和 EasyEnsemble

这些方法在 Validation 上部分指标较好，但在 Test 上没有形成稳定的大幅提升。例如：

| 实验 | Test AP | Test AUC | Top 2% C4 | Top 5% C4 | 结论 |
|---|---:|---:|---:|---:|---|
| Test 33 单一 XGBoost | 0.03349 | 0.72773 | 3 | 14 | 覆盖较高，但队列前端 precision 较弱 |
| Test 34 XGB/LGB rank ensemble | 0.03284 | 0.71590 | 5 | 13 | Top 2% 有改善 |
| Test 35 advanced XGB | 0.03017 | 0.71279 | 4 | 9 | 未超过 ensemble |
| Test 36 稳定特征 ensemble | 0.03415 | 0.71490 | 4 | 11 | AP 略升，前端提升有限 |
| Test 37 OOF stacking | 0.03286 | 0.71428 | 5 | 11 | 复杂度增加但收益有限 |
| Test 38 cascade/EasyEnsemble | 0.03284 | 0.71590 | 5 | 13 | 最终仍选择原 rank ensemble |

### 4.6 最终 agreement-veto 设计

Test 39 在 XGBoost 与 LightGBM 的基础排序之上增加 C0/C4 veto 模型。最终分数取基础排序与 veto 排序的较小值：

```text
xgb_rank  = percentile_rank(P_XGB(C4))
lgb_rank  = percentile_rank(P_LGB(C4))
base_rank = percentile_rank((xgb_rank + lgb_rank) / 2)
veto_rank = percentile_rank(P_veto(C4 | C0 or C4))

Risk6 score = min(base_rank, veto_rank)
```

这相当于要求“基础模型认为它高风险，同时专门区分 C0/C4 的模型也不能强烈反对”。最终方案牺牲了一部分 Top 5% recall，但提高了最前部队列的 precision、整体 AP 和排序可信度：

- Test 33 Top 2%：3/101，Precision 2.97%
- 最终模型 Top 2%：7/101，Precision 6.93%
- 最终模型 Test AP：0.04118，高于 Test 33 的 0.03349

## 5. 最终训练样本构造

训练池先根据维修和 censoring 条件筛选可使用的历史观察点，然后进行车辆内抽样。

### 5.1 维修车辆

对于发生维修的车辆，按照 `vehicle_id + class_label` 分组：

- C4 每个车辆阶段最多选 10 条记录
- C0–C3 每个车辆阶段最多选 3 条记录
- 如果记录多于上限，使用时间轴上近似等间距的记录，而不是全部集中在某个局部时间段
- 同一车辆同一阶段选中的样本权重之和为 1，避免历史记录更多的车辆支配训练

### 5.2 未维修车辆

只使用仍有超过 48 单位有效随访的记录作为 C0 候选。每辆车随机抽取最多 2 条记录，随机种子为 42，选中记录的车辆内权重之和同样为 1。

### 5.3 最终训练矩阵

- 训练记录：67,633 条
- C4 训练记录：3,851 条
- 输入特征：534 个
- 训练数据仅来自 Train；Validation 和 Test 不进入最终训练代码

## 6. 特征工程

### 6.1 原始输入

- 105 个匿名 operational readout 字段
- 8 个车辆 specification 字段
- `vehicle_id` 用于车辆内排序和连接，不作为直接数值特征
- `time_step` 用于计算 rate、历史间隔和可用性，不直接作为普通原始特征输入

### 6.2 Operational 特征，共 444 个

#### Lifetime rate

对单值累计变量计算：

```text
rate = current value / (time_step + 1)
```

这样可以降低车辆观察时长不同造成的绝对累计值不可比问题。

#### Histogram proportion

对于具有相同前缀的 histogram bins，先计算组总量，再计算每个 bin 占该组总量的比例：

```text
prop(bin_j) = bin_j / sum(all bins in group)
```

它描述运行模式或分布结构，而不是车辆使用总量。

#### Histogram group rate

```text
group rate = histogram group total / (time_step + 1)
```

#### Past-only temporal features

使用 lag 1、3、5 个历史 readouts。每个 lag 都严格引用同一车辆更早的记录，不使用未来数据。

对于单值累计变量：

```text
recent rate = (current - previous) / elapsed time
ratio_lag   = recent rate / lifetime rate
```

对于 histogram groups：

```text
mix_lag(bin_j) = change in bin_j / total change of the group
ratio_lag(group) = recent group rate / lifetime group rate
```

当累计值下降时，代码将该差值设为缺失值，以避免把计数器重置或异常读数误解释为负使用量。

### 6.3 Specification 特征，共 90 个

8 个匿名 specification 字段使用训练集建立 one-hot schema，最终产生 90 个静态特征。Validation、Test 和数据库推理必须对齐同一 schema；未出现的新类别不会创建新的模型列。

### 6.4 特征总计

| 来源 | 特征数 |
|---|---:|
| Operational-derived | 444 |
| Specification one-hot | 90 |
| 合计 | 534 |

## 7. 最终模型结构与参数

最终模型文件为 `fleetcare_c4_risk_ensemble.joblib`，内部包含四个组件。

### 7.1 C4 XGBoost

| 参数 | 值 |
|---|---|
| objective | `binary:logistic` |
| eval_metric | `logloss` |
| tree_method | `hist` |
| n_estimators | 300 |
| learning_rate | 0.04 |
| max_depth | 5 |
| min_child_weight | 5 |
| gamma | 0.05 |
| subsample | 0.80 |
| colsample_bytree | 0.75 |
| reg_alpha | 0.10 |
| reg_lambda | 3.0 |
| random_state | 42 |
| n_jobs | 4 |

训练目标为 C4 vs non-C4。基础车辆内权重上进一步使用二分类权重指数 0.5：

```text
positive multiplier = (weighted negatives / weighted positives) ^ 0.5
```

### 7.2 C4 LightGBM

| 参数 | 值 |
|---|---|
| objective | `binary` |
| n_estimators | 400 |
| learning_rate | 0.035 |
| num_leaves | 15 |
| min_child_samples | 35 |
| colsample_bytree | 0.80 |
| subsample | 0.80 |
| subsample_freq | 1 |
| reg_lambda | 2.0 |
| random_state | 42 |
| n_jobs | 4 |

它与 C4 XGBoost 使用相同训练标签和 weight exponent 0.5。两种树实现的归纳偏差不同，最终通过 percentile rank 平均降低单模型偶然排序的影响。

### 7.3 C0/C4 veto XGBoost

该模型只使用训练标签为 C0 或 C4 的记录，直接学习最容易混淆且决策影响最大的两端类别。

| 参数 | 值 |
|---|---|
| objective | `binary:logistic` |
| eval_metric | `logloss` |
| tree_method | `hist` |
| n_estimators | 350 |
| learning_rate | 0.03 |
| max_depth | 3 |
| min_child_weight | 10 |
| gamma | 0.10 |
| subsample | 0.85 |
| colsample_bytree | 0.70 |
| reg_alpha | 0.20 |
| reg_lambda | 6.0 |
| random_state | 61 |
| n_jobs | 4 |

veto 模型更浅、正则化更强，weight exponent 为 0.25。它的用途不是单独发现全部 C4，而是压低“基础 ensemble 很高、但 C0/C4 专项模型不支持”的风险排名。

### 7.4 Maintenance-horizon 五分类 XGBoost

| 参数 | 值 |
|---|---|
| objective | `multi:softprob` |
| num_class | 5 |
| eval_metric | `mlogloss` |
| n_estimators | 350 |
| learning_rate | 0.035 |
| max_depth | 5 |
| min_child_weight | 8 |
| gamma | 0.10 |
| subsample | 0.80 |
| colsample_bytree | 0.70 |
| reg_alpha | 0.20 |
| reg_lambda | 5.0 |
| tree_method | `hist` |
| random_state | 42 |
| n_jobs | 4 |

多分类权重按照每个类别的加权样本总量计算，weight exponent 为 0.25：

```text
class multiplier_k = (largest class weight / class k weight) ^ 0.25
```

该组件输出完整的 C0–C4 概率分布，并提供：

```text
Risk48 = P(C1) + P(C2) + P(C3) + P(C4)
Risk24 = P(C2) + P(C3) + P(C4)
Risk12 = P(C3) + P(C4)
state  = argmax(P(C0), ..., P(C4))
confidence = max(P(C0), ..., P(C4))
```

Risk6 不直接使用 `P(C4)`，而是使用前述 agreement-veto ensemble 排名。

## 8. 模型评估结果

### 8.1 Validation

Validation 包含 5,046 辆车，其中 76 辆 C4，C4 prevalence 约为 1.51%。

| 指标 | 结果 |
|---|---:|
| Average Precision | 0.06674 |
| ROC-AUC | 0.78765 |
| Top 1% | 6/51 C4，Precision 11.76%，Recall 7.89% |
| Top 2% | 13/101 C4，Precision 12.87%，Recall 17.11% |
| Top 5% | 24/253 C4，Precision 9.49%，Recall 31.58% |
| Top 10% | 31/505 C4，Precision 6.14%，Recall 40.79% |
| Top 20% | 42/1,010 C4，Precision 4.16%，Recall 55.26% |

### 8.2 Official Test

Test 包含 5,045 辆车，其中 60 辆 C4，C4 prevalence 约为 1.19%。

| 指标 | 结果 |
|---|---:|
| Average Precision | 0.04118 |
| ROC-AUC | 0.72049 |
| Top 1% | 3/51 C4，Precision 5.88%，Recall 5.00% |
| Top 2% | 7/101 C4，Precision 6.93%，Recall 11.67% |
| Top 5% | 11/253 C4，Precision 4.35%，Recall 18.33% |
| Top 10% | 17/505 C4，Precision 3.37%，Recall 28.33% |
| Top 20% | 26/1,009 C4，Precision 2.58%，Recall 43.33% |

Test Top 5% precision 约为总体 prevalence 的 3.66 倍，说明风险排序能把 C4 集中到队列前部，但绝对 precision 仍然较低。因此系统输出应解释为检查优先级，不应直接等同于确诊结果。

### 8.3 为什么 AP 仍然有用

AP 不直接告诉运营人员每天检查多少辆车，但它衡量整个 precision-recall 排序质量，特别适合 C4 这种严重不平衡的正类。项目真正的业务指标仍应优先看 Precision@K、Recall@K 和 C4 found@K；AP 用于判断模型在不同检查容量下是否整体保持更好的排序，而 ROC-AUC 用于衡量 C4 与非 C4 的总体区分能力。

## 9. Rule Engine

### 9.1 最终参数

| 参数 | 值 | 含义 |
|---|---:|---|
| Risk6 priority threshold | P99 | 极高短期风险 |
| Risk12 inspect threshold | P97 | 高近期风险 |
| Risk6 secondary inspect threshold | P97 | 配合多个 horizon 上升 |
| Risk24 threshold | P90 | 增强监控 |
| Risk48 threshold | P80 | 常规监控 |
| persistence | 0.80 | 最近最多 5 次中至少 80% 高于对应阈值 |
| fast deterioration | P85 | 正向风险变化位于当批次前 15% |
| minimum rising horizons | 2 | 至少两个 risk horizons 同时上升 |
| volatility threshold | P95 | Risk6 rolling volatility 位于前 5% |
| uncertainty threshold | P90 | 模型分歧和五分类不确定度都位于前 10% |

### 9.2 规则顺序

规则按顺序执行，命中第一条后停止：

| Rule | 条件 | Action |
|---|---|---|
| TRD | 少于 2 次有效 risk observations | `COLLECT_MORE_DATA` |
| TR0 | 少于 3 次观察，且 Risk6 或 Risk12 ≥ P97 | `MANUAL_REVIEW` |
| TRV | Risk6 波动 ≥ P95，且同时存在独立 uncertainty flag | `MANUAL_REVIEW` |
| TR1 | Risk6 ≥ P99，且 Risk6 持续、Risk6 快速恶化或 Risk12 快速恶化 | `PRIORITY_INSPECT` |
| TR2 | Risk12 ≥ P97，且 Risk12 持续、Risk12 上升或 Risk6 快速恶化 | `INSPECT` |
| TR3 | Risk6 ≥ P97，且至少两个 horizons 上升 | `INSPECT` |
| TR4 | Risk24 ≥ P90，且 Risk24 持续或上升 | `ENHANCED_MONITOR` |
| TR5 | Risk48 ≥ P80，且 Risk48 持续或上升 | `MONITOR` |
| TR6 | 其他情况 | `CONTINUE` |

### 9.3 冻结 Test 评估

| 范围 | 数量 | 识别 C4 | Precision | Recall | 同规模纯模型 |
|---|---:|---:|---:|---:|---:|
| PRIORITY_INSPECT | 15 | 1 | 6.67% | 1.67% | 1 |
| PRIORITY + INSPECT | 158 | 9 | 5.70% | 15.00% | 10 |
| 全部 attention actions | 1,207 | 34 | 2.82% | 56.67% | 29 |

因此应采用以下混合解释：

- 立即检查队列：保留模型排序，rule engine 不替代模型。
- 更广的 monitor/review attention：rule engine 利用历史趋势多找到 5 辆 C4。
- `MANUAL_REVIEW` 表示由于高风险但历史不足，或风险剧烈波动且模型不确定，系统无法稳定得出自动结论；它不同于 `INSPECT`，后者已有相对明确的近期风险证据。
- `COLLECT_MORE_DATA` 不是低风险判断，而是当前数据不足以判断。

## 10. 推理和数据库流程

### 10.1 输入车辆范围

推理只处理：

```sql
is_active IS TRUE
AND COALESCE(in_maintenance, FALSE) IS FALSE
```

即只分析仍在车队中、且当前不处于维修状态的车辆。

### 10.2 历史回放方式

默认生成 n−6 到 n−1 共 6 个 snapshot。需要特别说明：当前代码中的 offset 是 **readout index offset**，不是严格的自然日 offset。`n−6` 表示去掉每辆车最后 6 条 readouts 后所能看到的最新状态，`n−1` 表示去掉最后 1 条 readout 后的状态；系统再将这些 snapshot 标记为连续日期。

由于数据采样间隔不一定规律，这六个点不能严格称为“过去六个真实自然日”。更准确的说法是“最近六个历史 readout positions 的模拟回放”。如果报告必须宣称按天预测，需要进一步以真实 timestamp 或 time_step 对齐日历窗口，而不是按行数截断。

### 10.3 每个 snapshot 的计算

1. 按 `vehicle_id, time_step` 排序。
2. 截断未来 readouts，保证只使用该 snapshot 当时可见的信息。
3. 选择每辆车截断后的最后一条记录。
4. 构造 534 个特征，并严格核对训练时保存的 feature names。
5. 计算 C4 ensemble Risk6 排名。
6. 使用五分类组件计算 state、confidence、Risk12、Risk24、Risk48。
7. 在同一 prediction date 内把各 risk score 转为 fleet percentile。
8. 拼接多个 snapshot，计算 trend、persistence、fast deterioration、volatility 和 uncertainty。
9. 执行 rule engine，生成 action、priority、reason 和 rule trace。
10. 输出 CSV preview；只有显式提供 `--write-db` 才写入 PostgreSQL。

### 10.4 数据库存储安全

- 默认 dry run，不修改数据库。
- `--write-db` 才执行写入。
- `--replace-existing` 仅替换当前批次涉及的车辆和日期，不清理其他历史数据。
- analysis 与 SHAP 在同一 transaction 中写入。
- SHAP 通过插入后返回的 `analysis_id + vehicle_id + prediction_at` 关系绑定，不依赖 `prediction_step`。

## 11. 可解释性

每个有效预测保存 XGBoost C4 组件绝对 SHAP 值最大的前 5 个特征，包括：

- `feature_name`
- `feature_value`
- `shap_value`
- `contribution_rank`

需要注意，当前 SHAP 只解释 ensemble 中的 `c4_xgboost` 组件，不是对完整 XGBoost + LightGBM + veto 最终排序公式的严格整体解释。因此界面上应写成“Top XGBoost C4 feature contributions”，而不是“最终 ensemble 的完整因果解释”。

Rule engine 同时保存 `rule_id`、阈值 percentile、历史 observation 数量、source offset、最终 action 和 reason。SHAP 回答“哪些特征推动了一个基础模型的 C4 分数”，rule trace 回答“系统为什么给出这个 action”，两者作用不同。

## 12. 当前模型的优势

1. **目标与实际检查容量一致。** 直接优化 C4 排序，而不是被 C0 主导的总体 Accuracy。
2. **利用模型互补性。** XGBoost 和 LightGBM rank averaging 降低单一模型的偶然偏差。
3. **veto 控制前端误报。** C0/C4 专项模型抑制证据不足的高风险车辆。
4. **past-only 特征降低泄漏风险。** lag 特征只使用同一车辆的历史 readouts。
5. **车辆内样本权重更合理。** 避免读数多的车辆对训练产生过度影响。
6. **输出适合决策。** 同时提供风险排序、状态、时间范围、历史趋势、规则 action 和解释信息。
7. **部署结构清晰。** 一个推理脚本和一个 ensemble 文件，运行时只加载一个模型包。

## 13. 主要限制和风险

### 13.1 Test 已被查看

Test 39 的组合方法进行了 post-hoc Test 对比，因此 Test 指标可能略带选择乐观性。若要获得严格独立结果，需要新的完全未查看车辆集，或在 Train 内重新建立嵌套 cross-validation 后锁定模型。

### 13.2 Validation 到 Test 存在性能下降

AP 从 0.0667 降至 0.0412，AUC 从 0.7877 降至 0.7205；Top 5% precision 从 9.49% 降至 4.35%。这表明不同数据切分之间存在显著分布变化，不能只报告 Validation。

### 13.3 Risk6 不是概率

Risk6 是批次内 percentile rank。它适合排序，但不能跨不同批次直接比较绝对值，也不能解释为真实故障概率。

### 13.4 状态和 Risk6 来自不同组件

`state` 来自五分类 XGBoost，Risk6 来自 C4 ensemble。因此可能出现 state 为 C0、但 Risk6 排名较高的情况。这不一定是代码错误，而是两个目标不同：一个选择最可能类别，一个专门排序 C4 风险。界面应同时展示 confidence、risk percentiles 和 rule evidence，不能只展示 state。

### 13.5 历史点不等于自然日

当前六点回放按 readout 序号截断。采样不规则时，趋势图横轴如果显示连续日期，会给人“每日等间隔测量”的错误印象。建议界面标记为 snapshot sequence，或后续改成真实时间窗口。

### 13.6 SHAP 不是完整 ensemble 解释

当前 SHAP 只来自 XGBoost C4 组件。LightGBM、veto 和 rank/min 操作对最终排序的影响通过 `base_rank`、`veto_rank` 和 `model_disagreement` 反映，但没有统一的 ensemble SHAP。

### 13.7 C4 绝对 precision 仍然有限

Test Top 5% precision 为 4.35%，意味着 253 辆高风险车辆中有 11 辆 C4。模型能显著富集 C4，但不能作为自动诊断工具。最终维护判断仍应由人工检查和真实维修证据完成。

## 14. 建议的项目表述

建议在报告或答辩中使用以下表述：

> We developed a capacity-aware C4 risk-ranking ensemble rather than relying on overall classification accuracy. The final model combines XGBoost and LightGBM C4 rankings with a conservative C0-versus-C4 veto. On the official Test split, it achieved a C4 AP of 0.0412 and ROC-AUC of 0.7205, identifying 11 of 60 C4 vehicles within the top 5% of the fleet. A validation-selected temporal rule engine is used as a monitoring and human-review layer; it improves broader C4 attention coverage but does not replace the model ranking for the limited immediate-inspection queue.

不建议使用以下过度表述：

- “模型能准确预测车辆什么时候一定会维修。”
- “Risk6 代表未来 6 天发生故障的真实概率。”
- “Rule engine 在相同立即检查数量下提高了 C4 识别。”
- “Test 是完全未查看的独立盲测。”
- “六个 snapshot 就是六个真实自然日。”

## 15. 可复现文件

| 内容 | 路径 |
|---|---|
| 最终训练脚本 | `SystemCode/data/train_fleetcare_c4_risk_model.py` |
| 独立推理与 rule-engine 脚本 | `SystemCode/data/fleetcare_c4_risk_inference.py` |
| 单一模型包 | `SystemCode/data/processed/fleetcare_c4_risk_model/fleetcare_c4_risk_ensemble.joblib` |
| 模型 metadata | `SystemCode/data/processed/fleetcare_c4_risk_model/metadata.json` |
| Validation/Test 指标 | `SystemCode/data/processed/fleetcare_c4_risk_model/evaluation_results.json` |
| Test 39 结果 | `SystemCode/data/processed/test_39_c4_veto_sensitivity/summary.json` |
| Rule profile 搜索 | `SystemCode/data/processed/test_40_rule_engine_search/selection_and_test.json` |
| 冻结规则 Test 评估 | `SystemCode/data/processed/test_42_rule_engine_final/final_evaluation.json` |

## 16. 最终结论

当前模型已经适合作为学校项目中的完整技术方案：它有明确的问题定义、针对严重不平衡数据的开发过程、可解释的 feature engineering、多模型比较、独立的风险排序设计、validation/test 定量结果、rule-based reasoning、SHAP 解释和数据库部署流程。

最重要的结论不是“模型已经可以自动决定维修”，而是：**相比早期 baseline，当前 ensemble 明显提高了 C4 在有限检查队列中的集中度；rule engine 再利用历史趋势扩大监控覆盖，同时把数据不足和不稳定预测交给人工复核。** 这符合一个 intelligent reasoning system 的定位：模型提供风险证据，规则把证据转化为可追踪 action，最终决策保留给人。
