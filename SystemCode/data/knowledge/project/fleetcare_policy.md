# FleetCare 项目定义与知识边界
依据：model 分支提交 339c8b2297e181919853366a99ea518198f3abde 的 fleetcare_inference.py 与 rag.py。项目政策，不是官方 Scania 维修标准。
C0 表示模型未预测48运行时间单位内维修；C1 为24–48，C2 为12–24，C3 为6–12，C4 为0–6。运行时间单位不是小时或日历天。
Risk48=P(C1)+P(C2)+P(C3)+P(C4)，Risk24=P(C2)+P(C3)+P(C4)，Risk12=P(C3)+P(C4)，Risk6=P(C4)。这些是模型分数，不是已校准实际维修概率。分位数是0–100的车队相对排名，不是概率。

# TR0 历史证据不足 Manual review
当前 C3/C4 且历史少于3条，规则先返回 MANUAL_REVIEW。规则按顺序匹配；最终 action、rule_id 和 evidence 应读取数据库，而非让语言模型重新计算。

# TR1 优先检查 Priority inspect
C4 且 Risk6 分位数至少95，并有持续高 Risk6 或快速近端恶化证据，返回 PRIORITY_INSPECT。

# TR2 与 TR3 检查 Inspect
TR2：C3/C4，且至少满足 Risk12 分位数至少90、持续高 Risk12、快速近端恶化或类别恶化之一，返回 INSPECT。
TR3：类别低于C3，但 Risk6 分位数至少95，且持续高 Risk6 并快速恶化，返回 INSPECT。仓库目前进展.txt 中的 TR3 摘要与代码有差异，应以代码/数据库为准。

# TR4 至 TR6 监控与继续
TR4：C2，或高 Risk12 且有持续或恶化支持，返回 ENHANCED_MONITOR。
TR5：C1，或高 Risk48 且有持续或恶化支持，返回 MONITOR。
未命中前述规则则 TR6 返回 CONTINUE。这些动作是项目优先级，不是物理故障诊断。

# SHAP 与已存历史的解释限制
SHAP 描述匿名特征对预测的贡献，不能确定具体故障部件。
analysis.prediction_at 是运行批次时间，现有接口查询的是最近数据库分析记录；重复运行相同读数也会追加记录。因此不能把多次存储记录直接当成独立运行读数或真实风险恶化。
最新 PostgreSQL analysis 表已保存 Risk48、Risk24、Risk12、Risk6 原始模型分数及 time_step、history_count、rule_trace 等字段。应读取实际返回值；NULL 表示缺失，不得编造。time_step 是运行时间单位，不是日历日期；相同 time_step 的分析重跑不能视为独立车辆读数。
