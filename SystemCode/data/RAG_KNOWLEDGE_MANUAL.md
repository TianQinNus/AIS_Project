# FleetCare 基础维修知识手册

版本：2026-10-08，项目演示参考。已核对公开来源，未获得维修专家认证。Component X、真实车型及发动机身份未知，本手册不能作为该部件的实际维修规程。

手册分为公开资料摘要和项目自定义政策。英国参考不作为新加坡法规，澳大利亚历史指南不直接设置车辆维护周期。具体维修参数须核对实际车辆适用的 OEM 资料。


# 出车前通用检查 Daily walkaround inspection
来源：DVSA 官方 HGV 日常检查指南，2023-09-21 更新，2026-10-08 查阅。适用：英国重型货车的一般检查参考，不能作为新加坡法规或 Component X 故障诊断。
原始资料：[出车前通用检查 Daily walkaround inspection](https://www.gov.uk/guidance/carry-out-daily-heavy-goods-vehicle-hgv-walkaround-checks)。
检查仪表和发动机、排放、ABS、EBS 警告灯，以及灯光、视野、转向和安全带。检查可见燃油或机油泄漏、蓄电池状态及异常排烟。柴油车检查 AdBlue 余量。
这些检查项目属于通用车辆检查；模型 C4 或高 Risk6 不表示已经发现上述异常。

# 制动与轮胎通用检查 Brakes and tyres
来源：DVSA 官方 HGV 日常检查指南，2026-10-08 查阅。适用：英国重型货车的一般检查参考。
原始资料：[制动与轮胎通用检查 Brakes and tyres](https://www.gov.uk/guidance/carry-out-daily-heavy-goods-vehicle-hgv-walkaround-checks)。
检查气压建立及报警是否正常、是否漏气，验证行车和驻车制动功能。检查制动管路、接头和拖车驻车制动。检查轮胎充气、破损、露帘线及车轮紧固状态。
具体压力、紧固力矩及法定标准须使用实际车辆资料和所在地规定；本摘要不提供统一数值。

# 缺陷记录与途中处置 Defect reporting
来源：DVSA 官方 HGV 日常检查指南，2026-10-08 查阅。适用：通用缺陷记录参考。
原始资料：[缺陷记录与途中处置 Defect reporting](https://www.gov.uk/guidance/carry-out-daily-heavy-goods-vehicle-hgv-walkaround-checks)。
记录车辆标识、日期、缺陷或症状、严重程度评估、检查人员及报告对象；未发现缺陷也应记录。途中发现异常，应在安全地点停车评估并报告；危险缺陷须先修复再继续行程。
模型 CONTINUE 不能证明车辆没有实际缺陷，也不能替代现场安全检查。

# 日常与定期检查 Maintenance inspection planning
来源：英国 HSE 官方 Inspection, maintenance and repair 页面，2026-10-08 查阅。适用：工作场所运输车辆的一般维护参考，非 Scania 特定维修指令。
原始资料：[日常与定期检查 Maintenance inspection planning](https://www.hse.gov.uk/workplacetransport/vehicles/inspection.htm)。
按制造商手册安排维护；检查人员应受过相应培训并知道如何报告问题。日常项目可包括轮胎、灯光、车轮紧固、制动、转向、喇叭和刮水器，关注可见损伤。
具体维护项目应遵循制造商手册。项目解释限制：不能把模型的 operational units 换成日历保养周期。

# Scania 保养计划的适用范围 Preventative maintenance scope
来源：Scania Australia Preventative Maintenance Guide，October 2020，第1页；2026-10-08 查阅。适用：所列 P-G-R-T / L-P-G-R-S 车辆及澳大利亚使用条件，仅作历史通用参考。
原始资料：[Scania 保养计划的适用范围 Preventative maintenance scope](https://www.scania.com/content/dam/scanianoe/market/au/products-and-services/services/ScaniaPreventativeMaintenanceGuides_A3_October2020_FINAL.pdf)。
该指南区分长途、重载、建筑及配送用途。驾驶方式、路线、载重与维护质量会影响部件寿命；冷却系统、传动、车轮、电气等属于计划维护范围。
本项目未知车辆和发动机配置，因此不移植表中的里程、小时、油品或更换周期，也不将水泵等具体部件等同于 Component X。实际计划须核对当前适用的 OEM 资料。

# 获取准确的 Scania 维修资料 OEM repair information
来源：Scania Group Repair and maintenance information 官方入口，2026-10-08 查阅。
原始资料：[获取准确的 Scania 维修资料 OEM repair information](https://www.scania.com/group/en/home/products-and-services/services/rmi.html)。
官方页面提供 RMI Portal、技术资料商店、诊断及技术支持入口。本文仅说明资料获取渠道，没有取得特定车辆的维修步骤。
在获得可核实的车辆型号、发动机标识及症状后，应查询对应资料；本项目的匿名 vehicle_id 不等于真实 VIN。

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
数据库当前没有持久化完整 Risk48、Risk24、Risk12 原始分数及所有运行时间信息。未提供的数值应说明缺失，不能由 LLM 编造。


# FleetCare 检查交接建议 Project-authored workflow
这是本次为项目设计的交接流程，不是 OEM 维修规程，尚未经过维修专家验证。
当数据库返回 PRIORITY_INSPECT 或 INSPECT，向维修人员交接车辆标识、最新分析时间、原始 state、action、rule_id、evidence、reason 和可用 SHAP。先确认实际车辆、工况、司机症状及故障码；通用检查依据维修知识库中的公开来源，实际维修依据对应 OEM 手册。
MANUAL_REVIEW 应说明证据不足并交人工核查。MONITOR、ENHANCED_MONITOR 应记录后续可比较的运行读数和症状，再由既有模型及规则评估。CONTINUE 仍需常规检查。
没有症状、部件身份及相符手册时，不能指定更换部件、停驶时长、保养周期或维修工时。项目动作等级不等于车辆适行许可。
