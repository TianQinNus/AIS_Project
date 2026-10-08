# FleetCare 检查交接建议 Project-authored workflow
这是本次为项目设计的交接流程，不是 OEM 维修规程，尚未经过维修专家验证。
当数据库返回 PRIORITY_INSPECT 或 INSPECT，向维修人员交接车辆标识、最新分析时间、原始 state、action、rule_id、evidence、reason 和可用 SHAP。先确认实际车辆、工况、司机症状及故障码；通用检查依据维修知识库中的公开来源，实际维修依据对应 OEM 手册。
MANUAL_REVIEW 应说明证据不足并交人工核查。MONITOR、ENHANCED_MONITOR 应记录后续可比较的运行读数和症状，再由既有模型及规则评估。CONTINUE 仍需常规检查。
没有症状、部件身份及相符手册时，不能指定更换部件、停驶时长、保养周期或维修工时。项目动作等级不等于车辆适行许可。
