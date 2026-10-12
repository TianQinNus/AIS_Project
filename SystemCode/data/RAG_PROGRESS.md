# RAG v3 进度：2026-10-12

当前架构：同一 PostgreSQL 数据库，public 业务表 + rag 知识表，pgvector 精确余弦查询。替代前一版 MySQL + SQLite/NPZ 运行架构。

完成：
- 按 fleetcare_new.sql 核对 public.analysis / prediction_shap / truck / truck_readout。
- 统一 PostgreSQL 配置、psycopg 驱动；车辆查询扩展真实风险/时间/规则字段。
- 新增 rag schema 建表 SQL，JSONB 原文元数据、BYTEA 原文、vector(384) 嵌入。
- 保留切分、BM25、融合、地区/期限和引用检查；相似度在 pgvector 中计算。
- 建库事务、版本切换、回滚、校验和维数约束；发布可直接导入的 v3 向量种子。
- 网页 ask/search 路径与主要字段兼容；backend 标识改为 postgresql_pgvector。
- 模型数据库连接改 PostgreSQL；INSERT RETURNING 对应本批分析 ID；预测公式不改。
- 当前 4 份官方原文 +1 份项目说明，274 片段、384 维。

验证：41 项测试、13 条真实检索用例；提供 SQL 的分析/SHAP 样本查询，插入 ID、种子重复导入及组合查询已验证。官方知识的 SQLite/pgvector 距离对照见交付测试报告。

边界：本轮 SQL/pgvector 执行使用 PGlite 的 PostgreSQL 18.3 测试引擎，原生 PostgreSQL 受共享内存限制。未连接团队 PostgreSQL 16.15 服务；没有实际 OpenAI 生成或完整模型管线运行。团队部署时应按 README 配置实际数据库、扩展、权限和模型缓存后复核。

maintenance 回答通用检查、缺陷报告和维护安全参考，不做具体零件故障诊断或未经 OEM 手册支持的拆装步骤。官方原文与政策适用边界保持不变。

网页交接：新增 WEB_RAG_DEPLOYMENT.md 逐步操作指南；prepare_embedding_model.py 只准备缓存；check_rag_setup.py 只读检查数据库、扩展、知识和车辆查询。GitHub同步包含代码/种子/报告，不包含业务dump或真实凭据。
