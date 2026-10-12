# 原文查看与来源审计（v3）

官方快照：knowledge/official/raw/dvsa.html、hse.html、lta_inspection.html、lta_commercial.html。
可读正文：knowledge/official/extracted/ 同名 JSON；项目说明：knowledge/project/fleetcare_policy.md。
有效目录为 knowledge/catalog.json。完整原文也保存在 PostgreSQL rag.kb_documents.original（BYTEA），来源信息在 metadata（JSONB）。网页使用 source_url、publisher、jurisdiction、verified_on、review_due 与 citation_id 显示来源和适用范围。

现在的数据库不是 SQLite 文件。SQLite 种子仅作历史迁移参考；运行只连接项目 PostgreSQL。当前可分发的 v3 种子为 knowledge_seed/pgvector_seed.json.gz，通过 migrate_knowledge.py 导入。

4 份官方原文核验日期 2026-10-08，项目字段说明更新为 2026-10-12。官方资料仍是通用检查和政策，不是车型 OEM 拆装维修手册。
