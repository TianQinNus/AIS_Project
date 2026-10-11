# 知识原文查看与审计

有效知识以 `catalog.json` 为准，旧维护摘要未纳入当前数据库。

| 发布机构 | 完整原文快照 | 可读提取文本 | 分类/适用范围 |
|---|---|---|---|
| DVSA | knowledge/official/raw/dvsa.html | knowledge/official/extracted/dvsa.json | 英国通用检查指导 |
| HSE | knowledge/official/raw/hse.html | knowledge/official/extracted/hse.json | 英国检查维护指导 |
| LTA | knowledge/official/raw/lta_inspection.html | knowledge/official/extracted/lta_inspection.json | 新加坡检验政策 |
| LTA | knowledge/official/raw/lta_commercial.html | knowledge/official/extracted/lta_commercial.json | 新加坡商用车政策 |
| FleetCare | knowledge/project/fleetcare_policy.md | 同左 | 项目规则，非官方维修标准 |

HTML 是从官方站点获取的完整网页快照，可能依赖远程样式/图片。JSON 是其正文文字，不是重新编写的知识总结。原文 URL、获取时间和 SHA-256 都记录在目录和 SQLite 中。请通过原始 URL 复核政策变更；日常问答只使用已发布数据库快照。

检索结果包含 section、block、token_start 和 citation_id，可沿这些字段定位正文。表格保留并重复原表头；中文查询使用多语言嵌入和本地搜索词映射，输出知识引用仍对应英文原文。

数据库完整备份应复制 knowledge_index/fleetcare_knowledge.sqlite3，同时保留嵌入模型缓存或本地模型目录。数据库中包含来源原文，不需要额外文档文件才能检索。SQLite 可通过 DB Browser for SQLite 查看。

具体安装、使用、更新和测试说明见 README_RAG.md。
