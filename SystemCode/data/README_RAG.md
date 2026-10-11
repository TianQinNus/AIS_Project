# FleetCare RAG v2：官方原文与本地向量数据库

更新：2026-10-11。网页团队请优先阅读 [README_RAG_WEB.md](README_RAG_WEB.md)，最新交接见 [RAG_PROGRESS.md](RAG_PROGRESS.md)。新增 HTTP 适配器与可分发 SQLite 种子库；本文说明核心知识构建与维护。

知识检索只访问你自己的本地 SQLite 数据库；车辆模型结果继续读取你配置的 MySQL。生成综合回答仍调用 OpenAI API，检索到的片段和车辆记录会作为上下文发送给模型。这不是完全离线的问答系统。

## 当前有效来源

- DVSA：英国 HGV daily walkaround checks 官方原文。
- HSE：英国车辆 inspection, maintenance and repair 官方原文。
- LTA Singapore：Periodic Vehicle Inspection 与 Commercial Vehicles 官方原文。
- FleetCare 项目规则说明：单独标识 project_policy，仅说明项目代码，不是官方维修标准。

`knowledge/catalog.json` 是有效目录。旧 `knowledge/maintenance/*.md`、交接建议与原有 dataset PDF 保留供历史参考，但不进入 v2 检索库。它们不能作为已核验的 OEM 维修手册。当前知识覆盖通用检查和官方政策，没有车型专属的拆装、扭矩、零件和维修工序。

官方完整 HTML 位于 `knowledge/official/raw/`；确定性提取的段落和表格行位于 `knowledge/official/extracted/`。提取过程不使用 LLM、不改写或翻译原文；表格行重复其原文表头。所有来源带发布机构、URL、司法地区、获取/核验日期、审核期限、适用范围和 SHA-256。`reviewed=true` 只指来源身份、提取结构的检查，不表示专家签署或保证所有内容永远正确。哈希能检测修改，不能证明内容正确。

## 安装与首次建立

在仓库 `SystemCode` 下：

```bash
python -m pip install -r requirements-rag.txt
cd data
python build_knowledge.py
python evaluate_knowledge.py
python -m pip install -r ../requirements-rag-web.txt
python -m unittest discover -s tests -v
```

当前机器已存在本地嵌入模型缓存，已经建立数据库。若另一台机器尚无缓存，只在建库阶段明确允许下载：

```bash
python build_knowledge.py --allow-model-download
```

默认模型是 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`，384 维。查询阶段固定 `local_files_only=True`；模型缺失时报告错误，不自动联网下载。CPU 上运行。可用 `RAG_EMBEDDING_MODEL` 在建库时指定预先下载的本地模型目录；查询时读取数据库中的该路径，部署时路径必须有效。

## 查询

无需 OpenAI/MySQL 密钥即可验证知识检索：

```bash
python knowledge_rag.py '新加坡超过10年的货车多久需要定期车辆检验？' --kind policy
python knowledge_rag.py '气压建立异常和制动管路漏气应检查什么？' --kind maintenance
```

检索不会读外部文档、catalog，也不会搜索网页。数据库包含原文快照，因此原始文件目录被移走后仍可查询。原文件修改不会即时改变已发布数据库，需核验并重新建库。数据库只读打开；缺失时不会悄悄创建空库。使用 NumPy 计算本地向量余弦相似度，结合 BM25 和排序融合；不是服务器服务或 ANN 引擎，适合此小型知识库。中文词汇映射仅帮助查找英文原文，不增加政策事实。

综合模型结果与知识回答：先按 `.env.rag.example` 将真实凭据 export 到环境，然后：

```python
from fleetcare_agent import ask_fleetcare, ask_fleetcare_detailed
print(ask_fleetcare('车辆目前为什么需要检查？请结合官方资料说明下一步检查参考。', vehicle_id=实际车辆ID))
result = ask_fleetcare_detailed('新加坡货车的定期检验政策是什么？')
# result['sources'] 含原文片段、出处、地区、核验日期和引用ID。
```

`实际车辆ID` 请换成数据库中存在的整数。Python 示例不是可直接运行的占位变量定义。网页/应用现有调用接口无需改变。默认知识库是 `data/knowledge_index/fleetcare_knowledge.sqlite3`，可通过 `FLEETCARE_KNOWLEDGE_DB` 指定绝对路径。遗留 `FLEETCARE_KNOWLEDGE_INDEX` 如指向 index.npz，请 unset 或改为 sqlite3 路径。NPZ 不再用于问答；不自动导入旧摘要。

## 维护来源与政策

只在维护时联网：

```bash
python fetch_official_sources.py
# 检查 raw、extracted 和 catalog 的变更后
python build_knowledge.py
python evaluate_knowledge.py
```

抓取仅针对代码中的固定官方 HTTPS 页面，拒绝异域跳转、缺失正文与不符预期的页面。抓取命令刷新快照但不会改变问答使用的数据库，建库才发布新版本。若网站版式变化，需要人工检查提取选择器。

政策每 90 天、维护指导每 365 天安排重新审核；这是项目审核期限，不是法律规定或官方有效期。查询时超过审核期限的来源自动排除，不联网补齐。即使未到期限，法规也可能变化，因此新政策变更后应主动刷新，避免把快照称为实时法规。政策查询默认 SG；英国指导不会被当作新加坡法律。法律适用还需确认车型、注册车龄、重量等真实资料，模型风险类别不能决定法定检验周期。

新增 OEM 手册需取得合法原文、准确车型/版本/适用范围与来源凭据，在可信发布者允许列表和 catalog 中注册。未核验文档不能直接丢进目录就进入检索。现在的 Component X 匿名数据无法确定零件身份，不能据此生成具体维修工序。

## 数据结构与版本

- `kb_versions`：模型名、维数、来源目录摘要和版本元数据。
- `kb_documents`：来源元数据、提取文本文件和原文快照 BLOB。
- `kb_chunks`：知识片段、向量 float32 BLOB、文本/向量 SHA-256。
- `kb_state`：当前生效版本。

建库在 SQLite 事务中发布，旧版本保留；失败不切换有效版本。检索器检测有效版本变化并加载新版本。载入校验数据库内的来源、目录、片段与向量完整性，拒绝损坏数据。一次查询使用一个已加载的完整版本。

SQLite 与 MySQL 由 agent 汇总：MySQL 提供预测/规则/SHAP，SQLite 提供原文证据。这是两个各司其职的本地存储，不是将关系表和向量物理合并为一个 MySQL 数据库。

## 验证范围

单元测试覆盖原文修改拒绝、数据库损坏、版本切换、过期和地区筛选、删除外部文档后查询、工具编排和引用 ID。`evaluate_knowledge.py` 使用真实嵌入模型，检查命中来源及部分关键表格行。它不能证明 LLM 回答没有幻觉，也不能验证具体维修适用性。没有真实 API/MySQL 凭据的运行环境不进行线上综合回答验证。
