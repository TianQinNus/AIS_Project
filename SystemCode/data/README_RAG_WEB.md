# FleetCare RAG v3 — PostgreSQL + pgvector

**网页成员首次操作请先看 [WEB_RAG_DEPLOYMENT.md](WEB_RAG_DEPLOYMENT.md)，按环境检查、种子导入、模型准备、search、ask、页面验收顺序执行。**

更新：2026-10-12。本版替代 MySQL + SQLite 的运行架构。模型业务数据和知识证据统一使用项目 PostgreSQL；知识相似度由数据库内 pgvector 的余弦距离 `<=>` 计算。网页请求/返回字段保持兼容，`retrieval_backend` 更新为 `postgresql_pgvector`。

## 结构与实验思路

用户提供的 fleetcare_new.sql 为 PostgreSQL 16.15 dump：业务表是 public.analysis、public.prediction_shap、public.truck、public.truck_readout；虽然文件创建 fleetcareai schema，但业务表不在该 schema。文件不包含 vector 扩展或知识表。

同一个数据库新增 rag schema：kb_versions、kb_documents、kb_chunks、kb_state。JSONB 保存来源与片段，BYTEA 保存原文和校验字节，embedding 为 public.vector(384)。已有业务表不由知识迁移更改；不会自动执行业务 dump。

保持官方文档、384 维多语言模型、切分、BM25、融合排序、政策地区/期限过滤一致，先比较存储迁移后的结果。当前库小，使用精确扫描全部有效版本片段，不引入 ANN 候选截断；日后扩大库时再独立评估 HNSW 召回率。

已按实际 PostgreSQL 字段扩展车辆结果：time_step、history_count、Risk48/24/12 原始分数、risk24_percentile、rule_trace、hybrid_ranking_score、priority_score；原字段不删除。NULL 是缺失，prediction_at 是分析时间；time_step 是运行单位，相同 time_step 的重跑不是独立读数。

fleetcare_inference.py 的连接/读表改用 PostgreSQL，并用 INSERT RETURNING 取得本次插入 ID，去掉 MySQL 引号和按秒时间恢复 ID 的方式；模型训练、预测公式及规则不改。只有明确运行模型管线才写入业务表，RAG 问答不会重新运行预测。

## 配置与启动

Python 3.10+，在仓库根目录：

```bash
python -m pip install -r SystemCode/requirements-rag-web.txt
cd SystemCode/data
export FLEETCARE_DB_HOST='127.0.0.1'
export FLEETCARE_DB_PORT='5432'
export FLEETCARE_DB_USER='postgres'
export FLEETCARE_DB_PASSWORD='填写实际密码'
export FLEETCARE_DB_NAME='填写项目实际数据库名'
# 或 FLEETCARE_DATABASE_URL=postgresql+psycopg://user:password@host:5432/database
```

URL 优先于单独变量，密码特殊字符需 URL 编码；按单独变量配置时驱动处理转义。配置来自服务器环境，.env.rag.example 不自动加载。车辆查询和知识查询共用同一个连接配置，不再使用 FLEETCARE_KNOWLEDGE_DB / FLEETCARE_KNOWLEDGE_INDEX。

PostgreSQL 服务器必须安装 pgvector。数据库管理员启用 `CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;`，知识建库命令也会尝试启用并建立 rag 表。扩展若已安装在其他 schema，需要按团队配置调整，代码不会迁移已有扩展。建表脚本：sql/001_rag_pgvector.sql。普通只读查询账号无需建表权限。

首次导入已计算好的向量：

```bash
python migrate_knowledge.py
```

默认读取 knowledge_seed/pgvector_seed.json.gz，约 1.1 MiB，内含当前有效原文、274 个片段和向量；不需调用嵌入模型，重复导入幂等。只向 rag schema 写数据，事务发布新版本。模型权重不在种子包中。

另一种方式是从原文重新建库，同时在首次准备阶段下载模型：

```bash
python build_knowledge.py --allow-model-download
```

已有模型缓存时直接 python build_knowledge.py；查询时 local_files_only=True，模型缺失报错，不自动联网下载。使用种子导入后也必须在服务主机准备同一模型缓存；最简单方法是在准备阶段执行上述建库命令。旧 SQLite 文件仅用于显式一次性迁移，不再用于问答。旧种子与当前审核目录不符时 --sqlite 迁移拒绝，应使用 v3 种子或重新建库。

服务启动：

```bash
export OPENAI_API_KEY='实际密钥'
export FLEETCARE_LLM_MODEL='账户可用且支持Responses工具调用的模型'
export FLEETCARE_ALLOWED_ORIGINS='http://localhost:3000,http://localhost:5173'
python -m uvicorn rag_http_api:app --host 127.0.0.1 --port 8000
```

接口与 Swagger 仍在 /api/rag/ask、/api/rag/search、/docs。检索只需 PostgreSQL 和缓存模型；生成自然语言回答还需 OpenAI。浏览器不存储密钥/数据库密码。适配器为受控联调入口，公网接入应通过团队已有受保护后端。

## 网页调用契约

### POST /api/rag/ask

```json
{"question":"请解释这辆车的建议，并给出轮胎检查参考。","vehicle_id":123}
```

123 是格式示例，要替换为实际整数 ID；纯知识问题可省略 vehicle_id 或传 null。question 为非空字符串，最多 6000 字符，不接受其他字段或 kind。

返回 answer、vehicle_analysis、sources、retrieval_warnings、retrieval_trace。先检查 vehicle_analysis 每项 success，再读取 current/history/shap。优先展示 sources 中 cited_in_answer=true 的引用卡片，并展示警告；HTTP 200 不代表所有证据查询成功。相似度不是故障概率。current 中新增字段为可空值，不删除原字段。

### POST /api/rag/search

```json
{"question":"轮胎与车轮固定件应检查什么？","kind":"maintenance"}
```

kind 支持 maintenance / policy / project / dataset，也可省略；当前没有有效 dataset 来源。返回 success、query、hits、database_version、retrieval_backend=postgresql_pgvector、policy_jurisdiction、excluded_overdue_documents 等。这个入口严格按 kind 筛选，只给原文证据，不调用 OpenAI，不生成自然语言答案。

```javascript
const base = 'http://127.0.0.1:8000'; // 修改为团队实际后端地址
async function askFleetCare(question, vehicleId = null) {
  const response = await fetch(`${base}/api/rag/ask`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, vehicle_id: vehicleId }),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(JSON.stringify(result.detail));
  return result;
}
```

已有 Python 后端可以 from fleetcare_agent import ask_fleetcare_detailed，然后直接调用并返回 JSON，无需独立 HTTP 适配器。请求为单轮、非流式，UI 显示 loading 和错误，避免重复点击。CORS 配置准确来源，包括端口。状态：422 参数错误；ask 502 服务/生成异常；search 503 数据库/模型不可用。health 只表示进程存活。

## maintenance 回答范围

| 分类 | 依据 | 回答范围 |
|---|---|---|
| maintenance | DVSA HGV daily checks、HSE inspection/maintenance/repair | 出车前检查，制动/轮胎/警告灯/泄漏检查参考，缺陷记录报告，计划维护和安全注意事项 |
| policy | LTA Inspection、LTA Commercial Vehicles | 新加坡检验频率和商用车政策，需确认车型、车龄、重量 |
| project | FleetCare 项目说明 | 风险指标、TR0–TR6、SHAP、运行时间解释 |
| 车辆事实 | public.analysis / public.prediction_shap | 实际预测、行动、优先级、规则依据、SHAP |

maintenance 不提供未经对应 OEM 手册支持的部件诊断、Component X 身份、拆装步骤、扭矩、油液型号、零件号或更换周期。组合问答回答“为什么建议检查 + 官方资料提供什么通用检查参考”，不是自动维修方案。UK 指导不作为 Singapore 法律。

## 原文与更新

4 份官方快照核验日期 2026-10-08；项目说明按新 PostgreSQL 字段更新为 2026-10-12。共 5 份来源、274 片段、384 维。原文 knowledge/official/raw/*.html；正文 knowledge/official/extracted/*.json；有效目录 knowledge/catalog.json。原文同时存在 PostgreSQL rag.kb_documents.original。旧摘要与 dataset PDF 保留历史，不参与有效检索。

政策项目审核期限 2027-01-06；维护指导 2027-10-08；项目说明 2027-10-12。到期源自动排除；期限不是法律有效期，实际政策变更应提前刷新。哈希发现修改，不等于专家认证或来源签名。

```bash
python fetch_official_sources.py
# 人工检查原文、提取文本和目录；此命令不会修改已发布数据库
python build_knowledge.py
python evaluate_knowledge.py
```

查询不读源文件、不查网页。它会连接你配置的 PostgreSQL，是否位于本机由 host 配置决定；不会使用外部向量服务。LLM 生成仍通过 OpenAI，证据和车辆记录会成为模型上下文。

## 数据库查看

```sql
SELECT metadata FROM rag.kb_versions
WHERE version=(SELECT active_version FROM rag.kb_state WHERE id=1);
SELECT source,metadata FROM rag.kb_documents
WHERE version=(SELECT active_version FROM rag.kb_state WHERE id=1);
SELECT ordinal,payload->>'text' AS text,vector_dims(embedding) AS dimensions
FROM rag.kb_chunks
WHERE version=(SELECT active_version FROM rag.kb_state WHERE id=1)
ORDER BY ordinal LIMIT 5;
SELECT source,convert_from(original,'UTF8') AS original_html
FROM rag.kb_documents
WHERE version=(SELECT active_version FROM rag.kb_state WHERE id=1);
```

## 验证与环境边界

41 项单元/数据库/HTTP 契约测试，13 条真实模型检索用例；另验证提供 SQL 的业务表字段和分析/SHAP 样本、模型 INSERT RETURNING、种子重导入、原文目录无需读取、模拟生成下的组合查询。迁移比较保持官方知识相同，比较旧 SQLite 向量点积与 pgvector 余弦结果。

本轮原生 PostgreSQL 共享内存被运行环境限制，因此数据库执行验证使用 PostgreSQL 18.3 的 PGlite 测试引擎 + pgvector 0.8.1，经 psycopg/SQLAlchemy 连接；它不能代替团队 PostgreSQL 16.15 上的部署、并发和权限测试。未连接或修改团队实际数据库，未调用真实 OpenAI 生成，也未重新训练或跑完整预测管线。

```bash
python evaluate_knowledge.py
# 无测试 URL 时数据库测试会跳过
python -m unittest discover -s tests -v
# 要执行全部测试：配置一份独立、可清空的 PostgreSQL 测试数据库
# FLEETCARE_TEST_DATABASE_URL=postgresql+psycopg://.../fleetcare_rag_test
```

数据库测试会 TRUNCATE rag.kb_versions CASCADE；绝不能把测试 URL 设置为实际项目库。evaluate_knowledge.py 只读检索，可在正常库运行。测试通过不表示 LLM 事实准确率或机械维修适用性已经认证。
