# 网页同学操作指南：PostgreSQL + pgvector RAG

更新：2026-10-12。维护人员本机没有可用的原生 PostgreSQL 环境，本版 SQL/pgvector 已在隔离 PGlite 测试引擎复核；网页成员负责在团队 PostgreSQL16部署环境和真实OpenAI账户完成最终联调。不要把隔离测试结果当作线上已部署。

## 0. 本次接入不变与变化

接口路径仍为 POST /api/rag/ask、POST /api/rag/search；主要响应字段仍为 answer / vehicle_analysis / sources / retrieval_warnings / retrieval_trace。变化是后端存储统一 PostgreSQL，检索 backend 为 postgresql_pgvector，车辆结果增加风险分数/运行时间/规则字段。

业务表在 public，不是 fleetcareai schema。知识表在同一数据库 rag schema，384维。本版种子为 knowledge_seed/pgvector_seed.json.gz，5份资料、274片段。旧 SQLite/NPZ 不参与运行。

## 1. 获取代码并准备 Python

在已有仓库中，先保存自己的本地改动，再更新 model 分支；不要用强制覆盖命令丢弃网页代码。

```bash
git switch model
git pull --ff-only origin model
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r SystemCode/requirements-rag-web.txt
cd SystemCode/data
```

Windows PowerShell 虚拟环境激活为 `.venv\Scripts\Activate.ps1`。后续命令都在 SystemCode/data 执行。

## 2. 选择数据库，确认业务数据

优先用团队已经导入数据的 PostgreSQL16库；需要服务器已安装 pgvector 扩展文件，pip 安装 Python 驱动不会安装数据库服务或服务器扩展。

若新建一份验证环境，使用独立数据库，再由持有数据的成员通过匹配 dump 的 psql 客户端导入 fleetcare_new.sql。它是完整业务dump，不在Git仓库中，不要在已有项目库重复导入。

可选：有 Docker 的成员可以准备临时 PostgreSQL16+pgvector环境。以下容器没有业务数据，仍需单独导入到此新库；密码仅为隔离开发示例，实际配置请自选。

```bash
docker run --name fleetcare-rag-dev   -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=dev_only_change_me   -e POSTGRES_DB=fleetcareai -p 127.0.0.1:5433:5432   -v fleetcare-rag-dev-data:/var/lib/postgresql/data   -d pgvector/pgvector:pg16
```

该方案在本轮未执行；已有团队数据库不需要再启动容器。依据 [pgvector 官方安装说明](https://github.com/pgvector/pgvector#docker)。导入dump的psql需支持其 `\restrict` 命令，推荐与提供dump匹配的16.15或更新维护版本客户端；报unknown command应升级客户端，不修改原始dump。

配置实际连接（macOS/Linux）：

```bash
unset FLEETCARE_DATABASE_URL
export FLEETCARE_DB_HOST='127.0.0.1'
export FLEETCARE_DB_PORT='5432' # 上述 Docker 示例用5433
export FLEETCARE_DB_USER='实际用户'
export FLEETCARE_DB_PASSWORD='实际密码'
export FLEETCARE_DB_NAME='实际数据库名'
```

Windows PowerShell：

```powershell
Remove-Item Env:FLEETCARE_DATABASE_URL -ErrorAction SilentlyContinue
$env:FLEETCARE_DB_HOST='127.0.0.1'
$env:FLEETCARE_DB_PORT='5432'
$env:FLEETCARE_DB_USER='实际用户'
$env:FLEETCARE_DB_PASSWORD='实际密码'
$env:FLEETCARE_DB_NAME='实际数据库名'
```

也可用 FLEETCARE_DATABASE_URL，设置后优先于上述独立变量；URL里的特殊密码字符需编码。`.env.rag.example` 只是模板，不会自动加载。秘密只在服务器环境，不写前端或Git。

在数据库客户端执行：

```sql
SELECT version();
SELECT n.nspname,e.extversion FROM pg_extension e
JOIN pg_namespace n ON n.oid=e.extnamespace WHERE e.extname='vector';
SELECT vehicle_id,analysis_id,state,action,time_step
FROM public.analysis ORDER BY analysis_id DESC LIMIT 5;
```

目标：能连接、vector在public、有业务记录。缺少扩展时由管理员执行 `CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;`。若扩展文件未安装，先在服务器安装；若已在其他schema，不要自行移动，联系后端确认配置。知识迁移不改已有业务表。

## 3. 导入知识、准备模型

```bash
python migrate_knowledge.py
```

使用有建表/写入rag权限的接入账号。它创建rag表并导入现有向量；应输出 backend=postgresql_pgvector、documents=5、chunks=274、dimension=384。不会重新推理或写入public业务表，也不会访问OpenAI。

准备嵌入模型，首次明确允许下载：

```bash
python prepare_embedding_model.py --allow-download
```

这一步只下载/缓存并检查384维模型，不改数据库。有缓存时执行 `python prepare_embedding_model.py`，应该显示 cache_ready=true。查询阶段不联网下载。

从原文重新建库是另一种选择：python build_knowledge.py --allow-model-download；当前种子已计算好，不需要为了下载模型而重新建库。

## 4. 只读检查与知识验证（无需OpenAI）

```bash
python check_rag_setup.py --skip-model
python check_rag_setup.py
python evaluate_knowledge.py
python knowledge_rag.py '轮胎与车轮固定件应检查什么？' --kind maintenance
```

check_rag_setup 不建表、不导入、不清空数据库，自动选现有车辆或接受 --vehicle-id。完整检查期望 ready_for_search=true、ready_for_vehicle_queries=true；同时查看业务字段、知识数量和SHAP行数。--skip-model只检查数据库部分，不证明模型已准备。

evaluate_knowledge.py预期13/13。结果应带 source_url、publisher、jurisdiction、verified_on 与 citation_id；检索命中不是故障概率。空hits不是正常答案，应提示缺少相关证据。

## 5. 启动HTTP，先验证search

```bash
export FLEETCARE_ALLOWED_ORIGINS='http://localhost:3000,http://localhost:5173'
python -m uvicorn rag_http_api:app --host 127.0.0.1 --port 8000
```

PowerShell 环境变量写法为 `$env:FLEETCARE_ALLOWED_ORIGINS='...'`。另一终端或浏览器打开 http://127.0.0.1:8000/docs ，执行 POST /api/rag/search：

```json
{"question":"轮胎有哪些检查事项？","kind":"maintenance"}
```

预期HTTP200、非空hits、retrieval_backend=postgresql_pgvector。这个接口只检索原文，不依赖OpenAI或车辆数据。/health只是进程存活。

## 6. 配置OpenAI，再验证ask

在启动服务的终端配置真实OpenAI密钥和账户可用模型，再重启服务：

```bash
export OPENAI_API_KEY='实际密钥'
export FLEETCARE_LLM_MODEL='账户实际可用且支持Responses工具调用的模型'
```

按顺序在Swagger验证：

1. 不传vehicle_id：`{"question":"出车前轮胎有哪些通用检查事项？"}`。
2. 传check_rag_setup返回的真实整数ID：`{"question":"请解释这辆车的行动建议，并提供轮胎检查参考。","vehicle_id":实际整数}`。此写法中的实际整数是说明占位，替换后才是合法JSON。
3. 一次不存在的车辆ID：页面应显示查询失败信息，不能编造车辆结果。

预期 answer 非空，sources 有出处和引用；车辆问题 vehicle_analysis.success=true，current 与数据库实际一致。查看retrieval_warnings；HTTP200不等于所有证据成功。记下真实analysis_id、车辆ID和返回截图供组内联调，不分享密码。

## 7. 前端接入与验收

浏览器只调用后端HTTP，不直接访问数据库或OpenAI。已有Python后端也可以直接调用 ask_fleetcare_detailed。使用 [README_RAG_WEB.md](README_RAG_WEB.md) 的 JavaScript 示例。

跨机器时填写实际后端IP/域名；对方的localhost不是你的机器。按受控团队网络绑定地址，配置实际前端origin和代理超时；公网接入通过已有受保护后端。当前服务为非流式单轮问答。

| 页面事项 | 验收要求 |
|---|---|
| 回答 | 显示answer，等待期间loading，防止重复提交 |
| 车辆结果 | 先判断每项success；真实整数ID，不凭文本随意猜测 |
| 引用 | 优先显示cited_in_answer=true的sources，含机构、URL、地区、核验日期 |
| 警告 | 展示retrieval_warnings、空hits和HTTP422/502/503 |
| backend | 支持postgresql_pgvector，不继续要求local_sqlite |
| 维护知识定位 | “检查参考”，非自动故障诊断/拆装维修方案 |

maintenance覆盖出车检查、制动/轮胎/警告灯/泄漏参考、缺陷记录报告、维护计划和操作安全。法定检验属policy，Risk/TR/SHAP说明属project；车辆优先级来自实际模型记录。具体维修工序需对应OEM手册。

完成后回传：提交号、PostgreSQL/pgvector版本、只读检查JSON、13/13评估结果、一个真实车辆组合回答、前端引用/警告截图、后端联调地址。截图隐藏凭据。

## 8. 完整测试与常见错误

```bash
python -m unittest discover -s tests -v
```

未配置FLEETCARE_TEST_DATABASE_URL时，数据库测试跳过。要复核全部41项，单独准备可清空的PostgreSQL测试库并设置此URL。测试会TRUNCATE rag表，绝不能指向业务/联调库。

| 错误 | 检查顺序 |
|---|---|
| connection refused / authentication failed | host、port、服务状态、账号、数据库名和网络 |
| extension vector is not available | PostgreSQL服务器安装pgvector，不是pip问题 |
| permission denied | 管理员/接入账号准备扩展与rag表；运行账号需SELECT权限 |
| relation rag.kb_state does not exist | 尚未导入种子，或连接错数据库 |
| business_schema缺失字段 | 表必须在public，结构应与最新fleetcare_new.sql一致 |
| 模型缓存缺失 / search503 | 执行prepare_embedding_model.py --allow-download |
| ask502、search正常 | 检查OpenAI密钥、模型权限与服务器日志 |
| 浏览器CORS失败 | 实际origin的协议、域名、端口以及服务重启 |
| 车辆无分析或无SHAP | 查业务表记录；RAG不生成预测和SHAP |

官方快照核验日期2026-10-08，政策审核期限2027-01-06；维护指导2027-10-08。期限是项目审核安排，不是法律有效期。后续资料更新通过人工检查后建库；官方快照不能称为实时法规。
