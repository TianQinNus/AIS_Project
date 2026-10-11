# FleetCare RAG v2 — 网页团队接入 README

更新：2026-10-11。此文档替代旧 NPZ RAG 接入说明。核心接口保持兼容，新增可启动的 HTTP 适配器 `rag_http_api.py`。代码在 model 分支；提供本地运行入口，未部署公网服务。

## 1. 当前能力与维护知识范围

系统将 MySQL 的车辆模型/规则/SHAP 结果与本地 SQLite 的原文证据汇总为回答。当前有效库：4 份官方原文 + 1 份项目规则说明，273 个片段，384 维向量；快照核验日期 2026-10-08。

| 分类 | 资料 | 应回答的问题 |
|---|---|---|
| maintenance | DVSA HGV daily checks、HSE inspection/maintenance/repair | 通用日常检查、缺陷记录和报告、计划检查维护、安全注意事项 |
| policy | LTA Inspection、LTA Commercial Vehicles | 新加坡定期检验及商用车政策，需核实车型/车龄/重量等适用条件 |
| project | fleetcare_policy.md | Risk6、TR0–TR6、SHAP、prediction_at 和匿名特征的项目解释 |
| 车辆事实 | MySQL analysis / prediction_shap | 已保存的车辆预测、优先级、行动、规则依据和 SHAP |

maintenance 可以回答的示例：

- 出车前应检查哪些警告灯、制动、轮胎、灯光或泄漏？
- 制动气压建立异常、制动管路漏气时，官方指导列出了哪些检查事项？
- 轮胎或车轮固定件有哪些日常检查内容？
- 司机发现缺陷后需要记录、报告哪些信息？
- 日常检查与计划维护如何安排，维护操作有哪些安全注意事项？
- 车辆被模型标为 INSPECT：为什么被标记，以及有哪些通用检查参考？最后一个属于组合问答，须提供实际 vehicle_id。

maintenance 当前不能确认具体车型零件故障、Component X 的真实身份、维修扭矩、零件号、油液型号、拆装步骤或 OEM 更换周期。没有对应原文时应说明证据不足。SHAP 是预测贡献，不是机械诊断。新加坡“货车多久检验”应归 policy，TR1 的含义应归 project，某辆车为什么 INSPECT 先依据 MySQL。

界面建议：页面使用“维护知识 / 检查参考”，避免“故障确诊 / 自动维修方案”。政策单独展示“政策参考”。官方原文可追溯，但不意味着专家认证或政策实时更新；UK 指导不作为 Singapore 法规。

## 2. 网页端架构与安装

浏览器 → HTTP 后端 → FleetCare Agent → MySQL + SQLite → LLM → JSON。

Python 3.10+。在仓库根目录：

```bash
python -m pip install -r SystemCode/requirements-rag-web.txt
cd SystemCode/data
```

已提供可分发种子库 `knowledge_seed/fleetcare_knowledge.sqlite3`，只含当前有效版本。工作库是 `knowledge_index/fleetcare_knowledge.sqlite3`，被 gitignore 忽略；种子库会随 Git 上传。嵌入模型权重不包含在仓库中。

第一次建立/准备模型，允许在准备阶段下载模型：

```bash
python build_knowledge.py --allow-model-download
python evaluate_knowledge.py
```

已缓存模型时只需 `python build_knowledge.py`。如模型已缓存但不想重算向量，可以在首次运行、工作库不存在时复制种子库：

```bash
python - <<'PYINIT'
from pathlib import Path
import shutil
working = Path('knowledge_index/fleetcare_knowledge.sqlite3')
if not working.exists():
    working.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2('knowledge_seed/fleetcare_knowledge.sqlite3', working)
PYINIT
```

查询固定读取本地缓存模型，不自动下载。复制数据库不等于安装模型，另一台机器还需准备权重。`FLEETCARE_KNOWLEDGE_DB` 可设置自定义 SQLite 绝对路径；旧 `FLEETCARE_KNOWLEDGE_INDEX` 指向 NPZ 时请 unset。

服务端环境变量（`.env.rag.example` 是示例，代码不会自动加载 .env）：

```bash
export OPENAI_API_KEY='填写实际密钥'
export FLEETCARE_LLM_MODEL='填写账户有权限且支持Responses工具调用的模型名'
export FLEETCARE_DB_HOST='127.0.0.1'
export FLEETCARE_DB_PORT='3306'
export FLEETCARE_DB_USER='填写数据库用户'
export FLEETCARE_DB_PASSWORD='填写数据库密码'
export FLEETCARE_DB_NAME='fleetcareai'
export FLEETCARE_ALLOWED_ORIGINS='http://localhost:3000,http://localhost:5173'
python -m uvicorn rag_http_api:app --host 127.0.0.1 --port 8000
```

应按账户实际可用模型设置模型名，不要假定示例默认模型可用。已有 MySQL 表及分析记录由模型团队准备，RAG 接口不创建业务表或运行模型推理。

Swagger: http://127.0.0.1:8000/docs。跨机器联调时将服务绑定团队网络地址并把实际 URL 配置给前端；CORS 中配置准确的协议、域名、端口。密钥和 MySQL 凭据只在服务器，不能放前端。适配器用于本地/受控团队联调，尚无用户认证和限流；公网部署应接入现有受保护的后端。参考 [FastAPI CORS 官方说明](https://fastapi.tiangolo.com/tutorial/cors/)。

## 3. 接口契约

### POST /api/rag/ask — 综合回答

```json
{"question":"这辆车为什么需要检查？请结合官方文档提供检查参考。","vehicle_id":123}
```

123 只是格式示例，必须替换为 analysis 表中存在的整数。`vehicle_id` 可省略或为 null；纯知识问题无需绑定车辆。`question` 为去除首尾空白后非空字符串，原始长度 1–6000；vehicle_id 不接受字符串、浮点或布尔值。额外字段被拒绝。接口不接受 `kind`；此入口由 Agent 汇总检索的多类证据。

返回以下五个字段，不增加伪造的顶层 success：

| 字段 | 类型 | 前端用途 |
|---|---|---|
| answer | string | 回答正文，含 [K-xxxxxxxxxxxx] / [DB:analysis_id] 引用 |
| vehicle_analysis | array | 查询过的车辆结果；先检查每项 success，再读取 current、history、shap |
| sources | array | 来源卡片/引用详情：citation_id、text、title、publisher、source_url、jurisdiction、verified_on、review_due、kind、cited_in_answer、similarity 等 |
| retrieval_warnings | array[string] | 警告条，说明无相关证据、知识不可用或数据库查询失败等 |
| retrieval_trace | array | 调试用检索轨迹，普通页面可折叠 |

完整车辆响应由 `rag.py:get_vehicle_analysis` 定义；成功项含 `vehicle_id`、`current.analysis_id/state/action/priority/rule_id/reason` 等。查询失败项没有 current，应显示 error。空数组或成功 HTTP 响应不等于所有证据都成功。

引用卡片优先展示 `cited_in_answer=true` 的 sources；未被回答引用的检索片段可另放“参考检索结果”。`similarity` 是相关性，不是故障概率或事实置信度。车辆风险分数也不是已校准维修概率。正文建议作为文本或经过安全处理的 Markdown 渲染。

### POST /api/rag/search — 严格分类的本地证据检索

```json
{"question":"制动管路漏气有哪些检查事项？","kind":"maintenance"}
```

`kind` 可为 maintenance / policy / project / dataset，也可省略；当前没有有效 dataset 文档，因此该筛选可返回空结果。本入口只返回数据库原文命中，不调用 OpenAI 或 MySQL、不生成自然语言回答。仅需本地模型和 SQLite。

响应含 success、query、hits、database_version、retrieval_backend=local_sqlite、policy_jurisdiction=SG、excluded_overdue_documents、maintenance_evidence_found、note。`hits=[]` 表示未找到符合阈值/类别/地区/审核期限的证据，应显示“暂无相关知识证据”，不要补造答案。

### 状态与异常

- 200：正常返回；仍检查 retrieval_warnings、车辆项 success 和空 hits。
- 422：请求参数不合法。
- ask 502：回答生成或服务调用异常，返回去除凭据的固定错误信息。
- search 503：本地数据库/模型不可用。
- GET /health 返回 alive，只表示进程存活，不证明 MySQL、OpenAI 或知识库可用。

同步接口在线程池执行，单进程用锁串行访问共享模型/检索器；当前为非流式、单轮问答，无聊天历史参数。UI 提供 loading、防重复点击和错误提示；部署网关超时要覆盖可能的多轮工具调用。增加 worker 会各自加载模型并占内存。

## 4. 网页调用示例

```javascript
const RAG_BASE_URL = 'http://127.0.0.1:8000'; // 改为团队实际后端地址

export async function askFleetCare(question, vehicleId = null) {
  const response = await fetch(`${RAG_BASE_URL}/api/rag/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, vehicle_id: vehicleId }),
  });
  const result = await response.json();
  if (!response.ok) {
    const detail = typeof result.detail === 'string'
      ? result.detail : JSON.stringify(result.detail);
    throw new Error(detail || `HTTP ${response.status}`);
  }
  return result;
}

// 无车辆绑定的知识问答
const result = await askFleetCare('出车前轮胎有哪些检查事项？');
console.log(result.answer, result.sources, result.retrieval_warnings);

// 维护分类证据（不生成回答，不调用 OpenAI）
const searchResponse = await fetch(`${RAG_BASE_URL}/api/rag/search`, {
  method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ question: '轮胎与车轮固定件检查', kind: 'maintenance' }),
});
if (!searchResponse.ok) throw new Error(`HTTP ${searchResponse.status}`);
const evidence = await searchResponse.json();
console.log(evidence.hits);
```

curl 联调：

```bash
curl -X POST http://127.0.0.1:8000/api/rag/search   -H 'Content-Type: application/json'   -d '{"question":"轮胎有哪些检查事项？","kind":"maintenance"}'

curl -X POST http://127.0.0.1:8000/api/rag/ask   -H 'Content-Type: application/json'   -d '{"question":"新加坡货车有哪些定期检验要求？"}'
```

已有 Python 后端也可直接调用，无需另启适配服务：

```python
from fleetcare_agent import ask_fleetcare_detailed
result = ask_fleetcare_detailed(question, vehicle_id=vehicle_id)
# 将 result 作为 JSON 返回浏览器；导入路径需包含 SystemCode/data。
```

## 5. 数据、原文与版本维护

原文：knowledge/official/raw/*.html；确定性提取：knowledge/official/extracted/*.json；有效目录：knowledge/catalog.json；项目说明：knowledge/project/fleetcare_policy.md。旧摘要不在有效目录，不进入 v2 检索。种子库内同时保存原文 BLOB，可以离开原始目录检索。

查询不会读 catalog/外部文档，也不访问网页。只有维护命令才抓取官方页面：

```bash
python fetch_official_sources.py
# 人工核对 raw、extracted、catalog 变更
python build_knowledge.py
python evaluate_knowledge.py
```

维护指引每 365 天、政策每 90 天重新审核，查询自动排除过期源；这是项目审核周期，不是官方法律有效期。政策审核日期 2027-01-06、维护资料 2027-10-08。政策快照之后仍可能改变，应按实际变更主动更新。哈希检测修改，不等于真伪签名或专家认证。

SQLite 表：kb_versions / kb_documents / kb_chunks / kb_state。原文和向量按版本存储，事务发布有效版本，旧版本保留。Git 种子库仅发布当前版本；新的发布应重新生成种子库，避免复制凭据、业务记录或运行缓存。

## 6. 验证与当前限制

```bash
python -m unittest discover -s tests -v
python evaluate_knowledge.py
```

32 项单元测试（26 项核心 RAG + 6 项 HTTP 契约）和 13 条真实嵌入检索用例通过。原文目录缺失、网络阻断下检索已验证；源修改、数据库损坏、过期/地区过滤和版本切换有覆盖。HTTP 参数/返回透传/CORS/脱敏错误用模拟后端测试；本地真实 HTTP search 另外验证。不把这些结果称为 LLM 回答事实准确率。

尚未用团队实际 OpenAI/MySQL 凭据完成端到端综合回答联调。问答不重新计算车辆预测或法定到期日；不能把 prediction_at 当车辆采样时间或多次分析当真实风险变化。具体维修适用性需要车型身份、症状、故障码和对应 OEM 手册。

知识检索和嵌入完全本地，LLM 回答仍使用 OpenAI API，车辆结果与原文证据会作为模型上下文发送。MySQL 与 SQLite 是两个存储，由 Agent 汇总；不是物理合并为一个数据库。
