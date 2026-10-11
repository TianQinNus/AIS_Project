# FleetCare AI — Group 47

FleetCare 是车队维护决策支持原型：XGBoost 预测 + 时序规则推理 + SHAP 解释 + 官方知识 RAG。此 model 分支包含模型工作和 RAG 模块；网页团队可通过 Python 或 HTTP 入口接入。RAG 不执行真实维修或代替机械故障诊断。

## 当前 RAG 进度（2026-10-11）

- 官方原文快照：DVSA、HSE、LTA 检验、LTA 商用车，另有明确标注的项目规则说明。
- 本地 SQLite 向量库：5 份有效资料、273 个片段、384 维向量；替代旧 index.npz 查询。
- 原文出处、地区、核验日期、审核期限和 SHA-256 可追溯；过期源不参与检索。
- Agent 汇总 MySQL 中已有车辆分析与知识库证据，输出正文、车辆结果、引用及警告。
- 新增 rag_http_api.py：综合问答与严格分类的知识检索两个 HTTP 入口。
- 已提供只含有效版本的 knowledge_seed/fleetcare_knowledge.sqlite3；模型权重另行准备。

## 文档入口

- **网页团队接入必读：[README_RAG_WEB.md](SystemCode/data/README_RAG_WEB.md)**：安装、HTTP 请求/响应、JavaScript 示例、maintenance 范围与异常处理。
- [RAG 安装与知识维护](SystemCode/data/README_RAG.md)
- [原文目录](SystemCode/data/RAG_KNOWLEDGE_MANUAL.md)
- [最新 RAG 进度与交接](SystemCode/data/RAG_PROGRESS.md)
- [模型开发与评估记录](SystemCode/data/目前进展.txt)：原有模型记录保留，RAG 更新在文件顶部。

## 网页团队快速启动

```bash
python -m pip install -r SystemCode/requirements-rag-web.txt
cd SystemCode/data
python build_knowledge.py --allow-model-download
python evaluate_knowledge.py
# 根据 .env.rag.example export 后端实际配置
python -m uvicorn rag_http_api:app --host 127.0.0.1 --port 8000
```

POST /api/rag/search 仅检索本地证据，不需要 OpenAI/MySQL；POST /api/rag/ask 生成自然语言回答，车辆问题读取 MySQL。OpenAI 与数据库凭据仅保存在服务端。Swagger 位于本地 /docs；尚未部署公网地址。

## maintenance 知识的定位

提供日常检查、制动/轮胎/警告灯/泄漏检查参考、缺陷报告、计划维护与操作安全指导。新加坡检验频率属于 policy，TR/Risk/SHAP 定义属于 project，具体车辆优先级来自 MySQL。没有车型 OEM 手册时，不能确认具体故障、零件、扭矩或拆装工序。

## 验证与提交材料

32 项单元测试、13 条真实向量检索用例通过；检索已验证可在无网络、无原文目录条件下工作。HTTP 联调不等于真实 OpenAI/MySQL 端到端验证，后者由团队配置实际凭据后完成。

最终提交包含可运行系统、项目报告、5 分钟技术视频与 5 分钟宣传视频。视频链接和成员贡献信息由团队在完成后填写，当前没有发布链接。
