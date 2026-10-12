# FleetCare AI — Group 47

模型预测 + 时序规则 + SHAP + 官方知识 RAG。model 分支最新实现已统一 PostgreSQL + pgvector。

- **[网页同学逐步操作指南](SystemCode/data/WEB_RAG_DEPLOYMENT.md)**
- [网页团队接入与迁移 README](SystemCode/data/README_RAG_WEB.md)
- [验证记录](SystemCode/data/RAG_VALIDATION.md)
- [RAG 最新进度](SystemCode/data/RAG_PROGRESS.md)
- [原文目录](SystemCode/data/RAG_KNOWLEDGE_MANUAL.md)
- [模型开发记录](SystemCode/data/目前进展.txt)

当前有效知识：4 份政府官方原文 +1 份项目说明，274 个片段，384 维。车辆事实来自 public.analysis / prediction_shap；知识保存在同库 rag schema。网页接口 /api/rag/ask、/api/rag/search 保持兼容。

安装 SystemCode/requirements-rag-web.txt 后，配置项目 PostgreSQL（默认5432）并准备 pgvector。在 SystemCode/data 中执行 python migrate_knowledge.py 导入当前向量种子，或 python build_knowledge.py --allow-model-download 建库并准备嵌入模型。启动 python -m uvicorn rag_http_api:app --host 127.0.0.1 --port 8000。完整步骤见接入 README。

41 项测试和13条检索用例在隔离 PostgreSQL/PGlite 测试引擎验证；团队 PostgreSQL16与真实OpenAI综合问答仍需联调。测试 URL 必须指向独立可清空的数据库。

maintenance 是通用检查参考，不是具体车型维修手册。官方政策和项目规则分开标注；SHAP 不构成机械故障诊断。检索在配置的 PostgreSQL 中进行，LLM 生成仍调用 OpenAI。
