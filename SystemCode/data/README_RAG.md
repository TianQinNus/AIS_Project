# FleetCare RAG v3 使用与迁移

运行存储现已统一为 PostgreSQL + pgvector。完整安装、网页契约、配置、建库、维护、实验结果和问题范围请阅读 [README_RAG_WEB.md](README_RAG_WEB.md)。

快速准备：安装 requirements-rag-web.txt，配置 PostgreSQL 环境变量并由管理员准备 pgvector，然后在 data 目录执行 python migrate_knowledge.py 导入 v3 种子。服务器还需准备本地多语言模型缓存。也可 python build_knowledge.py --allow-model-download 从原文建库并准备模型。

知识查询：python knowledge_rag.py '轮胎应检查什么？' --kind maintenance。
网页接口：python -m uvicorn rag_http_api:app --host 127.0.0.1 --port 8000。
车辆查询和向量检索共用 PostgreSQL 连接，端口默认为 5432；旧 MySQL、SQLite/NPZ 路径配置不再生效。

当前 5 份文档、274 片段、384 维。maintenance 为通用检查参考；policy 为新加坡政策；project 为项目规则；具体车型维修步骤需要 OEM 手册。

网页成员首次操作指南：[WEB_RAG_DEPLOYMENT.md](WEB_RAG_DEPLOYMENT.md)。准备模型可单独运行 prepare_embedding_model.py --allow-download，无需重建知识库。只读自检用 check_rag_setup.py。
