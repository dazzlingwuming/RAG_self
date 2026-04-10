# Hybrid Graph RAG App

这是一个独立目录版的知识问答应用，不修改原来的 `rag_web_app`。

它组合了两类知识源：

- 文档库：复用现有 `vectorstore_rag`
- 图谱库：复用独立 `neo4j_kg_db`

## 目录

- `app.py`: 独立 FastAPI 入口
- `hybrid_service.py`: 混合检索与问答总流程
- `graph_retriever.py`: Neo4j 图谱检索
- `vector_retriever.py`: 文档检索
- `neo4j_runtime.py`: 按需拉起独立 Neo4j 实例
- `templates/chat.html`: 简单聊天页面
- `data/conversation_history.json`: 本目录自己的会话历史

## 启动

在项目根目录执行：

```powershell
D:\github\RAG_self\hybrid_graph_rag_app\start_hybrid_app.bat
```

或者直接执行：

```powershell
python -m uvicorn hybrid_graph_rag_app.app:app --host 0.0.0.0 --port 8010
```

打开：

```text
http://127.0.0.1:8010
```

## 当前运行方式

- 图谱检索：使用 `neo4j_kg_db`
- 文档检索：当前环境自动走 `FTS` 全文检索后备模式
- 如果后续本机 `sentence-transformers / numpy / scipy / scikit-learn` 环境恢复兼容，会自动切回 `Chroma` 语义检索

## 健康检查

```powershell
curl http://127.0.0.1:8010/api/health
```

返回中会包含：

- `vector_backend`
- `graph_endpoint`

## 说明

- 当前版本已经可以直接运行并提供“图谱 + 文档”混合问答
- 当外部大模型接口不可用时，应用会自动降级为基于检索结果的保守回答
- Neo4j 如果未启动，应用会自动尝试拉起 `neo4j_kg_db`
