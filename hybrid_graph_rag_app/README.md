# Hybrid Graph RAG App

这是当前项目的新版主应用，用来演示 **Agentic RAG 可信知识问答系统**。旧版 `rag_web_app` 保留为历史版本；新版能力集中在 `hybrid_graph_rag_app` 中维护。

系统组合两类知识源：

- 文档库：复用现有 `vectorstore_rag`，优先 Chroma 语义检索，必要时降级到 SQLite FTS/BM25 风格关键词检索。
- 图谱库：复用独立 `neo4j_kg_db`，通过 Neo4j 查询实体关系证据。

## 核心能力

- Agent Workflow：显式拆分 Memory、Router、Query、Retriever、Evidence、Answer、Verifier、Reflection 等模块。
- Retriever：统一融合文档片段和图谱事实，生成可引用证据 ID，例如 `[S1]`、`[G1]`。
- Query Enhancer：对用户问题做轻量改写和多查询扩展，提高文档/图谱检索召回；HyDE 作为配置开关保留。
- Answer：只基于本轮检索证据回答，关键事实必须带证据编号。
- Verifier：校验答案是否引用了有效证据，并用 LLM/规则输出可信状态和置信度。
- Refusal：证据不足时返回拒答，避免直接依赖模型常识编造。
- UI：页面展示路由、回答、可信校验、引用来源、文档证据和图谱证据。

## 目录

- `agents/`: 显式 Agent 工作流模块。
- `app.py`: 独立 FastAPI 入口。
- `hybrid_service.py`: 对外服务 façade，初始化依赖并委托 AgentWorkflow。
- `schemas.py`: 路由、证据、校验、最终响应的数据结构。
- `query_enhancer.py`: 查询改写、多查询扩展和可选 HyDE 检索增强。
- `graph_retriever.py`: Neo4j 图谱检索。
- `vector_retriever.py`: 文档语义检索和 FTS 后备检索。
- `neo4j_runtime.py`: 按需拉起独立 Neo4j 实例。
- `templates/chat.html`: 聊天和证据链展示页面。
- `data/conversation_history.json`: 本目录自己的会话历史。

## 启动

复制环境变量示例并填入自己的模型配置：

```powershell
Copy-Item config/.env.example config/.env
```

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

## 健康检查

```powershell
curl http://127.0.0.1:8010/api/health
```

返回中会包含：

- `vector_backend`
- `graph_endpoint`
- `routing`
- `verifier`
- `response_schema`
- `query_expansion`
- `hyde`

## API 输出

`POST /api/chat` 会返回可信问答结构：

- `answer`: 最终回答或拒答。
- `status`: `verified`、`uncertain` 或 `refused`。
- `confidence`: 置信度。
- `route`: 路由模式、分数和原因。
- `verification`: 校验结果和原因。
- `sources`: 被引用或最关键的证据列表。
- `vector_results`: 文档检索原始结果。
- `graph_results`: 图谱检索原始结果。
- `memory_usage`: 本轮记忆使用决策。
- `used_memories`: 本轮召回的长期记忆。
- `memory_write_result`: 本轮长期记忆写入或跳过结果。

## 评估

项目提供轻量评估入口：

```powershell
python eval/eval_rag.py
```

测试问题位于：

```text
eval/test_questions.jsonl
```

运行后会生成：

```text
eval/eval_report.jsonl
eval/eval_summary.json
```

`eval_report.jsonl` 保存每个问题的详细结果；`eval_summary.json` 汇总路由分布、状态分布、平均置信度、平均延迟、平均证据数量和按问题类型统计的指标。后续如果依赖完整，可以在此基础上接入 RAGAS 指标：

```powershell
python eval/eval_ragas.py
```

如果没有安装 `ragas` 和 `datasets`，脚本会给出提示并退出，不影响主流程。

## Docker

项目提供基础 Dockerfile：

```powershell
docker build -t agentic-rag-self .
docker run --rm -p 8010:8010 --env-file config/.env agentic-rag-self
```

注意：Docker 镜像默认不会包含本地大模型、向量库和 Neo4j 数据目录；如需完整检索能力，需要额外挂载 `model/`、`vectorstore_rag/`、`neo4j_kg_db/` 或改用宿主机运行。

## 降级行为

- 如果 embedding 环境不可用，文档检索会尝试使用 FTS 后备模式。
- 如果外部大模型不可用，回答生成会退回到基于检索证据的保守模板。
- 如果 Verifier 模型不可用，会先使用证据引用规则做保守校验。
- 如果文档和图谱都没有证据，系统会拒答。

## 注意事项

- `config/.env` 可能包含真实密钥，不应提交或公开。
- Neo4j 启动依赖本地 Windows 路径；如果本机没有对应数据，可只验证文档检索和拒答流程。
- 对话历史只用于理解追问，不作为最终事实依据；最终答案必须引用本轮检索证据。
