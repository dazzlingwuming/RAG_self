# Hybrid Graph RAG App 项目说明

## 1. 项目定位

`hybrid_graph_rag_app` 是一个独立于原项目的知识问答子应用。

目标是把两类知识源组合起来：

- 文档知识库
- Neo4j 知识图谱

这个目录下的代码不依赖你去继续修改原来的 `rag_web_app` 主流程，方便单独开发、单独运行、单独验证。

## 2. 当前实现了什么

当前已经完成：

- 独立 FastAPI 服务
- 独立前端页面
- Neo4j 图谱检索
- 文档检索
- 简单问题路由
- 会话历史保存
- 图谱服务自动拉起
- 大模型不可用时的降级回答

## 3. 目录结构

- `app.py`
  FastAPI 入口，提供页面、聊天接口、健康检查接口。

- `hybrid_service.py`
  核心服务层。负责：
  - 问题路由
  - 调用图谱检索
  - 调用文档检索
  - 组织上下文
  - 调用 LLM
  - 失败时降级返回

- `graph_retriever.py`
  Neo4j 图谱检索逻辑。通过 HTTP 接口查询本地独立 Neo4j 实例。

- `vector_retriever.py`
  文档检索逻辑。
  当前支持两种模式：
  - `semantic`
  - `fts`

- `neo4j_runtime.py`
  检查 Neo4j 是否在线；如果没启动，尝试自动拉起。

- `history_store.py`
  负责读写会话历史。

- `llm_support.py`
  负责初始化 LLM 和 embedding 模型。

- `settings.py`
  统一配置目录、模型目录、向量库目录、Neo4j 目录等。

- `templates/chat.html`
  前端页面。

- `data/conversation_history.json`
  当前目录自己的历史对话文件。

## 4. 检索策略

这个项目当前不是“只选一个检索器”，而是混合检索。

### 4.1 图谱检索

图谱检索使用：

- Neo4j

适合的问题类型：

- 外文名
- 简称
- 别名
- 作者
- 时间
- 地点
- 属性关系
- 实体之间的关联

例如：

- `夏朝的外文名是什么`
- `复旦大学的简称是什么`

### 4.2 文档检索

文档检索当前优先级如下：

1. 如果 embedding 环境正常，走 `semantic`
2. 如果 embedding 环境异常，自动降级到 `fts`

当前这台机器上，由于 `sentence-transformers / numpy / scipy / sklearn` 存在兼容问题，所以运行时实际走的是：

- `fts`

也就是说，当前文档检索是从 `vectorstore_rag/chroma.sqlite3` 中做全文检索。

适合的问题类型：

- 讲了什么
- 内容是什么
- 原文是什么
- 总结一下
- 如何解释

例如：

- `孙子曰讲的是什么`
- `民法典是什么`

### 4.3 问题路由

当前在 `hybrid_service.py` 中实现了三种路由模式：

- `graph_first`
- `document_first`
- `hybrid`

含义：

- `graph_first`
  优先采用图谱证据

- `document_first`
  优先采用文档证据

- `hybrid`
  两边都查，再根据结果选择主依据

## 5. 当前运行方式

### 5.1 启动应用

在项目根目录执行：

```powershell
D:\github\RAG_self\hybrid_graph_rag_app\start_hybrid_app.bat
```

或者：

```powershell
python -m uvicorn hybrid_graph_rag_app.app:app --host 0.0.0.0 --port 8010
```

### 5.2 访问页面

```text
http://127.0.0.1:8010
```

### 5.3 健康检查

```text
GET /api/health
```

返回内容里会包含：

- `status`
- `app`
- `vector_backend`
- `graph_endpoint`
- `routing`

## 6. 当前依赖状态

当前应用已经可以直接运行，但要明确区分两层能力：

### 6.1 已稳定可用

- FastAPI 页面
- Neo4j 图谱检索
- FTS 文档检索
- 问题路由
- 检索失败时降级回答


## 7. 回答生成逻辑

回答生成分两层：

1. 尝试调用外部大模型
2. 如果大模型不可用，则根据检索结果返回保守答案

因此系统在外部接口不稳定时，仍然能回答，但回答会更偏：

- 图谱关系直接返回
- 文档片段摘要式返回
