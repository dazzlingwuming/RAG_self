from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from hybrid_graph_rag_app import settings
from hybrid_graph_rag_app.graph_retriever import GraphRetriever
from hybrid_graph_rag_app.history_store import history_to_text, load_history, save_turn
from hybrid_graph_rag_app.llm_support import get_embedding_model, get_llm
from hybrid_graph_rag_app.neo4j_runtime import Neo4jRuntime
from hybrid_graph_rag_app.vector_retriever import VectorRetriever


class HybridGraphRAGService:
    GRAPH_HINTS = (
        "关系",
        "简称",
        "别名",
        "外文名",
        "英文名",
        "属于",
        "位于",
        "谁",
        "哪里",
        "时间",
        "作者",
        "朝代",
        "国籍",
        "出生",
        "死亡",
    )
    DOC_HINTS = (
        "讲了什么",
        "内容",
        "原文",
        "全文",
        "总结",
        "概述",
        "解释",
        "怎么说",
        "如何",
        "为什么",
    )

    def __init__(self) -> None:
        self.llm = get_llm()
        self.embedding_model = get_embedding_model()
        self.vector_retriever = VectorRetriever(self.embedding_model)
        self.vector_enabled = self.vector_retriever.backend != "disabled"
        self.neo4j_runtime = Neo4jRuntime()
        self.graph_retriever = GraphRetriever(self.neo4j_runtime)
        self.prompt = PromptTemplate.from_template(
            """你是一个中文知识问答助手。
请基于下面两类上下文回答问题：
1. 文档知识库片段
2. Neo4j 图谱事实

如果上下文不足以支撑结论，明确说明“知识库和图谱中没有足够依据”，不要编造。

问题路由建议：
{route_summary}

对话历史：
{history}

文档知识库：
{vector_context}

图谱事实：
{graph_context}

用户问题：
{query}

请按这个格式作答：
1. 核心回答：直接回答问题
2. 依据说明：说明主要依据来自文档片段还是图谱事实
3. 关键证据：列出最关键的 2 到 4 条证据"""
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def route_query(self, query: str) -> dict:
        graph_score = 0
        doc_score = 0

        for hint in self.GRAPH_HINTS:
            if hint in query:
                graph_score += 2
        for hint in self.DOC_HINTS:
            if hint in query:
                doc_score += 2

        if len(query) <= 10:
            graph_score += 1
        if len(query) >= 14:
            doc_score += 1

        if graph_score >= doc_score + 2:
            mode = "graph_first"
        elif doc_score >= graph_score + 2:
            mode = "document_first"
        else:
            mode = "hybrid"

        return {
            "mode": mode,
            "graph_score": graph_score,
            "doc_score": doc_score,
            "summary": f"mode={mode}, graph_score={graph_score}, doc_score={doc_score}",
        }

    @staticmethod
    def _graph_priority(graph_results: list[dict]) -> int:
        if not graph_results:
            return 0
        return max(int(item.get("score", 0) or 0) for item in graph_results[:4])

    @staticmethod
    def _fallback_answer(query: str, route: dict, graph_results: list[dict], vector_results: list[dict]) -> str:
        evidence: list[str] = []
        graph_score = HybridGraphRAGService._graph_priority(graph_results)

        if route["mode"] == "graph_first":
            use_graph = bool(graph_results)
            use_vector = bool(vector_results[:1]) and graph_score < 100
        elif route["mode"] == "document_first":
            use_graph = bool(graph_results) and graph_score >= 100
            use_vector = bool(vector_results)
        else:
            use_graph = graph_score >= 60
            use_vector = bool(vector_results)

        if use_graph:
            for item in graph_results[:4]:
                evidence.append(
                    f"- 图谱：{item.get('source_name', '未知实体')} -[{item.get('rel_type', '相关')}]- {item.get('target_name', '未知值')}"
                )

        if use_vector:
            for item in vector_results[:2]:
                snippet = item.get("content", "")[:160].replace("\n", " ")
                source = item.get("metadata", {}).get("source", "unknown")
                evidence.append(f"- 文档：source={source} {snippet}")

        if use_graph and graph_results:
            first = graph_results[0]
            core = (
                f"根据图谱检索，较相关的结果是："
                f"{first.get('source_name', '未知实体')} -[{first.get('rel_type', '相关')}]- {first.get('target_name', '未知值')}。"
            )
            basis = f"当前回答主要依据 Neo4j 图谱结果生成，路由模式为 {route['mode']}。"
        elif use_vector and vector_results:
            first = vector_results[0]
            snippet = first.get("content", "")[:180].replace("\n", " ")
            core = f"根据文档检索，当前最相关的片段是：{snippet}"
            basis = (
                f"当前回答主要依据文档知识库检索结果生成，"
                f"检索模式为 {first.get('retrieval_mode', 'unknown')}，路由模式为 {route['mode']}。"
            )
        else:
            core = f"针对问题“{query}”，当前没有检索到足够的图谱或文档依据。"
            basis = "当前环境下未获得可用的外部大模型响应，因此只能返回基于检索结果的保守结论。"

        if not evidence:
            evidence.append("- 未检索到足够证据")

        return "\n".join(
            [
                f"1. 核心回答：{core}",
                f"2. 依据说明：{basis}",
                "3. 关键证据：",
                *evidence,
            ]
        )

    def ask(self, query: str, session_id: str) -> dict:
        history = load_history(settings.HISTORY_PATH, session_id=session_id, turns=8)
        route = self.route_query(query)

        try:
            vector_results = self.vector_retriever.search(query)
        except Exception:
            vector_results = []
        try:
            graph_results = self.graph_retriever.search(query)
        except Exception:
            graph_results = []

        if route["mode"] == "graph_first":
            graph_results = graph_results[:8]
            vector_results = vector_results[:3]
        elif route["mode"] == "document_first":
            vector_results = vector_results[:8]
            graph_results = graph_results[:4]
        else:
            vector_results = vector_results[:6]
            graph_results = graph_results[:6]

        try:
            answer = self.chain.invoke(
                {
                    "route_summary": route["summary"],
                    "history": history_to_text(history),
                    "vector_context": self.vector_retriever.format_for_prompt(vector_results),
                    "graph_context": self.graph_retriever.format_for_prompt(graph_results),
                    "query": query,
                }
            )
        except Exception:
            answer = self._fallback_answer(query, route, graph_results, vector_results)

        save_turn(settings.HISTORY_PATH, session_id=session_id, query=query, answer=answer)
        return {
            "answer": answer,
            "route": route,
            "vector_enabled": self.vector_enabled,
            "vector_backend": self.vector_retriever.backend,
            "vector_results": vector_results,
            "graph_results": graph_results,
        }
