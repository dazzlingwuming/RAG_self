from hybrid_graph_rag_app import settings
from hybrid_graph_rag_app.graph_retriever import GraphRetriever
from hybrid_graph_rag_app.schemas import Evidence, RouteDecision
from hybrid_graph_rag_app.vector_retriever import VectorRetriever


class EvidenceAgent:
    def __init__(self, vector_retriever: VectorRetriever, graph_retriever: GraphRetriever) -> None:
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever

    def prepare(
        self,
        vector_results: list[dict],
        graph_results: list[dict],
        route: RouteDecision,
    ) -> tuple[list[dict], list[dict], list[Evidence]]:
        vector_results = self.dedupe_vector_results(vector_results)
        graph_results = self.dedupe_graph_results(graph_results)

        if route.mode == "graph_first":
            graph_results = graph_results[:8]
            vector_results = vector_results[:3]
        elif route.mode == "document_first":
            vector_results = vector_results[:8]
            graph_results = graph_results[:4]
        else:
            vector_results = vector_results[:6]
            graph_results = graph_results[:6]

        document_evidence = self.vector_retriever.to_evidence(vector_results)
        graph_evidence = self.graph_retriever.to_evidence(graph_results)
        evidence = self.fuse_evidence(document_evidence, graph_evidence, route)
        return vector_results, graph_results, evidence

    @staticmethod
    def dedupe_vector_results(results: list[dict]) -> list[dict]:
        unique: list[dict] = []
        seen: set[str] = set()
        for item in results:
            meta = item.get("metadata", {}) or {}
            key = f"{meta.get('source', 'unknown')}::{meta.get('chunk_id', '')}::{item.get('content', '')[:120]}"
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique

    @staticmethod
    def dedupe_graph_results(results: list[dict]) -> list[dict]:
        unique: list[dict] = []
        seen: set[str] = set()
        for item in results:
            key = f"{item.get('source_name')}::{item.get('rel_type')}::{item.get('target_name')}"
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        unique.sort(key=lambda item: item.get("score", 0) or 0, reverse=True)
        return unique

    @staticmethod
    def fuse_evidence(document_evidence: list[Evidence], graph_evidence: list[Evidence], route: RouteDecision) -> list[Evidence]:
        ordered = graph_evidence + document_evidence if route.mode == "graph_first" else document_evidence + graph_evidence
        if route.mode == "hybrid":
            ordered = []
            for idx in range(max(len(document_evidence), len(graph_evidence))):
                if idx < len(document_evidence):
                    ordered.append(document_evidence[idx])
                if idx < len(graph_evidence):
                    ordered.append(graph_evidence[idx])

        fused: list[Evidence] = []
        seen: set[tuple[str, str]] = set()
        for item in ordered:
            key = (item.type, item.source + item.content[:80])
            if key in seen:
                continue
            seen.add(key)
            fused.append(item)
        return fused[: settings.FINAL_EVIDENCE_LIMIT]

    @staticmethod
    def format_evidence_context(evidence: list[Evidence]) -> str:
        if not evidence:
            return "未检索到可用证据。"
        blocks = []
        for item in evidence:
            score_text = "" if item.score is None else f" score={item.score}"
            blocks.append(f"[{item.evidence_id}] {item.type} source={item.source}{score_text}\n{item.content}")
        return "\n\n".join(blocks)
