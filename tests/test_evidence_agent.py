from hybrid_graph_rag_app.agents.evidence_agent import EvidenceAgent
from hybrid_graph_rag_app.schemas import Evidence, RouteDecision


class DummyVectorRetriever:
    def to_evidence(self, results):
        return [Evidence(evidence_id=f"S{idx + 1}", type="document", content=item["content"], source=item["metadata"]["source"]) for idx, item in enumerate(results)]


class DummyGraphRetriever:
    def to_evidence(self, results):
        return [Evidence(evidence_id=f"G{idx + 1}", type="graph", content=f"{item['source_name']} -[{item['rel_type']}]- {item['target_name']}", source=f"Neo4j:{item['source_name']}") for idx, item in enumerate(results)]


def route(mode: str) -> RouteDecision:
    return RouteDecision(mode=mode, graph_score=0, doc_score=0, summary=mode, strategies=["document", "graph"], reason="test")


def test_evidence_agent_hybrid_interleaves_evidence():
    agent = EvidenceAgent(DummyVectorRetriever(), DummyGraphRetriever())
    vector_results = [{"content": "文档1", "metadata": {"source": "a.txt", "chunk_id": "1"}}]
    graph_results = [{"source_name": "A", "rel_type": "关系", "target_name": "B", "score": 2}]

    _, _, evidence = agent.prepare(vector_results, graph_results, route("hybrid"))

    assert [item.evidence_id for item in evidence] == ["S1", "G1"]


def test_evidence_agent_graph_first_orders_graph_before_document():
    agent = EvidenceAgent(DummyVectorRetriever(), DummyGraphRetriever())
    vector_results = [{"content": "文档1", "metadata": {"source": "a.txt", "chunk_id": "1"}}]
    graph_results = [{"source_name": "A", "rel_type": "关系", "target_name": "B", "score": 2}]

    _, _, evidence = agent.prepare(vector_results, graph_results, route("graph_first"))

    assert [item.evidence_id for item in evidence] == ["G1", "S1"]


def test_evidence_agent_dedupes_vector_results():
    agent = EvidenceAgent(DummyVectorRetriever(), DummyGraphRetriever())
    vector_results = [
        {"content": "重复文档", "metadata": {"source": "a.txt", "chunk_id": "1"}},
        {"content": "重复文档", "metadata": {"source": "a.txt", "chunk_id": "1"}},
    ]

    vector_results, _, evidence = agent.prepare(vector_results, [], route("document_first"))

    assert len(vector_results) == 1
    assert len(evidence) == 1
