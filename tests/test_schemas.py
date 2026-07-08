from hybrid_graph_rag_app.schemas import Evidence, FinalResponse, RouteDecision, VerificationResult
from hybrid_graph_rag_app.verifier import format_evidence


def test_evidence_formatting():
    evidence = [Evidence(evidence_id="S1", type="document", content="测试内容", source="sample.txt")]
    text = format_evidence(evidence)
    assert "[S1]" in text
    assert "测试内容" in text
    assert "sample.txt" in text


def test_final_response_to_dict():
    route = RouteDecision(
        mode="hybrid",
        graph_score=1,
        doc_score=1,
        summary="mode=hybrid",
        strategies=["document", "graph"],
        reason="test",
    )
    verification = VerificationResult(
        supported=True,
        confidence=0.8,
        status="verified",
        reason="ok",
    )
    evidence = [Evidence(evidence_id="G1", type="graph", content="A -[关系]- B", source="Neo4j:A")]
    response = FinalResponse(
        answer="核心回答：A 与 B 有关系 [G1]",
        status="verified",
        confidence=0.8,
        route=route,
        verification=verification,
        sources=evidence,
        vector_enabled=True,
        vector_backend="semantic",
        vector_results=[],
        graph_results=[],
    ).to_dict()
    assert response["status"] == "verified"
    assert response["route"]["mode"] == "hybrid"
    assert response["sources"][0]["evidence_id"] == "G1"
