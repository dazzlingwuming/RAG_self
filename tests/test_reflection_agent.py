from hybrid_graph_rag_app.agents.reflection_agent import ReflectionAgent
from hybrid_graph_rag_app.schemas import VerificationResult


def test_reflection_finishes_verified_answer():
    verification = VerificationResult(supported=True, confidence=0.8, status="verified", reason="ok")
    decision = ReflectionAgent().decide(verification, retry_count=0)
    assert decision.action == "finish"


def test_reflection_repairs_uncertain_answer_once():
    verification = VerificationResult(supported=False, confidence=0.4, status="uncertain", reason="weak")
    decision = ReflectionAgent().decide(verification, retry_count=0)
    assert decision.action == "repair"


def test_reflection_refuses_after_retry_when_confidence_low():
    verification = VerificationResult(supported=False, confidence=0.2, status="uncertain", reason="weak")
    decision = ReflectionAgent().decide(verification, retry_count=1)
    assert decision.action == "refuse"
