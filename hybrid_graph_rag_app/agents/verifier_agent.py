from hybrid_graph_rag_app.schemas import Evidence, VerificationResult
from hybrid_graph_rag_app.verifier import AnswerVerifier


class VerifierAgent:
    def __init__(self, verifier: AnswerVerifier) -> None:
        self.verifier = verifier

    def verify(self, query: str, answer: str, evidence: list[Evidence]) -> VerificationResult:
        return self.verifier.verify(query=query, answer=answer, evidence=evidence)
