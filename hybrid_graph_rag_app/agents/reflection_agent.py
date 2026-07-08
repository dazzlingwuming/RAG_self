from hybrid_graph_rag_app import settings
from hybrid_graph_rag_app.agents.state import ReflectionDecision
from hybrid_graph_rag_app.schemas import VerificationResult


class ReflectionAgent:
    def decide(self, verification: VerificationResult, retry_count: int) -> ReflectionDecision:
        if verification.status in {"verified", "refused"}:
            return ReflectionDecision(action="finish", reason="校验已经给出终态。")
        if verification.status == "uncertain" and retry_count < settings.REFLECTION_MAX_RETRIES:
            return ReflectionDecision(action="repair", reason="校验不确定，尝试一次保守修订。")
        if verification.status == "uncertain" and verification.confidence < settings.MIN_VERIFICATION_CONFIDENCE:
            return ReflectionDecision(action="refuse", reason="修订后仍低于最小可信阈值，进入拒答。")
        return ReflectionDecision(action="finish", reason="校验置信度达到保守返回条件。")
