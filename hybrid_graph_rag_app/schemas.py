from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


EvidenceType = Literal["document", "graph"]
ResponseStatus = Literal["verified", "uncertain", "refused"]
MemoryRiskLevel = Literal["low", "medium", "high"]
MemoryStatus = Literal["active", "outdated", "contradicted", "deleted", "low_confidence"]


@dataclass
class RouteDecision:
    mode: str
    graph_score: int
    doc_score: int
    summary: str
    strategies: list[str]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Evidence:
    evidence_id: str
    type: EvidenceType
    content: str
    source: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryUsageDecision:
    use_memory: bool
    memory_types: list[str]
    reason: str
    risk_level: MemoryRiskLevel = "low"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryRecord:
    memory_id: str
    user_id: str
    content: str
    category: str
    importance: float
    confidence: float
    status: MemoryStatus
    source: str
    source_turn_id: str | None
    evidence_type: str
    created_at: str
    updated_at: str
    last_accessed_at: str
    access_count: int
    ttl_days: int
    tags: list[str] = field(default_factory=list)
    contradicted_by: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryWriteResult:
    written: bool
    memory_id: str | None = None
    reason: str = ""
    skipped_reason: str | None = None
    contradicted_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnswerDraft:
    answer: str
    cited_sources: list[str]


@dataclass
class VerificationResult:
    supported: bool
    confidence: float
    status: ResponseStatus
    reason: str
    unsupported_claims: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FinalResponse:
    answer: str
    status: ResponseStatus
    confidence: float
    route: RouteDecision
    verification: VerificationResult
    sources: list[Evidence]
    vector_enabled: bool
    vector_backend: str
    vector_results: list[dict[str, Any]]
    graph_results: list[dict[str, Any]]
    memory_usage: MemoryUsageDecision | None = None
    used_memories: list[MemoryRecord] = field(default_factory=list)
    memory_write_result: MemoryWriteResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "status": self.status,
            "confidence": self.confidence,
            "route": self.route.to_dict(),
            "verification": self.verification.to_dict(),
            "sources": [source.to_dict() for source in self.sources],
            "vector_enabled": self.vector_enabled,
            "vector_backend": self.vector_backend,
            "vector_results": self.vector_results,
            "graph_results": self.graph_results,
            "memory_usage": self.memory_usage.to_dict() if self.memory_usage else None,
            "used_memories": [memory.to_dict() for memory in self.used_memories],
            "memory_write_result": self.memory_write_result.to_dict() if self.memory_write_result else None,
        }
