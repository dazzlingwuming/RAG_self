from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from hybrid_graph_rag_app.memory_manager import MemoryContext
from hybrid_graph_rag_app.schemas import AnswerDraft, Evidence, FinalResponse, RouteDecision, VerificationResult


@dataclass
class AgentStepTrace:
    name: str
    status: str
    input_summary: str = ""
    output_summary: str = ""
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


@dataclass
class PlannerResult:
    enabled: bool
    complex_query: bool
    subqueries: list[str]
    reason: str


ReflectionAction = Literal["finish", "repair", "refuse"]


@dataclass
class ReflectionDecision:
    action: ReflectionAction
    reason: str


@dataclass
class RetrievalResult:
    vector_results: list[dict]
    graph_results: list[dict]


@dataclass
class AgentWorkflowState:
    query: str
    session_id: str
    memory_context: MemoryContext | None = None
    memory_prompt: str = ""
    route: RouteDecision | None = None
    planner_result: PlannerResult | None = None
    planned_queries: list[str] = field(default_factory=list)
    expanded_queries: list[str] = field(default_factory=list)
    vector_results: list[dict] = field(default_factory=list)
    graph_results: list[dict] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    draft: AnswerDraft | None = None
    verification: VerificationResult | None = None
    retry_count: int = 0
    final_response: FinalResponse | None = None
    trace: list[AgentStepTrace] = field(default_factory=list)
