from hybrid_graph_rag_app.agents.workflow import AgentWorkflow
from hybrid_graph_rag_app.memory_manager import MemoryContext
from hybrid_graph_rag_app.schemas import MemoryUsageDecision, MemoryWriteResult, RouteDecision


class FakeMemoryAgent:
    def load(self, session_id, query, turns=8):
        return MemoryContext(
            short_history=[],
            short_history_text="",
            summary="",
            long_term_memories=[],
            memory_usage=MemoryUsageDecision(use_memory=False, memory_types=[], reason="test"),
        )

    def save(self, session_id, query, answer, status, confidence):
        return MemoryWriteResult(written=False, skipped_reason="test")


class FakeRouterAgent:
    def route(self, query):
        return RouteDecision(
            mode="insufficient_or_out_of_scope",
            graph_score=0,
            doc_score=0,
            summary="mode=insufficient_or_out_of_scope",
            strategies=[],
            reason="out of scope",
        )


class UnusedAgent:
    def __getattr__(self, name):
        raise AssertionError(f"unexpected workflow call: {name}")


def test_agent_workflow_refuses_out_of_scope_without_retrieval():
    workflow = AgentWorkflow(
        memory_agent=FakeMemoryAgent(),
        router_agent=FakeRouterAgent(),
        planner_agent=UnusedAgent(),
        query_agent=UnusedAgent(),
        retriever_agent=UnusedAgent(),
        evidence_agent=UnusedAgent(),
        answer_agent=UnusedAgent(),
        verifier_agent=UnusedAgent(),
        reflection_agent=UnusedAgent(),
        vector_enabled=True,
        vector_backend="fts",
    )

    result = workflow.run(query="火星殖民计划预算是多少", session_id="test")

    assert result["status"] == "refused"
    assert result["route"]["mode"] == "insufficient_or_out_of_scope"
    assert result["memory_write_result"]["written"] is False
