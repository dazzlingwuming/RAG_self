from __future__ import annotations

import time

from hybrid_graph_rag_app import settings
from hybrid_graph_rag_app.agents.answer_agent import AnswerAgent
from hybrid_graph_rag_app.agents.evidence_agent import EvidenceAgent
from hybrid_graph_rag_app.agents.memory_agent import MemoryAgent
from hybrid_graph_rag_app.agents.planner_agent import PlannerAgent
from hybrid_graph_rag_app.agents.query_agent import QueryAgent
from hybrid_graph_rag_app.agents.reflection_agent import ReflectionAgent
from hybrid_graph_rag_app.agents.retriever_agent import RetrieverAgent
from hybrid_graph_rag_app.agents.router_agent import RouterAgent
from hybrid_graph_rag_app.agents.state import AgentStepTrace, AgentWorkflowState
from hybrid_graph_rag_app.agents.verifier_agent import VerifierAgent
from hybrid_graph_rag_app.schemas import FinalResponse, RouteDecision, VerificationResult


class AgentWorkflow:
    def __init__(
        self,
        memory_agent: MemoryAgent,
        router_agent: RouterAgent,
        planner_agent: PlannerAgent,
        query_agent: QueryAgent,
        retriever_agent: RetrieverAgent,
        evidence_agent: EvidenceAgent,
        answer_agent: AnswerAgent,
        verifier_agent: VerifierAgent,
        reflection_agent: ReflectionAgent,
        vector_enabled: bool,
        vector_backend: str,
    ) -> None:
        self.memory_agent = memory_agent
        self.router_agent = router_agent
        self.planner_agent = planner_agent
        self.query_agent = query_agent
        self.retriever_agent = retriever_agent
        self.evidence_agent = evidence_agent
        self.answer_agent = answer_agent
        self.verifier_agent = verifier_agent
        self.reflection_agent = reflection_agent
        self.vector_enabled = vector_enabled
        self.vector_backend = vector_backend

    def run(self, query: str, session_id: str) -> dict:
        state = AgentWorkflowState(query=query, session_id=session_id)
        self._load_memory(state)
        self._route(state)

        if state.route and state.route.mode == "insufficient_or_out_of_scope":
            response = self._refusal(state.query, state.route, [], [], state.route.reason)
            return self._finish(state, response)

        self._plan(state)
        self._expand_queries(state)
        self._retrieve(state)
        self._prepare_evidence(state)

        if state.route and not state.evidence:
            response = self._refusal(
                state.query,
                state.route,
                state.vector_results,
                state.graph_results,
                "文档检索和图谱检索都没有返回可用于回答的证据。",
            )
            return self._finish(state, response)

        self._answer(state)
        self._verify(state)
        self._reflect(state)
        return self._finish_with_answer(state)

    def _load_memory(self, state: AgentWorkflowState) -> None:
        started = time.perf_counter()
        state.memory_context = self.memory_agent.load(session_id=state.session_id, query=state.query, turns=8)
        state.memory_prompt = state.memory_context.format_for_prompt()
        self._trace(state, "memory_read", "ok", state.query, state.memory_context.memory_usage.reason, started)

    def _route(self, state: AgentWorkflowState) -> None:
        started = time.perf_counter()
        state.route = self.router_agent.route(state.query)
        self._trace(state, "route", "ok", state.query, state.route.summary, started)

    def _plan(self, state: AgentWorkflowState) -> None:
        started = time.perf_counter()
        state.planner_result = self.planner_agent.plan(state.query)
        state.planned_queries = state.planner_result.subqueries or [state.query]
        self._trace(state, "plan", "ok", state.query, state.planner_result.reason, started)

    def _expand_queries(self, state: AgentWorkflowState) -> None:
        started = time.perf_counter()
        state.expanded_queries = self.query_agent.expand(
            query=state.query,
            history=state.memory_prompt,
            planned_queries=state.planned_queries,
        )
        self._trace(state, "query_expand", "ok", state.query, f"{len(state.expanded_queries)} queries", started)

    def _retrieve(self, state: AgentWorkflowState) -> None:
        started = time.perf_counter()
        result = self.retriever_agent.retrieve(state.expanded_queries or [state.query])
        state.vector_results = result.vector_results
        state.graph_results = result.graph_results
        self._trace(state, "retrieve", "ok", state.query, f"vector={len(state.vector_results)}, graph={len(state.graph_results)}", started)

    def _prepare_evidence(self, state: AgentWorkflowState) -> None:
        started = time.perf_counter()
        if not state.route:
            raise ValueError("route is required before evidence preparation")
        state.vector_results, state.graph_results, state.evidence = self.evidence_agent.prepare(
            vector_results=state.vector_results,
            graph_results=state.graph_results,
            route=state.route,
        )
        self._trace(state, "evidence_fuse", "ok", state.query, f"evidence={len(state.evidence)}", started)

    def _answer(self, state: AgentWorkflowState) -> None:
        started = time.perf_counter()
        if not state.route:
            raise ValueError("route is required before answer generation")
        state.draft = self.answer_agent.generate(
            query=state.query,
            memory_context=state.memory_prompt,
            route=state.route,
            evidence=state.evidence,
        )
        self._trace(state, "answer", "ok", state.query, f"citations={len(state.draft.cited_sources)}", started)

    def _verify(self, state: AgentWorkflowState) -> None:
        started = time.perf_counter()
        if not state.draft:
            raise ValueError("draft is required before verification")
        state.verification = self.verifier_agent.verify(query=state.query, answer=state.draft.answer, evidence=state.evidence)
        self._trace(state, "verify", "ok", state.query, state.verification.status, started)

    def _reflect(self, state: AgentWorkflowState) -> None:
        while state.verification and state.draft:
            started = time.perf_counter()
            decision = self.reflection_agent.decide(state.verification, state.retry_count)
            self._trace(state, "reflect", "ok", state.verification.status, decision.reason, started)
            if decision.action == "repair":
                repair_started = time.perf_counter()
                state.draft = self.answer_agent.repair(state.query, state.draft, state.evidence, state.verification)
                state.retry_count += 1
                self._trace(state, "answer_repair", "ok", state.query, f"retry={state.retry_count}", repair_started)
                self._verify(state)
                continue
            if decision.action == "refuse":
                state.verification = VerificationResult(
                    supported=False,
                    confidence=state.verification.confidence,
                    status="refused",
                    reason=state.verification.reason,
                    unsupported_claims=state.verification.unsupported_claims,
                )
            return

    def _finish_with_answer(self, state: AgentWorkflowState) -> dict:
        if not state.route or not state.draft or not state.verification:
            raise ValueError("route, draft and verification are required before finishing")

        if state.verification.status == "refused":
            answer = f"核心回答：知识库中没有足够依据回答“{state.query}”。\n依据说明：{state.verification.reason}\n关键证据：无"
        else:
            answer = state.draft.answer

        cited = set(self.answer_agent.cited_sources(answer))
        sources = [item for item in state.evidence if item.evidence_id in cited] or state.evidence[: settings.FINAL_SOURCE_LIMIT]
        response = FinalResponse(
            answer=answer,
            status=state.verification.status,
            confidence=state.verification.confidence,
            route=state.route,
            verification=state.verification,
            sources=sources,
            vector_enabled=self.vector_enabled,
            vector_backend=self.vector_backend,
            vector_results=state.vector_results,
            graph_results=state.graph_results,
            memory_usage=state.memory_context.memory_usage if state.memory_context else None,
            used_memories=state.memory_context.long_term_memories if state.memory_context else [],
        )
        return self._finish(state, response)

    def _finish(self, state: AgentWorkflowState, response: FinalResponse) -> dict:
        write_result = self.memory_agent.save(
            session_id=state.session_id,
            query=state.query,
            answer=response.answer,
            status=response.status,
            confidence=response.confidence,
        )
        response.vector_enabled = self.vector_enabled
        response.vector_backend = self.vector_backend
        if state.memory_context:
            response.memory_usage = state.memory_context.memory_usage
            response.used_memories = state.memory_context.long_term_memories
        response.memory_write_result = write_result
        result = response.to_dict()
        if getattr(settings, "INCLUDE_AGENT_TRACE", False):
            result["agent_trace"] = [item.to_dict() for item in state.trace]
        return result

    @staticmethod
    def _refusal(query: str, route: RouteDecision, vector_results: list[dict], graph_results: list[dict], reason: str) -> FinalResponse:
        verification = VerificationResult(
            supported=False,
            confidence=0.0,
            status="refused",
            reason=reason,
            unsupported_claims=[],
        )
        return FinalResponse(
            answer=f"核心回答：知识库中没有足够依据回答“{query}”。\n依据说明：{reason}\n关键证据：无",
            status="refused",
            confidence=0.0,
            route=route,
            verification=verification,
            sources=[],
            vector_enabled=False,
            vector_backend="disabled",
            vector_results=vector_results,
            graph_results=graph_results,
        )

    @staticmethod
    def _trace(
        state: AgentWorkflowState,
        name: str,
        status: str,
        input_summary: str,
        output_summary: str,
        started: float,
    ) -> None:
        state.trace.append(
            AgentStepTrace(
                name=name,
                status=status,
                input_summary=input_summary[:160],
                output_summary=output_summary[:220],
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        )
