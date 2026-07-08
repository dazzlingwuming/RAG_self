import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from hybrid_graph_rag_app.agents.evidence_agent import EvidenceAgent
from hybrid_graph_rag_app.schemas import AnswerDraft, Evidence, RouteDecision, VerificationResult


class AnswerAgent:
    def __init__(self, llm) -> None:
        self.prompt = PromptTemplate.from_template(
            """你是一个中文知识库可信问答助手。
你只能依据“本轮证据”回答，不能使用外部常识补充事实。
对话历史只用于理解追问，不可作为事实证据。

路由决策：{route_summary}
记忆上下文：
{memory_context}

本轮证据：
{evidence_context}

用户问题：{query}

作答要求：
1. 如果证据不足，直接回答“知识库中没有足够依据回答该问题”。
2. 如果可以回答，答案中的关键事实必须带证据编号，例如 [S1] 或 [G2]。
3. 输出格式固定为：
核心回答：...
依据说明：...
关键证据：...
"""
        )
        self.repair_prompt = PromptTemplate.from_template(
            """请把下面回答改写为更保守、严格基于证据的版本。
必须保留有效证据编号。如果证据不足，请明确拒答。

用户问题：{query}
证据：
{evidence_context}

原回答：
{answer}

校验问题：{verification_reason}

请输出修订后的回答。"""
        )
        self.chain = self.prompt | llm | StrOutputParser()
        self.repair_chain = self.repair_prompt | llm | StrOutputParser()

    def generate(self, query: str, memory_context: str, route: RouteDecision, evidence: list[Evidence]) -> AnswerDraft:
        if not evidence:
            answer = "核心回答：知识库中没有足够依据回答该问题。\n依据说明：本轮未检索到文档或图谱证据。\n关键证据：无"
            return AnswerDraft(answer=answer, cited_sources=[])

        evidence_context = EvidenceAgent.format_evidence_context(evidence)
        try:
            answer = self.chain.invoke(
                {
                    "route_summary": f"{route.summary}; {route.reason}",
                    "memory_context": memory_context,
                    "evidence_context": evidence_context,
                    "query": query,
                }
            )
        except Exception:
            answer = self.fallback_answer(query, route, evidence)
        return AnswerDraft(answer=answer, cited_sources=self.cited_sources(answer))

    def repair(
        self,
        query: str,
        draft: AnswerDraft,
        evidence: list[Evidence],
        verification: VerificationResult,
    ) -> AnswerDraft:
        try:
            answer = self.repair_chain.invoke(
                {
                    "query": query,
                    "evidence_context": EvidenceAgent.format_evidence_context(evidence),
                    "answer": draft.answer,
                    "verification_reason": verification.reason,
                }
            )
        except Exception:
            answer = self.fallback_answer(
                query,
                RouteDecision("hybrid", 0, 0, "repair", ["document", "graph"], "校验失败后的保守修订。"),
                evidence,
            )
        return AnswerDraft(answer=answer, cited_sources=self.cited_sources(answer))

    @staticmethod
    def fallback_answer(query: str, route: RouteDecision, evidence: list[Evidence]) -> str:
        if not evidence:
            return "核心回答：知识库中没有足够依据回答该问题。\n依据说明：本轮没有检索到可引用证据。\n关键证据：无"

        primary = evidence[0]
        evidence_lines = [f"- [{item.evidence_id}] {item.content}" for item in evidence[:4]]
        return "\n".join(
            [
                f"核心回答：根据当前最相关证据，{primary.content} [{primary.evidence_id}]",
                f"依据说明：当前回答基于 {route.mode} 路由下检索到的证据生成，未使用外部知识。",
                "关键证据：",
                *evidence_lines,
            ]
        )

    @staticmethod
    def cited_sources(answer: str) -> list[str]:
        sources = re.findall(r"\[(S\d+|G\d+)\]", answer)
        unique: list[str] = []
        for source in sources:
            if source not in unique:
                unique.append(source)
        return unique
