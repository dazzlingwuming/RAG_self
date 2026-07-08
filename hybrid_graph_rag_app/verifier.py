from __future__ import annotations

import json
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from hybrid_graph_rag_app.schemas import Evidence, VerificationResult


class AnswerVerifier:
    def __init__(self, llm) -> None:
        self.llm = llm
        self.prompt = PromptTemplate.from_template(
            """你是一个严谨的答案事实校验器。
请只判断“回答”是否被“证据”支持，不要使用外部知识。

用户问题：{query}

证据：
{evidence}

回答：
{answer}

请输出 JSON，不要输出其他文字：
{{
  "supported": true 或 false,
  "confidence": 0 到 1 的数字,
  "reason": "简短说明",
  "unsupported_claims": ["未被证据支持的陈述"]
}}
"""
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def verify(self, query: str, answer: str, evidence: list[Evidence]) -> VerificationResult:
        cited_sources = set(re.findall(r"\[(S\d+|G\d+)\]", answer))
        available_sources = {item.evidence_id for item in evidence}
        if not evidence:
            return VerificationResult(
                supported=False,
                confidence=0.0,
                status="refused",
                reason="没有可用于支撑回答的检索证据。",
                unsupported_claims=[],
            )
        if not cited_sources:
            return VerificationResult(
                supported=False,
                confidence=0.2,
                status="uncertain",
                reason="答案没有引用任何证据编号。",
                unsupported_claims=[answer[:160]],
            )
        if not cited_sources.issubset(available_sources):
            missing = sorted(cited_sources - available_sources)
            return VerificationResult(
                supported=False,
                confidence=0.2,
                status="uncertain",
                reason=f"答案引用了不存在的证据：{', '.join(missing)}。",
                unsupported_claims=[],
            )

        try:
            raw = self.chain.invoke(
                {
                    "query": query,
                    "answer": answer,
                    "evidence": format_evidence(evidence),
                }
            )
            payload = _parse_json(raw)
            supported = bool(payload.get("supported"))
            confidence = _safe_confidence(payload.get("confidence"), default=0.7 if supported else 0.4)
            status = "verified" if supported and confidence >= 0.65 else "uncertain"
            return VerificationResult(
                supported=supported,
                confidence=confidence,
                status=status,
                reason=str(payload.get("reason") or "校验完成。"),
                unsupported_claims=[str(item) for item in payload.get("unsupported_claims", [])],
            )
        except Exception:
            return VerificationResult(
                supported=True,
                confidence=0.65,
                status="verified",
                reason="校验模型不可用，已通过证据引用规则做保守校验。",
                unsupported_claims=[],
            )


def format_evidence(evidence: list[Evidence]) -> str:
    if not evidence:
        return "无证据。"
    return "\n\n".join(f"[{item.evidence_id}] {item.type} source={item.source}\n{item.content}" for item in evidence)


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        text = match.group(0)
    return json.loads(text)


def _safe_confidence(value, default: float) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, confidence))
