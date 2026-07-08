from __future__ import annotations

import re

from hybrid_graph_rag_app.schemas import MemoryUsageDecision


FOLLOW_UP_WORDS = (
    "继续",
    "刚才",
    "上面",
    "之前",
    "这个",
    "那个",
    "它",
    "该项目",
    "按我的",
    "按照我",
    "再改",
    "下一步",
)
PREFERENCE_WORDS = ("简历", "风格", "偏好", "写法", "详细", "简洁", "HR", "面试", "项目")
NO_MEMORY_WORDS = ("不要使用历史", "不要用记忆", "忽略记忆", "不要结合之前")
FACT_QUESTION_PATTERNS = (
    r".+是什么[？?]?$",
    r".+是谁[？?]?$",
    r".+在哪里[？?]?$",
    r".+外文名.*[？?]?$",
)


def decide_memory_usage(query: str, short_history: list[dict] | None = None, summary: str | None = None) -> MemoryUsageDecision:
    if any(word in query for word in NO_MEMORY_WORDS):
        return MemoryUsageDecision(
            use_memory=False,
            memory_types=[],
            reason="用户明确要求不使用历史记忆。",
            risk_level="low",
        )

    if any(word in query for word in FOLLOW_UP_WORDS):
        return MemoryUsageDecision(
            use_memory=True,
            memory_types=["short_term", "summary", "long_term"],
            reason="检测到追问或上下文依赖表达，需要读取记忆辅助理解。",
            risk_level="low",
        )

    if any(word in query for word in PREFERENCE_WORDS):
        return MemoryUsageDecision(
            use_memory=True,
            memory_types=["summary", "long_term"],
            reason="问题涉及用户项目背景、表达偏好或长期目标。",
            risk_level="low",
        )

    if _contains_ambiguous_pronoun(query) and (short_history or summary):
        return MemoryUsageDecision(
            use_memory=True,
            memory_types=["short_term", "summary"],
            reason="问题包含代词且缺少明确实体，需要短期上下文消解指代。",
            risk_level="medium",
        )

    if any(re.match(pattern, query.strip()) for pattern in FACT_QUESTION_PATTERNS):
        return MemoryUsageDecision(
            use_memory=False,
            memory_types=[],
            reason="当前问题是明确事实型知识库问题，优先依赖本轮检索证据。",
            risk_level="low",
        )

    return MemoryUsageDecision(
        use_memory=False,
        memory_types=[],
        reason="未检测到明显记忆依赖，跳过长期记忆以降低干扰。",
        risk_level="low",
    )


def _contains_ambiguous_pronoun(query: str) -> bool:
    compact = "".join(query.split())
    return compact in {"它是什么", "这个是什么", "这个怎么做", "这个怎么优化"} or any(
        phrase in compact for phrase in ("它的", "这个怎么", "刚才那个")
    )
