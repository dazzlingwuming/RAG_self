from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from hybrid_graph_rag_app import settings
from hybrid_graph_rag_app.history_store import history_to_text, load_history, save_turn
from hybrid_graph_rag_app.memory_policy import decide_memory_usage
from hybrid_graph_rag_app.schemas import MemoryRecord, MemoryUsageDecision, MemoryWriteResult


ALLOWED_CATEGORIES = {"preference", "profile", "project", "constraint", "task_state", "style"}
SENSITIVE_HINTS = ("密码", "身份证", "银行卡", "手机号", "住址", "api key", "apikey", "secret", "token")
KNOWLEDGE_FACT_HINTS = ("是什么", "是谁", "外文名", "位于", "朝代", "作者")
CONFLICT_PAIRS = (
    (("简洁", "简短", "短一点"), ("详细", "展开", "完整")),
    (("不要", "不需要"), ("需要", "希望", "必须")),
)


@dataclass
class ExtractedMemory:
    should_write: bool
    content: str
    category: str
    importance: float
    confidence: float
    evidence_type: str
    tags: list[str]
    reason: str


@dataclass
class MemoryContext:
    short_history: list[dict]
    short_history_text: str
    summary: str
    long_term_memories: list[MemoryRecord]
    memory_usage: MemoryUsageDecision

    def format_for_prompt(self) -> str:
        sections: list[str] = [f"记忆使用决策：{self.memory_usage.reason}"]
        if "summary" in self.memory_usage.memory_types:
            sections.append(f"会话摘要：\n{self.summary or '暂无会话摘要。'}")
        if "long_term" in self.memory_usage.memory_types:
            long_term_text = "\n".join(
                f"- [{item.memory_id}] ({item.category}, confidence={item.confidence:.2f}, status={item.status}) {item.content}"
                for item in self.long_term_memories
            )
            sections.append(f"相关长期记忆：\n{long_term_text or '无相关长期记忆。'}")
        if "short_term" in self.memory_usage.memory_types:
            sections.append(f"短期历史：\n{self.short_history_text or '暂无短期历史。'}")
        if not self.memory_usage.use_memory:
            sections.append("本轮未启用长期记忆；事实回答只依赖本轮知识库证据。")
        return "\n\n".join(sections)


class MemoryManager:
    def __init__(self) -> None:
        self.history_path = settings.HISTORY_PATH
        self.summary_path = settings.SESSION_SUMMARY_PATH
        self.long_term_path = settings.LONG_TERM_MEMORY_PATH

    def load_context(self, session_id: str, query: str, turns: int = 8) -> MemoryContext:
        short_history = load_history(path=self.history_path, session_id=session_id, turns=turns)
        summary = self._load_summary(session_id).get("summary", "")
        usage = decide_memory_usage(query=query, short_history=short_history, summary=summary)
        memories: list[MemoryRecord] = []
        if usage.use_memory and "long_term" in usage.memory_types:
            memories = self.search_long_term(session_id=session_id, query=query, k=settings.LONG_TERM_MEMORY_TOP_K)
        return MemoryContext(
            short_history=short_history if "short_term" in usage.memory_types else [],
            short_history_text=history_to_text(short_history) if "short_term" in usage.memory_types else "",
            summary=summary if "summary" in usage.memory_types else "",
            long_term_memories=memories,
            memory_usage=usage,
        )

    def save_turn(self, session_id: str, query: str, answer: str, status: str, confidence: float) -> MemoryWriteResult:
        save_turn(path=self.history_path, session_id=session_id, query=query, answer=answer)
        history = load_history(path=self.history_path, session_id=session_id, turns=settings.MEMORY_SUMMARY_TURNS)
        self._update_summary_if_needed(session_id=session_id, history=history)
        return self._update_long_term(session_id=session_id, query=query, answer=answer, status=status, confidence=confidence)

    def search_long_term(self, session_id: str, query: str, k: int = 4) -> list[MemoryRecord]:
        data = self._read_json(self.long_term_path, default={})
        items = [self._record_from_dict(session_id, item) for item in data.get(session_id, [])]
        now = datetime.now()
        scored: list[tuple[float, MemoryRecord]] = []
        query_terms = _terms(query)
        for item in items:
            if item.status != "active" or self._is_expired(item, now):
                continue
            overlap = len(query_terms & _terms(item.content))
            keyword_score = overlap / max(len(query_terms), 1)
            access_bonus = min(item.access_count, 5) * 0.05
            age_days = max((now - _parse_time(item.updated_at)).days, 0)
            age_decay = min(age_days / max(item.ttl_days, 1), 1.0) * 0.1
            score = keyword_score * 2 + item.importance * 0.8 + item.confidence * 0.7 + access_bonus - age_decay
            if score > settings.LONG_TERM_MEMORY_MIN_SCORE:
                item.score = round(score, 4)
                scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        selected = [item for _, item in scored[:k]]
        if selected:
            self._touch_memories(session_id, selected)
        return selected

    def _load_summary(self, session_id: str) -> dict:
        data = self._read_json(self.summary_path, default={})
        return data.get(session_id, {})

    def _update_summary_if_needed(self, session_id: str, history: list[dict]) -> None:
        user_messages = [item.get("content", "") for item in history if item.get("type") == "human"]
        existing = self._load_summary(session_id)
        turn_count = len(user_messages)
        previous_turn_count = int(existing.get("turn_count", 0) or 0)
        should_update = (
            turn_count == 1
            or turn_count - previous_turn_count >= settings.SUMMARY_UPDATE_EVERY_N_TURNS
            or sum(len(item.get("content", "")) for item in history) >= settings.SUMMARY_MAX_HISTORY_CHARS
        )
        if not should_update:
            return

        data = self._read_json(self.summary_path, default={})
        previous_summary = existing.get("summary", "")
        recent_topics = "；".join(user_messages[-settings.MEMORY_SUMMARY_TOPIC_LIMIT :])
        if previous_summary:
            summary = f"{previous_summary}\n最近新增关注：{recent_topics}"
        else:
            summary = f"用户最近关注的问题：{recent_topics}" if recent_topics else "暂无摘要。"
        data[session_id] = {
            "summary": summary[: settings.SUMMARY_MAX_CHARS],
            "updated_at": _now(),
            "turn_count": turn_count,
        }
        self._write_json(self.summary_path, data)

    def _update_long_term(self, session_id: str, query: str, answer: str, status: str, confidence: float) -> MemoryWriteResult:
        candidates = self._extract_memory_candidates(query=query, answer=answer, status=status, confidence=confidence)
        if not candidates:
            return MemoryWriteResult(written=False, skipped_reason="本轮没有高价值长期记忆候选。")

        data = self._read_json(self.long_term_path, default={})
        existing = [self._record_from_dict(session_id, item) for item in data.get(session_id, [])]
        write_result = MemoryWriteResult(written=False, skipped_reason="候选记忆未通过写入门控。")

        for candidate in candidates:
            gate_reason = self._write_gate(candidate)
            if gate_reason:
                write_result = MemoryWriteResult(written=False, skipped_reason=gate_reason)
                continue

            duplicate = self._find_duplicate(existing, candidate.content)
            if duplicate:
                duplicate.content = candidate.content if len(candidate.content) > len(duplicate.content) else duplicate.content
                duplicate.importance = max(duplicate.importance, candidate.importance)
                duplicate.confidence = max(duplicate.confidence, candidate.confidence)
                duplicate.updated_at = _now()
                duplicate.tags = sorted(set(duplicate.tags + candidate.tags))
                write_result = MemoryWriteResult(written=True, memory_id=duplicate.memory_id, reason="更新已有长期记忆。")
                continue

            new_record = MemoryRecord(
                memory_id=f"mem-{uuid4().hex[:12]}",
                user_id=session_id,
                content=candidate.content,
                category=candidate.category,
                importance=candidate.importance,
                confidence=candidate.confidence,
                status="active" if candidate.confidence >= settings.LONG_TERM_MEMORY_WRITE_CONFIDENCE else "low_confidence",
                source="conversation",
                source_turn_id=f"{session_id}-{_now()}",
                evidence_type=candidate.evidence_type,
                created_at=_now(),
                updated_at=_now(),
                last_accessed_at=_now(),
                access_count=0,
                ttl_days=settings.LONG_TERM_MEMORY_TTL_DAYS,
                tags=candidate.tags,
                metadata={"extract_reason": candidate.reason},
            )
            contradicted_ids = self._mark_conflicts(existing, new_record)
            existing.append(new_record)
            write_result = MemoryWriteResult(
                written=True,
                memory_id=new_record.memory_id,
                reason=candidate.reason,
                contradicted_ids=contradicted_ids,
            )

        existing = self._compact_memories(existing)
        data[session_id] = [record.to_dict() for record in existing]
        self._write_json(self.long_term_path, data)
        return write_result

    @staticmethod
    def _extract_memory_candidates(query: str, answer: str, status: str, confidence: float) -> list[ExtractedMemory]:
        candidates: list[ExtractedMemory] = []
        preference_patterns = ("我希望", "我想", "我喜欢", "我需要", "以后", "请你", "不要")
        project_patterns = ("项目", "简历", "面试", "RAG", "Agent", "记忆模块")
        constraint_patterns = ("必须", "不能", "不要", "只", "证据", "引用")

        if any(pattern in query for pattern in preference_patterns):
            candidates.append(
                ExtractedMemory(
                    should_write=True,
                    content=f"用户明确偏好或要求：{query}",
                    category="preference",
                    importance=0.85,
                    confidence=0.9,
                    evidence_type="user_explicit",
                    tags=_tags(query),
                    reason="用户明确表达了偏好或要求。",
                )
            )
        if any(pattern in query for pattern in project_patterns):
            candidates.append(
                ExtractedMemory(
                    should_write=True,
                    content=f"用户当前项目上下文：{query}",
                    category="project",
                    importance=0.75,
                    confidence=0.85,
                    evidence_type="user_explicit",
                    tags=_tags(query),
                    reason="用户问题涉及长期项目背景。",
                )
            )
        if any(pattern in query for pattern in constraint_patterns) and any(pattern in query for pattern in ("证据", "引用", "不要", "不能")):
            candidates.append(
                ExtractedMemory(
                    should_write=True,
                    content=f"用户长期约束：{query}",
                    category="constraint",
                    importance=0.8,
                    confidence=0.88,
                    evidence_type="user_explicit",
                    tags=_tags(query),
                    reason="用户明确提出约束条件。",
                )
            )
        if status == "verified" and confidence >= settings.LONG_TERM_MEMORY_MIN_CONFIDENCE and any(pattern in query for pattern in project_patterns):
            compact_answer = " ".join(answer.split())[:220]
            candidates.append(
                ExtractedMemory(
                    should_write=True,
                    content=f"已验证项目讨论：问题={query}；回答要点={compact_answer}",
                    category="task_state",
                    importance=0.62,
                    confidence=min(confidence, 0.75),
                    evidence_type="verified_by_doc",
                    tags=_tags(query),
                    reason="本轮项目讨论已通过回答校验，可作为任务进展记忆。",
                )
            )
        return candidates

    @staticmethod
    def _write_gate(candidate: ExtractedMemory) -> str | None:
        if not candidate.should_write:
            return "候选记忆标记为不写入。"
        if candidate.category not in ALLOWED_CATEGORIES:
            return "记忆类别不在允许范围内。"
        if candidate.confidence < settings.LONG_TERM_MEMORY_WRITE_CONFIDENCE:
            return "候选记忆置信度不足。"
        if candidate.importance < settings.LONG_TERM_MEMORY_WRITE_IMPORTANCE:
            return "候选记忆重要度不足。"
        lowered = candidate.content.lower()
        if any(hint in lowered for hint in SENSITIVE_HINTS):
            return "候选记忆疑似包含敏感信息。"
        if candidate.category not in {"preference", "project", "constraint", "task_state", "style", "profile"}:
            return "候选记忆不是用户相关长期信息。"
        if candidate.category == "task_state" and any(hint in candidate.content for hint in KNOWLEDGE_FACT_HINTS) and "项目" not in candidate.content:
            return "候选记忆更像知识库事实，不写入长期用户记忆。"
        return None

    @staticmethod
    def _find_duplicate(items: list[MemoryRecord], content: str) -> MemoryRecord | None:
        content_terms = _terms(content)
        for item in items:
            item_terms = _terms(item.content)
            if content == item.content:
                return item
            if content_terms and len(content_terms & item_terms) / max(len(content_terms), 1) >= 0.75:
                return item
        return None

    @staticmethod
    def _mark_conflicts(items: list[MemoryRecord], new_record: MemoryRecord) -> list[str]:
        contradicted: list[str] = []
        for item in items:
            if item.status != "active" or item.category != new_record.category:
                continue
            if _looks_conflicting(item.content, new_record.content) and new_record.confidence >= item.confidence:
                item.status = "contradicted"
                item.contradicted_by = new_record.memory_id
                item.updated_at = _now()
                contradicted.append(item.memory_id)
        return contradicted

    @staticmethod
    def _compact_memories(items: list[MemoryRecord]) -> list[MemoryRecord]:
        now = datetime.now()
        active = []
        for item in items:
            if MemoryManager._is_expired(item, now) and item.status == "active":
                item.status = "outdated"
            active.append(item)
        active.sort(key=lambda item: (item.status == "active", item.importance, item.confidence, item.access_count, item.updated_at), reverse=True)
        return active[: settings.LONG_TERM_MEMORY_LIMIT]

    @staticmethod
    def _is_expired(item: MemoryRecord, now: datetime) -> bool:
        updated_at = _parse_time(item.updated_at)
        return (now - updated_at).days > item.ttl_days

    def _touch_memories(self, session_id: str, selected: list[MemoryRecord]) -> None:
        selected_ids = {item.memory_id for item in selected}
        data = self._read_json(self.long_term_path, default={})
        changed = False
        for item in data.get(session_id, []):
            if item.get("memory_id") in selected_ids:
                item["last_accessed_at"] = _now()
                item["access_count"] = int(item.get("access_count", 0)) + 1
                changed = True
        if changed:
            self._write_json(self.long_term_path, data)

    @staticmethod
    def _record_from_dict(session_id: str, item: dict) -> MemoryRecord:
        now = _now()
        return MemoryRecord(
            memory_id=item.get("memory_id", f"mem-{uuid4().hex[:12]}"),
            user_id=item.get("user_id", session_id),
            content=item.get("content", ""),
            category=item.get("category", "project"),
            importance=float(item.get("importance", 0.5)),
            confidence=float(item.get("confidence", 0.8)),
            status=item.get("status", "active"),
            source=item.get("source", "conversation"),
            source_turn_id=item.get("source_turn_id"),
            evidence_type=item.get("evidence_type", "user_explicit"),
            created_at=item.get("created_at", now),
            updated_at=item.get("updated_at", now),
            last_accessed_at=item.get("last_accessed_at", now),
            access_count=int(item.get("access_count", 0)),
            ttl_days=int(item.get("ttl_days", settings.LONG_TERM_MEMORY_TTL_DAYS)),
            tags=list(item.get("tags", [])),
            contradicted_by=item.get("contradicted_by"),
            metadata=dict(item.get("metadata", {})),
            score=item.get("score"),
        )

    @staticmethod
    def _read_json(path: Path, default):
        if not path.exists() or path.stat().st_size == 0:
            return default
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _write_json(path: Path, data) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def _terms(text: str) -> set[str]:
    normalized = re.sub(r"[^一-鿿A-Za-z0-9]+", " ", text).lower()
    terms = {part for part in normalized.split() if len(part) >= 2}
    compact = normalized.replace(" ", "")
    for idx in range(max(len(compact) - 1, 0)):
        terms.add(compact[idx : idx + 2])
    return terms


def _tags(text: str) -> list[str]:
    tags = []
    mapping = {
        "简历": "resume",
        "项目": "project",
        "面试": "interview",
        "RAG": "rag",
        "Agent": "agent",
        "记忆": "memory",
        "证据": "evidence",
        "引用": "citation",
    }
    for key, value in mapping.items():
        if key in text and value not in tags:
            tags.append(value)
    return tags


def _looks_conflicting(old: str, new: str) -> bool:
    for left, right in CONFLICT_PAIRS:
        old_left = any(word in old for word in left)
        old_right = any(word in old for word in right)
        new_left = any(word in new for word in left)
        new_right = any(word in new for word in right)
        if (old_left and new_right) or (old_right and new_left):
            return True
    return False


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _parse_time(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime.now()
