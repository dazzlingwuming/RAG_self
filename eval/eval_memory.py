import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_graph_rag_app import settings
from hybrid_graph_rag_app.memory_manager import MemoryManager
from hybrid_graph_rag_app.memory_policy import decide_memory_usage

ROOT = Path(__file__).resolve().parent
QUESTION_PATH = ROOT / "eval_memory_questions.jsonl"
REPORT_PATH = ROOT / "eval_memory_report.jsonl"
SUMMARY_PATH = ROOT / "eval_memory_summary.json"
SESSION_ID = "eval-memory"


def load_questions(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def reset_eval_memory(manager: MemoryManager) -> None:
    for path in (settings.HISTORY_PATH, settings.SESSION_SUMMARY_PATH, settings.LONG_TERM_MEMORY_PATH):
        data = manager._read_json(path, default={})
        if isinstance(data, dict) and SESSION_ID in data:
            data.pop(SESSION_ID, None)
            manager._write_json(path, data)


def eval_usage_policy(item: dict) -> dict:
    decision = decide_memory_usage(
        query=item["query"],
        short_history=[{"type": "human", "content": "前面讨论了简历风格。"}],
        summary="用户希望项目说明适合 HR 和面试官阅读。",
    )
    expected_types = set(item.get("expected_memory_types", []))
    actual_types = set(decision.memory_types)
    passed = decision.use_memory == item.get("expected_use_memory") and expected_types.issubset(actual_types)
    return {
        "id": item["id"],
        "type": item["type"],
        "query": item["query"],
        "expected_use_memory": item.get("expected_use_memory"),
        "actual_use_memory": decision.use_memory,
        "expected_memory_types": sorted(expected_types),
        "actual_memory_types": decision.memory_types,
        "reason": decision.reason,
        "outcome": "pass" if passed else "fail",
    }


def eval_write_gate(manager: MemoryManager, item: dict) -> dict:
    result = manager.save_turn(
        session_id=SESSION_ID,
        query=item["query"],
        answer="核心回答：这是一次记忆评估模拟回答。",
        status="verified",
        confidence=0.9,
    )
    memories = manager.search_long_term(session_id=SESSION_ID, query=item["query"], k=5)
    expected_written = bool(item.get("expected_written"))
    category = item.get("expected_category")
    category_hit = category is None or any(memory.category == category for memory in memories)
    passed = result.written == expected_written and (not expected_written or category_hit)
    return {
        "id": item["id"],
        "type": item["type"],
        "query": item["query"],
        "expected_written": expected_written,
        "actual_written": result.written,
        "memory_id": result.memory_id,
        "expected_category": category,
        "retrieved_categories": [memory.category for memory in memories],
        "skipped_reason": result.skipped_reason,
        "reason": result.reason,
        "outcome": "pass" if passed else "fail",
    }


def eval_recall(manager: MemoryManager, item: dict) -> dict:
    manager.save_turn(
        session_id=SESSION_ID,
        query=item["seed_query"],
        answer="核心回答：已记录用户对项目说明风格的偏好。",
        status="verified",
        confidence=0.9,
    )
    context = manager.load_context(session_id=SESSION_ID, query=item["recall_query"], turns=8)
    recalled = [memory.content for memory in context.long_term_memories]
    expected_keyword = item.get("expected_recall_keyword", "")
    passed = context.memory_usage.use_memory and any(expected_keyword in content for content in recalled)
    return {
        "id": item["id"],
        "type": item["type"],
        "seed_query": item["seed_query"],
        "recall_query": item["recall_query"],
        "expected_recall_keyword": expected_keyword,
        "actual_use_memory": context.memory_usage.use_memory,
        "recalled_count": len(recalled),
        "recalled_memories": recalled,
        "outcome": "pass" if passed else "fail",
    }


def summarize(rows: list[dict]) -> dict:
    return {
        "total": len(rows),
        "outcome_counts": dict(Counter(row["outcome"] for row in rows)),
        "type_counts": dict(Counter(row["type"] for row in rows)),
        "pass_rate": round(sum(row["outcome"] == "pass" for row in rows) / max(len(rows), 1), 4),
    }


def evaluate() -> None:
    manager = MemoryManager()
    reset_eval_memory(manager)
    rows: list[dict] = []

    for item in load_questions(QUESTION_PATH):
        if item["type"] == "usage_policy":
            rows.append(eval_usage_policy(item))
        elif item["type"] == "write_gate":
            rows.append(eval_write_gate(manager, item))
        elif item["type"] == "recall":
            rows.append(eval_recall(manager, item))

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize(rows)
    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"wrote {len(rows)} rows to {REPORT_PATH}")
    print(f"wrote summary to {SUMMARY_PATH}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    evaluate()
