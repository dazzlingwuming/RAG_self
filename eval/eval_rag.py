import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_graph_rag_app import settings
from hybrid_graph_rag_app.hybrid_service import HybridGraphRAGService


ROOT = Path(__file__).resolve().parent
QUESTION_PATH = ROOT / "test_questions.jsonl"
REPORT_PATH = ROOT / "eval_report.jsonl"
SUMMARY_PATH = ROOT / "eval_summary.json"


def load_questions(path: Path) -> list[dict]:
    questions: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def infer_outcome(row: dict) -> str:
    expected = row.get("expected_status")
    if expected and row.get("status") == expected:
        return "pass"
    if expected:
        return "fail"
    if row.get("type") == "no_answer":
        return "pass" if row.get("status") == "refused" else "review"
    if row.get("source_count", 0) > 0 and row.get("status") in {"verified", "uncertain"}:
        return "review"
    return "review"


def summarize(rows: list[dict]) -> dict:
    status_counts = Counter(row["status"] for row in rows)
    route_counts = Counter(row["route"] for row in rows)
    type_counts = Counter(row["type"] for row in rows)
    outcome_counts = Counter(row["outcome"] for row in rows)
    by_type: dict[str, dict] = {}

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["type"]].append(row)

    for question_type, items in grouped.items():
        by_type[question_type] = {
            "count": len(items),
            "status_counts": dict(Counter(item["status"] for item in items)),
            "route_counts": dict(Counter(item["route"] for item in items)),
            "avg_confidence": round(mean(float(item.get("confidence") or 0) for item in items), 4),
            "avg_latency_ms": round(mean(float(item.get("latency_ms") or 0) for item in items), 2),
            "avg_source_count": round(mean(int(item.get("source_count") or 0) for item in items), 2),
        }

    return {
        "total": len(rows),
        "status_counts": dict(status_counts),
        "route_counts": dict(route_counts),
        "type_counts": dict(type_counts),
        "outcome_counts": dict(outcome_counts),
        "avg_confidence": round(mean(float(row.get("confidence") or 0) for row in rows), 4) if rows else 0,
        "avg_latency_ms": round(mean(float(row.get("latency_ms") or 0) for row in rows), 2) if rows else 0,
        "avg_source_count": round(mean(int(row.get("source_count") or 0) for row in rows), 2) if rows else 0,
        "by_type": by_type,
    }


def evaluate() -> None:
    service = HybridGraphRAGService()
    questions = load_questions(QUESTION_PATH)
    rows: list[dict] = []

    for item in questions:
        started = time.perf_counter()
        result = service.ask(query=item["query"], session_id="eval")
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        row = {
            "id": item.get("id"),
            "type": item.get("type"),
            "query": item["query"],
            "expected_behavior": item.get("expected_behavior"),
            "expected_status": item.get("expected_status"),
            "route": result.get("route", {}).get("mode"),
            "route_reason": result.get("route", {}).get("reason"),
            "status": result.get("status"),
            "confidence": result.get("confidence"),
            "source_count": len(result.get("sources", [])),
            "vector_count": len(result.get("vector_results", [])),
            "graph_count": len(result.get("graph_results", [])),
            "latency_ms": latency_ms,
            "answer": result.get("answer"),
            "verification_reason": result.get("verification", {}).get("reason"),
        }
        row["outcome"] = infer_outcome(row)
        rows.append(row)

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
