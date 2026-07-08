import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "eval_report.jsonl"
RAGAS_PATH = ROOT / "ragas_report.json"


def load_rows() -> list[dict]:
    rows: list[dict] = []
    with REPORT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness
    except ImportError as exc:
        print("RAGAS dependencies are not installed. Install ragas and datasets to run this optional evaluation.")
        print(f"Import error: {exc}")
        return

    rows = load_rows()
    dataset = Dataset.from_list(
        [
            {
                "question": row["query"],
                "answer": row.get("answer") or "",
                "contexts": [row.get("answer") or ""],
            }
            for row in rows
        ]
    )
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    payload = result.to_pandas().to_dict(orient="records")
    with RAGAS_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"wrote RAGAS report to {RAGAS_PATH}")


if __name__ == "__main__":
    main()
