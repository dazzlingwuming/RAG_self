import json
from datetime import datetime
from pathlib import Path


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_history(path: Path, session_id: str, turns: int = 8) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    session_history = data.get(session_id, [])
    recent = session_history[-turns * 2 :]
    return recent


def save_turn(path: Path, session_id: str, query: str, answer: str) -> None:
    _ensure_parent(path)
    payload = {}
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

    payload.setdefault(session_id, [])
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload[session_id].extend(
        [
            {"type": "human", "content": query, "time": now},
            {"type": "ai", "content": answer, "time": now},
        ]
    )

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def history_to_text(history: list[dict]) -> str:
    lines: list[str] = []
    for item in history:
        role = item.get("type")
        content = item.get("content", "")
        if role == "human":
            lines.append(f"用户: {content}")
        elif role == "ai":
            lines.append(f"助手: {content}")
    return "\n".join(lines)

