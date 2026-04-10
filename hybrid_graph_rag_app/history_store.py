import json
from datetime import datetime
from pathlib import Path


def _ensure_parent(path: Path) -> None:
    # 这里保证历史文件所在目录一定存在，避免首次写入时报目录不存在。
    path.parent.mkdir(parents=True, exist_ok=True)


def load_history(path: Path, session_id: str, turns: int = 8) -> list[dict]:
    # 文件不存在或空文件时，直接返回空历史，避免无意义异常。
    if not path.exists() or path.stat().st_size == 0:
        return []

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    session_history = data.get(session_id, [])
    # 一轮对话包含 human 和 ai 两条记录，所以这里按 turns * 2 截取。
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
    # 这里把一问一答作为两条记录顺序写入，后面还原历史时更直观。
    payload[session_id].extend(
        [
            {"type": "human", "content": query, "time": now},
            {"type": "ai", "content": answer, "time": now},
        ]
    )

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def history_to_text(history: list[dict]) -> str:
    # 这里把结构化历史转成 prompt 可直接拼接的纯文本格式。
    lines: list[str] = []
    for item in history:
        role = item.get("type")
        content = item.get("content", "")
        if role == "human":
            lines.append(f"用户: {content}")
        elif role == "ai":
            lines.append(f"助手: {content}")
    return "\n".join(lines)
