from hybrid_graph_rag_app.memory_manager import MemoryContext, MemoryManager
from hybrid_graph_rag_app.schemas import MemoryWriteResult


class MemoryAgent:
    def __init__(self, memory_manager: MemoryManager) -> None:
        self.memory_manager = memory_manager

    def load(self, session_id: str, query: str, turns: int = 8) -> MemoryContext:
        return self.memory_manager.load_context(session_id=session_id, query=query, turns=turns)

    def save(self, session_id: str, query: str, answer: str, status: str, confidence: float) -> MemoryWriteResult:
        return self.memory_manager.save_turn(
            session_id=session_id,
            query=query,
            answer=answer,
            status=status,
            confidence=confidence,
        )
