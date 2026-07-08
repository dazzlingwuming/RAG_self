from hybrid_graph_rag_app.memory_manager import MemoryManager, _terms


def test_terms_extracts_chinese_bigrams():
    terms = _terms("我希望简历表达更适合HR")
    assert "我希" in terms
    assert "hr" in terms


def test_memory_candidate_extraction():
    candidates = MemoryManager._extract_memory_candidates(
        query="我希望这个项目介绍更适合写进简历",
        answer="核心回答：已按简历风格整理 [S1]",
        status="verified",
        confidence=0.8,
    )
    categories = {item[1] for item in candidates}
    assert "preference" in categories
    assert "project" in categories
