from hybrid_graph_rag_app import settings
from hybrid_graph_rag_app.agents.state import PlannerResult


class PlannerAgent:
    COMPLEX_HINTS = ("比较", "分别", "对比", "多个方面", "先", "再", "结合", "异同", "区别")

    def plan(self, query: str) -> PlannerResult:
        enabled = bool(getattr(settings, "PLANNER_ENABLED", False))
        complex_query = any(hint in query for hint in self.COMPLEX_HINTS)
        if not enabled:
            return PlannerResult(enabled=False, complex_query=complex_query, subqueries=[query], reason="Planner 默认关闭，直接使用原问题。")
        if not complex_query:
            return PlannerResult(enabled=True, complex_query=False, subqueries=[query], reason="问题不需要拆解，直接检索原问题。")

        separators = ("，", "；", ";", "、")
        subqueries = [query]
        for separator in separators:
            if separator in query:
                parts = [part.strip() for part in query.split(separator) if part.strip()]
                if len(parts) > 1:
                    subqueries = parts[:3]
                    break
        return PlannerResult(enabled=True, complex_query=True, subqueries=subqueries, reason="检测到复杂问题，生成有限子问题用于检索。")
