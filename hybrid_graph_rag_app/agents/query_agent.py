from hybrid_graph_rag_app import settings
from hybrid_graph_rag_app.query_enhancer import QueryEnhancer


class QueryAgent:
    def __init__(self, query_enhancer: QueryEnhancer) -> None:
        self.query_enhancer = query_enhancer

    def expand(self, query: str, history: str, planned_queries: list[str] | None = None) -> list[str]:
        seed_queries = planned_queries or [query]
        expanded: list[str] = []
        for current_query in seed_queries:
            if settings.QUERY_EXPANSION_ENABLED:
                candidates = self.query_enhancer.expand(
                    query=current_query,
                    history=history,
                    query_num=settings.QUERY_EXPANSION_NUM,
                    enable_hyde=settings.HYDE_ENABLED,
                )
            else:
                candidates = [current_query]
            for candidate in candidates:
                if candidate and candidate not in expanded:
                    expanded.append(candidate)
        return expanded or [query]
