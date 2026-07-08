from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from hybrid_graph_rag_app import settings
from hybrid_graph_rag_app.agents.state import RetrievalResult
from hybrid_graph_rag_app.graph_retriever import GraphRetriever
from hybrid_graph_rag_app.vector_retriever import VectorRetriever


class RetrieverAgent:
    def __init__(self, vector_retriever: VectorRetriever, graph_retriever: GraphRetriever) -> None:
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        self._cache: dict[tuple[str, str], tuple[float, list[dict]]] = {}
        self._cache_lock = Lock()

    def retrieve(self, queries: list[str]) -> RetrievalResult:
        if not queries:
            return RetrievalResult(vector_results=[], graph_results=[])

        if settings.RETRIEVAL_PARALLEL_ENABLED:
            return self._retrieve_parallel(queries)
        return self._retrieve_sequential(queries)

    def _retrieve_sequential(self, queries: list[str]) -> RetrievalResult:
        vector_results: list[dict] = []
        graph_results: list[dict] = []
        for query in queries:
            vector_results.extend(self._search_vector(query))
            graph_results.extend(self._search_graph(query))
        return RetrievalResult(vector_results=vector_results, graph_results=graph_results)

    def _retrieve_parallel(self, queries: list[str]) -> RetrievalResult:
        vector_results: list[dict] = []
        graph_results: list[dict] = []
        max_workers = max(1, min(settings.RETRIEVAL_MAX_WORKERS, len(queries) * 2))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {}
            for query in queries:
                future_map[executor.submit(self._search_vector, query)] = "vector"
                future_map[executor.submit(self._search_graph, query)] = "graph"
            for future in as_completed(future_map, timeout=settings.RETRIEVAL_TASK_TIMEOUT_SECONDS * max(len(future_map), 1)):
                kind = future_map[future]
                try:
                    results = future.result(timeout=0)
                except Exception:
                    results = []
                if kind == "vector":
                    vector_results.extend(results)
                else:
                    graph_results.extend(results)
        return RetrievalResult(vector_results=vector_results, graph_results=graph_results)

    def _search_vector(self, query: str) -> list[dict]:
        cached = self._get_cache("vector", query)
        if cached is not None:
            return cached
        try:
            results = self.vector_retriever.search(query, k=settings.PER_QUERY_VECTOR_TOP_K)
        except Exception:
            results = []
        self._set_cache("vector", query, results)
        return results

    def _search_graph(self, query: str) -> list[dict]:
        cached = self._get_cache("graph", query)
        if cached is not None:
            return cached
        try:
            results = self.graph_retriever.search(query)[: settings.PER_QUERY_GRAPH_LIMIT]
        except Exception:
            results = []
        self._set_cache("graph", query, results)
        return results

    def _get_cache(self, kind: str, query: str) -> list[dict] | None:
        if not settings.RETRIEVAL_CACHE_ENABLED:
            return None
        key = (kind, query)
        now = time.time()
        with self._cache_lock:
            item = self._cache.get(key)
            if not item:
                return None
            created_at, results = item
            if now - created_at > settings.RETRIEVAL_CACHE_TTL_SECONDS:
                self._cache.pop(key, None)
                return None
            return [dict(result) for result in results]

    def _set_cache(self, kind: str, query: str, results: list[dict]) -> None:
        if not settings.RETRIEVAL_CACHE_ENABLED:
            return
        key = (kind, query)
        with self._cache_lock:
            if len(self._cache) >= settings.RETRIEVAL_CACHE_MAX_ITEMS:
                oldest_key = min(self._cache, key=lambda cache_key: self._cache[cache_key][0])
                self._cache.pop(oldest_key, None)
            self._cache[key] = (time.time(), [dict(result) for result in results])
