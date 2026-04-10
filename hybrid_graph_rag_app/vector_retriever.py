import sqlite3

from langchain_chroma import Chroma

from hybrid_graph_rag_app import settings


class VectorRetriever:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.vectorstore = None
        self.backend = "disabled"
        self.sqlite_path = settings.VECTORSTORE_DIR / "chroma.sqlite3"

        # embedding 模型可用时，优先走真正的语义检索。
        if embedding_model is not None:
            try:
                self.vectorstore = Chroma(
                    persist_directory=str(settings.VECTORSTORE_DIR),
                    embedding_function=embedding_model,
                )
                self.backend = "semantic"
            except Exception:
                self.vectorstore = None

        # 当前环境里 embedding 依赖可能有问题，这里保留 FTS 作为后备模式。
        if self.backend == "disabled" and self.sqlite_path.exists():
            self.backend = "fts"

    def search(self, query: str, k: int | None = None) -> list[dict]:
        top_k = k or settings.VECTOR_TOP_K
        if self.backend == "semantic" and self.vectorstore is not None:
            return self._semantic_search(query, top_k)
        if self.backend == "fts":
            return self._fts_search(query, top_k)
        return []

    def _semantic_search(self, query: str, k: int) -> list[dict]:
        docs = self.vectorstore.similarity_search(query, k=k)
        results: list[dict] = []
        for doc in docs:
            results.append(
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": None,
                    "retrieval_mode": "semantic",
                }
            )
        return results

    def _fts_search(self, query: str, k: int) -> list[dict]:
        terms = self._candidate_terms(query)
        if not terms:
            return []

        # 这里直接从 Chroma 的 sqlite 文件里做全文检索，
        # 目的是在语义检索不可用时，仍然能把文档片段取出来。
        sql = """
        SELECT
            fts.rowid AS chunk_id,
            fts.string_value AS content,
            bm25(embedding_fulltext_search) AS rank,
            (
                SELECT em.string_value
                FROM embedding_metadata AS em
                WHERE em.id = fts.rowid AND em.key = 'source'
                LIMIT 1
            ) AS source
        FROM embedding_fulltext_search AS fts
        WHERE embedding_fulltext_search MATCH ?
        ORDER BY rank
        LIMIT ?
        """

        match_query = " OR ".join(f'"{term}"' for term in terms)
        with sqlite3.connect(self.sqlite_path) as conn:
            rows = conn.execute(sql, (match_query, k)).fetchall()

        results: list[dict] = []
        for chunk_id, content, rank, source in rows:
            results.append(
                {
                    "content": content,
                    "metadata": {
                        "source": source or "unknown",
                        "chunk_id": chunk_id,
                    },
                    "score": rank,
                    "retrieval_mode": "fts",
                }
            )
        return results

    @staticmethod
    def _candidate_terms(query: str) -> list[str]:
        # FTS 对特别短的中文问句不够敏感，这里额外切 3 字滑窗，提升命中率。
        cleaned = " ".join(query.split()).strip()
        if not cleaned:
            return []

        compact = cleaned.replace(" ", "")
        terms: list[str] = []
        chunks = [part for part in cleaned.split(" ") if part]
        for chunk in chunks:
            if len(chunk) >= 3:
                terms.append(chunk)
            for idx in range(max(len(chunk) - 2, 0)):
                terms.append(chunk[idx : idx + 3])

        if len(compact) >= 3:
            for idx in range(len(compact) - 2):
                terms.append(compact[idx : idx + 3])

        if not terms and len(cleaned) >= 3:
            terms.append(cleaned)

        unique: list[str] = []
        for term in terms:
            if term not in unique:
                unique.append(term)
        return unique[: max(settings.VECTOR_TOP_K, 6)]

    @staticmethod
    def format_for_prompt(results: list[dict]) -> str:
        # 这里同样把结构化结果压平成 prompt 友好的文本块。
        if not results:
            return "未检索到文档知识库内容。"

        blocks = []
        for idx, item in enumerate(results, start=1):
            meta = item.get("metadata", {})
            source = meta.get("source", "unknown")
            mode = item.get("retrieval_mode", "unknown")
            score = item.get("score")
            score_text = "" if score is None else f" score={score:.4f}"
            blocks.append(f"[文档片段{idx}] mode={mode} source={source}{score_text}\n{item['content']}")
        return "\n\n".join(blocks)
