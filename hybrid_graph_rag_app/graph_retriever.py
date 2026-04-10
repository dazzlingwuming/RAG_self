import re
import time
from typing import Any

import requests

from hybrid_graph_rag_app import settings
from hybrid_graph_rag_app.neo4j_runtime import Neo4jRuntime


class GraphRetriever:
    def __init__(self, runtime: Neo4jRuntime):
        self.runtime = runtime

    def _run_query(self, statement: str, parameters: dict[str, Any]) -> list[dict]:
        last_error = None
        # 这里给图谱请求留一次重试机会，主要是为了应对本地 Neo4j 偶发连接抖动。
        for _ in range(2):
            try:
                self.runtime.ensure_started()
                response = requests.post(
                    f"{settings.NEO4J_HTTP_URL}/db/neo4j/query/v2",
                    json={"statement": statement, "parameters": parameters},
                    timeout=20,
                )
                response.raise_for_status()
                payload = response.json()
                data = payload.get("data", {})
                fields = data.get("fields", [])
                values = data.get("values", [])
                return [dict(zip(fields, row)) for row in values]
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(1)
        if last_error is not None:
            raise last_error
        return []

    @staticmethod
    def extract_keywords(query: str) -> list[str]:
        # 这里先把问句里的常见标点和口语尾词清掉，尽量保留实体和关系词。
        normalized = re.sub(r"[，。！？、,.!?;:：\"'“”‘’（）()\[\]【】\s]+", " ", query).strip()
        stop_phrases = [
            "是什么",
            "是啥",
            "什么",
            "有哪些",
            "有啥",
            "多少",
            "请问",
            "一下",
            "告诉我",
            "介绍下",
            "介绍一下",
            "吗",
            "呢",
            "啊",
            "呀",
        ]

        for phrase in stop_phrases:
            normalized = normalized.replace(phrase, " ")

        pieces = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", normalized)
        # “X的Y是什么” 这类问句里，按 “的” 再拆一层，更容易拿到核心实体。
        for part in normalized.split("的"):
            part = part.strip()
            if len(part) >= 2:
                pieces.append(part)
        pieces.sort(key=len, reverse=True)

        unique: list[str] = []
        for piece in pieces:
            if piece not in unique:
                unique.append(piece)
        return unique[: settings.GRAPH_KEYWORD_LIMIT]

    def search(self, query: str) -> list[dict]:
        keywords = self.extract_keywords(query)
        if not keywords:
            return []

        # 这里的打分不是严格的召回模型，而是一个轻量排序规则：
        # 实体完全命中优先，其次是包含命中，再其次才是简介字段命中。
        # 如果问题里直接出现了关系名，比如“外文名”，也额外加分。
        statement = """
        MATCH (n)-[r]->(m)
        WITH
          n,
          r,
          m,
          (
          CASE
            WHEN coalesce(n.entity_name, '') = $keyword THEN 100
            WHEN coalesce(m.entity_name, '') = $keyword THEN 95
            WHEN coalesce(n.entity_name, '') CONTAINS $keyword THEN 80
            WHEN coalesce(m.entity_name, '') CONTAINS $keyword THEN 75
            WHEN coalesce(n.entity_profile, '') CONTAINS $keyword THEN 60
            WHEN coalesce(m.entity_profile, '') CONTAINS $keyword THEN 55
            ELSE 0
          END +
          CASE
            WHEN $query_text CONTAINS type(r) THEN 40
            ELSE 0
          END
          ) AS score
        WHERE score > 0
        RETURN
          n.entity_name AS source_name,
          n.entity_profile AS source_profile,
          type(r) AS rel_type,
          m.entity_name AS target_name,
          m.entity_profile AS target_profile,
          score
        ORDER BY score DESC
        LIMIT $limit
        """

        merged: list[dict] = []
        seen: set[str] = set()
        for keyword in keywords:
            rows = self._run_query(
                statement,
                {"keyword": keyword, "query_text": query, "limit": settings.GRAPH_RESULT_LIMIT},
            )
            # 多个关键词会查出重复边，这里做一次去重，避免前端和 prompt 里出现重复证据。
            for row in rows:
                row_key = str(row)
                if row_key not in seen:
                    seen.add(row_key)
                    row["keyword"] = keyword
                    merged.append(row)
        merged.sort(key=lambda item: item.get("score", 0), reverse=True)
        return merged

    @staticmethod
    def format_for_prompt(results: list[dict]) -> str:
        # 这里把结构化图谱结果转成紧凑文本，方便直接放进 prompt。
        if not results:
            return "未检索到图谱关系。"

        lines = []
        for idx, item in enumerate(results[: settings.GRAPH_RESULT_LIMIT], start=1):
            source_name = item.get("source_name") or "未知实体"
            rel_type = item.get("rel_type") or "相关"
            target_name = item.get("target_name") or "未知值"
            score = item.get("score")
            score_text = "" if score is None else f" score={score}"
            lines.append(f"[图谱事实{idx}] {source_name} -[{rel_type}]- {target_name}{score_text}")
        return "\n".join(lines)
