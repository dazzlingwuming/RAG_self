from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


class QueryEnhancer:
    def __init__(self, llm) -> None:
        self.rewrite_prompt = PromptTemplate.from_template(
            """根据对话历史简要完善最新用户问题，使它更适合知识库检索。
如果不需要完善，直接输出原问题。只输出一个问题。

对话历史：
{history}

用户问题：{query}
"""
        )
        self.multi_query_prompt = PromptTemplate.from_template(
            """请为下面问题生成 {query_num} 个不同检索问法，用于提升知识库召回。
要求：
- 每行一个问法
- 保留原问题中的关键实体和术语
- 不要编造具体答案

原问题：{query}
"""
        )
        self.hyde_prompt = PromptTemplate.from_template(
            """请根据下面问题生成一段“可能出现在知识库中的答案式文本”，用于向量检索。
要求只生成检索用文本，不要说明过程。

问题：{query}
"""
        )
        self.rewrite_chain = self.rewrite_prompt | llm | StrOutputParser()
        self.multi_query_chain = self.multi_query_prompt | llm | StrOutputParser()
        self.hyde_chain = self.hyde_prompt | llm | StrOutputParser()

    def expand(self, query: str, history: str, query_num: int, enable_hyde: bool) -> list[str]:
        candidates = [query]
        try:
            rewritten = self.rewrite_chain.invoke({"query": query, "history": history}).strip()
            if rewritten:
                candidates.append(rewritten)
        except Exception:
            pass

        try:
            raw = self.multi_query_chain.invoke({"query": candidates[-1], "query_num": query_num})
            candidates.extend(line.strip() for line in raw.splitlines() if line.strip())
        except Exception:
            pass

        if enable_hyde:
            try:
                hyde = self.hyde_chain.invoke({"query": candidates[-1]}).strip()
                if hyde:
                    candidates.append(hyde)
            except Exception:
                pass

        unique: list[str] = []
        for item in candidates:
            compact = " ".join(item.split())
            if compact and compact not in unique:
                unique.append(compact)
        return unique[: max(query_num + 2, 2)]
