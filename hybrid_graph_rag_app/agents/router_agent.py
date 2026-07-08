from hybrid_graph_rag_app.schemas import RouteDecision


class RouterAgent:
    GRAPH_HINTS = (
        "关系",
        "简称",
        "别名",
        "外文名",
        "英文名",
        "属于",
        "位于",
        "谁",
        "哪里",
        "时间",
        "作者",
        "朝代",
        "国籍",
        "出生",
        "死亡",
    )
    OUT_OF_SCOPE_HINTS = (
        "火星",
        "殖民",
        "预算",
        "股票",
        "天气",
        "新闻",
        "比赛",
    )
    DOC_HINTS = (
        "讲了什么",
        "内容",
        "原文",
        "全文",
        "总结",
        "概述",
        "解释",
        "怎么说",
        "如何",
        "为什么",
    )

    def route(self, query: str) -> RouteDecision:
        if any(hint in query for hint in self.OUT_OF_SCOPE_HINTS):
            return RouteDecision(
                mode="insufficient_or_out_of_scope",
                graph_score=0,
                doc_score=0,
                summary="mode=insufficient_or_out_of_scope, graph_score=0, doc_score=0",
                strategies=[],
                reason="问题包含明显超出当前知识库范围的主题，优先拒答。",
            )

        graph_score = 0
        doc_score = 0
        for hint in self.GRAPH_HINTS:
            if hint in query:
                graph_score += 2
        for hint in self.DOC_HINTS:
            if hint in query:
                doc_score += 2

        if len(query) <= 10:
            graph_score += 1
        if len(query) >= 14:
            doc_score += 1

        if graph_score >= doc_score + 2:
            mode = "graph_first"
            strategies = ["graph", "document"]
            reason = "问题更像实体属性或关系查询，优先保留图谱证据。"
        elif doc_score >= graph_score + 2:
            mode = "document_first"
            strategies = ["document", "graph"]
            reason = "问题更像解释、总结或原文内容查询，优先保留文档证据。"
        else:
            mode = "hybrid"
            strategies = ["document", "graph"]
            reason = "问题同时可能需要文档片段和图谱事实，采用混合检索。"

        return RouteDecision(
            mode=mode,
            graph_score=graph_score,
            doc_score=doc_score,
            summary=f"mode={mode}, graph_score={graph_score}, doc_score={doc_score}",
            strategies=strategies,
            reason=reason,
        )
