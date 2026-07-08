from hybrid_graph_rag_app.agents.router_agent import RouterAgent


def test_router_agent_graph_first():
    route = RouterAgent().route("夏朝的外文名是什么")
    assert route.mode == "graph_first"
    assert "graph" in route.strategies


def test_router_agent_document_first():
    route = RouterAgent().route("请总结这段内容讲了什么")
    assert route.mode == "document_first"
    assert "document" in route.strategies


def test_router_agent_out_of_scope():
    route = RouterAgent().route("火星殖民计划预算是多少")
    assert route.mode == "insufficient_or_out_of_scope"
    assert route.strategies == []
