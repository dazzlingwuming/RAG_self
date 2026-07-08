from hybrid_graph_rag_app.agents import (
    AgentWorkflow,
    AnswerAgent,
    EvidenceAgent,
    MemoryAgent,
    PlannerAgent,
    QueryAgent,
    ReflectionAgent,
    RetrieverAgent,
    RouterAgent,
    VerifierAgent,
)
from hybrid_graph_rag_app.graph_retriever import GraphRetriever
from hybrid_graph_rag_app.llm_support import get_embedding_model, get_llm
from hybrid_graph_rag_app.memory_manager import MemoryManager
from hybrid_graph_rag_app.neo4j_runtime import Neo4jRuntime
from hybrid_graph_rag_app.query_enhancer import QueryEnhancer
from hybrid_graph_rag_app.vector_retriever import VectorRetriever
from hybrid_graph_rag_app.verifier import AnswerVerifier


class HybridGraphRAGService:
    def __init__(self) -> None:
        self.llm = get_llm()
        self.embedding_model = get_embedding_model()
        self.vector_retriever = VectorRetriever(self.embedding_model)
        self.vector_enabled = self.vector_retriever.backend != "disabled"
        self.neo4j_runtime = Neo4jRuntime()
        self.graph_retriever = GraphRetriever(self.neo4j_runtime)
        self.memory_manager = MemoryManager()
        self.query_enhancer = QueryEnhancer(self.llm)
        self.verifier = AnswerVerifier(self.llm)
        self.workflow = AgentWorkflow(
            memory_agent=MemoryAgent(self.memory_manager),
            router_agent=RouterAgent(),
            planner_agent=PlannerAgent(),
            query_agent=QueryAgent(self.query_enhancer),
            retriever_agent=RetrieverAgent(self.vector_retriever, self.graph_retriever),
            evidence_agent=EvidenceAgent(self.vector_retriever, self.graph_retriever),
            answer_agent=AnswerAgent(self.llm),
            verifier_agent=VerifierAgent(self.verifier),
            reflection_agent=ReflectionAgent(),
            vector_enabled=self.vector_enabled,
            vector_backend=self.vector_retriever.backend,
        )

    def ask(self, query: str, session_id: str) -> dict:
        return self.workflow.run(query=query, session_id=session_id)
