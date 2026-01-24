# 加载嵌入模型
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="D:/github/RAG_self/model/bge-base-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={
        "normalize_embeddings": True
    },  # 输出归一化向量，更适合余弦相似度计算
)
# 加载已有向量库
vectorstore = Chroma(
    embedding_function=embedding_model,
    persist_directory="D:/github/RAG_self/数据处理/vectorstore_rag",
)

query_text = "宣告该自然人死亡应该是怎么样子才行"
print(vectorstore.similarity_search(query_text, k=3))
# chroma_max_marginal_query(query_text, k=5)
# chroma_retriever_query(query_text, k=5)
