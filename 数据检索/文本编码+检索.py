# 加载嵌入模型
import chromadb
import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def chroma_base_query(text , k ):
    '''基础查询向量库'''
    # 进行相似度搜索
    results = vectorstore.similarity_search(text, k=k)

    # 输出结果
    print(f"查询内容：{text}")
    print(f"找到的Top {k} 相似内容：")
    for idx, doc in enumerate(results):
        print(f"[{idx + 1}] 内容：{doc.page_content[:200]}...")  # 只显示前200个字符

def chroma_max_marginal_query(text , k ):
    '''最大边际相关性查询向量库
    这个是在相似度搜索的基础上，进一步考虑结果之间的多样性，避免返回过于相似的内容，从而提供更丰富的信息。
    '''
    # 进行最大边际相关性搜索
    results = vectorstore.max_marginal_relevance_search(text, k=k)

    # 输出结果
    print(f"查询内容：{text}")
    print(f"找到的Top {k} 最大边际相关内容：")
    for idx, doc in enumerate(results):
        print(f"[{idx + 1}] 内容：{doc.page_content[:200]}...")  # 只显示前200个字符



#先获取检索器，再检索
def chroma_retriever_query(text , k ):
    '''通过检索器查询向量库，这里返回的是 LangChain 接口的 Document 列表'''
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    results = retriever.invoke(text)

    # 输出结果
    print(f"查询内容：{text}")
    print(f"通过检索器找到的Top {k} 相似内容：")
    for idx, doc in enumerate(results):
        print(f"[{idx + 1}] 内容：{doc.page_content[:200]}...")  # 只显示前200个字符

if __name__ == "__main__":
    # 加载嵌入模型
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
        persist_directory="../vectorstore_rag",
    )

    query_text = "宣告该自然人死亡应该是怎么样子才行"
    # print(vectorstore.similarity_search(query_text, k=3))
    # chroma_max_marginal_query(query_text, k=5)
    chroma_retriever_query(query_text, k=5)

