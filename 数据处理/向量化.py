import pickle
import torch
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


def hugging_model_vect(texts, embedding_model):

    # 从 Document 中取出文本
    page_content_list = [text.page_content for text in texts]
    # 进行嵌入
    embeddings = embedding_model.embed_documents(page_content_list)
    # 打印嵌入结果
    for i, (page_content, vector) in enumerate(zip(page_content_list, embeddings)):
        print("Text:\n", page_content)
        print("Embedding:\n", vector[:5])
        print()
        if i == 5:
            break

def chroma_vect(texts , embedding_model):


    # 嵌入并存储到向量数据库
    vectorstore = Chroma.from_documents(
        texts,  # 文档列表
        embedding_model,  # 嵌入模型
        persist_directory="vectorstore_rag",  # 存储路径
    )
    print(vectorstore.get().keys())  # 查看所有属性
    print(vectorstore.get(include=["embeddings"])["embeddings"][:5, :5])  # 查看嵌入向量

def chroma_add(vectorstore, documents):
    '''增加文档到已有向量库'''


    # 核心：动态添加到已有的向量库
    vectorstore.add_documents(documents=documents)

    # 验证添加结果（查看总数据量）
    total_docs = len(vectorstore.get()["ids"])
    print(f"动态添加后，向量库总文档数：{total_docs}")  # 输出：4（初始2 + 新增2）


def chroma_delete(vectorstore,ids=None):
    '''删除向量库中的文档
    删除一定要知道具体的id
    '''
    # 1. 获取要删除的文档ID
    if ids is None:
        Exception ("请提供要删除的文档ID")

    delete_id = ids

    # 2. 执行删除
    vectorstore.delete(ids=[delete_id])

    # 验证删除结果
    remaining_ids = vectorstore.get()["ids"]
    print(f"删除后剩余ID：{remaining_ids}")
    print(f"剩余文档数：{len(remaining_ids)}")

def chroma_query_id(vectorstore,text):
    '''通过问本查询向量库获得具体的id，然后通过id删除文档'''
    matched_docs = vectorstore.get(where_document={"$contains":text})
    if not matched_docs["ids"]:
        print(f"未找到包含「{text}」的内容")
        return False
        # 若匹配到多条，打印让用户选择（避免误删）
    if len(matched_docs["ids"]) > 1:
        print(f"⚠️ 找到{len(matched_docs['ids'])}条包含「{text}」的内容：")
        for idx, (doc_id, content) in enumerate(zip(matched_docs["ids"], matched_docs["documents"])):
            print(f"[{idx + 1}] ID：{doc_id} | 内容：{content[:100]}...")
    else:
        # 仅匹配到1条，直接取该ID
        target_id = matched_docs["ids"][0]
        print(f"找到匹配内容，ID：{target_id}")
        print(f"确认内容：{matched_docs['documents'][0][:100]}...")



if __name__ == "__main__":
    # 加载嵌入模型
    embedding_model = HuggingFaceEmbeddings(
        model_name="../model/bge-base-zh-v1.5",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={
            "normalize_embeddings": True
        },  # 输出归一化向量，更适合余弦相似度计算
    )
    # with open("数据分块.pkl", "rb") as f:
    #     texts = pickle.load(f)
    # print(texts)
    # chroma_vect(texts,embedding_model)

    # # 加载已有向量库
    vectorstore = Chroma(
        embedding_function=embedding_model,
        persist_directory="vectorstore_rag",
    )
    # # 实时新增的 PDF Document（比如从第2页提取的内容）
    # '''需要注意的是如果重复添加相同内容，Chroma不会自动去重，会将其作为新的文档处理，从而导致向量库中存在重复的嵌入向量和文档条目。因此，在添加新文档之前，最好先检查是否已经存在相同或相似的内容，以避免冗余数据的积累。'''
    # new_docs = [
    #     Document(
    #         page_content="兵者，诡道也。故能而示之不能，用而示之不用。",
    #         metadata={"source": "111111", "page_number": 2}
    #     ),
    #     Document(
    #         page_content="不战而屈人之兵，善之善者也。",
    #         metadata={"source": "1111111", "page_number": 3}
    #     )
    # ]
    # # chroma_add(vectorstore , new_docs)
    # # print(vectorstore.get())
    #
    #
    #查询内容获得具体的id
    query_doc = "宣告该自然人死亡应该是怎么样子才行"
    # chroma_query_id(vectorstore, query_doc)
    print(vectorstore.similarity_search(query_doc,k=3))
    #
    # # 删除向量库中的文档
    # # chroma_delete(vectorstore,ids='7fb62ae8-d8fb-4e77-8b50-62f6eec2f3e4')
    pass