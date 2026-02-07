'''
对于检索前的处理，多查询请求，原理类似于重新生成问题，其实也就是让模型先生成多个查询，再用生成的查询去检索，然后再把检索到的内容进行整合，最后再回答用户的问题。
'''
import os
from hashlib import sha256
from typing import Dict
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.tongyi import Tongyi
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI, ChatOpenAI
from untils.untils_rag import load_history_from_file



def format_docs(docs: list[Document]) -> str:
    """格式化 docs"""
    return "\n\n".join(doc.page_content for doc in docs)


def get_retriever(k=20, embedding_model=None):
    """获取向量数据库的检索器"""

    # 1、初始化 Chroma 客户端
    vectorstore = Chroma(
        persist_directory="../../vectorstore_rag",
        embedding_function=embedding_model,
    )

    # 2、创建向量数据库检索器
    retriever = vectorstore.as_retriever(
        search_type="similarity",  # 检索方式，similarity 或 mmr
        search_kwargs={"k": k},
    )

    return retriever


def get_llm():
    # 大模型
    load_dotenv("../../config/.env")  # 加载环境变量文件 .env
    api_key = os.getenv("API_KEY")  # 从环境
    base_url = os.getenv("BASE_URL")  # 从环境变量获取 API 基础 URL
    model_name = os.getenv("MODEL_NAME")  # 从环境变量获取模型名称

    # llm = OpenAI(
    #     model=model_name,
    #     api_key=api_key,
    #     base_url=base_url,
    #     temperature=0.1,
    # )
    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0.1,
    )
    return llm
    return llm


def get_unique_docs(docs: list[list[Document]]) -> list[Document]:
    """
    文档去重函数
    去除多个文档列表中的重复文档

    Args:
        docs (list[list[Document]]): 多个文档列表

    Returns:
        list[Document]: 去重后的文档列表
    """
    seen = set()  # 用于存储已经遇到的文档内容的哈希值，用于快速检查是否重复
    unique_docs = []  # 用于存储去重后的文档
    # 遍历所有文档列表
    for sublist in docs:
        for doc in sublist:
            # 将文档内容作为唯一标识符
            # SHA256哈希函数需要字节输入，而不是字符串，所以需要encode
            # hexdigest()：将二进制哈希值转换为十六进制字符串表示，便于存储、比较和显示
            key = sha256(doc.page_content.encode("utf-8")).hexdigest()
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)
    return unique_docs


def rephrase_retrieve(input: Dict[str, str], llm, retriever,multi_query_num):
    """
    重述用户query，检索向量数据库
    这里使用多查询的优化，多传入一个参数multi_query_num表示多查询的数量
    """

    # 1、重述query的prompt
    rephrase_prompt = PromptTemplate.from_template(
        """
        根据对话历史简要完善最新的用户消息，使其更加具体。只输出完善后的问题。如果问题不需要完善，请直接输出原始问题。历史记录：{history}，用户：{query}
        """
    )

    # 2、重述链条：根据历史和当前 query 生成更具体问题
    rephrase_chain = (
            {
                "history": lambda x: x.get("history"),
                "query": lambda x: x.get("query"),
            }
            | rephrase_prompt
            | llm
            | StrOutputParser()
            | (lambda x: print(f"===== 重述后的查询: {x}=====") or x)
    )

    # ---------------检索前优化：多查询----------------------
    # 3、多查询提示模板
    multi_query_prompt = PromptTemplate.from_template(
        """
        你是一名AI语言模型助理。你的任务是生成给定问题的{query_num}个不同版本，以从矢量数据库中检索相关文档。
        你需要通过从多个视角生成问题，来克服基于距离的相似性搜索的一些局限性。请使用换行符分隔备选问题。

        原始问题：{query}
        """
    )

    # 4、扩展查询的链条
    expend_query_chain = (
            multi_query_prompt
            | llm
            | StrOutputParser()
            | (lambda x: [item.strip() for item in x.split("\n") if item.strip()])
    )

    # 5、最终多查询的链条
    multi_query_chain = (rephrase_chain
                         | (lambda x: {"query": x, "query_num": multi_query_num})  # 添加一个参数，用于控制生成多少个查询
                         | expend_query_chain  # 生成多个查询
                         | (lambda x: print(f"===== 扩展的多查询: {x}=====") or x)
                         | (lambda x: [retriever.invoke(i, k=3) for i in x])  # 遍历检索多个查询
                         | get_unique_docs  # 文档去重
                         )

    # 6、执行多查询链条，获取检索结果
    retrieve_result = multi_query_chain.invoke({"history": input.get("history"), "query": input.get("query")})

    return retrieve_result


def get_rag_chain(retrieve_result, llm):
    """使用检索结果、历史记录、用户查询，提交大模型生成回复"""

    # 1、Prompt 模板
    prompt = PromptTemplate(
        input_variables=["context", "history", "query"],
        template="""
    你是一个专业的中文问答助手，擅长基于提供的资料回答问题。
    请仅根据以下背景资料以及历史消息回答问题，如无法找到答案，请直接回答“我不知道”。

    背景资料：{context}

    历史消息：[{history}]

    问题：{query}

    回答：""",
    )

    # 2、定义 RAG 链条
    rag_chain = (
            {
                "context": lambda x: format_docs(retrieve_result),
                "history": lambda x: x.get("history"),
                "query": lambda x: x.get("query"),
            }
            | prompt
            | (lambda x: print(x.text, end="") or x)
            | llm
            | StrOutputParser()  # 输出解析器，将输出解析为字符串
    )

    return rag_chain



if __name__ == "__main__":
    from config import configRag
    history_path = configRag.history_path
    print(f"历史记录：{history_path}")
    # 获取 LLM
    llm = get_llm()
    #获取模型嵌入
    embedding_model_path = str(configRag.retrieval_model_path)
    print(f"嵌入模型：{embedding_model_path}")
    # 加载嵌入模型
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_path,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True
        },  # 输出归一化向量，更适合余弦相似度计算
    )
    # 获取检索器
    retriever = get_retriever(k=10, embedding_model=embedding_model)
    #获取历史记录
    history = load_history_from_file(session_id="小明", file_path=history_path,k=10)
    print(f"加载历史记录，共{len(history)}条")
    # 构建输入
    user_question = "失踪人重新出现,财产怎么管理?"
    input_data = {
        "history": history,
        "query": user_question
    }
    # 重述+多查询+检索
    retrieve_result = rephrase_retrieve(input_data, llm=llm, retriever = retriever,multi_query_num=3)
    print(f"检索到 {len(retrieve_result)} 条去重后的文档")

    chain = get_rag_chain(retrieve_result, llm)

    response = chain.invoke(input_data)
    print(f"\n\nRAG 模型回答：{response}")
