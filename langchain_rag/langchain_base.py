import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
import config.configRag as config
from untils.untils_rag import format_docs


#调用大模型api
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

#获取向量数据库的检索器
def get_retriever(k=20, embedding_model=None):
    """获取向量数据库的检索器"""

    # 1、初始化 Chroma 客户端
    vectorstore = Chroma(
        persist_directory=config.persist_directory,
        embedding_function=embedding_model,
    )

    # 2、创建向量数据库检索器
    retriever = vectorstore.as_retriever(
        search_type="similarity",  # 检索方式，similarity 或 mmr
        search_kwargs={"k": k},
    )

    return retriever
#获取embedding模型
def get_embedding_model(model_path):
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True
        },  # 输出归一化向量，更适合余弦相似度计算
    )
    return embedding_model

#获取重排序模型
def get_rerank_model(model_path):
    from modelscope import AutoModelForSequenceClassification, AutoTokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer

#基础问答
def llm_chain(llm):
    """
    基础问答链
    :param inputs: 包含 "context"（背景资料）-->list[Document]、"history"（历史消息）和 "query"（问题）的输入字典
    :param llm: 大模型实例
    :return: 大模型的回答字符串
    """
    #提示词
    prompt = PromptTemplate.from_template(
            '''你是一个专业的中文问答助手，擅长基于提供的资料回答问题。
    请仅根据以下背景资料以及历史消息回答问题，如无法找到答案，请直接回答“我不知道”。
    
    背景资料：{context}
    
    历史消息：[{history}]
    
    问题：{query}
    
    回答：''',

    )

    # 定义 RAG 链条
    rag_chain = (
            {
                "context": lambda x: format_docs(x.get("context")),
                "history": lambda x: x.get("history"),
                "query": lambda x: x.get("query"),
            }
            | prompt
            | (lambda x: print(x.text, end="") or x)
            | llm
            | StrOutputParser()  # 输出解析器，将输出解析为字符串
    )

    return rag_chain



if __name__ == '__main__':
    pass
