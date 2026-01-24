import os
import torch
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAI

from untils.untils_rag import format_docs, load_history_from_file, build_history_from_llm_response, save_memory_to_file

load_dotenv("../config/.env")  # 加载环境变量文件 .env
api_key = os.getenv("API_KEY")  # 从环境
base_url = os.getenv("BASE_URL")  # 从环境变量获取 API 基础 URL
model_name = os.getenv("MODEL_NAME")  # 从环境变量获取模型名称

llm = OpenAI(
    model=model_name,
    api_key=api_key,
    base_url=base_url,
    temperature=0.1,
    model_kwargs={"stream_options" : True}
    )

prompt = PromptTemplate(
    input_variables=["context","question","history"],
    template="""请基于以下内容，简要回答用户的问题。如果无法从内容中找到答案，请回复“抱歉，我无法回答您的问题”。
    背景内容：{context}，
    历史记录:{history}
    用户问题：{question}”。
    回答：""",
)

#检索器相关
# 加载嵌入模型
embedding_model = HuggingFaceEmbeddings(
    model_name="../model/bge-base-zh-v1.5",
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
#获取检索器
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

#
history = load_history_from_file("小明", "../数据处理/history_save/conversation_history.json")

#构建RAG 链条
#在ICEL里面的函数必须要要以Runable接口类型
rag_chain = (
    # 第一步：组装输入（生成 context + query 字典+历史记录）
    {"context": retriever | format_docs, "question": lambda x :x, "history": lambda x: history}
    # 第二步：把第一步的字典输入给 prompt，生成格式化的提示词
    | prompt
    # 第三步：（调试用）打印 prompt 的文本，然后透传（or x 是为了不影响后续流程）
    | (lambda x: print(x.text, end="") or x)
    # 第四步：把格式化的提示词传给 LLM，生成回答
    | llm
    # 第五步：把 LLM 的输出解析为纯字符串
    | StrOutputParser()
)



if __name__ == "__main__":
    question = "失踪人重新出现,财产怎么管理?"
    response = rag_chain.invoke(question)
    print(f"\n\nRAG 模型回答：{response}")
    new_history = build_history_from_llm_response(question,response)
    save_memory_to_file(new_history, session_id="小兰", file_path="../数据处理/history_save/conversation_history.json")
    print(new_history)

