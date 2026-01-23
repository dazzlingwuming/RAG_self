import os
import torch
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

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
    input_variables=["question"],
    template="""请基于以下内容，简要回答用户的问题。如果无法从内容中找到答案，请回复“抱歉，我无法回答您的问题”。
    背景内容：
    {context}
    用户问题：{question}”。
    回答：""",
)



if __name__ == "__main__":
    prompt = "请简要介绍一下人工智能的发展历程。"
    response = llm.invoke(prompt)
    print("模型回答：", response)

