import asyncio
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_rag.langchain_base import get_llm, get_retriever, llm_chain_retrieve, llm_chain_unretrieve
from langchain_rag.retrieve_rag.HyDE import rephrase_retrieve
from untils.untils_rag import load_history_from_file, build_history_from_llm_response_no_think, save_memory_to_file
import config.configRag as config


# 1、初始化Embedding模型
model_path = str(config.retrieval_model_path)
embedding_model = HuggingFaceEmbeddings(
    model_name=model_path,
    model_kwargs={"device": "cpu"},
    encode_kwargs={
        "normalize_embeddings": True
    },  # 输出归一化向量，更适合余弦相似度计算
)

# 2、初始化 LLM
llm = get_llm()
#使用HyDE方法的RAG链条，重述+多查询+检索
async def invoke_rag(query,session_id):

    answer = ""
    # 对话历史
    chat_history = load_history_from_file(session_id=session_id, file_path=config.history_path, k=10)
    input = {"query": query, "history": chat_history}
    '''# 1、获取检索器
    retriever=get_retriever(k=20,embedding_model=embedding_model)
    # 2、执行重述、检索，看具体使用哪一个检索增强方法
    retrieve_result= rephrase_retrieve(input,llm,retriever)
    input["context"]=retrieve_result
    # 3、获取RAG链
    rag_chain = llm_chain_retrieve(llm)'''
    rag_chain = llm_chain_unretrieve(llm)
    # 4、异步执行RAG链，流式输出
    async for chunk in rag_chain.astream(input):
        answer += chunk
        yield chunk # 将大模型生成的内容逐块(chunk)地返回给调用者，而不是等待整个回答完成后一次性返回

    # 5、更新对话历史，添加用户查询和AI回答
    # 保存历史记录
    new_Dialogue = build_history_from_llm_response_no_think(query, answer)
    save_memory_to_file(session_id=session_id, memory=new_Dialogue, file_path=config.history_path)


if __name__ == '__main__':
    async def main():
        query_list = ["你知道公司法第22条是怎么规定的吗？",]
        for query in query_list:
            print(f"===== 查询: {query} =====")
            async for chunk in invoke_rag(query,"小天"):
                print(chunk, end="", flush=True)

    asyncio.run(main())
