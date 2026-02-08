import torch
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from torch import device
import config.configRag as configRag
from langchain_rag.langchain_base import get_llm, get_rerank_model, get_embedding_model, get_retriever, llm_chain_retrieve
from untils.untils_rag import load_history_from_file, save_memory_to_file, build_history_from_llm_response_no_think


def save_reranker_model():
    from modelscope import AutoModelForSequenceClassification, AutoTokenizer

    model_name = "BAAI/bge-reranker-base"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # 保存到本地
    save_path = "../../model"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


def load_reranker_model():
    from modelscope import AutoModelForSequenceClassification, AutoTokenizer

    model_path = "../../model/bge-reranker-base"
    model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer


#实现RRF算法来对检索结果进行重排序
def rrf_rerank(retrieve_results: list[list[Document]], k=60, top_k=10):
    '''
    倒数排序融合：将多个排序列表融合为一个最终的排序列表，优先考虑在多个列表中排名靠前的文档
    对多查询返回的多个文档列表进行排序，返回一个排序后的文档列表
    RRF(d)=∑_(r∈R)▒1/(k+rank_r (d) )
    :param retrieve_results:
    :param query:
    :param k:k是一个常数，通常设置为60，表示排名的平滑参数。较大的k值会降低排名靠前文档的权重，使得排名靠后的文档也有机会被选中。
    :param top_k:返回的最终文档数量
    :return:
    '''
    # 计算每个文档的RRF分数
    doc_scores = {}
    #去重后的文档列表，键是文档ID，值是文档内容
    unique_docs = {}
    for result in retrieve_results:
        for rank, doc in enumerate(result):
            doc_id = doc.metadata.get("id", str(doc))  # 获取文档ID，如果没有则使用文档内容作为ID
            score = 1 / (k + rank)  # 计算RRF分数
            if doc_id in doc_scores:
                doc_scores[doc_id] += score  # 累加分数
            else:
                doc_scores[doc_id] = score
            unique_docs[doc_id] = doc  # 保存文档内容
    #对文档按照RRF分数进行排序
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    #根据排序结果返回前top_k的文档
    rerank_doc = []
    #返回排名前top_k的文档
    for doc_id, _ in sorted_docs[:top_k]:
        rerank_doc.append(unique_docs[doc_id].page_content)
    return rerank_doc

#使用模型对检索结果进行重排序
def model_rerank(retrieve_results: list[list[Document]], query, model, tokenizer,device, top_k=10):
    #将多个文档列表展平为一个列表
    all_docs = []
    for result in retrieve_results:
        all_docs.extend(result)
    #对每个文档计算与查询的相关性得分
    doc_scores = []
    #批次处理文档以提高效率
    batch_size = 4
    for i in range(0, len(all_docs), batch_size):
        batch_docs = all_docs[i:i + batch_size]
        inputs = tokenizer(
            text = [query] * len(batch_docs),
            text_pair =[doc.page_content for doc in batch_docs],
            padding=True,
            max_length=tokenizer.model_max_length,
            truncation=True,#是否截断
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            #对输出的logits进行处理，得到相关性得分[batch_size,1]
            scores = outputs.logits.squeeze().tolist()
            if isinstance(scores, float):
                scores = [scores]
            doc_scores.extend(scores)
    #将文档与对应的得分进行排序
    doc_score_pairs = list(zip(all_docs, doc_scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
    #返回排序后的文档内容
    rerank_doc = [doc for doc, _ in doc_score_pairs]
    #返回排名前top_k的文档
    rerank_doc = rerank_doc[:top_k]
    return rerank_doc


def rerank_langchain(inputs:dict, llm, retriever, rerank_model, rerank_model_tokenizer, device, top_k=10,multi_query_num=5):
    """
    :param inputs:{"query": "用户的查询", "history": "对话历史"}
    :param llm:
    :param retriever:
    :param rerank_model:
    :param rerank_model_tokenizer:
    :param device:
    :param top_k:
    :return:
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
                         # | (lambda x: rrf_rerank(x, k=60, top_k=top_k))
                         | (lambda x:model_rerank(x,inputs.get("query"), rerank_model, rerank_model_tokenizer,device, top_k=top_k))  # 文档去重
                         )

    # 6、执行多查询链条，获取检索结果
    retrieve_result = multi_query_chain.invoke({"history": inputs.get("history"), "query": inputs.get("query")})

    return retrieve_result


if __name__ == "__main__":
    # model, tokenizer = load_reranker_model()
    # print(model, "\n")
    # print(tokenizer)
    #获取大模型
    llm = get_llm()
    #获取设备
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    #获取重排序模型和embedding模型
    rerank_model_path = configRag.rerank_model_path
    embedding_model_path = str(configRag.retrieval_model_path)
    rerank_model,rerank_model_tokenizer = get_rerank_model(rerank_model_path)
    embedding_model = get_embedding_model(embedding_model_path)
    #加载检索器
    retriever = get_retriever(k=20, embedding_model=embedding_model)
    #构建输入
    session_id = "小米"
    history = load_history_from_file(session_id=session_id, file_path=configRag.history_path)
    query = "国家、集体和私人依法可以出资设立有限责任公司、股份有限公司或者其他企业吗？"
    inputs = {"query": query, "history": history}
    #执行重排序
    rerank_results = rerank_langchain(inputs, llm, retriever, rerank_model, rerank_model_tokenizer, device, top_k=10,multi_query_num=5)
    print(rerank_results)
    chain = llm_chain_retrieve(llm)
    response = chain.invoke({"history": history, "query": query, "context": rerank_results})
    print(response)
    #保存历史记录
    new_Dialogue = build_history_from_llm_response_no_think(query, response)
    save_memory_to_file(session_id=session_id, memory=new_Dialogue,file_path=configRag.history_path)


    pass
