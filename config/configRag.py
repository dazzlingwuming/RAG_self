#参数
from pathlib import Path
#获取当前文件工作所在目录
current_dir = Path(__file__).parent.parent.resolve()

#检索生成模型位置
retrieval_model_path= current_dir/"model/bge-base-zh-v1.5"
#历史记录保存位置
history_path=current_dir/"数据处理/history_save/conversation_history.json"
#重排序模型位置
rerank_model_path=current_dir/"model/bge-reranker-base"
#向量数据库位置
persist_directory = current_dir/"vectorstore_rag"



if __name__ == "__main__":
    print(f"嵌入模型路径: {retrieval_model_path}")
    print(f"历史记录路径: {history_path}")