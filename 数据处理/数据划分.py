import pickle

from 数据处理.数据加载 import t6


def text_split(documents):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # 文本分块
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "。"],  # 分隔符列表，按优先级顺序使用，优先从前面的分隔符进行分割
        #例子子中使用了两个分隔符，首先尝试使用两个换行符进行分割，如果无法满足chunk_size的要求，再尝试使用句号进行分割
        chunk_size=400,  # 每个块的最大长度
        chunk_overlap=40,  # 每个块重叠的长度
    )
    texts = text_splitter.split_documents(documents)
    print(texts)
    return texts



if __name__ == "__main__":
    with open("数据.pkl", "rb") as f:
        documents = pickle.load(f)
    print(documents)
    text_split(documents)
    path = "数据分块.pkl"
    with open(path, "wb") as f:
        pickle.dump(documents, f)
