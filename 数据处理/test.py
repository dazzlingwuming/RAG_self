from langchain_community.document_loaders import UnstructuredWordDocumentLoader

from 数据处理.数据划分 import text_split

word_documents = UnstructuredWordDocumentLoader(
    "../knowledge_base/sample.docx",
    mode="elements",
    strategy="fast",
).load()
text_split(word_documents)
