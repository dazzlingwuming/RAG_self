import pickle


def t1():
    """
    TextLoader会将一个文档文件加载为一个Document对象，该对象有两个属性：
        metadata:       存储该文档的来源路径等元数据
        page_content:   存储文档的内容
    """

    from langchain_community.document_loaders import TextLoader

    text_documents = TextLoader("../knowledge_base/sample.txt", encoding="utf-8").load()
    print(text_documents)
    return text_documents


def t2():
    """
    UnstructuredMarkdownLoader可用于加载Markdown文件
        mode: 加载模式
            "single"    返回单个Document对象，一般采用，因为后续的文本切分器会对文档进行切分，
            "elements"  按标题等元素切分文档，这个切分会非常细，有时候几个字就会被切分为一个Document对象
        strategy: 加载策略
            "fast"      快速粗粒度加载 ,一般用于预览文档内容
            "hi_res"    细粒度加载，按标题层级、列表项、表格等结构细分 ，一般用于pdf等复杂文档
    """

    from langchain_community.document_loaders import UnstructuredMarkdownLoader

    md_documents = UnstructuredMarkdownLoader(
        "../knowledge_base/sample.md",
        mode="elements",
        strategy="fast",
    ).load()
    print(md_documents)
    return md_documents

def t3():
    """
    PyPDFLoader
        支持页级拆分，每一页作为一个Document返回
        支持提取图像、提取布局
        extraction_mode: 提取模式
            "plain"     提取纯文本
            "layout"    提取布局
    """

    from langchain_community.document_loaders import PyPDFLoader

    pdf_documents = PyPDFLoader(
        "../knowledge_base/sample.pdf",
        extraction_mode="plain",
    ).load()
    print(pdf_documents)
    return pdf_documents

#使用
def t4():
    """
    UnstructuredPDFLoader
        支持结构化提取，支持OCR
        仅当 PDF 文档中不存在文本时，才会应用 OCR
        mode: 加载模式
            "single"    返回单个Document对象
            "elements"  按标题等元素切分文档
        strategy: 加载策略
            "fast"      提取并处理文本
            "ocr_only"  先进行 OCR 处理，再处理原始文本
            "hi_res"    识别文档布局并处理，自动下载YOLOX模型（识别页面布局）
        infer_table_structure: 是否推断表格结构
            仅 hi_res 下起效
            如果为 True，提取出的表格元素会包含一个 text_as_html 元数据，将文本内容转换为 html 格式
        languages: OCR使用的语言
            需传入语言列表
            语言列表参考 https://github.com/tesseract-ocr/langdata
        更多参数详见 https://github.com/Unstructured-IO/unstructured/blob/main/unstructured/partition/pdf.py
    """

    from langchain_community.document_loaders import UnstructuredPDFLoader

    pdf_documents = UnstructuredPDFLoader(
        "D:/lihaodong/Documents/Doc1.pdf",
        mode="elements",
        strategy="hi_res",
        infer_table_structure=True,
        languages=["eng", "chi_sim"],#注意这里的ocr只能识别英文
    ).load()
    print(pdf_documents)
    return pdf_documents

def t5():
    """
    UnstructuredWordDocumentLoader
        适用于 .docx 和 .doc 文件
        mode: 加载模式
            "single"    返回单个Document对象
            "elements"  按标题等元素切分文档
        strategy: 加载策略
            "fast"      快速粗粒度加载
            "hi_res"    细粒度加载，按结构细分
    """

    from langchain_community.document_loaders import UnstructuredWordDocumentLoader

    word_documents = UnstructuredWordDocumentLoader(
        "../knowledge_base/sample.docx",
        mode="elements",
        strategy="fast",
    ).load()
    print(word_documents)
    return word_documents

def t6(documents):
    import re
    import json

    def clean_content(documents: list):

        """文本清洗"""
        cleaned_docs = []

        for doc in documents:

            # 1、page_content处理：去除多余换行和空格
            text = doc.page_content

            # 将连续的换行符替换为两个换行符，正则表达式模式：r"\n{2,}"
            # r"" 表示原始字符串（raw string），避免转义字符的特殊处理
            # \n 表示换行符
            # {2,} 是量词，表示前面的字符（换行符）出现 2 次或更多次
            text = re.sub(r"\n{2,}", "\n\n", text)
            text = text.strip()

            # 2、metadata处理：将所有非 Chroma 支持类型的值转为 JSON 格式字符串
            allowed_types = (str, int, float, bool)
            for key, value in doc.metadata.items():
                if not isinstance(value, allowed_types):
                    try:
                        doc.metadata[key] = json.dumps(value, ensure_ascii=False)
                    except (TypeError, ValueError):
                        # 如果 json.dumps 失败（如包含不可序列化对象），转为 str
                        doc.metadata[key] = str(value)

            # 3、更新文档内容
            doc.page_content = text
            cleaned_docs.append(doc)

        return cleaned_docs
    return clean_content(documents)

def t7():
    con1 = t1()
    con2 = t2()
    # t3()
    con3 = t4()
    con = con1 + con2 + con3
    clean_docs = t5(con)

    return clean_docs


if __name__ == "__main__":
    con1 = t1()
    con2 = t2()
    # t3()
    con3 = t4()
    con4 = t5()
    con = con1 + con2 + con3 + con4
    clean_docs = t6(con)
    for doc in clean_docs:
        print(doc)
    path = "数据.pkl"
    with open(path, "wb") as f:
        pickle.dump(con, f)



