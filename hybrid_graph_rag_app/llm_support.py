import contextlib
import io
import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from hybrid_graph_rag_app import settings


def get_llm() -> ChatOpenAI:
    # 这里从 .env 读取模型相关配置，避免把 key 和地址硬编码进代码。
    load_dotenv(settings.DOTENV_PATH)
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME"),
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
        temperature=0.1,
    )


def get_embedding_model() -> HuggingFaceEmbeddings:
    try:
        # 当前机器上 embedding 依赖有兼容问题，这里把底层报错先吞掉，
        # 让上层自动走降级分支，而不是启动阶段直接失败。
        with contextlib.redirect_stderr(io.StringIO()):
            return HuggingFaceEmbeddings(
                model_name=str(settings.EMBEDDING_MODEL_PATH),
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
    except Exception:
        return None
