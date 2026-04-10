import os
import io
import contextlib

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from hybrid_graph_rag_app import settings


def get_llm() -> ChatOpenAI:
    load_dotenv(settings.DOTENV_PATH)
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME"),
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
        temperature=0.1,
    )


def get_embedding_model() -> HuggingFaceEmbeddings:
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            return HuggingFaceEmbeddings(
                model_name=str(settings.EMBEDDING_MODEL_PATH),
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
    except Exception:
        return None
