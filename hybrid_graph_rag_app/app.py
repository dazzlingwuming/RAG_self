import urllib.parse
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from hybrid_graph_rag_app.hybrid_service import HybridGraphRAGService


APP_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

# 这里直接把服务对象初始化成单例，后面的接口都复用这一份实例。
app = FastAPI(title="Hybrid Graph RAG")
service = HybridGraphRAGService()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # 首页只负责返回前端页面，真正的问答逻辑走 /api/chat。
    return templates.TemplateResponse("chat.html", {"request": request})


@app.post("/api/chat")
async def chat(query: str = Form(...), session_id: str = Form("default")):
    # session_id 可能来自前端表单或 URL 编码内容，这里先统一解码再进入问答流程。
    result = service.ask(query=query, session_id=urllib.parse.unquote(session_id))
    return JSONResponse(result)


@app.get("/api/health")
async def health():
    # 健康检查接口主要给排错和状态确认使用，不负责真正的业务问答。
    return {
        "status": "ok",
        "app": "hybrid_graph_rag",
        "vector_backend": service.vector_retriever.backend,
        "graph_endpoint": "neo4j@127.0.0.1:8687",
        "routing": "enabled",
    }


if __name__ == "__main__":
    import uvicorn

    # 这里保留一个直接运行入口，方便不用 bat 文件时直接本地启动。
    uvicorn.run("hybrid_graph_rag_app.app:app", host="0.0.0.0", port=8010, reload=False)
