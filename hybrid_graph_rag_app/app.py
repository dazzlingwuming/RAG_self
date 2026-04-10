import urllib.parse
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from hybrid_graph_rag_app.hybrid_service import HybridGraphRAGService


APP_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

app = FastAPI(title="Hybrid Graph RAG")
service = HybridGraphRAGService()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


@app.post("/api/chat")
async def chat(query: str = Form(...), session_id: str = Form("default")):
    result = service.ask(query=query, session_id=urllib.parse.unquote(session_id))
    return JSONResponse(result)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "app": "hybrid_graph_rag",
        "vector_backend": service.vector_retriever.backend,
        "graph_endpoint": "neo4j@127.0.0.1:8687",
        "routing": "enabled",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("hybrid_graph_rag_app.app:app", host="0.0.0.0", port=8010, reload=False)
