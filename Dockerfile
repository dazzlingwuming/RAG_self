FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY hybrid_graph_rag_app ./hybrid_graph_rag_app
COPY eval ./eval
COPY config/.env.example ./config/.env.example

EXPOSE 8010

CMD ["python", "-m", "uvicorn", "hybrid_graph_rag_app.app:app", "--host", "0.0.0.0", "--port", "8010"]
