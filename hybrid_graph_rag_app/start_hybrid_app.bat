@echo off
setlocal

cd /d %~dp0..
python -m uvicorn hybrid_graph_rag_app.app:app --host 0.0.0.0 --port 8010
