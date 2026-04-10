@echo off
setlocal

REM 这里先切回项目根目录，保证 uvicorn 按仓库根路径启动。
cd /d %~dp0..

REM 这里直接启动独立的 hybrid graph rag 应用入口。
python -m uvicorn hybrid_graph_rag_app.app:app --host 0.0.0.0 --port 8010
