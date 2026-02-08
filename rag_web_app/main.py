import urllib.parse

from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse,JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import asyncio
import uuid
from typing import Optional
import json
import config.configRag as config
# 导入你的 RAG 模块
from rag_chain_fastapi_base import invoke_rag
from untils.untils_rag import load_history_from_file

app = FastAPI(title="RAG Chat Assistant")

# 配置静态文件和模板
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# 首页/登录页面
@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


# 登录处理
@app.post("/login")
async def login(request: Request):
    form_data = await request.form()
    username = form_data.get("username", "").strip()

    if not username:
        raise HTTPException(status_code=400, detail="用户名不能为空")

    # 直接重定向到聊天页面
    encoded_username = urllib.parse.quote(username)
    return JSONResponse({
        "session_id": username,
        "username": username,
        "redirect": f"/chat?session_id={encoded_username}"
    })


# 聊天页面
@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request, session_id: Optional[str] = None):
    if not session_id:
        # 如果没有session_id，重定向到登录页
        return templates.TemplateResponse("login.html", {"request": request})

    # 解码session_id
    decoded_session_id = urllib.parse.unquote(session_id)

    return templates.TemplateResponse("chat.html", {
        "request": request,
        "username": decoded_session_id,
        "session_id": decoded_session_id
    })


# 聊天API接口（流式响应） - 修复参数接收方式
@app.post("/api/chat")
async def chat_stream(request: Request):
    try:
        form_data = await request.form()
        session_id = form_data.get("session_id")
        query = form_data.get("query")

        if not session_id:
            raise HTTPException(status_code=400, detail="缺少session_id")
        if not query:
            raise HTTPException(status_code=400, detail="请输入问题")

        # 解码session_id
        decoded_session_id = urllib.parse.unquote(session_id)

        # 创建流式响应
        async def generate():
            full_response = ""
            try:
                # 调用你的RAG链条
                async for chunk in invoke_rag(query, decoded_session_id):
                    if chunk:
                        full_response += chunk
                        # 将每个chunk作为SSE格式发送
                        yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            except Exception as e:
                error_msg = f"发生错误: {str(e)}"
                print(f"RAG处理错误: {e}")  # 打印错误信息到控制台
                yield f"data: {json.dumps({'error': error_msg})}\n\n"
            finally:
                yield f"data: {json.dumps({'done': True})}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")


# 获取历史消息 - 从文件读取
@app.get("/api/history")
async def get_history(session_id: str):
    try:
        # 解码session_id
        decoded_session_id = urllib.parse.unquote(session_id)

        # 从文件加载历史记录
        raw_history = load_history_from_file(
            session_id=decoded_session_id,
            file_path=config.history_path,
            k=10
        )

        if not raw_history:
            return {"messages": []}

        # 转换历史记录格式，以匹配前端期望的格式
        messages = []
        for item in raw_history:
            if "human" in item:
                messages.append({
                    "role": "user",
                    "content": item["human"]
                })
            elif "ai" in item:
                messages.append({
                    "role": "assistant",
                    "content": item["ai"]
                })

        return {"messages": messages}
    except Exception as e:
        print(f"获取历史记录错误: {e}")
        return {"messages": []}


# 清除历史 - 清除文件中的历史记录
@app.post("/api/clear")
async def clear_history(request: Request):
    try:
        form_data = await request.form()
        session_id = form_data.get("session_id")

        if not session_id:
            raise HTTPException(status_code=400, detail="缺少session_id")

        # 解码session_id
        decoded_session_id = urllib.parse.unquote(session_id)

        # 清除文件中的历史记录
        import os
        import json as json_module

        file_path = config.history_path

        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json_module.load(f)

            # 移除该session_id的历史记录
            if decoded_session_id in data:
                data.pop(decoded_session_id)

                # 写回文件
                with open(file_path, "w", encoding="utf-8") as f:
                    json_module.dump(data, f, ensure_ascii=False, indent=2)

        return {"status": "success", "message": "历史记录已清除"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清除历史失败: {str(e)}")


# 登出 - 只是重定向，不需要删除文件中的历史记录
@app.post("/api/logout")
async def logout():
    return {"status": "success", "redirect": "/"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)