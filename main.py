import os
import json
import uuid
from typing import List, Optional
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path

from graph import graph
from tools import rag


load_dotenv()


FILE_PATH = Path(__file__).resolve()
ROOT_DIR = FILE_PATH.parent
DATA_DEV = ROOT_DIR / "file"
DATA_DIR = Path(os.getenv("DATA_DIR", DATA_DEV))
STATIC_DIR = ROOT_DIR / "static"


class QuestionRequest(BaseModel):
    conversation_id: Optional[str] = None
    question: str
    dataset_ids: Optional[List[str]] = []
    top_k: int = 5


class SourceDocument(BaseModel):
    document_id: str = ""
    document_name: str = ""
    score: float = 0.0
    dataset_id: str = ""


class AnswerResponse(BaseModel):
    question: str
    answer: str
    context_count: int = 0
    sources: List[SourceDocument] = []


app = FastAPI(title="LangGraph + RAGFlow API", version="1.0.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _make_config(request: QuestionRequest, conversation_id: str) -> RunnableConfig:
    return {
        "configurable": {
            "thread_id": conversation_id,
            "dataset_ids": request.dataset_ids,
            "top_k": request.top_k,
        }
    }


@app.post("/api/chat", response_model=AnswerResponse)
async def chat(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    conversation_id = request.conversation_id or str(uuid.uuid4())
    config = _make_config(request, conversation_id)

    sources = []

    async for event in graph.astream_events(
        {"messages": [HumanMessage(content=request.question)]}, # type: ignore
        config=config,
        version="v2",
    ):
        if (
            event["event"] == "on_custom_event"
            and event["name"] == "retrieved_sources"
        ):
            for s in event["data"].get("sources", []):
                sources.append(SourceDocument(**s))

    result = await graph.aget_state(config)

    last_message = result.values["messages"][-1]
    answer = last_message.content if hasattr(last_message, 'content') else str(last_message)

    return AnswerResponse(
        question=request.question,
        answer=answer,
        sources=sources,
        context_count=len(sources),
    )


@app.post("/api/chat/stream")
async def chat_stream(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    conversation_id = request.conversation_id or str(uuid.uuid4())
    config = _make_config(request, conversation_id)

    async def event_generator():
        # 先告知前端 conversation_id
        yield f"data: {json.dumps({'type': 'meta', 'conversation_id': conversation_id})}\n\n"

        async for event in graph.astream_events(
            {"messages": [HumanMessage(content=request.question)]}, # type: ignore
            config=config,
            version="v2",
        ):
            kind = event["event"]

            # LLM 输出 token
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"].content
                if chunk:
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"

            # 检索到的文档来源
            elif kind == "on_custom_event" and event["name"] == "retrieved_sources":
                sources = event["data"].get("sources", [])
                yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 关闭 nginx 缓冲，避免流式卡顿
        },
    )


@app.get("/api/health")
async def health():
    ragflow_connected = rag is not None
    return {"status": "healthy", "ragflow_connected": ragflow_connected}


app.mount("/file", StaticFiles(directory=str(DATA_DIR)), name="file")
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)