import uuid
from typing import List, Optional
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from graph import graph
from tools import rag


class QuestionRequest(BaseModel):
    conversation_id: Optional[str] = None
    question: str
    dataset_ids: Optional[List[str]] = []
    top_k: int = 5


class AnswerResponse(BaseModel):
    question: str
    answer: str
    context_count: int = 0


app = FastAPI(title="LangGraph + RAGFlow API", version="1.0.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/chat", response_model=AnswerResponse)
async def chat(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    input_msg = HumanMessage(content=request.question)
    conversation_id = request.conversation_id or str(uuid.uuid4())
    config: RunnableConfig = {
        "configurable": {
            "thread_id": conversation_id,
            "dataset_ids": request.dataset_ids,
            "top_k": request.top_k
        }
    }

    result = await graph.ainvoke({"messages": [input_msg]}, config=config) # type: ignore

    last_message = result["messages"][-1]
    answer = last_message.content if hasattr(last_message, 'content') else str(last_message)

    return AnswerResponse(
        question=request.question,
        answer=answer,
        context_count=0
    )


@app.get("/api/health")
async def health():
    ragflow_connected = rag is not None
    return {"status": "healthy", "ragflow_connected": ragflow_connected}


app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)