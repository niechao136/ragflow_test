import uuid
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from typing import TypedDict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from graph import graph


class GraphState(TypedDict):
    question: str
    context: List[str]
    answer: str


class QuestionRequest(BaseModel):
    conversation_id: Optional[str] = None
    question: str
    dataset_ids: Optional[List[str]] = []
    top_k: int = 5


class AnswerResponse(BaseModel):
    question: str
    answer: str


app = FastAPI(title="LangGraph + RAGFlow API", version="1.0.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许访问的域名列表，["*"] 表示允许所有
    allow_credentials=True,  # 是否允许携带 cookie
    allow_methods=["*"],      # 允许的方法，例如 ["GET", "POST"]
    allow_headers=["*"],      # 允许的请求头
)


@app.post("/api/chat", response_model=AnswerResponse)
async def chat(request: QuestionRequest):

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    input = HumanMessage(content=request.question)
    conversation_id = request.conversation_id or str(uuid.uuid4())
    config: RunnableConfig = {
        "configurable": {
            "thread_id": conversation_id,
            "dataset_ids": request.dataset_ids,
            "top_k": request.top_k
        }
    } 
    
    
    result = await graph.ainvoke(input, config=config) # type: ignore
    
    return AnswerResponse(
        question=request.question,
        answer=result["answer"]
    )


@app.get("/api/health")
async def health():
    return {"status": "healthy"}


app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)
