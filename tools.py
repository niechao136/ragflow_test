import os
from langchain_core.tools import tool
from ragflow_sdk import RAGFlow, Chunk
from dotenv import load_dotenv
from typing import List

load_dotenv()

RAGFLOW_BASE_URL = os.getenv("RAGFLOW_BASE_URL", "http://localhost:8084")
RAGFLOW_API_KEY = os.getenv("RAGFLOW_API_KEY")
RAGFLOW_CHAT_ID = os.getenv("RAGFLOW_CHAT_ID")
RAGFLOW_DATASET_ID = os.getenv("RAGFLOW_DATASET_ID")

rag = RAGFlow(
    api_key=RAGFLOW_API_KEY,
    base_url=RAGFLOW_BASE_URL
)
dataset = rag.list_datasets(id=RAGFLOW_DATASET_ID)[0]

@tool
def ragflow_retrieve(query: str) -> str:
    """从 RAGFlow 知识库中检索相关内容"""
    chunks: List[Chunk] = rag.retrieve(
        question=query,
        dataset_ids=[dataset.id],
        top_k=5
    )
    return "\n\n".join([c.content for c in chunks])
