import os
from typing import List, Optional
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from ragflow_sdk import RAGFlow, Chunk
from dotenv import load_dotenv

load_dotenv()

RAGFLOW_BASE_URL = os.getenv("RAGFLOW_BASE_URL", "http://localhost:8084")
RAGFLOW_API_KEY = os.getenv("RAGFLOW_API_KEY")
RAGFLOW_DATASET_ID = os.getenv("RAGFLOW_DATASET_ID")

rag = RAGFlow(
    api_key=RAGFLOW_API_KEY,
    base_url=RAGFLOW_BASE_URL
)


def get_dataset_ids(config: Optional[RunnableConfig] = None) -> List[str]:
    if config and "configurable" in config:
        dataset_ids = config["configurable"].get("dataset_ids", [])
        if dataset_ids:
            return dataset_ids
    if RAGFLOW_DATASET_ID:
        return [RAGFLOW_DATASET_ID]
    return []


def get_top_k(config: Optional[RunnableConfig] = None) -> int:
    if config and "configurable" in config:
        return config["configurable"].get("top_k", 5)
    return 5


@tool
def ragflow_retrieve(query: str, config: Optional[RunnableConfig] = None) -> str:
    """从 RAGFlow 知识库中检索相关内容"""
    
    dataset_ids = get_dataset_ids(config)
    top_k = get_top_k(config)
    
    if not dataset_ids:
        return "未配置数据集，请设置 RAGFLOW_DATASET_ID 环境变量或通过 API 参数传递 dataset_ids"
    
    chunks: List[Chunk] = rag.retrieve(
        question=query,
        dataset_ids=dataset_ids,
        top_k=top_k
    )
    return "\n\n".join([c.content for c in chunks])