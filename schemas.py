from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing import List, Annotated, Optional


class AgentState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages]
    answer: Optional[str] = None
