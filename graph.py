from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from llm import llm
from prompts import SYSTEM_PROMPT
from schemas import AgentState
from tools import ragflow_retrieve


tool = [ragflow_retrieve]
llm_with_tools = llm.bind_tools(tool)


tool_node = ToolNode(tools=tool)


async def call_agent(state: AgentState, config: RunnableConfig):
    """调用智能体"""
    sys_msg = SystemMessage(content=SYSTEM_PROMPT)
    response = await llm_with_tools.ainvoke([sys_msg] + state.messages, config)
    return AgentState(messages=[response])


def should_continue(state: AgentState):
    """
    判断逻辑：
    如果模型最后一条消息包含 tool_calls，则跳转到 tools 节点；
    否则结束会话。
    """
    messages = state.messages
    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "end"


graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", call_agent)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})

graph = graph_builder.compile()
