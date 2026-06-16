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
    sys_msg = SystemMessage(content=SYSTEM_PROMPT)
    response = await llm_with_tools.ainvoke([sys_msg] + state.messages, config)
    return {"messages": [response]}


def should_continue(state: AgentState):
    messages = state.messages
    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", call_agent)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "agent")

graph = graph_builder.compile()