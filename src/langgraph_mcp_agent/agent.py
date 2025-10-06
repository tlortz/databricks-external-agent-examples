"""LangGraph agent implementation with tool calling support."""

from typing import Annotated, Any, Optional, Sequence, Union
from typing_extensions import TypedDict

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State for the agent graph."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    custom_inputs: Optional[dict[str, Any]]
    custom_outputs: Optional[dict[str, Any]]


def create_tool_calling_agent(
    model: LanguageModelLike,
    tools: Union[ToolNode, Sequence[BaseTool]],
    system_prompt: Optional[str] = None,
):
    """
    Create a LangGraph agent with tool calling capabilities.

    Args:
        model: The language model to use
        tools: Tools available to the agent
        system_prompt: Optional system prompt to prepend to messages

    Returns:
        Compiled LangGraph agent
    """
    # Bind tools to the model
    model_with_tools = model.bind_tools(tools)

    # Create tool node
    tool_node = ToolNode(tools) if not isinstance(tools, ToolNode) else tools

    def should_continue(state: AgentState):
        """Determine whether to continue to tools or end."""
        messages = state["messages"]
        last_message = messages[-1]

        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "continue"
        return "end"

    def call_model(state: AgentState, config: RunnableConfig):
        """Call the model with the current state."""
        messages = state["messages"]

        # Add system prompt if provided
        if system_prompt:
            from langchain_core.messages import SystemMessage
            messages = [SystemMessage(content=system_prompt)] + list(messages)

        response = model_with_tools.invoke(messages, config)
        return {"messages": [response]}

    # Build the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    # Set entry point
    workflow.set_entry_point("agent")

    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )

    # Add edge from tools back to agent
    workflow.add_edge("tools", "agent")

    # Compile and return
    return workflow.compile()
