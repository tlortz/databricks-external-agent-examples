"""Tests for the agent module."""

import pytest
from unittest.mock import Mock, AsyncMock
from langchain_core.messages import AIMessage, HumanMessage

from src.langgraph_mcp_agent.agent import create_tool_calling_agent, AgentState


@pytest.fixture
def mock_model():
    """Create a mock language model."""
    model = Mock()
    model.bind_tools = Mock(return_value=model)
    return model


@pytest.fixture
def mock_tools():
    """Create mock tools."""
    from langchain_core.tools import tool

    @tool
    def test_tool(query: str) -> str:
        """A test tool."""
        return "test result"

    return [test_tool]


def test_agent_state_structure():
    """Test AgentState structure."""
    state = AgentState(
        messages=[HumanMessage(content="test")],
        custom_inputs={"key": "value"},
        custom_outputs=None,
    )
    assert len(state["messages"]) == 1
    assert state["custom_inputs"]["key"] == "value"
    assert state["custom_outputs"] is None


def test_create_tool_calling_agent(mock_model, mock_tools):
    """Test agent creation."""
    agent = create_tool_calling_agent(
        model=mock_model,
        tools=mock_tools,
        system_prompt="Test prompt",
    )

    assert agent is not None
    mock_model.bind_tools.assert_called_once_with(mock_tools)


@pytest.mark.asyncio
async def test_agent_execution(mock_model, mock_tools):
    """Test agent execution flow."""
    # Mock the model response
    mock_response = AIMessage(content="Test response", tool_calls=[])
    mock_model.invoke = Mock(return_value=mock_response)

    agent = create_tool_calling_agent(
        model=mock_model,
        tools=mock_tools,
    )

    result = await agent.ainvoke({
        "messages": [HumanMessage(content="Test query")]
    })

    assert "messages" in result
    assert len(result["messages"]) > 0
