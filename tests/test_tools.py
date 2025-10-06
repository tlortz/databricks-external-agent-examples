"""Tests for the tools module."""

import pytest
from unittest.mock import Mock, AsyncMock

from src.langgraph_mcp_agent.tools import get_mcp_tools, list_tool_names, get_tool_by_name


@pytest.fixture
def mock_tools():
    """Create mock tools."""
    tool1 = Mock()
    tool1.name = "tool_one"
    tool2 = Mock()
    tool2.name = "tool_two"
    return [tool1, tool2]


@pytest.mark.asyncio
async def test_get_mcp_tools():
    """Test getting tools from MCP client."""
    mock_client = AsyncMock()

    # Create mock MCP tools with proper structure
    mock_mcp_tool = Mock()
    mock_mcp_tool.name = "test_tool"
    mock_mcp_tool.description = "A test tool"
    mock_mcp_tool.inputSchema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The query string"
            }
        },
        "required": ["query"]
    }

    mock_client._get_tools_async.return_value = [mock_mcp_tool]
    mock_client.call_tool = AsyncMock(return_value="test result")

    tools = await get_mcp_tools(mock_client)

    assert len(tools) == 1
    assert tools[0].name == "test_tool"
    assert tools[0].description == "A test tool"
    mock_client._get_tools_async.assert_called_once()


def test_list_tool_names(mock_tools):
    """Test listing tool names."""
    names = list_tool_names(mock_tools)

    assert len(names) == 2
    assert "tool_one" in names
    assert "tool_two" in names


def test_get_tool_by_name(mock_tools):
    """Test getting tool by name."""
    tool = get_tool_by_name(mock_tools, "tool_one")
    assert tool.name == "tool_one"


def test_get_tool_by_name_not_found(mock_tools):
    """Test getting non-existent tool raises error."""
    with pytest.raises(ValueError) as exc_info:
        get_tool_by_name(mock_tools, "nonexistent")

    assert "not found" in str(exc_info.value)
    assert "tool_one" in str(exc_info.value)
    assert "tool_two" in str(exc_info.value)
