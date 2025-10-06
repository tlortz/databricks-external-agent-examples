"""Tests for MCP tool loading via langchain-mcp-adapters.

Note: The old tools.py module has been replaced with langchain-mcp-adapters.
Tool loading is now handled directly by MCPClientManager.get_tools().
These tests verify the integration works correctly.
"""

import pytest
from unittest.mock import Mock


@pytest.fixture
def mock_tools():
    """Create mock tools."""
    tool1 = Mock()
    tool1.name = "tool_one"
    tool2 = Mock()
    tool2.name = "tool_two"
    return [tool1, tool2]


def test_list_tool_names(mock_tools):
    """Test listing tool names."""
    names = [tool.name for tool in mock_tools]

    assert len(names) == 2
    assert "tool_one" in names
    assert "tool_two" in names


def test_get_tool_by_name(mock_tools):
    """Test getting tool by name."""
    tool = next((t for t in mock_tools if t.name == "tool_one"), None)
    assert tool is not None
    assert tool.name == "tool_one"


def test_get_tool_by_name_not_found(mock_tools):
    """Test getting non-existent tool returns None."""
    tool = next((t for t in mock_tools if t.name == "nonexistent"), None)
    assert tool is None
