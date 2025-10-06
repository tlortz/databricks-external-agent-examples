"""Tests for the MCP client manager."""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from src.langgraph_mcp_agent.mcp_client import MCPClientManager, build_databricks_server_urls


@pytest.fixture
def mock_workspace_client():
    """Create a mock workspace client."""
    client = Mock()
    client.config.host = "https://test.databricks.com"
    return client


def test_build_databricks_server_urls():
    """Test building Databricks server URLs."""
    workspace_url = "https://test.databricks.com"
    paths = ["/api/2.0/mcp/functions/system/ai", "/api/2.0/mcp/genie/space1"]

    urls = build_databricks_server_urls(workspace_url, paths)

    assert len(urls) == 2
    assert urls[0] == "https://test.databricks.com/api/2.0/mcp/functions/system/ai"
    assert urls[1] == "https://test.databricks.com/api/2.0/mcp/genie/space1"


def test_build_databricks_server_urls_with_trailing_slash():
    """Test URL building with trailing slash in workspace URL."""
    workspace_url = "https://test.databricks.com/"
    paths = ["/api/2.0/mcp/functions/system/ai"]

    urls = build_databricks_server_urls(workspace_url, paths)

    assert urls[0] == "https://test.databricks.com/api/2.0/mcp/functions/system/ai"


def test_get_all_server_urls(mock_workspace_client):
    """Test getting all server URLs."""
    databricks_urls = ["https://test.databricks.com/api/2.0/mcp/functions/system/ai"]
    external_urls = ["https://custom1.com", "https://custom2.com"]
    manager = MCPClientManager(
        workspace_client=mock_workspace_client,
        databricks_server_urls=databricks_urls,
        external_server_urls=external_urls,
    )

    urls = manager.get_all_server_urls()
    assert len(urls) == 3  # 1 databricks + 2 external


def test_only_databricks_servers(mock_workspace_client):
    """Test with only Databricks servers."""
    databricks_urls = ["https://test.databricks.com/api/2.0/mcp/functions/system/ai"]
    manager = MCPClientManager(
        workspace_client=mock_workspace_client,
        databricks_server_urls=databricks_urls,
    )

    urls = manager.get_all_server_urls()
    assert len(urls) == 1
    assert urls[0] == "https://test.databricks.com/api/2.0/mcp/functions/system/ai"


@pytest.mark.asyncio
async def test_client_initialization(mock_workspace_client):
    """Test MCP client initialization with MultiServerMCPClient."""
    with patch("src.langgraph_mcp_agent.mcp_client.MultiServerMCPClient") as mock_client_class:
        # Mock the MultiServerMCPClient.get_tools() method
        mock_client = AsyncMock()
        mock_tool = AsyncMock()
        mock_tool.name = "test_tool"
        mock_client.get_tools.return_value = [mock_tool]
        mock_client_class.return_value = mock_client

        databricks_urls = ["https://test.databricks.com/api/2.0/mcp/functions/system/ai"]
        manager = MCPClientManager(
            workspace_client=mock_workspace_client,
            databricks_server_urls=databricks_urls,
        )
        tools = await manager.get_tools()

        assert tools is not None
        assert len(tools) == 1
        assert tools[0].name == "test_tool"
        mock_client_class.assert_called_once()


@pytest.mark.asyncio
async def test_context_manager(mock_workspace_client):
    """Test async context manager."""
    databricks_urls = ["https://test.databricks.com/api/2.0/mcp/functions/system/ai"]
    manager = MCPClientManager(
        workspace_client=mock_workspace_client,
        databricks_server_urls=databricks_urls,
    )

    # Test that context manager works without errors
    async with manager as mgr:
        assert mgr is not None
        assert isinstance(mgr, MCPClientManager)
