"""Integration tests for MCP client connectivity to Databricks."""

import pytest
from src.langgraph_mcp_agent.mcp_client import (
    MCPClientManager,
    build_databricks_server_urls,
    parse_server_list_from_env,
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_client_connects_to_databricks(workspace_client, workspace_url, databricks_server_paths):
    """Test that MCP client can connect to Databricks workspace."""
    # Build full server URLs
    databricks_servers = build_databricks_server_urls(workspace_url, databricks_server_paths)

    # Create MCP client manager
    manager = MCPClientManager(
        workspace_client=workspace_client,
        databricks_server_urls=databricks_servers,
    )

    # Test connection
    try:
        tools = await manager.get_tools()
        assert tools is not None, "Tools should be retrieved"
        assert len(tools) > 0, "Should retrieve at least one tool"
    finally:
        await manager.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_client_retrieves_tools(workspace_client, workspace_url, databricks_server_paths):
    """Test that MCP client can retrieve tools from Databricks servers."""
    # Build full server URLs
    databricks_servers = build_databricks_server_urls(workspace_url, databricks_server_paths)

    # Create MCP client manager
    manager = MCPClientManager(
        workspace_client=workspace_client,
        databricks_server_urls=databricks_servers,
    )

    try:
        # Get tools
        tools = await manager.get_tools()

        # Verify tools were retrieved
        assert tools is not None, "Tools should be retrieved"
        assert len(tools) > 0, f"Should retrieve at least one tool from servers: {databricks_servers}"

        # Log available tools
        tool_names = [tool.name for tool in tools]
        print(f"\nRetrieved {len(tools)} tools from {len(databricks_servers)} servers: {tool_names}")

    finally:
        await manager.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workspace_client_authentication(workspace_client):
    """Test that workspace client is properly authenticated."""
    # Try to get current user info to verify authentication
    current_user = workspace_client.current_user.me()

    assert current_user is not None, "Should retrieve current user"
    assert current_user.user_name is not None, "Current user should have a username"

    print(f"\nAuthenticated as: {current_user.user_name}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_servers_are_accessible(workspace_client, workspace_url, databricks_server_paths):
    """Test that configured MCP server URLs are accessible."""
    # Build full server URLs
    databricks_servers = build_databricks_server_urls(workspace_url, databricks_server_paths)

    print(f"\nTesting access to {len(databricks_servers)} MCP servers:")
    for server in databricks_servers:
        print(f"  - {server}")

    # Create MCP client manager
    manager = MCPClientManager(
        workspace_client=workspace_client,
        databricks_server_urls=databricks_servers,
    )

    try:
        # Try to get tools to verify servers are responding
        tools = await manager.get_tools()

        # Verify tools were retrieved
        assert tools is not None, "Tools should be retrieved"
        assert len(tools) > 0, "Should retrieve at least one tool"

        # Report results
        print(f"\nSuccessfully connected to {len(databricks_servers)} servers, retrieved {len(tools)} tools")

    except Exception as e:
        pytest.fail(f"Failed to connect to MCP servers: {e}")
    finally:
        await manager.close()
