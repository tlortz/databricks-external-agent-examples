"""Integration tests specifically for Genie MCP server."""

import pytest
from src.langgraph_mcp_agent.mcp_client import (
    MCPClientManager,
    build_databricks_server_urls,
)
from src.langgraph_mcp_agent.tools import get_mcp_tools


@pytest.mark.integration
@pytest.mark.asyncio
async def test_genie_mcp_server_connectivity(workspace_client, workspace_url):
    """Test that we can connect to Genie MCP server."""
    # Use only the Genie server
    genie_path = "/api/2.0/mcp/genie/01f07c8be026197b989df6c647fe6970"
    genie_url = build_databricks_server_urls(workspace_url, [genie_path])

    manager = MCPClientManager(
        workspace_client=workspace_client,
        databricks_server_urls=genie_url,
    )

    try:
        clients = await manager.get_clients()
        assert len(clients) == 1, "Should have one Genie MCP client"

        # Get tools from Genie
        tools = await get_mcp_tools(clients)

        print(f"\n=== Genie MCP Server Tools ===")
        print(f"Number of tools: {len(tools)}")

        for tool in tools:
            print(f"\nTool: {tool.name}")
            print(f"Description: {tool.description}")
            print(f"Args schema: {tool.args_schema.model_json_schema() if tool.args_schema else 'None'}")

        assert len(tools) > 0, "Should have at least one Genie tool"

    finally:
        await manager.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_genie_tool_execution(workspace_client, workspace_url):
    """Test executing a Genie tool to understand its behavior."""
    # Use only the Genie server
    genie_path = "/api/2.0/mcp/genie/01f07c8be026197b989df6c647fe6970"
    genie_url = build_databricks_server_urls(workspace_url, [genie_path])

    manager = MCPClientManager(
        workspace_client=workspace_client,
        databricks_server_urls=genie_url,
    )

    try:
        clients = await manager.get_clients()
        client = clients[0]

        # Get the raw MCP tools to inspect their structure
        mcp_tools = await client._get_tools_async()

        print(f"\n=== Raw MCP Tool Structure ===")
        for mcp_tool in mcp_tools:
            print(f"\nName: {mcp_tool.name}")
            print(f"Description: {mcp_tool.description}")
            print(f"Input Schema: {mcp_tool.inputSchema}")

            # Try to call the tool with a simple query
            try:
                print(f"\n=== Testing tool execution ===")
                # Genie tools typically take a query parameter
                # Use _call_tools_async to avoid asyncio.run() issues
                result = await client._call_tools_async(mcp_tool.name, {"query": "What data is available?"})

                print(f"Result type: {type(result)}")
                print(f"Result attributes: {dir(result)}")

                if hasattr(result, 'content'):
                    print(f"Content: {result.content}")
                    if result.content:
                        for i, item in enumerate(result.content):
                            print(f"  Content item {i}: {type(item)}")
                            print(f"    Attributes: {dir(item)}")
                            if hasattr(item, 'text'):
                                print(f"    Text: {item.text[:200] if len(item.text) > 200 else item.text}")

                if hasattr(result, 'isError'):
                    print(f"Is Error: {result.isError}")

            except Exception as e:
                print(f"Error calling tool: {e}")
                import traceback
                traceback.print_exc()

    finally:
        await manager.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_genie_vs_system_ai_tools(workspace_client, workspace_url):
    """Compare tool structures between Genie and system/ai servers."""
    # Get both server types
    genie_path = "/api/2.0/mcp/genie/01f07c8be026197b989df6c647fe6970"
    system_ai_path = "/api/2.0/mcp/functions/system/ai"

    genie_url = build_databricks_server_urls(workspace_url, [genie_path])
    system_ai_url = build_databricks_server_urls(workspace_url, [system_ai_path])

    # Test Genie
    genie_manager = MCPClientManager(
        workspace_client=workspace_client,
        databricks_server_urls=genie_url,
    )

    # Test system/ai
    system_ai_manager = MCPClientManager(
        workspace_client=workspace_client,
        databricks_server_urls=system_ai_url,
    )

    try:
        print("\n=== Genie Tools ===")
        genie_clients = await genie_manager.get_clients()
        genie_mcp_tools = await genie_clients[0]._get_tools_async()
        for tool in genie_mcp_tools:
            print(f"Name: {tool.name}")
            print(f"Input Schema Keys: {list(tool.inputSchema.get('properties', {}).keys())}")
            print(f"Required: {tool.inputSchema.get('required', [])}")

        print("\n=== System/AI Tools ===")
        system_ai_clients = await system_ai_manager.get_clients()
        system_ai_mcp_tools = await system_ai_clients[0]._get_tools_async()
        for tool in system_ai_mcp_tools:
            print(f"Name: {tool.name}")
            print(f"Input Schema Keys: {list(tool.inputSchema.get('properties', {}).keys())}")
            print(f"Required: {tool.inputSchema.get('required', [])}")

    finally:
        await genie_manager.close()
        await system_ai_manager.close()
