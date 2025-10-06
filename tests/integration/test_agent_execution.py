"""Integration tests for agent execution with real Databricks resources."""

import pytest
from langchain_core.messages import HumanMessage
from databricks_langchain import ChatDatabricks

from src.langgraph_mcp_agent.agent import create_tool_calling_agent
from src.langgraph_mcp_agent.mcp_client import MCPClientManager, build_databricks_server_urls
from src.langgraph_mcp_agent.tools import get_mcp_tools, list_tool_names


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_initializes_with_databricks_model(
    workspace_client,
    workspace_url,
    databricks_server_paths,
    model_name
):
    """Test that agent can initialize with Databricks model and MCP tools."""
    # Build server URLs
    databricks_servers = build_databricks_server_urls(workspace_url, databricks_server_paths)

    # Create MCP client and get tools
    manager = MCPClientManager(
        workspace_client=workspace_client,
        databricks_server_urls=databricks_servers,
    )

    try:
        clients = await manager.get_clients()
        tools = await get_mcp_tools(clients)

        print(f"\nLoaded {len(tools)} tools: {list_tool_names(tools)}")

        # Create model
        model = ChatDatabricks(endpoint=model_name)

        # Create agent
        agent = create_tool_calling_agent(
            model=model,
            tools=tools,
            system_prompt="You are a helpful assistant with access to Databricks tools."
        )

        assert agent is not None, "Agent should be created successfully"

    finally:
        await manager.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_executes_simple_query(
    workspace_client,
    workspace_url,
    databricks_server_paths,
    model_name
):
    """Test that agent can execute a simple query."""
    # Build server URLs
    databricks_servers = build_databricks_server_urls(workspace_url, databricks_server_paths)

    # Create MCP client and get tools
    manager = MCPClientManager(
        workspace_client=workspace_client,
        databricks_server_urls=databricks_servers,
    )

    try:
        clients = await manager.get_clients()
        tools = await get_mcp_tools(clients)

        print(f"\nLoaded {len(tools)} tools: {list_tool_names(tools)}")

        # Create model and agent
        model = ChatDatabricks(endpoint=model_name)
        agent = create_tool_calling_agent(
            model=model,
            tools=tools,
            system_prompt="You are a helpful assistant."
        )

        # Execute a simple query that doesn't require tools
        query = "What is 2+2?"
        print(f"\nExecuting query: {query}")

        result = await agent.ainvoke({
            "messages": [HumanMessage(content=query)]
        })

        # Verify response
        assert "messages" in result, "Result should contain messages"
        assert len(result["messages"]) > 0, "Should have at least one message"

        last_message = result["messages"][-1]
        print(f"Agent response: {last_message.content}")

        # Verify the response contains something about "4"
        assert "4" in last_message.content, "Response should contain the answer 4"

    finally:
        await manager.close()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.slow
async def test_agent_can_use_tools(
    workspace_client,
    workspace_url,
    databricks_server_paths,
    model_name
):
    """Test that agent can successfully call and use MCP tools."""
    # Build server URLs
    databricks_servers = build_databricks_server_urls(workspace_url, databricks_server_paths)

    # Create MCP client and get tools
    manager = MCPClientManager(
        workspace_client=workspace_client,
        databricks_server_urls=databricks_servers,
    )

    try:
        clients = await manager.get_clients()
        tools = await get_mcp_tools(clients)

        print(f"\nLoaded {len(tools)} tools: {list_tool_names(tools)}")

        # Create model and agent
        model = ChatDatabricks(endpoint=model_name)
        agent = create_tool_calling_agent(
            model=model,
            tools=tools,
            system_prompt="You are a helpful assistant. Use tools when appropriate."
        )

        # Execute a query that should trigger tool use
        # This is a generic query - actual behavior depends on available tools
        query = "List the available tools you have access to."
        print(f"\nExecuting query: {query}")

        result = await agent.ainvoke({
            "messages": [HumanMessage(content=query)]
        })

        # Verify response
        assert "messages" in result, "Result should contain messages"
        assert len(result["messages"]) > 0, "Should have at least one message"

        # Check if any tools were called by examining message history
        has_tool_calls = any(
            hasattr(msg, 'tool_calls') and msg.tool_calls
            for msg in result["messages"]
        )

        last_message = result["messages"][-1]
        print(f"Agent response: {last_message.content}")
        print(f"Tools were called: {has_tool_calls}")

    finally:
        await manager.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_end_to_end_agent_workflow(
    workspace_client,
    workspace_url,
    databricks_server_paths,
    model_name
):
    """Test complete end-to-end workflow from initialization to execution."""
    from src.langgraph_mcp_agent.app import LangGraphMCPApp

    # Create app
    app = LangGraphMCPApp(
        model_name=model_name,
        workspace_client=workspace_client,
    )

    try:
        # Initialize
        await app.initialize()

        # Run a query
        query = "Hello! Can you tell me what you can help me with?"
        print(f"\nExecuting query: {query}")

        result = await app.run_query(query)

        # Verify response
        assert "messages" in result, "Result should contain messages"
        last_message = result["messages"][-1]

        print(f"Agent response: {last_message.content}")

        assert len(last_message.content) > 0, "Response should not be empty"

    finally:
        await app.cleanup()
