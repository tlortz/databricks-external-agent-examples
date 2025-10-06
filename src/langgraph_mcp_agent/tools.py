"""Tool management for MCP-based tools."""

from typing import List, Union

from langchain_core.tools import BaseTool, StructuredTool
from databricks_mcp import DatabricksMCPClient
from pydantic import BaseModel, create_model


def _create_tool_schema(input_schema: dict) -> type[BaseModel]:
    """Create a Pydantic model from MCP tool input schema."""
    fields = {}
    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])

    for field_name, field_info in properties.items():
        field_type = str  # Default to string
        field_description = field_info.get("description", "")

        # Map JSON schema types to Python types
        json_type = field_info.get("type", "string")
        if json_type == "integer":
            field_type = int
        elif json_type == "number":
            field_type = float
        elif json_type == "boolean":
            field_type = bool
        elif json_type == "array":
            field_type = list
        elif json_type == "object":
            field_type = dict

        # Set default and required status
        is_required = field_name in required
        if is_required:
            fields[field_name] = (field_type, ...)
        else:
            fields[field_name] = (field_type, None)

    return create_model("ToolInput", **fields)


async def get_mcp_tools(mcp_clients: Union[DatabricksMCPClient, List[DatabricksMCPClient]]) -> List[BaseTool]:
    """
    Get tools from MCP client(s) and convert to LangChain tools.

    Args:
        mcp_clients: Single MCP client or list of MCP clients

    Returns:
        List of LangChain tools from MCP servers
    """
    # Ensure we have a list of clients
    if not isinstance(mcp_clients, list):
        mcp_clients = [mcp_clients]

    # Aggregate tools from all clients, keeping track of which client each tool came from
    tool_client_map = {}  # Maps tool_name -> (client, mcp_tool)
    for mcp_client in mcp_clients:
        tools = await mcp_client._get_tools_async()
        for tool in tools:
            tool_client_map[tool.name] = (mcp_client, tool)

    # Convert MCP tools to LangChain tools
    langchain_tools = []
    for tool_name, (mcp_client, mcp_tool) in tool_client_map.items():

        # Create a wrapper function that calls the MCP tool on the correct client
        # Use default arguments to capture values in the closure
        def make_tool_func(client: DatabricksMCPClient, tool_name: str):
            async def call_mcp_tool(**kwargs):
                """Execute the MCP tool."""
                # Use _call_tools_async instead of call_tool to avoid asyncio.run() in async context
                result = await client._call_tools_async(tool_name, kwargs)

                # Extract the actual content from CallToolResult
                # CallToolResult has a 'content' field which is a list of content items
                if hasattr(result, 'content') and result.content:
                    # Get the first content item (typically text)
                    content_item = result.content[0]
                    if hasattr(content_item, 'text'):
                        return content_item.text
                    elif hasattr(content_item, 'content'):
                        return content_item.content
                    else:
                        return str(content_item)

                # Fallback to string representation
                return str(result)
            return call_mcp_tool

        # Create the Pydantic schema from the MCP tool's input schema
        args_schema = _create_tool_schema(mcp_tool.inputSchema)

        # Create a LangChain StructuredTool
        # Use coroutine parameter for async functions
        langchain_tool = StructuredTool(
            name=mcp_tool.name,
            description=mcp_tool.description or f"Tool: {mcp_tool.name}",
            coroutine=make_tool_func(mcp_client, mcp_tool.name),
            args_schema=args_schema,
        )

        langchain_tools.append(langchain_tool)

    return langchain_tools


def list_tool_names(tools: List[BaseTool]) -> List[str]:
    """
    Get list of tool names.

    Args:
        tools: List of tools

    Returns:
        List of tool names
    """
    return [tool.name for tool in tools]


def get_tool_by_name(tools: List[BaseTool], name: str) -> BaseTool:
    """
    Get a specific tool by name.

    Args:
        tools: List of tools
        name: Name of the tool to retrieve

    Returns:
        The requested tool

    Raises:
        ValueError: If tool not found
    """
    for tool in tools:
        if tool.name == name:
            return tool
    raise ValueError(f"Tool '{name}' not found. Available tools: {list_tool_names(tools)}")
