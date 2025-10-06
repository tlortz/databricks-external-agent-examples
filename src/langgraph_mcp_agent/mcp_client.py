"""MCP client manager for Databricks MCP servers using langchain-mcp-adapters."""

import os
from typing import List, Optional

from databricks.sdk import WorkspaceClient
from databricks_mcp.oauth_provider import DatabricksOAuthClientProvider
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient


def parse_server_list_from_env(env_var: str) -> List[str]:
    """
    Parse MCP server URLs from environment variable.

    Reads env var which should be a comma-separated list.
    Filters out empty strings and strips whitespace.

    Args:
        env_var: Name of environment variable to read

    Returns:
        List of server URLs
    """
    servers_str = os.getenv(env_var, "")
    if not servers_str.strip():
        return []

    # Split by comma and clean up whitespace
    servers = [s.strip() for s in servers_str.split(",")]
    # Filter out empty strings
    return [s for s in servers if s]


def build_databricks_server_urls(workspace_url: str, server_paths: List[str]) -> List[str]:
    """
    Build full Databricks MCP server URLs from workspace URL and paths.

    Args:
        workspace_url: Base workspace URL (e.g., https://workspace.databricks.com)
        server_paths: List of server paths (e.g., ['/api/2.0/mcp/functions/system/ai'])

    Returns:
        List of full server URLs
    """
    # Remove trailing slash from workspace URL
    workspace_url = workspace_url.rstrip('/')

    urls = []
    for path in server_paths:
        # Ensure path starts with /
        if not path.startswith('/'):
            path = '/' + path
        urls.append(f"{workspace_url}{path}")

    return urls


class MCPClientManager:
    """Manager for MCP client connections using langchain-mcp-adapters."""

    def __init__(
        self,
        workspace_client: Optional[WorkspaceClient] = None,
        profile: Optional[str] = None,
        databricks_server_urls: Optional[List[str]] = None,
        external_server_urls: Optional[List[str]] = None,
    ):
        """
        Initialize MCP client manager.

        Args:
            workspace_client: Databricks workspace client (creates default if None)
            profile: Databricks config profile name (e.g., 'DEFAULT', 'prod')
            databricks_server_urls: List of Databricks managed MCP server URLs
            external_server_urls: List of external MCP server URLs
        """
        # Create workspace client with profile if specified
        if workspace_client:
            self.workspace_client = workspace_client
        elif profile:
            self.workspace_client = WorkspaceClient(profile=profile)
        else:
            # Fall back to default profile or environment variables
            self.workspace_client = WorkspaceClient()

        self.databricks_server_urls = databricks_server_urls or []
        self.external_server_urls = external_server_urls or []
        self._client: Optional[MultiServerMCPClient] = None

    def get_all_server_urls(self) -> List[str]:
        """Get all configured server URLs."""
        urls = []
        urls.extend(self.databricks_server_urls)
        urls.extend(self.external_server_urls)
        return urls

    async def get_tools(self) -> List[BaseTool]:
        """
        Get tools from all configured MCP servers.

        Returns:
            List of LangChain tools from all MCP servers
        """
        if not self._client:
            # Build connections dictionary for MultiServerMCPClient
            all_urls = self.get_all_server_urls()
            if not all_urls:
                raise ValueError("At least one MCP server URL must be configured")

            # Create Databricks OAuth provider
            auth_provider = DatabricksOAuthClientProvider(self.workspace_client)

            # Build connections - use URL as server name for simplicity
            connections = {}
            for idx, server_url in enumerate(all_urls):
                # Create a unique server name
                server_name = f"server_{idx}"

                # Check if this is a Databricks or external server
                if server_url in self.databricks_server_urls:
                    # Databricks server - use OAuth provider
                    connections[server_name] = {
                        "transport": "streamable_http",
                        "url": server_url,
                        "auth": auth_provider,
                    }
                else:
                    # External server - use without auth (or could support custom auth)
                    connections[server_name] = {
                        "transport": "streamable_http",
                        "url": server_url,
                    }

            self._client = MultiServerMCPClient(connections)

        # Get all tools from all servers
        return await self._client.get_tools()

    async def close(self):
        """Close all MCP client connections."""
        # MultiServerMCPClient is stateless, just clear reference
        self._client = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
