"""Shared fixtures for integration tests."""

import os
import pytest
from dotenv import load_dotenv
from databricks.sdk import WorkspaceClient

# Load environment variables
load_dotenv()


@pytest.fixture(scope="session")
def workspace_client():
    """Create a real Databricks workspace client from environment."""
    profile = os.getenv("DATABRICKS_CONFIG_PROFILE")

    if profile:
        return WorkspaceClient(profile=profile)
    elif os.getenv("DATABRICKS_HOST") and os.getenv("DATABRICKS_TOKEN"):
        return WorkspaceClient(
            host=os.getenv("DATABRICKS_HOST"),
            token=os.getenv("DATABRICKS_TOKEN")
        )
    else:
        pytest.skip("No Databricks credentials configured")


@pytest.fixture(scope="session")
def workspace_url():
    """Get workspace URL from environment."""
    url = os.getenv("DATABRICKS_WORKSPACE_URL")
    if not url:
        pytest.skip("DATABRICKS_WORKSPACE_URL not configured")
    return url


@pytest.fixture(scope="session")
def model_name():
    """Get model name from environment."""
    return os.getenv("DATABRICKS_MODEL_NAME", "databricks-claude-sonnet-4")


@pytest.fixture(scope="session")
def databricks_server_paths():
    """Get Databricks MCP server paths from environment."""
    from src.langgraph_mcp_agent.mcp_client import parse_server_list_from_env
    paths = parse_server_list_from_env("DATABRICKS_MCP_SERVERS")
    if not paths:
        # Default to system/ai if not configured
        return ["/api/2.0/mcp/functions/system/ai"]
    return paths
