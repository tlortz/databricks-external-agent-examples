"""Main application for running the LangGraph MCP agent."""

import asyncio
import os
from typing import Optional
from dotenv import load_dotenv

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks
from langchain_core.messages import HumanMessage

from .agent import create_tool_calling_agent
from .mcp_client import MCPClientManager, parse_server_list_from_env, build_databricks_server_urls
from .tools import get_mcp_tools, list_tool_names


class LangGraphMCPApp:
    """Main application class for LangGraph MCP agent."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        workspace_client: Optional[WorkspaceClient] = None,
        profile: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize the application.

        Args:
            model_name: Name of the Databricks model to use (defaults to env var)
            workspace_client: Databricks workspace client (creates from profile/env if None)
            profile: Databricks config profile name (e.g., 'DEFAULT', 'prod')
            system_prompt: Optional system prompt for the agent
        """
        # Load environment variables
        load_dotenv()

        self.model_name = model_name or os.getenv(
            "DATABRICKS_MODEL_NAME", "databricks-meta-llama-3-1-70b-instruct"
        )
        self.profile = profile or os.getenv("DATABRICKS_CONFIG_PROFILE")
        self.workspace_client = workspace_client
        self.system_prompt = system_prompt or (
            "You are a helpful AI assistant with access to various tools. "
            "Use the tools available to you to answer questions and complete tasks."
        )
        self.mcp_manager = None
        self.agent = None

        # Configure MLflow tracing
        self._setup_mlflow_tracing()

    def _setup_mlflow_tracing(self):
        """Configure MLflow tracing for agent interactions."""
        enable_tracing = os.getenv("MLFLOW_ENABLE_TRACING", "false").lower() == "true"

        if not enable_tracing:
            print("MLflow tracing is disabled")
            return

        print("Setting up MLflow tracing...")

        # Set tracking URI (defaults to Databricks)
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")
        mlflow.set_tracking_uri(tracking_uri)

        # Set or create experiment
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")
        if experiment_name:
            try:
                mlflow.set_experiment(experiment_name)
                print(f"  - MLflow experiment: {experiment_name}")
            except Exception as e:
                print(f"  - Warning: Could not set experiment '{experiment_name}': {e}")
                print(f"  - Using default experiment")

        # Configure trace export settings to prevent hanging on exit
        os.environ.setdefault("MLFLOW_EXPORT_TIMEOUT", "30")  # 30 second timeout

        # Enable LangChain autologging for tracing
        mlflow.langchain.autolog()
        print("  - MLflow LangChain autolog enabled")
        print(f"  - Tracking URI: {tracking_uri}")

    async def initialize(self):
        """Initialize MCP client and agent."""
        print("Initializing MCP client...")

        # Get workspace URL from environment or workspace client
        workspace_url = os.getenv("DATABRICKS_WORKSPACE_URL")
        if not workspace_url and self.workspace_client:
            workspace_url = self.workspace_client.config.host

        if not workspace_url:
            raise ValueError(
                "DATABRICKS_WORKSPACE_URL must be set in .env or workspace client must be configured"
            )

        # Parse MCP server configuration from environment
        databricks_server_paths = parse_server_list_from_env("DATABRICKS_MCP_SERVERS")
        databricks_servers = build_databricks_server_urls(workspace_url, databricks_server_paths)
        external_servers = parse_server_list_from_env("EXTERNAL_MCP_SERVERS")

        # Log configuration
        print(f"  - Workspace URL: {workspace_url}")
        print(f"  - Databricks managed servers: {len(databricks_servers)} configured")
        for server in databricks_servers:
            print(f"    • {server}")

        if external_servers:
            print(f"  - External servers: {len(external_servers)} configured")
            for server in external_servers:
                print(f"    • {server}")

        self.mcp_manager = MCPClientManager(
            workspace_client=self.workspace_client,
            profile=self.profile,
            databricks_server_urls=databricks_servers,
            external_server_urls=external_servers,
        )
        mcp_clients = await self.mcp_manager.get_clients()

        print("Fetching tools from MCP servers...")
        tools = await get_mcp_tools(mcp_clients)
        print(f"Loaded {len(tools)} tools: {list_tool_names(tools)}")

        print(f"Creating agent with model: {self.model_name}")
        model = ChatDatabricks(endpoint=self.model_name)

        self.agent = create_tool_calling_agent(
            model=model,
            tools=tools,
            system_prompt=self.system_prompt,
        )

        print("Agent initialized successfully!")

    @mlflow.trace(name="agent_query", span_type="AGENT")
    async def run_query(self, query: str) -> dict:
        """
        Run a query through the agent.

        Args:
            query: User query string

        Returns:
            Agent response with messages
        """
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        print(f"\nProcessing query: {query}")

        # The @mlflow.trace decorator handles tracing automatically
        # LangChain autolog will capture the full agent execution
        result = await self.agent.ainvoke(
            {"messages": [HumanMessage(content=query)]}
        )

        return result

    async def chat(self):
        """Interactive chat loop."""
        if not self.agent:
            await self.initialize()

        print("\n=== LangGraph MCP Agent Chat ===")
        print("Type 'exit' or 'quit' to end the session.\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() in ["exit", "quit"]:
                    print("Goodbye!")
                    break

                if not user_input:
                    continue

                result = await self.run_query(user_input)

                # Get the last AI message
                last_message = result["messages"][-1]
                print(f"\nAssistant: {last_message.content}\n")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")

    async def cleanup(self):
        """Cleanup resources."""
        if self.mcp_manager:
            await self.mcp_manager.close()


async def main():
    """Main entry point."""
    app = LangGraphMCPApp()

    try:
        await app.initialize()
        await app.chat()
    finally:
        await app.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
