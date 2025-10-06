"""Entry point for the LangGraph MCP agent application."""

import asyncio
from src.langgraph_mcp_agent.app import main

if __name__ == "__main__":
    asyncio.run(main())
