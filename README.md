# LangGraph MCP Agent with Databricks

A LangGraph-based agent that integrates with Databricks MCP (Model Context Protocol) servers for tool calling capabilities.

## Features

- **LangGraph Agent**: Tool-calling agent built with LangGraph
- **MCP Integration**: Connects to Databricks managed and custom MCP servers
- **Dynamic Tools**: Automatically loads tools from MCP servers
- **Interactive Chat**: Command-line chat interface
- **MLflow Tracing**: Automatic tracing and logging to Databricks MLflow
- **Fully Tested**: Comprehensive test suite with pytest

## Project Structure

```
.
├── src/langgraph_mcp_agent/
│   ├── __init__.py
│   ├── agent.py          # LangGraph agent implementation
│   ├── mcp_client.py     # MCP client manager
│   ├── tools.py          # Tool management utilities
│   └── app.py            # Main application
├── tests/                # Test suite
├── notebooks/            # Reference notebooks
├── main.py              # Entry point
└── pyproject.toml       # Project configuration
```

## Installation

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Databricks workspace with MCP server access
- **MLflow Experiment**: Create or identify an MLflow experiment for logging agent traces (e.g., `/Users/your-username/langgraph-mcp-agent`)
- **MCP Servers**: Identify the Databricks-hosted MCP servers you want to use:
  - System/AI functions: `/api/2.0/mcp/functions/system/ai`
  - Unity Catalog functions: `/api/2.0/mcp/functions/{catalog}/{schema}`
  - Genie spaces: `/api/2.0/mcp/genie/{space_id}` (get space ID from Genie UI)
  - Vector Search: `/api/2.0/mcp/vector-search/{endpoint_name}`
  - See [Databricks MCP documentation](https://docs.databricks.com/aws/en/generative-ai/mcp/managed-mcp) for details

### Setup

1. **Clone the repository** (or use existing directory):

```bash
cd external-langgraph-with-databricks-mcp
```

2. **Install dependencies** with uv:

```bash
uv pip install -e .
```

3. **For development** (includes testing tools):

```bash
uv pip install -e ".[dev]"
```

4. **Configure Databricks authentication**:

**Option A: Configuration Profile (Recommended)**

Set up your Databricks CLI configuration file at `~/.databrickscfg`:

```ini
[DEFAULT]
host = https://your-workspace.cloud.databricks.com
auth_type = oauth
```

Then create `.env`:
```bash
cp .env.example .env
```

Edit `.env` to specify your profile, workspace, and MCP servers:
```env
DATABRICKS_CONFIG_PROFILE=DEFAULT
DATABRICKS_WORKSPACE_URL=https://your-workspace.cloud.databricks.com
DATABRICKS_MODEL_NAME=databricks-claude-sonnet-4

# MCP Servers (see https://docs.databricks.com/aws/en/generative-ai/mcp/managed-mcp)
# Example: System AI + Genie space
DATABRICKS_MCP_SERVERS=/api/2.0/mcp/functions/system/ai,/api/2.0/mcp/genie/01ef9a8b123456789abcdef

# MLflow Tracing
MLFLOW_ENABLE_TRACING=true
MLFLOW_EXPERIMENT_NAME=/Users/your-username/langgraph-mcp-agent
```

**Option B: Personal Access Token (Alternative)**

Edit `.env`:
```env
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=your_databricks_token
DATABRICKS_WORKSPACE_URL=https://your-workspace.cloud.databricks.com
DATABRICKS_MODEL_NAME=databricks-claude-sonnet-4

# MCP Servers (see https://docs.databricks.com/aws/en/generative-ai/mcp/managed-mcp)
DATABRICKS_MCP_SERVERS=/api/2.0/mcp/functions/system/ai

# MLflow Tracing
MLFLOW_ENABLE_TRACING=true
MLFLOW_EXPERIMENT_NAME=/Users/your-username/langgraph-mcp-agent
```

## Usage

### Running the Agent

Activate the virtual environment and run:

```bash
source .venv/bin/activate
python main.py
```

This starts an interactive chat session:

```
=== LangGraph MCP Agent Chat ===
Type 'exit' or 'quit' to end the session.

You: What tools do you have access to?
Assistant: [Agent response with available tools]

You: [Your question]
```

### Programmatic Usage

```python
import asyncio
from src.langgraph_mcp_agent.app import LangGraphMCPApp

async def main():
    # Using a config profile
    app = LangGraphMCPApp(
        profile="DEFAULT",  # or "prod", "staging", etc.
        model_name="databricks-claude-sonnet-4",
        system_prompt="You are a helpful assistant."
    )

    await app.initialize()

    result = await app.run_query("What is 2+2?")
    print(result["messages"][-1].content)

    await app.cleanup()

asyncio.run(main())
```

### Advanced: Programmatic Server Configuration

You can also configure servers programmatically:

```python
from src.langgraph_mcp_agent.mcp_client import MCPClientManager

manager = MCPClientManager(
    profile="DEFAULT",
    databricks_server_urls=[
        "https://workspace.databricks.com/api/2.0/mcp/functions/system/ai",
        "https://workspace.databricks.com/api/2.0/mcp/genie/my_space"
    ],
    external_server_urls=[
        "https://custom-server.com/mcp"
    ]
)
```

## MLflow Tracing

The agent automatically logs traces to Databricks MLflow when enabled, capturing:
- Agent queries and responses
- Tool calls and results
- Model interactions
- Execution metadata

### Enable Tracing

Set in `.env`:

```env
MLFLOW_ENABLE_TRACING=true
MLFLOW_EXPERIMENT_NAME=/Users/your-username/langgraph-mcp-agent
MLFLOW_TRACKING_URI=databricks
```

### View Traces

1. Navigate to your Databricks workspace
2. Go to **Experiments** in the sidebar
3. Find your experiment (e.g., `/Users/your-username/langgraph-mcp-agent`)
4. Click on runs to view detailed traces including:
   - Full conversation history
   - Tool invocations and outputs
   - Model parameters
   - Execution times

### Programmatic Tracing

Traces are automatically created when `MLFLOW_ENABLE_TRACING=true`. Each call to `run_query()` creates a new trace:

```python
app = LangGraphMCPApp()
await app.initialize()

# This query will be automatically traced to MLflow
result = await app.run_query("Calculate factorial of 5")
```

## Testing

### Unit Tests

Run unit tests (fast, no Databricks connectivity required):

```bash
uv run pytest -m "not integration"
```

Run all unit tests with coverage:

```bash
uv run pytest -m "not integration" --cov=src/langgraph_mcp_agent --cov-report=html
```

Run specific test file:

```bash
uv run pytest tests/test_agent.py -v
```

### Integration Tests

Integration tests verify actual connectivity to Databricks resources. They require:
- Valid Databricks credentials configured in `.env`
- Network access to your Databricks workspace
- Configured MCP servers

Run integration tests only:

```bash
uv run pytest -m integration -v
```

Run all tests (unit + integration):

```bash
uv run pytest
```

Skip slow tests:

```bash
uv run pytest -m "not slow"
```

**Integration test coverage:**
- MCP client authentication and connectivity
- Tool retrieval from MCP servers
- Agent initialization with real Databricks models
- End-to-end agent query execution

**Note:** Integration tests may take longer and consume Databricks resources.

## Development

### Code Formatting

Format code with black:

```bash
black src/ tests/
```

Lint with ruff:

```bash
ruff check src/ tests/
```

### Project Commands

- **Install dependencies**: `uv pip install -e .`
- **Run tests**: `pytest`
- **Format code**: `black src/ tests/`
- **Lint code**: `ruff check src/`
- **Run application**: `python main.py`

## Architecture

### Components

1. **Agent (`agent.py`)**: LangGraph-based agent with tool calling
2. **MCP Client (`mcp_client.py`)**: Manages connections to MCP servers
3. **Tools (`tools.py`)**: Utilities for tool management
4. **App (`app.py`)**: Main application with chat interface

### Flow

```
User Query -> Agent -> Tool Selection -> MCP Server -> Tool Execution -> Response
```

## Configuration

### MCP Servers

**This is the most important configuration!** Control which MCP servers provide tools to your agent.

Edit your `.env` file:

```env
# Workspace URL is used to build Databricks MCP server URLs
DATABRICKS_WORKSPACE_URL=https://your-workspace.cloud.databricks.com

# Databricks Managed MCP Server Paths (comma-separated, without workspace URL)
# The workspace URL is automatically prepended
# Includes: system/ai, Unity Catalog functions, Genie spaces, Vector Search
DATABRICKS_MCP_SERVERS=/api/2.0/mcp/functions/system/ai,/api/2.0/mcp/functions/catalog1/schema1

# External MCP Servers (comma-separated, full URLs)
# Self-hosted or third-party MCP servers
EXTERNAL_MCP_SERVERS=https://custom-mcp-server.company.com
```

**Databricks MCP Server Path Formats:**
- System AI functions: `/api/2.0/mcp/functions/system/ai`
- Unity Catalog functions: `/api/2.0/mcp/functions/<catalog>/<schema>`
- Genie spaces: `/api/2.0/mcp/genie/<space_id>`
- Vector Search: `/api/2.0/mcp/vector-search/<endpoint>`

**Examples:**
- `DATABRICKS_MCP_SERVERS=/api/2.0/mcp/functions/system/ai,/api/2.0/mcp/genie/my_space`
- `EXTERNAL_MCP_SERVERS=https://custom-server.com/mcp`

The workspace URL and paths are automatically combined to create full URLs like:
`https://your-workspace.cloud.databricks.com/api/2.0/mcp/functions/system/ai`

### Model Options

Recommended Databricks models:
- `databricks-claude-sonnet-4` (recommended, default)
- `databricks-gpt-oss-20b`
- Coming soon: `gpt-5`, `gemini`

Other supported models:
- Any custom Databricks model endpoint

## Troubleshooting

### Common Issues

**Import errors**: Ensure you've installed the package:
```bash
uv pip install -e .
```

**Authentication errors**: Check your `.env` file has correct credentials

**MCP connection errors**: Verify MCP server URLs and network connectivity

## Reference

Based on the [Databricks LangGraph MCP Tool Calling Agent notebook](https://docs.databricks.com/aws/en/notebooks/source/generative-ai/langgraph-mcp-tool-calling-agent.html).

## License

MIT