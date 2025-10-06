# databricks integration
- assume that you will use OAuth to access any Databricks resources
- prioritize the Databricks Python SDK for accessing system resources
- use the Databricks CLI in deployment scripts, e.g. `databricks sync` to sync this repo with a Databricks app
- capture all databricks resources in a config file, and reference that config wherever resources are used
- build in test scripts that verify working access to Databricks resources prior to deploying the app; in particular, foundation model API endpoints, MCP server API endpoints and MLflow tracking server

# agent framework
- prioritize langgraph; write as little custom code as possible
- base the solution on the example notebook in notebooks/langgraph-mcp-tool-calling-agent.html
- use processes outlined in https://docs.databricks.com/aws/en/mlflow3/genai/tracing/prod-tracing-external for capturing and logging traces to MLflow on Databricks

# app
- the app only needs to provide a text input for the prompt and text output
- light theme
- app responses should use streaming if possible, along with an option to view traces
- assume single-turn conversations only, no need to maintain chat history
- python is preferred to javascript or typescript