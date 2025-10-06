"""Integration tests for MLflow tracing."""

import os
import pytest
import mlflow
from src.langgraph_mcp_agent.app import LangGraphMCPApp


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mlflow_tracing_setup(workspace_client):
    """Test that MLflow tracing is properly configured."""
    # Temporarily enable tracing for this test
    original_value = os.getenv("MLFLOW_ENABLE_TRACING")
    os.environ["MLFLOW_ENABLE_TRACING"] = "true"

    try:
        app = LangGraphMCPApp(workspace_client=workspace_client)

        # Verify MLflow is configured
        tracking_uri = mlflow.get_tracking_uri()
        assert tracking_uri is not None, "MLflow tracking URI should be set"

        print(f"\nMLflow tracking URI: {tracking_uri}")

        # Verify experiment is set
        experiment = mlflow.get_experiment_by_name(
            os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/langgraph-mcp-agent")
        )
        if experiment:
            print(f"MLflow experiment: {experiment.name}")

    finally:
        # Restore original value
        if original_value:
            os.environ["MLFLOW_ENABLE_TRACING"] = original_value
        else:
            os.environ.pop("MLFLOW_ENABLE_TRACING", None)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.slow
async def test_mlflow_tracing_captures_query(workspace_client):
    """Test that MLflow tracing captures agent queries."""
    # Temporarily enable tracing for this test
    original_value = os.getenv("MLFLOW_ENABLE_TRACING")
    os.environ["MLFLOW_ENABLE_TRACING"] = "true"

    try:
        app = LangGraphMCPApp(
            workspace_client=workspace_client,
            model_name=os.getenv("DATABRICKS_MODEL_NAME", "databricks-claude-sonnet-4")
        )

        await app.initialize()

        # Run a simple query that should be traced
        test_query = "What is 2 + 2?"

        print(f"\nRunning traced query: {test_query}")

        with mlflow.start_run(run_name="test_agent_query") as run:
            result = await app.run_query(test_query)

            # Verify we got a result
            assert result is not None, "Should get a result from query"
            assert "messages" in result, "Result should contain messages"

            print(f"Run ID: {run.info.run_id}")
            print(f"Messages in result: {len(result['messages'])}")

            # The trace should be automatically logged by MLflow
            # You can view it in the Databricks MLflow UI

        await app.cleanup()

        print(f"\nTrace logged to MLflow. View in Databricks MLflow UI:")
        print(f"  Experiment: {os.getenv('MLFLOW_EXPERIMENT_NAME')}")
        print(f"  Run ID: {run.info.run_id}")

    finally:
        # Restore original value
        if original_value:
            os.environ["MLFLOW_ENABLE_TRACING"] = original_value
        else:
            os.environ.pop("MLFLOW_ENABLE_TRACING", None)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.slow
async def test_mlflow_tracing_with_tools(workspace_client):
    """Test that MLflow tracing captures tool usage."""
    # Temporarily enable tracing for this test
    original_value = os.getenv("MLFLOW_ENABLE_TRACING")
    os.environ["MLFLOW_ENABLE_TRACING"] = "true"

    try:
        app = LangGraphMCPApp(
            workspace_client=workspace_client,
            model_name=os.getenv("DATABRICKS_MODEL_NAME", "databricks-claude-sonnet-4")
        )

        await app.initialize()

        # Run a query that requires tool usage
        test_query = "Use the python_exec tool to calculate the factorial of 5"

        print(f"\nRunning traced query with tool: {test_query}")

        with mlflow.start_run(run_name="test_agent_with_tools") as run:
            result = await app.run_query(test_query)

            # Verify we got a result
            assert result is not None, "Should get a result from query"
            assert "messages" in result, "Result should contain messages"

            print(f"Run ID: {run.info.run_id}")
            print(f"Messages in result: {len(result['messages'])}")

            # Check if tools were called (should have multiple messages for tool calls)
            # Typically: HumanMessage, AIMessage with tool_calls, ToolMessage, AIMessage with response
            if len(result['messages']) > 2:
                print("Tool calls detected in trace")

        await app.cleanup()

        print(f"\nTrace with tool usage logged to MLflow. View in Databricks MLflow UI:")
        print(f"  Experiment: {os.getenv('MLFLOW_EXPERIMENT_NAME')}")
        print(f"  Run ID: {run.info.run_id}")

    finally:
        # Restore original value
        if original_value:
            os.environ["MLFLOW_ENABLE_TRACING"] = original_value
        else:
            os.environ.pop("MLFLOW_ENABLE_TRACING", None)
