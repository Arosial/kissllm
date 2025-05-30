import os
import socket
import subprocess
import sys
import time
from contextlib import closing

import pytest
from dotenv import load_dotenv

from kissllm.client import LLMClient
from kissllm.mcp import SSEMCPConfig, StdioMCPConfig
from kissllm.mcp.manager import MCPManager
from kissllm.tools import ToolManager

load_dotenv()
test_provider = os.environ.get(
    "TEST_PROVIDER", "openai"
)  # Default to openai if not set
test_model = os.environ["TEST_MODEL"]


# Helper function to find a free port
def find_free_port():
    """Find an available TCP port."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


# Path to the consolidated example server script
EXAMPLE_SERVER_SCRIPT = os.path.join(
    os.path.dirname(__file__), "example_mcp_servers.py"
)


@pytest.fixture(scope="module")
def mcp_server_path():
    """Provides the path to the example MCP server script."""
    # No need to create/delete a temp file anymore
    yield EXAMPLE_SERVER_SCRIPT


@pytest.fixture(scope="module")
def sse_mcp_server(mcp_server_path):
    """Starts an MCP server with SSE transport in a subprocess."""
    port = find_free_port()
    host = "localhost"
    base_url = f"http://{host}:{port}"
    sse_url = f"{base_url}/sse"

    # Command to run the simple server in SSE mode
    cmd = [
        sys.executable,
        mcp_server_path,
        "simple",
        "sse",
        "--port",
        str(port),
        "--host",
        host,
    ]

    # Start the server process
    print(f"\nStarting SSE MCP server: {' '.join(cmd)}")
    server_process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # Wait a moment for the server to start up
    time.sleep(2)  # Adjust sleep time if needed

    # Check if the process started correctly
    if server_process.poll() is not None:
        stdout, stderr = server_process.communicate()
        pytest.fail(
            f"SSE MCP server failed to start. Return code: {server_process.returncode}\n"
            f"Stdout:\n{stdout.decode()}\nStderr:\n{stderr.decode()}"
        )

    print(f"SSE MCP server running at {sse_url}")
    yield sse_url

    # Cleanup: terminate the server process
    print(f"\nStopping SSE MCP server (PID: {server_process.pid})...")
    server_process.terminate()
    try:
        server_process.wait(timeout=5)  # Wait for graceful termination
        print("SSE MCP server stopped.")
    except subprocess.TimeoutExpired:
        print("Server did not terminate gracefully, killing...")
        server_process.kill()
        server_process.wait()
        print("SSE MCP server killed.")


@pytest.fixture(scope="module")
def mcp_aggregator_server(mcp_server_path):
    """Starts an MCP Aggregator server with two stdio backends."""
    aggregator_port = find_free_port()
    aggregator_host = "localhost"
    aggregator_base_url = f"http://{aggregator_host}:{aggregator_port}"
    aggregator_sse_url = f"{aggregator_base_url}/sse"

    cmd = [
        sys.executable,
        mcp_server_path,
        "aggregator",
        "sse",
        "--port",
        str(aggregator_port),
        "--host",
        aggregator_host,
    ]

    server_process = None
    try:
        # Start the server process
        print(f"\nStarting MCP Aggregator server: {' '.join(cmd)}")
        # Ensure the subprocess can find the kissllm package
        env = os.environ.copy()
        current_pythonpath = env.get("PYTHONPATH", "")
        # Prepend sys.path to existing PYTHONPATH or create it
        env["PYTHONPATH"] = os.pathsep.join(sys.path) + os.pathsep + current_pythonpath

        server_process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
        )

        # Wait a moment for the server to start up
        time.sleep(3)  # Allow slightly more time for aggregator + backends

        # Check if the process started correctly
        if server_process.poll() is not None:
            stdout, stderr = server_process.communicate()
            pytest.fail(
                f"MCP Aggregator server failed to start. Return code: {server_process.returncode}\n"
                f"Stdout:\n{stdout.decode()}\nStderr:\n{stderr.decode()}"
            )

        print(f"MCP Aggregator server running at {aggregator_sse_url}")
        yield aggregator_sse_url

    finally:
        # Cleanup: terminate the server process
        if server_process and server_process.poll() is None:
            print(f"\nStopping MCP Aggregator server (PID: {server_process.pid})...")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
                print("MCP Aggregator server stopped.")
            except subprocess.TimeoutExpired:
                print("Aggregator server did not terminate gracefully, killing...")
                server_process.kill()
                server_process.wait()
                print("MCP Aggregator server killed.")

        # No temporary file to clean up anymore


@pytest.fixture(scope="function")
def tool_registry():
    """Provides a ToolManager instance, potentially without MCP configured initially."""
    # Tests will add MCPManager as needed
    registry = ToolManager()
    return registry


# Helper function to perform the core LLM interaction and assertions
async def _perform_mcp_tool_test(client: LLMClient):
    """Performs the LLM interaction part of the MCP tool test."""
    # Test with automatic tool execution
    final_response = await client.async_completion_with_tool_execution(
        messages=[
            {
                "role": "user",
                "content": "What is 15 + 27 and 8 * 9?",
            }
        ],
        stream=True,
    )

    # Print the final response content
    content = final_response.choices[0].message.content
    print(f"\nFinal response content: {content}")

    # Basic check that the final response contains the calculated numbers
    assert "42" in content  # 15 + 27
    assert "72" in content or "72.0" in content  # 8 * 9


@pytest.mark.asyncio
async def test_mcp_stdio_tools(mcp_server_path):
    """Test MCP tools functionality using stdio transport."""
    server_name = "test_stdio_server"
    config = StdioMCPConfig(
        name=server_name,
        command=sys.executable,
        args=[mcp_server_path, "simple", "stdio"],
    )

    # Create MCPManager with the config
    mcp_manager = MCPManager(mcp_configs=[config])

    # Use MCPManager as context manager to handle connection lifecycle
    async with mcp_manager:
        # Create ToolManager using the context-managed MCPManager
        tool_manager = ToolManager(mcp_manager=mcp_manager)

        # Verify tools are registered within the manager's context
        registered_tool_specs = tool_manager.get_tools_specs()
        registered_tool_names = [
            spec["function"]["name"] for spec in registered_tool_specs
        ]
        print(f"Registered tools (stdio): {registered_tool_names}")
        assert f"{server_name}_add" in registered_tool_names
        assert f"{server_name}_multiply" in registered_tool_names

        # Initialize client with the ToolManager
        client = LLMClient(
            provider_model=f"{test_provider}/{test_model}", tool_registry=tool_manager
        )

        # Perform the actual LLM interaction test
        await _perform_mcp_tool_test(client)

    # No explicit unregister needed, handled by async with


@pytest.mark.asyncio
async def test_mcp_aggregator_tools(mcp_aggregator_server):
    """Test MCP tools functionality via the MCPAggregatorServer."""
    aggregator_sse_url = mcp_aggregator_server
    aggregator_server_name = "test_aggregator"
    config = SSEMCPConfig(name=aggregator_server_name, url=aggregator_sse_url)

    # Create MCPManager for the aggregator server
    mcp_manager = MCPManager(mcp_configs=[config])

    async with mcp_manager:
        # Create ToolManager using the aggregator's MCPManager
        tool_manager = ToolManager(mcp_manager=mcp_manager)

        # Verify tools from backends are registered via the aggregator
        registered_tool_specs = tool_manager.get_tools_specs()
        registered_tool_names = [
            spec["function"]["name"] for spec in registered_tool_specs
        ]
        print(f"Registered tools (aggregator): {registered_tool_names}")
        # ToolManager prefixes the aggregator's name to the tools exposed by the aggregator
        assert f"{aggregator_server_name}_backend_0_add" in registered_tool_names
        assert f"{aggregator_server_name}_backend_0_multiply" in registered_tool_names
        assert f"{aggregator_server_name}_backend_1_add" in registered_tool_names
        assert f"{aggregator_server_name}_backend_1_multiply" in registered_tool_names

        # Initialize client with the ToolManager
        client = LLMClient(
            provider_model=f"{test_provider}/{test_model}", tool_registry=tool_manager
        )

        # Perform the actual LLM interaction test
        await _perform_mcp_tool_test(client)

    # No explicit unregister needed


@pytest.mark.asyncio
async def test_mcp_sse_tools(sse_mcp_server):
    """Test MCP tools functionality using SSE transport."""
    sse_url = sse_mcp_server
    server_name = "test_sse_server"
    config = SSEMCPConfig(name=server_name, url=sse_url)

    # Create MCPManager with the config
    mcp_manager = MCPManager(mcp_configs=[config])

    async with mcp_manager:
        # Create ToolManager using the context-managed MCPManager
        tool_manager = ToolManager(mcp_manager=mcp_manager)

        # Verify tools are registered
        registered_tool_specs = tool_manager.get_tools_specs()
        registered_tool_names = [
            spec["function"]["name"] for spec in registered_tool_specs
        ]
        print(f"Registered tools (sse): {registered_tool_names}")
        assert f"{server_name}_add" in registered_tool_names
        assert f"{server_name}_multiply" in registered_tool_names

        # Initialize client with the ToolManager
        client = LLMClient(
            provider_model=f"{test_provider}/{test_model}", tool_registry=tool_manager
        )

        # Perform the actual LLM interaction test
        await _perform_mcp_tool_test(client)

    # No explicit unregister needed
