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
    """Provides a ToolRegistry instance with managers for MCP tests."""

    mcp_manager = MCPManager()
    registry = ToolManager(mcp_manager=mcp_manager)
    return registry


async def register_and_connect_mcp_server(
    registry: ToolManager, config: StdioMCPConfig | SSEMCPConfig
):
    """Helper to register and connect to an MCP server using a ToolRegistry instance."""
    server_name = config.name
    # Registration and connection are now handled by MCPManager within the registry
    print(
        f"Registering and connecting to MCP server '{server_name}' with config: {config}"
    )
    # register_server now handles connection and returns the server_name if successful
    await registry.mcp_manager.register_server(config)
    # Retrieve discovered tools after connection
    connection = registry.mcp_manager._mcp_connections.get(server_name)
    discovered_tools = [tool.name for tool in connection.tools] if connection else []
    print(f"Discovered tools for '{server_name}': {discovered_tools}")
    return discovered_tools


# Helper function to perform the core LLM interaction and assertions
async def _perform_mcp_tool_test(client: LLMClient, server_name: str):
    """Performs the LLM interaction part of the MCP tool test."""
    # Test with MCP tools
    response = await client.async_completion(
        messages=[
            {
                "role": "user",
                "content": "What is 15 + 27 and 8 * 9?",
            }
        ],
        tools=True,  # Use all registered tools (including MCP ones)
        tool_choice="auto",
        stream=True,
    )

    print(f"\nStreaming response with MCP tool calls (Server: {server_name}):")
    async for content in response.iter_content():
        print(content, end="", flush=True)
    print("\n")

    response = await response.accumulate_stream()

    # Get tool calls and results
    tool_calls = response.get_tool_calls()
    print("\nTool Calls:")
    # Check that the correct tools were called (names might include server_id prefix)
    called_tool_names = {call["function"]["name"] for call in tool_calls}
    assert any(name.endswith("_add") for name in called_tool_names)
    assert any(name.endswith("_multiply") for name in called_tool_names)

    for call in tool_calls:
        print(f"- {call['function']['name']}: {call['function']['arguments']}")
        # Basic check for argument structure
        assert "arguments" in call["function"]
        # Depending on the model, arguments might be a string or dict
        # assert isinstance(call['function']['arguments'], (str, dict))

    # get_tool_results uses the registry stored in the response object
    tool_results = await response.get_tool_results()
    print("\nTool Results:")
    assert len(tool_results) == len(tool_calls)  # Ensure one result per call
    for result in tool_results:
        assert "tool_call_id" in result
        assert "content" in result
        print(f"- {result['tool_call_id']}: {result['content']}")

    # Continue conversation with tool results
    if tool_calls:
        print("\nContinuing conversation with tool results:")
        continuation = await client.continue_with_tool_results(response, test_model)

        final_content = ""
        async for content in continuation.iter_content():
            print(content, end="", flush=True)
            final_content += content
        print("\n")
        # Basic check that the final response contains the calculated numbers
        assert "42" in final_content  # 15 + 27
        assert "72" in final_content  # 8 * 9


@pytest.mark.asyncio
async def test_mcp_stdio_tools(mcp_server_path, tool_registry):
    """Test MCP tools functionality using stdio transport."""
    server_name = "test_stdio_server"
    # Update config to call the example script with 'simple stdio' args and add name
    config = StdioMCPConfig(
        name=server_name,
        command=sys.executable,
        args=[mcp_server_path, "simple", "stdio"],
    )
    try:
        # Register and connect using the provided registry instance
        discovered_tools = await register_and_connect_mcp_server(tool_registry, config)

        # Verify that the tools were discovered
        assert "add" in discovered_tools
        assert "multiply" in discovered_tools

        # Verify tools are registered with correct IDs in the specific registry instance
        registered_tool_specs = tool_registry.get_tools_specs()
        registered_tool_names = [
            spec["function"]["name"] for spec in registered_tool_specs
        ]
        # Tool names are now serverName_toolName
        assert f"{server_name}_add" in registered_tool_names
        assert f"{server_name}_multiply" in registered_tool_names

        # Initialize client with the registry containing the MCP connection
        client = LLMClient(
            provider_model=f"{test_provider}/{test_model}", tool_registry=tool_registry
        )

        # Perform the actual LLM interaction test
        await _perform_mcp_tool_test(client, server_name)
    finally:
        # Clean up by unregistering from the specific registry instance
        await tool_registry.mcp_manager.unregister_server(server_name)


@pytest.mark.asyncio
async def test_mcp_aggregator_tools(mcp_aggregator_server, tool_registry):
    """Test MCP tools functionality via the MCPAggregatorServer."""
    aggregator_sse_url = mcp_aggregator_server
    # The aggregator server itself needs a name in its config for ToolManager
    aggregator_server_name = "test_aggregator"
    config = SSEMCPConfig(name=aggregator_server_name, url=aggregator_sse_url)

    try:
        # Register and connect the ToolManager to the MCP Aggregator Server
        # The aggregator itself connects to the backends internally.
        # The helper will discover tools *exposed by the aggregator*.
        discovered_tools = await register_and_connect_mcp_server(tool_registry, config)

        # Verify tools from both backends were discovered via the aggregator
        # The aggregator now exposes tools named: backendName_toolName
        # And ToolManager registers them as: aggregatorName_backendName_toolName
        assert "backend_0_add" in discovered_tools
        assert "backend_0_multiply" in discovered_tools
        assert "backend_1_add" in discovered_tools
        assert "backend_1_multiply" in discovered_tools

        # Verify tools are registered in ToolManager with its own prefix
        registered_tool_specs = tool_registry.get_tools_specs()
        registered_tool_names = [
            spec["function"]["name"] for spec in registered_tool_specs
        ]
        # ToolManager adds its own prefix (aggregator_server_name) to the name exposed by the aggregator
        assert f"{aggregator_server_name}_backend_0_add" in registered_tool_names
        assert f"{aggregator_server_name}_backend_0_multiply" in registered_tool_names
        assert f"{aggregator_server_name}_backend_1_add" in registered_tool_names
        assert f"{aggregator_server_name}_backend_1_multiply" in registered_tool_names

        # Initialize client with the registry containing the aggregator connection
        client = LLMClient(
            provider_model=f"{test_provider}/{test_model}", tool_registry=tool_registry
        )

        # Perform the actual LLM interaction test using the aggregator
        # The helper function's assertions should work as tool names still end with _add/_multiply
        # Pass the aggregator's registered name to the helper
        await _perform_mcp_tool_test(client, aggregator_server_name)

    finally:
        # Clean up by unregistering the aggregator connection from ToolManager
        await tool_registry.mcp_manager.unregister_server(aggregator_server_name)


@pytest.mark.asyncio
async def test_mcp_sse_tools(sse_mcp_server, tool_registry):
    """Test MCP tools functionality using SSE transport."""
    sse_url = sse_mcp_server
    server_name = "test_sse_server"
    config = SSEMCPConfig(name=server_name, url=sse_url)

    try:
        # Register and connect using the provided registry instance
        discovered_tools = await register_and_connect_mcp_server(tool_registry, config)

        # Verify that the tools were discovered
        assert "add" in discovered_tools
        assert "multiply" in discovered_tools

        # Verify tools are registered with correct IDs in the specific registry instance
        registered_tool_specs = tool_registry.get_tools_specs()
        registered_tool_names = [
            spec["function"]["name"] for spec in registered_tool_specs
        ]
        assert f"{server_name}_add" in registered_tool_names
        assert f"{server_name}_multiply" in registered_tool_names

        # Initialize client with the registry containing the MCP connection
        client = LLMClient(
            provider_model=f"{test_provider}/{test_model}", tool_registry=tool_registry
        )

        # Perform the actual LLM interaction test
        await _perform_mcp_tool_test(client, server_name)

    finally:
        # Clean up by unregistering from the specific registry instance
        await tool_registry.mcp_manager.unregister_server(server_name)
