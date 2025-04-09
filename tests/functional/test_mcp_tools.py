import os
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import closing

import pytest
from dotenv import load_dotenv

from kissllm.client import LLMClient
from kissllm.mcp import MCPManager, SSEMCPConfig, StdioMCPConfig
from kissllm.tools import ToolManager

load_dotenv()
test_provider = os.environ.get(
    "TEST_PROVIDER", "openai"
)  # Default to openai if not set
test_model = os.environ["TEST_MODEL"]

# Create a simple MCP server file for testing
MCP_SERVER_CONTENT = """
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Test Server")

@mcp.tool()
def add(a: int, b: int) -> int:
    \"\"\"Add two numbers\"\"\"
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    \"\"\"Multiply two numbers\"\"\"
    return a * b

if __name__ == "__main__":
    import sys
    mode = sys.argv[1]
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
        mcp.settings.port = port
    mcp.run(mode)
"""


def find_free_port():
    """Find an available TCP port."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def mcp_server_path():
    """Create a temporary MCP server file for testing"""
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
        f.write(MCP_SERVER_CONTENT.encode("utf-8"))
        server_path = f.name

    yield server_path

    # Clean up the temporary file only if it exists
    if server_path and os.path.exists(server_path):
        try:
            os.unlink(server_path)
        except OSError as e:
            # Log error but don't fail the test run
            print(f"Error removing temporary file {server_path}: {e}")


@pytest.fixture(scope="module")
def sse_mcp_server(mcp_server_path):
    """Starts an MCP server with SSE transport in a subprocess."""
    port = find_free_port()
    host = "localhost"
    base_url = f"http://{host}:{port}"
    sse_url = f"{base_url}/sse"

    # Command to run the server script with the chosen port
    cmd = [sys.executable, mcp_server_path, "sse", str(port)]

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


@pytest.fixture(scope="function")
def tool_registry():
    """Provides a ToolRegistry instance with managers for MCP tests."""

    mcp_manager = MCPManager()
    registry = ToolManager(mcp_manager=mcp_manager)
    return registry


async def register_and_connect_mcp_server(
    registry: ToolManager, server_id: str, config: StdioMCPConfig | SSEMCPConfig
):
    """Helper to register and connect to an MCP server using a ToolRegistry instance."""
    # Registration and connection are now handled by MCPManager within the registry
    print(
        f"Registering and connecting to MCP server '{server_id}' with config: {config}"
    )
    # register_server now handles connection and returns the server_id if successful
    await registry.mcp_manager.register_server(server_id, config)
    # Retrieve discovered tools after connection (assuming register_server connects)
    connection = registry.mcp_manager._mcp_connections.get(server_id)
    discovered_tools = [tool.name for tool in connection.tools] if connection else []
    print(f"Discovered tools for '{server_id}': {discovered_tools}")
    return discovered_tools


# Helper function to perform the core LLM interaction and assertions
async def _perform_mcp_tool_test(client: LLMClient, server_id: str):
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

    print(f"\nStreaming response with MCP tool calls (Server: {server_id}):")
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
    server_id = "test_stdio_server"
    config = StdioMCPConfig(command=sys.executable, args=[mcp_server_path, "stdio"])
    try:
        # Register and connect using the provided registry instance
        discovered_tools = await register_and_connect_mcp_server(
            tool_registry, server_id, config
        )

        # Verify that the tools were discovered
        assert "add" in discovered_tools
        assert "multiply" in discovered_tools

        # Verify tools are registered with correct IDs in the specific registry instance
        registered_tool_specs = tool_registry.get_tools_specs()
        registered_tool_names = [
            spec["function"]["name"] for spec in registered_tool_specs
        ]
        assert f"{server_id}_add" in registered_tool_names
        assert f"{server_id}_multiply" in registered_tool_names

        # Initialize client with the registry containing the MCP connection
        client = LLMClient(
            provider_model=f"{test_provider}/{test_model}", tool_registry=tool_registry
        )

        # Perform the actual LLM interaction test
        await _perform_mcp_tool_test(client, server_id)
    finally:
        # Clean up by unregistering from the specific registry instance
        await tool_registry.mcp_manager.unregister_server(server_id)


@pytest.mark.asyncio
async def test_mcp_sse_tools(sse_mcp_server, tool_registry):
    """Test MCP tools functionality using SSE transport."""
    sse_url = sse_mcp_server
    server_id = "test_sse_server"
    config = SSEMCPConfig(url=sse_url)

    try:
        # Register and connect using the provided registry instance
        discovered_tools = await register_and_connect_mcp_server(
            tool_registry, server_id, config
        )

        # Verify that the tools were discovered
        assert "add" in discovered_tools
        assert "multiply" in discovered_tools

        # Verify tools are registered with correct IDs in the specific registry instance
        registered_tool_specs = tool_registry.get_tools_specs()
        registered_tool_names = [
            spec["function"]["name"] for spec in registered_tool_specs
        ]
        assert f"{server_id}_add" in registered_tool_names
        assert f"{server_id}_multiply" in registered_tool_names

        # Initialize client with the registry containing the MCP connection
        client = LLMClient(
            provider_model=f"{test_provider}/{test_model}", tool_registry=tool_registry
        )

        # Perform the actual LLM interaction test
        await _perform_mcp_tool_test(client, server_id)

    finally:
        # Clean up by unregistering from the specific registry instance
        await tool_registry.mcp_manager.unregister_server(server_id)
