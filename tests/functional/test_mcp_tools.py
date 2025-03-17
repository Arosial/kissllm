import asyncio
import os
import tempfile

import pytest
from dotenv import load_dotenv

from simplellm.client import LLMClient
from simplellm.tools import ToolRegistry

load_dotenv()
test_provider = os.environ["TEST_PROVIDER"]
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
    mcp.run()
"""


@pytest.fixture
def mcp_server_path():
    """Create a temporary MCP server file for testing"""
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
        f.write(MCP_SERVER_CONTENT.encode("utf-8"))
        server_path = f.name

    yield server_path

    # Clean up the temporary file
    try:
        os.unlink(server_path)
    except:
        pass


async def register_and_connect_mcp_server(server_path):
    """Helper to register and connect to an MCP server"""
    server_id = ToolRegistry.register_mcp_server(server_path)
    tools = await ToolRegistry.connect_mcp_server(server_id)
    return server_id, tools


@pytest.mark.asyncio
async def test_mcp_tools(mcp_server_path):
    """Test MCP tools functionality"""
    # Register and connect to the MCP server
    server_id, tools = await register_and_connect_mcp_server(mcp_server_path)
    try:
        # Verify that the tools were registered
        assert "add" in tools
        assert "multiply" in tools

        client = LLMClient(provider_model=f"{test_provider}/{test_model}")

        # Test with MCP tools
        response = client.completion(
            messages=[
                {
                    "role": "user",
                    "content": "What is 15 + 27 and 8 * 9?",
                }
            ],
            tools=True,  # Use all registered tools
            tool_choice="auto",
            stream=True,
        )

        print("\nStreaming response with MCP tool calls:")
        for content in response.iter_content():
            print(content, end="", flush=True)
        print("\n")

        response = response.accumulate_stream()

        # Get tool calls and results
        tool_calls = response.get_tool_calls()
        print("\nTool Calls:")
        for call in tool_calls:
            print(f"- {call['function']['name']}: {call['function']['arguments']}")

        tool_results = await response.get_tool_results()
        print("\nTool Results:")
        for result in tool_results:
            print(f"- {result['tool_call_id']}: {result['content']}")

        # Continue conversation with tool results
        if tool_calls:
            print("\nContinuing conversation with tool results:")
            continuation = await response.continue_with_tool_results(client, test_model)

            for content in continuation.iter_content():
                print(content, end="", flush=True)
            print("\n")
    finally:
        # Clean up by disconnecting from the MCP server
        await ToolRegistry.disconnect_mcp_server(server_id)
