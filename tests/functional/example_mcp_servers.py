import argparse
import os
import sys

# Determine the absolute path to this script file
# This is needed so the aggregator can correctly configure its backends
# to call this same script.
SCRIPT_PATH = os.path.abspath(__file__)


# --- Simple MCP Server Definition ---
def create_simple_mcp_server():
    """Creates and returns a simple FastMCP server instance."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("Test Server")

    @mcp.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    @mcp.tool()
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers"""
        return a * b

    return mcp


# --- MCP Aggregator Server Definition ---
def create_aggregator_mcp_server():
    """Creates and returns an MCPAggregatorServer instance."""
    from kissllm.mcp import StdioMCPConfig
    from kissllm.mcp.aggregator import MCPAggregatorServer

    # Backend configs point back to this script, running in 'simple stdio' mode
    # Add unique names for each backend
    backend_configs = [
        StdioMCPConfig(
            name="backend_0",
            command=sys.executable,
            args=[SCRIPT_PATH, "simple", "stdio"],
        ),
        StdioMCPConfig(
            name="backend_1",
            command=sys.executable,
            args=[SCRIPT_PATH, "simple", "stdio"],
        ),
    ]

    aggregator = MCPAggregatorServer(
        backend_configs=backend_configs,
        name="TestAggregatorServer",
    )
    return aggregator


# --- Main Execution Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Example MCP Servers for Testing")
    parser.add_argument(
        "server_type", choices=["simple", "aggregator"], help="Type of server to run"
    )
    parser.add_argument(
        "mode", choices=["stdio", "sse"], help="Transport mode (stdio or sse)"
    )
    parser.add_argument(
        "--port", type=int, help="Port number for SSE mode", default=None
    )
    parser.add_argument(
        "--host", type=str, help="Host for SSE mode", default="localhost"
    )

    args = parser.parse_args()

    server = None
    if args.server_type == "simple":
        server = create_simple_mcp_server()
        print(f"Starting Simple MCP Server in {args.mode} mode...")
    elif args.server_type == "aggregator":
        server = create_aggregator_mcp_server()
        print(f"Starting MCP Aggregator Server in {args.mode} mode...")

    if server:
        if args.mode == "sse" and args.port is not None:
            server.settings.port = args.port
            server.settings.host = args.host
            print(f"SSE configured for {server.settings.host}:{server.settings.port}")

        # Run the selected server
        server.run(args.mode)
    else:
        print(f"Error: Unknown server type '{args.server_type}'", file=sys.stderr)
        sys.exit(1)
