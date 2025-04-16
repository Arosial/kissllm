import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

from . import MCPConfig, SSEMCPConfig, StdioMCPConfig

logger = logging.getLogger(__name__)


@dataclass
class MCPConnection:
    """Represents an active MCP server connection."""

    config: MCPConfig
    session: Optional[ClientSession] = None
    exit_stack: Optional[AsyncExitStack] = None
    tools: List[Any] = field(default_factory=list)


class MCPManager:
    """Manages connections and tool interactions with MCP servers."""

    def __init__(self):
        # Connections keyed by server name (from config)
        self._mcp_connections: Dict[str, MCPConnection] = {}
        # Tools keyed by unique aggregator name (name_toolname)
        self._mcp_tools: Dict[str, Dict[str, Any]] = {}

    async def register_server(self, config: MCPConfig):
        """Register and connect to an MCP server configuration."""
        name = config.name
        if name in self._mcp_connections:
            raise ValueError(f"MCP server name '{name}' already registered.")
        self._mcp_connections[name] = MCPConnection(config=config)
        await self._connect_server(name)
        logger.info(f"Registered and connected MCP server '{name}'")
        return name

    async def _connect_server(self, name: str):
        """Connect to a registered MCP server and discover its tools."""
        if name not in self._mcp_connections:
            raise ValueError(f"MCP server '{name}' not registered.")

        connection = self._mcp_connections[name]
        if connection.session:
            logger.info(f"MCP server '{name}' already connected.")
            return [tool.name for tool in connection.tools]

        config = connection.config
        logger.info(
            f"Attempting to connect to MCP server '{name}' using {config.type} transport."
        )
        exit_stack = AsyncExitStack()
        connection.exit_stack = exit_stack

        try:
            if isinstance(config, StdioMCPConfig):
                server_params = StdioServerParameters(
                    command=config.command, args=config.args, env=config.env
                )
                transport = await exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
                logger.info(f"Stdio transport created for {name}")
                read_stream, write_stream = transport

            elif isinstance(config, SSEMCPConfig):
                sse_endpoint_url = config.url
                transport = await exit_stack.enter_async_context(
                    sse_client(sse_endpoint_url)
                )
                logger.info(
                    f"SSE transport created for {name} with SSE endpoint: {sse_endpoint_url}"
                )
                read_stream, write_stream = transport
            else:
                logger.error(
                    f"Unsupported MCP configuration type: {type(config)} for server {name}"
                )
                raise TypeError(f"Unsupported MCP configuration type: {type(config)}")

            # Initialize session
            session = await exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            logger.debug(f"Transport established for {name}, initializing session.")
            connection.session = session
            await session.initialize()
            logger.info(f"MCP session initialized for '{name}'.")

            # List available tools
            response = await session.list_tools()
            logger.debug(f"Received tool list response for '{name}'.")
            tools = response.tools
            connection.tools = tools

            # Register each tool from the MCP server
            for tool in tools:
                # Create unique tool ID using server name and tool name
                tool_id = f"{name}_{tool.name}".replace(".", "_").replace("-", "_")
                self._mcp_tools[tool_id] = {
                    "server_name": name,  # Store the server name this tool belongs to
                    "original_name": tool.name,  # Original MCP tool name
                    "description": tool.description,
                    "spec": {
                        "type": "function",
                        "function": {
                            "name": tool_id,  # Unique name for LLM
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        },
                    },
                }
                logger.debug(
                    f"Registered MCP tool '{tool.name}' as '{tool_id}' from server '{name}'."
                )

            discovered_tool_names = [tool.name for tool in tools]
            logger.info(
                f"Connected to MCP server '{name}' with tools: {discovered_tool_names}"
            )
            return discovered_tool_names

        except Exception as e:
            logger.error(f"Error connecting to MCP server '{name}': {e}", exc_info=True)
            # Ensure cleanup if connection fails
            await exit_stack.aclose()
            connection.exit_stack = None
            connection.session = None
            raise ConnectionError(
                f"Failed to connect to MCP server '{name}': {e}"
            ) from e

    async def unregister_server(self, name: str):
        """Disconnect from an MCP server and clean up resources."""
        if name not in self._mcp_connections:
            return

        connection = self._mcp_connections[name]
        if connection.exit_stack:
            await connection.exit_stack.aclose()
            connection.exit_stack = None
            connection.session = None
            connection.tools = []
            logger.info(f"Disconnected from MCP server '{name}'.")

        # Remove associated tools from the registry
        logger.debug(f"Removing tools associated with server '{name}' from registry.")
        tool_ids_to_remove = [
            tool_id
            for tool_id, tool_info in self._mcp_tools.items()
            if tool_info["server_name"] == name
        ]
        for tool_id in tool_ids_to_remove:
            if tool_id in self._mcp_tools:
                del self._mcp_tools[tool_id]
                logger.debug(f"Removed MCP tool '{tool_id}' from registry.")

        del self._mcp_connections[name]

    async def unregister_all(self):
        """Disconnect from all connected MCP servers."""
        server_names = list(self._mcp_connections.keys())
        for name in server_names:
            await self.unregister_server(name)

    async def execute_tool(self, function_name: str, args: Dict) -> Any:
        """Execute an MCP tool call using the appropriate server connection."""
        if function_name not in self._mcp_tools:
            raise ValueError(f"MCP tool '{function_name}' not found in registry.")

        mcp_tool_info = self._mcp_tools[function_name]
        server_name = mcp_tool_info["server_name"]
        original_tool_name = mcp_tool_info["original_name"]  # MCP tool name

        if server_name not in self._mcp_connections:
            raise ValueError(
                f"MCP server '{server_name}' for tool '{function_name}' not registered or disconnected."
            )

        connection = self._mcp_connections[server_name]
        session = connection.session

        if not session:
            logger.warning(
                f"Session for MCP server '{server_name}' not active. Attempting to reconnect..."
            )
            try:
                await self._connect_server(server_name)
                session = self._mcp_connections[
                    server_name
                ].session  # Re-fetch session after connect attempt
                if not session:
                    logger.error(f"Reconnection to MCP server '{server_name}' failed.")
                    raise ValueError("Reconnection failed.")
                logger.info(f"Successfully reconnected to MCP server '{server_name}'.")
            except Exception as e:
                logger.error(
                    f"Failed to reconnect to MCP server '{server_name}': {e}",
                    exc_info=True,
                )
                raise ConnectionError(
                    f"Failed to reconnect to MCP server '{server_name}' for tool"
                    f" '{function_name}': {e}"
                ) from e

        logger.debug(
            f"Executing MCP tool '{original_tool_name}' on server '{server_name}' with args: {args}"
        )

        try:
            result = await session.call_tool(original_tool_name, args)
            logger.debug(
                f"MCP tool '{original_tool_name}' executed successfully. Result content type: {type(result.content)}"
            )
        except Exception as e:
            logger.error(
                f"Error calling MCP tool '{original_tool_name}' on server '{server_name}': {e}",
                exc_info=True,
            )
            raise RuntimeError(
                f"Failed to execute MCP tool {original_tool_name}"
            ) from e
        return result.content

    def get_tool_specs(self) -> List[Dict[str, Any]]:
        """Get all registered MCP tool specifications"""
        return [tool["spec"] for tool in self._mcp_tools.values()]

    def is_mcp_tool(self, function_name: str) -> bool:
        """Check if a function name corresponds to a registered MCP tool."""
        return function_name in self._mcp_tools
