import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


@dataclass
class StdioMCPConfig:
    """Configuration for an MCP server connected via stdio."""

    command: str
    args: List[str]
    env: Optional[Dict[str, str]] = None
    type: Literal["stdio"] = "stdio"


@dataclass
class SSEMCPConfig:
    """Configuration for an MCP server connected via SSE."""

    url: str
    type: Literal["sse"] = "sse"


MCPConfig = Union[StdioMCPConfig, SSEMCPConfig]


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
        self._mcp_connections: Dict[str, MCPConnection] = {}
        self._mcp_tools: Dict[str, Dict[str, Any]] = {}

    async def register_server(self, server_id: str, config: MCPConfig):
        """Register an MCP server configuration."""
        if server_id in self._mcp_connections:
            raise ValueError(f"MCP server ID '{server_id}' already registered.")
        self._mcp_connections[server_id] = MCPConnection(config=config)
        await self._connect_server(server_id)
        logger.info(f"Registered MCP server '{server_id}' with config: {config}")
        return server_id

    async def _connect_server(self, server_id: str):
        """Connect to a registered MCP server and discover its tools."""
        if server_id not in self._mcp_connections:
            raise ValueError(f"MCP server '{server_id}' not registered.")

        connection = self._mcp_connections[server_id]
        if connection.session:
            logger.info(f"MCP server '{server_id}' already connected.")
            return [tool.name for tool in connection.tools]

        config = connection.config
        logger.info(
            f"Attempting to connect to MCP server '{server_id}' using {config.type} transport."
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
                logger.debug(f"Stdio transport created for {server_id}")
                read_stream, write_stream = transport

            elif isinstance(config, SSEMCPConfig):
                sse_endpoint_url = config.url
                logger.debug(
                    f"Creating SSE transport for {server_id} with SSE endpoint: {sse_endpoint_url}"
                )
                transport = await exit_stack.enter_async_context(
                    sse_client(sse_endpoint_url)
                )
                read_stream, write_stream = transport
            else:
                logger.error(
                    f"Unsupported MCP configuration type: {type(config)} for server {server_id}"
                )
                raise TypeError(f"Unsupported MCP configuration type: {type(config)}")

            # Initialize session
            session = await exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            logger.debug(
                f"Transport established for {server_id}, initializing session."
            )
            connection.session = session
            await session.initialize()
            logger.info(f"MCP session initialized for '{server_id}'.")

            # List available tools
            response = await session.list_tools()
            logger.debug(f"Received tool list response for '{server_id}'.")
            tools = response.tools
            connection.tools = tools

            # Register each tool from the MCP server
            for tool in tools:
                # Sanitize tool name for unique ID
                tool_id = f"{server_id}_{tool.name}".replace(".", "_").replace("-", "_")
                self._mcp_tools[tool_id] = {
                    "server_id": server_id,
                    "name": tool.name,  # Original MCP tool name
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
                    f"Registered MCP tool '{tool.name}' as '{tool_id}' from server '{server_id}'."
                )

            discovered_tool_names = [tool.name for tool in tools]
            logger.info(
                f"Connected to MCP server '{server_id}' with tools: {discovered_tool_names}"
            )
            return discovered_tool_names

        except Exception as e:
            logger.error(
                f"Error connecting to MCP server '{server_id}': {e}", exc_info=True
            )
            # Ensure cleanup if connection fails
            await exit_stack.aclose()
            connection.exit_stack = None
            connection.session = None
            raise ConnectionError(
                f"Failed to connect to MCP server '{server_id}': {e}"
            ) from e

    async def unregister_server(self, server_id: str):
        """Disconnect from an MCP server and clean up resources."""
        if server_id not in self._mcp_connections:
            return

        connection = self._mcp_connections[server_id]
        if connection.exit_stack:
            await connection.exit_stack.aclose()
            connection.exit_stack = None
            connection.session = None
            connection.tools = []
            logger.info(f"Disconnected from MCP server '{server_id}'.")

        # Remove associated tools from the registry
        logger.debug(
            f"Removing tools associated with server '{server_id}' from registry."
        )
        tool_ids_to_remove = [
            tool_id
            for tool_id, tool_info in self._mcp_tools.items()
            if tool_info["server_id"] == server_id
        ]
        for tool_id in tool_ids_to_remove:
            if tool_id in self._mcp_tools:
                del self._mcp_tools[tool_id]
                logger.debug(f"Removed MCP tool '{tool_id}' from registry.")

        del self._mcp_connections[server_id]

    async def unregister_all(self):
        """Disconnect from all connected MCP servers."""
        server_ids = list(self._mcp_connections.keys())
        for server_id in server_ids:
            await self.unregister_server(server_id)

    async def execute_tool(self, function_name: str, args: Dict) -> Any:
        """Execute an MCP tool call using the appropriate server connection."""
        mcp_tool_info = self._mcp_tools[function_name]
        server_id = mcp_tool_info["server_id"]
        tool_name = mcp_tool_info["name"]  # MCP tool name

        if server_id not in self._mcp_connections:
            raise ValueError(
                f"MCP server '{server_id}' for tool '{function_name}' not registered."
            )

        connection = self._mcp_connections[server_id]
        session = connection.session

        if not session:
            logger.warning(
                f"Session for MCP server '{server_id}' not active. Attempting to reconnect..."
            )
            try:
                await self._connect_server(server_id)
                session = self._mcp_connections[
                    server_id
                ].session  # Re-fetch session after connect attempt
                if not session:
                    logger.error(f"Reconnection to MCP server '{server_id}' failed.")
                    raise ValueError("Reconnection failed.")
                logger.info(f"Successfully reconnected to MCP server '{server_id}'.")
            except Exception as e:
                logger.error(
                    f"Failed to reconnect to MCP server '{server_id}': {e}",
                    exc_info=True,
                )
                raise ConnectionError(
                    f"Failed to reconnect to MCP server '{server_id}' for tool"
                    f" '{function_name}': {e}"
                ) from e

        logger.debug(
            f"Executing MCP tool '{tool_name}' on server '{server_id}' with args: {args}"
        )

        try:
            result = await session.call_tool(tool_name, args)
            logger.debug(
                f"MCP tool '{tool_name}' executed successfully. Result content type: {type(result.content)}"
            )
        except Exception as e:
            logger.error(
                f"Error calling MCP tool '{tool_name}' on server '{server_id}': {e}",
                exc_info=True,
            )
            raise RuntimeError(f"Failed to execute MCP tool {tool_name}") from e
        return result.content

    def get_tool_specs(self) -> List[Dict[str, Any]]:
        """Get all registered MCP tool specifications"""
        return [tool["spec"] for tool in self._mcp_tools.values()]

    def is_mcp_tool(self, function_name: str) -> bool:
        """Check if a function name corresponds to a registered MCP tool."""
        return function_name in self._mcp_tools
