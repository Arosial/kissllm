import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.fastmcp.server import Settings
from mcp.server.lowlevel.server import Server as MCPServer
from mcp.types import (
    ImageContent,
    TextContent,
)
from mcp.types import Tool as MCPTool

from . import MCPConfig
from .manager import MCPManager

logger = logging.getLogger(__name__)


class MCPAggregatorServer(FastMCP):
    """
    An MCP aggregator server that aggregates multiple backend MCP servers.
    Inheriting from FastMCP for `run` methods.

    TODO: Only Tool-Related functions are added now.
    """

    def __init__(
        self,
        backend_configs: List[MCPConfig],
        name: str = "kissllm-mcp-aggregator",
        **settings: Any,
    ):
        self.settings = Settings(**settings)

        # Initialize MCPManager with backend configs
        self._mcp_manager = MCPManager(mcp_configs=backend_configs)

        # Mappings to route requests
        # aggregator_tool_name -> (server_name, original_tool_name)
        self._tool_map: Dict[str, Tuple[str, str]] = {}

        self._mcp_server = MCPServer(
            name=name,
            lifespan=self._lifespan,
        )

        # Register aggregator handlers (overriding or supplementing base handlers)
        self._register_handlers()

    def _register_handlers(self):
        """Register MCP handlers for proxied methods, overriding base handlers."""
        self._mcp_server.list_tools()(self.list_tools)
        self._mcp_server.call_tool()(self.call_tool)

    @asynccontextmanager
    async def _lifespan(self, server: MCPServer) -> AsyncIterator[object]:
        """Lifespan context manager to manage backend connections using MCPManager."""
        logger.info("MCP Aggregator Server lifespan starting...")
        # Use MCPManager as a context manager to handle connections
        async with self._mcp_manager:
            logger.info("MCP Manager entered, backends should be connected.")
            try:
                yield {}  # Server runs while in this block
            finally:
                logger.info("MCP Aggregator Server lifespan shutting down...")
        logger.info("MCP Manager exited, backends should be disconnected.")

    async def list_tools(self) -> list[MCPTool]:
        """List tools aggregated from all connected backends."""
        logger.debug("Aggregating list_tools request.")
        all_tools: List[MCPTool] = []
        self._tool_map.clear()

        backend_tools = self._mcp_manager._mcp_tools

        for aggregator_tool_name, tool_info in backend_tools.items():
            server_name = tool_info["server_name"]
            original_name = tool_info["original_name"]
            spec = tool_info["spec"]["function"]  # Extract function spec

            proxied_tool = MCPTool(
                name=aggregator_tool_name,
                description=f"[{server_name}] {spec.get('description', '')}",  # Add server name prefix to description
                inputSchema=spec.get("parameters", {}),
            )
            all_tools.append(proxied_tool)
            self._tool_map[aggregator_tool_name] = (server_name, original_name)
            logger.debug(
                f"Mapping aggregator tool '{aggregator_tool_name}' to server '{server_name}', original name '{original_name}'"
            )

        logger.info(f"Aggregated {len(all_tools)} tools from backends.")
        return all_tools

    async def call_tool(
        self, name: str, arguments: Dict[str, Any], context: Optional[Context] = None
    ) -> List[TextContent | ImageContent]:
        """Call a tool, routing it to the correct backend."""
        logger.debug(f"Aggregating call_tool request for tool: {name}")

        if name not in self._tool_map:
            logger.error(f"Unknown tool requested via aggregator: {name}")
            raise ToolError(f"Unknown tool: {name}")

        server_name, original_tool_name = self._tool_map[name]
        # backend_name = self._get_backend_name(server_name) # Now just server_name
        logger.info(
            f"Routing tool call '{name}' (original: '{original_tool_name}') to backend: {server_name}"
        )

        try:
            result_content = await self._mcp_manager.execute_tool(name, arguments)

            # TODO: Handle different content types (Image, EmbeddedResource) if execute_tool returns them.
            if isinstance(result_content, (str, bytes)):
                # Decode bytes assuming UTF-8, might need adjustment
                text_result = (
                    result_content.decode("utf-8")
                    if isinstance(result_content, bytes)
                    else result_content
                )
                return [TextContent(type="text", text=text_result)]
            elif isinstance(result_content, list):  # Handle list results from backend
                # Assuming list contains simple text for now
                return [
                    TextContent(type="text", text=str(item)) for item in result_content
                ]
            else:
                # Attempt to convert other types to string
                logger.warning(
                    f"Unexpected result type {type(result_content)} from tool '{name}', converting to string."
                )
                return [TextContent(type="text", text=str(result_content))]

        except ConnectionError as e:
            logger.error(
                f"Connection error calling tool '{name}' on backend {server_name}: {e}"
            )
            raise ToolError(f"Connection error executing tool '{name}': {e}") from e
        except Exception as e:
            logger.error(
                f"Error executing tool '{name}' (original: '{original_tool_name}') on backend {server_name}: {e}",
                exc_info=True,
            )
            # Reraise as ToolError for FastMCP
            raise ToolError(f"Error executing tool '{name}': {e}") from e
