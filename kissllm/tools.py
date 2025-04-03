import json
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Union, get_type_hints

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


class ToolRegistry:
    """Registry for tool functions and MCP server connections."""

    _tools: Dict[str, Dict[str, Any]] = {}
    _mcp_connections: Dict[str, MCPConnection] = {}
    _mcp_tools: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(cls, func=None, *, name=None, description=None):
        """Decorator to register a function as a tool"""

        def decorator(func):
            func_name = name or func.__name__
            func_description = description or func.__doc__ or ""

            # Extract parameter information from type hints and docstring
            type_hints = get_type_hints(func)
            parameters = {"type": "object", "properties": {}, "required": []}

            # Process function signature to get parameters
            import inspect

            sig = inspect.signature(func)
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue

                param_type = type_hints.get(param_name, Any)
                param_info = {"type": "string"}  # Default to string

                # Map Python types to JSON Schema types
                if param_type is int:
                    param_info = {"type": "integer"}
                elif param_type is float:
                    param_info = {"type": "number"}
                elif param_type is bool:
                    param_info = {"type": "boolean"}
                elif param_type is list or param_type is List:
                    param_info = {"type": "array", "items": {"type": "string"}}

                parameters["properties"][param_name] = param_info

                # Add to required parameters if no default value
                if param.default == inspect.Parameter.empty:
                    parameters["required"].append(param_name)

            # Register the tool
            cls._tools[func_name] = {
                "function": func,
                "spec": {
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "description": func_description,
                        "parameters": parameters,
                    },
                },
            }

            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        # Handle both @register and @register() syntax
        if func is None:
            return decorator
        return decorator(func)

    @classmethod
    def register_mcp_server(cls, server_id: str, config: MCPConfig):
        """Register an MCP server configuration."""
        if server_id in cls._mcp_connections:
            raise ValueError(f"MCP server ID '{server_id}' already registered.")
        cls._mcp_connections[server_id] = MCPConnection(config=config)
        return server_id

    @classmethod
    async def connect_mcp_server(cls, server_id: str):
        """Connect to a registered MCP server and discover its tools."""
        if server_id not in cls._mcp_connections:
            raise ValueError(f"MCP server '{server_id}' not registered.")

        connection = cls._mcp_connections[server_id]
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
            logger.debug(f"Listing tools for '{server_id}'...")
            response = await session.list_tools()
            logger.debug(f"Received tool list response for '{server_id}'.")
            tools = response.tools
            connection.tools = tools

            # Register each tool from the MCP server
            for tool in tools:
                # Sanitize tool name for unique ID
                tool_id = f"{server_id}_{tool.name}".replace(".", "_").replace("-", "_")
                cls._mcp_tools[tool_id] = {
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

    @classmethod
    def _parse_tool_arguments(
        cls, function_args_str: Union[str, Dict[str, Any]], function_name: str
    ) -> Dict[str, Any]:
        """Parse tool arguments from string or dict."""
        if isinstance(function_args_str, str):
            try:
                args = json.loads(function_args_str)
            except json.JSONDecodeError as e:
                logger.error(
                    f"Failed to parse JSON arguments for tool '{function_name}': {e}. Args: '{function_args_str}'"
                )
                raise ValueError(
                    f"Invalid JSON arguments for tool {function_name}"
                ) from e
        elif isinstance(function_args_str, dict):
            args = function_args_str  # Already a dict
        else:
            logger.warning(
                f"Unexpected argument type for tool '{function_name}': {type(function_args_str)}. Attempting to use as empty dict."
            )
            args = {}
        return args

    @classmethod
    async def disconnect_mcp_server(cls, server_id: str):
        """Disconnect from an MCP server and clean up resources."""
        if server_id not in cls._mcp_connections:
            return

        connection = cls._mcp_connections[server_id]
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
            for tool_id, tool_info in cls._mcp_tools.items()
            if tool_info["server_id"] == server_id
        ]
        for tool_id in tool_ids_to_remove:
            if tool_id in cls._mcp_tools:
                del cls._mcp_tools[tool_id]
                logger.debug(f"Removed MCP tool '{tool_id}' from registry.")

    @classmethod
    async def disconnect_all_mcp_servers(cls):
        """Disconnect from all connected MCP servers."""
        server_ids = list(cls._mcp_connections.keys())
        for server_id in server_ids:
            await cls.disconnect_mcp_server(server_id)

    @classmethod
    async def execute_mcp_tool_call(cls, function_name: str, args: Dict) -> Any:
        """Execute an MCP tool call using the appropriate server connection."""
        mcp_tool_info = cls._mcp_tools[function_name]
        server_id = mcp_tool_info["server_id"]
        tool_name = mcp_tool_info["name"]  # MCP tool name

        if server_id not in cls._mcp_connections:
            raise ValueError(
                f"MCP server '{server_id}' for tool '{function_name}' not registered."
            )

        connection = cls._mcp_connections[server_id]
        session = connection.session

        if not session:
            logger.warning(
                f"Session for MCP server '{server_id}' not active. Attempting to reconnect..."
            )
            try:
                # Ensure the connection attempt doesn't re-register if already connecting
                # connect_mcp_server handles the check for existing session now
                await cls.connect_mcp_server(server_id)
                session = cls._mcp_connections[
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

    @classmethod
    def get_tools_specs(cls) -> List[Dict[str, Any]]:
        """Get all registered tool specifications"""
        specs = [tool["spec"] for tool in cls._tools.values()]
        # Add MCP tool specs
        specs.extend([tool["spec"] for tool in cls._mcp_tools.values()])
        return specs

    @classmethod
    def get_tool_function(cls, name: str) -> Optional[Callable]:
        """Get a registered tool function by name"""
        tool = cls._tools.get(name)
        return tool["function"] if tool else None

    @classmethod
    async def execute_tool_call(cls, tool_call: Dict[str, Any]) -> Any:
        """Execute a tool call with the given parameters"""
        function_name = tool_call.get("function", {}).get("name")
        function_args_str = tool_call.get("function", {}).get("arguments", "{}")
        if isinstance(function_args_str, str):
            try:
                args = json.loads(function_args_str)
            except json.JSONDecodeError as e:
                logger.error(
                    f"Failed to parse JSON arguments for tool '{function_name}': {e}. Args: '{function_args_str}'"
                )
                raise ValueError(
                    f"Invalid JSON arguments for tool {function_name}"
                ) from e
        else:
            args = function_args_str

        if function_name in cls._mcp_tools:
            return await cls.execute_mcp_tool_call(function_name, args)

        # Get and execute the function
        func = cls.get_tool_function(function_name)
        if not func:
            raise ValueError(f"Tool function '{function_name}' not found")

        return func(**args)


# Decorator for registering tool functions
tool = ToolRegistry.register


class ToolMixin:
    """Mixin class for tool-related functionality in responses"""

    def get_tool_calls(self) -> List[Dict[str, Any]]:
        """Get all tool calls from the response"""
        if hasattr(self, "tool_calls") and self.tool_calls:
            return self.tool_calls

        # For non-streaming responses
        if hasattr(self, "choices") and self.choices:
            for choice in self.choices:
                if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                    return [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in choice.message.tool_calls
                    ]
        return []

    async def get_tool_results(self) -> List[Dict[str, Any]]:
        """Get results from executed tool calls"""
        if hasattr(self, "tool_results") and self.tool_results:
            return self.tool_results

        tool_results = []
        for tool_call in self.get_tool_calls():
            try:
                result = await ToolRegistry.execute_tool_call(tool_call)
                tool_results.append(
                    {
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "content": str(result),
                    }
                )
            except Exception as e:
                tool_results.append(
                    {
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "content": f"Error executing tool: {str(e)}",
                    }
                )

        # Store results for future calls
        if not hasattr(self, "tool_results"):
            self.tool_results = tool_results

        return tool_results
