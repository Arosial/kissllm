import json
from contextlib import AsyncExitStack
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, get_type_hints

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class ToolRegistry:
    """Registry for tool functions"""

    _tools: Dict[str, Dict[str, Any]] = {}
    _mcp_servers: Dict[str, Dict[str, Any]] = {}
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
    def register_mcp_server(cls, server_path: str, server_name: str = None):
        """Register an MCP server to make its tools available"""
        server_id = server_name or server_path.split("/")[-1].split(".", 1)[0]
        cls._mcp_servers[server_id] = {
            "path": server_path,
            "name": server_name or server_path,
            "session": None,
            "tools": [],
        }
        return server_id

    @classmethod
    async def connect_mcp_server(cls, server_id: str):
        """Connect to an MCP server and register its tools"""
        if server_id not in cls._mcp_servers:
            raise ValueError(f"MCP server '{server_id}' not registered")

        server_info = cls._mcp_servers[server_id]
        server_path = server_info["path"]

        # Validate server type and create parameters
        is_python = server_path.endswith(".py")
        is_js = server_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command, args=[server_path], env=None
        )

        # Create async exit stack for proper resource management
        server_info["exit_stack"] = AsyncExitStack()
        exit_stack = server_info["exit_stack"]

        # Connect to server using stdio transport
        stdio_transport = await exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read_stream, write_stream = stdio_transport

        # Initialize session with proper cleanup
        session = await exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        server_info["session"] = session
        await session.initialize()

        # List available tools
        response = await session.list_tools()
        tools = response.tools

        # Register each tool from the MCP server
        for tool in tools:
            tool_id = f"{server_id}_{tool.name}".replace(".", "_")
            cls._mcp_tools[tool_id] = {
                "server_id": server_id,
                "name": tool.name,
                "description": tool.description,
                "spec": {
                    "type": "function",
                    "function": {
                        "name": tool_id,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                },
            }

        # Store the tools in the server info
        server_info["tools"] = tools
        return [tool.name for tool in tools]

    @classmethod
    async def disconnect_mcp_server(cls, server_id: str):
        """Disconnect from an MCP server"""
        if server_id not in cls._mcp_servers:
            return

        server_info = cls._mcp_servers[server_id]
        # Clean up resources using the exit stack
        if "exit_stack" in server_info and server_info["exit_stack"]:
            await server_info["exit_stack"].aclose()
            server_info["exit_stack"] = None

        # Remove all tools from this server
        tool_ids_to_remove = []
        for tool_id, tool_info in cls._mcp_tools.items():
            if tool_info["server_id"] == server_id:
                tool_ids_to_remove.append(tool_id)

        for tool_id in tool_ids_to_remove:
            del cls._mcp_tools[tool_id]

    @classmethod
    async def execute_mcp_tool_call(cls, tool_call: Dict[str, Any]) -> Any:
        """Execute an MCP tool call"""
        function_name = tool_call.get("function", {}).get("name")
        function_args = tool_call.get("function", {}).get("arguments", "{}")

        if function_name not in cls._mcp_tools:
            raise ValueError(f"MCP tool '{function_name}' not found")

        tool_info = cls._mcp_tools[function_name]
        server_id = tool_info["server_id"]
        tool_name = tool_info["name"]

        server_info = cls._mcp_servers[server_id]
        session = server_info["session"]

        if not session:
            # Try to reconnect
            await cls.connect_mcp_server(server_id)
            session = server_info["session"]
            if not session:
                raise ValueError(f"MCP server '{server_id}' not connected")

        # Parse arguments (handle both string and dict formats)
        if isinstance(function_args, str):
            try:
                args = json.loads(function_args)
            except json.JSONDecodeError:
                args = {}
        else:
            args = function_args

        result = await session.call_tool(tool_name, args)
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
        function_args = tool_call.get("function", {}).get("arguments", "{}")

        # Check if this is an MCP tool
        if function_name in cls._mcp_tools:
            return await cls.execute_mcp_tool_call(tool_call)

        # Parse arguments (handle both string and dict formats)
        if isinstance(function_args, str):
            import json

            try:
                args = json.loads(function_args)
            except json.JSONDecodeError:
                args = {}
        else:
            args = function_args

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

    async def continue_with_tool_results(self, client, model=None):
        """Continue the conversation with tool results"""
        tool_results = await self.get_tool_results()
        if not tool_results:
            return None

        # Get the tool calls
        tool_calls = self.get_tool_calls()

        # Create messages for continuation
        messages = []

        # For streaming response
        if hasattr(self, "accumulate_stream"):
            completion = self.accumulate_stream()
            for choice in completion.choices:
                messages.append(
                    {
                        "role": "assistant",
                        "content": choice.message.content or "",
                        "tool_calls": tool_calls,
                    }
                )
        # For non-streaming response
        else:
            for choice in self.choices:
                messages.append(
                    {
                        "role": "assistant",
                        "content": choice.message.content or "",
                        "tool_calls": tool_calls,
                    }
                )

        # Add tool results
        for result in tool_results:
            messages.append(result)

        # Make a new completion with the tool results
        return client.completion(messages=messages, model=model, stream=True)
