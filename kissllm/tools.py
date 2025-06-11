import asyncio
import json
import logging
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, get_type_hints

from kissllm.mcp.manager import MCPManager

logger = logging.getLogger(__name__)


class LocalToolManager:
    """Registry for locally defined tool functions."""

    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(self, func=None, *, name=None, description=None):
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
            self._tools[func_name] = {
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
            logger.debug(f"Registered local tool: {func_name}")

            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        # Handle both @register and @register() syntax
        if func is None:
            return decorator
        return decorator(func)

    def get_tools_specs(self) -> List[Dict[str, Any]]:
        """Get all registered local tool specifications."""
        return [tool["spec"] for tool in self._tools.values()]

    def _get_tool_function(self, name: str) -> Optional[Callable]:
        """Get a registered local tool function by name"""
        tool = self._tools.get(name)
        return tool["function"] if tool else None

    def is_local_tool(self, name: str) -> bool:
        """Check if a tool name corresponds to a registered local tool."""
        return name in self._tools

    async def execute_tool(self, name: str, args: Dict[str, Any]) -> Any:
        """Execute a registered local tool function."""
        func = self._get_tool_function(name)
        if not func:
            raise ValueError(f"Local tool function '{name}' not found")
        logger.debug(f"Executing local tool: {name} with args: {args}")
        if asyncio.iscoroutinefunction(func):
            return await func(**args)
        else:
            return func(**args)


class ToolManager:
    """Registry coordinating local tools and MCP server connections."""

    def __init__(
        self,
        local_manager: LocalToolManager | None = None,
        mcp_manager: MCPManager | None = None,
    ):
        """
        Initialize the ToolRegistry with provided managers.

        Args:
            local_manager: An instance of LocalToolManager.
            mcp_manager: An instance of MCPManager.
        """
        self.local_manager = local_manager
        self.mcp_manager = mcp_manager

    async def __aenter__(self):
        """Setup tools"""
        if self.mcp_manager:
            return await self.mcp_manager.register_all()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Tear down tools."""
        if self.mcp_manager:
            await self.mcp_manager.unregister_all()

    def get_tools_specs(self) -> List[Dict[str, Any]]:
        """Get all registered tool specifications (local and MCP)."""
        local_specs = self.local_manager.get_tools_specs() if self.local_manager else []
        mcp_specs = self.mcp_manager.get_tool_specs() if self.mcp_manager else []
        return local_specs + mcp_specs

    async def execute_tool_call(self, tool_call: Dict[str, Any]) -> Any:
        """Execute a tool call (local or MCP) with the given parameters"""
        function_name = tool_call.get("function", {}).get("name")
        function_args_str = tool_call.get("function", {}).get("arguments", "{}")

        if not function_name:
            raise ValueError("Tool call missing function name.")

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
            # Assume it's already a dict if not a string
            args = function_args_str if isinstance(function_args_str, dict) else {}

        # Check local tools first
        if self.local_manager and self.local_manager.is_local_tool(function_name):
            return await self.local_manager.execute_tool(function_name, args)

        # Check MCP tools
        if self.mcp_manager and self.mcp_manager.is_mcp_tool(function_name):
            return await self.mcp_manager.execute_tool(function_name, args)

        # Tool not found
        logger.error(
            f"Tool '{function_name}' not found in local registry or connected MCP servers."
        )
        raise ValueError(
            f"Tool function '{function_name}' not found or corresponding MCP server not connected."
        )


class ToolMixin:
    """Mixin class for tool-related functionality in responses"""

    def __init__(self, tool_registry) -> None:
        self.tool_registry = tool_registry

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
        """Get results from executed tool calls using the provided ToolRegistry."""
        # Avoid re-executing if results are already stored
        if hasattr(self, "_tool_results") and self._tool_results:
            return self._tool_results

        tool_results = []
        tool_calls = self.get_tool_calls()  # Ensure tool calls are populated if needed

        if not tool_calls:
            return []

        for tool_call in tool_calls:
            try:
                result = await self.tool_registry.execute_tool_call(tool_call)
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

        # Store results to prevent re-execution
        self._tool_results = tool_results
        return self._tool_results
