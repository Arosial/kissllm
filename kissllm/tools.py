import asyncio
import inspect
import json
import logging
import re
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    get_args,
    get_origin,
    get_type_hints,
)

from kissllm.mcp.manager import MCPManager

logger = logging.getLogger(__name__)


class LocalToolManager:
    """Registry for locally defined tool functions."""

    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(self, func=None, *, name="", description=""):
        """Decorator to register a function as a tool"""

        def decorator(func):
            func_spec = self.generate_function_spec(func, name, description)
            logger.debug(f"Generated function spec: {func_spec}")
            func_name = func_spec["name"]

            # Register the tool
            self._tools[func_name] = {
                "function": func,
                "spec": {"type": "function", "function": func_spec},
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

    @classmethod
    def generate_function_spec(
        cls, func: Callable, name: str = "", description: str = ""
    ) -> Dict[str, Any]:
        """Generate a function specification from a callable.

        Args:
            func: The function to generate a spec for.
            name: The name of the function.
            description: Description of the function.

        Returns:
            Dict[str, Any]: The generated function specification.
        """

        def conv_type(py_type):
            """Map Python types to JSON Schema types."""
            if py_type is int:
                json_type = {"type": "integer"}
            elif py_type is float:
                json_type = {"type": "number"}
            elif py_type is bool:
                json_type = {"type": "boolean"}
            elif (origin := get_origin(py_type)) is not None and (
                origin is list or origin is List
            ):
                args = get_args(py_type)
                item_type = conv_type(args[0])
                json_type = {"type": "array", "items": item_type}
            else:
                json_type = {"type": "string"}
            return json_type

        name = name or func.__name__
        description = description or inspect.getdoc(func) or ""
        # Extract parameter information from type hints and docstring
        type_hints = get_type_hints(func)
        parameters = {"type": "object", "properties": {}, "required": []}

        sig = inspect.signature(func)
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            py_type = type_hints.get(param_name, Any)
            param_info = conv_type(py_type)
            parameters["properties"][param_name] = param_info

            # Add to required parameters if no default value
            if param.default == inspect.Parameter.empty:
                parameters["required"].append(param_name)

        spec = {
            "name": name,
            "description": description,
            "parameters": parameters,
        }
        return spec

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

    def __init__(self, tool_registry, use_flexible_toolcall=True) -> None:
        self.tool_registry = tool_registry
        self.use_flexible_toolcall = use_flexible_toolcall
        self.tool_calls = None

    def get_tool_calls(self) -> List[Dict[str, Any]]:
        """Get all tool calls from the response"""
        if self.tool_calls is not None:
            return self.tool_calls

        self.tool_calls = []
        if self.use_flexible_toolcall:
            self.tool_calls = self._parse_flexible_tool_calls()
        else:
            if hasattr(self, "choices") and self.choices:
                for choice in self.choices:
                    if (
                        hasattr(choice.message, "tool_calls")
                        and choice.message.tool_calls
                    ):
                        self.tool_calls = [
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
        return self.tool_calls

    def _parse_flexible_tool_calls(self) -> List[Dict[str, Any]]:
        """Parse simulated tool calls from message content"""
        if not hasattr(self, "choices") or not self.choices:
            return []

        content = self.choices[0].message.content or ""
        tool_calls = []

        # Parse tool calls from content using <TOOL_CALL> tags
        pattern = r"^\s*<TOOL_CALL>\s*(\{.*?\})\s*</TOOL_CALL>\s*$"
        matches = re.findall(pattern, content, re.DOTALL | re.MULTILINE)

        for i, match in enumerate(matches):
            try:
                tool_call_data = json.loads(match.strip())
                tool_calls.append(
                    {
                        "id": tool_call_data.get("id"),
                        "type": "function",
                        "function": {
                            "name": tool_call_data.get("name"),
                            "arguments": json.dumps(
                                tool_call_data.get("arguments", {})
                            ),
                        },
                    }
                )
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse simulated tool call JSON: {match}")

        return tool_calls

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
