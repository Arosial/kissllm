from functools import wraps
from typing import Any, Callable, Dict, List, Optional, get_type_hints


class ToolRegistry:
    """Registry for tool functions"""

    _tools: Dict[str, Dict[str, Any]] = {}

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
    def get_tools_specs(cls) -> List[Dict[str, Any]]:
        """Get all registered tool specifications"""
        return [tool["spec"] for tool in cls._tools.values()]

    @classmethod
    def get_tool_function(cls, name: str) -> Optional[Callable]:
        """Get a registered tool function by name"""
        tool = cls._tools.get(name)
        return tool["function"] if tool else None

    @classmethod
    def execute_tool_call(cls, tool_call: Dict[str, Any]) -> Any:
        """Execute a tool call with the given parameters"""
        function_name = tool_call.get("function", {}).get("name")
        function_args = tool_call.get("function", {}).get("arguments", "{}")

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

    def get_tool_results(self) -> List[Dict[str, Any]]:
        """Get results from executed tool calls"""
        if hasattr(self, "tool_results") and self.tool_results:
            return self.tool_results

        # For non-streaming responses, execute tools on demand
        tool_results = []
        for tool_call in self.get_tool_calls():
            try:
                result = ToolRegistry.execute_tool_call(tool_call)
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

    def continue_with_tool_results(self, client, model=None):
        """Continue the conversation with tool results"""
        tool_results = self.get_tool_results()
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
