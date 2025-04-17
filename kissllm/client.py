from typing import Any, Dict, List, Optional, Union

from openai.types.completion import Completion

from kissllm.observation.decorators import observe
from kissllm.providers import get_provider_driver
from kissllm.stream import CompletionStream
from kissllm.tools import ToolManager, ToolMixin


class CompletionResponse(ToolMixin):
    def __init__(self, response: Completion, tool_registry: Optional[ToolManager]):
        self.__dict__.update(response.__dict__)
        ToolMixin.__init__(self, tool_registry)


class LLMClient:
    """Unified LLM Client for multiple model providers"""

    def __init__(
        self,
        provider: str | None = None,
        provider_model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        tool_registry: Optional[ToolManager] = None,
    ):
        """
        Initialize LLM client with specific provider.

        Args:
            provider: Provider name (e.g. "openai", "anthropic").
            provider_model: Provider along with default model to use (e.g., "openai/gpt-4").
            api_key: Provider API key.
            base_url: Provider base URL.
            tool_registry: An optional ToolRegistry instance. If None, a new one is created.
        """
        self.default_model = None
        if provider_model:
            self.provider, self.default_model = provider_model.split("/", 1)
        if provider:
            self.provider = provider
        if self.provider is None:
            raise ValueError(
                "Provider must be specified either through provider or provider_model parameter"
            )
        self.provider_driver = get_provider_driver(self.provider)(
            self.provider, api_key=api_key, base_url=base_url
        )
        self.tool_registry = tool_registry

    def get_model(self, model):
        if model is None:
            model = self.default_model
        if model is None:
            raise ValueError(
                "Model must be specified either through model or provider_model parameter"
            )
        return model

    @observe
    async def async_completion(
        self,
        messages: List[Dict[str, str]],
        model: str | None = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = False,
        tools: Optional[List[Dict[str, Any]]] | bool = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs,
    ) -> Any:
        """Execute LLM completion with provider-specific implementation"""
        model = self.get_model(model)

        # Use registered tools from the client's registry if tools parameter is True
        if tools is True and self.tool_registry:
            tools = self.tool_registry.get_tools_specs()
            if not tools:
                # If tools=True but no tools are registered, don't send empty list
                # Some providers might error on empty tools list
                tools = None

        res = await self.provider_driver.async_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )
        if not stream:
            # Pass the client's tool registry to the response object
            return CompletionResponse(res, self.tool_registry)
        else:
            # Pass the client's tool registry to the stream object
            return CompletionStream(res, self.tool_registry)

    async def continue_with_tool_results(self, response, model=None):
        """Continue the conversation with tool results"""
        # Pass the client's tool registry to get results
        tool_results = await response.get_tool_results()
        if not tool_results:
            return None

        # Get the tool calls
        tool_calls = response.get_tool_calls()

        # Create messages for continuation
        messages = []

        for choice in response.choices:
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
        return await self.async_completion(messages=messages, model=model, stream=True)
