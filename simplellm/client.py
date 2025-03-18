from typing import Any, Dict, List, Optional, Union

from openai.types.completion import Completion

from simplellm.observation.decorators import observe
from simplellm.providers import get_provider_driver
from simplellm.stream import CompletionStream
from simplellm.tools import ToolMixin, ToolRegistry


class CompletionResponse(ToolMixin):
    def __init__(self, response: Completion):
        self.__dict__.update(response.__dict__)


class LLMClient:
    """Unified LLM Client for multiple model providers"""

    def __init__(
        self,
        provider: str | None = None,
        provider_model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        """
        Initialize LLM client with specific provider

        Args:
            provider: Provider name (e.g. "openai", "anthropic")
            provider_model: Provider along with default model to use.
            api_key: Provider API key
            base_url: Provider base url
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

        # Use registered tools if tools parameter is True
        if tools is True:
            tools = ToolRegistry.get_tools_specs()

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
            return CompletionResponse(res)
        else:
            return CompletionStream(res)

    async def continue_with_tool_results(self, response, model=None):
        """Continue the conversation with tool results"""
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
