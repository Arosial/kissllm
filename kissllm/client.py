import logging
from typing import Any, Dict, List, Optional, Union

from openai.types.completion import Completion

from kissllm.observation.decorators import observe
from kissllm.providers import get_provider_driver
from kissllm.stream import CompletionStream
from kissllm.tools import ToolManager, ToolMixin
from kissllm.utils import logging_prompt

logger = logging.getLogger(__name__)


class DefaultResponseHandler:
    def __init__(self, messages):
        self.messages = messages

    async def accumulate_response(self, response):
        if isinstance(response, CompletionStream):
            response = await response.accumulate_stream()
        return response

    async def __call__(self, response):
        messages = self.messages
        response = await self.accumulate_response(response)
        if not response.get_tool_calls():
            messages.append(
                {
                    "role": "assistant",
                    "content": response.choices[0].message.content or "",
                }
            )
            return messages, False
        else:
            tool_results = await response.get_tool_results()

            messages.append(
                {
                    "role": "assistant",
                    "content": response.choices[0].message.content or "",
                    "tool_calls": response.get_tool_calls(),
                }
            )

            for result in tool_results:
                messages.append(result)
            return messages, True


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
            tools = None
            tool_choice = None

        logging_prompt(logger, "===Raw Prompt Messages:===", messages)
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

    async def async_completion_with_tool_execution(
        self,
        messages: List[Dict[str, str]],
        model: str | None = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = False,
        handle_response=None,
        max_steps=10,
        **kwargs,
    ):
        """Execute LLM completion with automatic tool execution until no more tool calls"""
        step = 0
        if handle_response is None:
            handle_response = DefaultResponseHandler(messages)

        while step < max_steps:
            step += 1
            response = await self.async_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                tools=True,
                tool_choice="auto",
                **kwargs,
            )
            messages, continu = await handle_response(response)
            if not continu:
                break
