from typing import Any, Dict, List, Optional

from simplellm.observation.decorators import observe
from simplellm.providers import get_provider_driver
from simplellm.stream import CompletionStream


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
    def completion(
        self,
        messages: List[Dict[str, str]],
        model: str | None = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = False,
        **kwargs,
    ) -> Any:
        """Execute LLM completion with provider-specific implementation"""
        model = self.get_model(model)
        res = self.provider_driver.completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )
        if not stream:
            return res
        else:
            return CompletionStream(res)
