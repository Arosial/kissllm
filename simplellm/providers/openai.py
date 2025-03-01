from typing import Any, Dict, List, Optional

from openai import OpenAI

from simplellm import providers
from simplellm.providers.base import BaseDriver
from simplellm.utils import get_from_env


class GeneralOpenAIDriver(BaseDriver):
    """OpenAI API provider implementation"""

    id = "general_openai"

    def __init__(
        self,
        provider: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.provider = provider
        self.client = self.get_client(api_key, base_url)

    def get_client(self, api_key: Optional[str], base_url: Optional[str]):
        """Create configured OpenAI client with fallback to env vars"""
        return OpenAI(
            api_key=api_key or get_from_env(f"{self.provider.upper()}_API_KEY"),
            base_url=base_url
            or get_from_env(f"{self.provider.upper()}_BASE_URL")
            or get_from_env(f"{self.provider.upper()}_API_BASE"),
        )

    def completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = False,
    ) -> Any:
        """Execute OpenAI chat completion"""
        return self.client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )


providers.register_provider_driver(GeneralOpenAIDriver, as_default=True)
