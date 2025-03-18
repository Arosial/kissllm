from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseDriver(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def __init__(
        self,
        provider: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """Initialize provider driver with credentials"""

    @abstractmethod
    async def async_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        **kwargs,
    ) -> Any:
        """Execute LLM completion request asynchronously"""
