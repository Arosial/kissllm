from abc import ABC, abstractmethod
from typing import Any, Dict, List

from kissllm.utils import get_from_env as get_from_env


class BaseObserver(ABC):
    """Abstract base class for observation implementations"""

    @abstractmethod
    def observe_completion_start(
        self,
        provider: str,
        model: str,
        messages: List[Dict[str, str]],
        params: Dict[str, Any],
    ):
        pass

    @abstractmethod
    def stream_end(self, response):
        pass

    @abstractmethod
    def completion_end(self, response):
        pass
