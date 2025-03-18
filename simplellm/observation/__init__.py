from typing import Dict, Optional, Type

from simplellm.observation.base import BaseObserver
from simplellm.observation.langfuse import LangfuseObserver
from simplellm.utils import get_from_env as get_from_env

# Observer registry mapping IDs to classes
_OBSERVER_REGISTRY: Dict[str, Type["BaseObserver"]] = {"langfuse": LangfuseObserver}
_observer_id: Optional[str] = None
_observer_kwargs: Dict = {}


def configure_observer(observer: Optional[str] = None, **kwargs):
    """Configure the global observer for LLM completions

    Args:
        observer: Observer ID string (e.g. "langfuse") or None to disable
        **kwargs: Observer-specific configuration
    """
    global _observer_id, _observer_kwargs
    _observer_id = observer or get_from_env("LLM_OBSERVER")
    _observer_kwargs = kwargs


def get_observer() -> Optional["BaseObserver"]:
    """Get the configured observer instance"""
    if _observer_id is None or _observer_id not in _OBSERVER_REGISTRY:
        return None
    return _OBSERVER_REGISTRY[_observer_id](**_observer_kwargs)
