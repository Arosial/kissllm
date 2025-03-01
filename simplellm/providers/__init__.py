import importlib
import pkgutil
from pathlib import Path
from typing import Type

from simplellm.providers.base import BaseDriver
from simplellm.utils import get_from_env

PROVIDER_DRIVERS = {}
DEFAULT_DRIVER = None


def register_provider_driver(driver: Type[BaseDriver], as_default=False):
    """Register provider driver, optionally as default"""
    global PROVIDER_DRIVERS
    PROVIDER_DRIVERS[driver.id] = driver
    if as_default:
        global DEFAULT_DRIVER
        DEFAULT_DRIVER = driver


def get_provider_driver(provider: str) -> Type[BaseDriver]:
    """Get registered provider driver class with fallback handling"""
    provider_id = get_from_env(f"{provider}_driver", provider)
    driver_cls = PROVIDER_DRIVERS.get(provider_id) or DEFAULT_DRIVER
    return driver_cls


def _auto_import_providers():
    """Automatically scan and import all provider modules"""
    package_dir = Path(__file__).parent
    for _, module_name, _ in pkgutil.iter_modules([str(package_dir)]):
        # Skip base.py and files starting with _
        if module_name != "base" and not module_name.startswith("_"):
            importlib.import_module(f"simplellm.providers.{module_name}")


# Automatically import all providers
_auto_import_providers()
