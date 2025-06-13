import json
import logging
import os

import yaml


def get_from_env(env: str, default=None) -> str | None:
    """Retrieve a value from environment variables, converting the key to uppercase.

    Args:
        env: The environment variable name.
        default: Default value if the environment variable is not set.

    Returns:
        The value of the environment variable or the default value.
    """
    return os.environ.get(env.upper(), default)


class PrettyDumper(yaml.Dumper):
    """Custom YAML dumper for improved readability."""

    pass


def literal_presenter(dumper, data):
    """Custom YAML presenter for human-readable string formatting.

    Attempts to format strings with newlines as literal blocks (| style).
    WARNING: This may not preserve the original object structure when reloaded.
    """

    if "\n" in data:
        # Remove trailing whitespace for compatibility with literal block style.
        if " \n" in data or "\t\n" in data:
            data = "\n".join(line.rstrip() for line in data.splitlines())
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    else:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)


PrettyDumper.add_representer(str, literal_presenter)


def logging_prompt(logger: logging.Logger, *messages):
    """Log messages with a custom log level for prompts.

    The log level is determined by the PROMPT_LOG_LEVEL environment variable.
    If not set, defaults to a level of 100 (custom level).

    Args:
        logger: The logger instance to use.
        *messages: Messages to log, formatted as YAML for readability.
    """

    if not hasattr(logging_prompt, "_PROMPT_LOG_LEVEL"):
        logging_prompt._PROMPT_LOG_LEVEL = logging.getLevelNamesMapping().get(
            (get_from_env("PROMPT_LOG_LEVEL") or "").upper(), 100
        )

    def recursive_convert(obj):
        if isinstance(obj, dict):
            return {k: recursive_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_convert(v) for v in obj]
        elif isinstance(obj, str):
            try:
                # Parse JSON-like strings for better YAML representation.
                return json.loads(obj)
            except Exception:
                return obj
        else:
            return obj

    for message in messages:
        logger.log(
            logging_prompt._PROMPT_LOG_LEVEL,
            yaml.dump(
                recursive_convert(message),
                allow_unicode=True,
                default_flow_style=False,
                Dumper=PrettyDumper,
            ),
        )
