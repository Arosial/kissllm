import logging
import os

import yaml


def get_from_env(env, default=None) -> str | None:
    return os.environ.get(env.upper(), default)


class PrettyDumper(yaml.Dumper):
    pass


def literal_presenter(dumper, data):
    if "\n" in data:
        # Remove trailing whitespace from each line.
        # "|" style is unable to represent trailing whitespace.
        if " \n" in data or "\t\n" in data:
            data = "\n".join(line.rstrip() for line in data.splitlines())
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    else:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)


PrettyDumper.add_representer(str, literal_presenter)


_PROMPT_LOG_LEVEL = None


def logging_prompt(logger, *messages):
    global _PROMPT_LOG_LEVEL
    if _PROMPT_LOG_LEVEL is None:
        _PROMPT_LOG_LEVEL = logging.getLevelNamesMapping().get(
            (get_from_env("PROMPT_LOG_LEVEL") or "").upper(), 100
        )

    for message in messages:
        logger.log(
            _PROMPT_LOG_LEVEL,
            yaml.dump(
                message,
                allow_unicode=True,
                default_flow_style=False,
                Dumper=PrettyDumper,
            ),
        )
