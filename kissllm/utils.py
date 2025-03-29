import os


def get_from_env(env, default=None) -> str:
    return os.environ.get(env.upper(), default)
