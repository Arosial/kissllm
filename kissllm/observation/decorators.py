import logging
from functools import partial

from kissllm.observation import get_observer
from kissllm.stream import CompletionStream
from kissllm.utils import get_from_env as get_from_env


def observe(func):
    """Decorator to add observation to methods using the configured observer"""

    async def wrapper(*args, **kwargs):
        observer = get_observer()
        if observer is None:
            return await func(*args, **kwargs)

        self = args[0]
        provider = self.provider
        model = kwargs.get("model")
        model = self.get_model(model)
        messages = kwargs.get("messages", [])
        model_parameters = {
            "temperature": kwargs.get("temperature"),
            "max_tokens": kwargs.get("max_tokens"),
        }

        try:
            observer.observe_completion_start(
                provider=provider,
                model=model,
                messages=messages,
                model_parameters=model_parameters,
            )
        except Exception as e:
            logging.warning(f"Failed to start observe completion: {str(e)}")

        response = await func(*args, **kwargs)
        try:
            if not isinstance(response, CompletionStream):
                observer.completion_end(response=response)
            else:
                response.register_callback(partial(observer.stream_end, response))
        except Exception as e:
            logging.warning(f"Failed to finish observe completion: {str(e)}")

        return response

    return wrapper
