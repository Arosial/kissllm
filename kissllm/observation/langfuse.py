from typing import Any, Dict, List

from langfuse import Langfuse

from kissllm.observation.base import BaseObserver


class LangfuseObserver(BaseObserver):
    """Langfuse implementation of observation"""

    def __init__(
        self,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str | None = None,
    ):
        self.langfuse = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        self.trace = None
        self.generation = None

    def observe_completion_start(
        self,
        provider: str,
        model: str,
        messages: List[Dict[str, str]],
        model_parameters: Dict[str, Any],
    ):
        self.trace = self.langfuse.trace(name=f"{provider}/{model}")

        self.generation = self.trace.generation(
            name="completion",
            model=model,
            model_parameters=model_parameters,
            input=messages,
            metadata={
                "provider": provider,
                "model": model,
                "parameters": model_parameters,
            },
        )

    def _format_response(self, response):
        if hasattr(response, "choices") and response.choices:
            content = response.choices[0].message.content
            metadata = {
                "type": "completion",
                "complete_response": response.model_dump()
                if hasattr(response, "model_dump")
                else str(response),
            }
        else:
            content = str(response)
            metadata = {"type": "completion", "complete_response": str(response)}

        return content, metadata

    def stream_end(self, response):
        response = response.accumulate_stream()
        content, metadata = self._format_response(response)
        if self.generation:
            self.generation.end(
                output=content,
                metadata=metadata,
            )

    def completion_end(self, response):
        content, metadata = self._format_response(response)
        if self.generation:
            self.generation.end(output=content, metadata=metadata)
