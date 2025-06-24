from typing import Optional

from openai.lib.streaming.chat import ChatCompletionStreamState
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion

from kissllm.tools import ToolManager, ToolMixin


class AccumulatedCompletionResponse(ToolMixin):
    def __init__(
        self,
        response: ParsedChatCompletion,
        tool_registry: Optional[ToolManager],
        use_flexible_toolcall=True,
    ):
        self.__dict__.update(response.__dict__)
        ToolMixin.__init__(self, tool_registry, use_flexible_toolcall)


class CompletionStream:
    def __init__(
        self, chunks, tool_registry: Optional[ToolManager], use_flexible_toolcall=True
    ):
        self.chunks = chunks
        self._tool_registry = tool_registry
        self._consumed = False
        self.callbacks = []
        self._state = ChatCompletionStreamState()
        self._role_defined = False
        self.use_flexible_toolcall = use_flexible_toolcall

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            chunk = await self.chunks.__anext__()

            # workaround for https://github.com/openai/openai-python/issues/2129
            if self._role_defined:
                chunk.choices[0].delta.role = None
            elif chunk.choices[0].delta.role:
                self._role_defined = True

            self._state.handle_chunk(chunk)
            return chunk

        except StopAsyncIteration:
            self._consumed = True
            for callback in self.callbacks:
                callback()
            raise

    async def iter_content(self, reasoning=True, include_tool_calls=True):
        if reasoning:
            reasoning_started = False
            async for chunk in self:
                # hardcoded reasoning_content or reasoning attribute.
                reasoning_content = getattr(
                    chunk.choices[0].delta, "reasoning_content", None
                )
                if not reasoning_content:
                    reasoning_content = getattr(
                        chunk.choices[0].delta, "reasoning", None
                    )
                if not reasoning_started and reasoning_content:
                    yield "<Reasoning>\n"
                    reasoning_started = True
                if reasoning_content:
                    yield reasoning_content

                content = chunk.choices[0].delta.content
                if reasoning_started and content:
                    yield "</Reasoning>\n\n"
                    reasoning_started = False
                if content:
                    yield content

                # Handle tool calls in streaming
                if (
                    include_tool_calls
                    and hasattr(chunk.choices[0].delta, "tool_calls")
                    and chunk.choices[0].delta.tool_calls
                ):
                    for tool_call_delta in chunk.choices[0].delta.tool_calls:
                        if tool_call_delta.function and tool_call_delta.function.name:
                            yield f"\n<Tool Call: {tool_call_delta.function.name}>\n"
                        if (
                            tool_call_delta.function
                            and tool_call_delta.function.arguments
                        ):
                            yield tool_call_delta.function.arguments
        else:
            async for chunk in self.iter():
                content = chunk.choices[0].delta.content
                if content:
                    yield content

    async def accumulate_stream(self):
        if not self._consumed:
            async for _ in self:  # Ensure stream is consumed
                pass
        parsed = self._state.get_final_completion()
        # Pass the registry to the accumulated response
        acc_response = AccumulatedCompletionResponse(
            parsed,
            self._tool_registry,
            use_flexible_toolcall=self.use_flexible_toolcall,
        )
        return acc_response
