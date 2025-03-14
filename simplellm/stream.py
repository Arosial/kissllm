from typing import Any, Dict, List

from openai.lib.streaming.chat import ChatCompletionStreamState
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion

from simplellm.tools import ToolMixin, ToolRegistry


class AccumulatedCompletionResponse(ToolMixin):
    def __init__(self, response: ParsedChatCompletion):
        self.__dict__.update(response.__dict__)


class CompletionStream:
    def __init__(self, chunks):
        self.chunks = chunks
        self._openai_state = None
        self.callbacks = []
        self.tool_calls = []
        self.current_tool_call = None
        self.tool_results = []

    def register_callback(self, func):
        self.callbacks.append(func)

    def iter(self):
        state = ChatCompletionStreamState()
        role_defined = False
        for c in self.chunks:
            # workaround for https://github.com/openai/openai-python/issues/2129
            if role_defined:
                c.choices[0].delta.role = None
            elif c.choices[0].delta.role:
                role_defined = True

            # Track tool calls
            delta = c.choices[0].delta
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                for tool_call_delta in delta.tool_calls:
                    index = tool_call_delta.index

                    # Initialize new tool call if needed
                    if len(self.tool_calls) <= index:
                        self.tool_calls.append(
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        )

                    # Update tool call with delta information
                    if tool_call_delta.id:
                        self.tool_calls[index]["id"] = tool_call_delta.id

                    if tool_call_delta.function:
                        if tool_call_delta.function.name:
                            self.tool_calls[index]["function"]["name"] = (
                                tool_call_delta.function.name
                            )
                        if tool_call_delta.function.arguments:
                            self.tool_calls[index]["function"]["arguments"] += (
                                tool_call_delta.function.arguments
                            )

            state.handle_chunk(c)
            yield c

        self._openai_state = state

        # Execute tool calls if any
        for tool_call in self.tool_calls:
            try:
                result = ToolRegistry.execute_tool_call(tool_call)
                self.tool_results.append(
                    {
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "content": str(result),
                    }
                )
            except Exception as e:
                self.tool_results.append(
                    {
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "content": f"Error executing tool: {str(e)}",
                    }
                )

        for callback in self.callbacks:
            callback()

    def iter_content(self, reasoning=True, include_tool_calls=True):
        if reasoning:
            reasoning_started = False
            for chunk in self.iter():
                reasoning_content = getattr(
                    chunk.choices[0].delta, "reasoning_content", None
                )
                if not reasoning_started and reasoning_content:
                    yield "<Reasoning>\n"
                    reasoning_started = True
                if reasoning_content:
                    yield reasoning_content

                content = chunk.choices[0].delta.content
                if reasoning_started and content:
                    yield "</Reasoning>\n"
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
            for chunk in self.iter():
                content = chunk.choices[0].delta.content
                if content:
                    yield content

    def accumulate_stream(self):
        if self._openai_state is None:
            for _ in self.iter():
                pass
        parsed = self._openai_state.get_final_completion()
        return AccumulatedCompletionResponse(parsed)

    def get_tool_calls(self) -> List[Dict[str, Any]]:
        """Get all tool calls from the stream"""
        if self._openai_state is None:
            for _ in self.iter():
                pass
        return self.tool_calls

    def get_tool_results(self) -> List[Dict[str, Any]]:
        """Get results from executed tool calls"""
        if self._openai_state is None:
            for _ in self.iter():
                pass
        return super().get_tool_results()
