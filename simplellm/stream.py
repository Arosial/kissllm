from openai.lib.streaming.chat import ChatCompletionStreamState


class CompletionStream:
    def __init__(self, chunks):
        self.chunks = chunks
        self._openai_state = None
        self.callbacks = []

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
            state.handle_chunk(c)
            yield c
        self._openai_state = state
        for callback in self.callbacks:
            callback()

    def iter_content(self, reasoning=True):
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
        return parsed
