from kissllm.client import DefaultResponseHandler
from kissllm.stream import CompletionStream


class ResponseHandlerForTest(DefaultResponseHandler):
    async def accumulate_response(self, response):
        if isinstance(response, CompletionStream):
            print("\n======Streaming Assistant Response:======")
            async for content in response.iter_content():
                if not content:
                    continue
                print(content, end="", flush=True)
            print("\n")

        return await super().accumulate_response(response)
