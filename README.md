# Unified LLM Python API Library

A unified interface for interacting with multiple LLM providers (OpenAI compatible only for now) with consistent API design.

## Features

- Single interface for multiple LLM providers
- Standardized request/response formats
- Easy provider configuration

## Basic Usage

### Async call
```
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_API_KEY=sk-xxxx
```

```python
from kissllm import LLMClient
from kissllm.tools import tool

# Example tool definition
@tool
def get_weather(location: str, unit: str = "celsius"):
    """Get current weather for a location"""
    # Mock implementation
    return f"Sunny 22°C in {location}"

# Initialize client with provider/model
client = LLMClient(provider_model="deepseek/deepseek-chat")

async def main():
    # Basic async completion
    response = await client.async_completion(
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        tools=True,
        tool_choice="auto"
    )
    print(response.choices[0].message.content)

    # Async streaming
    stream = await client.async_completion(
        messages=[{"role": "user", "content": "What's 15*27?"}],
        stream=True,
        tools=[{
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Calculate math expressions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"}
                    }
                }
            }
        }]
    )

    print("\nStreaming response:")
    async for chunk in stream.iter_content():
        print(chunk, end="")

    # Continue with tool results
    if response.tool_calls:
        continuation = await client.continue_with_tool_results(response)
        print("\n\nFinal answer:")
        print(continuation.choices[0].message.content)

# Run the async application
import asyncio
asyncio.run(main())
```
