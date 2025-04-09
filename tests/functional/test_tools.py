import os

import pytest
from dotenv import load_dotenv

from kissllm.client import LLMClient
from kissllm.tools import LocalToolManager, ToolManager

load_dotenv()
test_provider = os.environ["TEST_PROVIDER"]
test_model = os.environ["TEST_MODEL"]

local_manager = LocalToolManager()
register_tool = local_manager.register


# Define tool functions (without decorator initially)
@register_tool
def get_weather(location: str, unit: str = "celsius"):
    """Get the current weather in a given location"""
    # This is a mock implementation
    weather_data = {
        "New York": {"celsius": 22, "fahrenheit": 72, "condition": "Sunny"},
        "London": {"celsius": 18, "fahrenheit": 64, "condition": "Cloudy"},
        "Tokyo": {"celsius": 25, "fahrenheit": 77, "condition": "Rainy"},
    }

    if location not in weather_data:
        return f"Weather data for {location} not available"

    data = weather_data[location]
    temp = data[unit.lower()] if unit.lower() in data else data["celsius"]
    return f"The weather in {location} is {data['condition']} with a temperature of {temp}°{unit[0].upper()}"


@register_tool
def calculate(expression: str):
    """Calculate the result of a mathematical expression"""
    try:
        # Be careful with eval - this is just for demonstration
        # In production, use a safer evaluation method
        return eval(expression)
    except Exception as e:
        return f"Error calculating: {str(e)}"


@pytest.fixture(scope="function")
def tool_registry():
    """Creates managers and a ToolRegistry, registering local tools."""

    # Create the registry, passing the managers
    registry = ToolManager(local_manager=local_manager)
    return registry


@pytest.mark.asyncio
async def test_tool_calls(tool_registry):
    """Test tool calls functionality"""
    # Pass the pre-configured registry to the client
    client = LLMClient(
        provider_model=f"{test_provider}/{test_model}", tool_registry=tool_registry
    )

    # Test with specific tools
    response = await client.async_completion(
        messages=[
            {
                "role": "user",
                "content": "What's the weather in Tokyo and what is 15 * 7?",
            }
        ],
        tools=True,  # Use all registered tools
        tool_choice="auto",
        stream=True,
    )

    print("\nStreaming response with tool calls:")
    async for content in response.iter_content():
        print(content, end="", flush=True)
    print("\n")

    response = await response.accumulate_stream()
    # Get tool calls and results
    tool_calls = response.get_tool_calls()
    print("\nTool Calls:")
    for call in tool_calls:
        print(f"- {call['function']['name']}: {call['function']['arguments']}")

    # get_tool_results now uses the registry stored in the response object
    tool_results = await response.get_tool_results()
    print("\nTool Results:")
    for result in tool_results:
        print(f"- {result['tool_call_id']}: {result['content']}")

    # Continue conversation with tool results
    if tool_calls:
        print("\nContinuing conversation with tool results:")
        continuation = await client.continue_with_tool_results(response, test_model)

        async for content in continuation.iter_content():
            print(content, end="", flush=True)
        print("\n")


@pytest.mark.asyncio
async def test_single_tool_call(tool_registry):
    """Test using a single specific tool"""
    # Pass the pre-configured registry to the client
    # Although we specify tools manually here, the client needs the registry
    # if tool execution were needed later via get_tool_results.
    client = LLMClient(
        provider_model=f"{test_provider}/{test_model}", tool_registry=tool_registry
    )

    # Test with forcing a specific tool
    response = await client.async_completion(
        messages=[{"role": "user", "content": "What's 25 * 16?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Calculate the result of a mathematical expression",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    },
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "calculate"}},
        stream=False,
    )

    print("\nForced tool call response:")
    print(response.choices[0].message.content)

    if hasattr(response.choices[0].message, "tool_calls"):
        tool_calls = response.choices[0].message.tool_calls
        print("\nTool Calls:")
        for call in tool_calls:
            print(f"- {call.function.name}: {call.function.arguments}")

            # Execute the tool call using the registry associated with the response
            # (This demonstrates manual execution if needed, though response.get_tool_results() is preferred)
            tool_results = (
                await response.get_tool_results()
            )  # This uses the registry implicitly
            result = None
            for res in tool_results:
                if res["tool_call_id"] == call.id:
                    result = res["content"]
                    break

            print(f"  Result: {result}")
