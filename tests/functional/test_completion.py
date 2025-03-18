import os

import pytest
from dotenv import load_dotenv

from simplellm.client import LLMClient

load_dotenv()
test_provider = os.environ["TEST_PROVIDER"]
test_model = os.environ["TEST_MODEL"]


@pytest.mark.asyncio
async def test_completion():
    """Test provider through the OpenAI-compatible interface"""
    client = LLMClient(provider=test_provider)

    # Test basic completion
    response = await client.async_completion(
        messages=[{"role": "user", "content": "Hello"}],
        model=test_model,
        temperature=0.5,
    )
    # Verify response structure
    assert hasattr(response, "choices"), "Response should have 'choices' field"
    assert len(response.choices) > 0, "Response should contain at least one choice"
    print(response.choices[0].message.content)


@pytest.mark.asyncio
async def test_completion_streaming():
    """Test provider streaming functionality"""
    # Test streaming completion
    cli = LLMClient(provider_model=f"{test_provider}/{test_model}")
    stream = await cli.async_completion(
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.5,
        stream=True,
    )

    async for c in stream.iter_content():
        print(c, end="", flush=True)
    print()

    response = await stream.accumulate_stream()
    print("Accumulated content:")
    msg = response.choices[0].message
    print(msg)
