# Unified LLM Python API Library

A unified interface for interacting with multiple LLM providers (OpenAI, Anthropic, etc.) with consistent API design.

## Features

- Single interface for multiple LLM providers
- Standardized request/response formats
- Easy provider configuration
- Type annotations and full documentation

## Installation
TODO

## Basic Usage

### Synchronous call
```
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_API_KEY=sk-xxxx
```

```python
from simplellm import LLMClient

# Initialize client for OpenAI
client = LLMClient(provider="deepseek")

response = client.completion(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.content)

# Or provide provider along with model
response = LLMClient(provider_model="deepseek/deepseek-chat").completion(
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.content)
```
