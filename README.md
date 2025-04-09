# Unified LLM Python API Library

A unified interface for interacting with multiple LLM providers (OpenAI compatible only for now) with consistent API design.

## Features

- Single interface for multiple LLM providers
- Standardized request/response formats
- Easy provider configuration
- MCP and local Tool support

## Usage

See [tests/functional](tests/functional) for usage.

Most functional tests read env with dotenv. An example:
```
$ cat .env
TEST_PROVIDER=deepseek
TEST_MODEL=deepseek-chat
DEEPSEEK_API_BASE = "https://api.deepseek.com/beta"
DEEPSEEK_API_KEY = "sk-your-api-key"
```
