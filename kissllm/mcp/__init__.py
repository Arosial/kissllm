from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Union


@dataclass
class StdioMCPConfig:
    """Configuration for an MCP server connected via stdio."""

    command: str
    args: List[str]
    env: Optional[Dict[str, str]] = None
    type: Literal["stdio"] = "stdio"


@dataclass
class SSEMCPConfig:
    """Configuration for an MCP server connected via SSE."""

    url: str
    type: Literal["sse"] = "sse"


MCPConfig = Union[StdioMCPConfig, SSEMCPConfig]
