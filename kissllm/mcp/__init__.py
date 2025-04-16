from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Union


@dataclass
class StdioMCPConfig:
    """Configuration for an MCP server connected via stdio."""

    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Optional[Dict[str, str]] = None
    type: Literal["stdio"] = "stdio"


@dataclass
class SSEMCPConfig:
    """Configuration for an MCP server connected via SSE."""

    name: str
    url: str
    type: Literal["sse"] = "sse"


MCPConfig = Union[StdioMCPConfig, SSEMCPConfig]


# Example mcp_servers.json structure:
"""
{
  "servers": [
    {
      "name": "My Stdio Server",
      "type": "stdio",
      "command": "python",
      "args": ["/path/to/my_mcp_server.py", "stdio"],
      "env": {"MY_VAR": "value"}
    },
    {
      "name": "My SSE Server",
      "type": "sse",
      "url": "http://localhost:8080/sse"
    },
    {
       // Minimal stdio config
      "command": "another_server_cmd"
    }
  ]
}
"""
