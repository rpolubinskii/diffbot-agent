# diffbot-agent

Long-running Python orchestrator for DiffBot command turns.

V1 owns one OpenAI Agents SDK session, reads fresh `robot://status` from
`diffbot-mcp`, and sends each manual or voice command as a new turn into the
same session. The selected named agent profile can use OpenAI directly or
Ollama's OpenAI-compatible `/v1` API.

## Setup

```bash
uv sync
cp config.example.toml config.toml
```

Edit ignored `config.toml` for local ports, model selection, and the active
agent profile. For OpenAI, set the key inside the selected profile:

```toml
[agents.openai-main]
backend = "openai"
model = "gpt-5.1"
session_id = "diffbot-main"
session_db = "diffbot-agent.sqlite3"
openai_api_key = "sk-..."
```

For local Ollama, point the profile at Ollama's OpenAI-compatible endpoint and
make it active:

```toml
active_agent = "local-ollama"

[agents.local-ollama]
backend = "ollama"
model = "qwen3"
base_url = "http://localhost:11434/v1"
session_id = "diffbot-ollama"
session_db = "diffbot-agent.sqlite3"
api_key = "ollama"
```

Ollama tool-calling quality depends on the selected local model.

## Run

Start `diffbot-mcp` first, then run:

```bash
uv run diffbot-agent --config config.toml
```

Until diffbot-audio exposes a VTT command stream, keep:

```toml
[audio]
voice_stream_enabled = false
```

The agent then accepts manual commands on stdin:

```text
diffbot> stop
diffbot> describe what you can see
```

## Runtime Boundary

The internal runtime interface is:

```python
class AgentRuntime:
    async def start(self) -> None: ...
    async def run_turn(self, user_text: str, robot_status: str) -> None: ...
    async def stop(self) -> None: ...
```

`OpenAIAgentsRuntime` is the V1 implementation. It keeps the same Agents SDK,
MCP tool, streaming, and SQLite session flow for both OpenAI and Ollama
profiles.

## Command Flow

For each command:

1. Apply the configured `[agent_runtime]` busy policy. V1 supports `ignore`.
2. Read `robot://status` from `diffbot-mcp`.
3. Fetch `diffbot.command_turn` from `diffbot-mcp` when available.
4. Send one user turn into the existing OpenAI Agents SDK `SQLiteSession`.
5. Stream the run to completion while the agent may call MCP tools.

The orchestrator does not start a new agent process or session per command.
