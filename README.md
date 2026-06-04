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

Runtime behavior is configured under `[agent_runtime]`:

```toml
[agent_runtime]
busy_policy = "ignore"
max_turns = 50
```

`busy_policy = "ignore"` drops new voice or stdin commands while another command
turn is still running. `max_turns` controls the OpenAI Agents SDK turn-loop cap
for a single command.

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

Ollama tool-calling and visual analysis quality depend on the selected local
model. To inspect camera images returned by `vision.get_camera_image`, use a
vision-capable Ollama model.

`vision.get_camera_image` returns image content from `diffbot-mcp`. OpenAI and
Ollama multimodal models can inspect that image directly. Text-only models may
call the tool but cannot reason about the returned image.

## Run

Start `diffbot-mcp` first, then run:

```bash
uv run diffbot-agent --config config.toml
```

To receive finalized VTT commands directly from `diffbot-audio`, enable the
voice command stream:

```toml
[audio]
host = "localhost"
port = 50052
voice_stream_enabled = true
reconnect_delay_seconds = 2.0
```

For manual development without audio, set `voice_stream_enabled = false`. The
agent then accepts commands on stdin:

```text
diffbot> stop
diffbot> describe what you can see
```

Operational logs are written to stderr as one compact JSON payload per line.
Assistant responses are not printed to stdout; LLM inputs/responses and MCP
requests/responses are captured in logs, with likely secrets redacted by default.

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
3. Compose the command-turn prompt locally in this service.
4. Send one user turn into the existing OpenAI Agents SDK `SQLiteSession`, with
   the run capped by `[agent_runtime].max_turns`.
5. Stream the run to completion while the agent may call MCP tools.

The orchestrator does not start a new agent process or session per command.
