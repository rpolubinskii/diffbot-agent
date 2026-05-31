# diffbot-agent

Long-running Python orchestrator for DiffBot command turns.

V1 owns one OpenAI Agents SDK session, reads fresh `robot://status` from
`diffbot-mcp`, and sends each manual or voice command as a new turn into the
same session. The active runtime is `codex`, meaning an OpenAI Agents SDK
backend/profile, not a Codex CLI subprocess.

## Setup

```bash
uv sync
cp config.example.toml config.toml
```

Edit ignored `config.toml` for local ports, model selection, and the OpenAI API
key:

```toml
[secrets]
openai_api_key = "sk-..."
```

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

`OpenAICodexRuntime` is the only implementation in V1. Future backends such as
Claude, direct OpenAI API, local models, or CLI-backed runtimes should plug in
behind this boundary.

## Command Flow

For each command:

1. Apply the configured busy policy. V1 supports `ignore`.
2. Read `robot://status` from `diffbot-mcp`.
3. Fetch `diffbot.command_turn` from `diffbot-mcp` when available.
4. Send one user turn into the existing OpenAI Agents SDK `SQLiteSession`.
5. Stream the run to completion while the agent may call MCP tools.

The orchestrator does not start a new agent process or session per command.
