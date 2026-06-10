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
history_commands = 4
full_tool_rounds = 6
```

`busy_policy = "ignore"` drops new voice or stdin commands while another command
turn is still running. `max_turns` counts model invocations within one command.
`history_commands` controls how many completed canonical command records are
included as recent memory; set it to `0` to disable recent command memory.
`full_tool_rounds` controls how many of the current command's latest model/tool
rounds remain exact; older rounds are compacted deterministically. Set it to `0`
to compact every completed tool round after the model has used it once.

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
requests/responses are captured in logs, with likely secrets and image payloads
redacted by default. Sensitive SDK trace payloads are disabled.

## Runtime Boundary

The internal runtime interface is:

```python
class AgentRuntime:
    async def start(self) -> None: ...
    async def run_turn(self, command: str, robot_status: str) -> None: ...
    async def stop(self) -> None: ...
```

`OpenAIAgentsRuntime` is the V1 implementation. It keeps the same Agents SDK,
MCP tool, streaming, and SQLite session flow for both OpenAI and Ollama
profiles.

## Command Flow

For each command:

1. Apply the configured `[agent_runtime]` busy policy. V1 supports `ignore`.
2. Read `robot://status` from `diffbot-mcp`.
3. Load the latest canonical command-memory records and compose the command
   prompt inside the runtime. Fresh `robot://status` is authoritative.
4. Send one user turn through the OpenAI Agents SDK, with the run capped by
   `[agent_runtime].max_turns`. Persisted raw SDK history is excluded from model
   input across operator commands.
5. Before every model request, keep the configured number of latest tool rounds
   exact, compact older rounds, and replace already-consumed camera images.
6. Stream the run to completion while the agent may call MCP tools.
7. Store one canonical command record for completed, failed, or max-turn runs.

Raw SDK rows remain available in the same SQLite file for debugging, but image
data is removed before those rows are persisted. Canonical records are stored in
the `command_memories` table and contain the original command, completion
status, assistant and spoken text, compact tool outcomes, and deterministic
searchable text. Reasoning, status snapshots, acknowledgements, telemetry, and
image data are excluded.

`--reset-session` clears both SDK history and canonical command records for the
active profile. The orchestrator does not start a new agent process per command.
