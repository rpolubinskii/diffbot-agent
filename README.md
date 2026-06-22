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
compact_threshold = 240000
full_tool_rounds = 6

[memory]
backend = "sqlite"
```

`busy_policy = "ignore"` drops new voice or stdin commands while another command
turn is still running. `max_turns` counts model invocations within one command.
`history_commands` controls how many completed canonical command records the
memory backend recalls each turn; set it to `0` to disable recent command memory.

In-command context stays bounded differently per backend. On the OpenAI path the
Responses API compacts server-side via `context_management` once the running
context crosses `compact_threshold` tokens. The Ollama chat-completions path has
no server compaction, so it keeps the latest `full_tool_rounds` tool rounds exact
and compacts older rounds locally (`full_tool_rounds = 0` compacts every completed
round after one use). Camera-image freshness is managed in both cases: only the
latest valid image is kept, and it is invalidated when the robot moves.

`[memory].backend` selects the cross-command memory backend: `sqlite` (recency-based
local records, the default) or `none`. The backend is pluggable behind a small
interface so a future `diffbot-rag` (Graphiti) backend can replace it without
touching the runtime. Tool categories (speech/navigation/safety/status/vision) are
advertised by diffbot-mcp in each tool's MCP `_meta` and read at startup — the agent
does not enumerate tools. Override a specific tool under `[tool_categories]`.

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
Configure `[logging] level = "info"` (the default) to print called tool names,
available model reasoning, warnings, and errors. Use `level = "debug"` to retain
the detailed LLM, MCP, resource, audio, and timing events. Likely secrets and
image payloads are redacted in both modes. Assistant responses are not printed
to stdout, and sensitive SDK trace payloads are disabled.

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
3. Recall historical memory from the configured backend and compose the command
   prompt inside the runtime.
4. Send one user turn through the OpenAI Agents SDK, with the run capped by
   `[agent_runtime].max_turns`. Persisted raw SDK history is excluded from model
   input across operator commands.
5. Before every model request, a context filter rebuilds the input: it replaces
   the leading item with the current command, robot status, and timeline; stamps
   tool outputs and assistant messages with elapsed timestamps; keeps only the
   latest valid camera image; and appends the historical-memory block.
6. Bound the in-command context: server-side compaction on the OpenAI path,
   local round compaction on the Ollama path.
7. Stream the run to completion while the agent may call MCP tools.
8. Store one canonical command record for completed, failed, or max-turn runs.

Raw SDK rows remain available in the same SQLite file for debugging, but image
data is removed before those rows are persisted. Canonical records are stored in
the `command_memories` table and contain the original command, completion
status, assistant and spoken text, and compact tool outcomes. Reasoning, status
snapshots, acknowledgements, telemetry, and image data are excluded.

`--reset-session` clears both SDK history and canonical command records for the
active profile before the process starts. A live "reset context" / "clear memory"
command does the same without a restart. The orchestrator does not start a new
agent process per command.
