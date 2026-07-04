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
compact_threshold = 240000
full_tool_rounds = 6

[memory]
backend = "none"
```

`busy_policy = "ignore"` drops new voice or stdin commands while another command
turn is still running. `max_turns` counts model invocations within one command.

The conversation thread is the agent's short-term memory: it is owned by the SDK
session and carried across commands. Each command enters the thread as a timestamped
user turn, and a fresh robot-status note is appended to every model request. Context
stays bounded per backend: on the OpenAI path the Responses API compacts server-side
via `context_management` once the running context crosses `compact_threshold` tokens;
the Ollama chat-completions path has no server compaction, so it keeps the latest
`full_tool_rounds` tool rounds exact and compacts older rounds locally
(`full_tool_rounds = 0` compacts every completed round after one use). Camera-image
freshness is managed in both cases: only the latest valid image is kept, and it is
invalidated when the robot moves; older frames in the thread are stored as text
placeholders, so they can never be mistaken for the current view.

`[memory].backend` selects the cross-command long-term memory backend: `none` (the
default) or `diffbot_memory` (persistent Graphiti memory via diffbot-mcp). Recall is
deliberate — the model calls diffbot-mcp's `memory.recall` tool when it needs durable
facts (locations, people, past outcomes); the backend only persists a distilled
episode after each command via `memory.remember`. Writes are fire-and-forget, so the
agent runs whether or not the service is reachable. Tool categories (speech/navigation/safety/status/vision/tool)
are owned by diffbot-mcp: each tool advertises its category in its MCP `_meta`, and the
agent reads that map at startup and classifies nothing itself (an unadvertised tool falls
back to `tool`). Categories drive episode classification and camera-image invalidation.

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
2. Append the command to the SDK conversation thread as a timestamped user turn.
3. Send one user turn through the OpenAI Agents SDK, with the run capped by
   `[agent_runtime].max_turns`. The thread (short-term memory) carries across
   commands; long-term recall is deliberate, via the `memory.recall` tool.
4. Before every model request, a context filter does only two robot-specific
   things: keep the latest valid camera image (older frames are already text
   placeholders) and bound the thread length on the Ollama path (sliding window;
   the OpenAI path compacts server-side).
5. Stream the run to completion while the agent may call MCP tools.
6. After the run, write one distilled episode to long-term memory via
   `memory.remember` (when `[memory].backend = "diffbot_memory"`).

The episode is mcp-agnostic: the original command, completion status, the model's
final text, and an opaque list of tool calls (name, mcp-provided category,
arguments, and a bounded text preview of the output). Status reads are dropped as
noise; tool outputs are not parsed — the memory service's LLM extracts the facts.

`--reset-session` clears the SDK conversation thread for the active profile before
the process starts. Typing `/reset` during a session does the same without a restart
(the persistent memory graph is not wiped). It is handled between turns — like any
command, a `/reset` sent while a turn is running is ignored under `busy_policy = "ignore"`.
The orchestrator does not start a new agent process per command.

## Agent Interop Server

Coding agents can delegate robot-facing requests to the same long-running
`diffbot-agent` runtime instead of calling `diffbot-mcp` directly:

```bash
uv run diffbot-agent serve --config config.toml --host 127.0.0.1 --port 8090
```

The server exposes:

- A2A JSON-RPC at `http://127.0.0.1:8090/`.
- The A2A agent card at `http://127.0.0.1:8090/.well-known/agent-card.json`.
- Streamable HTTP MCP at `http://127.0.0.1:8090/mcp`.
- Health/status at `http://127.0.0.1:8090/health`.

For Codex, add the MCP endpoint in a trusted project `.codex/config.toml` or
the global Codex config:

```toml
[mcp_servers.diffbot_agent]
url = "http://127.0.0.1:8090/mcp"
required = false
tool_timeout_sec = 600
```

For Claude Code:

```bash
claude mcp add --transport http diffbot-agent http://127.0.0.1:8090/mcp
```

The main MCP tool is:

```text
diffbot_agent.run_command(command, timeout_seconds=600)
```

It sends one command through the existing DiffBot agent session and returns the
final textual result plus bounded structured metadata. `diffbot_agent.status`
reports server/session state, and `diffbot_agent.reset_session` clears the
conversation session.
