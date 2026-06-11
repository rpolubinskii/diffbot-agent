# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.11+ uv application. Production code lives in
`src/diffbot_agent/`. `main.py` defines the CLI entry point, `orchestrator.py`
coordinates command turns, runtime implementations live beside
`agent_runtime.py`, and integration clients are in `mcp_client.py` and
`audio_client.py`. Configuration parsing is centralized in `config.py`.

Protocol definitions are under `src/diffbot_agent/proto/`; update
`audio.proto` and regenerate `audio_pb2.py` rather than editing generated code
by hand. Add tests under `tests/`, mirroring module names such as
`tests/test_config.py`. Local configuration and SQLite session files are
intentionally ignored.