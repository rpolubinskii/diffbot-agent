from __future__ import annotations

from starlette.testclient import TestClient

from diffbot_agent.agent_runtime import TurnResult
from diffbot_agent.server import create_app
from diffbot_agent.session_usage import SessionUsage


class FakeRuntime:
    def __init__(self) -> None:
        self.usage = SessionUsage("fake-model")
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def run_turn(self, command: str) -> TurnResult:
        return TurnResult(status="completed", final_text=f"done: {command}")

    async def reset(self) -> None:
        self.usage.reset()

    async def stop(self) -> None:
        self.stopped = True


def test_server_health_agent_card_and_mcp_route() -> None:
    runtime = FakeRuntime()
    app = create_app(runtime, host="127.0.0.1", port=8090)

    with TestClient(app, base_url="http://127.0.0.1:8090") as client:
        health = client.get("/health")
        card = client.get("/.well-known/agent-card.json")
        mcp_options = client.options("/mcp")

    assert runtime.started is True
    assert runtime.stopped is True
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert card.status_code == 200
    assert card.json()["name"] == "DiffBot Agent"
    assert mcp_options.status_code in {200, 405}
