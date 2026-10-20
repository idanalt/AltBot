import json

import pytest
from fastapi.testclient import TestClient

from altbot import Orchestrator, load_team, load_team_file
from altbot.web import TEAMS_DIR, app


class FakeLLM:
    """Records calls; each agent 'says' a fixed line, the leader follows a script."""

    def __init__(self, decisions=()):
        self.decisions = list(decisions)
        self.prompts = []

    async def stream_text(self, *, model, system, prompt, effort, max_tokens):
        self.prompts.append(prompt)
        name = system.split(",", 1)[0].removeprefix("You are ")
        for word in (f"hello from {name}", "!"):
            yield word

    async def decide(self, *, model, system, prompt, schema, effort):
        return self.decisions.pop(0)


TEAM = {
    "name": "t",
    "agents": [{"name": "A", "role": "a"}, {"name": "B", "role": "b"}],
    "leader": {"name": "L", "role": "lead"},
    "rounds": 3,
}


async def collect(orch, task="do it"):
    return [e async for e in orch.run(task)]


def speakers(events):
    return [e["agent"] for e in events if e["type"] == "turn_start"]


@pytest.mark.asyncio
async def test_round_robin_runs_all_rounds_then_leader():
    llm = FakeLLM([{"done": False, "reason": "more"}, {"done": False, "reason": "more"}])
    events = await collect(Orchestrator(load_team(TEAM), llm))
    assert speakers(events) == ["A", "B", "A", "B", "A", "B", "L"]
    assert events[-2] == {"type": "final", "text": "hello from L!"}
    assert events[-1]["type"] == "done"
    # later speakers see earlier messages
    assert "[A]\nhello from A!" in llm.prompts[1]


@pytest.mark.asyncio
async def test_round_robin_early_stop():
    llm = FakeLLM([{"done": True, "reason": "converged"}])
    events = await collect(Orchestrator(load_team(TEAM), llm))
    assert speakers(events) == ["A", "B", "L"]
    assert any(e["type"] == "decision" and e["text"] == "converged" for e in events)


@pytest.mark.asyncio
async def test_moderated_follows_leader_choices():
    team = load_team({**TEAM, "mode": "moderated", "max_turns": 5})
    llm = FakeLLM([
        {"next_speaker": "B", "instruction": "start"},
        {"next_speaker": "B", "instruction": "again"},
        {"next_speaker": "A", "instruction": "review"},
        {"next_speaker": "FINISH", "instruction": "done"},
    ])
    events = await collect(Orchestrator(team, llm))
    assert speakers(events) == ["B", "B", "A", "L"]
    assert "asks you: review" in llm.prompts[2]


def test_validation():
    with pytest.raises(ValueError):
        load_team({"name": "x", "agents": []})
    with pytest.raises(ValueError):
        load_team({"name": "x", "agents": [{"name": "A", "role": "r"}, {"name": "A", "role": "r"}]})


def test_example_teams_load():
    paths = list(TEAMS_DIR.glob("*.yaml"))
    assert paths
    for p in paths:
        load_team_file(p)


def test_web_endpoints(monkeypatch):
    import altbot.web as web

    real = web.Orchestrator
    monkeypatch.setattr(web, "Orchestrator", lambda team: real(team, FakeLLM([{"done": True, "reason": "ok"}])))
    client = TestClient(app)
    assert client.get("/").status_code == 200
    assert len(client.get("/api/teams").json()) >= 3
    res = client.post("/api/run", json={"task": "x", "team": TEAM})
    events = [json.loads(l[6:]) for l in res.text.split("\n\n") if l.startswith("data: ")]
    assert events[-1]["type"] == "done"
    assert client.post("/api/run", json={"task": "x", "team": {"name": "x", "agents": []}}).status_code == 400
