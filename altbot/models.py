"""Team and agent definitions, loadable from YAML/JSON or plain dicts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

DEFAULT_MODEL = "claude-opus-5-5"
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
Mode = Literal["round_robin", "moderated"]


@dataclass
class Agent:
    name: str
    role: str
    emoji: str = "🤖"
    model: str | None = None  # falls back to Team.model
    effort: str | None = None  # falls back to Team.effort

    def __post_init__(self) -> None:
        self.name = self.name.strip()
        self.role = self.role.strip()
        if not self.name:
            raise ValueError("every agent needs a name")
        if not self.role:
            raise ValueError(f"agent '{self.name}' needs a role")
        if self.effort is not None and self.effort not in EFFORT_LEVELS:
            raise ValueError(f"agent '{self.name}': effort must be one of {EFFORT_LEVELS}")


@dataclass
class Team:
    name: str
    agents: list[Agent]
    # The leader moderates the discussion and writes the final answer.
    leader: Agent = field(
        default_factory=lambda: Agent(
            name="מנהל הצוות",
            emoji="🧭",
            role=(
                "You lead the team. You keep the discussion focused on the user's request, "
                "decide when the team has converged, and write the final deliverable."
            ),
        )
    )
    mode: Mode = "round_robin"
    rounds: int = 2  # round_robin: number of full rounds
    max_turns: int = 8  # moderated: max number of agent turns
    early_stop: bool = True  # round_robin: let the leader end the discussion early
    language: str = "Hebrew"
    model: str = DEFAULT_MODEL
    effort: str = "medium"
    fallbacks: bool = True  # server-side refusal fallbacks
    max_tokens: int = 16000

    def __post_init__(self) -> None:
        if not self.agents:
            raise ValueError("a team needs at least one agent")
        names = [a.name for a in self.agents] + [self.leader.name]
        if len(set(names)) != len(names):
            raise ValueError("agent names (including the leader) must be unique")
        if self.mode not in ("round_robin", "moderated"):
            raise ValueError("mode must be 'round_robin' or 'moderated'")
        if self.effort not in EFFORT_LEVELS:
            raise ValueError(f"effort must be one of {EFFORT_LEVELS}")
        self.rounds = max(1, min(int(self.rounds), 10))
        self.max_turns = max(1, min(int(self.max_turns), 40))

    def model_for(self, agent: Agent) -> str:
        return agent.model or self.model

    def effort_for(self, agent: Agent) -> str:
        return agent.effort or self.effort


def _agent(data: dict[str, Any]) -> Agent:
    known = {"name", "role", "emoji", "model", "effort"}
    return Agent(**{k: v for k, v in data.items() if k in known and v not in (None, "")})


def load_team(data: dict[str, Any]) -> Team:
    data = dict(data)
    agents = [_agent(a) for a in data.pop("agents", [])]
    leader = data.pop("leader", None)
    known = set(Team.__dataclass_fields__) - {"agents", "leader"}
    kwargs = {k: v for k, v in data.items() if k in known and v is not None}
    if leader:
        kwargs["leader"] = _agent(leader)
    return Team(agents=agents, **kwargs)


def load_team_file(path: str | Path) -> Team:
    with open(path, encoding="utf-8") as f:
        return load_team(yaml.safe_load(f))
