"""Runs a team discussion and yields events describing everything that happens.

Event types (all are plain dicts, ready to be sent as JSON):
  start       {team, mode, agents: [{name, emoji, role}], leader}
  phase       {label}
  turn_start  {agent, emoji, final: bool}
  delta       {agent, text}
  turn_end    {agent, text}
  decision    {agent, emoji, text}
  error       {agent, message}
  final       {text}
  done        {}
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import anthropic

from .llm import LLM, ClaudeLLM, RefusalError
from .models import Agent, Team

FINISH = "FINISH"


@dataclass
class Message:
    agent: str
    text: str


def _roster(team: Team) -> str:
    lines = [f"- {a.name}: {a.role}" for a in team.agents]
    lines.append(f"- {team.leader.name} (leader): {team.leader.role}")
    return "\n".join(lines)


def _transcript(history: list[Message]) -> str:
    if not history:
        return "(nobody has spoken yet)"
    return "\n\n".join(f"[{m.agent}]\n{m.text}" for m in history)


class Orchestrator:
    def __init__(self, team: Team, llm: LLM | None = None):
        self.team = team
        self.llm = llm or ClaudeLLM(fallbacks=team.fallbacks)

    # ---------- prompts ----------

    def _agent_system(self, agent: Agent) -> str:
        t = self.team
        return f"""You are {agent.name}, one member of a team of AI agents collaborating on a request from the user.

Your role:
{agent.role}

The team:
{_roster(t)}

How the discussion works: members take turns speaking in a shared discussion. When it is your turn, read everything said so far and contribute from the perspective of your role. Build on, challenge or refine what others said and address them by name. Add something new rather than repeating points already made. If you disagree, say so and explain why. Keep each turn focused and reasonably short (roughly 100-300 words) unless the task genuinely needs more detail. Do not write the final deliverable for the user; the leader, {t.leader.name}, will do that at the end based on the discussion.

Write in {t.language}."""

    def _leader_system(self) -> str:
        t = self.team
        return f"""You are {t.leader.name}, the leader of a team of AI agents collaborating on a request from the user.

Your role:
{t.leader.role}

The team:
{_roster(t)}

Write in {t.language}."""

    def _turn_prompt(self, task: str, history: list[Message], agent: Agent, note: str) -> str:
        return f"""<user_request>
{task}
</user_request>

<discussion>
{_transcript(history)}
</discussion>

{note}
It is now your turn, {agent.name}. Speak as {agent.name}; write only your own contribution."""

    # ---------- streaming helpers ----------

    async def _speak(
        self, agent: Agent, system: str, prompt: str, final: bool = False
    ) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "turn_start", "agent": agent.name, "emoji": agent.emoji, "final": final}
        parts: list[str] = []
        try:
            async for chunk in self.llm.stream_text(
                model=self.team.model_for(agent),
                system=system,
                prompt=prompt,
                effort=self.team.effort_for(agent),
                max_tokens=self.team.max_tokens,
            ):
                parts.append(chunk)
                yield {"type": "delta", "agent": agent.name, "text": chunk}
        except (anthropic.APIError, RefusalError) as e:
            yield {"type": "error", "agent": agent.name, "message": str(e)}
        yield {"type": "turn_end", "agent": agent.name, "text": "".join(parts)}

    async def _decide(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any] | None:
        leader = self.team.leader
        try:
            return await self.llm.decide(
                model=self.team.model_for(leader),
                system=self._leader_system(),
                prompt=prompt,
                schema=schema,
                effort="low",
            )
        except (anthropic.APIError, RefusalError, ValueError, StopIteration):
            return None

    # ---------- main loop ----------

    async def run(self, task: str) -> AsyncIterator[dict[str, Any]]:
        t = self.team
        task = task.strip()
        if not task:
            raise ValueError("the task is empty")
        history: list[Message] = []

        yield {
            "type": "start",
            "team": t.name,
            "mode": t.mode,
            "agents": [{"name": a.name, "emoji": a.emoji, "role": a.role} for a in t.agents],
            "leader": {"name": t.leader.name, "emoji": t.leader.emoji, "role": t.leader.role},
        }

        turns = self._round_robin(task, history) if t.mode == "round_robin" else self._moderated(task, history)
        async for event in turns:
            yield event

        # Final synthesis by the leader.
        yield {"type": "phase", "label": "תוצאה סופית"}
        prompt = f"""<user_request>
{task}
</user_request>

<discussion>
{_transcript(history)}
</discussion>

The discussion is over. Write the final result for the user: the actual deliverable they asked for, built from the strongest ideas in the discussion. Resolve disagreements by making a clear call. Address the user directly; do not narrate the discussion or say who suggested what unless it genuinely helps the user."""
        final_text = ""
        async for event in self._speak(t.leader, self._leader_system(), prompt, final=True):
            if event["type"] == "turn_end":
                final_text = event["text"]
            yield event
        yield {"type": "final", "text": final_text}
        yield {"type": "done"}

    async def _agent_turn(
        self, task: str, history: list[Message], agent: Agent, note: str
    ) -> AsyncIterator[dict[str, Any]]:
        prompt = self._turn_prompt(task, history, agent, note)
        async for event in self._speak(agent, self._agent_system(agent), prompt):
            if event["type"] == "turn_end" and event["text"].strip():
                history.append(Message(agent.name, event["text"]))
            yield event

    async def _round_robin(self, task: str, history: list[Message]) -> AsyncIterator[dict[str, Any]]:
        t = self.team
        for rnd in range(1, t.rounds + 1):
            yield {"type": "phase", "label": f"סבב {rnd} מתוך {t.rounds}"}
            for agent in t.agents:
                note = f"(Round {rnd} of {t.rounds}.)"
                async for event in self._agent_turn(task, history, agent, note):
                    yield event

            if t.early_stop and rnd < t.rounds:
                decision = await self._decide(
                    f"""<user_request>
{task}
</user_request>

<discussion>
{_transcript(history)}
</discussion>

Round {rnd} of {t.rounds} just ended. Has the team converged enough that another round would add little? Answer "done": true to stop the discussion now, or false to run another round. Give a one-sentence reason.""",
                    {
                        "type": "object",
                        "properties": {"done": {"type": "boolean"}, "reason": {"type": "string"}},
                        "required": ["done", "reason"],
                        "additionalProperties": False,
                    },
                )
                if decision:
                    yield {
                        "type": "decision",
                        "agent": t.leader.name,
                        "emoji": t.leader.emoji,
                        "text": decision.get("reason", ""),
                    }
                    if decision.get("done"):
                        return

    async def _moderated(self, task: str, history: list[Message]) -> AsyncIterator[dict[str, Any]]:
        t = self.team
        by_name = {a.name: a for a in t.agents}
        schema = {
            "type": "object",
            "properties": {
                "next_speaker": {"type": "string", "enum": [*by_name, FINISH]},
                "instruction": {"type": "string"},
            },
            "required": ["next_speaker", "instruction"],
            "additionalProperties": False,
        }
        yield {"type": "phase", "label": "דיון בהנחיית המנהל"}
        for turn in range(1, t.max_turns + 1):
            decision = await self._decide(
                f"""<user_request>
{task}
</user_request>

<discussion>
{_transcript(history)}
</discussion>

You moderate this discussion (turn {turn} of at most {t.max_turns}). Pick who should speak next and give them a short, specific instruction for what to contribute. Choose "{FINISH}" once the team has enough to produce an excellent final result.""",
                schema,
            )
            if decision is None:
                # Leader failed to decide: fall back to simple rotation.
                agent = t.agents[(turn - 1) % len(t.agents)]
                instruction = ""
            else:
                speaker = decision["next_speaker"]
                instruction = decision.get("instruction", "")
                if speaker == FINISH or speaker not in by_name:
                    yield {"type": "decision", "agent": t.leader.name, "emoji": t.leader.emoji,
                           "text": instruction or "סיימנו את הדיון."}
                    return
                agent = by_name[speaker]
                yield {"type": "decision", "agent": t.leader.name, "emoji": t.leader.emoji,
                       "text": f"{agent.name}: {instruction}"}
            note = f"The leader, {t.leader.name}, asks you: {instruction}" if instruction else ""
            async for event in self._agent_turn(task, history, agent, note):
                yield event
