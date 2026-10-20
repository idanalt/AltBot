"""AltBot - a multi-agent "team room" powered by Claude.

Define a team of agents, each with its own role, give them a task, and let
them discuss it with each other until a leader writes the final result.
"""

from .models import Agent, Team, load_team, load_team_file
from .orchestrator import Orchestrator

__all__ = ["Agent", "Team", "load_team", "load_team_file", "Orchestrator"]
