"""Command line entry point.

  python -m altbot web [--host 127.0.0.1] [--port 8000]
  python -m altbot run teams/product.yaml "המשימה שלך"
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from .models import load_team_file
from .orchestrator import Orchestrator

BOLD, DIM, CYAN, GREEN, RED, RESET = "\033[1m", "\033[2m", "\033[36m", "\033[32m", "\033[31m", "\033[0m"


async def _run(team_path: str, task: str, rounds: int | None, mode: str | None) -> None:
    team = load_team_file(team_path)
    if rounds:
        team.rounds = rounds
    if mode:
        team.mode = mode
    async for e in Orchestrator(team).run(task):
        kind = e["type"]
        if kind == "start":
            print(f"{BOLD}צוות: {e['team']}{RESET}  ({', '.join(a['name'] for a in e['agents'])})")
        elif kind == "phase":
            print(f"\n{CYAN}━━━ {e['label']} ━━━{RESET}")
        elif kind == "turn_start":
            color = GREEN if e["final"] else BOLD
            print(f"\n{color}{e['emoji']} {e['agent']}:{RESET}")
        elif kind == "delta":
            print(e["text"], end="", flush=True)
        elif kind == "turn_end":
            print()
        elif kind == "decision":
            print(f"\n{DIM}{e['emoji']} {e['agent']} → {e['text']}{RESET}")
        elif kind == "error":
            print(f"\n{RED}שגיאה ({e['agent']}): {e['message']}{RESET}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(prog="altbot", description="Multi-agent team room powered by Claude")
    sub = parser.add_subparsers(dest="cmd", required=True)

    web = sub.add_parser("web", help="start the web UI")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8000)

    run = sub.add_parser("run", help="run a team from the terminal")
    run.add_argument("team", help="path to a team YAML file")
    run.add_argument("task", help="what you want the team to produce")
    run.add_argument("--rounds", type=int)
    run.add_argument("--mode", choices=["round_robin", "moderated"])

    args = parser.parse_args()
    if args.cmd == "web":
        import uvicorn

        print(f"AltBot is running at http://{args.host}:{args.port}")
        uvicorn.run("altbot.web:app", host=args.host, port=args.port)
    else:
        asyncio.run(_run(args.team, args.task, args.rounds, args.mode))


if __name__ == "__main__":
    main()
