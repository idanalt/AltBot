"""Web UI: build a team in the browser and watch the agents talk live (SSE)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .models import load_team, load_team_file
from .orchestrator import Orchestrator

ROOT = Path(__file__).resolve().parent
TEAMS_DIR = ROOT.parent / "teams"

app = FastAPI(title="AltBot")


class RunRequest(BaseModel):
    task: str
    team: dict[str, Any]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/teams")
def list_teams() -> list[dict[str, Any]]:
    import yaml

    teams = []
    for path in sorted(TEAMS_DIR.glob("*.y*ml")):
        load_team_file(path)  # validate
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        teams.append({"id": path.stem, **data})
    return teams


@app.post("/api/run")
async def run(req: RunRequest) -> StreamingResponse:
    try:
        team = load_team(req.team)
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if not req.task.strip():
        raise HTTPException(status_code=400, detail="המשימה ריקה")

    async def events():
        try:
            async for event in Orchestrator(team).run(req.task):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:  # surface unexpected failures to the browser
            yield f"data: {json.dumps({'type': 'error', 'agent': '', 'message': str(e)}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
