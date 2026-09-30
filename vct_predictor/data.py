"""Loading of the JSON data files in ``data/``."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Player:
    name: str
    impact: float


@dataclass
class Team:
    name: str
    tag: str
    region: str
    seed: str
    prior_elo: float
    players: list[Player]
    map_delta: dict[str, float]
    resume_2026: list[str] = field(default_factory=list)
    bench: list[str] = field(default_factory=list)


@dataclass
class Dataset:
    event: dict
    teams: dict[str, Team]
    external: dict[str, float]
    history: dict
    results: dict
    data_dir: Path = DATA_DIR

    @property
    def map_pool(self) -> list[str]:
        return list(self.event["map_pool"])

    def find_team(self, query: str) -> Team:
        """Resolve a team by exact name, tag, or unambiguous prefix (case-insensitive)."""
        q = query.strip().lower()
        for t in self.teams.values():
            if q in (t.name.lower(), t.tag.lower()):
                return t
        hits = [t for t in self.teams.values() if t.name.lower().startswith(q) or q in t.name.lower()]
        if len(hits) == 1:
            return hits[0]
        names = ", ".join(f"{t.name} ({t.tag})" for t in self.teams.values())
        if not hits:
            raise KeyError(f"Unknown team '{query}'. Known teams: {names}")
        raise KeyError(f"Ambiguous team '{query}': {', '.join(t.name for t in hits)}")


def _read(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_dataset(data_dir: Path | str = DATA_DIR) -> Dataset:
    data_dir = Path(data_dir)
    raw_teams = _read(data_dir / "teams.json")
    teams = {}
    for t in raw_teams["teams"]:
        teams[t["name"]] = Team(
            name=t["name"],
            tag=t["tag"],
            region=t["region"],
            seed=t["seed"],
            prior_elo=float(t["prior_elo"]),
            players=[Player(p["name"], float(p["impact"])) for p in t["players"]],
            map_delta={k: float(v) for k, v in t["map_delta"].items()},
            resume_2026=t.get("resume_2026", []),
            bench=t.get("bench", []),
        )
    return Dataset(
        event=_read(data_dir / "event.json"),
        teams=teams,
        external={k: float(v) for k, v in raw_teams.get("external_teams", {}).items()},
        history=_read(data_dir / "history.json"),
        results=_read(data_dir / "champions_results.json"),
        data_dir=data_dir,
    )
