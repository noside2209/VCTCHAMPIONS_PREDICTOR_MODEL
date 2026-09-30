"""Sync player agent pools and stats from VLR.gg player profiles.

Run on a machine that can reach vlr.gg:

    python -m vct_predictor sync-vlr                  # every player, last 90 days
    python -m vct_predictor sync-vlr --team LOUD      # one team
    python -m vct_predictor sync-vlr --timespan all   # career numbers

For each player it:
  1. finds the VLR player id (uses "vlr_id" in data/players.json if set, otherwise
     searches VLR and picks the exact-name match whose profile mentions the team),
  2. reads the agent table on the profile (agent, uses, rounds, rating, ACS, K:D, ADR,
     KAST, KPR, APR, FKPR, FDPR ...),
  3. writes back into data/players.json: agent pool ordered by rounds played, role
     (from the agents actually played), vlr_id, and round-weighted vlr_stats,
  4. sets the player's impact in data/teams.json to his VLR rating (unless --keep-impact).

Players with a user-verified agent pool keep it unless --overwrite-verified is given.
"""
from __future__ import annotations

import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

from .data import DATA_DIR

BASE = "https://www.vlr.gg"
UA = "Mozilla/5.0 (vct-predictor; +https://github.com/noside2209/vctchampions_predictor_model)"

# VLR header label -> our key
COLUMNS = {
    "use": "use", "rnd": "rounds", "rating": "rating", "r2.0": "rating", "acs": "acs", "k:d": "kd",
    "adr": "adr", "kast": "kast", "kpr": "kpr", "apr": "apr", "fkpr": "fkpr", "fdpr": "fdpr",
    "k": "k", "d": "d", "a": "a", "fk": "fk", "fd": "fd",
}
DEFAULT_ORDER = ["agent", "use", "rounds", "rating", "acs", "kd", "adr", "kast", "kpr", "apr",
                 "fkpr", "fdpr", "k", "d", "a", "fk", "fd"]


def _column_key(label: str) -> str:
    label = label.lower().replace(" ", "")
    if label.startswith("rating") or label.startswith("r2") or label.startswith("r3"):
        return "rating"
    return COLUMNS.get(label, label)


def fetch(url: str, retries: int = 3) -> str:
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception as exc:  # network hiccup: back off and retry
            last = exc
            time.sleep(2 ** i)
    raise RuntimeError(f"could not fetch {url}: {last}")


def _text(cell: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", cell)).strip()


def _num(s: str) -> float | None:
    s = s.replace("%", "").replace(",", "").strip()
    m = re.search(r"-?\d+(\.\d+)?", s)
    return float(m.group()) if m else None


def search_player_ids(name: str) -> list[tuple[int, str]]:
    page = fetch(f"{BASE}/search/?q={urllib.parse.quote(name)}&type=players")
    out = []
    for m in re.finditer(r'href="/player/(\d+)/([^"/?#]+)"(.*?)</a>', page, re.S):
        pid, slug, inner = int(m.group(1)), m.group(2), m.group(3)
        title = re.search(r'search-item-title[^>]*>(.*?)</div>', inner, re.S)
        shown = _text(title.group(1)) if title else slug
        out.append((pid, shown))
    return out


def parse_agent_table(page: str) -> list[dict]:
    """Parse the agent stats table from a VLR player profile."""
    rows = []
    for table in re.findall(r"<table[^>]*>(.*?)</table>", page, re.S):
        if "/img/vlr/game/agents/" not in table:
            continue
        headers = [_text(h).lower() for h in re.findall(r"<th[^>]*>(.*?)</th>", table, re.S)]
        order = ["agent"] + [_column_key(h) for h in headers[1:]] if headers else DEFAULT_ORDER
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", table, re.S):
            img = re.search(r'/img/vlr/game/agents/([a-z0-9_\-]+)\.png', tr)
            if not img:
                continue
            title = re.search(r'<img[^>]*title="([^"]+)"', tr)
            agent = html.unescape(title.group(1)) if title else img.group(1)
            agent = {"kayo": "KAY/O", "kay/o": "KAY/O"}.get(agent.lower(), agent[:1].upper() + agent[1:])
            cells = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)
            row: dict = {"agent": agent}
            for key, cell in zip(order[1:], cells[1:]):
                txt = _text(cell)
                if key == "use":
                    uses = re.search(r"\((\d+)\)", txt)
                    row["use"] = int(uses.group(1)) if uses else _num(txt)
                else:
                    row[key] = _num(txt)
            rows.append(row)
        if rows:
            break
    return rows


def aggregate(rows: list[dict]) -> dict:
    total = sum(r.get("rounds") or 0 for r in rows)
    if not total:
        return {}
    out: dict = {"rounds": int(total)}
    for key in ("rating", "acs", "adr", "kast", "kpr", "apr", "fkpr", "fdpr"):
        vals = [(r.get(key), r.get("rounds") or 0) for r in rows if r.get(key) is not None]
        if vals:
            v = sum(x * w for x, w in vals) / max(sum(w for _, w in vals), 1)
            out[key] = round(v / 100 if key == "kast" and v > 1.5 else v, 3)
    kills = sum(r.get("k") or 0 for r in rows)
    deaths = sum(r.get("d") or 0 for r in rows)
    if deaths:
        out["dpr"] = round(deaths / total, 3)
    elif out.get("kpr") and rows:
        kd = sum((r.get("kd") or 1) * (r.get("rounds") or 0) for r in rows) / total
        out["dpr"] = round(out["kpr"] / max(kd, 0.1), 3)
    if kills:
        out["kpr"] = round(kills / total, 3)
    return out


def role_from_agents(rows: list[dict], agent_roles: dict[str, str]) -> str | None:
    by_role: dict[str, float] = {}
    total = 0.0
    for r in rows:
        role = agent_roles.get(r["agent"])
        if role:
            by_role[role] = by_role.get(role, 0) + (r.get("rounds") or 0)
            total += r.get("rounds") or 0
    if not total:
        return None
    role, share = max(by_role.items(), key=lambda kv: kv[1])
    return role if share / total >= 0.6 else "Flex"


def resolve_id(player: dict, team: str, log) -> int | None:
    if player.get("vlr_id"):
        return int(player["vlr_id"])
    cands = [pid for pid, shown in search_player_ids(player["name"]) if shown.lower() == player["name"].lower()]
    if not cands:
        log(f"  ! {player['name']}: no exact VLR search match; set \"vlr_id\" in data/players.json")
        return None
    if len(cands) == 1:
        return cands[0]
    team_key = team.lower().split()[0]
    for pid in cands[:5]:
        page = fetch(f"{BASE}/player/{pid}")
        if team.lower() in page.lower() or team_key in page.lower():
            return pid
        time.sleep(1)
    return cands[0]


def sync(team_filter: str | None = None, timespan: str = "90d", keep_impact: bool = False,
         overwrite_verified: bool = False, data_dir: Path = DATA_DIR, delay: float = 1.5, log=print) -> dict:
    players_path, teams_path = data_dir / "players.json", data_dir / "teams.json"
    pdata = json.loads(players_path.read_text(encoding="utf-8"))
    tdata = json.loads(teams_path.read_text(encoding="utf-8"))
    agent_roles = pdata["agent_roles"]
    summary = {"updated": [], "failed": []}
    for team, roster in pdata["teams"].items():
        if team_filter and team_filter.lower() not in (team.lower(),):
            continue
        log(f"{team}")
        for pl in roster:
            try:
                pid = resolve_id(pl, team, log)
                if pid is None:
                    summary["failed"].append(pl["name"])
                    continue
                rows = parse_agent_table(fetch(f"{BASE}/player/{pid}/?timespan={timespan}"))
                if not rows and timespan != "all":
                    rows = parse_agent_table(fetch(f"{BASE}/player/{pid}/?timespan=all"))
                if not rows:
                    log(f"  ! {pl['name']}: no agent table found on /player/{pid}")
                    summary["failed"].append(pl["name"])
                    continue
                rows.sort(key=lambda r: -(r.get("rounds") or 0))
                # Ignore one-off picks (<5% of rounds) so the pool reflects what he really plays.
                total = sum(r.get("rounds") or 0 for r in rows) or 1
                agents = [r["agent"] for r in rows if (r.get("rounds") or 0) / total >= 0.05] or [rows[0]["agent"]]
                old = list(pl["agents"])
                pl["vlr_id"] = pid
                pl["vlr_stats"] = aggregate(rows) | {"timespan": timespan}
                if pl.get("source") != "user-verified" or overwrite_verified:
                    pl["agents"] = agents
                    role = role_from_agents(rows, agent_roles)
                    if role:
                        pl["role"] = role
                    pl["source"] = "vlr"
                for a in agents:
                    agent_roles.setdefault(a, "Flex")
                if not keep_impact and pl["vlr_stats"].get("rating"):
                    for t in tdata["teams"]:
                        if t["name"] == team:
                            for p in t["players"]:
                                if p["name"] == pl["name"]:
                                    p["impact"] = round(pl["vlr_stats"]["rating"], 2)
                change = "" if old == pl["agents"] else f"   (was {', '.join(old)})"
                log(f"  {pl['name']:<12} {pl['role']:<10} {', '.join(pl['agents'])}{change}"
                    f"  rating {pl['vlr_stats'].get('rating', '?')} over {pl['vlr_stats'].get('rounds', 0)} rds")
                summary["updated"].append(pl["name"])
            except Exception as exc:
                log(f"  ! {pl['name']}: {exc}")
                summary["failed"].append(pl["name"])
            time.sleep(delay)
    players_path.write_text(json.dumps(pdata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    teams_path.write_text(json.dumps(tdata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"\nUpdated {len(summary['updated'])} players; failed {len(summary['failed'])}: {', '.join(summary['failed']) or '-'}")
    return summary
