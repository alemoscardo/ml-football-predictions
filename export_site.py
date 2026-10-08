"""Export what the static site shows to JSON in ``site/public/data/``.

The site computes nothing: this script runs after ``forecast.py`` (locally or in the
deploy workflow) and the site only reads its output.

- ``matches.json``: every match of the season under way. Forecasts committed to the
  ledger before kick-off are ``live``; matches played before the ledger existed are
  ``simulated``: the frozen model run on them after the fact, and labelled as such.
- ``record.json``: the backtest season, the live record and the simulated one so far.
- ``teams.json``: Elo ratings, rating paths and form for the season's clubs.
- ``meta.json``: labels, model details and when the data was built.

Usage:
    python export_site.py
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import BOOKMAKER_PROBS, MODEL_FEATURES, build_features
from forecast import (
    KICKOFF_TZ,
    LEDGER_PATH,
    OUTCOMES,
    fit_frozen_model,
    kickoff_utc,
    load_ledger,
    load_live_results,
    model_version,
    season_label,
    season_stem,
)
from train_models import (
    BOOKIE_COLUMNS,
    DATA_DIR,
    LABELS,
    METRICS_PATH,
    fit_spec,
    load_dataset,
    load_raw_matches,
    predict_sorted,
    score,
    split_seasons,
)

OUT_DIR = Path("site/public/data")
REPO_URL = "https://github.com/alemoscardo/ml-football-predictions"
CONFIDENCE_BANDS = [0.0, 0.4, 0.5, 0.6, 0.7, 1.0]
TEAM_CODES = {
    "Arsenal": "ARS",
    "Aston Villa": "AVL",
    "Bournemouth": "BOU",
    "Brentford": "BRE",
    "Brighton": "BHA",
    "Burnley": "BUR",
    "Chelsea": "CHE",
    "Coventry": "COV",
    "Crystal Palace": "CRY",
    "Everton": "EVE",
    "Fulham": "FUL",
    "Hull": "HUL",
    "Ipswich": "IPS",
    "Leeds": "LEE",
    "Leicester": "LEI",
    "Liverpool": "LIV",
    "Luton": "LUT",
    "Man City": "MCI",
    "Man United": "MUN",
    "Middlesbrough": "MID",
    "Newcastle": "NEW",
    "Norwich": "NOR",
    "Nott'm Forest": "NFO",
    "Sheffield United": "SHU",
    "Southampton": "SOU",
    "Sunderland": "SUN",
    "Tottenham": "TOT",
    "Watford": "WAT",
    "West Brom": "WBA",
    "West Ham": "WHU",
    "Wolves": "WOL",
}


def team_code(name: str) -> str:
    return TEAM_CODES.get(name, re.sub(r"[^A-Za-z]", "", name)[:3].upper())


def clean(value: object) -> object:
    """JSON-safe: NaN becomes null, numpy scalars become Python numbers."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def probabilities(values: list[float]) -> dict[str, float] | None:
    """{H, D, A} rounded to four places, or None when any value is missing."""
    if any(pd.isna(v) for v in values):
        return None
    return {o: round(float(v), 4) for o, v in zip(OUTCOMES, values, strict=True)}


def pick(probs: dict[str, float] | None) -> str | None:
    return max(OUTCOMES, key=lambda o: probs[o]) if probs else None


def assign_matchweeks(home: pd.Series, away: pd.Series) -> list[int]:
    """Round of each match, given in kick-off order.

    Football-Data has no round numbers. A match goes in the round after the most games
    either side has had before it, so a rearranged match lands where it is played.
    """
    games: dict[str, int] = {}
    rounds = []
    for h, a in zip(home, away, strict=True):
        rounds.append(max(games.get(h, 0), games.get(a, 0)) + 1)
        games[h] = games.get(h, 0) + 1
        games[a] = games.get(a, 0) + 1
    return rounds


def ledger_commits(path: Path = LEDGER_PATH) -> list[str | None]:
    """The commit that added each ledger data row; None where git cannot tell."""
    try:
        blame = subprocess.run(
            ["git", "blame", "--line-porcelain", "--", str(path)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    shas = [m.group(1) for m in re.finditer(r"^([0-9a-f]{40}) \d+ \d+", blame, re.M)]
    return [None if set(sha) == {"0"} else sha for sha in shas[1:]]  # skip the header


def live_season(results: pd.DataFrame, ledger: pd.DataFrame) -> str:
    seasons = set(results.get("SeasonFile", [])) | set(ledger.get("Season", []))
    return max(seasons) if seasons else season_stem(datetime.now(UTC).date())


def season_matches(model, results: pd.DataFrame, ledger: pd.DataFrame, season: str) -> list[dict]:
    """Every match of ``season``: played ones with their result, logged fixtures without."""
    played = results[(results["SeasonFile"] == season) & results["FTR"].notna()].copy()
    if len(played):
        played["KickoffUTC"] = kickoff_utc(played)
    ledger = ledger.assign(Commit=(ledger_commits() + [None] * len(ledger))[: len(ledger)])
    ledger = ledger[ledger["Season"] == season]
    logged = ledger.set_index(["HomeTeam", "AwayTeam"])

    done = list(zip(played["HomeTeam"], played["AwayTeam"], strict=True))
    pending = ledger[~pd.MultiIndex.from_frame(ledger[["HomeTeam", "AwayTeam"]]).isin(done)]
    upcoming = pd.DataFrame(
        {
            "Date": pending["KickoffUTC"].dt.tz_convert(KICKOFF_TZ).dt.strftime("%d/%m/%Y"),
            "HomeTeam": pending["HomeTeam"],
            "AwayTeam": pending["AwayTeam"],
            "SeasonFile": season,
            "KickoffUTC": pending["KickoffUTC"],
        }
    )
    raw = pd.concat([load_raw_matches(), played, upcoming], ignore_index=True)
    features = build_features(raw)
    features = features[features["SeasonFile"] == season].copy()
    if features.empty:
        return []
    simulated = pd.DataFrame(predict_sorted(model, features[MODEL_FEATURES]), columns=LABELS)
    features[[f"S_{o}" for o in LABELS]] = simulated.to_numpy()
    features["KickoffUTC"] = pd.to_datetime(features["KickoffUTC"], utc=True)
    features = features.sort_values(["KickoffUTC", "HomeTeam"], kind="stable")
    features["Matchweek"] = assign_matchweeks(features["HomeTeam"], features["AwayTeam"])

    matches = []
    for row in features.itertuples(index=False):
        key = (row.HomeTeam, row.AwayTeam)
        is_live = key in logged.index
        if is_live:
            entry = logged.loc[key]
            model_p = probabilities([entry[f"P_{o}"] for o in OUTCOMES])
            bookie_p = probabilities([entry[f"B_{o}"] for o in OUTCOMES])
            logged_at = entry["LoggedAtUTC"].strftime("%Y-%m-%dT%H:%MZ")
            commit = entry["Commit"]
        else:
            model_p = probabilities([getattr(row, f"S_{o}") for o in OUTCOMES])
            bookie_p = probabilities([getattr(row, BOOKMAKER_PROBS[o]) for o in OUTCOMES])
            logged_at, commit = None, None
        has_result = isinstance(row.Result, str)
        matches.append(
            {
                "id": f"{season}-{row.HomeTeam}-{row.AwayTeam}".replace(" ", "_"),
                "matchweek": int(row.Matchweek),
                "kickoff": row.KickoffUTC.strftime("%Y-%m-%dT%H:%MZ"),
                "home": row.HomeTeam,
                "away": row.AwayTeam,
                "homeCode": team_code(row.HomeTeam),
                "awayCode": team_code(row.AwayTeam),
                "score": [int(row.HomeGoals), int(row.AwayGoals)] if has_result else None,
                "result": row.Result if has_result else None,
                "source": "live" if is_live else "simulated",
                "model": model_p,
                "bookmaker": bookie_p,
                "pick": pick(model_p),
                "bookmakerPick": pick(bookie_p),
                "loggedAt": logged_at,
                "commit": commit,
                "features": {
                    "homeElo": round(row.HomeElo),
                    "awayElo": round(row.AwayElo),
                    "homeForm": None if pd.isna(row.HomePointsL5) else round(row.HomePointsL5, 1),
                    "awayForm": None if pd.isna(row.AwayPointsL5) else round(row.AwayPointsL5, 1),
                },
            }
        )
    return matches


def gap_series(results: list[str], model: np.ndarray, bookie: np.ndarray) -> list[float]:
    """Running mean of model log-loss minus bookmaker log-loss, one value per match."""
    idx = np.array([LABELS.index(r) for r in results])
    rows = np.arange(len(idx))
    gap = -np.log(model[rows, idx]) + np.log(bookie[rows, idx])
    return [round(float(g), 4) for g in np.cumsum(gap) / np.arange(1, len(gap) + 1)]


def backtest_record(spec: dict) -> dict:
    """The test season, scored the way ``train_models.py`` does, plus what the charts need."""
    matches = load_dataset()
    split = split_seasons(matches)
    fit = matches[matches["SeasonFile"].isin(split["train"] + split["validation"])]
    test = matches[matches["SeasonFile"].isin(split["test"])].sort_values("Date", kind="stable")
    model = fit_spec(spec, fit)
    proba = predict_sorted(model, test[MODEL_FEATURES])
    bookie = test[BOOKIE_COLUMNS].to_numpy()
    y = test["Result"]
    picks = np.array(LABELS)[proba.argmax(axis=1)]

    confidence = proba.max(axis=1)
    hit = picks == y.to_numpy()
    bands = []
    for low, high in zip(CONFIDENCE_BANDS[:-1], CONFIDENCE_BANDS[1:], strict=True):
        inside = (confidence > low) & (confidence <= high)
        if inside.any():
            bands.append(
                {
                    "low": low,
                    "high": high,
                    "stated": round(float(confidence[inside].mean()), 4),
                    "actual": round(float(hit[inside].mean()), 4),
                    "matches": int(inside.sum()),
                }
            )
    return {
        "season": season_label(split["test"][0]),
        "matches": len(test),
        "model": score(y, proba),
        "bookmaker": score(y, bookie),
        "baselines": {
            "random": 1 / 3,
            "home": float((y == "H").mean()),
            "bookmaker": score(y, bookie)["accuracy"],
            "model": score(y, proba)["accuracy"],
        },
        "draws": {"happened": int((y == "D").sum()), "called": int((picks == "D").sum())},
        "gap": gap_series(list(y), proba, bookie),
        "dates": test["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "calibration": bands,
    }


def season_record(matches: list[dict], source: str) -> dict:
    """Hits, log-loss and the running gap for the settled matches from one source."""
    settled = [m for m in matches if m["source"] == source and m["result"] and m["bookmaker"]]
    if not settled:
        return {"matches": 0}
    model = np.array([[m["model"][o] for o in LABELS] for m in settled])
    bookie = np.array([[m["bookmaker"][o] for o in LABELS] for m in settled])
    model /= model.sum(axis=1, keepdims=True)  # stored to four places
    bookie /= bookie.sum(axis=1, keepdims=True)
    results = [m["result"] for m in settled]
    return {
        "matches": len(settled),
        "model": score(pd.Series(results), model),
        "bookmaker": score(pd.Series(results), bookie),
        "modelHits": sum(m["pick"] == m["result"] for m in settled),
        "bookmakerHits": sum(m["bookmakerPick"] == m["result"] for m in settled),
        "gap": gap_series(results, model, bookie),
    }


def team_ratings(results: pd.DataFrame, season: str) -> list[dict]:
    """Each club's Elo before its first match of the season and after every one since."""
    raw = pd.concat([load_raw_matches(), results], ignore_index=True)
    played = build_features(raw).dropna(subset=["Result"])
    if season not in set(played["SeasonFile"]):  # no match played yet: last season's table
        season = played["SeasonFile"].max()
    rated = played[played["SeasonFile"] == season]
    sides = []
    for side, other in (("Home", "Away"), ("Away", "Home")):
        frame = rated[["Date", f"{side}Team", f"{side}Elo", f"{side}EloAfter"]].copy()
        frame.columns = ["Date", "Team", "Before", "After"]
        frame["For"] = rated[f"{side}Goals"].to_numpy()
        frame["Against"] = rated[f"{other}Goals"].to_numpy()
        sides.append(frame)
    long = pd.concat(sides).sort_values("Date", kind="stable")
    teams = []
    for team, games in long.groupby("Team"):
        path = [round(games["Before"].iloc[0])] + [round(e) for e in games["After"]]
        form = [
            "W" if f > a else "D" if f == a else "L"
            for f, a in zip(games["For"], games["Against"], strict=True)
        ]
        teams.append(
            {
                "name": team,
                "code": team_code(team),
                "elo": path[-1],
                "start": path[0],
                "change": path[-1] - path[0],
                "path": path,
                "form": form[-5:],
            }
        )
    teams.sort(key=lambda t: -t["elo"])
    for rank, team in enumerate(teams, 1):
        team["rank"] = rank
    return teams


def build(now: datetime | None = None) -> dict[str, object]:
    """Everything the site needs, keyed by output file name."""
    now = now or datetime.now(UTC)
    report = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    spec = report["model"]
    results = load_live_results()
    ledger = load_ledger()
    if results.empty:
        results = pd.DataFrame(columns=["SeasonFile", "HomeTeam", "AwayTeam", "FTR"])
    season = live_season(results, ledger)
    model = fit_frozen_model(spec)
    matches = season_matches(model, results, ledger, season)
    completed = sorted(p.stem for p in DATA_DIR.glob("E0_*.csv"))

    live = season_record(matches, "live")
    return {
        "meta": {
            "generatedAt": now.strftime("%Y-%m-%dT%H:%MZ"),
            "season": season_label(season),
            "repo": REPO_URL,
            "model": {
                "algorithm": spec["algorithm"],
                "features": len(MODEL_FEATURES),
                "version": model_version(spec),
                "fittedOn": f"{season_label(completed[1])} – {season_label(completed[-1])}",
            },
            "ledger": {
                "logged": sum(m["source"] == "live" for m in matches),
                "settled": live["matches"],
                "pending": sum(m["source"] == "live" and not m["result"] for m in matches),
            },
        },
        "matches": matches,
        "record": {
            "backtest": backtest_record(spec),
            "live": live,
            "simulated": season_record(matches, "simulated"),
        },
        "teams": team_ratings(results, season),
    }


def write(data: dict[str, object], out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in data.items():
        text = json.dumps(clean(payload), allow_nan=False, separators=(",", ":"))
        (out_dir / f"{name}.json").write_text(text, encoding="utf-8")


def main() -> None:
    data = build()
    write(data)
    meta = data["meta"]
    print(
        f"{meta['season']}: {len(data['matches'])} matches "
        f"({meta['ledger']['logged']} live, {meta['ledger']['pending']} pending), "
        f"{len(data['teams'])} clubs, written to {OUT_DIR}"
    )


if __name__ == "__main__":
    main()
