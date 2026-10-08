"""Live forward test: forecast every upcoming Premier League match before kick-off.

Run on a schedule by ``.github/workflows/forecast.yml``. Each run:

1. refreshes this season's results and the upcoming fixtures from Football-Data.co.uk,
2. builds pre-match features for the fixtures from every result played so far,
3. appends a forecast for each fixture that has not kicked off and is not logged yet.

``predictions/live.csv`` is an append-only ledger: a row is written once and never
edited, and each run that adds rows is a commit by GitHub Actions, so the history shows
every forecast was made before its match. The model is frozen for the season: the spec
chosen on the validation season, refitted on every completed season.

Usage:
    python forecast.py
"""

from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from feature_engineering import BOOKMAKER_PROBS, MODEL_FEATURES, build_features
from fetch_data import BASE_URL, download_csv, season_code
from train_models import (
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

FIXTURES_URL = "https://football-data.co.uk/fixtures.csv"
LIVE_DIR = DATA_DIR / "live"
LEDGER_PATH = Path("predictions") / "live.csv"
LEAGUE = "E0"
KICKOFF_TZ = "Europe/London"
OUTCOMES = ["H", "D", "A"]
KEY = ["Season", "HomeTeam", "AwayTeam"]  # each pairing is played once per season
LEDGER_COLUMNS = [
    *KEY,
    "KickoffUTC",
    "LoggedAtUTC",
    "ModelVersion",
    *[f"P_{o}" for o in OUTCOMES],
    *[f"B_{o}" for o in OUTCOMES],
]
TIME_FORMAT = "%Y-%m-%dT%H:%MZ"
SUM_TOLERANCE = 1e-3  # probabilities are logged to four decimals


def season_stem(today: date) -> str:
    """The season under way on ``today``: seasons start in August, so July onwards."""
    start = today.year if today.month >= 7 else today.year - 1
    return f"{LEAGUE}_{season_code(start)}"


def season_label(stem: str) -> str:
    """``E0_2324`` → ``2023/24``; anything else comes back unchanged."""
    code = stem.split("_")[-1]
    return f"20{code[:2]}/{code[2:]}" if len(code) == 4 and code.isdigit() else stem


def load_live_results() -> pd.DataFrame:
    """Results of seasons under way, from ``data/live/``.

    Skips any season already moved into ``data/`` for training, so a finished
    season is never counted twice.
    """
    completed = {path.stem for path in DATA_DIR.glob(f"{LEAGUE}_*.csv")}
    frames = [
        pd.read_csv(path, encoding_errors="replace").copy().assign(SeasonFile=path.stem)
        for path in sorted(LIVE_DIR.glob(f"{LEAGUE}_*.csv"))
        if path.stem not in completed
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def kickoff_utc(fixtures: pd.DataFrame) -> pd.Series:
    """UK local date and time to UTC. A missing time counts as midnight, the earliest case."""
    local = pd.to_datetime(
        fixtures["Date"] + " " + fixtures["Time"].fillna("00:00"), dayfirst=True, format="mixed"
    )
    return local.dt.tz_localize(
        KICKOFF_TZ, ambiguous="NaT", nonexistent="shift_forward"
    ).dt.tz_convert("UTC")


def upcoming_fixtures(fixtures: pd.DataFrame, results: pd.DataFrame, season: str) -> pd.DataFrame:
    """League fixtures from the weekly file that are not in the results yet."""
    league = fixtures[fixtures["Div"] == LEAGUE].copy()
    played: set[tuple[str, str]] = set()
    if len(results):
        current = results[results["SeasonFile"] == season]
        played = set(zip(current["HomeTeam"], current["AwayTeam"], strict=True))
    pairs = pd.MultiIndex.from_frame(league[["HomeTeam", "AwayTeam"]])
    league = league[~pairs.isin(list(played))].copy()
    league["SeasonFile"] = season
    league["KickoffUTC"] = kickoff_utc(league)
    return league.dropna(subset=["KickoffUTC"]).reset_index(drop=True)


def fit_frozen_model(spec: dict) -> Pipeline:
    """The chosen spec, refitted on every completed season except the burn-in."""
    history = load_dataset()
    burn_in = split_seasons(history)["burn_in"]
    fit = history[~history["SeasonFile"].isin(burn_in)]
    return fit_spec(spec, fit)


def model_version(spec: dict) -> str:
    """Short hash of the model spec and the completed seasons it is fitted on."""
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode())
    for path in sorted(DATA_DIR.glob(f"{LEAGUE}_*.csv")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:10]


def forecast(
    history: pd.DataFrame, results: pd.DataFrame, fixtures: pd.DataFrame, model: Pipeline
) -> pd.DataFrame:
    """Model and bookmaker probabilities for each fixture.

    Features come from every completed and live result before the fixture. Fixtures
    have no goals, so they move no Elo rating and add no form to later fixtures.
    """
    raw = pd.concat([history, results, fixtures.assign(IsFixture=True)], ignore_index=True)
    features = build_features(raw)
    rows = features[features["IsFixture"].eq(True)].reset_index(drop=True)
    proba = pd.DataFrame(predict_sorted(model, rows[MODEL_FEATURES]), columns=LABELS)
    out = rows[["SeasonFile", "HomeTeam", "AwayTeam", "KickoffUTC"]].rename(
        columns={"SeasonFile": "Season"}
    )
    for outcome in OUTCOMES:
        out[f"P_{outcome}"] = proba[outcome]
        out[f"B_{outcome}"] = rows[BOOKMAKER_PROBS[outcome]]
    return out


def new_entries(
    forecasts: pd.DataFrame, ledger: pd.DataFrame, now: datetime, version: str
) -> pd.DataFrame:
    """Forecasts not logged yet whose match has not kicked off."""
    logged = list(ledger[KEY].itertuples(index=False, name=None))
    keep = (forecasts["KickoffUTC"] > now) & ~pd.MultiIndex.from_frame(forecasts[KEY]).isin(logged)
    entries = forecasts[keep].sort_values(["KickoffUTC", "HomeTeam"], kind="stable").copy()
    entries["KickoffUTC"] = entries["KickoffUTC"].dt.strftime(TIME_FORMAT)
    entries["LoggedAtUTC"] = now.strftime(TIME_FORMAT)
    entries["ModelVersion"] = version
    return entries[LEDGER_COLUMNS].round(4).reset_index(drop=True)


def check_entries(entries: pd.DataFrame) -> list[str]:
    """Reasons these rows are unfit for the ledger; empty when every row is valid.

    The ledger cannot be corrected once written, so a bad row fails the run instead.
    Bookmaker odds may be missing; the model's probabilities may not.
    """
    problems = []
    for source, prefix in (("model", "P"), ("bookmaker", "B")):
        probs = entries[[f"{prefix}_{o}" for o in OUTCOMES]]
        present = probs.dropna()
        if prefix == "P" and len(present) < len(probs):
            problems.append("model probabilities missing")
        if ((present < 0) | (present > 1)).to_numpy().any():
            problems.append(f"{source} probability outside [0, 1]")
        if (present.sum(axis=1) - 1).abs().gt(SUM_TOLERANCE).any():
            problems.append(f"{source} probabilities do not sum to 1")
    kickoff = pd.to_datetime(entries["KickoffUTC"], utc=True)
    logged = pd.to_datetime(entries["LoggedAtUTC"], utc=True)
    if (kickoff <= logged).any():
        problems.append("forecast logged at or after kick-off")
    if (entries["HomeTeam"] == entries["AwayTeam"]).any():
        problems.append("a team playing itself")
    if entries.duplicated(KEY).any():
        problems.append("match logged twice")
    return problems


def load_ledger(path: Path = LEDGER_PATH) -> pd.DataFrame:
    ledger = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=LEDGER_COLUMNS)
    for column in ("KickoffUTC", "LoggedAtUTC"):
        ledger[column] = pd.to_datetime(ledger[column], format=TIME_FORMAT, utc=True)
    return ledger


def append_to_ledger(entries: pd.DataFrame, path: Path = LEDGER_PATH) -> None:
    """Append rows without rewriting the file, so logged forecasts stay byte-identical."""
    if entries.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    entries.to_csv(path, mode="a", header=not path.exists(), index=False, lineterminator="\n")


def settle(ledger: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """The ledger with the final score of every match played so far (NaN otherwise)."""
    columns = ["HomeGoals", "AwayGoals", "Result"]
    if ledger.empty or results.empty:
        return ledger.assign(**{c: np.nan for c in columns})
    final = results.rename(
        columns={"SeasonFile": "Season", "FTHG": "HomeGoals", "FTAG": "AwayGoals", "FTR": "Result"}
    )[[*KEY, *columns]].dropna(subset=["Result"])
    return ledger.merge(final, on=KEY, how="left", validate="one_to_one")


def live_scores(settled: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Accuracy and log-loss of the model and the bookmaker over settled forecasts."""
    played = settled.dropna(subset=["Result", *[f"B_{o}" for o in OUTCOMES]])
    scores = {}
    for source, prefix in (("model", "P"), ("bookmaker", "B")):
        probs = played[[f"{prefix}_{o}" for o in LABELS]].to_numpy()
        # Logged to four decimals, rows sum to 1 ± 1e-4: renormalise before scoring.
        scores[source] = score(played["Result"], probs / probs.sum(axis=1, keepdims=True))
    return scores


def main() -> None:
    now = datetime.now(UTC)
    season = season_stem(now.date())
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    results_csv = download_csv(BASE_URL.format(code=season.removeprefix(f"{LEAGUE}_")))
    (LIVE_DIR / f"{season}.csv").write_bytes(results_csv)
    fixtures_raw = pd.read_csv(io.BytesIO(download_csv(FIXTURES_URL)), encoding_errors="replace")

    results = load_live_results()
    fixtures = upcoming_fixtures(fixtures_raw, results, season)
    ledger = load_ledger()
    print(f"{season}: {len(results)} results so far, {len(fixtures)} upcoming fixtures")

    if len(fixtures):
        spec = json.loads(METRICS_PATH.read_text(encoding="utf-8"))["model"]
        forecasts = forecast(load_raw_matches(), results, fixtures, fit_frozen_model(spec))
        entries = new_entries(forecasts, ledger, now, model_version(spec))
        if problems := check_entries(entries):
            raise ValueError(f"Refusing to log invalid forecasts: {'; '.join(problems)}")
        append_to_ledger(entries)
        for row in entries.itertuples(index=False):
            print(
                f"  logged {row.KickoffUTC}  {row.HomeTeam} v {row.AwayTeam}"
                f"  H {row.P_H:.0%}  D {row.P_D:.0%}  A {row.P_A:.0%}"
            )
        print(f"Logged {len(entries)} new forecasts")
        ledger = load_ledger()

    settled = settle(ledger, results).dropna(subset=["Result"])
    if len(settled):
        print(f"Settled {len(settled)} live forecasts:")
        for source, result in live_scores(settled).items():
            print(f"  {source:10s} acc {result['accuracy']:.3f}  log-loss {result['log_loss']:.4f}")


if __name__ == "__main__":
    main()
