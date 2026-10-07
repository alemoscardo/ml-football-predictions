"""Streamlit showcase for the Premier League pre-match outcome models.

Fits the model chosen on the validation season, scores it on the most recent
season next to the benchmarks that give it meaning (random guessing, always
backing the home side, the bookmaker favourite), and shows this season's live
forecasts from the ledger that ``forecast.py`` appends to before each kick-off.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss, precision_recall_fscore_support

from feature_engineering import (
    BOOKMAKER_PROBS,
    ELO_START,
    FORM_WINDOW,
    MODEL_FEATURES,
    build_features,
)
from forecast import (
    KICKOFF_TZ,
    LIVE_DIR,
    live_scores,
    load_ledger,
    load_live_results,
    season_stem,
    settle,
)
from train_models import (
    DATA_DIR,
    METRICS_PATH,
    load_dataset,
    load_raw_matches,
    make_model,
    predict_sorted,
    split_seasons,
)

st.set_page_config(
    page_title="EPL Outcome Model",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    [data-testid="stMainBlockContainer"] { max-width: 1180px; padding-top: 2.5rem; }
    [data-testid="stSidebarCollapsedControl"] { display: none; }
    footer { visibility: hidden; }
    [data-testid="stTab"] p { font-size: 1.05rem; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

CLASS_ORDER = ["H", "D", "A"]
LABEL_MAP = {"H": "Home win", "D": "Draw", "A": "Away win"}
REPO_URL = "https://github.com/alemoscardo/ml-football-predictions"

W = FORM_WINDOW
FEATURE_LABELS = {
    "HomeElo": "Home Elo rating",
    "AwayElo": "Away Elo rating",
    "EloDiff": "Elo difference",
    "EloHomeWinProb": "Elo home-win probability",
    **{
        f"{side}{stat}L{W}": f"{side} {label}, last {W}"
        for side in ("Home", "Away")
        for stat, label in [
            ("Points", "points"),
            ("GoalsFor", "goals scored"),
            ("GoalsAgainst", "goals conceded"),
            ("ShotsTargetFor", "shots on target"),
            ("ShotsTargetAgainst", "shots on target faced"),
        ]
    },
    "HomeRestDays": "Home rest days",
    "AwayRestDays": "Away rest days",
    "FormPointsDiff": f"Points-per-game gap, last {W}",
    "GoalDiffDiff": f"Goal-difference gap, last {W}",
    "ShotsTargetDiffDiff": f"Shots-on-target gap, last {W}",
}
FAMILIES = ["Team strength (Elo)", "Recent form & rest"]


def feature_family(feature: str) -> str:
    return FAMILIES[0] if "Elo" in feature else FAMILIES[1]


# Chart colours per theme. The model is the one highlighted series; benchmarks
# stay neutral. Feature families use slots 1–2 of a CVD-validated palette.
# Outcomes run home (blue) → draw (neutral grey) → away (orange), validated for
# colour-vision deficiency against each theme's surface.
PALETTES = {
    "dark": {
        "surface": "#0e1117",
        "outcomes": ["#3987e5", "#6f6e69", "#d95926"],
        "accent": "#10B981",
        "neutral": "#5A6B85",
        "text": "#E6EAF3",
        "muted": "#8B98AD",
        "families": ["#3987e5", "#d95926"],
        "ramp": ["#0f2a22", "#10B981"],
    },
    "light": {
        "surface": "#ffffff",
        "outcomes": ["#2a78d6", "#8d8c87", "#eb6834"],
        "accent": "#059669",
        "neutral": "#A3ACB9",
        "text": "#1F2937",
        "muted": "#6B7280",
        "families": ["#2a78d6", "#eb6834"],
        "ramp": ["#ecfdf5", "#047857"],
    },
}


def season_label(stem: str) -> str:
    """``E0_2324`` → ``2023/24``."""
    code = stem.split("_")[-1]
    return f"20{code[:2]}/{code[2:]}" if len(code) == 4 and code.isdigit() else stem


def season_span(stems: list[str]) -> str:
    first, last = season_label(stems[0]), season_label(stems[-1])
    return first if first == last else f"{first} – {last}"


# --- Data & models -----------------------------------------------------------


def load_report() -> dict:
    """Model selection and scores written by ``train_models.py``.

    Not cached: the file is tiny, and a cache keyed on this function's source would
    keep serving a stale report after ``train_models.py`` rewrites it.
    """
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def files_fingerprint(paths: list[Path]) -> str:
    """Hash of the given files' names and contents.

    Passed to the cached functions below so a retrain or new results invalidate
    them; Streamlit otherwise keys caches on the function's code alone.
    """
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def inputs_fingerprint() -> str:
    """The model report and every completed season."""
    return files_fingerprint([METRICS_PATH, *sorted(DATA_DIR.glob("E0_*.csv"))])


def ratings_fingerprint() -> str:
    """Every completed and live season: what the Elo history is built from."""
    return files_fingerprint(
        [*sorted(DATA_DIR.glob("E0_*.csv")), *sorted(LIVE_DIR.glob("E0_*.csv"))]
    )


@st.cache_resource(show_spinner="Fitting the selected model…")
def load_experiment(fingerprint: str):
    """Refit the chosen spec on the train + validation seasons.

    Fitting takes a few seconds, so the app does it in-process instead of
    unpickling artifacts — no dependency on the scikit-learn version that wrote them.
    """
    matches = load_dataset()
    split = split_seasons(matches)
    fit_df = matches[matches["SeasonFile"].isin(split["train"] + split["validation"])]
    test_df = matches[matches["SeasonFile"].isin(split["test"])].reset_index(drop=True)
    spec = load_report()["model"]
    model = make_model(spec["algorithm"], spec["params"]).fit(
        fit_df[MODEL_FEATURES], fit_df["Result"]
    )
    return fit_df, test_df, model, split


@st.cache_data(show_spinner=False)
def score_model(fingerprint: str) -> pd.DataFrame:
    """One row per test-season match: model and bookmaker probabilities, picks, hits."""
    _, test_df, model, _ = load_experiment(fingerprint)
    proba = pd.DataFrame(predict_sorted(model, test_df[MODEL_FEATURES]), columns=["A", "D", "H"])[
        CLASS_ORDER
    ]
    bookie = pd.DataFrame({o: test_df[BOOKMAKER_PROBS[o]] for o in CLASS_ORDER})

    scored = pd.DataFrame(
        {
            "Date": test_df["Date"],
            "Home": test_df["HomeTeam"],
            "Away": test_df["AwayTeam"],
            "HomeGoals": test_df["HomeGoals"].astype("Int64"),
            "AwayGoals": test_df["AwayGoals"].astype("Int64"),
            "Result": test_df["Result"],
            "Pick": proba.idxmax(axis=1),
            "Confidence": proba.max(axis=1),
            "BookiePick": bookie.idxmax(axis=1),
            "BookieConfidence": bookie.max(axis=1),
        }
    )
    for outcome in CLASS_ORDER:
        scored[f"P_{outcome}"] = proba[outcome]
        scored[f"B_{outcome}"] = bookie[outcome]
    scored["Hit"] = scored["Pick"] == scored["Result"]
    scored["BookieHit"] = scored["BookiePick"] == scored["Result"]
    return scored.sort_values("Date", kind="stable").reset_index(drop=True)


@st.cache_data(show_spinner="Measuring feature importance…")
def feature_importance(fingerprint: str) -> pd.DataFrame:
    """Permutation importance on the test season (increase in log-loss)."""
    _, test_df, model, _ = load_experiment(fingerprint)
    features = MODEL_FEATURES
    result = permutation_importance(
        model,
        test_df[features],
        test_df["Result"],
        scoring="neg_log_loss",
        n_repeats=8,
        random_state=0,
    )
    importance = pd.DataFrame(
        {
            "Feature": [FEATURE_LABELS.get(f, f) for f in features],
            "Family": [feature_family(f) for f in features],
            "Importance": result.importances_mean,
        }
    )
    return importance.sort_values("Importance", ascending=False).head(10)


@st.cache_data(show_spinner=False)
def elo_history(fingerprint: str) -> pd.DataFrame:
    """Every club's Elo after each league match, completed and live seasons."""
    raw = [load_raw_matches()]
    live_results = load_live_results()
    if len(live_results):
        raw.append(live_results)
    played = build_features(pd.concat(raw, ignore_index=True)).dropna(subset=["Result"])
    sides = [
        played[["Date", "SeasonFile", f"{side}Team", f"{side}EloAfter"]].set_axis(
            ["Date", "Season", "Team", "Elo"], axis=1
        )
        for side in ("Home", "Away")
    ]
    return pd.concat(sides).sort_values("Date", kind="stable").reset_index(drop=True)


def current_ranking(history: pd.DataFrame) -> pd.DataFrame:
    """Each club in the latest season, by Elo after its most recent match."""
    latest = history[history["Season"] == history["Season"].max()]
    ranking = latest.groupby("Team").tail(1).sort_values("Elo", ascending=False)
    return ranking.assign(Rank=np.arange(1, len(ranking) + 1)).reset_index(drop=True)


def prepare_live(live: pd.DataFrame) -> pd.DataFrame:
    """Picks, UK kick-off times and display names for the settled ledger."""
    extra = ["Pick", "BookiePick", "Kickoff", "Match"]
    if live.empty:
        return live.reindex(columns=[*live.columns, *extra])
    live = live.copy()
    outcomes = np.array(CLASS_ORDER)
    model = live[[f"P_{o}" for o in CLASS_ORDER]]
    bookie = live[[f"B_{o}" for o in CLASS_ORDER]]
    live["Pick"] = outcomes[model.to_numpy().argmax(axis=1)]
    live["BookiePick"] = np.where(
        bookie.isna().any(axis=1), None, outcomes[bookie.fillna(0).to_numpy().argmax(axis=1)]
    )
    live["Kickoff"] = live["KickoffUTC"].dt.tz_convert(KICKOFF_TZ).dt.tz_localize(None)
    live["Match"] = live["HomeTeam"] + " v " + live["AwayTeam"]
    return live


def probs(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    """Probability columns in sorted label order (A, D, H), as ``log_loss`` expects."""
    return frame[[f"{prefix}_{o}" for o in sorted(CLASS_ORDER)]].to_numpy()


# --- Charts ------------------------------------------------------------------


def benchmark_chart(rows: pd.DataFrame, palette: dict[str, str]) -> alt.Chart:
    order = rows["Approach"].tolist()
    base = alt.Chart(rows).encode(
        y=alt.Y(
            "Approach:N", sort=order, title=None, axis=alt.Axis(labelLimit=220, labelFontSize=12)
        ),
        x=alt.X(
            "Accuracy:Q",
            scale=alt.Scale(domain=[0, 0.8]),
            axis=alt.Axis(format="%", tickCount=5, grid=True),
            title="Share of matches called correctly",
        ),
        tooltip=[
            alt.Tooltip("Approach:N"),
            alt.Tooltip("Accuracy:Q", format=".1%"),
            alt.Tooltip("How:N", title="How it works"),
        ],
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height=26).encode(
        color=alt.condition(
            alt.datum.IsModel,
            alt.value(palette["accent"]),
            alt.value(palette["neutral"]),
        )
    )
    labels = base.mark_text(
        align="left", dx=6, fontSize=13, fontWeight=600, color=palette["text"]
    ).encode(text=alt.Text("Accuracy:Q", format=".1%"))
    return (bars + labels).properties(height=210)


def season_chart(
    scored: pd.DataFrame,
    model_name: str,
    palette: dict[str, str],
    skip: int = 29,
    by_match: bool = False,
) -> alt.Chart:
    """Cumulative accuracy against the bookmaker favourite, by date or by match count.

    A few weeks of live matches read better by count: dates bunch up on match days.
    """
    x = (
        alt.X("Match:Q", title="Matches played", axis=alt.Axis(tickMinStep=1, format="d"))
        if by_match
        else alt.X("Date:T", title=None, axis=alt.Axis(format="%b", tickCount=10))
    )
    running = pd.DataFrame(
        {
            "Match": np.arange(1, len(scored) + 1),
            "Date": scored["Date"],
            model_name: scored["Hit"].expanding().mean(),
            "Bookmaker favourite": scored["BookieHit"].expanding().mean(),
        }
    ).iloc[skip:]  # the first few weeks swing wildly on tiny samples
    long = running.melt(["Match", "Date"], var_name="Series", value_name="Accuracy")
    color = alt.Color(
        "Series:N",
        scale=alt.Scale(
            domain=[model_name, "Bookmaker favourite"],
            range=[palette["accent"], palette["neutral"]],
        ),
        legend=alt.Legend(orient="top", title=None),
    )
    lines = (
        alt.Chart(long)
        .mark_line(strokeWidth=2)
        .encode(
            x=x,
            y=alt.Y(
                "Accuracy:Q",
                scale=alt.Scale(domain=[0.3, 0.7]),
                axis=alt.Axis(format="%", tickCount=4),
                title="Cumulative accuracy",
            ),
            color=color,
        )
    )
    hover = alt.selection_point(fields=["Match"], nearest=True, on="pointerover", empty=False)
    rule = (
        alt.Chart(running)
        .mark_rule(color=palette["muted"], strokeWidth=1)
        .encode(
            x=x,
            opacity=alt.condition(hover, alt.value(0.6), alt.value(0)),
            tooltip=[
                alt.Tooltip("Date:T", format="%d %b %Y"),
                alt.Tooltip("Match:Q", title="Matches played"),
                alt.Tooltip(f"{model_name}:Q", format=".1%"),
                alt.Tooltip("Bookmaker favourite:Q", format=".1%"),
            ],
        )
        .add_params(hover)
    )
    return (lines + rule).properties(height=260)


def confidence_chart(scored: pd.DataFrame, palette: dict[str, str]) -> alt.Chart:
    bands = pd.cut(
        scored["Confidence"],
        bins=[0, 0.4, 0.5, 0.6, 0.7, 1.0],
        labels=["< 40%", "40–50%", "50–60%", "60–70%", "70%+"],
    )
    grouped = (
        scored.groupby(bands, observed=True)
        .agg(HitRate=("Hit", "mean"), Stated=("Confidence", "mean"), Matches=("Hit", "size"))
        .reset_index(names="Band")
    )
    order = grouped["Band"].astype(str).tolist()
    grouped["Band"] = grouped["Band"].astype(str)
    tooltip = [
        alt.Tooltip("Band:N", title="Model confidence"),
        alt.Tooltip("HitRate:Q", title="Actually correct", format=".0%"),
        alt.Tooltip("Stated:Q", title="Average stated confidence", format=".0%"),
        alt.Tooltip("Matches:Q"),
    ]
    base = alt.Chart(grouped).encode(
        x=alt.X("Band:N", sort=order, title="Model confidence", axis=alt.Axis(labelAngle=0))
    )
    bars = base.mark_bar(cornerRadiusEnd=4, width={"band": 0.55}, color=palette["accent"]).encode(
        y=alt.Y(
            "HitRate:Q",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(format="%", tickCount=5),
            title="Actually correct",
        ),
        tooltip=tooltip,
    )
    ticks = base.mark_tick(thickness=2, size=44, color=palette["muted"]).encode(
        y="Stated:Q", tooltip=tooltip
    )
    counts = base.mark_text(dy=-8, fontSize=11, color=palette["muted"]).encode(
        y="HitRate:Q", text=alt.Text("Matches:Q", format="d")
    )
    return (bars + ticks + counts).properties(height=260)


def confusion_chart(scored: pd.DataFrame, palette: dict[str, str]) -> alt.Chart:
    labels = list(LABEL_MAP.values())
    counts = pd.crosstab(scored["Result"].map(LABEL_MAP), scored["Pick"].map(LABEL_MAP))
    counts = counts.reindex(index=labels, columns=labels, fill_value=0)
    share = counts.div(counts.sum(axis=1), axis=0)
    cells = (
        counts.stack()
        .rename("Count")
        .to_frame()
        .join(share.stack().rename("Share"))
        .reset_index(names=["Actual", "Predicted"])
    )
    base = alt.Chart(cells).encode(
        x=alt.X("Predicted:N", sort=labels, axis=alt.Axis(orient="top", labelAngle=0)),
        y=alt.Y("Actual:N", sort=labels),
    )
    rect = base.mark_rect(cornerRadius=4, stroke=None).encode(
        color=alt.Color(
            "Share:Q",
            scale=alt.Scale(domain=[0, 1], range=palette["ramp"]),
            legend=None,
        ),
        tooltip=[
            "Actual:N",
            "Predicted:N",
            alt.Tooltip("Count:Q", title="Matches"),
            alt.Tooltip("Share:Q", title="Share of actual outcome", format=".0%"),
        ],
    )
    text = base.mark_text(fontSize=14, fontWeight=600).encode(
        text="Count:Q",
        color=alt.condition(alt.datum.Share > 0.5, alt.value("white"), alt.value(palette["muted"])),
    )
    return (rect + text).properties(height=260)


def importance_chart(importance: pd.DataFrame, palette: dict[str, str]) -> alt.Chart:
    order = importance["Feature"].tolist()
    present = [f for f in FAMILIES if f in set(importance["Family"])]
    colours = [palette["families"][FAMILIES.index(f)] for f in present]
    base = alt.Chart(importance).encode(
        y=alt.Y(
            "Feature:N",
            sort=order,
            title=None,
            axis=alt.Axis(labelLimit=220, labelOverlap=False, labelFontSize=12),
        ),
        x=alt.X(
            "Importance:Q", title="Increase in log-loss when shuffled", axis=alt.Axis(tickCount=4)
        ),
        tooltip=[
            "Feature:N",
            "Family:N",
            alt.Tooltip("Importance:Q", format=".4f"),
        ],
    )
    return (
        base.mark_bar(cornerRadiusEnd=4, height=14)
        .encode(
            color=alt.Color(
                "Family:N",
                scale=alt.Scale(
                    domain=present,
                    range=colours,
                ),
                legend=alt.Legend(orient="top", title=None),
            )
        )
        .properties(height=360)
    )


def match_chart(match: pd.Series, model_name: str, palette: dict[str, str]) -> alt.Chart:
    rows = pd.DataFrame(
        [
            {"Outcome": LABEL_MAP[o], "Source": source, "Probability": match[f"{prefix}_{o}"]}
            for o in CLASS_ORDER
            for source, prefix in ((model_name, "P"), ("Bookmaker", "B"))
        ]
    )
    labels = list(LABEL_MAP.values())
    base = alt.Chart(rows).encode(
        y=alt.Y("Outcome:N", sort=labels, title=None),
        yOffset=alt.YOffset("Source:N", sort=[model_name, "Bookmaker"]),
        x=alt.X(
            "Probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title=None
        ),
        tooltip=["Outcome:N", "Source:N", alt.Tooltip("Probability:Q", format=".0%")],
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height=12).encode(
        color=alt.Color(
            "Source:N",
            scale=alt.Scale(
                domain=[model_name, "Bookmaker"], range=[palette["accent"], palette["neutral"]]
            ),
            legend=alt.Legend(orient="top", title=None),
        )
    )
    text = base.mark_text(align="left", dx=5, fontSize=11, color=palette["muted"]).encode(
        text=alt.Text("Probability:Q", format=".0%")
    )
    return (bars + text).properties(height=190)


def fixtures_chart(pending: pd.DataFrame, palette: dict[str, str]) -> alt.Chart:
    """Each upcoming match: a title row, then the model's split and Bet365's.

    Match names sit on their own row rather than on the axis, so the bars keep the
    full width on a phone.
    """
    rows, titles, order = [], [], []
    for i, match in enumerate(pending.itertuples(index=False)):
        # Unique row keys; the axis shows only the part after "|".
        title_key = f"{i:02d}T|"
        order.append(title_key)
        titles.append(
            {"Row": title_key, "Label": f"{match.Match} · {match.Kickoff:%a %d %b, %H:%M}"}
        )
        for source, prefix in (("Model", "P"), ("Bet365", "B")):
            values = [getattr(match, f"{prefix}_{o}") for o in CLASS_ORDER]
            if any(pd.isna(v) for v in values):
                continue
            key = f"{i:02d}{prefix}|{source}"
            order.append(key)
            start = 0.0
            for outcome, p in zip(CLASS_ORDER, values, strict=True):
                rows.append(
                    {
                        "Row": key,
                        "Match": match.Match,
                        "Source": source,
                        "Outcome": LABEL_MAP[outcome],
                        "Probability": p,
                        "Start": start,
                        "End": start + p,
                        "Mid": start + p / 2,
                        "IsModel": prefix == "P",
                    }
                )
                start += p
    data = pd.DataFrame(rows)
    scale = alt.Scale(domain=[0, 1])
    y = alt.Y(
        "Row:N",
        sort=order,
        title=None,
        scale=alt.Scale(paddingInner=0.15),
        axis=alt.Axis(
            labelExpr="split(datum.value, '|')[1]",
            labelFontSize=11,
            labelColor=alt.ExprRef(
                expr=f"indexof(datum.value, 'B|') > 0 ? '{palette['muted']}' : '{palette['text']}'"
            ),
            ticks=False,
            domain=False,
        ),
    )
    tooltip = [
        alt.Tooltip("Match:N"),
        alt.Tooltip("Source:N"),
        alt.Tooltip("Outcome:N"),
        alt.Tooltip("Probability:Q", format=".0%"),
    ]
    bars = (
        alt.Chart(data)
        .mark_bar(cornerRadius=2, stroke=palette["surface"], strokeWidth=2)
        .encode(
            y=y,
            x=alt.X(
                "Start:Q",
                scale=scale,
                title=None,
                axis=alt.Axis(format="%", tickCount=4, grid=False),
            ),
            x2="End:Q",
            color=alt.Color(
                "Outcome:N",
                scale=alt.Scale(domain=list(LABEL_MAP.values()), range=palette["outcomes"]),
                legend=alt.Legend(orient="top", title=None),
            ),
            size=alt.condition(alt.datum.IsModel, alt.value(16), alt.value(9)),
            opacity=alt.condition(alt.datum.IsModel, alt.value(1.0), alt.value(0.55)),
            tooltip=tooltip,
        )
    )
    values = (
        alt.Chart(data[data["IsModel"] & (data["Probability"] >= 0.12)])
        .mark_text(fontSize=11, fontWeight=600, color="white")
        .encode(
            y=y,
            x=alt.X("Mid:Q", scale=scale),
            text=alt.Text("Probability:Q", format=".0%"),
            tooltip=tooltip,
        )
    )
    names = (
        alt.Chart(pd.DataFrame(titles))
        .mark_text(align="left", baseline="middle", fontSize=13, fontWeight=600)
        .encode(
            y=y,
            x=alt.X("x:Q", scale=scale),
            text="Label:N",
            color=alt.value(palette["text"]),
        )
        .transform_calculate(x="0")
    )
    return (bars + values + names).properties(height=22 * len(order) + 10)


def elo_ranking_chart(ranking: pd.DataFrame, team: str, palette: dict[str, str]) -> alt.Chart:
    """Lollipops from the league average (1500) to each club's current rating."""
    data = ranking.assign(Selected=ranking["Team"] == team, Average=ELO_START)
    low = min(data["Elo"].min(), ELO_START) - 25
    high = max(data["Elo"].max(), ELO_START) + 25
    scale = alt.Scale(domain=[low, high], zero=False)
    y = alt.Y(
        "Team:N",
        sort=data["Team"].tolist(),
        title=None,
        axis=alt.Axis(
            labelLimit=150, labelFontSize=12, labelOverlap=False, ticks=False, domain=False
        ),
    )
    colour = alt.condition(
        alt.datum.Selected, alt.value(palette["accent"]), alt.value(palette["neutral"])
    )
    tooltip = [
        alt.Tooltip("Rank:Q"),
        alt.Tooltip("Team:N"),
        alt.Tooltip("Elo:Q", format=".0f"),
    ]
    average = (
        alt.Chart(pd.DataFrame({"Elo": [ELO_START]}))
        .mark_rule(color=palette["muted"], strokeWidth=1)
        .encode(x=alt.X("Elo:Q", scale=scale))
    )
    stems = (
        alt.Chart(data)
        .mark_rule(strokeWidth=2)
        .encode(y=y, x=alt.X("Average:Q", scale=scale), x2="Elo:Q", color=colour, tooltip=tooltip)
    )
    dots = (
        alt.Chart(data)
        .mark_circle(size=90, opacity=1)
        .encode(
            y=y,
            x=alt.X("Elo:Q", scale=scale, title="Elo rating", axis=alt.Axis(tickCount=5)),
            color=colour,
            tooltip=tooltip,
        )
    )
    return (average + stems + dots).properties(height=22 * len(data) + 20)


def elo_history_chart(history: pd.DataFrame, team: str, palette: dict[str, str]) -> alt.Chart:
    """One club's rating after every match; the axis is shared by all clubs."""
    scale = alt.Scale(domain=[history["Elo"].min() - 20, history["Elo"].max() + 20], zero=False)
    data = history[history["Team"] == team].assign(
        SeasonLabel=lambda d: d["Season"].map(season_label)
    )
    hover = alt.selection_point(fields=["Date"], nearest=True, on="pointerover", empty=False)
    base = alt.Chart(data).encode(
        x=alt.X("Date:T", title=None, axis=alt.Axis(format="%Y", tickCount=8)),
        y=alt.Y("Elo:Q", scale=scale, title="Elo rating", axis=alt.Axis(tickCount=5)),
    )
    average = (
        alt.Chart(pd.DataFrame({"Elo": [ELO_START]}))
        .mark_rule(color=palette["muted"], strokeWidth=1)
        .encode(y=alt.Y("Elo:Q", scale=scale))
    )
    # One line per season, so years outside the league stay a gap.
    line = base.mark_line(strokeWidth=2, color=palette["accent"]).encode(detail="Season:N")
    points = (
        base.mark_circle(size=70, color=palette["accent"])
        .encode(
            opacity=alt.condition(hover, alt.value(1), alt.value(0)),
            tooltip=[
                alt.Tooltip("Date:T", format="%d %b %Y"),
                alt.Tooltip("SeasonLabel:N", title="Season"),
                alt.Tooltip("Elo:Q", format=".0f"),
            ],
        )
        .add_params(hover)
    )
    rule = base.mark_rule(color=palette["muted"], strokeWidth=1).transform_filter(hover)
    return (average + line + rule + points).properties(height=440)


# --- Load --------------------------------------------------------------------

MODEL_LABEL = "Model"
LIVE_CHART_MIN = 20  # matches before the running live accuracy is worth a chart

try:
    report = load_report()
    fingerprint = inputs_fingerprint()
    fit_df, _, _, split = load_experiment(fingerprint)
except Exception as exc:
    st.error(f"Unable to fit the models: {exc}")
    st.stop()

palette = PALETTES.get(st.context.theme.type or "dark", PALETTES["dark"])
train_label = season_span(split["train"])
fit_label = season_span(split["train"] + split["validation"])
validation_label = season_label(split["validation"][0])
test_label = season_label(split["test"][0])
live_label = season_label(season_stem(pd.Timestamp.now(tz="UTC").date()))
spec = report["model"]

scored = score_model(fingerprint)
y_true = scored["Result"]
accuracy = scored["Hit"].mean()
bookie_accuracy = scored["BookieHit"].mean()
model_loss = log_loss(y_true, probs(scored, "P"), labels=sorted(CLASS_ORDER))
bookie_loss = log_loss(y_true, probs(scored, "B"), labels=sorted(CLASS_ORDER))

live = prepare_live(settle(load_ledger(), load_live_results()))
played = live.dropna(subset=["Result"])
pending = live[live["Result"].isna()].sort_values("KickoffUTC", kind="stable")
live_score = live_scores(played) if len(played) else None

# --- Header ------------------------------------------------------------------

st.title("Premier League Outcome Model")
st.markdown(
    "Forecasts Premier League results **before kick-off** from public match history alone "
    "(Elo ratings and recent form, no bookmaker odds) and keeps score against the "
    f"bookmaker: once on {test_label}, a season the model never saw, and live through "
    f"{live_label}, one round at a time."
)
st.caption(
    f"{spec['algorithm']} · {len(MODEL_FEATURES)} pre-match features · fitted on "
    f"{len(fit_df):,} matches ({fit_label})"
)

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    f"Backtest accuracy · {test_label}",
    f"{accuracy:.1%}",
    delta=f"{(accuracy - bookie_accuracy) * 100:+.1f} pts vs Bet365",
    help=f"{int(scored['Hit'].sum())} of {len(scored)} matches called correctly, "
    "on a season the model never saw.",
    border=True,
    height="stretch",
)
k2.metric(
    "Backtest log-loss",
    f"{model_loss:.3f}",
    delta=f"{model_loss - bookie_loss:+.3f} vs Bet365",
    delta_color="inverse",
    help="How good the probabilities are, not just the picks. Lower is better; "
    "always saying ⅓-⅓-⅓ scores 1.099.",
    border=True,
    height="stretch",
)
k3.metric(
    f"Live forecasts · {live_label}",
    f"{len(live)}" if len(live) else "—",
    help=(
        f"Logged before kick-off since {live['LoggedAtUTC'].min():%d %B %Y}; "
        f"{len(pending)} awaiting a result."
        if len(live)
        else "The first forecasts are logged when the next round of fixtures is published."
    ),
    border=True,
    height="stretch",
)
if live_score:
    k4.metric(
        "Live log-loss",
        f"{live_score['model']['log_loss']:.3f}",
        delta=f"{live_score['model']['log_loss'] - live_score['bookmaker']['log_loss']:+.3f} "
        "vs Bet365",
        delta_color="inverse",
        help=f"Over the {len(played)} live forecasts played so far.",
        border=True,
        height="stretch",
    )
else:
    k4.metric(
        "Live log-loss",
        "—",
        help="Shown once the first forecast match is played.",
        border=True,
        height="stretch",
    )

# ?tab=live|backtest|teams|method links straight to a tab; otherwise open on the
# live record once it exists, and on the backtest until then.
TABS = {
    "live": f"Live {live_label}",
    "backtest": f"Backtest {test_label}",
    "teams": "Teams",
    "method": "How it works",
}
opening = TABS.get(st.query_params.get("tab", ""), TABS["live" if len(live) else "backtest"])
live_tab, backtest_tab, teams_tab, method_tab = st.tabs(list(TABS.values()), default=opening)

# --- Live --------------------------------------------------------------------

with live_tab:
    st.markdown(
        "A backtest can always hide a subtle leak; a forecast published before the match "
        "cannot. A scheduled GitHub Actions job forecasts every match **before kick-off** and "
        "commits it to an append-only ledger, so the "
        f"[commit history]({REPO_URL}/commits/main/predictions/live.csv) timestamps each one. "
        "The model is frozen for the season."
    )

    st.subheader("Next matches")
    if len(pending):
        with st.container(border=True):
            st.altair_chart(fixtures_chart(pending, palette), width="stretch")
            st.caption(
                "For each match, the model's forecast as logged, then Bet365's odds with the "
                "margin removed. Kick-off times are UK time."
            )
        with st.expander("Table view"):
            st.dataframe(
                pd.DataFrame(
                    {
                        "Kick-off": pending["Kickoff"],
                        "Match": pending["Match"],
                        **{LABEL_MAP[o]: pending[f"P_{o}"] * 100 for o in CLASS_ORDER},
                        "Bet365": [
                            " · ".join(f"{row[f'B_{o}']:.0%}" for o in CLASS_ORDER)
                            for _, row in pending.iterrows()
                        ],
                        "Logged": [
                            f"{lead.days}d {lead.seconds // 3600}h before"
                            for lead in pending["KickoffUTC"] - pending["LoggedAtUTC"]
                        ],
                    }
                ),
                hide_index=True,
                column_config={
                    "Kick-off": st.column_config.DatetimeColumn(format="ddd D MMM, HH:mm"),
                    **{
                        LABEL_MAP[o]: st.column_config.ProgressColumn(
                            format="%.0f%%", min_value=0, max_value=100, width=90
                        )
                        for o in CLASS_ORDER
                    },
                    "Bet365": st.column_config.TextColumn(
                        help="Bet365 probabilities, margin removed: home · draw · away"
                    ),
                },
            )
            st.caption("Kick-off times are UK time.")
    else:
        st.info(
            "No match is waiting for a result. The next forecasts appear here as soon as "
            "Football-Data.co.uk publishes the next round, usually a few days before kick-off.",
            icon=":material/schedule:",
        )

    st.subheader("Results so far")
    if len(played):
        settled = played.sort_values("KickoffUTC", kind="stable")
        live_hits = settled.assign(
            Date=settled["Kickoff"],
            Hit=settled["Pick"] == settled["Result"],
            BookieHit=settled["BookiePick"] == settled["Result"],
        )
        if len(played) >= LIVE_CHART_MIN:
            with st.container(border=True):
                st.markdown("**Running accuracy**")
                st.caption("Share of live picks called correctly, against the bookmaker favourite.")
                st.altair_chart(
                    season_chart(live_hits, MODEL_LABEL, palette, skip=9, by_match=True),
                    width="stretch",
                )
        recent = live_hits.iloc[::-1]
        st.dataframe(
            pd.DataFrame(
                {
                    "Date": recent["Kickoff"],
                    "Match": recent["HomeTeam"]
                    + " "
                    + recent["HomeGoals"].astype(int).astype(str)
                    + "–"
                    + recent["AwayGoals"].astype(int).astype(str)
                    + " "
                    + recent["AwayTeam"],
                    "Prediction": recent["Pick"].map(LABEL_MAP),
                    "Confidence": recent[[f"P_{o}" for o in CLASS_ORDER]].max(axis=1) * 100,
                    "Bookmaker": recent["BookiePick"].map(LABEL_MAP),
                    "": np.where(recent["Hit"], "✓", "✗"),
                }
            ),
            hide_index=True,
            height=min(420, 35 * len(recent) + 38),
            column_config={
                "Date": st.column_config.DateColumn(format="DD MMM"),
                "Confidence": st.column_config.ProgressColumn(
                    format="%.0f%%", min_value=0, max_value=100, width=100
                ),
                "Bookmaker": st.column_config.TextColumn("Bookmaker pick"),
                "": st.column_config.TextColumn(width="small"),
            },
        )
        st.caption(
            f"{len(played)} matches so far. Telling the model from the bookmaker takes a few "
            "hundred, so read the live record as a check on the backtest, not a verdict yet."
        )
    else:
        st.caption("The live record starts when the first forecast match is played.")

# --- Backtest ----------------------------------------------------------------

with backtest_tab:
    st.markdown(
        f"Candidates were trained on {train_label} and compared on {validation_label}; the "
        f"winner was refitted on both and scored **once** on {test_label}: "
        f"{int(scored['Hit'].sum())} of {len(scored)} matches called correctly."
    )

    st.subheader("How it compares")
    benchmarks = pd.DataFrame(
        [
            ("Random guess", 1 / 3, "Pick one of the three outcomes at random", False),
            ("Always back the home side", (y_true == "H").mean(), "Home win every time", False),
            ("Bookmaker favourite", bookie_accuracy, "Shortest Bet365 odds", False),
            ("Model", accuracy, "This model, on the unseen season", True),
        ],
        columns=["Approach", "Accuracy", "How", "IsModel"],
    )
    with st.container(border=True):
        st.altair_chart(benchmark_chart(benchmarks, palette), width="stretch")
        st.caption(
            "The bookmaker is the benchmark to beat: its odds already price in injuries, "
            "line-ups and market money that the model never sees. Matching it from public "
            "match history alone is the realistic goal."
        )

    st.subheader("How it behaves")
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown("**Across the season**")
        st.caption("Running accuracy as the season unfolds, against the bookmaker favourite.")
        st.altair_chart(season_chart(scored, MODEL_LABEL, palette), width="stretch")
    with right, st.container(border=True):
        st.markdown("**When it is confident, is it right?**")
        st.caption(
            "Bars: how often picks in each band were correct. Ticks: the confidence the model "
            "stated; ticks close to the bars mean well-calibrated probabilities. Labels: matches."
        )
        st.altair_chart(confidence_chart(scored, palette), width="stretch")

    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown("**Where the misses go**")
        draws_called = int((scored["Pick"] == "D").sum())
        st.caption(
            f"Rows are real results, columns are predictions. Draws are the blind spot: "
            f"{int((y_true == 'D').sum())} happened, the model called {draws_called}. "
            "A draw is rarely the single most likely outcome, for bookmakers too."
        )
        st.altair_chart(confusion_chart(scored, palette), width="stretch")
    with right, st.container(border=True):
        st.markdown("**What drives the predictions**")
        st.caption(
            "Top ten features by how much shuffling each one hurts the test-season log-loss."
        )
        st.altair_chart(importance_chart(feature_importance(fingerprint), palette), width="stretch")

    st.subheader("Every match")
    f1, f2 = st.columns([2, 1], vertical_alignment="bottom")
    teams = sorted(set(scored["Home"]) | set(scored["Away"]))
    selected_teams = f1.multiselect("Team", teams, placeholder="All teams")
    show = f2.segmented_control(
        "Show", ["All", "Misses", "Model beat bookmaker"], default="All", required=True
    )

    view = scored
    if selected_teams:
        view = view[view["Home"].isin(selected_teams) | view["Away"].isin(selected_teams)]
    if show == "Misses":
        view = view[~view["Hit"]]
    elif show == "Model beat bookmaker":
        view = view[view["Hit"] & ~view["BookieHit"]]

    table = pd.DataFrame(
        {
            "Date": view["Date"],
            "Match": view["Home"]
            + " "
            + view["HomeGoals"].astype(str)
            + "–"
            + view["AwayGoals"].astype(str)
            + " "
            + view["Away"],
            "Prediction": view["Pick"].map(LABEL_MAP),
            "Confidence": view["Confidence"] * 100,
            "Bookmaker": view["BookiePick"].map(LABEL_MAP),
            "": np.where(view["Hit"], "✓", "✗"),
        }
    )

    table_col, detail_col = st.columns([5, 3])
    with table_col:
        selection = st.dataframe(
            table,
            hide_index=True,
            height=420,
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "Date": st.column_config.DateColumn(format="DD MMM", width=60),
                "Match": st.column_config.TextColumn(width=220),
                "Prediction": st.column_config.TextColumn(width=80),
                "Confidence": st.column_config.ProgressColumn(
                    format="%.0f%%", min_value=0, max_value=100, width=100
                ),
                "Bookmaker": st.column_config.TextColumn("Bookmaker pick", width=105),
                "": st.column_config.TextColumn(width="small"),
            },
        )
        st.caption(f"{len(table)} matches · select a row to compare probabilities.")

    with detail_col, st.container(border=True):
        rows = selection.selection.rows if selection else []
        if len(view):
            match = view.iloc[rows[0]] if rows else view.iloc[0]
            st.markdown(
                f"**{match['Home']} {match['HomeGoals']}–{match['AwayGoals']} {match['Away']}**"
                f"  \n{match['Date']:%A %d %B %Y} · result: {LABEL_MAP[match['Result']]}"
            )
            st.altair_chart(match_chart(match, MODEL_LABEL, palette), width="stretch")
            verdict = "✓ model right" if match["Hit"] else "✗ model wrong"
            bookie_verdict = "✓ bookmaker right" if match["BookieHit"] else "✗ bookmaker wrong"
            st.caption(f"{verdict} · {bookie_verdict}")
        else:
            st.info("No matches for this filter.")

    st.download_button(
        "Download predictions (CSV)",
        table.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{split['test'][0]}.csv",
        mime="text/csv",
    )

# --- Teams -------------------------------------------------------------------

with teams_tab:
    history = elo_history(ratings_fingerprint())
    ranking = current_ranking(history)
    names = sorted(ranking["Team"])
    picker, _ = st.columns([1, 2])
    team = picker.selectbox("Club", names, index=names.index(ranking["Team"].iloc[0]))
    position = ranking.set_index("Team").loc[team]

    left, right = st.columns([2, 3])
    with left, st.container(border=True):
        st.markdown(f"**Strongest right now · {season_label(history['Season'].max())}**")
        st.caption(
            "Elo after each club's latest match. The line is 1500, the long-run league "
            "average; promoted clubs new to the data start at 1420."
        )
        st.altair_chart(elo_ranking_chart(ranking, team, palette), width="stretch")
    with right, st.container(border=True):
        st.markdown(f"**{team} since {season_label(history['Season'].min())}**")
        st.caption(
            f"Rated {position['Elo']:.0f} after its latest match, #{int(position['Rank'])} of "
            f"{len(ranking)}. Gaps are seasons outside the Premier League."
        )
        st.altair_chart(elo_history_chart(history, team, palette), width="stretch")

# --- How it works ------------------------------------------------------------

with method_tab:
    method, metrics_col = st.columns([3, 2])
    with method:
        st.markdown("#### Method and limitations")
        st.markdown(
            f"""
- **Task:** three-way classification (home win / draw / away win), scored on probabilities
  (log-loss) as well as picks (accuracy).
- **Features ({len(MODEL_FEATURES)}):** Elo ratings updated after every result and carried
  across seasons; each side's points, goals and shots on target over its last {FORM_WINDOW}
  matches; rest days. Every value is computed from matches played *before* the one being
  predicted. Bet365 odds are used only as the benchmark, never as inputs.
- **Validation:** strictly chronological. {season_label(split["burn_in"][0])} warms up the
  ratings. Candidates train on {train_label} ({report["rows"]["train"]:,} matches) and are
  compared on {validation_label}; the winner is refitted on {fit_label}
  ({len(fit_df):,} matches) and scored once on {test_label}.
- **Live:** every few hours a GitHub Actions job forecasts each fixture not yet kicked off
  and appends it to [`predictions/live.csv`]({REPO_URL}/blob/main/predictions/live.csv).
  Rows are checked (probabilities in range and summing to one, logged before kick-off) and
  never rewritten.
- **Limitations:** no line-ups, injuries or expected-goals data, which is where bookmakers
  get their edge. Draws are almost never the top pick.

Data: [Football-Data.co.uk](https://www.football-data.co.uk/englandm.php) ·
Code: [GitHub]({REPO_URL})
"""
        )
    with metrics_col:
        st.markdown("#### Per-outcome metrics")
        precision, recall, f1_score, support = precision_recall_fscore_support(
            y_true, scored["Pick"], labels=CLASS_ORDER, zero_division=0
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "Outcome": [LABEL_MAP[o] for o in CLASS_ORDER],
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1_score,
                    "Matches": support,
                }
            ),
            hide_index=True,
            column_config={
                metric: st.column_config.NumberColumn(format="%.2f")
                for metric in ("Precision", "Recall", "F1")
            },
        )
        st.markdown(f"#### Model selection on {validation_label}")
        trials = pd.DataFrame(
            {
                "Algorithm": [t["algorithm"] for t in spec["trials"]],
                "Settings": [
                    ", ".join(f"{k}={v}" for k, v in t["params"].items()).replace(
                        "min_samples_leaf", "min_leaf"
                    )
                    for t in spec["trials"]
                ],
                "Log-loss": [t["log_loss"] for t in spec["trials"]],
            }
        )
        st.dataframe(
            trials,
            hide_index=True,
            height=250,
            column_config={
                "Algorithm": st.column_config.TextColumn(width=130),
                "Settings": st.column_config.TextColumn(width=165),
                "Log-loss": st.column_config.NumberColumn(format="%.4f", width=70),
            },
        )
        st.caption(
            f"Every candidate fitted on {train_label} and scored on {validation_label}; "
            "the top row is refitted on both and used in the backtest."
        )
