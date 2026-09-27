"""Streamlit showcase for the Premier League pre-match outcome models.

Fits the model chosen on the validation season, scores it on the most recent
season, and puts the result next to the benchmarks that give it meaning:
random guessing, always backing the home side, and the bookmaker favourite.
"""

from __future__ import annotations

import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss, precision_recall_fscore_support

from feature_engineering import FEATURE_SETS, FORM_WINDOW, ODDS_FEATURES
from train_models import METRICS_PATH, load_dataset, make_model, predict_sorted, split_seasons

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
    </style>
    """,
    unsafe_allow_html=True,
)

CLASS_ORDER = ["H", "D", "A"]
LABEL_MAP = {"H": "Home win", "D": "Draw", "A": "Away win"}
BOOKIE_COLUMNS = {"H": "NormProbHome_B365", "D": "NormProbDraw_B365", "A": "NormProbAway_B365"}

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
    "NormProbHome_B365": "Bookmaker P(home win)",
    "NormProbDraw_B365": "Bookmaker P(draw)",
    "NormProbAway_B365": "Bookmaker P(away win)",
}
FAMILIES = ["Team strength (Elo)", "Recent form & rest", "Bookmaker odds"]


def feature_family(feature: str) -> str:
    if feature in ODDS_FEATURES:
        return FAMILIES[2]
    return FAMILIES[0] if "Elo" in feature else FAMILIES[1]


# Chart colours per theme. The model is the one highlighted series; benchmarks
# stay neutral. Feature families use slots 1–3 of a CVD-validated palette.
PALETTES = {
    "dark": {
        "accent": "#10B981",
        "neutral": "#5A6B85",
        "text": "#E6EAF3",
        "muted": "#8B98AD",
        "families": ["#3987e5", "#d95926", "#199e70"],
        "ramp": ["#0f2a22", "#10B981"],
    },
    "light": {
        "accent": "#059669",
        "neutral": "#A3ACB9",
        "text": "#1F2937",
        "muted": "#6B7280",
        "families": ["#2a78d6", "#eb6834", "#1baf7a"],
        "ramp": ["#ecfdf5", "#047857"],
    },
}


def season_label(stem: str) -> str:
    """``E0_2324`` → ``2023/24``."""
    code = stem.split("_")[-1]
    return f"20{code[:2]}/{code[2:]}" if len(code) == 4 and code.isdigit() else stem


def season_span(stems: list[str]) -> str:
    return season_label(stems[0]) if len(stems) == 1 else f"{season_label(stems[0])} – {season_label(stems[-1])}"


# --- Data & models -----------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_report() -> dict:
    """Model selection and scores written by ``train_models.py``."""
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="Fitting the selected models…")
def load_experiment():
    """Refit each feature set's chosen spec on train + validation seasons.

    Fitting takes a few seconds, so the app does it in-process instead of
    unpickling artifacts — no dependency on the scikit-learn version that wrote them.
    """
    matches = load_dataset()
    split = split_seasons(matches)
    fit_df = matches[matches["SeasonFile"].isin(split["train"] + split["validation"])]
    test_df = matches[matches["SeasonFile"].isin(split["test"])].reset_index(drop=True)
    report = load_report()
    models = {}
    for set_name, features in FEATURE_SETS.items():
        spec = report["feature_sets"][set_name]
        models[set_name] = make_model(spec["algorithm"], spec["params"]).fit(
            fit_df[features], fit_df["Result"]
        )
    return fit_df, test_df, models, split


@st.cache_data(show_spinner=False)
def score_model(set_name: str) -> pd.DataFrame:
    """One row per test-season match: model and bookmaker probabilities, picks, hits."""
    _, test_df, models, _ = load_experiment()
    proba = pd.DataFrame(
        predict_sorted(models[set_name], test_df[FEATURE_SETS[set_name]]), columns=["A", "D", "H"]
    )[CLASS_ORDER]
    bookie = pd.DataFrame({o: test_df[c] for o, c in BOOKIE_COLUMNS.items()})

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
def feature_importance(set_name: str) -> pd.DataFrame:
    """Permutation importance on the test season (increase in log-loss)."""
    _, test_df, models, _ = load_experiment()
    features = FEATURE_SETS[set_name]
    result = permutation_importance(
        models[set_name],
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
    ).encode(
        text=alt.Text("Accuracy:Q", format=".1%")
    )
    return (bars + labels).properties(height=210)


def season_chart(scored: pd.DataFrame, model_name: str, palette: dict[str, str]) -> alt.Chart:
    running = pd.DataFrame(
        {
            "Match": np.arange(1, len(scored) + 1),
            "Date": scored["Date"],
            model_name: scored["Hit"].expanding().mean(),
            "Bookmaker favourite": scored["BookieHit"].expanding().mean(),
        }
    ).iloc[29:]  # the first few weeks swing wildly on tiny samples
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
            x=alt.X("Date:T", title=None, axis=alt.Axis(format="%b", tickCount=10)),
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
            x="Date:T",
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
        counts.stack().rename("Count").to_frame().join(share.stack().rename("Share"))
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
        x=alt.X("Importance:Q", title="Increase in log-loss when shuffled", axis=alt.Axis(tickCount=4)),
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
        x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title=None),
        tooltip=["Outcome:N", "Source:N", alt.Tooltip("Probability:Q", format=".0%")],
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height=12).encode(
        color=alt.Color(
            "Source:N",
            scale=alt.Scale(domain=[model_name, "Bookmaker"], range=[palette["accent"], palette["neutral"]]),
            legend=alt.Legend(orient="top", title=None),
        )
    )
    text = base.mark_text(align="left", dx=5, fontSize=11, color=palette["muted"]).encode(
        text=alt.Text("Probability:Q", format=".0%")
    )
    return (bars + text).properties(height=190)


# --- Load --------------------------------------------------------------------

MODEL_LABEL = "Model"

try:
    report = load_report()
    fit_df, _, _, split = load_experiment()
except Exception as exc:
    st.error(f"Unable to fit the models: {exc}")
    st.stop()

palette = PALETTES.get(st.context.theme.type or "dark", PALETTES["dark"])
train_label = season_span(split["train"])
validation_label = season_label(split["validation"][0])
test_label = season_label(split["test"][0])

# --- Header ------------------------------------------------------------------

head, picker = st.columns([3, 1], vertical_alignment="bottom")
with head:
    st.title("Premier League Outcome Model")
    st.markdown(
        f"Forecasts every {test_label} Premier League match **before kick-off**, using only "
        f"what was known at the time. Trained on {train_label}, tuned on {validation_label}, "
        f"scored once on **{test_label}**."
    )
with picker:
    set_name = st.segmented_control(
        "Features",
        list(FEATURE_SETS),
        default=list(FEATURE_SETS)[0],
        required=True,
        help="Form & Elo uses only public match history. The second set adds the "
        "bookmaker's own probabilities as inputs.",
    )

spec = report["feature_sets"][set_name]
head.caption(f"{spec['algorithm']} · picked by validation log-loss on {validation_label}")
scored = score_model(set_name)
y_true = scored["Result"]
accuracy = scored["Hit"].mean()
bookie_accuracy = scored["BookieHit"].mean()
model_loss = log_loss(y_true, probs(scored, "P"), labels=sorted(CLASS_ORDER))
bookie_loss = log_loss(y_true, probs(scored, "B"), labels=sorted(CLASS_ORDER))

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Accuracy",
    f"{accuracy:.1%}",
    delta=f"{(accuracy - bookie_accuracy) * 100:+.1f} pts vs bookmaker",
    border=True,
    height="stretch",
)
k2.metric(
    "Log-loss",
    f"{model_loss:.3f}",
    delta=f"{model_loss - bookie_loss:+.3f} vs bookmaker",
    delta_color="inverse",
    help="How good the probabilities are, not just the picks. Lower is better; "
    "always saying ⅓-⅓-⅓ scores 1.099.",
    border=True,
    height="stretch",
)
k3.metric(
    "Called correctly", f"{int(scored['Hit'].sum())} / {len(scored)}", border=True, height="stretch"
)
k4.metric("Training matches", f"{len(fit_df):,}", border=True, height="stretch")

# --- Benchmarks --------------------------------------------------------------

st.subheader("How it compares")
benchmarks = pd.DataFrame(
    [
        ("Random guess", 1 / 3, "Pick one of the three outcomes at random", False),
        ("Always back the home side", (y_true == "H").mean(), "Predict a home win every time", False),
        ("Bookmaker favourite", bookie_accuracy, "Pick the outcome with the shortest Bet365 odds", False),
        (f"Model ({set_name})", accuracy, "This model, on the unseen season", True),
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

# --- Model behaviour ---------------------------------------------------------

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
    st.caption("Top ten features by how much shuffling each one hurts the test-season log-loss.")
    st.altair_chart(importance_chart(feature_importance(set_name), palette), width="stretch")

# --- Match explorer ----------------------------------------------------------

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
            f"**{match['Home']} {match['HomeGoals']}–{match['AwayGoals']} {match['Away']}**  \n"
            f"{match['Date']:%A %d %B %Y} · result: {LABEL_MAP[match['Result']]}"
        )
        st.altair_chart(match_chart(match, MODEL_LABEL, palette), width="stretch")
        verdict = "✓ model right" if match["Hit"] else "✗ model wrong"
        bookie_verdict = "✓ bookmaker right" if match["BookieHit"] else "✗ bookmaker wrong"
        st.caption(f"{verdict} · {bookie_verdict}")
    else:
        st.info("No matches for this filter.")

slug = "form_elo_odds" if "odds" in set_name else "form_elo"
st.download_button(
    "Download predictions (CSV)",
    table.to_csv(index=False).encode("utf-8"),
    file_name=f"predictions_{split['test'][0]}_{slug}.csv",
    mime="text/csv",
)

# --- Method ------------------------------------------------------------------

st.divider()
method, metrics_col = st.columns([3, 2])
with method:
    st.markdown("#### Method and limitations")
    st.markdown(
        f"""
- **Task:** three-way classification (home win / draw / away win), scored on probabilities
  (log-loss) as well as picks (accuracy).
- **Features ({len(FEATURE_SETS[set_name])}):** Elo ratings updated after every result and carried
  across seasons; each side's points, goals and shots on target over its last {FORM_WINDOW}
  matches; rest days. The second feature set adds Bet365's margin-free implied probabilities.
  Every value is computed from matches played *before* the one being predicted.
- **Validation:** strictly chronological. {season_label(split['burn_in'][0])} warms up the
  ratings, {train_label} trains ({report['rows']['train']:,} matches), {validation_label}
  picks the algorithm and hyper-parameters, and {test_label} is scored once at the end.
- **Limitations:** no line-ups, injuries or expected-goals data, which is where bookmakers
  get their edge. Draws are almost never the top pick.

Data: [Football-Data.co.uk](https://www.football-data.co.uk/englandm.php) ·
Code: [GitHub](https://github.com/alemoscardo/ml-football-predictions)
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
        "the top row is refitted on both and used above."
    )
