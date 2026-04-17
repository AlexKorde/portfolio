"""
CFB Game Outcome Prediction Dashboard
Run: streamlit run dashboard.py
"""

import glob
import os
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from scipy import stats
from scipy.special import expit as sigmoid
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
with open('/Users/alexkorde/balluptop.txt', 'r') as f: API_KEY = f.read().strip()
BASE_URL = "https://api.collegefootballdata.com"
NIL_YEAR = 2021
SEASONS  = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]
HEADERS  = {"Authorization": f"Bearer {API_KEY}"}

STAT_MAP = {
    "turnovers":            "turnovers",
    "thirdDownConversions": "third_down_conv",
    "rushingYards":         "rush_yards",
    "netPassingYards":      "pass_yards",
    "totalYards":           "total_yards",
    "sacks":                "sacks",
}
STAT_LABELS = {
    "turnovers":       "Turnovers",
    "third_down_conv": "3rd Down Conv",
    "rush_yards":      "Rush Yards",
    "pass_yards":      "Pass Yards",
    "total_yards":     "Total Yards",
    "sacks":           "Sacks",
}
LOWER_IS_BETTER = {"turnovers"}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="CFB Analytics", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@300;400;600&display=swap');
  html, body, [class*="css"] { font-family: 'Source Serif 4', Georgia, serif; }
  h1, h2, h3, h4             { font-family: 'Source Serif 4', Georgia, serif; font-weight: 600; }
  .stMetric label            { font-size: 12px; }
  .block-container           { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(color="#222", family="Georgia, serif"),
    margin=dict(t=40, b=40, l=40, r=40),
)


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _latest_cache():
    files = sorted(glob.glob("cfb_cache_*.csv"))
    return files[-1] if files else None

def _next_cache():
    files = sorted(glob.glob("cfb_cache_*.csv"))
    n = int(files[-1].split("_")[-1].replace(".csv", "")) + 1 if files else 0
    return f"cfb_cache_{n}.csv"


# ── Data fetching ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Fetching season data...")
def load_data() -> pd.DataFrame:
    latest = _latest_cache()
    if latest:
        return pd.read_csv(latest)

    if not API_KEY:
        st.error("CFB_API_KEY not set. Run: export CFB_API_KEY='your_key'")
        st.stop()

    all_rows = []
    for season in SEASONS:
        gr = requests.get(f"{BASE_URL}/games",        headers=HEADERS, params={"year": season, "seasonType": "regular"})
        sr = requests.get(f"{BASE_URL}/stats/season", headers=HEADERS, params={"year": season})
        gr.raise_for_status()
        sr.raise_for_status()

        games = pd.DataFrame(gr.json()).rename(columns={
            "homeTeam": "home_team", "awayTeam": "away_team",
            "homePoints": "home_points", "awayPoints": "away_points",
        })
        stats_df = pd.DataFrame(sr.json())

        pivot = stats_df.pivot_table(index="team", columns="statName", values="statValue", aggfunc="first").reset_index()
        pivot = pivot.rename(columns={k: v for k, v in STAT_MAP.items() if k in pivot.columns})
        keep  = ["team"] + [v for v in STAT_MAP.values() if v in pivot.columns]
        pivot = pivot[keep]

        for _, g in games.iterrows():
            home, away   = g.get("home_team"), g.get("away_team")
            h_pts, a_pts = g.get("home_points"), g.get("away_points")
            if pd.isna(h_pts) or pd.isna(a_pts):
                continue
            hs  = pivot[pivot["team"] == home]
            as_ = pivot[pivot["team"] == away]
            if hs.empty or as_.empty:
                continue
            h, a = hs.iloc[0], as_.iloc[0]
            row = {
                "season":    season,
                "home_team": home,
                "away_team": away,
                "home_win":  int(h_pts > a_pts),
                "nil_era":   int(season >= NIL_YEAR),
            }
            for col in STAT_MAP.values():
                if col in pivot.columns:
                    row[f"diff_{col}"] = float(h.get(col, 0)) - float(a.get(col, 0))
                    row[f"nil_{col}"]  = row["nil_era"] * row[f"diff_{col}"]
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.to_csv(_next_cache(), index=False)
    return df


# ── Model ─────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Training model...")
def train_model(cache_key: str):
    df        = load_data()
    feat_cols = [c for c in df.columns if c.startswith("diff_") or c.startswith("nil_")]
    feat_cols.append("nil_era")

    X = df[feat_cols].fillna(0).values
    y = df["home_win"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    acc      = accuracy_score(y_test, model.predict(X_test))
    baseline = float(y_test.mean())
    coef_df  = pd.DataFrame({"feature": feat_cols, "coef": model.coef_[0]}).sort_values("coef", ascending=False)

    return model, scaler, feat_cols, coef_df, acc, baseline, df


# ── Chow Test ─────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Running Chow Test...")
def chow_test(cache_key: str):
    df         = load_data()
    base_feats = [c for c in df.columns if c.startswith("diff_")]
    pre        = df[df["nil_era"] == 0]
    post       = df[df["nil_era"] == 1]

    def rss_and_acc(train_df, test_df):
        X_tr = train_df[base_feats].fillna(0).values
        y_tr = train_df["home_win"].values
        X_te = test_df[base_feats].fillna(0).values
        y_te = test_df["home_win"].values
        sc   = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)
        m    = LogisticRegression(max_iter=1000)
        m.fit(X_tr, y_tr)
        probs     = m.predict_proba(X_te)[:, 1]
        rss       = float(((y_te - probs) ** 2).sum())
        acc       = accuracy_score(y_te, m.predict(X_te))
        return rss, acc, len(y_te)

    rss_pre,    acc_pre,   n_pre  = rss_and_acc(pre,  pre)
    rss_post,   acc_post,  n_post = rss_and_acc(post, post)
    rss_pooled, _,         _      = rss_and_acc(df,   df)
    _,          acc_cross, _      = rss_and_acc(pre,  post)

    k           = len(base_feats) + 1
    n           = n_pre + n_post
    numerator   = (rss_pooled - (rss_pre + rss_post)) / k
    denominator = (rss_pre + rss_post) / max(n - 2 * k, 1)
    f_stat      = numerator / denominator
    p_value     = float(1 - stats.f.cdf(f_stat, k, n - 2 * k))

    acc_results = {
        "pre → pre":   round(acc_pre,   4),
        "pre → post":  round(acc_cross, 4),
        "post → post": round(acc_post,  4),
    }
    return f_stat, p_value, acc_results


# ── Ratings ───────────────────────────────────────────────────────────────────

def _build_raw(df: pd.DataFrame) -> pd.DataFrame:
    stat_cols = list(STAT_MAP.values())
    rows = []
    for season in SEASONS:
        for _, g in df[df["season"] == season].iterrows():
            row = {"team": g["home_team"], "season": season, "nil_era": g["nil_era"]}
            for col in stat_cols:
                if f"diff_{col}" in g.index:
                    row[col] = g[f"diff_{col}"]
            rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Computing ratings...")
def compute_ratings(cache_key: str):
    df        = load_data()
    raw       = _build_raw(df)
    stat_cols = list(STAT_MAP.values())

    def build_ratings(raw_df, within_season: bool) -> pd.DataFrame:
        out = raw_df.copy()
        for col in stat_cols:
            if col not in out.columns:
                continue
            if within_season:
                for season in SEASONS:
                    mask = out["season"] == season
                    mu, sigma = out.loc[mask, col].mean(), out.loc[mask, col].std()
                    out.loc[mask, f"z_{col}"] = ((out.loc[mask, col] - mu) / sigma) if sigma > 0 else 0.0
            else:
                mu, sigma = out[col].mean(), out[col].std()
                out[f"z_{col}"] = ((out[col] - mu) / sigma) if sigma > 0 else 0.0

        z_cols = [f"z_{c}" for c in stat_cols if f"z_{c}" in out.columns]
        out["rating"] = out[z_cols].mean(axis=1)
        if "z_turnovers" in out.columns:
            out["rating"] -= out["z_turnovers"]
        mn, mx = out["rating"].min(), out["rating"].max()
        out["rating_100"] = ((out["rating"] - mn) / (mx - mn) * 100).round(1)
        return out[["team", "season", "nil_era", "rating_100"] + z_cols]

    era_ratings = build_ratings(raw, within_season=True)
    abs_ratings = build_ratings(raw, within_season=False)
    return era_ratings, abs_ratings


def matchup_sim(team_a, season_a, team_b, season_b, era_ratings, abs_ratings):
    def lookup(df, team, season):
        r = df[(df["team"] == team) & (df["season"] == season)]
        return r.iloc[0] if not r.empty else None

    ra_era = lookup(era_ratings, team_a, season_a)
    rb_era = lookup(era_ratings, team_b, season_b)
    ra_abs = lookup(abs_ratings, team_a, season_a)
    rb_abs = lookup(abs_ratings, team_b, season_b)

    if ra_era is None or rb_era is None:
        return None

    diff   = float(ra_era["rating_100"]) - float(rb_era["rating_100"])
    prob_a = float(sigmoid(diff * 0.07))

    z_cols    = [c for c in era_ratings.columns if c.startswith("z_")]
    stat_comp = {z[2:]: {"a": round(float(ra_era[z]), 2), "b": round(float(rb_era[z]), 2)} for z in z_cols}

    return {
        "era_rating_a": float(ra_era["rating_100"]),
        "era_rating_b": float(rb_era["rating_100"]),
        "abs_rating_a": float(ra_abs["rating_100"]) if ra_abs is not None else None,
        "abs_rating_b": float(rb_abs["rating_100"]) if rb_abs is not None else None,
        "prob_a":       round(prob_a * 100, 1),
        "prob_b":       round((1 - prob_a) * 100, 1),
        "stat_comp":    stat_comp,
    }


# ── Bootstrap ─────────────────────────────────────────────────────────────────
with st.spinner("Loading..."):
    model, scaler, feat_cols, coef_df, acc, baseline, df = train_model("v4")
    era_ratings, abs_ratings = compute_ratings("v4")
    f_stat, p_value, chow_acc = chow_test("v4")

teams_by_season = {s: sorted(df[df["season"] == s]["home_team"].unique().tolist()) for s in SEASONS}
all_teams       = sorted(era_ratings["team"].unique().tolist())

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("CFB Analytics")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["Overview", "Matchup Simulator", "Team Ratings", "Structural Break", "Data Export"])
st.sidebar.markdown("---")
st.sidebar.caption(f"Accuracy: {acc:.1%} | Baseline: {baseline:.1%} | Lift: {acc - baseline:+.1%}")
st.sidebar.caption("Source: College Football Data API")
st.sidebar.caption("NIL / Transfer Portal break: 2021")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("CFB Game Outcome Prediction")
    st.markdown(
        "Logistic regression model trained on regular season games 2016–2024 "
        "(2020 excluded). Predicts home win from season-level team stat differentials. "
        "NIL and Transfer Portal era begins 2021."
    )
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model Accuracy",   f"{acc:.1%}")
    c2.metric("Baseline (home%)", f"{baseline:.1%}")
    c3.metric("Lift",             f"{acc - baseline:+.1%}")
    c4.metric("Total Games",      f"{len(df):,}")

    st.markdown("### Games per season")
    gps = df.groupby("season").size().reset_index(name="games")
    gps["Era"] = gps["season"].apply(lambda x: "Post-NIL" if x >= NIL_YEAR else "Pre-NIL")
    fig = px.bar(gps, x="season", y="games", color="Era",
                 color_discrete_map={"Pre-NIL": "#4472C4", "Post-NIL": "#ED7D31"},
                 labels={"season": "Season", "games": "Games"})
    fig.update_layout(**PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Home win rate by season")
    hwr = df.groupby("season")["home_win"].mean().reset_index()
    fig2 = px.line(hwr, x="season", y="home_win", markers=True,
                   labels={"home_win": "Home Win %", "season": "Season"},
                   color_discrete_sequence=["#222"])
    fig2.add_hline(y=baseline, line_dash="dash", line_color="#999",
                   annotation_text=f"Baseline {baseline:.1%}", annotation_position="bottom right")
    fig2.update_layout(**PLOT_LAYOUT)
    fig2.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Model coefficients")
    st.caption(
        "Positive = favours home win. diff_* = base effect. "
        "nil_* = how that effect shifted post-NIL (interaction term)."
    )
    diff_coefs = coef_df[coef_df["feature"].str.startswith("diff_")].copy()
    nil_coefs  = coef_df[coef_df["feature"].str.startswith("nil_")].copy()
    diff_coefs["stat"] = diff_coefs["feature"].str.replace("diff_", "", regex=False)
    nil_coefs["stat"]  = nil_coefs["feature"].str.replace("nil_",  "", regex=False)
    merged = diff_coefs[["stat", "coef"]].merge(nil_coefs[["stat", "coef"]], on="stat", suffixes=("_base", "_shift"))
    merged["label"] = merged["stat"].map(STAT_LABELS).fillna(merged["stat"])

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(name="Base (pre-NIL)", x=merged["label"], y=merged["coef_base"],  marker_color="#4472C4"))
    fig3.add_trace(go.Bar(name="NIL-era shift",  x=merged["label"], y=merged["coef_shift"], marker_color="#ED7D31"))
    fig3.update_layout(**PLOT_LAYOUT, barmode="group", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig3, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Matchup Simulator
# ════════════════════════════════════════════════════════════════════════════
elif page == "Matchup Simulator":
    st.title("Era-Relative Dominance Comparison")
    st.markdown(
        "Compares how statistically dominant each team was relative to their own season's competition. "
        "This is **not** a score prediction and does not account for strength of schedule. "
        "Era-normalised and absolute (cross-era) ratings are shown side by side."
    )
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Team A")
        season_a = st.selectbox("Season", SEASONS, index=len(SEASONS) - 2, key="sa")
        team_a   = st.selectbox("Team",   teams_by_season[season_a], key="ta")
        st.caption("Post-NIL era" if season_a >= NIL_YEAR else "Pre-NIL era")

    with col2:
        st.subheader("Team B")
        season_b = st.selectbox("Season", SEASONS, index=0, key="sb")
        team_b   = st.selectbox("Team",   teams_by_season[season_b], key="tb")
        st.caption("Post-NIL era" if season_b >= NIL_YEAR else "Pre-NIL era")

    if st.button("Compare", use_container_width=True):
        result = matchup_sim(team_a, season_a, team_b, season_b, era_ratings, abs_ratings)
        if result is None:
            st.error("Could not find stats for one or both teams.")
        else:
            st.markdown("---")
            r1, r2, r3 = st.columns([5, 1, 5])
            with r1:
                st.markdown(f"**{team_a} ({season_a})**")
                st.markdown(f"### {result['prob_a']}%")
                st.caption("Relative dominance advantage")
                st.metric("Era-normalised rating", f"{result['era_rating_a']:.1f} / 100")
                if result["abs_rating_a"] is not None:
                    st.metric("Absolute (cross-era) rating", f"{result['abs_rating_a']:.1f} / 100")
            with r2:
                st.markdown("<br><br><p style='text-align:center;color:#aaa'>vs</p>", unsafe_allow_html=True)
            with r3:
                st.markdown(f"**{team_b} ({season_b})**")
                st.markdown(f"### {result['prob_b']}%")
                st.caption("Relative dominance advantage")
                st.metric("Era-normalised rating", f"{result['era_rating_b']:.1f} / 100")
                if result["abs_rating_b"] is not None:
                    st.metric("Absolute (cross-era) rating", f"{result['abs_rating_b']:.1f} / 100")

            st.markdown("---")
            st.markdown("**Stat comparison — Z-scores, era-normalised**")
            st.caption("Standard deviations above/below that season's average. Turnovers: lower is better.")

            stat_rows = []
            for stat, vals in result["stat_comp"].items():
                label  = STAT_LABELS.get(stat, stat)
                winner = team_a if (vals["a"] < vals["b"] if stat in LOWER_IS_BETTER else vals["a"] > vals["b"]) else team_b
                stat_rows.append({"Stat": label, team_a: vals["a"], team_b: vals["b"], "Edge": winner})
            st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)

            cats   = [STAT_LABELS.get(s, s) for s in result["stat_comp"]]
            vals_a = [v["a"] for v in result["stat_comp"].values()]
            vals_b = [v["b"] for v in result["stat_comp"].values()]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=vals_a + [vals_a[0]], theta=cats + [cats[0]],
                fill="toself", name=f"{team_a} {season_a}", line_color="#4472C4"))
            fig.add_trace(go.Scatterpolar(r=vals_b + [vals_b[0]], theta=cats + [cats[0]],
                fill="toself", name=f"{team_b} {season_b}", line_color="#ED7D31", opacity=0.7))
            fig.update_layout(polar=dict(bgcolor="#f9f9f9", radialaxis=dict(visible=True, color="#aaa")),
                               showlegend=True, **PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Team Ratings
# ════════════════════════════════════════════════════════════════════════════
elif page == "Team Ratings":
    st.title("Team Ratings by Season")

    rating_mode = st.radio(
        "Rating type", ["Era-normalised", "Absolute (cross-era)"], horizontal=True,
        help="Era-normalised: scored relative to that season. Absolute: scored across all seasons on one scale.",
    )
    ratings_df = era_ratings if rating_mode == "Era-normalised" else abs_ratings

    if rating_mode == "Era-normalised":
        st.caption("Each team scored relative to the average team in their own season. High score = dominant for that era.")
    else:
        st.caption("All teams scored on a single shared scale. Penalises weak-schedule inflation (e.g. FCS teams will drop).")

    sel_season = st.selectbox("Season", SEASONS, index=len(SEASONS) - 1)
    top_n      = st.slider("Show top N teams", 10, 50, 25)

    season_ratings = ratings_df[ratings_df["season"] == sel_season].nlargest(top_n, "rating_100")
    fig = px.bar(season_ratings, x="rating_100", y="team", orientation="h",
                 labels={"rating_100": "Rating (0–100)", "team": ""},
                 color_discrete_sequence=["#4472C4"])
    fig.update_layout(**PLOT_LAYOUT, yaxis=dict(autorange="reversed"), height=max(400, top_n * 22))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Team history")
    team_lookup = st.selectbox("Team", all_teams)
    team_hist   = ratings_df[ratings_df["team"] == team_lookup][["season", "rating_100"]].sort_values("season")

    if not team_hist.empty:
        fig2 = px.line(team_hist, x="season", y="rating_100", markers=True,
                       labels={"rating_100": "Rating", "season": "Season"},
                       color_discrete_sequence=["#222"])
        fig2.add_vline(x=NIL_YEAR - 0.5, line_dash="dash", line_color="#aaa",
                       annotation_text="NIL begins", annotation_position="top right")
        fig2.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Structural Break
# ════════════════════════════════════════════════════════════════════════════
elif page == "Structural Break":
    st.title("Structural Break Analysis")
    st.markdown(
        "Did NIL and the Transfer Portal change how well traditional statistics predict wins? "
        "A structural break means the *regression coefficients* shifted — not just the outcomes. "
        "Tested with a proper Chow Test (F-statistic) and an accuracy cross-test."
    )
    st.markdown("---")

    st.markdown("### Chow Test result")
    c1, c2 = st.columns(2)
    c1.metric("F-statistic", f"{f_stat:.3f}")
    c2.metric("p-value",     f"{p_value:.4f}")

    if p_value < 0.01:
        st.success(f"Strong evidence of a structural break (p = {p_value:.4f}). The regression structure changed significantly post-NIL.")
    elif p_value < 0.05:
        st.warning(f"Moderate evidence of a structural break (p = {p_value:.4f}).")
    else:
        st.info(f"No significant structural break detected at the 5% level (p = {p_value:.4f}).")

    st.caption(
        "Compares residual sum of squares from a pooled model vs. separate pre/post models. "
        "A significant F-statistic means the coefficients are statistically different across eras."
    )

    st.markdown("---")
    st.markdown("### Accuracy cross-test")
    st.caption(
        "pre → pre: trained and tested on pre-NIL data. "
        "pre → post: same model applied to post-NIL games — drops if structure changed. "
        "post → post: trained and tested on post-NIL data."
    )

    chow_df = pd.DataFrame({"Scenario": list(chow_acc.keys()), "Accuracy": list(chow_acc.values())})
    fig = px.bar(chow_df, x="Scenario", y="Accuracy",
                 color="Scenario",
                 color_discrete_sequence=["#4472C4", "#ED7D31", "#70AD47"],
                 labels={"Accuracy": "Accuracy"})
    fig.add_hline(y=baseline, line_dash="dash", line_color="#aaa",
                  annotation_text=f"Overall baseline {baseline:.1%}", annotation_position="bottom right")
    fig.update_layout(**PLOT_LAYOUT, showlegend=False)
    fig.update_yaxes(tickformat=".0%", range=[0.45, 0.75])
    st.plotly_chart(fig, use_container_width=True)

    drop = chow_acc.get("pre → pre", 0) - chow_acc.get("pre → post", 0)
    st.markdown(f"Accuracy drop (pre→pre vs pre→post): **{drop:+.1%}**")

    st.markdown("---")
    st.markdown("### Coefficient shifts")
    st.caption("Base = pre-NIL predictive weight. NIL-era shift = change in that weight post-2021.")

    diff_coefs = coef_df[coef_df["feature"].str.startswith("diff_")].copy()
    nil_coefs  = coef_df[coef_df["feature"].str.startswith("nil_")].copy()
    diff_coefs["stat"] = diff_coefs["feature"].str.replace("diff_", "", regex=False)
    nil_coefs["stat"]  = nil_coefs["feature"].str.replace("nil_",  "", regex=False)
    merged = diff_coefs[["stat", "coef"]].merge(nil_coefs[["stat", "coef"]], on="stat", suffixes=("_base", "_shift"))
    merged["label"] = merged["stat"].map(STAT_LABELS).fillna(merged["stat"])

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name="Base (pre-NIL)", x=merged["label"], y=merged["coef_base"],  marker_color="#4472C4"))
    fig2.add_trace(go.Bar(name="NIL-era shift",  x=merged["label"], y=merged["coef_shift"], marker_color="#ED7D31"))
    fig2.update_layout(**PLOT_LAYOUT, barmode="group", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Data Export
# ════════════════════════════════════════════════════════════════════════════
elif page == "Data Export":
    st.title("Data Export")
    st.markdown("Browse and download the raw game dataset and team ratings.")

    tab1, tab2 = st.tabs(["Game Data", "Team Ratings"])

    with tab1:
        st.markdown(f"{len(df):,} games across {df['season'].nunique()} seasons.")
        fc1, fc2, fc3 = st.columns(3)
        sel_seasons = fc1.multiselect("Season", SEASONS, default=SEASONS)
        sel_era     = fc2.selectbox("Era", ["All", "Pre-NIL", "Post-NIL"])
        team_filter = fc3.text_input("Team filter")

        view = df[df["season"].isin(sel_seasons)].copy()
        if sel_era == "Pre-NIL":
            view = view[view["nil_era"] == 0]
        elif sel_era == "Post-NIL":
            view = view[view["nil_era"] == 1]
        if team_filter:
            mask = (view["home_team"].str.contains(team_filter, case=False, na=False) |
                    view["away_team"].str.contains(team_filter, case=False, na=False))
            view = view[mask]

        display_cols = ["season", "home_team", "away_team", "home_win", "nil_era"] + \
                       [c for c in view.columns if c.startswith("diff_")]
        st.dataframe(view[display_cols].reset_index(drop=True), use_container_width=True, height=400)
        st.download_button("Download game data (CSV)", data=view[display_cols].to_csv(index=False),
                           file_name="cfb_games.csv", mime="text/csv")

    with tab2:
        rc1, rc2, rc3 = st.columns(3)
        rat_mode   = rc1.selectbox("Rating type", ["Era-normalised", "Absolute"])
        rat_season = rc2.multiselect("Season", SEASONS, default=SEASONS, key="rs")
        rat_team   = rc3.text_input("Team filter", key="rt")

        rat_df   = era_ratings if rat_mode == "Era-normalised" else abs_ratings
        rat_view = rat_df[rat_df["season"].isin(rat_season)].copy()
        if rat_team:
            rat_view = rat_view[rat_view["team"].str.contains(rat_team, case=False, na=False)]

        rat_view = rat_view.sort_values(["season", "rating_100"], ascending=[True, False]).reset_index(drop=True)
        st.dataframe(rat_view, use_container_width=True, height=400)
        st.download_button("Download team ratings (CSV)", data=rat_view.to_csv(index=False),
                           file_name="cfb_ratings.csv", mime="text/csv")
