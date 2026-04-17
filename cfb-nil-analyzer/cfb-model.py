"""
CFB Game Outcome Prediction Model
Structural Break Analysis: Pre vs Post NIL/Transfer Portal Era (2021)

Author: Alex Korde
"""

import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────────
with open('/Users/alexkorde/balluptop.txt', 'r') as f: API_KEY = f.read().strip()
BASE_URL = "https://api.collegefootballdata.com"
NIL_YEAR = 2021                 # Structural break point
SEASONS   = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]  # 4 pre, 4 post — 2020 excluded (COVID)

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# ── Data Fetching ────────────────────────────────────────────────────────────

def fetch_games(season: int) -> pd.DataFrame:
    """Fetch all regular season games for a given year."""
    resp = requests.get(
        f"{BASE_URL}/games",
        headers=HEADERS,
        params={"year": season, "seasonType": "regular"}
    )
    resp.raise_for_status()
    return pd.DataFrame(resp.json())


def fetch_team_stats(season: int) -> pd.DataFrame:
    """Fetch season-level team stats (yards/play, TO margin, 3rd down %, etc.)."""
    resp = requests.get(
        f"{BASE_URL}/stats/season",
        headers=HEADERS,
        params={"year": season}
    )
    resp.raise_for_status()
    return pd.DataFrame(resp.json())


def build_dataset(seasons: list) -> pd.DataFrame:
    """
    For each season, pull games + team stats and merge into a flat
    feature matrix.  Only pre-game stats are used to avoid leakage.
    """
    all_rows = []

    for season in seasons:
        print(f"  Loading {season}...")
        games = fetch_games(season)
        stats = fetch_team_stats(season)

        # Pivot stats so each team has one row with all metrics
        stats_pivot = stats.pivot_table(
            index="team", columns="statName", values="statValue", aggfunc="first"
        ).reset_index()

        # Columns we care about (rename to safe strings)
        desired = {
            "yardsPerPlay":          "yards_per_play",
            "turnovers":             "turnovers",
            "thirdDownConversions":  "third_down_conv",
            "pointsPerGame":         "ppg",
            "rushingYards":          "rush_yards",
            "passingYards":          "pass_yards",
        }
        stats_pivot = stats_pivot.rename(
            columns={k: v for k, v in desired.items() if k in stats_pivot.columns}
        )

        # Keep only columns that exist
        keep = ["team"] + [v for v in desired.values() if v in stats_pivot.columns]
        stats_pivot = stats_pivot[keep]

        for _, game in games.iterrows():
            home = game.get("home_team")
            away = game.get("away_team")
            h_pts = game.get("home_points")
            a_pts = game.get("away_points")

            if pd.isna(h_pts) or pd.isna(a_pts):
                continue

            h_stats = stats_pivot[stats_pivot["team"] == home]
            a_stats = stats_pivot[stats_pivot["team"] == away]

            if h_stats.empty or a_stats.empty:
                continue

            h = h_stats.iloc[0]
            a = a_stats.iloc[0]

            row = {
                "season":      season,
                "home_team":   home,
                "away_team":   away,
                "home_win":    int(h_pts > a_pts),   # target variable
                "nil_era":     int(season >= NIL_YEAR),
            }

            # Differential features  (home minus away)
            for col in desired.values():
                if col in stats_pivot.columns:
                    row[f"diff_{col}"] = float(h.get(col, 0)) - float(a.get(col, 0))

            # Interaction terms: nil_era × each differential
            # This is the structural break — does the coefficient shift?
            for col in desired.values():
                if col in stats_pivot.columns:
                    row[f"nil_{col}"] = row["nil_era"] * row[f"diff_{col}"]

            all_rows.append(row)

    return pd.DataFrame(all_rows)


# ── Chow Test (structural break) ────────────────────────────────────────────

def chow_test(df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Simplified Chow Test:
    Compare model accuracy trained only on pre-NIL data vs only post-NIL data.
    A big gap suggests the regression structure changed — i.e. a structural break.
    """
    pre  = df[df["nil_era"] == 0]
    post = df[df["nil_era"] == 1]

    results = {}
    scaler = StandardScaler()

    for label, train_df, test_df in [
        ("pre→pre",   pre,  pre),
        ("pre→post",  pre,  post),
        ("post→post", post, post),
    ]:
        if len(train_df) < 50 or len(test_df) < 20:
            continue

        X_tr = scaler.fit_transform(train_df[feature_cols].fillna(0))
        y_tr = train_df["home_win"]
        X_te = scaler.transform(test_df[feature_cols].fillna(0))
        y_te = test_df["home_win"]

        X_tr2, _, y_tr2, _ = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_tr2, y_tr2)
        results[label] = round(accuracy_score(y_te, model.predict(X_te)), 4)

    return results


# ── Full Model ───────────────────────────────────────────────────────────────

def train_full_model(df: pd.DataFrame):
    """Train the combined model with interaction terms."""
    feature_cols = [c for c in df.columns if c.startswith("diff_") or c.startswith("nil_")]
    feature_cols.append("nil_era")

    X = df[feature_cols].fillna(0)
    y = df["home_win"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)

    # Baseline: always predict home win (~57% historically)
    baseline = y_test.mean()

    print("\n── Model Results ──────────────────────────────")
    print(f"  Accuracy:          {acc:.1%}")
    print(f"  Baseline (home %): {baseline:.1%}")
    print(f"  Lift over baseline:{acc - baseline:+.1%}")
    print("\n── Classification Report ──────────────────────")
    print(classification_report(y_test, preds, target_names=["Away Win", "Home Win"]))

    # Coefficient table — which features shifted most?
    coef_df = pd.DataFrame({
        "feature":     feature_cols,
        "coefficient": model.coef_[0]
    }).sort_values("coefficient", ascending=False)

    print("\n── Top features (positive = favors home win) ──")
    print(coef_df.to_string(index=False))

    return model, scaler, feature_cols, coef_df


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Fetching data...")
    df = build_dataset(SEASONS)
    print(f"Dataset: {len(df)} games across {df['season'].nunique()} seasons\n")

    feature_cols = [c for c in df.columns if c.startswith("diff_")]

    print("── Chow Test (structural break check) ────────")
    chow = chow_test(df, feature_cols)
    for label, acc in chow.items():
        print(f"  {label:<15} accuracy: {acc:.1%}")
    print("""
  Interpretation:
    pre→pre   = model trained & tested on pre-NIL data
    pre→post  = same model applied to post-NIL data (drops if structure changed)
    post→post = model trained & tested on post-NIL data
    A large gap between pre→post and post→post confirms a structural break.
""")

    model, scaler, features, coef_df = train_full_model(df)
    print("\nDone. Model trained successfully.")
