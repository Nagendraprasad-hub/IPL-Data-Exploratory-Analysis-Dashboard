"""Reusable analysis utilities for IPL EDA."""

from __future__ import annotations

import pandas as pd


def load_data(matches_path: str, deliveries_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load matches and deliveries datasets."""
    matches = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)
    return matches, deliveries


def basic_cleaning(matches: pd.DataFrame, deliveries: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply basic cleaning rules and return cleaned copies."""
    m = matches.copy()
    d = deliveries.copy()

    # Standardize column names
    m.columns = [c.strip().lower() for c in m.columns]
    d.columns = [c.strip().lower() for c in d.columns]

    # Handle common nulls
    if "winner" in m.columns:
        m["winner"] = m["winner"].fillna("No Result")

    # Derived delivery-level features
    if "total_runs" in d.columns and "over" in d.columns:
        d["phase"] = pd.cut(
            d["over"],
            bins=[0, 6, 15, 20],
            labels=["Powerplay", "Middle", "Death"],
            include_lowest=True,
        )

    return m, d


def top_run_scorers(deliveries: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return top run scorers by batsman/batter."""
    d = deliveries.copy()
    batter_col = "batter" if "batter" in d.columns else "batsman"
    runs = (
        d.groupby(batter_col, as_index=False)["batsman_runs"]
        .sum()
        .sort_values("batsman_runs", ascending=False)
        .head(top_n)
    )
    runs.rename(columns={batter_col: "player", "batsman_runs": "runs"}, inplace=True)
    return runs


def toss_impact(matches: pd.DataFrame) -> pd.DataFrame:
    """Analyze toss winner vs match winner impact."""
    m = matches.copy()
    required = {"toss_winner", "winner"}
    if not required.issubset(m.columns):
        return pd.DataFrame(columns=["metric", "value"])

    toss_and_match_same = (m["toss_winner"] == m["winner"]).mean() * 100
    return pd.DataFrame(
        {
            "metric": ["Toss winner also won match (%)"],
            "value": [round(toss_and_match_same, 2)],
        }
    )
