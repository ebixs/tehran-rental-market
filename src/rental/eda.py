"""Exploratory analysis figures + key statistics.  Usage: python -m rental.eda"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import FIGURES_DIR, MIN_LISTINGS_FOR_RANKING, REF_YEAR, REPORTS_DIR
from .data import clean, load_raw
from .features import add_features
from .plotting import fa, setup


def neighborhood_stats(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("neighborhood").agg(
        n=("value_m", "size"),
        median_value_m=("value_m", "median"),
        median_value_per_m2=("value_per_m2", "median"),
        median_area=("area", "median"),
        full_deposit_share=("full_deposit", "mean"),
    )
    return g[g["n"] >= MIN_LISTINGS_FOR_RANKING].sort_values("median_value_per_m2", ascending=False)


def within_neighborhood_premium(df: pd.DataFrame, col: str) -> float:
    """Median % premium of listings with `col`==1 vs. the neighborhood median (controls for location)."""
    counts = df["neighborhood"].value_counts()
    d = df[df["neighborhood"].isin(counts[counts >= MIN_LISTINGS_FOR_RANKING].index)].copy()
    d["rel"] = d["value_per_m2"] / d.groupby("neighborhood")["value_per_m2"].transform("median")
    return float((d.loc[d[col] == 1, "rel"].median() / d.loc[d[col] == 0, "rel"].median() - 1) * 100)


def run():
    setup()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    df, rep = clean(raw, return_report=True)
    df = add_features(df)

    # 1. cleaning waterfall
    r = rep.iloc[1:]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    sns.barplot(x=r["removed"], y=r["rule"], color="#c44e52", ax=ax)
    for i, v in enumerate(r["removed"]):
        ax.text(v + 10, i, f"{v:,}", va="center", fontsize=9)
    ax.set_title(f"Rows removed by each cleaning rule  ({len(raw):,} -> {len(df):,})")
    ax.set_xlabel("rows removed"); ax.set_ylabel("")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "01_cleaning_rules.png", dpi=150); plt.close(fig)

    # 2. distributions
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    sns.histplot(df["value_m"], bins=60, log_scale=True, ax=axes[0], color="#4c72b0")
    axes[0].set_title("Deposit-equivalent value (million Toman, log scale)")
    sns.histplot(df["area"], bins=50, ax=axes[1], color="#55a868"); axes[1].set_title("Area (m2)")
    sns.histplot(df["age"], bins=30, ax=axes[2], color="#8172b2"); axes[2].set_title("Building age (years)")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "02_distributions.png", dpi=150); plt.close(fig)

    # 3. neighborhood ranking
    ns = neighborhood_stats(df)
    top, bottom = ns.head(15), ns.tail(15)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, d, title, color in [(axes[0], top, "15 most expensive neighborhoods", "#c44e52"),
                                (axes[1], bottom, "15 most affordable neighborhoods", "#55a868")]:
        ax.barh([fa(i) for i in d.index][::-1], d["median_value_per_m2"][::-1], color=color)
        ax.set_title(f"{title}\n(median value per m2, million Toman; n>={MIN_LISTINGS_FOR_RANKING})")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "03_neighborhood_ranking.png", dpi=150); plt.close(fig)

    # 4. deposit vs rent trade-off
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    mixed = df[df["rent"] > 0]
    axes[0].scatter(mixed["deposit"] / 1e6, mixed["rent"] / 1e6, s=6, alpha=0.25, color="#4c72b0")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("deposit (million Toman)"); axes[0].set_ylabel("monthly rent (million Toman)")
    axes[0].set_title("Deposit vs. rent trade-off (listings with rent > 0)")
    big = ns.sort_values("n", ascending=False).head(15).sort_values("full_deposit_share")
    axes[1].barh([fa(i) for i in big.index], big["full_deposit_share"] * 100, color="#8172b2")
    axes[1].set_xlabel("% of listings that are full-deposit (no monthly rent)")
    axes[1].set_title("Full-deposit (رهن کامل) share, 15 busiest neighborhoods".replace("رهن کامل", fa("رهن کامل")))
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "04_deposit_vs_rent.png", dpi=150); plt.close(fig)

    # 5. amenities & age (raw vs. location-adjusted)
    prem = {c: within_neighborhood_premium(df, c) for c in ["elevator", "parking"]}
    raw_prem = {c: (df.loc[df[c] == 1, "value_per_m2"].median() / df.loc[df[c] == 0, "value_per_m2"].median() - 1) * 100
                for c in ["elevator", "parking"]}
    df["age_group"] = pd.cut(df["age"], [-1, 2, 7, 15, 25, 60], labels=["0-2", "3-7", "8-15", "16-25", "25+"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(2); w = 0.35
    axes[0].bar(x - w / 2, [raw_prem[c] for c in ["elevator", "parking"]], w, label="raw", color="#c9c9c9")
    axes[0].bar(x + w / 2, [prem[c] for c in ["elevator", "parking"]], w, label="within neighborhood", color="#4c72b0")
    axes[0].set_xticks(x); axes[0].set_xticklabels(["Elevator", "Parking"]); axes[0].legend()
    axes[0].set_ylabel("% premium in value per m2"); axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_title("Amenity premium: raw vs. controlling for neighborhood")
    sns.boxplot(data=df, x="age_group", y="value_per_m2", showfliers=False, color="#8172b2", ax=axes[1])
    axes[1].set_title("Value per m2 by building age (years)"); axes[1].set_xlabel("age"); axes[1].set_ylabel("million Toman / m2")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "05_amenities_and_age.png", dpi=150); plt.close(fig)

    stats = {
        "rows_raw": int(len(raw)), "rows_clean": int(len(df)),
        "n_neighborhoods": int(df["neighborhood"].nunique()),
        "median_value_million_toman": round(float(df["value_m"].median()), 1),
        "median_area": float(df["area"].median()),
        "full_deposit_share": round(float(df["full_deposit"].mean()), 3),
        "premium_raw_pct": {k: round(v, 1) for k, v in raw_prem.items()},
        "premium_within_neighborhood_pct": {k: round(v, 1) for k, v in prem.items()},
        "top5": list(ns.head(5).index), "bottom5": list(ns.tail(5).index),
        "top5_median_per_m2": [round(v, 1) for v in ns.head(5)["median_value_per_m2"]],
        "bottom5_median_per_m2": [round(v, 1) for v in ns.tail(5)["median_value_per_m2"]],
        "top_to_bottom_ratio": round(float(ns["median_value_per_m2"].iloc[0] / ns["median_value_per_m2"].iloc[-1]), 1),
    }
    (REPORTS_DIR / "eda_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    ns.round(3).to_csv(REPORTS_DIR / "neighborhood_ranking.csv", encoding="utf-8-sig")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run()
