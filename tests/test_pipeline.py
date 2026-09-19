import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rental.data import clean, normalize_text  # noqa: E402
from rental.features import add_features  # noqa: E402
from rental.pipeline import candidate_models, make_pipeline  # noqa: E402
from rental.predict import estimate  # noqa: E402


def synthetic(n=600, seed=0):
    """Small fake frame in the raw source format (no network needed)."""
    rng = np.random.default_rng(seed)
    nbh = rng.choice(["الف", "ب", "پ", "ت"], n)
    base = pd.Series(nbh).map({"الف": 8, "ب": 5, "پ": 3, "ت": 2}).values
    area = rng.integers(40, 250, n)
    deposit = (base * area * 1e6 * rng.uniform(0.8, 1.2, n)).round(-6)
    rent = np.where(rng.random(n) < 0.4, 0, rng.integers(1, 6, n) * 1e6)
    return pd.DataFrame({
        "total_value": deposit, "neighborhood": [f" {x} " for x in nbh], "area": area,
        "year": rng.integers(1375, 1400, n), "deposit": deposit, "rent": rent,
        "elavator": rng.integers(0, 2, n), "parking": rng.integers(0, 2, n), "warehouse": 1,
    })


def test_normalize_text_unifies_arabic_variants():
    s = normalize_text(pd.Series(["  كوي   فردوس ", "علي"]))
    assert s.tolist() == ["کوی فردوس", "علی"]


def test_clean_rules_remove_bad_rows_and_log_them():
    df = synthetic(300)
    bad = df.iloc[:4].copy()
    bad.loc[bad.index[0], ["deposit", "rent"]] = 0            # no price
    bad.loc[bad.index[1], "area"] = 300000                     # typo area
    bad.loc[bad.index[2], "rent"] = 100000                     # placeholder rent
    bad.loc[bad.index[3], "rent"] = 5e10                       # typo rent
    df = pd.concat([df, bad], ignore_index=True)
    out, rep = clean(df, return_report=True)
    assert "elevator" in out.columns and "elavator" not in out.columns
    assert out["neighborhood"].str.startswith(" ").sum() == 0
    assert (out["area"].between(20, 600)).all()
    assert rep["removed"].sum() >= 4
    assert rep["remaining"].iloc[-1] == len(out)
    assert {"value_m", "value_per_m2", "full_deposit"} <= set(out.columns)


def test_value_uses_deposit_plus_converted_rent():
    df = synthetic(200)
    out = clean(df, rent_to_deposit=30)
    row = out.iloc[0]
    assert row["value_m"] == pytest.approx(row["deposit"] / 1e6 + 30 * row["rent"] / 1e6)


def test_features():
    out = add_features(clean(synthetic(100)))
    assert (out["age"] >= 0).all()
    assert np.allclose(out["log_area"], np.log(out["area"]))


def test_pipeline_learns_signal_and_handles_unseen_neighborhood():
    df = clean(synthetic(800))
    X, y = df[["neighborhood", "area", "year", "elevator", "parking"]], np.log1p(df["value_m"])
    pipe = make_pipeline(candidate_models()["ridge"]).fit(X, y)
    pred = pipe.predict(X)
    assert np.corrcoef(pred, y)[0, 1] > 0.75
    X2 = X.head(3).copy()
    X2["neighborhood"] = "ناشناخته"
    assert np.isfinite(pipe.predict(X2)).all()


def test_estimate_interval_is_ordered():
    df = clean(synthetic(800))
    feats = ["neighborhood", "area", "year", "elevator", "parking"]
    pipe = make_pipeline(candidate_models()["ridge"]).fit(df[feats], np.log1p(df["value_m"]))
    art = {"pipeline": pipe, "resid_q10": -0.2, "resid_q90": 0.2, "features": feats}
    res = estimate(df[feats].head(10), art)
    assert (res["low_million_toman"] <= res["estimate_million_toman"]).all()
    assert (res["estimate_million_toman"] <= res["high_million_toman"]).all()
