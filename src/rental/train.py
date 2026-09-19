"""Train, compare and persist the best rent-value model.  Usage: python -m rental.train"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split

from .config import MODELS_DIR, RANDOM_STATE, REPORTS_DIR, TEST_SIZE
from .data import load_clean
from .pipeline import candidate_models, make_pipeline

FEATURES = ["neighborhood", "area", "year", "elevator", "parking"]


def to_log(v_million):
    return np.log1p(v_million)


def from_log(z):
    return np.expm1(z)


def metrics(y_log_true, y_log_pred) -> dict:
    y, p = from_log(y_log_true), from_log(y_log_pred)
    ape = np.abs(p - y) / y
    return {
        "r2_log": round(float(r2_score(y_log_true, y_log_pred)), 4),
        "mae_million_toman": round(float(mean_absolute_error(y, p)), 1),
        "median_ape_pct": round(float(np.median(ape) * 100), 1),
        "within_20pct": round(float((ape <= 0.20).mean()), 3),
        "within_30pct": round(float((ape <= 0.30).mean()), 3),
    }


def main():
    df = load_clean()
    X, y_val = df[FEATURES], df["value_m"]
    y = to_log(y_val)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)

    rows, oof = [], {}
    for name, model in candidate_models().items():
        pred = cross_val_predict(make_pipeline(model), X_tr, y_tr, cv=cv)
        oof[name] = pred
        rows.append({"model": name, **metrics(y_tr, pred)})
        print(f"{name:14s}", rows[-1])
    comp = pd.DataFrame(rows).sort_values("median_ape_pct")
    best = comp.iloc[0]["model"]
    print(f"\nBest model (CV median APE): {best}")

    final = make_pipeline(candidate_models()[best]).fit(X_tr, y_tr)
    test_pred = final.predict(X_te)
    test_metrics = metrics(y_te, test_pred)
    print("Hold-out test:", test_metrics)

    # Empirical 80% prediction interval in log space from out-of-fold residuals
    resid = y_tr.values - oof[best]
    q10, q90 = np.quantile(resid, [0.10, 0.90])
    cover = float(((y_te.values - test_pred >= q10) & (y_te.values - test_pred <= q90)).mean())
    print(f"80% interval empirical coverage on test: {cover:.3f}")

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    joblib.dump({"pipeline": final, "model_name": best, "resid_q10": float(q10),
                 "resid_q90": float(q90), "features": FEATURES,
                 "neighborhoods": sorted(df["neighborhood"].unique())},
                MODELS_DIR / "rent_model.joblib")
    comp.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(
        {"best_model": best, "test": test_metrics, "interval_80_test_coverage": round(cover, 3)}, indent=2))
    pd.concat([X_te, y_te.rename("log_value")], axis=1).to_csv(MODELS_DIR / "holdout.csv", index=False)
    pd.concat([X_tr, y_tr.rename("log_value")], axis=1).to_csv(MODELS_DIR / "train.csv", index=False)


if __name__ == "__main__":
    main()
