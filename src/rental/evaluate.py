"""Evaluation plots + SHAP explainability.  Usage: python -m rental.evaluate"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from .config import FIGURES_DIR, MODELS_DIR, MIN_LISTINGS_FOR_RANKING
from .plotting import fa, setup
from .train import from_log


def main():
    setup()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    art = joblib.load(MODELS_DIR / "rent_model.joblib")
    pipe = art["pipeline"]
    te = pd.read_csv(MODELS_DIR / "holdout.csv")
    X, y_log = te.drop(columns=["log_value"]), te["log_value"]
    pred_log = pipe.predict(X)
    y, p = from_log(y_log), from_log(pred_log)
    lo, hi = from_log(pred_log + art["resid_q10"]), from_log(pred_log + art["resid_q90"])

    # 1. predicted vs actual + residual distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].scatter(y, p, s=7, alpha=0.3, color="#4c72b0")
    lim = [y.min(), y.max()]
    axes[0].plot(lim, lim, "k--", lw=1)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("actual value (million Toman)"); axes[0].set_ylabel("predicted value (million Toman)")
    axes[0].set_title(f"Predicted vs. actual ({art['model_name']}, hold-out)")
    ape = (np.abs(p - y) / y * 100).clip(upper=100)
    sns.histplot(ape, bins=40, ax=axes[1], color="#55a868")
    axes[1].axvline(np.median(ape), color="k", ls="--", lw=1, label=f"median = {np.median(ape):.1f}%")
    axes[1].set_xlabel("absolute percentage error (%, clipped at 100)"); axes[1].legend()
    axes[1].set_title("Error distribution")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "06_model_evaluation.png", dpi=150); plt.close(fig)

    # 2. error by neighborhood tier (location price level)
    d = pd.DataFrame({"nbh": X["neighborhood"], "ape": np.abs(p - y) / y * 100, "y": y})
    med = d.groupby("nbh")["y"].transform("median")
    d["tier"] = pd.qcut(med.rank(method="first"), 4, labels=["Q1 cheapest", "Q2", "Q3", "Q4 priciest"])
    tier = d.groupby("tier", observed=True)["ape"].median()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    sns.barplot(x=tier.index, y=tier.values, color="#4c72b0", ax=ax)
    for i, v in enumerate(tier.values):
        ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("median abs. % error"); ax.set_xlabel("neighborhood price quartile")
    ax.set_title("Error by market segment")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "07_error_by_segment.png", dpi=150); plt.close(fig)
    tier.round(1).to_csv(FIGURES_DIR.parent / "error_by_segment.csv", header=["median_ape_pct"])

    # 3. SHAP
    feats, prep, model = pipe.named_steps["features"], pipe.named_steps["prep"], pipe.named_steps["model"]
    names = [n.replace("nbh__neighborhood", "neighborhood (target-encoded)").replace("num__", "")
             for n in prep.get_feature_names_out()]
    Xs = prep.transform(feats.transform(X))
    if hasattr(model, "get_booster"):
        import xgboost as xgb
        vals = model.get_booster().predict(xgb.DMatrix(Xs, feature_names=list(names)), pred_contribs=True)[:, :-1]
    else:
        idx = np.random.RandomState(0).choice(len(Xs), size=min(800, len(Xs)), replace=False)
        Xs = Xs[idx]
        vals = shap.TreeExplainer(model).shap_values(Xs)
    plt.figure()
    shap.summary_plot(vals, Xs, feature_names=names, show=False)
    plt.title("SHAP - what drives predicted rental value (log scale)")
    plt.tight_layout(); plt.savefig(FIGURES_DIR / "08_shap_summary.png", dpi=150, bbox_inches="tight"); plt.close()
    imp = pd.Series(np.abs(vals).mean(0), index=names).sort_values(ascending=False)
    print("mean |SHAP|:\n", imp.round(3).to_string())
    print("\nmedian APE by segment:\n", tier.round(1).to_string())
    cover = float(((y >= lo) & (y <= hi)).mean())
    print(f"80% interval coverage (hold-out): {cover:.3f}")


if __name__ == "__main__":
    main()
