"""How sensitive are the conclusions to the rent->deposit conversion constant?

Usage: python -m rental.sensitivity
"""
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE, REPORTS_DIR, RENT_TO_DEPOSIT, TEST_SIZE
from .data import clean, load_raw
from .eda import neighborhood_stats
from .pipeline import candidate_models, make_pipeline
from .train import FEATURES, metrics, to_log


def main(factors=(10, 30, 50, 100)):
    raw = load_raw()
    base = neighborhood_stats(clean(raw, rent_to_deposit=RENT_TO_DEPOSIT))["median_value_per_m2"]
    rows = []
    for f in factors:
        df = clean(raw, rent_to_deposit=f)
        ranks = neighborhood_stats(df)["median_value_per_m2"]
        common = base.index.intersection(ranks.index)
        rho = spearmanr(base[common], ranks[common])[0]
        X, y = df[FEATURES], to_log(df["value_m"])
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        pred = make_pipeline(candidate_models()["random_forest"]).fit(X_tr, y_tr).predict(X_te)
        rows.append({"rent_to_deposit": f, "neighborhood_rank_spearman_vs_30": round(rho, 3),
                     **metrics(y_te, pred)})
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS_DIR / "sensitivity.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
