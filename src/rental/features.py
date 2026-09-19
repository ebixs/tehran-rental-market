"""Feature engineering + preprocessing."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, TargetEncoder

from .config import CATEGORICAL_HIGH_CARD, NUMERIC, RANDOM_STATE, REF_YEAR


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age"] = (REF_YEAR - df["year"]).clip(lower=0)
    df["log_area"] = np.log(df["area"])
    return df


def build_preprocessor() -> ColumnTransformer:
    """Neighborhood has ~300 levels, so it is target-encoded *inside* the CV
    pipeline (smoothed and cross-fitted) instead of one-hot encoded."""
    return ColumnTransformer(
        [
            ("nbh", TargetEncoder(target_type="continuous", smooth="auto",
                                  cv=5, random_state=RANDOM_STATE), CATEGORICAL_HIGH_CARD),
            ("num", StandardScaler(), NUMERIC),
        ]
    )
