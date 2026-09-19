"""Estimate the rental value of a property with an 80% prediction range."""
import joblib
import numpy as np
import pandas as pd

from .config import MODELS_DIR
from .train import from_log


def load_model(path=MODELS_DIR / "rent_model.joblib"):
    return joblib.load(path)


def estimate(df: pd.DataFrame, artifact=None) -> pd.DataFrame:
    """df needs: neighborhood, area (m2), year (Jalali build year), elevator, parking (0/1).

    Returns deposit-equivalent value in million Toman with an 80% range.
    """
    artifact = artifact or load_model()
    z = artifact["pipeline"].predict(df[artifact["features"]])
    return pd.DataFrame({
        "estimate_million_toman": from_log(z).round(0),
        "low_million_toman": from_log(z + artifact["resid_q10"]).round(0),
        "high_million_toman": from_log(z + artifact["resid_q90"]).round(0),
    })


def as_full_deposit_and_rent(value_million: float, deposit_share: float = 0.5, rent_to_deposit: float = 30):
    """Split a deposit-equivalent value into (deposit, monthly rent) in million Toman."""
    deposit = value_million * deposit_share
    return deposit, (value_million - deposit) / rent_to_deposit
