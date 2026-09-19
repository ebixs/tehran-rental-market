"""Download, load and clean the Tehran rental listings.

Usage:  python -m rental.data        # download the raw file + write cleaning report
"""
import json
import urllib.request

import pandas as pd

from .config import (AREA_RANGE, DATA_URL, MAX_MONTHLY_RENT, MIN_MONTHLY_RENT, MIN_YEAR, PRICE_M2_TRIM,
                     RAW_DATA, REPORTS_DIR, RENT_TO_DEPOSIT)

MILLION = 1_000_000


def download(path=RAW_DATA, url=DATA_URL, force=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        print(f"already present: {path}")
        return path
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, path)
    return path


def load_raw(path=RAW_DATA) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m rental.data` (or `make data`) first.")
    return pd.read_csv(path)


def normalize_text(s: pd.Series) -> pd.Series:
    """Unify Arabic/Persian letter variants and whitespace."""
    return (s.astype(str)
             .str.replace("ي", "ی").str.replace("ك", "ک")
             .str.replace(r"\s+", " ", regex=True).str.strip())


def clean(df: pd.DataFrame, return_report: bool = False, rent_to_deposit: float = RENT_TO_DEPOSIT):
    """Apply documented cleaning rules and add the deposit-equivalent value.

    Every rule is logged so the cleaning is auditable (reports/cleaning_report.json).
    """
    df = df.copy()
    report = [{"rule": "raw rows", "removed": 0, "remaining": len(df)}]

    def step(name, keep):
        nonlocal df
        removed = int((~keep).sum())
        df = df[keep].copy()
        report.append({"rule": name, "removed": removed, "remaining": len(df)})

    df = df.rename(columns={"elavator": "elevator"})  # typo in the source column name
    df["neighborhood"] = normalize_text(df["neighborhood"])

    step("no price (deposit = 0 and rent = 0)", (df["deposit"] > 0) | (df["rent"] > 0))
    step(f"area outside {AREA_RANGE[0]}-{AREA_RANGE[1]} m2", df["area"].between(*AREA_RANGE))
    step(f"monthly rent > {MAX_MONTHLY_RENT:,} Toman (typo)", df["rent"] <= MAX_MONTHLY_RENT)
    step(f"0 < rent < {MIN_MONTHLY_RENT:,} Toman (placeholder / unit error)",
         (df["rent"] == 0) | (df["rent"] >= MIN_MONTHLY_RENT))
    step(f"build year < {MIN_YEAR}", df["year"] >= MIN_YEAR)

    df["value_m"] = (df["deposit"] + rent_to_deposit * df["rent"]) / MILLION  # million Toman
    df["value_per_m2"] = df["value_m"] / df["area"]
    lo, hi = df["value_per_m2"].quantile(PRICE_M2_TRIM)
    step(f"value/m2 outside {PRICE_M2_TRIM[0]:.1%}-{PRICE_M2_TRIM[1]:.1%} quantiles",
         df["value_per_m2"].between(lo, hi))

    df["full_deposit"] = (df["rent"] == 0).astype(int)      # رهن کامل
    # `total_value` in the source is (almost) just the deposit; `warehouse` is ~constant.
    df = df.drop(columns=["total_value", "warehouse"], errors="ignore").reset_index(drop=True)
    return (df, pd.DataFrame(report)) if return_report else df


def load_clean(path=RAW_DATA):
    return clean(load_raw(path))


def save_cleaning_report(path=RAW_DATA):
    _, rep = clean(load_raw(path), return_report=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "cleaning_report.json").write_text(
        json.dumps(rep.to_dict("records"), indent=2, ensure_ascii=False))
    return rep


if __name__ == "__main__":
    download()
    print(save_cleaning_report().to_string(index=False))
