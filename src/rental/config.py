from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = ROOT / "data" / "raw" / "tehran_rentals.csv"
DATA_URL = ("https://raw.githubusercontent.com/amiralimadadi/"
            "Regression_TheranHousing/main/Data.csv")
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Iranian rentals mix a lump-sum deposit (ودیعه/رهن) with monthly rent (اجاره).
# To compare listings we convert everything to a "deposit-equivalent" value:
#   value = deposit + RENT_TO_DEPOSIT * monthly_rent           (all in Toman)
# 30 is a common market rule of thumb (1M Toman monthly rent ~ 30M Toman deposit).
# Sensitivity to this constant is checked in `python -m rental.sensitivity`.
RENT_TO_DEPOSIT = 30

# Jalali (Solar Hijri) reference year for building age. The newest buildings in the
# snapshot were built in 1399, so the listings were scraped around 1399-1400.
REF_YEAR = 1400

# Documented, reproducible cleaning rules (see data.clean)
AREA_RANGE = (20, 600)          # m2
MAX_MONTHLY_RENT = 300_000_000  # Toman; above this is almost surely a typo
MIN_MONTHLY_RENT = 500_000      # Toman; a non-zero rent below this is a placeholder / unit error
                                # (e.g. 98 listings share the exact value 100,000)
MIN_YEAR = 1350
PRICE_M2_TRIM = (0.005, 0.995)  # quantile trimming on value per m2

MIN_LISTINGS_FOR_RANKING = 30

NUMERIC = ["area", "log_area", "age", "elevator", "parking"]
CATEGORICAL_HIGH_CARD = ["neighborhood"]
TARGET = "log_value"            # log1p(deposit-equivalent in million Toman)
