"""Rental value estimator.  Run: streamlit run app/streamlit_app.py  (after `make train`)"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rental.config import RENT_TO_DEPOSIT  # noqa: E402
from rental.predict import estimate, load_model  # noqa: E402

st.set_page_config(page_title="Tehran Rental Estimator", page_icon="🏠", layout="wide")
st.title("🏠 Tehran Rental Value Estimator")
st.caption("Estimates the deposit-equivalent value of an apartment listing (million Toman) "
           "from neighborhood, size, age and amenities, with an 80% prediction range.")


@st.cache_resource
def get_model():
    return load_model()


try:
    art = get_model()
except FileNotFoundError:
    st.error("Model not found. Run `make data && make train` first.")
    st.stop()

c1, c2 = st.columns(2)
with c1:
    nbh = st.selectbox("Neighborhood", art["neighborhoods"],
                       index=art["neighborhoods"].index("سعادت‌آباد") if "سعادت‌آباد" in art["neighborhoods"] else 0)
    area = st.slider("Area (m²)", 30, 400, 100)
    year = st.slider("Build year (Jalali)", 1365, 1400, 1390)
with c2:
    elevator = st.checkbox("Elevator", True)
    parking = st.checkbox("Parking", True)

row = pd.DataFrame([{"neighborhood": nbh, "area": area, "year": year,
                     "elevator": int(elevator), "parking": int(parking)}])
res = estimate(row, art).iloc[0]

st.divider()
m1, m2, m3 = st.columns(3)
m1.metric("Estimated value", f"{res.estimate_million_toman:,.0f} M Toman")
m2.metric("Low (10th pct)", f"{res.low_million_toman:,.0f} M")
m3.metric("High (90th pct)", f"{res.high_million_toman:,.0f} M")

st.subheader("Equivalent deposit / rent combinations")
st.caption(f"Using the convention 1 Toman monthly rent ≈ {RENT_TO_DEPOSIT} Toman deposit.")
val = float(res.estimate_million_toman)
combos = pd.DataFrame({
    "deposit share": ["100% (full deposit)", "75%", "50%", "25%"],
    "deposit (M Toman)": [round(val * s) for s in (1, .75, .5, .25)],
    "monthly rent (M Toman)": [round(val * (1 - s) / RENT_TO_DEPOSIT, 2) for s in (1, .75, .5, .25)],
})
st.dataframe(combos, hide_index=True)

ranking = ROOT / "reports" / "neighborhood_ranking.csv"
if ranking.exists():
    rk = pd.read_csv(ranking)
    hit = rk[rk["neighborhood"] == nbh]
    if len(hit):
        st.info(f"Neighborhood benchmark: median {hit.iloc[0].median_value_per_m2:.1f} M Toman per m² "
                f"across {int(hit.iloc[0].n)} listings.")
st.caption("Model trained on 1399-era listings; values are asking prices in Toman of that period, "
           "not current market prices.")
