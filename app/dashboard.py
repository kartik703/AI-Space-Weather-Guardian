import streamlit as st
import requests
import pandas as pd
import altair as alt
import sys, os
import os
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/forecast")
# Ensure project root is on path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestion import fetch_noaa_kp
from app.advisor import risk_level, advisory_message, sector_advisory


API_URL = "http://127.0.0.1:8000/forecast"  # FastAPI endpoint

# --------------------------
# Page Config
# --------------------------
st.set_page_config(page_title="AI Space Weather Guardian", layout="centered")
st.title("🌌 AI Space Weather Guardian")
st.markdown("AI-powered **multi-horizon forecasts** of geomagnetic activity (**Kp**).")


# --------------------------
# Helper: colored label for risk
# --------------------------
def risk_badge(kp_val: float) -> str:
    lvl = risk_level(kp_val)
    color = {"Low": "#16a34a", "Medium": "#f59e0b", "High": "#ef4444"}[lvl]
    return f"<span style='background:{color}22;color:{color};padding:3px 8px;border-radius:8px;font-weight:600'>{lvl}</span>"


# --------------------------
# Latest Kp index
# --------------------------
df = fetch_noaa_kp().tail(50)  # returns columns: time_tag, Kp
latest_kp = float(df["Kp"].iloc[-1])
prev_kp = float(df["Kp"].iloc[-2]) if len(df) >= 2 else latest_kp
st.metric("Latest Kp Index", f"{latest_kp:.2f}", delta=latest_kp - prev_kp)

cur_risk = risk_level(latest_kp)

# 🚨 High Risk Banner
if cur_risk == "High":
    st.markdown(
        f"""
        <div style="background-color:#ff4c4c; padding:15px; border-radius:10px; text-align:center; animation: blinker 1s linear infinite;">
            🚨 <strong>High Risk Alert!</strong> Kp {latest_kp:.1f} — Severe geomagnetic storm potential. 
            Possible disruptions to satellites, aviation, and energy grids.
        </div>
        <style>
        @keyframes blinker {{ 50% {{ opacity: 0; }} }}
        </style>
        """,
        unsafe_allow_html=True
    )


# --------------------------
# General Risk Assessment
# --------------------------
st.subheader("Current Risk Assessment")
st.markdown(f"**Level**: {risk_badge(latest_kp)}", unsafe_allow_html=True)
st.info(advisory_message(latest_kp))


# --------------------------
# Sector-Specific Advisories
# --------------------------
st.subheader("📡 Sector-Specific Advisories")
msgs = sector_advisory(latest_kp)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("✈️ **Aviation**")
    if "⚠️" in msgs["Aviation"] or "Reroute" in msgs["Aviation"]:
        st.warning(msgs["Aviation"])
    else:
        st.success(msgs["Aviation"])

with c2:
    st.markdown("🛰 **Satellites**")
    if "⚠️" in msgs["Satellites"] or "safe mode" in msgs["Satellites"]:
        st.warning(msgs["Satellites"])
    else:
        st.success(msgs["Satellites"])

with c3:
    st.markdown("⚡ **Energy Grids**")
    if "⚡" in msgs["Energy"] or "risk" in msgs["Energy"]:
        st.warning(msgs["Energy"])
    else:
        st.success(msgs["Energy"])


# --------------------------
# Forecasts from API (real Kp) with uncertainty
# --------------------------
st.subheader("🔮 Forecasts (Kp) — 1h / 3h / 6h")

results = None
uncert = None
try:
    r = requests.get(API_URL, timeout=10)
    r.raise_for_status()
    payload = r.json()
    results = payload.get("forecasts", {})
    uncert = float(payload.get("uncertainty_kp", 0.3))
except Exception as e:
    st.error(f"❌ Could not fetch forecasts: {e}")

if results:
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Best Model Forecasts**")
        for horizon, val in results.items():
            v = float(val)
            st.markdown(
                f"{horizon}: **{v:.2f} ± {uncert:.2f}** &nbsp; {risk_badge(v)}",
                unsafe_allow_html=True
            )
    with col2:
        peak_h = max(results, key=lambda k: results[k])
        peak_v = float(results[peak_h])
        st.write("**Advisory (Peak Horizon)**")
        st.markdown(f"Peak: **{peak_h}**")
        st.markdown(f"Kp Forecast: **{peak_v:.2f} ± {uncert:.2f}** &nbsp; {risk_badge(peak_v)}", unsafe_allow_html=True)
        st.caption(advisory_message(peak_v))
else:
    st.info("No forecasts available yet. Ensure the FastAPI server is running.")


# --------------------------
# Chart: Past 24h + Forecast with band
# --------------------------
st.subheader("📈 Kp: Past 24h + Forecast")

history = df[["time_tag", "Kp"]].rename(columns={"time_tag": "Time"}).tail(24)
history["Type"] = "History"

if results:
    last_t = history["Time"].iloc[-1]
    items = sorted(((int(k.rstrip("h")), float(v)) for k, v in results.items()), key=lambda x: x[0])
    f_times = [last_t + pd.Timedelta(hours=h) for h, _ in items]
    f_vals  = [v for _, v in items]
    fdf = pd.DataFrame({"Time": pd.to_datetime(f_times), "Kp": f_vals, "Type": "Forecast"})
    # ribbons
    fdf["Kp_lo"] = fdf["Kp"] - (uncert or 0.0)
    fdf["Kp_hi"] = fdf["Kp"] + (uncert or 0.0)

    combined = pd.concat([history, fdf])

    base = alt.Chart(combined).encode(x="Time:T")

    hist_line = base.transform_filter(alt.datum.Type == "History").mark_line().encode(
        y=alt.Y("Kp:Q", title="Kp"),
        color=alt.value("#60a5fa")
    )

    band = alt.Chart(fdf).mark_area(opacity=0.2).encode(
        x="Time:T",
        y="Kp_lo:Q",
        y2="Kp_hi:Q",
        color=alt.value("#f59e0b")
    )

    forecast_line = alt.Chart(fdf).mark_line(color="#f59e0b").encode(
        x="Time:T",
        y="Kp:Q"
    )

    st.altair_chart((hist_line + band + forecast_line).properties(height=320), use_container_width=True)
else:
    st.line_chart(history.set_index("Time")["Kp"])
