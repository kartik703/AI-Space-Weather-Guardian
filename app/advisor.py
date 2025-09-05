"""
advisor.py
----------
Risk classification and advisory messaging for Kp-based space weather.
Provides:
- risk_level(kp): "Low" | "Medium" | "High"
- advisory_message(kp): general guidance text
- sector_advisory(kp): dict advisories for Aviation, Satellites, Energy
- assess_forecast(forecasts): summarize highest risk across multi-horizon forecasts
"""

from typing import Dict, Tuple


# ---- Thresholds (tweak here if you want different cutoffs) ----
LOW_MAX = 4.0      # Kp < 4       → Low
MED_MAX = 6.0      # 4 ≤ Kp < 6    → Medium
# Kp ≥ 6            → High


def risk_level(kp_value: float) -> str:
    """
    Classify risk level based on Kp.
    """
    if kp_value < LOW_MAX:
        return "Low"
    elif kp_value < MED_MAX:
        return "Medium"
    else:
        return "High"


def advisory_message(kp_value: float) -> str:
    """
    General audience advisory string for a given Kp value.
    """
    level = risk_level(kp_value)
    if level == "Low":
        return "✅ No significant space weather impact expected."
    elif level == "Medium":
        return "⚠️ Moderate geomagnetic activity. GPS and HF radio may degrade; slight satellite drag possible."
    else:
        return ("🚨 High geomagnetic storm risk! Potential disruptions to satellites, "
                "aviation (HF/GNSS), and power grids. Prepare mitigation steps.")


def sector_advisory(kp_value: float) -> Dict[str, str]:
    """
    Sector-specific guidance for Aviation, Satellites, and Energy Grids.
    Returns a dict of {sector: message}.
    """
    level = risk_level(kp_value)
    out: Dict[str, str] = {}

    if level == "Low":
        out["Aviation"] = "Normal ops. No significant HF/GNSS risk."
        out["Satellites"] = "Nominal environment; minimal radiation/drag."
        out["Energy"] = "No geomagnetic disturbances expected."
    elif level == "Medium":
        out["Aviation"] = "Polar/HT ops: possible HF blackouts, GNSS accuracy dips. Plan alternates."
        out["Satellites"] = "Slight drag/comms degradation possible. Heighten monitoring."
        out["Energy"] = "Minor GICs at high latitudes possible. Review standby procedures."
    else:  # High
        out["Aviation"] = "Reroute polar flights where feasible. Expect HF/GNSS unreliability."
        out["Satellites"] = "Elevated radiation risk; consider safe-mode windows & uplink schedule review."
        out["Energy"] = "⚡ Heightened GIC risk. Engage storm procedures & load-balancing plans."

    return out


# ---------- Helpers for multi-horizon forecasts ----------

def assess_forecast(forecasts: Dict[str, float]) -> Tuple[str, float, str]:
    """
    Given a dict of forecasts like {"1h": 2.1, "3h": 3.0, "6h": 6.2},
    return (worst_horizon, worst_kp, worst_risk_level).
    """
    if not forecasts:
        return ("", 0.0, "Low")
    # Identify horizon with maximum Kp
    worst_h = max(forecasts.keys(), key=lambda h: forecasts[h])
    worst_kp = float(forecasts[worst_h])
    return (worst_h, worst_kp, risk_level(worst_kp))


def advisory_for_forecast(forecasts: Dict[str, float]) -> str:
    """
    Short summary string for a multi-horizon forecast dict.
    """
    worst_h, worst_kp, worst_level = assess_forecast(forecasts)
    if not worst_h:
        return "No forecast available."
    base = f"Peak forecast: Kp {worst_kp:.1f} at {worst_h} → {worst_level} risk."
    return f"{base} {advisory_message(worst_kp)}"
