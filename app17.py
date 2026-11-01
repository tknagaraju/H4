"""
Centum Electronics — Working Capital Decision Support System
=================================================================
Methodology (per dissertation):
  - Historical KPIs (DSO, DIO, DPO, CCC) are CALCULATED from uploaded
    Revenue / COGS / Inventory / Receivables / Payables data.
  - REVENUE will be forecasted via historical data input or Manual input (entered by the user).
  - COGS, Receivables, Inventory and Payables are AUTO-FORECASTED using
    driver-based linear regressions fitted on historical data:
        COGS         = f(Revenue)
        Receivables  = f(Revenue)
        Inventory    = f(COGS)
        Payables     = f(COGS)
  - DSO, DIO, DPO, CCC are then derived automatically from these.
  - Everything is benchmarked against a weighted peer benchmark,
    and DPO is treated as a funding-risk indicator (not a strength) once
    it exceeds standard trade terms (default 90 days).

Multi-sheet workbook support (peer benchmarking):
  - Excel upload: FIRST sheet = subject company (drives the whole DSS, exactly as
    before); every other sheet = a peer company (sheet name = company name).
  - Historical average DSO / DIO / DPO / CCC is calculated for every sheet.
  - The user tags each peer as EMS-based or Design-based and sets the EMS : Design
    weight ratio. The best peer of each category (lowest CCC) is weighted into the
    final benchmark. With no peer sheets, or no peer classified, the default
    benchmark values are kept unchanged.
"""

import io
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels
import sklearn
import plotly
from scipy.optimize import minimize

from statsmodels.tsa.holtwinters import (
    SimpleExpSmoothing,
    Holt
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)
# =========================================================
# THEME / CONSTANTS
# =========================================================
NAVY = "#132A4C"
STEEL = "#3D5A80"
GOLD = "#D69A2D"
GREEN = "#2E7D4F"
RED = "#B5432B"
AMBER = "#B5842B"
ICE = "#E9EEF6"
GREY = "#5B6B82"

COVID_YEARS = {"FY2020-21", "FY2021-22"}

REQUIRED_COLS = ["Year", "Revenue", "COGS", "Inventory", "Receivables", "Payables"]

st.set_page_config(
    page_title="Centum WC Decision Support Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# LINKED SCENARIO INPUTS — shared state between the sidebar
# widgets ("Scenario DIO/DSO Reduction", "Cost of Funds") and
# the Tab 7 ("Scenario Analysis") widgets.
#
# Streamlit widgets are only linked to each other if they share
# the exact same `key`. Since the sidebar and the Tab 7 controls
# need different labels/ranges, we give each its own key and use
# on_change callbacks to mirror one into the other. Whichever one
# the user edits, the other is updated before the next rerun.
# =========================================================
_SCENARIO_DEFAULTS = {
    "side_dio_reduction": 10.0,
    "tab_dio_reduction": 10.0,
    "side_dso_reduction": 3.0,
    "tab_dso_reduction": 3.0,
    "side_interest_rate": 8.0,
    "tab_interest_rate": 8.0,
}
for _k, _v in _SCENARIO_DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


def _sync_dio_from_side():
    st.session_state["tab_dio_reduction"] = st.session_state["side_dio_reduction"]


def _sync_dio_from_tab():
    st.session_state["side_dio_reduction"] = st.session_state["tab_dio_reduction"]


def _sync_dso_from_side():
    st.session_state["tab_dso_reduction"] = st.session_state["side_dso_reduction"]


def _sync_dso_from_tab():
    st.session_state["side_dso_reduction"] = st.session_state["tab_dso_reduction"]


def _sync_rate_from_side():
    st.session_state["tab_interest_rate"] = st.session_state["side_interest_rate"]


def _sync_rate_from_tab():
    st.session_state["side_interest_rate"] = st.session_state["tab_interest_rate"]


# ---------------------------------------------------------
# Global CSS for KPI cards
# ---------------------------------------------------------
st.markdown(f"""
<style>
.kpi-card {{
    border-radius: 10px;
    padding: 16px 18px;
    color: white;
    margin-bottom: 10px;
}}
.kpi-title {{ font-size: 13px; font-weight: 600; opacity: 0.9; }}
.kpi-value {{ font-size: 30px; font-weight: 800; margin: 4px 0; }}
.kpi-sub   {{ font-size: 12px; opacity: 0.85; }}
.status-pill {{
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    font-size: 11.5px; font-weight: 700; margin-top: 6px;
}}
</style>
""", unsafe_allow_html=True)


def kpi_card(title, value, sub, bg):
    st.markdown(f"""
    <div class="kpi-card" style="background:{bg};">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def status_pill(label, color):
    st.markdown(
        f'<span class="status-pill" style="background:{color}22;color:{color};">{label}</span>',
        unsafe_allow_html=True,
    )


# =========================================================
# SAMPLE / DEMO DATA  (used only if no file is uploaded)
# NOTE: illustrative only — replace with your real CCC_DATA2.xlsx export.
# =========================================================
def sample_data():
    years = ["FY2015-16", "FY2016-17", "FY2017-18", "FY2018-19", "FY2019-20",
              "FY2022-23", "FY2023-24", "FY2024-25", "FY2025-26"]
    rng = np.random.default_rng(7)
    revenue = np.array([3350, 3860, 3580, 5060, 4830, 6340, 7480, 9740, 9720], dtype=float)
    cogs = revenue * 0.66 + rng.normal(0, 40, len(revenue))
    inventory = cogs * 0.62 + rng.normal(0, 60, len(revenue))
    receivables = revenue * 0.33 + rng.normal(0, 60, len(revenue))
    payables = cogs * 0.34 + rng.normal(0, 40, len(revenue))
    return pd.DataFrame({
        "Year": years, "Revenue": revenue.round(2), "COGS": cogs.round(2),
        "Inventory": inventory.round(2), "Receivables": receivables.round(2),
        "Payables": payables.round(2),
    })


def make_template_csv():
    df = sample_data()
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


# =========================================================
# DATA LOADING
# =========================================================
@st.cache_data(show_spinner=False)
def load_uploaded(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}. "
                          f"Expected columns: {REQUIRED_COLS}")
    return df[REQUIRED_COLS].copy()


def compute_kpis(df):
    df = df.copy()
    df["DSO"] = df["Receivables"] / df["Revenue"] * 360
    df["DIO"] = df["Inventory"] / df["COGS"] * 360
    df["DPO"] = df["Payables"] / df["COGS"] * 360
    df["CCC"] = df["DSO"] + df["DIO"] - df["DPO"]
    return df


# =========================================================
# MULTI-SHEET PEER BENCHMARKING  (NEW)
# Workbook layout:  sheet 1 = subject company (drives the whole DSS)
#                   sheet 2..n = peer companies (sheet name = company name)
# =========================================================
PEER_KPI_COLS = ["DSO", "DIO", "DPO", "CCC"]

_FY_LABEL = re.compile(r"^(?:FY)?\s*(\d{4})\s*[-‐-―/]\s*(\d{2}|\d{4})$")


def is_covid_year(label):
    """True if `label` is one of the COVID years, whichever way it is written:
    FY2020-21 · FY2020-2021 · FY 2020-21 · 2020-21 · FY2020–21 (en-dash) ...
    Every other label (including non-text values) returns False."""
    m = _FY_LABEL.match(str(label).strip().upper())
    if not m:
        return False
    return f"FY{m.group(1)}-{m.group(2)[-2:]}" in COVID_YEARS
BEST_PEER_BASIS = "Best peer (lowest CCC)"
AVG_PEER_BASIS = "Average of selected peers"


@st.cache_data(show_spinner=False)
def load_all_sheets(file_bytes, file_name):
    """Read EVERY sheet of an uploaded Excel workbook.

    Returns (sheets, skipped, sheet_order)
      sheets      : {sheet_name: cleaned DataFrame with REQUIRED_COLS}
      skipped     : [(sheet_name, reason), ...] for sheets that cannot be used
      sheet_order : all sheet names in workbook order (first = subject company)
    """
    xl = pd.ExcelFile(io.BytesIO(file_bytes))
    sheets, skipped = {}, []
    for name in xl.sheet_names:
        try:
            df = pd.read_excel(xl, sheet_name=name)
            df.columns = [str(c).strip() for c in df.columns]
            missing = [c for c in REQUIRED_COLS if c not in df.columns]
            if missing:
                skipped.append((name, f"missing column(s) {missing}"))
                continue
            df = df[REQUIRED_COLS].copy()
            for c in REQUIRED_COLS[1:]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=REQUIRED_COLS)
            df = df[(df["Revenue"] > 0) & (df["COGS"] > 0)].reset_index(drop=True)
            if df.empty:
                skipped.append((name, "no valid data rows"))
                continue
            sheets[name] = df
        except Exception as e:  # never let one bad sheet break the app
            skipped.append((name, f"could not be read ({e})"))
    return sheets, skipped, list(xl.sheet_names)


def compute_peer_averages(sheets, exclude_covid):
    """Historical average DSO / DIO / DPO / CCC for each sheet.
    Same KPI formulas as the subject company (mean of the annual ratios)."""
    rows, skipped = [], []
    for name, df in sheets.items():
        d = df
        if exclude_covid:
            d = d[~d["Year"].map(is_covid_year).astype(bool)]
        if d.empty:
            skipped.append((name, "no data left after COVID-year exclusion"))
            continue
        k = compute_kpis(d)
        rows.append({
            "Company": name, "Years": int(len(k)),
            "DSO": float(k["DSO"].mean()), "DIO": float(k["DIO"].mean()),
            "DPO": float(k["DPO"].mean()), "CCC": float(k["CCC"].mean()),
        })
    return pd.DataFrame(rows, columns=["Company", "Years"] + PEER_KPI_COLS), skipped


def derive_peer_benchmark(peer_avg_df, peer_class, ems_weight_pct, basis):
    """Final weighted benchmark from the user's EMS / Design tagging.

    For each category (EMS, Design):
      - basis = best peer   -> the peer with the lowest historical CCC is the
                                category benchmark (all four metrics from that same peer)
      - basis = average     -> simple average of the peers selected in that category
    Final benchmark = EMS weight x EMS benchmark + Design weight x Design benchmark.
    If only one category has peers, that category carries 100% of the weight.
    Returns None when no peer has been tagged (caller keeps the default benchmark).
    """
    def category(names):
        sub = peer_avg_df[peer_avg_df["Company"].isin(names)]
        if sub.empty:
            return None
        if basis == BEST_PEER_BASIS:
            best = sub.loc[sub["CCC"].idxmin()]
            vals = {m: float(best[m]) for m in ("DSO", "DIO", "DPO")}
            picked, label = str(best["Company"]), str(best["Company"])
        else:
            vals = {m: float(sub[m].mean()) for m in ("DSO", "DIO", "DPO")}
            picked, label = None, f"average of {len(sub)} peer(s)"
        vals["CCC"] = vals["DSO"] + vals["DIO"] - vals["DPO"]
        return {"values": vals, "picked": picked, "label": label,
                "members": list(sub["Company"])}

    ems = category([n for n, c in peer_class.items() if c == "EMS"])
    des = category([n for n, c in peer_class.items() if c == "Design"])
    if ems is None and des is None:
        return None

    if ems is not None and des is not None:
        w_ems = float(ems_weight_pct) / 100.0
        w_des = 1.0 - w_ems
    elif ems is not None:
        w_ems, w_des = 1.0, 0.0
    else:
        w_ems, w_des = 0.0, 1.0

    final = {}
    for m in ("DSO", "DIO", "DPO"):
        final[m] = (w_ems * ems["values"][m] if ems else 0.0) + \
                   (w_des * des["values"][m] if des else 0.0)
    final["CCC"] = final["DSO"] + final["DIO"] - final["DPO"]

    parts = []
    if ems is not None:
        parts.append(f"EMS {w_ems * 100:.0f}% × {ems['label']}")
    if des is not None:
        parts.append(f"Design {w_des * 100:.0f}% × {des['label']}")
    summary = " + ".join(parts)
    if ems is None or des is None:
        summary += " (only one category selected → carries 100%)"

    return {"final": final, "ems": ems, "design": des,
            "w_ems": w_ems, "w_design": w_des, "basis": basis, "summary": summary}


# =========================================================
# DRIVER-BASED REGRESSION MODELS (fit on historical data)
# =========================================================
@st.cache_data(show_spinner=False)
def fit_models(df):
    def fit(x, y):
        slope, intercept = np.polyfit(x, y, 1)
        pred = slope * x + intercept
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        mae = mean_absolute_error(y, pred)

        rmse = np.sqrt(
        mean_squared_error(y, pred)
        )

        mape = np.mean(
        np.abs((y - pred) / y)
        ) * 100

        return {"slope": slope, "intercept": intercept, "r2": r2, "mae": mae,"rmse": rmse, "mape": mape}

    models = {
        "cogs_on_revenue": fit(df["Revenue"].values, df["COGS"].values),
        "receivables_on_revenue": fit(df["Revenue"].values, df["Receivables"].values),
        "inventory_on_cogs": fit(df["COGS"].values, df["Inventory"].values),
        "payables_on_cogs": fit(df["COGS"].values, df["Payables"].values),
    }
    return models


def predict(m, x):
    return m["slope"] * x + m["intercept"]


def forecast_from_revenue(revenue, models):
    cogs = predict(models["cogs_on_revenue"], revenue)
    receivables = predict(models["receivables_on_revenue"], revenue)
    inventory = predict(models["inventory_on_cogs"], cogs)
    payables = predict(models["payables_on_cogs"], cogs)
    dso = receivables / revenue * 360
    dio = inventory / cogs * 360
    dpo = payables / cogs * 360
    ccc = dso + dio - dpo
    return {
        "Revenue": revenue, "COGS": cogs, "Inventory": inventory,
        "Receivables": receivables, "Payables": payables,
        "DSO": dso, "DIO": dio, "DPO": dpo, "CCC": ccc,
    }



# =========================================================
# PRESCRIPTIVE OPTIMISATION — shared by the Optimization tab and the
# Revenue Sensitivity section of the Scenario Analysis tab
# =========================================================
def optimise_wc(current_dio, current_dso, current_dpo, cogs, revenue,
                bm_dio, bm_dso, dso_closure_pct, min_inventory_exposure,
                interest_rate):
    """Maximise cash release  Z = x1 x COGS / 360 + x2 x Revenue / 360
    where x1 = DIO reduction (days) and x2 = DSO reduction (days), subject to

        1. DIO floor:                target DIO >= benchmark DIO
        2. Net inventory exposure:   (target DIO - DPO) >= minimum exposure
        3. DSO gap-closure cap:      x2 <= closure % x (DSO - benchmark DSO)
        4. Non-negativity:           x1, x2 >= 0

    The objective and every constraint are linear in (x1, x2), so this is a
    linear programme. It is solved with SciPy's SLSQP; a closed-form check
    guards against solver failure (for example an infeasible exposure floor).
    """
    max_dso_reduction = max((current_dso - bm_dso) * (dso_closure_pct / 100.0), 0.0)
    max_dio_reduction = max(current_dio - bm_dio, 0.0)
    exposure_cap = max(current_dio - current_dpo - min_inventory_exposure, 0.0)
    x1_exact = min(max_dio_reduction, exposure_cap)
    x2_exact = max_dso_reduction

    def objective(x):
        return -((cogs * x[0] / 360.0) + (revenue * x[1] / 360.0))

    def dio_constraint(x):
        return (current_dio - x[0]) - bm_dio

    def inventory_exposure_constraint(x):
        return (current_dio - x[0] - current_dpo) - min_inventory_exposure

    try:
        res = minimize(
            objective,
            x0=[min(5.0, max_dio_reduction), min(1.0, max_dso_reduction)],
            bounds=[(0.0, max_dio_reduction), (0.0, max_dso_reduction)],
            constraints=[{"type": "ineq", "fun": dio_constraint},
                         {"type": "ineq", "fun": inventory_exposure_constraint}],
            method="SLSQP",
        )
        x1, x2 = float(res.x[0]), float(res.x[1])
        ok = bool(res.success) and abs(x1 - x1_exact) < 1e-3 and abs(x2 - x2_exact) < 1e-3
    except Exception:
        ok = False
    if not ok:
        x1, x2 = x1_exact, x2_exact

    target_dio = current_dio - x1
    target_dso = current_dso - x2
    target_ccc = target_dso + target_dio - current_dpo
    current_ccc = current_dso + current_dio - current_dpo
    inv_cash = cogs * x1 / 360.0
    rec_cash = revenue * x2 / 360.0
    total_cash = inv_cash + rec_cash
    return {
        "dio_reduction": x1, "dso_reduction": x2,
        "target_dio": target_dio, "target_dso": target_dso, "target_ccc": target_ccc,
        "ccc_improvement": current_ccc - target_ccc,
        "inventory_cash_release": inv_cash, "receivable_cash_release": rec_cash,
        "total_cash_release": total_cash,
        "annual_interest_saved": total_cash * interest_rate / 100.0,
        "net_inventory_days": target_dio - current_dpo,
    }


def revenue_sensitivity(models, cases, bm_dio, bm_dso, dso_closure_pct,
                        min_inventory_exposure, interest_rate):
    """Re-run the full cascade, the benchmark gap and the optimiser for each
    (label, revenue) case. Fitted regressions, benchmark, constraints and cost
    of funds are held unchanged; only revenue varies."""
    rows = []
    for label, rev in cases:
        f = forecast_from_revenue(rev, models)
        dio_gap = max(f["DIO"] - bm_dio, 0.0)
        dso_gap = max(f["DSO"] - bm_dso, 0.0)
        opportunity = dio_gap * f["COGS"] / 360.0 + dso_gap * f["Revenue"] / 360.0
        o = optimise_wc(f["DIO"], f["DSO"], f["DPO"], f["COGS"], f["Revenue"],
                        bm_dio, bm_dso, dso_closure_pct, min_inventory_exposure,
                        interest_rate)
        rows.append({
            "Revenue case": label,
            "Revenue (₹ Mn)": f["Revenue"],
            "Benchmark-gap opportunity (₹ Mn)": opportunity,
            "Prescriptive cash release (₹ Mn)": o["total_cash_release"],
            "DIO reduction (days)": o["dio_reduction"],
            "DSO reduction (days)": o["dso_reduction"],
            "Interest saving (₹ Mn)": o["annual_interest_saved"],
            "Forecast DSO": f["DSO"], "Forecast DIO": f["DIO"],
            "Forecast DPO": f["DPO"], "Forecast CCC": f["CCC"],
        })
    return pd.DataFrame(rows)


# =========================================================
# STATUS / COLOR LOGIC
# =========================================================
def lower_is_better_status(value, benchmark):
    """For DSO, DIO, CCC — lower than benchmark is favorable."""
    if value <= benchmark:
        return "Better than Benchmark", GREEN
    gap_pct = (value - benchmark) / benchmark * 100
    if gap_pct <= 10:
        return "Slightly Above Benchmark", AMBER
    return "Improvement Required", RED


def dpo_status(value, standard_term, benchmark):
    """DPO is a funding-risk indicator once it exceeds standard trade terms —
    NOT simply 'higher is better'."""
    if value <= standard_term:
        return "Within Standard Trade Terms", GREEN
    if value <= benchmark:
        return "Above Standard Terms — Within Peer Norm", AMBER
    return "Supplier Funded Working Capital", RED

# =========================================================
# REVENUE FORECASTING ENGINE
# =========================================================

from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def revenue_forecasting_engine(revenue_df, forecast_periods=3):

    y = revenue_df["Revenue"].values
    n = len(y)

    results = []

    # ==================================
    # 1. LINEAR TREND
    # ==================================
    x = np.arange(1, n + 1)

    slope, intercept = np.polyfit(x, y, 1)

    fitted = slope * x + intercept
    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = 1 - (ss_res / ss_tot)
    mae = mean_absolute_error(y, fitted)
    rmse = np.sqrt(mean_squared_error(y, fitted))
    mape = np.mean(np.abs((y - fitted) / y)) * 100

    future_x = np.arange(n + 1, n + forecast_periods + 1)
    forecast = slope * future_x + intercept

    results.append({
    "Model": "Linear Trend",
    "MAE": mae,
    "RMSE": rmse,
    "MAPE": mape,
    "R2": r2,
    "Forecast": forecast
    })


     # ==================================
    # 2. MOVING AVERAGE
    # ==================================

    window = 3

    actual = []
    fitted = []

    for i in range(window, n):
        fitted.append(np.mean(y[i-window:i]))
        actual.append(y[i])

    actual = np.array(actual)
    fitted = np.array(fitted)
    ss_res = np.sum((actual - fitted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)

    r2 = 1 - (ss_res / ss_tot)

    mae = mean_absolute_error(actual, fitted)
    rmse = np.sqrt(mean_squared_error(actual, fitted))
    mape = np.mean(np.abs((actual - fitted) / actual)) * 100

    forecast = []
    history = list(y.copy())

    for _ in range(forecast_periods):
        next_value = np.mean(history[-window:])
        forecast.append(next_value)
        history.append(next_value)

    forecast = np.array(forecast)

    results.append({
        "Model": "Moving Average",
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "Forecast": forecast
    })

    # ==================================
    # 3. SES
    # ==================================
    ses_model = SimpleExpSmoothing(y, initialization_method="estimated").fit()

    fitted = ses_model.fittedvalues
    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = 1 - (ss_res / ss_tot)

    mae = mean_absolute_error(y, fitted)
    rmse = np.sqrt(mean_squared_error(y, fitted))
    mape = np.mean(np.abs((y - fitted) / y)) * 100

    forecast = ses_model.forecast(forecast_periods)

    results.append({
        "Model": "SES",
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "Forecast": forecast
    })

    # ==================================
    # 4. DES (HOLT)
    # ==================================
    
    des_model = Holt(y, initialization_method="estimated").fit()
  
    fitted = des_model.fittedvalues
    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = 1 - (ss_res / ss_tot)
    mae = mean_absolute_error(y, fitted)
    rmse = np.sqrt(mean_squared_error(y, fitted))
    mape = np.mean(np.abs((y - fitted) / y)) * 100

    forecast = des_model.forecast(forecast_periods)

    results.append({
        "Model": "DES",
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "Forecast": forecast
    })

    metrics_df = pd.DataFrame([
    {
        "Model": r["Model"],
        "MAE": r["MAE"],
        "RMSE": r["RMSE"],
        "MAPE (%)": r["MAPE"],
        "R²": r["R2"]
    }
    for r in results
    ])

    best_model = metrics_df.sort_values(
        by="MAPE (%)"
    ).iloc[0]["Model"]

    best_forecast = np.array(
        next(
            r["Forecast"]
            for r in results
            if r["Model"] == best_model
        )
    ).flatten()
    return metrics_df, best_model, best_forecast

# =========================================================
# SIDEBAR — DATA, BENCHMARK, FORECAST INPUT
# =========================================================
st.sidebar.markdown(f"""
<div style="background:{NAVY};color:white;padding:10px 12px;border-radius:8px;
            text-align:center;font-weight:800;font-size:14px;letter-spacing:0.5px;margin-bottom:8px;">
    CENTUM WC&nbsp;DSS
</div>
""", unsafe_allow_html=True)

st.sidebar.header("1 · Historical Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload historical financials (.xlsx / .csv)", type=["xlsx", "csv"]
)
st.sidebar.download_button(
    "⬇ Download data template (CSV)", data=make_template_csv(),
    file_name="centum_wc_data_template.csv", mime="text/csv",
)
exclude_covid = st.sidebar.checkbox(
    "Exclude FY2020-21 & FY2021-22 (COVID years)", value=True
)

if uploaded_file is not None:
    try:
        raw_df = load_uploaded(uploaded_file)
        using_demo = False
    except Exception as e:
        st.sidebar.error(str(e))
        raw_df = sample_data()
        using_demo = True
else:
    raw_df = sample_data()
    using_demo = True

if exclude_covid:
    raw_df = raw_df[~raw_df["Year"].map(is_covid_year).astype(bool)].reset_index(drop=True)

hist_df = compute_kpis(raw_df)
models = fit_models(hist_df)

# ---------------------------------------------------------
# NEW — Peer sheets. Sheet 1 (subject company) has been loaded above exactly as
# before; here the REMAINING sheets of the workbook are read as peer companies.
# ---------------------------------------------------------
peer_avg_df = pd.DataFrame(columns=["Company", "Years"] + PEER_KPI_COLS)
peer_warnings = []
subject_name = None
is_workbook = uploaded_file is not None and uploaded_file.name.lower().endswith(".xlsx")
if is_workbook:
    try:
        _pb_sheets, _pb_skipped, _pb_order = load_all_sheets(
            uploaded_file.getvalue(), uploaded_file.name
        )
        subject_name = _pb_order[0] if _pb_order else None
        _pb_peers = {n: d for n, d in _pb_sheets.items() if n != subject_name}
        peer_avg_df, _pb_skipped2 = compute_peer_averages(_pb_peers, exclude_covid)
        peer_warnings = [(n, r) for n, r in _pb_skipped if n != subject_name] + _pb_skipped2
    except Exception as _pb_err:
        st.sidebar.warning(
            f"Peer sheets could not be read — the default benchmark is used. ({_pb_err})"
        )
if peer_warnings:
    st.sidebar.warning(
        "Sheet(s) skipped: " + "; ".join(f"**{n}** — {r}" for n, r in peer_warnings)
    )



st.sidebar.header("2 · Weighted Peer Benchmark")

# --- NEW: peer-based benchmark builder. Only appears when the uploaded workbook
#     contains peer sheets; otherwise the sidebar is exactly as before. ---
peer_bm = None
peer_class = {}
if len(peer_avg_df) > 0:
    with st.sidebar.expander("🏭 Peer-based benchmark (from workbook sheets)", expanded=True):
        st.caption(
            "Tag each peer sheet as EMS-based or Design-based. The best peer (lowest CCC) "
            "of each group is weighted by your EMS : Design ratio to form the benchmark. "
            "Leave everything as 'Not used' to keep the default benchmark."
        )
        for _pb_i, _pb_r in peer_avg_df.iterrows():
            _pb_name = _pb_r["Company"]
            peer_class[_pb_name] = st.selectbox(
                f"{_pb_name} — avg CCC {_pb_r['CCC']:.1f}d ({int(_pb_r['Years'])} yrs)",
                ["Not used", "EMS", "Design"], key=f"peer_class_{_pb_name}",
            )
        ems_weight_pct = st.slider(
            "EMS weight (%) — Design gets the remainder",
            min_value=0, max_value=100, value=70, step=5, key="peer_ems_weight",
        )
        peer_basis = st.radio(
            "Benchmark basis within each category",
            [BEST_PEER_BASIS, AVG_PEER_BASIS],
            key="peer_basis",
        )
        peer_bm = derive_peer_benchmark(peer_avg_df, peer_class, ems_weight_pct, peer_basis)
        if peer_bm is None:
            st.info("No peer tagged yet — the default benchmark values below are in use.")
        else:
            _pb_f = peer_bm["final"]
            st.success(
                f"{peer_bm['summary']}\n\n"
                f"→ DSO {_pb_f['DSO']:.2f} · DIO {_pb_f['DIO']:.2f} · "
                f"DPO {_pb_f['DPO']:.2f} · CCC {_pb_f['CCC']:.2f}"
            )
elif is_workbook:
    st.sidebar.caption(
        "Tip: add peer companies as extra sheets (sheet 1 = subject company) to build a peer-based benchmark."
    )

# Benchmark inputs: default values are unchanged (119.91 / 210.75 / 99.73). When a
# peer-based benchmark has been built, it becomes the (still editable) default.
if peer_bm is not None:
    _pb_default = {m: round(peer_bm["final"][m], 2) for m in ("DSO", "DIO", "DPO")}
    _pb_sig = f"peer_{_pb_default['DSO']}_{_pb_default['DIO']}_{_pb_default['DPO']}"
else:
    _pb_default = {"DSO": 119.91, "DIO": 210.75, "DPO": 99.73}
    _pb_sig = "default"
bm_dso = st.sidebar.number_input("Benchmark DSO (days)", value=_pb_default["DSO"], step=0.1, key=f"bm_dso_{_pb_sig}")
bm_dio = st.sidebar.number_input("Benchmark DIO (days)", value=_pb_default["DIO"], step=0.1, key=f"bm_dio_{_pb_sig}")
bm_dpo = st.sidebar.number_input("Benchmark DPO (days)", value=_pb_default["DPO"], step=0.1, key=f"bm_dpo_{_pb_sig}")
bm_ccc = bm_dso + bm_dio - bm_dpo
_pb_edited = (abs(bm_dso - _pb_default["DSO"]) > 0.005 or abs(bm_dio - _pb_default["DIO"]) > 0.005
              or abs(bm_dpo - _pb_default["DPO"]) > 0.005)
_pb_source = ("peer-based" if peer_bm is not None else "default") + (" (manually adjusted)" if _pb_edited else "")
st.sidebar.caption(f"Benchmark source: {_pb_source} · Benchmark CCC = {bm_ccc:.2f} days")
standard_term = st.sidebar.number_input("Standard Trade Term (days)", value=90.0, step=1.0)

# --- Cost of Funds / interest rate — linked with the "Annual Interest
#     Rate (%)" control on the Scenario Analysis tab (see _sync_rate_*).
interest_rate = st.sidebar.slider(
    "Cost of Funds (%)",
    min_value=0.0,
    max_value=50.0,
    step=0.5,
    key="side_interest_rate",
    on_change=_sync_rate_from_side,
)
dso_closure_pct = st.sidebar.slider(
    "DSO Gap Closure (%)",
    min_value=5,
    max_value=100,
    value=25,
    step=5
)

min_inventory_exposure = st.sidebar.slider(
    "Minimum Net Inventory Exposure (Days)",
    min_value=30,
    max_value=180,
    value=90,
    step=5
)

# --- Scenario DIO/DSO reduction — linked with the identical controls
#     on the Scenario Analysis tab (see _sync_dio_*/_sync_dso_*).
scenario_dio_reduction = st.sidebar.number_input(
    "Scenario DIO Reduction (Days)",
    min_value=0.0,
    max_value=360.0,
    step=1.0,
    key="side_dio_reduction",
    on_change=_sync_dio_from_side,
)

scenario_dso_reduction = st.sidebar.number_input(
    "Scenario DSO Reduction (Days)",
    min_value=0.0,
    max_value=360.0,
    step=1.0,
    key="side_dso_reduction",
    on_change=_sync_dso_from_side,
)
# =========================================================
# REVENUE FORECAST
# =========================================================

st.sidebar.header("3 · Revenue Forecast")

forecast_mode = st.sidebar.radio(
    "Select Forecast Method",
    [
        "Automatic (Best Model)",
        "Manual Input"
    ]
)

if forecast_mode == "Automatic (Best Model)":

    revenue_df = hist_df[["Year", "Revenue"]]

    metrics_df, best_model, best_forecast = revenue_forecasting_engine(
        revenue_df,
        forecast_periods=3
    )

    revenue_forecast_df = pd.DataFrame({
        "Year": [
            "FY2026-27",
            "FY2027-28",
            "FY2028-29"
        ],
        "Revenue": np.array(best_forecast).flatten()
    })

    st.sidebar.success(
        f"Selected Model: {best_model}"
    )

else:

    revenue_forecast_df = st.sidebar.data_editor(
        pd.DataFrame({
            "Year": [
                "FY2026-27",
                "FY2027-28",
                "FY2028-29"
            ],
            "Revenue": [
                8965.82,
                9665.19,
                10364.56
            ]
        }),
        num_rows="dynamic",
        hide_index=True
    )

    metrics_df = pd.DataFrame()

    best_model = "Manual Revenue Input"


# =========================================================
# COMPUTE FORECASTS
# =========================================================

forecast_rows = []

for _, row in revenue_forecast_df.iterrows():

    if pd.isna(row["Revenue"]):
        continue

    res = forecast_from_revenue(
        float(row["Revenue"]),
        models
    )

    res["Year"] = row["Year"]

    forecast_rows.append(res)

forecast_df = pd.DataFrame(forecast_rows)

if not forecast_df.empty:

    forecast_df["Net Working Capital"] = (
        forecast_df["Inventory"]
        + forecast_df["Receivables"]
        - forecast_df["Payables"]
    )

    forecast_df = forecast_df[
        [
            "Year",
            "Revenue",
            "COGS",
            "Inventory",
            "Receivables",
            "Payables",
            "Net Working Capital",
            "DSO",
            "DIO",
            "DPO",
            "CCC"
        ]
    ]

# =========================================================
# HEADER
# =========================================================
st.markdown(f"""
<div style="background:{NAVY};padding:22px 28px;border-radius:10px;margin-bottom:18px;">
    <div style="color:{GOLD};font-size:11px;font-weight:700;letter-spacing:2px;">
        WORKING CAPITAL OPTIMIZATION — DECISION SUPPORT SYSTEM
    </div>
    <div style="color:white;font-size:26px;font-weight:800;margin-top:4px;">
        Centum Electronics Ltd. — Cash Conversion Cycle Analytics Dashboard
    </div>
    <div style="color:#C9D6E8;font-size:13px;margin-top:4px;">
        Historical financial data is required to establish driver-based regression relationships. Once available, Revenue may be forecast using either automated forecasting models or manual user inputs. Based on these historical relationships, COGS, Receivables, Inventory, Payables, and all working capital KPIs are automatically computed through driver-based regression models.
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab_inv, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Historical", "🔮 Forecast & KPIs", "📦 Inventory Deep-Dive",
    "🎯 Benchmark & Gap", "💰 Cash Release", "⚙️ Model Diagnostics", "🚀 Optimization", "📈 Scenario Analysis"
])

# =========================================================
# TAB 1 — CURRENT DATA (HISTORICAL, AS PER DISSERTATION)
# =========================================================
with tab1:
    st.subheader("Historical Financials & Calculated KPIs")
    st.dataframe(
        hist_df.style.format({
            "Revenue": "{:,.2f}", "COGS": "{:,.2f}", "Inventory": "{:,.2f}",
            "Receivables": "{:,.2f}", "Payables": "{:,.2f}",
            "DSO": "{:.2f}", "DIO": "{:.2f}", "DPO": "{:.2f}", "CCC": "{:.2f}",
        }),
        use_container_width=True, height=340,
    )

    latest = hist_df.iloc[-1]
    st.markdown(f"#### Latest Year Snapshot — {latest['Year']}")
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in zip(
        [c1, c2, c3, c4], ["DSO", "DIO", "DPO", "CCC"],
        [latest["DSO"], latest["DIO"], latest["DPO"], latest["CCC"]]
    ):
        with col:
            kpi_card(f"{label} (Days)", f"{val:.1f}", "Latest historical value", STEEL)

    st.markdown("#### Historical Trend")
    fig = go.Figure()
    for col, color in zip(["DSO", "DIO", "DPO", "CCC"], [STEEL, GOLD, GREEN, NAVY]):
        fig.add_trace(go.Scatter(x=hist_df["Year"], y=hist_df[col], mode="lines+markers", name=col, line=dict(color=color, width=3)))
    fig.update_layout(height=420, plot_bgcolor="white", legend=dict(orientation="h", y=-0.2),
                       yaxis_title="Days", margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 2 — FORECAST & KPIs (AUTO-COMPUTED FROM REVENUE INPUT)
# =========================================================
with tab2:
    if forecast_df.empty:
        st.info("Enter at least one forecasted Revenue value in the sidebar to see results.")
    else:
        _fc_src = (f"{best_model} Revenue Forecast" if forecast_mode == "Automatic (Best Model)"
                   else "Manual Revenue Input")
        st.subheader(f"Auto-Computed Forecast (from {_fc_src})")
        st.dataframe(
            forecast_df.style.format({
                "Revenue": "{:,.2f}", "COGS": "{:,.2f}", "Inventory": "{:,.2f}",
                "Receivables": "{:,.2f}", "Payables": "{:,.2f}", "Net Working Capital": "{:,.2f}",
                "DSO": "{:.2f}", "DIO": "{:.2f}", "DPO": "{:.2f}", "CCC": "{:.2f}",
            }),
            use_container_width=True,
        )

        sel_year = st.selectbox("Select forecast year for KPI detail:", forecast_df["Year"].tolist(),
                                 index=0)
        row = forecast_df[forecast_df["Year"] == sel_year].iloc[0]
        # ---------------- Row 1: efficiency metrics ----------------
        st.markdown(f"#### KPI Status — {sel_year}")
        c1, c2, c3, c4 = st.columns(4)

        dso_label, dso_color = lower_is_better_status(row["DSO"], bm_dso)
        dio_label, dio_color = lower_is_better_status(row["DIO"], bm_dio)
        dpo_label, dpo_color = dpo_status(row["DPO"], standard_term, bm_dpo)
        ccc_label, ccc_color = lower_is_better_status(row["CCC"], bm_ccc)

        with c1:
            kpi_card("DSO (Days)", f"{row['DSO']:.1f}", f"Benchmark: {bm_dso:.1f}", dso_color)
        with c2:
            kpi_card("DIO (Days)", f"{row['DIO']:.1f}", f"Benchmark: {bm_dio:.1f}", dio_color)
        with c3:
            kpi_card("DPO (Days)", f"{row['DPO']:.1f}", f"Standard: {standard_term:.0f} · Benchmark: {bm_dpo:.1f}", dpo_color)
        with c4:
            kpi_card("CCC (Days)", f"{row['CCC']:.1f}", f"Benchmark: {bm_ccc:.1f}", ccc_color)

        # ---------------- Row 2: efficiency metrics ----------------
        st.markdown("#### Forecasted Working Capital")

       
        c1, c2, c3 = st.columns(3)

        with c1:
            kpi_card(
                "Receivables",
                f"₹{row['Receivables']:,.2f} Mn",
                "Forecast",
                STEEL
            )

        with c2:
            kpi_card(
                "Inventory",
                f"₹{row['Inventory']:,.2f} Mn",
                "Forecast",
                GOLD
            )

        with c3:
            kpi_card(
                "Net Working Capital",
                f"₹{row['Net Working Capital']:,.2f} Mn",
                "Inventory + Receivables - Payables",
                NAVY
            )


        # ---------------- Row 3: efficiency metrics ----------------
        c1, c2, c3, c4 = st.columns(4)
        with c1: status_pill(dso_label, dso_color)
        with c2: status_pill(dio_label, dio_color)
        with c3: status_pill(dpo_label, dpo_color)
        with c4: status_pill(ccc_label, ccc_color)

        st.markdown("####  ")
        st.markdown("#### Historical + Forecast Trend")
        fig2 = go.Figure()
        combined_years = list(hist_df["Year"]) + list(forecast_df["Year"])
        for col, color in zip(["DSO", "DIO", "DPO", "CCC"], [STEEL, GOLD, GREEN, NAVY]):
            y_hist = list(hist_df[col])
            y_fc = list(forecast_df[col])
            fig2.add_trace(go.Scatter(x=hist_df["Year"], y=y_hist, mode="lines+markers", name=f"{col} (Actual)", line=dict(color=color, width=3)))
            fig2.add_trace(go.Scatter(
                x=[hist_df["Year"].iloc[-1]] + list(forecast_df["Year"]),
                y=[y_hist[-1]] + y_fc, mode="lines+markers", name=f"{col} (Forecast)",
                line=dict(color=color, width=3, dash="dash"), showlegend=False,
            ))
        fig2.update_layout(height=440, plot_bgcolor="white", legend=dict(orientation="h", y=-0.2),
                            yaxis_title="Days", margin=dict(t=20))
        st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# TAB  — INVENTORY DEEP-DIVE
# =========================================================
with tab_inv:
    if forecast_df.empty:
        st.info("Enter a forecasted Revenue value in the sidebar to see the inventory analysis.")
    else:
        sel_year_inv = st.selectbox("Forecast year:", forecast_df["Year"].tolist(),
                                     index=0, key="inv_year")
        row = forecast_df[forecast_df["Year"] == sel_year_inv].iloc[0]

        st.markdown("Inventory is the largest single driver of the working capital gap. "
                     "This page shows how much inventory *should* be held for the forecasted "
                     "business volume, versus how much the current trajectory implies.")

        # ---------------- Target DIO control ----------------
        c_target, c_info = st.columns([1, 2])
        with c_target:
            target_dio = st.slider(
                "Target DIO (days) — default is benchmark", min_value=float(standard_term),
                max_value=float(max(row["DIO"], bm_dio) + 20), value=float(bm_dio), step=0.5,
            )
        with c_info:
            st.caption(
                f"Default target = weighted peer benchmark DIO ({bm_dio:.1f} days). "
                f"Drag to test a more aggressive or more conservative inventory target — "
                f"every metric below recalculates instantly."
            )

        # ---------------- Core calculations ----------------
        daily_cogs = row["COGS"] / 360
        target_inventory = target_dio / 360 * row["COGS"]
        excess_inventory = row["Inventory"] - target_inventory
        excess_days = row["DIO"] - target_dio
        turnover_actual = row["COGS"] / row["Inventory"]
        turnover_target = 360 / target_dio
        inv_pct_revenue = row["Inventory"] / row["Revenue"] * 100
        inv_pct_cogs = row["Inventory"] / row["COGS"] * 100
        weeks_cover = row["DIO"] / 7
        months_cover = row["DIO"] / 30.44
        efficiency_index = target_dio / row["DIO"] * 100  # 100% = at target; <100% = holding excess
        cogs_pct_revenue = row["COGS"] / row["Revenue"] * 100

        # ---------------- Row 1: headline KPIs ----------------
        st.markdown(f"#### Inventory Position — {sel_year_inv}")
        c1, c2, c3 = st.columns(3)
        with c1:
            kpi_card("Forecasted Inventory (BAU)", f"₹{row['Inventory']:,.2f} Mn",
                     "Regression-based, business-as-usual trajectory", NAVY)
        with c2:
            kpi_card("Target Inventory", f"₹{target_inventory:,.2f} Mn",
                     f"At {target_dio:.1f}-day DIO target", STEEL)
        with c3:
            excess_color = RED if excess_inventory > 0 else GREEN
            excess_label = "Excess vs. Target" if excess_inventory > 0 else "Already Below Target"
            kpi_card(excess_label, f"₹{abs(excess_inventory):,.2f} Mn",
                     f"{excess_days:+.1f} days vs. target", excess_color)

        # ---------------- Row 2: efficiency metrics ----------------
        st.markdown("#### Efficiency Metrics")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            dio_label, dio_color = lower_is_better_status(row["DIO"], target_dio)
            kpi_card("DIO (Days)", f"{row['DIO']:.1f}", f"Target: {target_dio:.1f}", dio_color)
        with c2:
            turn_color = GREEN if turnover_actual >= turnover_target else RED
            kpi_card("Inventory Turnover", f"{turnover_actual:.2f}×",
                     f"Target: {turnover_target:.2f}× per year", turn_color)
        with c3:
            eff_color = GREEN if efficiency_index >= 95 else (AMBER if efficiency_index >= 80 else RED)
            kpi_card("Inventory Efficiency Index", f"{efficiency_index:.0f}%",
                     "100% = at target DIO", eff_color)
        with c4:
            kpi_card("Weeks / Months of Cover", f"{weeks_cover:.1f} wks",
                     f"≈ {months_cover:.1f} months", STEEL)

        st.markdown("#### Inventory as a Share of the Business")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Inventory % of Revenue", f"{inv_pct_revenue:.1f}%",
                     "Inventory ÷ Forecasted Revenue", NAVY)
        with c2:
            kpi_card("Inventory % of COGS", f"{inv_pct_cogs:.1f}%",
                     "Inventory ÷ Forecasted COGS", STEEL)
        with c3:
            kpi_card("Potential Cash Release", f"₹{max(excess_inventory,0):,.2f} Mn",
                     "If inventory is brought to target level", NAVY)
        with c4:
            kpi_card("COGS % of Revenue", f"{cogs_pct_revenue:.1f}%",
                     "Cost of goods sold as % of revenue", GOLD)

        # ---------------- Chart 1: BAU vs Target Inventory, all forecast years ----------------
        st.markdown("#### Forecasted Inventory vs. Target — All Forecast Years")
        inv_chart_df = forecast_df.copy()
        inv_chart_df["Target Inventory"] = target_dio / 360 * inv_chart_df["COGS"]
        inv_chart_df["Excess"] = inv_chart_df["Inventory"] - inv_chart_df["Target Inventory"]

        fig_inv1 = go.Figure()
        fig_inv1.add_trace(go.Bar(name="Forecasted Inventory (BAU)", x=inv_chart_df["Year"],
                                   y=inv_chart_df["Inventory"], marker_color=NAVY))
        fig_inv1.add_trace(go.Bar(name="Target Inventory", x=inv_chart_df["Year"],
                                   y=inv_chart_df["Target Inventory"], marker_color=GOLD))
        fig_inv1.update_layout(barmode="group", height=400, plot_bgcolor="white",
                                yaxis_title="₹ Mn", margin=dict(t=20),
                                legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_inv1, use_container_width=True)

        st.dataframe(
            inv_chart_df[["Year", "COGS", "Inventory", "Target Inventory", "Excess", "DIO"]]
            .rename(columns={"Inventory": "Forecasted Inventory (BAU)"})
            .style.format({
                "COGS": "{:,.2f}", "Forecasted Inventory (BAU)": "{:,.2f}",
                "Target Inventory": "{:,.2f}", "Excess": "{:+,.2f}", "DIO": "{:.2f}",
            }),
            use_container_width=True,
        )

        # ---------------- Chart 2: DIO trend with target line ----------------
        st.markdown("#### DIO Trend — Historical, Forecast & Target")
        fig_inv2 = go.Figure()
        fig_inv2.add_trace(go.Scatter(x=hist_df["Year"], y=hist_df["DIO"], mode="lines+markers",
                                       name="DIO (Actual)", line=dict(color=NAVY, width=3)))
        fig_inv2.add_trace(go.Scatter(
            x=[hist_df["Year"].iloc[-1]] + list(forecast_df["Year"]),
            y=[hist_df["DIO"].iloc[-1]] + list(forecast_df["DIO"]),
            mode="lines+markers", name="DIO (Forecast, BAU)",
            line=dict(color=NAVY, width=3, dash="dash"),
        ))
        fig_inv2.add_hline(y=bm_dio, line_dash="dot", line_color=GOLD,
                            annotation_text=f"Benchmark ({bm_dio:.1f}d)", annotation_position="top left")
        if abs(target_dio - bm_dio) > 0.01:
            fig_inv2.add_hline(y=target_dio, line_dash="dot", line_color=GREEN,
                                annotation_text=f"Target ({target_dio:.1f}d)", annotation_position="bottom left")
        fig_inv2.update_layout(height=400, plot_bgcolor="white", yaxis_title="Days", margin=dict(t=20),
                                legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_inv2, use_container_width=True)

        # ---------------- Phased glide path to target ----------------
        st.markdown("#### Phased Glide Path to Target")
        st.caption("Assumes a straight-line reduction from the current DIO to the target DIO, "
                    "spread evenly across the entered forecast years.")
        n_years = len(forecast_df)
        start_dio = hist_df["DIO"].iloc[-1]
        glide_rows = []
        for i, (_, fr) in enumerate(forecast_df.iterrows()):
            glide_dio = start_dio + (target_dio - start_dio) * (i + 1) / n_years
            glide_inventory = glide_dio / 360 * fr["COGS"]
            bau_inventory = fr["Inventory"]
            incremental_release = bau_inventory - glide_inventory
            glide_rows.append({
                "Year": fr["Year"], "BAU DIO": fr["DIO"], "Glide-Path Target DIO": glide_dio,
                "BAU Inventory": bau_inventory, "Glide-Path Inventory": glide_inventory,
                "Cumulative Cash Released vs. BAU": incremental_release,
            })
        glide_df = pd.DataFrame(glide_rows)
        st.dataframe(
            glide_df.style.format({
                "BAU DIO": "{:.1f}", "Glide-Path Target DIO": "{:.1f}",
                "BAU Inventory": "{:,.2f}", "Glide-Path Inventory": "{:,.2f}",
                "Cumulative Cash Released vs. BAU": "{:+,.2f}",
            }),
            use_container_width=True,
        )

        fig_glide = go.Figure()
        fig_glide.add_trace(go.Bar(name="BAU Inventory", x=glide_df["Year"], y=glide_df["BAU Inventory"], marker_color=STEEL))
        fig_glide.add_trace(go.Bar(name="Glide-Path Inventory", x=glide_df["Year"], y=glide_df["Glide-Path Inventory"], marker_color=GREEN))
        fig_glide.update_layout(barmode="group", height=380, plot_bgcolor="white", yaxis_title="₹ Mn",
                                 margin=dict(t=20), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_glide, use_container_width=True)

        _final_year = glide_df["Year"].iloc[-1]
        _net_release = glide_df["Cumulative Cash Released vs. BAU"].iloc[-1]
        if target_dio >= start_dio:
            _reco = (f"The latest actual DIO ({start_dio:.1f} days) is already at or below the target "
                     f"({target_dio:.1f} days), so no phased DIO reduction is needed.")
        else:
            _reco = (f"Reduce DIO from {start_dio:.1f} days toward {target_dio:.1f} days in equal steps over "
                     f"{n_years} year(s) via tighter procurement planning and demand-aligned inventory control — "
                     f"by {_final_year} this frees ₹{_net_release:,.2f} Mn of cash relative to the "
                     f"business-as-usual trajectory, without a disruptive one-time inventory correction.")
        st.markdown(f"""
        <div style="background:{GREEN}18;border-left:5px solid {GREEN};padding:14px 18px;border-radius:6px;margin-top:6px;">
        <b style="color:{GREEN};">Recommendation:</b> {_reco}
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# TAB 3 — BENCHMARK & GAP ANALYSIS
# =========================================================
with tab3:
    # ---- NEW: peer benchmarking (only when the uploaded workbook has peer sheets) ----
    if len(peer_avg_df) > 0:
        with st.expander("🏭 Peer Benchmarking — Historical Averages by Company", expanded=True):
            _pb_rows = []
            if (not using_demo) and subject_name:
                _pb_rows.append({
                    "Company": subject_name, "Role": "Subject", "Years": int(len(hist_df)),
                    "Avg DSO": float(hist_df["DSO"].mean()), "Avg DIO": float(hist_df["DIO"].mean()),
                    "Avg DPO": float(hist_df["DPO"].mean()), "Avg CCC": float(hist_df["CCC"].mean()),
                    "Benchmark use": "— (subject company)",
                })
            for _pb_i, _pb_r in peer_avg_df.iterrows():
                _pb_role = peer_class.get(_pb_r["Company"], "Not used")
                _pb_use = "Not used"
                if peer_bm is not None and _pb_role in ("EMS", "Design"):
                    _pb_grp = peer_bm["ems"] if _pb_role == "EMS" else peer_bm["design"]
                    if _pb_grp is not None:
                        if _pb_grp["picked"] is not None:
                            _pb_use = ("★ Category benchmark" if _pb_r["Company"] == _pb_grp["picked"]
                                       else "Considered — not the best")
                        else:
                            _pb_use = "Averaged into category"
                _pb_rows.append({
                    "Company": _pb_r["Company"], "Role": _pb_role, "Years": int(_pb_r["Years"]),
                    "Avg DSO": _pb_r["DSO"], "Avg DIO": _pb_r["DIO"],
                    "Avg DPO": _pb_r["DPO"], "Avg CCC": _pb_r["CCC"],
                    "Benchmark use": _pb_use,
                })
            _pb_tbl = pd.DataFrame(_pb_rows).set_index("Company")

            def _pb_highlight(_pb_row):
                if _pb_row["Role"] == "Subject":
                    return [f"background-color:{ICE};font-weight:700"] * len(_pb_row)
                if str(_pb_row["Benchmark use"]).startswith("★"):
                    return ["background-color:#FBF1DA;font-weight:700"] * len(_pb_row)
                return [""] * len(_pb_row)

            st.caption("Historical average of the annual ratios for every sheet in the workbook "
                       "(COVID-year exclusion follows the sidebar setting).")
            st.dataframe(
                _pb_tbl.style.apply(_pb_highlight, axis=1).format({
                    "Avg DSO": "{:.2f}", "Avg DIO": "{:.2f}", "Avg DPO": "{:.2f}", "Avg CCC": "{:.2f}",
                }),
                use_container_width=True,
            )

            if peer_bm is None:
                st.info("Tag the peers as EMS or Design in the sidebar (section 2) to build a peer-based "
                        "benchmark. Until then the default benchmark values are used.")
            else:
                _pb_der = []
                for _pb_lab, _pb_key, _pb_w in [("EMS benchmark", "ems", peer_bm["w_ems"]),
                                                ("Design benchmark", "design", peer_bm["w_design"])]:
                    _pb_g = peer_bm[_pb_key]
                    if _pb_g is not None:
                        _pb_der.append({
                            "Component": _pb_lab, "Weight": f"{_pb_w * 100:.0f}%", "Source": _pb_g["label"],
                            "DSO": _pb_g["values"]["DSO"], "DIO": _pb_g["values"]["DIO"],
                            "DPO": _pb_g["values"]["DPO"], "CCC": _pb_g["values"]["CCC"],
                        })
                _pb_fin = peer_bm["final"]
                _pb_der.append({
                    "Component": "Final weighted benchmark", "Weight": "100%", "Source": peer_bm["basis"],
                    "DSO": _pb_fin["DSO"], "DIO": _pb_fin["DIO"], "DPO": _pb_fin["DPO"], "CCC": _pb_fin["CCC"],
                })
                st.markdown("**How the benchmark was decided**")
                st.dataframe(
                    pd.DataFrame(_pb_der).set_index("Component").style.format({
                        "DSO": "{:.2f}", "DIO": "{:.2f}", "DPO": "{:.2f}", "CCC": "{:.2f}",
                    }),
                    use_container_width=True,
                )
                st.caption("The final weighted benchmark fills the benchmark inputs in the sidebar and is used "
                           "for every comparison, cash-release and optimisation calculation in this dashboard.")

    if forecast_df.empty:
        st.info("Enter a forecasted Revenue value in the sidebar to see the benchmark comparison.")
    else:
        sel_year3 = st.selectbox("Forecast year:", forecast_df["Year"].tolist(),
                                  index=0, key="gap_year")
        row = forecast_df[forecast_df["Year"] == sel_year3].iloc[0]

        gap_df = pd.DataFrame({
            "Metric": ["DSO", "DIO", "DPO", "CCC"],
            "Forecast": [row["DSO"], row["DIO"], row["DPO"], row["CCC"]],
            "Benchmark": [bm_dso, bm_dio, bm_dpo, bm_ccc],
        })
        gap_df["Gap (Days)"] = gap_df["Forecast"] - gap_df["Benchmark"]
        gap_df["Gap %"] = gap_df["Gap (Days)"] / gap_df["Benchmark"] * 100

        def row_status(m, gap):
            if m == "DPO":
                return dpo_status(row["DPO"], standard_term, bm_dpo)[0]
            return "Better than Benchmark" if gap <= 0 else "Improvement Required"

        gap_df["Status"] = [row_status(m, g) for m, g in zip(gap_df["Metric"], gap_df["Gap (Days)"])]

        st.subheader(f"Forecast vs. Weighted Benchmark — {sel_year3}")
        st.dataframe(
            gap_df.style.format({"Forecast": "{:.2f}", "Benchmark": "{:.2f}", "Gap (Days)": "{:+.2f}", "Gap %": "{:+.1f}%"}),
            use_container_width=True,
        )

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name="Forecast", x=gap_df["Metric"], y=gap_df["Forecast"], marker_color=NAVY))
        fig3.add_trace(go.Bar(name="Benchmark", x=gap_df["Metric"], y=gap_df["Benchmark"], marker_color=GOLD))
        fig3.update_layout(barmode="group", height=420, plot_bgcolor="white", yaxis_title="Days", margin=dict(t=20))
        st.plotly_chart(fig3, use_container_width=True)

# =========================================================
# TAB 4 — CASH RELEASE OPPORTUNITY (DURABLE vs SUPPLIER-FUNDED)
# =========================================================
with tab4:
    if forecast_df.empty:
        st.info("Enter a forecasted Revenue value in the sidebar to see the cash release analysis.")
    else:
        sel_year4 = st.selectbox("Forecast year:", forecast_df["Year"].tolist(),
                                  index=0, key="cash_year")
        row = forecast_df[forecast_df["Year"] == sel_year4].iloc[0]

        daily_cogs = row["COGS"] / 360
        daily_rev = row["Revenue"] / 360

        dio_gap = max(row["DIO"] - bm_dio, 0)
        dso_gap = max(row["DSO"] - bm_dso, 0)
        dpo_gap_bm = max(row["DPO"] - bm_dpo, 0)
        dpo_gap_std = max(row["DPO"] - standard_term, 0)

        cash_dio = dio_gap * daily_cogs
        cash_dso = dso_gap * daily_rev
        total_identified = cash_dio + cash_dso
        supplier_funded = dpo_gap_bm * daily_cogs
        durable_gain = total_identified - supplier_funded
        funding_risk_std = dpo_gap_std * daily_cogs

        inventory_interest_saving = (
            cash_dio * interest_rate / 100
        )

        receivables_interest_saving = (
            cash_dso * interest_rate / 100
        )

        total_interest_saving = (
            total_identified * interest_rate / 100
        )

        st.subheader(f"Cash Release Opportunity — {sel_year4}")
        c1, c2, c3 = st.columns(3)
        with c1:
            kpi_card("Inventory (DIO) Opportunity", f"₹{cash_dio:,.2f} Mn", f"{dio_gap:.2f} days vs. benchmark", NAVY)
        with c2:
            kpi_card("Receivables (DSO) Opportunity", f"₹{cash_dso:,.2f} Mn", f"{dso_gap:.2f} days vs. benchmark", NAVY)
        with c3:
            kpi_card("Total Identified Opportunity", f"₹{total_identified:,.2f} Mn", f"{dio_gap + dso_gap:.2f} days combined", GOLD)

        st.markdown("#### Annual Interest Savings")

        c1, c2, c3 = st.columns(3)

        with c1:
            kpi_card(
                "Inventory Interest Saving",
                f"₹{inventory_interest_saving:,.2f} Mn",
                f"At {interest_rate:.1f}% Cost of Funds",
                GREEN
            )

        with c2:
            kpi_card(
                "Receivables Interest Saving",
                f"₹{receivables_interest_saving:,.2f} Mn",
                f"At {interest_rate:.1f}% Cost of Funds",
                GREEN
            )

        with c3:
            kpi_card(
                "Total Interest Saving",
                f"₹{total_interest_saving:,.2f} Mn",
                f"Annual Saving at {interest_rate:.1f}%",
                NAVY
            )


        st.markdown("#### Durable Gain vs. Supplier-Funded Cushion")
        c1, c2 = st.columns(2)
        with c1:
            kpi_card("Durable Efficiency Gain", f"₹{durable_gain:,.2f} Mn",
                     "Genuine CCC-gap improvement vs. benchmark", GREEN)
        with c2:
            kpi_card("Currently Supplier-Funded", f"₹{supplier_funded:,.2f} Mn",
                     "Sustained only because DPO exceeds benchmark", RED)

        st.markdown(f"""
        <div style="background:{RED}18;border-left:5px solid {RED};padding:14px 18px;border-radius:6px;margin-top:6px;">
        <b style="color:{RED};">Risk statement:</b> If supplier terms tighten toward standard trade terms
        ({standard_term:.0f} days), an additional <b>₹{funding_risk_std:,.2f} Mn</b> of working capital funding
        would be required — with no change to DIO or DSO.
        </div>
        """, unsafe_allow_html=True)

        fig4 = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative", "relative", "total"],
            x=["Durable Efficiency Gain", "Supplier-Funded Cushion", "Total Identified Opportunity"],
            y=[durable_gain, supplier_funded, 0],
            text=[f"₹{durable_gain:,.0f} Mn", f"₹{supplier_funded:,.0f} Mn", f"₹{total_identified:,.0f} Mn"],
            textposition="outside",
            connector={"line": {"color": GREY}},
            decreasing={"marker": {"color": RED}},
            increasing={"marker": {"color": GREEN}},
            totals={"marker": {"color": NAVY}},
        ))
        fig4.update_layout(height=420, plot_bgcolor="white", margin=dict(t=20), yaxis_title="₹ Mn")
        st.plotly_chart(fig4, use_container_width=True)

# =========================================================
# TAB 5 — MODEL DIAGNOSTICS (TRANSPARENCY ON THE "AUTO" PART)
# =========================================================
with tab5:
    st.subheader("Driver-Based Regression Models (Fitted on Historical Data)")

    st.caption(
        "These equations are what automatically compute COGS, "
        "Receivables, Inventory and Payables once Revenue is entered."
    )

    diag_rows = [
        ("COGS = f(Revenue)", models["cogs_on_revenue"]),
        ("Receivables = f(Revenue)", models["receivables_on_revenue"]),
        ("Inventory = f(COGS)", models["inventory_on_cogs"]),
        ("Payables = f(COGS)", models["payables_on_cogs"]),
    ]

    diag_df = pd.DataFrame([
        {
            "Relationship": name,
            "Equation": f"y = {m['slope']:.4f}x + {m['intercept']:,.2f}",
            "R²": round(m["r2"], 4),
            "MAE": round(m["mae"], 2),
            "MAPE (%)": round(m["mape"], 2),
            "RMSE": round(m["rmse"], 2)
        }
        for name, m in diag_rows
    ])

    st.dataframe(
    diag_df.style.format({
        "R²": "{:.4f}",
        "MAE": "{:,.2f}",
        "MAPE (%)": "{:.2f}",
        "RMSE": "{:,.2f}"
    }),
    use_container_width=True
    )

    st.caption(
        f"Fitted on {len(hist_df)} historical annual observations"
        f"{' (COVID years excluded)' if exclude_covid else ''}."
    )

    st.markdown("---")

    st.subheader("Revenue Forecast Model Comparison")

    st.dataframe(
        metrics_df.style.format({
            "MAE": "{:,.2f}",
            "RMSE": "{:,.2f}",
            "MAPE (%)": "{:.2f}"
        }),
        use_container_width=True
    )

    st.success(
        f"Selected Revenue Forecast Model: {best_model}"
    )

# =========================================================
# TAB 6 — PRESCRIPTIVE OPTIMIZATION
# =========================================================
with tab6:

    if forecast_df.empty:

        st.info(
            "Enter a forecasted Revenue value in the sidebar to run the optimization model."
        )

    else:

        sel_year6 = st.selectbox(
            "Forecast year:",
            forecast_df["Year"].tolist(),
           # index=len(forecast_df) - 1,
            index=0,
            key="opt_year"
        )

        row = forecast_df[
            forecast_df["Year"] == sel_year6
        ].iloc[0]

        current_dio = row["DIO"]
        current_dso = row["DSO"]
        current_dpo = row["DPO"]
        current_ccc = row["CCC"]

        forecast_cogs = row["COGS"]
        forecast_revenue = row["Revenue"]
        benchmark_dio = bm_dio
        benchmark_dso = bm_dso

        

        # ==========================================
        # PRESCRIPTIVE OPTIMISATION — linear programme (see optimise_wc)
        # Objective: maximise cash release; constraints: DIO floor, minimum
        # net inventory exposure, DSO gap-closure cap, non-negativity.
        # ==========================================
        opt = optimise_wc(
            current_dio, current_dso, current_dpo, forecast_cogs, forecast_revenue,
            bm_dio, bm_dso, dso_closure_pct, min_inventory_exposure, interest_rate,
        )
        optimal_dio_reduction = opt["dio_reduction"]
        optimal_dso_reduction = opt["dso_reduction"]
        target_dio = opt["target_dio"]
        target_dso = opt["target_dso"]
        target_ccc = opt["target_ccc"]
        ccc_improvement = opt["ccc_improvement"]
        inventory_cash_release = opt["inventory_cash_release"]
        receivable_cash_release = opt["receivable_cash_release"]
        total_cash_release = opt["total_cash_release"]
        annual_interest_saved = opt["annual_interest_saved"]
        net_inventory_days = opt["net_inventory_days"]

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            kpi_card(
                "Forecast DSO",
                f"{current_dso:.1f}",
                "Days",
                STEEL
            )

        with c2:
            kpi_card(
                "Forecast DIO",
                f"{current_dio:.1f}",
                "Days",
                GOLD
            )

        with c3:
            kpi_card(
                "Forecast DPO",
                f"{current_dpo:.1f}",
                "Days",
                GREEN
            )

        with c4:
            kpi_card(
                "Forecast CCC",
                f"{current_ccc:.1f}",
                "Days",
                NAVY
            )


        st.subheader(
            f"Prescriptive Optimization — {sel_year6}"
        )

        st.markdown("#### Optimization Constraints")

        constraint_df = pd.DataFrame({
            "Constraint": [
                "Minimum DIO",
                "Maximum DSO Gap Closure",
                "Minimum Net Inventory Exposure",
                "Cost of Funds"
            ],
            "Value": [
                f"{bm_dio:.1f} Days",
                f"{dso_closure_pct:.0f}%",
                f"{min_inventory_exposure:.0f} Days",
                f"{interest_rate:.1f}%"
            ]
        })

        st.dataframe(
            constraint_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("#### Optimization Results")

        opt1, opt2, opt3 = st.columns(3)

        with opt1:
            kpi_card(
                "Optimized DSO",
                f"{target_dso:.1f}",
                f"Forecast: {current_dso:.1f}",
                STEEL
            )

        with opt2:
            kpi_card(
                "Optimized DIO",
                f"{target_dio:.1f}",
                f"Forecast: {current_dio:.1f}",
                GOLD
            )

        with opt3:
            kpi_card(
                "Optimized CCC",
                f"{target_ccc:.1f}",
                f"Forecast: {current_ccc:.1f}",
                NAVY
            )
        imp1, imp2, imp3 = st.columns(3)

        with imp1:
            kpi_card(
                "DSO Reduction",
                f"{optimal_dso_reduction:.1f} Days",
                "Optimization Benefit",
                GREEN
            )

        with imp2:
            kpi_card(
                "DIO Reduction",
                f"{optimal_dio_reduction:.1f} Days",
                "Optimization Benefit",
                GREEN
            )

        with imp3:
            kpi_card(
                "CCC Improvement",
                f"{ccc_improvement:.1f} Days",
                "Optimization Benefit",
                GREEN
            )

        fin1, fin2 = st.columns(2)

        with fin1:
            kpi_card(
                "Cash Released",
                f"₹{total_cash_release:,.2f} Mn",
                "Maximum Cash Released",
                NAVY
            )

        with fin2:
            kpi_card(
                "Interest Saving",
                f"₹{annual_interest_saved:,.2f} Mn",
                f"At {interest_rate:.1f}% Cost of Funds",
                GOLD
           )

# =========================================================
# TAB 7 — SCENARIO ANALYSIS
# =========================================================

with tab7:
    if forecast_df.empty:
        st.info("Forecast data not available.")
    else:
        # 1. Select the Forecast Year
        sel_year7 = st.selectbox(
            "Forecast Year",
            forecast_df["Year"].tolist(),
            index=0,
            key="scenario_year"
        )

        # --- Scenario controls — linked to the identical controls in the
        #     sidebar ("Scenario DIO/DSO Reduction (Days)", "Cost of Funds (%)").
        #     Changing either the sidebar or these tab controls updates both,
        #     via the on_change callbacks defined near the top of the script.
        st.markdown("### Scenario Parameters")
        col1, col2, col3 = st.columns(3)

        with col1:
            scenario_dio_reduction = st.number_input(
                "Target DIO Reduction (Days)",
                min_value=0.0, max_value=360.0, step=1.0,
                key="tab_dio_reduction",
                on_change=_sync_dio_from_tab,
            )
        with col2:
            scenario_dso_reduction = st.number_input(
                "Target DSO Reduction (Days)",
                min_value=0.0, max_value=360.0, step=1.0,
                key="tab_dso_reduction",
                on_change=_sync_dso_from_tab,
            )
        with col3:
            interest_rate = st.number_input(
                "Annual Interest Rate (%)",
                min_value=0.0, max_value=50.0, step=0.5,
                key="tab_interest_rate",
                on_change=_sync_rate_from_tab,
            )
        # -------------------------------------------------

        # Extract row data
        row = forecast_df[forecast_df["Year"] == sel_year7].iloc[0]

        forecast_ccc = row["CCC"]
        forecast_dio = row["DIO"]
        forecast_dso = row["DSO"]
        forecast_dpo = row["DPO"]

        daily_cogs = row["COGS"] / 360
        daily_rev = row["Revenue"] / 360

        scenarios = []

        # ==========================================
        # Scenario A - DIO Reduction
        # ==========================================
        ccc_a = forecast_dso + (forecast_dio - scenario_dio_reduction) - forecast_dpo
        cash_a = scenario_dio_reduction * daily_cogs
        int_a = cash_a * interest_rate / 100

        scenarios.append({
            "Scenario": f"A : DIO Reduction ({scenario_dio_reduction:.0f} Days)",
            "Forecast CCC": round(forecast_ccc, 1),
            "Resultant CCC": round(ccc_a, 1),
            "CCC Improvement": round(forecast_ccc - ccc_a, 1),
            "Cash Released (₹ Mn)": round(cash_a, 2),
            "Interest Saved (₹ Mn)": round(int_a, 2)
        })

        # ==========================================
        # Scenario B - DSO Reduction
        # ==========================================
        ccc_b = (forecast_dso - scenario_dso_reduction) + forecast_dio - forecast_dpo
        cash_b = scenario_dso_reduction * daily_rev
        int_b = cash_b * interest_rate / 100

        scenarios.append({
            "Scenario": f"B : DSO Reduction ({scenario_dso_reduction:.0f} Days)",
            "Forecast CCC": round(forecast_ccc, 1),
            "Resultant CCC": round(ccc_b, 1),
            "CCC Improvement": round(forecast_ccc - ccc_b, 1),
            "Cash Released (₹ Mn)": round(cash_b, 2),
            "Interest Saved (₹ Mn)": round(int_b, 2)
        })

        # ==========================================
        # Scenario C - DIO + DSO Reduction
        # ==========================================
        ccc_c = (forecast_dso - scenario_dso_reduction) + (forecast_dio - scenario_dio_reduction) - forecast_dpo
        cash_c = (scenario_dio_reduction * daily_cogs) + (scenario_dso_reduction * daily_rev)
        int_c = cash_c * interest_rate / 100

        scenarios.append({
            "Scenario": f"C : DIO ({scenario_dio_reduction:.0f}) + DSO ({scenario_dso_reduction:.0f})",
            "Forecast CCC": round(forecast_ccc, 1),
            "Resultant CCC": round(ccc_c, 1),
            "CCC Improvement": round(forecast_ccc - ccc_c, 1),
            "Cash Released (₹ Mn)": round(cash_c, 2),
            "Interest Saved (₹ Mn)": round(int_c, 2)
        })

        scenario_df = pd.DataFrame(scenarios)

        st.subheader(f"Scenario Analysis — {sel_year7}")
        st.dataframe(
            scenario_df,
            use_container_width=True,
            hide_index=True
        )


        # ==========================================================
        # REVENUE SENSITIVITY — what if revenue falls short of the forecast?
        # ==========================================================
        st.markdown("---")
        st.subheader(f"Revenue Sensitivity — {sel_year7}")
        st.caption(
            "Re-runs the full cascade (Revenue → COGS & Receivables → Inventory & Payables → "
            "DSO / DIO / DPO), the benchmark gap and the prescriptive optimiser at lower revenue "
            "levels. The fitted regressions, benchmark, constraints and cost of funds are held "
            "unchanged — only revenue varies."
        )

        base_rev = float(row["Revenue"])
        last_actual_rev = float(hist_df["Revenue"].iloc[-1])
        last_actual_label = str(hist_df["Year"].iloc[-1])
        _base_name = (f"{best_model} forecast" if forecast_mode == "Automatic (Best Model)"
                      else "Manual forecast")

        sens_catalog = {}
        sens_catalog[f"{_base_name}"] = (f"{_base_name} (₹{base_rev:,.0f} Mn)", base_rev)
        for pct in (5, 10, 15, 20):
            sens_catalog[f"{pct}% below forecast"] = (
                f"{pct}% below forecast (₹{base_rev * (1 - pct / 100):,.0f} Mn)",
                base_rev * (1 - pct / 100))
        sens_catalog["Zero growth (last actual revenue)"] = (
            f"Zero growth: {last_actual_label} actual (₹{last_actual_rev:,.0f} Mn)", last_actual_rev)

        _sens_default = [_base_name, "10% below forecast", "20% below forecast",
                         "Zero growth (last actual revenue)"]
        picked_cases = st.multiselect(
            "Revenue cases to compare", list(sens_catalog.keys()),
            default=_sens_default, key="sens_cases",
        )
        custom_rev = st.number_input(
            "Custom revenue (₹ Mn) — optional, 0 = not used",
            min_value=0.0, value=0.0, step=100.0, key="sens_custom_rev",
        )

        sens_cases = [sens_catalog[k] for k in sens_catalog if k in picked_cases]
        if custom_rev > 0:
            sens_cases.append((f"Custom revenue (₹{custom_rev:,.0f} Mn)", float(custom_rev)))

        if not sens_cases:
            st.info("Select at least one revenue case to see the sensitivity.")
        else:
            sens_df = revenue_sensitivity(
                models, sens_cases, bm_dio, bm_dso, dso_closure_pct,
                min_inventory_exposure, interest_rate,
            )
            st.dataframe(
                sens_df.style.format({
                    "Revenue (₹ Mn)": "{:,.2f}",
                    "Benchmark-gap opportunity (₹ Mn)": "{:,.2f}",
                    "Prescriptive cash release (₹ Mn)": "{:,.2f}",
                    "DIO reduction (days)": "{:.1f}",
                    "DSO reduction (days)": "{:.1f}",
                    "Interest saving (₹ Mn)": "{:,.2f}",
                    "Forecast DSO": "{:.1f}", "Forecast DIO": "{:.1f}",
                    "Forecast DPO": "{:.1f}", "Forecast CCC": "{:.1f}",
                }),
                use_container_width=True, hide_index=True,
            )

            fig_sens = go.Figure()
            fig_sens.add_trace(go.Bar(
                name="Benchmark-gap opportunity", x=sens_df["Revenue case"],
                y=sens_df["Benchmark-gap opportunity (₹ Mn)"], marker_color=NAVY))
            fig_sens.add_trace(go.Bar(
                name="Prescriptive cash release", x=sens_df["Revenue case"],
                y=sens_df["Prescriptive cash release (₹ Mn)"], marker_color=GOLD))
            fig_sens.update_layout(barmode="group", height=380, plot_bgcolor="white",
                                   yaxis_title="₹ Mn", margin=dict(t=20),
                                   legend=dict(orientation="h", y=-0.3))
            st.plotly_chart(fig_sens, use_container_width=True)

            if float(sens_df["Benchmark-gap opportunity (₹ Mn)"].max()) <= 0:
                st.info(
                    "At the current benchmark selection the forecast is already better than the "
                    "benchmark in every case, so the opportunity and the release are zero. Choose "
                    "'Best peer (lowest CCC)' in the sidebar to see the improvement opportunity."
                )
