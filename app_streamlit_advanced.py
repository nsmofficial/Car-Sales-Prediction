import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from src import advanced_solution as adv

st.set_page_config(
    page_title="Dealership KPI Forecasting",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Dealership KPI Forecasting Dashboard")

# --- Sidebar
st.sidebar.header("📂 Data & Settings")
uploaded = st.sidebar.file_uploader("Upload KPI CSV", type=["csv"])

default_file = Path("outputs/forecast_results.csv")
if uploaded:
    df = pd.read_csv(uploaded, parse_dates=["date"])
elif default_file.exists():
    df = pd.read_csv(default_file, parse_dates=["date"])
else:
    st.warning("Upload forecast_results.csv or run pipeline first.")
    st.stop()

# --- KPI Whitelist (business-relevant only)
whitelist_kpis = [
    "TOTAL MITSUBISHI CARS RETAIL (LINES 2  THRU 4)",
    "UV Units Sold Per NV & UV Sales Rep.",
    "NV Units Sold",
    "Used Vehicles - Retail",
    "NV Gross Profit",
    "UV Gross Profit",
    "Gross Profit (Total)",
    "Finance Income - New",
    "Finance Income - Used",
    "Finance Income (CPO)",
    "Service Contracts (CPO)",
    "Finance Penetration %",
    "Customer Pay Hours Sold",
    "Warranty Hours Sold",
    "Service Revenue",
    "Parts Revenue",
    "Labour Sales Revenue",
    "Marketing Expense",
    "Active Customers ",
    "Customer Retention %",
    "Dept’l Operating Profit",
    "Profit or Loss (Before Income Tax)",
]

# Filter dataset
df = df[df["english_name"].isin(whitelist_kpis)]

# Prepare lists
all_kpis = sorted(df["english_name"].unique())
st.sidebar.markdown("### 🔍 Choose KPI")
kpi_choice = st.sidebar.selectbox("Select KPI", all_kpis, index=0)

# Date filter
date_min = df["date"].min()
date_max = df["date"].max()
date_range = st.sidebar.slider(
    "Date Range", min_value=date_min.to_pydatetime(), max_value=date_max.to_pydatetime(),
    value=(date_min.to_pydatetime(), date_max.to_pydatetime())
)

# --- Main Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Forecast Explorer",
    "🔗 Correlation Heatmap", 
    "🤔 What-If Analysis",
    "📊 Model Report Explorer"
])

# --- Tab 1: Forecast Explorer
with tab1:
    st.subheader(f"📊 {kpi_choice} Forecast")
    g = df[df["english_name"] == kpi_choice].sort_values("date")
    g = g[(g["date"] >= pd.to_datetime(date_range[0])) & (g["date"] <= pd.to_datetime(date_range[1]))]

    fig = px.line(g, x="date", y="monthly_value", color="is_forecast",
                  labels={"monthly_value":"Monthly Value", "date":"Date", "is_forecast":"Data Type"},
                  title=f"{kpi_choice} – History vs Forecast")
    fig.update_traces(mode="lines+markers")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(g[["date","monthly_value","is_forecast","model_choice"]].tail(12))

# --- Tab 2: Correlation Heatmap
with tab2:
    st.subheader("🔗 KPI Correlation Heatmap")
    corr_path = Path("outputs/correlation_matrix.csv")
    if corr_path.exists():
        corr = pd.read_csv(corr_path, index_col=0)
        fig = px.imshow(corr, text_auto=False, aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Run pipeline to generate correlation_matrix.csv")

# --- Tab 3: What-If Analysis
with tab3:
    st.subheader("🤔 What-If Scenario")
    st.markdown("Change a KPI forecast and see correlated impacts.")

    with st.form("whatif_form"):
        kpi_sel = st.selectbox("KPI to change", all_kpis, index=0)
        target_dt = st.selectbox("Target forecast month",
                                 sorted(df[df["is_forecast"]==True]["date"].unique()))
        pct_change = st.slider("Percent Change", -50, 50, 10)
        submitted = st.form_submit_button("Run Scenario")

    if submitted:
        hist = df[df["is_forecast"]==False]
        out = adv.what_if_correlation_propagation(
            df_fcst=df,
            df_hist=hist,
            kpi_name=kpi_sel,
            target_date=target_dt,
            percent_change=pct_change
        )
        # Compare original vs adjusted for selected KPI
        g0 = df[df["english_name"]==kpi_sel].copy()
        g1 = out[out["english_name"]==kpi_sel].copy()
        g0["scenario"] = "Original"
        g1["scenario"] = "Adjusted"
        comp = pd.concat([g0,g1])

        fig = px.line(comp, x="date", y="monthly_value", color="scenario",
                      title=f"What-If: {kpi_sel} {pct_change:+d}% on {pd.to_datetime(target_dt).date()}")
        fig.update_traces(mode="lines+markers")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(comp.tail(12))

# --- Tab 4: Model Report Explorer
with tab4:
    st.subheader("📊 Model Report Explorer")

    report_path = Path("outputs/kpi_quality_summary.csv")
    if not report_path.exists():
        st.warning("Run pipeline + quality summary script first.")
    else:
        summary = pd.read_csv(report_path)

        # Sidebar filters
        quality_filter = st.multiselect(
            "Filter by Quality", ["🟢 Reliable", "🟡 Moderate", "🔴 Unreliable"],
            default=["🟢 Reliable","🟡 Moderate","🔴 Unreliable"]
        )
        df_show = summary[summary["Quality"].isin(quality_filter)]

        # Show stats
        st.markdown("### ✅ KPI Quality Breakdown")
        col1, col2, col3 = st.columns(3)
        col1.metric("Reliable", (summary["Quality"]=="🟢 Reliable").sum())
        col2.metric("Moderate", (summary["Quality"]=="🟡 Moderate").sum())
        col3.metric("Unreliable", (summary["Quality"]=="🔴 Unreliable").sum())

        # Interactive table
        st.dataframe(
            df_show.sort_values("MASE")[["english_name","model","MASE","sMAPE_%","Quality"]],
            use_container_width=True
        )

        # Plot distribution
        fig = px.histogram(summary, x="MASE", nbins=50, color="Quality",
                           title="Distribution of MASE across KPIs")
        st.plotly_chart(fig, use_container_width=True)


