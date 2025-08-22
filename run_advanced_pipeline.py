import pandas as pd
from pathlib import Path
from src import advanced_solution as adv

# ========= CHOOSE YOUR FILE HERE =========
# Use full history (recommended for accuracy)
DATA_PATH = Path("data/FS-data-80475.csv")
# Or, if you only want 2025:
# DATA_PATH = Path("data/FS-data-80475-2025-all-months.csv")
# Or demo:
# DATA_PATH = Path("data/sample_data.csv")
# =========================================

OUT = Path("outputs"); PLOTS = OUT / "plots"
OUT.mkdir(parents=True, exist_ok=True); PLOTS.mkdir(parents=True, exist_ok=True)

# Load
df = pd.read_csv(DATA_PATH)

# Prepare
prepared = adv.prepare_data(df)

# Forecast ALL KPIs with model selection (parallel)
fcst, model_report = adv.forecast_all_kpis(prepared, horizon=3, n_jobs=-1)

# Artifacts
fcst.sort_values(["english_name","date"]).to_csv(OUT / "forecast_results.csv", index=False)
model_report.sort_values(["english_name","MASE"]).to_csv(OUT / "model_report.csv", index=False)

# Generate quality summary
from src.generate_quality_summary import create_quality_summary
create_quality_summary()

corr = adv.correlation_matrix(fcst)
corr.to_csv(OUT / "correlation_matrix.csv")

# Plots
adv.plot_forecasts(fcst, PLOTS, history_months=24)
adv.plot_correlation_heatmap(corr, OUT / "correlation_heatmap.png")

# Example What-If: +10% Units Sold on earliest forecast month
first_fcst_dt = fcst[fcst["is_forecast"]].sort_values("date")["date"].iloc[0]
what_if = adv.what_if_correlation_propagation(
    df_fcst=fcst,
    df_hist=prepared,
    kpi_name="Units Sold",            # change to any KPI in your data
    target_date=first_fcst_dt,
    percent_change=10.0
)
what_if.to_csv(OUT / "what_if_example.csv", index=False)

print("Done. See outputs/ and outputs/plots/. Key files:")
print("- outputs/forecast_results.csv")
print("- outputs/model_report.csv  (per-KPI model selection + CV scores)")
print("- outputs/correlation_matrix.csv")
print("- outputs/what_if_example.csv")
print("- outputs/plots/*.png")
