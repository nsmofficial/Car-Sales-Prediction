# src/generate_quality_summary.py
import pandas as pd

def create_quality_summary(input_path="outputs/model_report.csv",
                           output_path="outputs/kpi_quality_summary.csv"):
    report = pd.read_csv(input_path)

    # Best model per KPI (lowest MASE, tie-break with sMAPE)
    best_models = (
        report.sort_values(["english_name", "MASE", "sMAPE_%"])
        .groupby("english_name")
        .first()
        .reset_index()
    )

    # Add quality flags
    def flag_quality(row):
        if row["MASE"] < 1 and row["sMAPE_%"] < 50:
            return "🟢 Reliable"
        elif row["MASE"] < 5 and row["sMAPE_%"] < 100:
            return "🟡 Moderate"
        else:
            return "🔴 Unreliable"

    best_models["Quality"] = best_models.apply(flag_quality, axis=1)

    # Save
    best_models.to_csv(output_path, index=False)
    print(f"✅ KPI quality summary saved to {output_path}")
    return best_models

if __name__ == "__main__":
    create_quality_summary()
