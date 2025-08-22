import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import pmdarima as pm
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ---------- Utilities

def _ensure_monthly_continuity(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name, g in df.groupby("english_name"):
        # Remove duplicates within each KPI group
        g = g.drop_duplicates(subset=["date"], keep="first")
        idx = pd.date_range(g["date"].min(), g["date"].max(), freq="MS")
        gf = g.set_index("date").reindex(idx).reset_index().rename(columns={"index":"date"})
        # carry id fields
        for c in ["account_id", "english_name", "dealer_code"]:
            gf[c] = g[c].iloc[0]
        gf["year"] = gf["date"].dt.year
        gf["month"] = gf["date"].dt.month
        out.append(gf)
    return pd.concat(out, ignore_index=True)

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    req = ["account_id","english_name","dealer_code","year","month","monthly_value"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.copy()
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    df = df.sort_values(["english_name","date"])
    df = _ensure_monthly_continuity(df)
    df["yearly_value"] = df.groupby(["english_name","year"])["monthly_value"].cumsum()
    return df

# Metrics robust to zeros
def smape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred) + eps)
    return np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0

def mase(y_true, y_pred, seasonal_periods: int = 12):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    n = y_true.shape[0]
    if n <= seasonal_periods + 1:
        # fallback to non-seasonal scaling
        scale = np.mean(np.abs(np.diff(y_true))) if n > 1 else 1.0
    else:
        scale = np.mean(np.abs(y_true[seasonal_periods:] - y_true[:-seasonal_periods]))
    scale = scale if scale != 0 and not np.isnan(scale) else 1.0
    return np.mean(np.abs(y_true - y_pred)) / scale

# ---------- Design matrices for Linear baseline
MONTH_DUMMY_COLS = [f"m_{i}" for i in range(2, 13)]  # m_2..m_12

def _design_matrix(dates: pd.Series) -> pd.DataFrame:
    dates = pd.to_datetime(dates)
    t = (dates.dt.year - dates.dt.year.min())*12 + (dates.dt.month - 1)
    dummies = pd.get_dummies(dates.dt.month, prefix="m")
    if "m_1" in dummies.columns:
        dummies = dummies.drop(columns=["m_1"])
    dummies = dummies.reindex(columns=MONTH_DUMMY_COLS, fill_value=0)
    return dummies.assign(t=t.values)

# ---------- Model wrappers

@dataclass
class FittedModel:
    model_type: str
    fitted_obj: object
    seasonal_periods: int = 12

def _fit_linear(train_df: pd.DataFrame) -> FittedModel:
    X = _design_matrix(train_df["date"])
    y = train_df["monthly_value"].values
    lr = LinearRegression().fit(X, y)
    return FittedModel("LINEAR", lr)

def _forecast_linear(fm: FittedModel, fut_dates: pd.Series) -> np.ndarray:
    Xf = _design_matrix(pd.Series(fut_dates))
    return fm.fitted_obj.predict(Xf)

def _fit_autoarima(train_df: pd.DataFrame, m: int = 12) -> FittedModel:
    y = train_df["monthly_value"].values
    # For short series, disable seasonality
    seasonal = True if len(y) >= 2*m else False
    
    try:
        model = pm.auto_arima(
            y,
            seasonal=seasonal,
            m=m,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            max_p=min(3, len(y)//3), max_q=min(3, len(y)//3), 
            max_P=1, max_Q=1,
            trace=False,
        )
        return FittedModel("ARIMA", model, seasonal_periods=m)
    except Exception:
        # Fallback to simple ARIMA(1,1,1) if auto_arima fails
        try:
            model = pm.ARIMA(order=(1,1,1)).fit(y)
            return FittedModel("ARIMA", model, seasonal_periods=m)
        except Exception:
            # Final fallback to naive
            return _fit_naive(train_df, m)

def _forecast_autoarima(fm: FittedModel, periods: int) -> np.ndarray:
    return fm.fitted_obj.predict(n_periods=periods)

def _fit_ets(train_df: pd.DataFrame, m: int = 12) -> FittedModel:
    y = train_df["monthly_value"].values
    seasonal = None
    trend = "add"
    if len(y) >= 2*m:
        # heuristic: multiplicative only if strictly positive and has seasonality
        seasonal = "mul" if np.min(y) > 0 else "add"
    hw = ExponentialSmoothing(
        train_df["monthly_value"].astype(float),
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=m if seasonal else None,
        initialization_method="estimated",
    ).fit(optimized=True)
    return FittedModel("ETS", hw, seasonal_periods=m)

def _forecast_ets(fm: FittedModel, periods: int) -> np.ndarray:
    return fm.fitted_obj.forecast(periods).values

def _fit_naive(train_df: pd.DataFrame, m: int = 12) -> FittedModel:
    return FittedModel("SNAIVE", {"last": train_df["monthly_value"].iloc[-1],
                                  "hist": train_df["monthly_value"].values}, seasonal_periods=m)

def _forecast_naive(fm: FittedModel, periods: int) -> np.ndarray:
    hist = fm.fitted_obj["hist"]
    m = fm.seasonal_periods
    if len(hist) >= m:
        seasonal = hist[-m:]
        reps = int(np.ceil(periods / m))
        fc = np.tile(seasonal, reps)[:periods]
        return fc
    else:
        return np.repeat(fm.fitted_obj["last"], periods)

# ---------- Rolling-origin CV and model selection

def _rolling_cv_forecast(train_df: pd.DataFrame, model_fit_fn: Callable, model_fcst_fn: Callable,
                         k_back: int = 6) -> Tuple[float, float]:
    """Return (MASE, sMAPE) for last k_back one-step-ahead forecasts."""
    g = train_df.sort_values("date").copy()
    dates = g["date"].unique()
    if len(dates) < k_back + 12:
        # not enough history: smaller CV window
        k_back = max(1, min(3, len(dates) - 1))
    y_true_all, y_pred_all = [], []
    for dt in dates[-k_back:]:
        train = g[g["date"] < dt]
        test = g[g["date"] == dt]
        if train.empty or test.empty:
            continue
        try:
            fm = model_fit_fn(train)
            # always 1-step forecast for CV
            if model_fcst_fn in (_forecast_autoarima, _forecast_ets, _forecast_naive):
                yhat = model_fcst_fn(fm, 1)[0]
            else:
                yhat = model_fcst_fn(fm, test["date"])[0]
        except Exception:
            continue
        y_true_all.append(test["monthly_value"].iloc[0])
        y_pred_all.append(yhat)
    if not y_true_all:
        return (np.inf, np.inf)
    return (mase(y_true_all, y_pred_all, seasonal_periods=12),
            smape(y_true_all, y_pred_all))

def select_best_model_for_series(series_df: pd.DataFrame) -> Tuple[FittedModel, Dict]:
    """Try ARIMA, ETS, LINEAR, SNAIVE. Pick by MASE; sMAPE as tie-breaker."""
    candidates = [
        ("ARIMA", _fit_autoarima, _forecast_autoarima),
        ("ETS",   _fit_ets,       _forecast_ets),
        ("LINEAR",_fit_linear,    _forecast_linear),
        ("SNAIVE",_fit_naive,     _forecast_naive),
    ]
    scores = []
    for name, fit_fn, fc_fn in candidates:
        m_mase, m_smape = _rolling_cv_forecast(series_df, fit_fn, fc_fn, k_back=6)
        scores.append((name, m_mase, m_smape))
    # choose best by MASE (lower is better), then sMAPE
    scores_sorted = sorted(scores, key=lambda t: (t[1], t[2]))
    best_name = scores_sorted[0][0]

    # fit the best on full series
    fit_fn = dict((n,f) for n,f,_ in candidates)[best_name]
    fm = fit_fn(series_df)

    return fm, {
        "scores": scores_sorted,
        "best": best_name
    }

# ---------- Forecast all KPIs

def forecast_all_kpis(prepared: pd.DataFrame, horizon: int = 3, n_jobs: int = -1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (forecast_df_with_history, model_report)"""
    groups = list(prepared.groupby("english_name"))

    def _process_one(name, g):
        g = g.dropna(subset=["monthly_value"]).sort_values("date")
        if len(g) < 8:
            # too short, fall back directly to SNAIVE
            fm = _fit_naive(g)
            best = "SNAIVE"; scores = [("SNAIVE", np.nan, np.nan)]
        else:
            fm, rpt = select_best_model_for_series(g)
            best = rpt["best"]; scores = rpt["scores"]

        last_date = g["date"].max()
        fut_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")

        try:
            if fm.model_type == "ARIMA":
                yf = _forecast_autoarima(fm, horizon)
            elif fm.model_type == "ETS":
                yf = _forecast_ets(fm, horizon)
            elif fm.model_type == "LINEAR":
                yf = _forecast_linear(fm, fut_dates)
            else:
                yf = _forecast_naive(fm, horizon)
        except Exception:
            # robust fallback
            yf = _forecast_naive(_fit_naive(g), horizon)

        f = pd.DataFrame({
            "account_id": g["account_id"].iloc[0],
            "english_name": name,
            "dealer_code": g["dealer_code"].iloc[0],
            "date": fut_dates,
            "year": fut_dates.year,
            "month": fut_dates.month,
            "monthly_value": yf,
            "is_forecast": True,
            "model_choice": best
        })
        g = g.copy()
        g["is_forecast"] = False
        g["model_choice"] = best
        return pd.concat([g, f], ignore_index=True), pd.DataFrame(
            [{"english_name": name, "model": n, "MASE": m, "sMAPE_%": s} for (n,m,s) in scores]
        )

    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_process_one)(name, g) for name, g in groups
    )

    parts, reports = zip(*results)
    out = pd.concat(parts, ignore_index=True)
    out["yearly_value"] = out.groupby(["english_name","year"])["monthly_value"].cumsum()
    report = pd.concat(reports, ignore_index=True)
    return out, report

# ---------- Correlation + What-If

def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    hist = df[df["is_forecast"] == False]
    piv = hist.pivot_table(index="date", columns="english_name", values="monthly_value")
    return piv.corr()

def what_if_correlation_propagation(df_fcst, df_hist, kpi_name, target_date, percent_change):
    out = df_fcst.copy()
    hist_piv = df_hist.pivot_table(index="date", columns="english_name", values="monthly_value")
    stds = hist_piv.std()
    corr = hist_piv.corr()

    mask_i = (out["english_name"] == kpi_name) & (out["date"] == pd.to_datetime(target_date)) & (out["is_forecast"])
    if not mask_i.any() or kpi_name not in stds.index:
        return out

    base_i = out.loc[mask_i, "monthly_value"].iloc[0]
    delta_i = base_i * (percent_change/100.0)
    out.loc[mask_i, "monthly_value"] = base_i + delta_i

    sigma_i = stds.get(kpi_name, np.nan)
    if pd.isna(sigma_i) or sigma_i == 0:
        out["yearly_value"] = out.groupby(["english_name","year"])["monthly_value"].cumsum()
        return out

    for other in stds.index:
        if other == kpi_name: continue
        rho = corr.loc[kpi_name, other] if (kpi_name in corr.index and other in corr.columns) else 0.0
        sigma_j = stds.get(other, np.nan)
        if pd.isna(sigma_j) or sigma_j == 0 or pd.isna(rho): continue
        delta_j = rho * (sigma_j/sigma_i) * delta_i
        mask_j = (out["english_name"] == other) & (out["date"] == pd.to_datetime(target_date)) & (out["is_forecast"])
        if mask_j.any():
            out.loc[mask_j, "monthly_value"] = out.loc[mask_j, "monthly_value"] + delta_j

    out["yearly_value"] = out.groupby(["english_name","year"])["monthly_value"].cumsum()
    return out

# ---------- Plots

def plot_forecasts(df_fcst: pd.DataFrame, out_dir: Path, history_months: int = 24):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, g in df_fcst.groupby("english_name"):
        g = g.sort_values("date")
        hist = g[g["is_forecast"] == False].tail(history_months)
        fcst = g[g["is_forecast"] == True]
        plt.figure()
        if not hist.empty:
            plt.plot(hist["date"], hist["monthly_value"], label="History")
        if not fcst.empty:
            plt.plot(fcst["date"], fcst["monthly_value"], label="Forecast")
        plt.title(f"{name} – History & Forecast")
        plt.xlabel("Date"); plt.ylabel("Monthly Value"); plt.legend()
        p = out_dir / f"{name.replace(' ', '_').replace('/', '_')}_forecast.png"
        plt.tight_layout(); plt.savefig(p); plt.close()

def plot_correlation_heatmap(corr: pd.DataFrame, out_path: Path):
    plt.figure()
    im = plt.imshow(corr.values, aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("KPI Correlation (Pearson)")
    plt.tight_layout(); plt.savefig(out_path); plt.close()


