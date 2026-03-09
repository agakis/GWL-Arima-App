import os
import sys
import warnings
import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA

# ---------- USER SETTINGS ----------
INPUT_BASENAME = "timeseries_yearmax"
SHEET_NAME = "timeseries_yearmax"

OUTPUT_FILE = "arima_yearmax_forecast.xlsx"
OUTPUT_SHEET_FORECAST = "forecast"
OUTPUT_SHEET_MODELINFO = "model_info"

MIN_YEAR = 1965
MIN_OBS = 12
TARGET_YEAR = 2032

P_CANDIDATES = [0, 1]
D_CANDIDATES = [0, 1]
Q_CANDIDATES = [0, 1]

MAX_BACKTEST_POINTS = 8
MIN_TRAIN_FOR_BACKTEST = 10

# Central intervals:
ALPHA_68 = 0.32   # central 68%
ALPHA_30 = 0.70   # central 30%  (because 1 - 0.30 = 0.70)
# ----------------------------------


def find_excel_file(script_dir: str, basename: str) -> str:
    candidates = [
        os.path.join(script_dir, f"{basename}.xlsx"),
        os.path.join(script_dir, f"{basename}.xlsm"),
        os.path.join(script_dir, f"{basename}.xls"),
        os.path.join(script_dir, basename),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"Could not find '{basename}' in {script_dir}. "
        f"Tried: {', '.join(os.path.basename(c) for c in candidates)}"
    )


def is_year_col(c) -> bool:
    try:
        y = int(float(str(c).strip()))
        return 1900 <= y <= 2100
    except Exception:
        return False


def year_int(c) -> int:
    return int(float(str(c).strip()))


def extract_year_cols(df: pd.DataFrame) -> list:
    cols = [c for c in df.columns if is_year_col(c)]
    return sorted(cols, key=year_int)


def longest_consecutive_block(y_series: pd.Series) -> pd.Series:
    s = y_series.dropna().copy()
    if s.empty:
        return s

    years = np.array(sorted(s.index.astype(int)))
    breaks = np.where(np.diff(years) != 1)[0]
    start_idxs = np.r_[0, breaks + 1]
    end_idxs = np.r_[breaks, len(years) - 1]

    best_years = None
    best_len = -1
    best_end = -1

    for a, b in zip(start_idxs, end_idxs):
        run = years[a:b + 1]
        run_len = len(run)
        run_end = int(run[-1])
        if (run_len > best_len) or (run_len == best_len and run_end > best_end):
            best_len = run_len
            best_end = run_end
            best_years = run

    out = s.loc[best_years].copy()
    out.index = out.index.astype(int)
    return out.sort_index()


def most_recent_consecutive_block(y_series: pd.Series) -> pd.Series:
    s = y_series.dropna().copy()
    if s.empty:
        return s

    years = np.array(sorted(s.index.astype(int)))
    end = years[-1]
    year_set = set(years)

    run = [end]
    cur = end
    while (cur - 1) in year_set:
        cur -= 1
        run.append(cur)

    run_years = np.array(sorted(run))
    out = s.loc[run_years].copy()
    out.index = out.index.astype(int)
    return out.sort_index()


def choose_training_block(y_series: pd.Series) -> pd.Series:
    recent = most_recent_consecutive_block(y_series)
    if len(recent) >= MIN_OBS:
        return recent
    return longest_consecutive_block(y_series)


def fit_arima(y: pd.Series, order: tuple):
    return ARIMA(y, order=order).fit()


def rolling_rmse_for_order(y: pd.Series, order: tuple) -> float:
    n = len(y)
    if n < (MIN_TRAIN_FOR_BACKTEST + 2):
        return np.inf

    K = min(MAX_BACKTEST_POINTS, n - MIN_TRAIN_FOR_BACKTEST)
    targets = list(range(n - K, n))

    errors = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in targets:
            train = y.iloc[:t]
            obs = float(y.iloc[t])
            try:
                res = fit_arima(train, order)
                pred = float(np.asarray(res.get_forecast(steps=1).predicted_mean)[0])
                errors.append(pred - obs)
            except Exception:
                return np.inf

    if not errors:
        return np.inf
    return float(np.sqrt(np.mean(np.square(errors))))


def best_arima_by_rmse_then_aic(y: pd.Series):
    best_model = None
    best_order = None
    best_rmse = np.inf
    best_aic = np.inf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in P_CANDIDATES:
            for d in D_CANDIDATES:
                for q in Q_CANDIDATES:
                    if p == 0 and d == 0 and q == 0:
                        continue
                    order = (p, d, q)

                    rmse = rolling_rmse_for_order(y, order)
                    if not np.isfinite(rmse):
                        continue

                    try:
                        res = fit_arima(y, order)
                        aic = float(res.aic)
                    except Exception:
                        continue

                    better = (rmse < best_rmse - 1e-12) or (abs(rmse - best_rmse) <= 1e-12 and aic < best_aic)
                    if better:
                        best_model, best_order, best_rmse, best_aic = res, order, rmse, aic

    return best_model, best_order, best_rmse, best_aic


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    in_path = find_excel_file(script_dir, INPUT_BASENAME)
    out_path = os.path.join(script_dir, OUTPUT_FILE)

    df = pd.read_excel(in_path, sheet_name=SHEET_NAME, engine="openpyxl")

    for c in ["station", "x", "y"]:
        if c not in df.columns:
            raise KeyError(f"Missing required column '{c}' in sheet '{SHEET_NAME}'.")

    year_cols = extract_year_cols(df)
    if not year_cols:
        raise KeyError("No yearly columns found (expected 2025, 2024, ...).")

    forecasts_rows = []
    modelinfo_rows = []

    for idx in range(len(df)):
        station = str(df.loc[idx, "station"])
        x = df.loc[idx, "x"]
        ycoord = df.loc[idx, "y"]

        vals = pd.to_numeric(df.loc[idx, year_cols], errors="coerce").to_numpy(dtype=float)
        yrs = np.array([year_int(c) for c in year_cols], dtype=int)

        y_series = pd.Series(vals, index=yrs).sort_index()
        y_series = y_series[y_series.index >= MIN_YEAR]

        y_train = choose_training_block(y_series)

        if len(y_train) < MIN_OBS:
            modelinfo_rows.append({
                "station": station, "x": x, "y": ycoord,
                "status": f"skipped (only {len(y_train)} consecutive years; need >= {MIN_OBS})",
                "order": "", "rmse": np.nan, "aic": np.nan,
                "train_start": int(y_train.index.min()) if len(y_train) else np.nan,
                "train_end": int(y_train.index.max()) if len(y_train) else np.nan,
            })
            continue

        best_model, best_order, best_rmse, best_aic = best_arima_by_rmse_then_aic(y_train)

        if best_model is None:
            modelinfo_rows.append({
                "station": station, "x": x, "y": ycoord,
                "status": "failed (no model converged)",
                "order": "", "rmse": np.nan, "aic": np.nan,
                "train_start": int(y_train.index.min()),
                "train_end": int(y_train.index.max()),
            })
            continue

        train_start = int(y_train.index.min())
        train_end = int(y_train.index.max())

        steps = int(max(0, TARGET_YEAR - train_end))
        if steps == 0:
            modelinfo_rows.append({
                "station": station, "x": x, "y": ycoord,
                "status": "ok (train_end >= target)",
                "order": str(best_order),
                "rmse": float(best_rmse), "aic": float(best_aic),
                "train_start": train_start, "train_end": train_end,
            })
            continue

        fc = best_model.get_forecast(steps=steps)
        mean = np.asarray(fc.predicted_mean)

        ci68 = fc.conf_int(alpha=ALPHA_68)
        ci30 = fc.conf_int(alpha=ALPHA_30)

        lo68 = np.asarray(ci68.iloc[:, 0])
        hi68 = np.asarray(ci68.iloc[:, 1])
        lo30 = np.asarray(ci30.iloc[:, 0])
        hi30 = np.asarray(ci30.iloc[:, 1])

        fy = np.arange(train_end + 1, train_end + steps + 1, dtype=int)

        for j, year in enumerate(fy):
            forecasts_rows.append({
                "station": station, "x": x, "y": ycoord,
                "year": int(year),
                "forecast": float(mean[j]),
                "lo_68": float(lo68[j]),
                "hi_68": float(hi68[j]),
                "lo_30": float(lo30[j]),
                "hi_30": float(hi30[j]),
                "order_p": best_order[0],
                "order_d": best_order[1],
                "order_q": best_order[2],
                "rmse": float(best_rmse),
                "aic": float(best_aic),
                "train_start": train_start,
                "train_end": train_end,
            })

        modelinfo_rows.append({
            "station": station, "x": x, "y": ycoord,
            "status": "ok",
            "order": str(best_order),
            "rmse": float(best_rmse),
            "aic": float(best_aic),
            "train_start": train_start,
            "train_end": train_end,
        })

        print(f"✅ {station}: ARIMA{best_order} train={train_start}..{train_end} -> {TARGET_YEAR}")

    fc_df = pd.DataFrame(forecasts_rows)
    mi_df = pd.DataFrame(modelinfo_rows)

    with pd.ExcelWriter(out_path, engine="openpyxl", mode="w") as writer:
        fc_df.to_excel(writer, sheet_name=OUTPUT_SHEET_FORECAST, index=False)
        mi_df.to_excel(writer, sheet_name=OUTPUT_SHEET_MODELINFO, index=False)

    print(f"\n✅ Done. Wrote: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)