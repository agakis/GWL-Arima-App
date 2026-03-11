import io
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

# ------------------------------
# Default settings
# ------------------------------
DEFAULT_INPUT_SHEET = "timeseries_yearmax"
DEFAULT_OUTPUT_FORECAST_SHEET = "forecast"
DEFAULT_OUTPUT_MODELINFO_SHEET = "model_info"

DEFAULT_MIN_YEAR = 1965
DEFAULT_MIN_OBS = 12
DEFAULT_TARGET_YEAR = 2032
DEFAULT_P = [0, 1]
DEFAULT_D = [0, 1]
DEFAULT_Q = [0, 1]
DEFAULT_MAX_BACKTEST_POINTS = 8
DEFAULT_MIN_TRAIN_FOR_BACKTEST = 10
DEFAULT_ALPHA_68 = 0.32  # central 68%
DEFAULT_ALPHA_30 = 0.70  # central 30%
DEFAULT_OLD_YEAR = 1988

min_obs = DEFAULT_MIN_OBS
max_backtest_points = DEFAULT_MAX_BACKTEST_POINTS
min_train_for_backtest = DEFAULT_MIN_TRAIN_FOR_BACKTEST
# ------------------------------
# Helper functions
# ------------------------------

# ------------------------------
# Simple access control
# ------------------------------

def check_login(username: str, password: str) -> bool:
    expected_user = st.secrets.get("AUTH_USERNAME", "")
    expected_pass = st.secrets.get("AUTH_PASSWORD", "")
    return username == expected_user and password == expected_pass


def require_login():
    if st.session_state.get("authenticated", False):
        with st.sidebar:
            st.success("Logged in")
            if st.button("Log out"):
                st.session_state.authenticated = False
                st.rerun()
        return

    st.title("ARIMA Groundwater Forecast App")
    st.subheader("Login")
    st.info("Please enter your username and password to access the app.")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in", type="primary")

    if submitted:
        if check_login(username, password):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid username or password.")

    st.stop()


def is_year_col(c) -> bool:
    try:
        y = int(float(str(c).strip()))
        return 1900 <= y <= 2100
    except Exception:
        return False


def year_int(c) -> int:
    return int(float(str(c).strip()))


def extract_year_cols(df: pd.DataFrame) -> List:
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
        run = years[a : b + 1]
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


def choose_training_block(y_series: pd.Series, min_obs: int) -> pd.Series:
    recent = most_recent_consecutive_block(y_series)
    if len(recent) >= min_obs:
        return recent
    return longest_consecutive_block(y_series)


def fit_arima(y: pd.Series, order: Tuple[int, int, int]):
    return ARIMA(y, order=order).fit()


def rolling_rmse_for_order(
    y: pd.Series,
    order: Tuple[int, int, int],
    max_backtest_points: int,
    min_train_for_backtest: int,
) -> float:
    n = len(y)
    if n < (min_train_for_backtest + 2):
        return np.inf

    k = min(max_backtest_points, n - min_train_for_backtest)
    targets = list(range(n - k, n))

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


def best_arima_by_rmse_then_aic(
    y: pd.Series,
    p_candidates: List[int],
    d_candidates: List[int],
    q_candidates: List[int],
    max_backtest_points: int,
    min_train_for_backtest: int,
):
    best_model = None
    best_order = None
    best_rmse = np.inf
    best_aic = np.inf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in p_candidates:
            for d in d_candidates:
                for q in q_candidates:
                    if p == 0 and d == 0 and q == 0:
                        continue
                    order = (p, d, q)

                    rmse = rolling_rmse_for_order(
                        y, order, max_backtest_points, min_train_for_backtest
                    )
                    if not np.isfinite(rmse):
                        continue

                    try:
                        res = fit_arima(y, order)
                        aic = float(res.aic)
                    except Exception:
                        continue

                    better = (rmse < best_rmse - 1e-12) or (
                        abs(rmse - best_rmse) <= 1e-12 and aic < best_aic
                    )
                    if better:
                        best_model, best_order, best_rmse, best_aic = (
                            res,
                            order,
                            rmse,
                            aic,
                        )

    return best_model, best_order, best_rmse, best_aic


def read_excel_from_upload(file, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(file, sheet_name=sheet_name, engine="openpyxl")


def run_forecast(
    df: pd.DataFrame,
    min_year: int,
    min_obs: int,
    target_year: int,
    p_candidates: List[int],
    d_candidates: List[int],
    q_candidates: List[int],
    max_backtest_points: int,
    min_train_for_backtest: int,
    alpha_68: float,
    alpha_30: float,
):
    for c in ["station", "x", "y"]:
        if c not in df.columns:
            raise KeyError(f"Missing required column '{c}' in input sheet.")

    year_cols = extract_year_cols(df)
    if not year_cols:
        raise KeyError("No yearly columns found (expected columns like 2025, 2024, ...).")

    forecasts_rows = []
    modelinfo_rows = []
    progress = st.progress(0)
    status = st.empty()

    total = len(df)
    for idx in range(total):
        progress.progress((idx + 1) / max(total, 1))
        station = str(df.loc[idx, "station"])
        status.write(f"Processing station {idx + 1}/{total}: **{station}**")

        x = df.loc[idx, "x"]
        ycoord = df.loc[idx, "y"]

        vals = pd.to_numeric(df.loc[idx, year_cols], errors="coerce").to_numpy(dtype=float)
        yrs = np.array([year_int(c) for c in year_cols], dtype=int)

        y_series = pd.Series(vals, index=yrs).sort_index()
        y_series = y_series[y_series.index >= min_year]

        y_train = choose_training_block(y_series, min_obs)

        if len(y_train) < min_obs:
            modelinfo_rows.append(
                {
                    "station": station,
                    "x": x,
                    "y": ycoord,
                    "status": f"skipped (only {len(y_train)} consecutive years; need >= {min_obs})",
                    "order": "",
                    "rmse": np.nan,
                    "aic": np.nan,
                    "train_start": int(y_train.index.min()) if len(y_train) else np.nan,
                    "train_end": int(y_train.index.max()) if len(y_train) else np.nan,
                }
            )
            continue

        best_model, best_order, best_rmse, best_aic = best_arima_by_rmse_then_aic(
            y_train,
            p_candidates,
            d_candidates,
            q_candidates,
            max_backtest_points,
            min_train_for_backtest,
        )

        if best_model is None:
            modelinfo_rows.append(
                {
                    "station": station,
                    "x": x,
                    "y": ycoord,
                    "status": "failed (no model converged)",
                    "order": "",
                    "rmse": np.nan,
                    "aic": np.nan,
                    "train_start": int(y_train.index.min()),
                    "train_end": int(y_train.index.max()),
                }
            )
            continue

        train_start = int(y_train.index.min())
        train_end = int(y_train.index.max())

        steps = int(max(0, target_year - train_end))
        if steps == 0:
            modelinfo_rows.append(
                {
                    "station": station,
                    "x": x,
                    "y": ycoord,
                    "status": "ok (train_end >= target)",
                    "order": str(best_order),
                    "rmse": float(best_rmse),
                    "aic": float(best_aic),
                    "train_start": train_start,
                    "train_end": train_end,
                }
            )
            continue

        fc = best_model.get_forecast(steps=steps)
        mean = np.asarray(fc.predicted_mean)

        ci68 = fc.conf_int(alpha=alpha_68)
        ci30 = fc.conf_int(alpha=alpha_30)

        lo68 = np.asarray(ci68.iloc[:, 0])
        hi68 = np.asarray(ci68.iloc[:, 1])
        lo30 = np.asarray(ci30.iloc[:, 0])
        hi30 = np.asarray(ci30.iloc[:, 1])

        fy = np.arange(train_end + 1, train_end + steps + 1, dtype=int)

        for j, year in enumerate(fy):
            forecasts_rows.append(
                {
                    "station": station,
                    "x": x,
                    "y": ycoord,
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
                }
            )

        modelinfo_rows.append(
            {
                "station": station,
                "x": x,
                "y": ycoord,
                "status": "ok",
                "order": str(best_order),
                "rmse": float(best_rmse),
                "aic": float(best_aic),
                "train_start": train_start,
                "train_end": train_end,
            }
        )

    progress.empty()
    status.empty()

    fc_df = pd.DataFrame(forecasts_rows)
    mi_df = pd.DataFrame(modelinfo_rows)
    return fc_df, mi_df


def build_hist_series(hist_df: pd.DataFrame, station: str) -> pd.Series:
    rows = hist_df[hist_df["station"].astype(str).str.strip() == str(station).strip()]
    if rows.empty:
        return pd.Series(dtype=float)

    row = rows.iloc[0]
    year_cols = sorted([c for c in hist_df.columns if is_year_col(c)], key=year_int)
    data = {year_int(c): row[c] for c in year_cols}
    s = pd.to_numeric(pd.Series(data), errors="coerce").dropna()
    return s.sort_index()


def make_forecast_excel(fc_df: pd.DataFrame, mi_df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        fc_df.to_excel(writer, sheet_name=DEFAULT_OUTPUT_FORECAST_SHEET, index=False)
        mi_df.to_excel(writer, sheet_name=DEFAULT_OUTPUT_MODELINFO_SHEET, index=False)
    buffer.seek(0)
    return buffer.read()


def make_plot(hist_df: pd.DataFrame, fc_df: pd.DataFrame, station: str, old_year: int, new_year: int,
              show_outer: bool, show_inner: bool, rotate_labels: bool, show_bounds_lines: bool):
    y_hist = build_hist_series(hist_df, station)
    if y_hist.empty:
        return None, "No historical data for this station."

    fc_station = fc_df[fc_df["station"].astype(str).str.strip() == str(station).strip()].copy()
    if fc_station.empty:
        return None, "No forecast results for this station."
    fc_station = fc_station.sort_values("year")

    train_end = int(fc_station["train_end"].iloc[0]) if "train_end" in fc_station.columns else int(y_hist.index.max())

    y_hist_plot = y_hist[(y_hist.index >= old_year) & (y_hist.index <= new_year)]
    fc_plot = fc_station[(fc_station["year"] >= old_year) & (fc_station["year"] <= new_year)].copy()

    fig = go.Figure()

    if not fc_plot.empty:
        x_fc = fc_plot["year"].to_numpy(int)
        y_fc = fc_plot["forecast"].to_numpy(float)
        lo68 = fc_plot["lo_68"].to_numpy(float)
        hi68 = fc_plot["hi_68"].to_numpy(float)
        lo30 = fc_plot["lo_30"].to_numpy(float)
        hi30 = fc_plot["hi_30"].to_numpy(float)

        if show_outer:
            fig.add_trace(go.Scatter(x=x_fc, y=hi68, mode="lines", line=dict(width=0), name="", showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=x_fc, y=lo68, mode="lines", line=dict(width=0), fill="tonexty", name="Outer band (68%)"))

        if show_inner:
            fig.add_trace(go.Scatter(x=x_fc, y=hi30, mode="lines", line=dict(width=0), name="", showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=x_fc, y=lo30, mode="lines", line=dict(width=0), fill="tonexty", name="Inner band (30%)"))

        if show_bounds_lines:
            fig.add_trace(go.Scatter(x=x_fc, y=hi68, mode="lines+markers", name="Upper 68"))
            fig.add_trace(go.Scatter(x=x_fc, y=hi30, mode="lines+markers", name="Upper 30"))
            fig.add_trace(go.Scatter(x=x_fc, y=lo30, mode="lines+markers", name="Lower 30"))
            fig.add_trace(go.Scatter(x=x_fc, y=lo68, mode="lines+markers", name="Lower 68"))

        fig.add_trace(go.Scatter(x=x_fc, y=y_fc, mode="lines+markers", name="Forecast"))

    x_h = y_hist_plot.index.to_numpy(int)
    y_h = y_hist_plot.to_numpy(float)
    fig.add_trace(go.Scatter(x=x_h, y=y_h, mode="lines+markers", name="Historical"))

    if len(y_h) > 0 and old_year <= train_end <= new_year:
        fig.add_vline(x=train_end, line_width=1, line_dash="dot")
        fig.add_annotation(x=train_end, y=float(np.nanmax(y_h)), text=f"Train end: {train_end}", showarrow=False, yanchor="bottom")

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Water level (m)",
        hovermode="x unified",
        height=580,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig.update_xaxes(type="linear", tickmode="linear", dtick=1, tickangle=45 if rotate_labels else 0)
    fig.update_yaxes(tickmode="linear", dtick=0.5, showgrid=True, minor=dict(tickmode="linear", dtick=0.1, showgrid=True))

    return fig, None


# ------------------------------
# App UI
# ------------------------------

st.set_page_config(page_title="ARIMA Groundwater Forecast App", layout="wide")
require_login()
st.title("ARIMA Groundwater Forecast App")
st.caption("Upload the historical Excel file, run the forecast, review plots, and download the results workbook.")

with st.expander("Input format expected", expanded=False):
    st.markdown(
        """
        The uploaded Excel sheet should contain:
        - columns **station**, **x**, **y**
        - yearly columns such as **1988, 1989, ..., 2025**
        - one row per station
        """
    )

uploaded_file = st.file_uploader("Upload historical Excel file", type=["xlsx", "xlsm", "xls"])

with st.sidebar:
    st.header("Forecast settings")
    input_sheet = st.text_input("Input sheet name", value=DEFAULT_INPUT_SHEET)
    min_year = st.number_input("Minimum year", value=DEFAULT_MIN_YEAR, step=1)
  #   min_obs = st.number_input("Minimum consecutive observations", value=DEFAULT_MIN_OBS, step=1)
    target_year = st.number_input("Target forecast year", value=DEFAULT_TARGET_YEAR, step=1)
   #  max_backtest_points = st.number_input("Max backtest points", value=DEFAULT_MAX_BACKTEST_POINTS, step=1)
   #  min_train_for_backtest = st.number_input("Min train length for backtest", value=DEFAULT_MIN_TRAIN_FOR_BACKTEST, step=1)
    alpha_68 = st.number_input("Alpha for UB", value=DEFAULT_ALPHA_68, min_value=0.0, max_value=1.0)
    alpha_30 = st.number_input("Alpha forLB", value=DEFAULT_ALPHA_30, min_value=0.0, max_value=1.0)

    st.subheader("ARIMA candidate orders")
    p_candidates = st.multiselect("p candidates", options=list(range(0, 6)), default=DEFAULT_P)
    d_candidates = st.multiselect("d candidates", options=list(range(0, 4)), default=DEFAULT_D)
    q_candidates = st.multiselect("q candidates", options=list(range(0, 6)), default=DEFAULT_Q)

run_button = st.button("Run forecast", type="primary", disabled=uploaded_file is None)

if "hist_df" not in st.session_state:
    st.session_state.hist_df = None
if "fc_df" not in st.session_state:
    st.session_state.fc_df = None
if "mi_df" not in st.session_state:
    st.session_state.mi_df = None

if run_button and uploaded_file is not None:
    if not p_candidates or not d_candidates or not q_candidates:
        st.error("Please choose at least one value for p, d, and q.")
    else:
        try:
            hist_df = read_excel_from_upload(uploaded_file, input_sheet)
            fc_df, mi_df = run_forecast(
                hist_df,
                int(min_year),
                int(min_obs),
                int(target_year),
                [int(x) for x in p_candidates],
                [int(x) for x in d_candidates],
                [int(x) for x in q_candidates],
                int(max_backtest_points),
                int(min_train_for_backtest),
                float(alpha_68),
                float(alpha_30),
            )
            st.session_state.hist_df = hist_df
            st.session_state.fc_df = fc_df
            st.session_state.mi_df = mi_df
            st.success("Forecast completed.")
        except Exception as e:
            st.exception(e)

hist_df = st.session_state.hist_df
fc_df = st.session_state.fc_df
mi_df = st.session_state.mi_df

if hist_df is not None:
    st.subheader("Input preview")
    st.dataframe(hist_df.head(10), use_container_width=True)

if fc_df is not None and mi_df is not None:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Forecast results")
    with c2:
        st.download_button(
            label="Download forecast workbook",
            data=make_forecast_excel(fc_df, mi_df),
            file_name="arima_yearmax_forecast.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    tab1, tab2, tab3 = st.tabs(["Plot viewer", "Forecast table", "Model info"])

    with tab1:
        stations = sorted(set(hist_df["station"].astype(str)) | set(fc_df["station"].astype(str)))
        station = st.selectbox("Choose station", stations)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            show_outer = st.checkbox("Show UB", value=True)
        with c2:
            show_inner = st.checkbox("Show LB", value=True)
        with c3:
            rotate_labels = st.checkbox("Rotate x-axis labels", value=True)
        with c4:
            show_bounds_lines = st.checkbox("Show bound lines", value=True)

        y_hist = build_hist_series(hist_df, station)
        fc_station = fc_df[fc_df["station"].astype(str).str.strip() == str(station).strip()].copy()

        if not y_hist.empty and not fc_station.empty:
            hist_min = int(y_hist.index.min())
            hist_max = int(y_hist.index.max())
            fc_min = int(fc_station["year"].min())
            fc_max = int(fc_station["year"].max())
            global_min = min(hist_min, fc_min)
            global_max = max(hist_max, fc_max)
            default_old = DEFAULT_OLD_YEAR if global_min <= DEFAULT_OLD_YEAR <= global_max else global_min

            old_year, new_year = st.slider(
                "Select years to display",
                min_value=int(global_min),
                max_value=int(global_max),
                value=(int(default_old), int(global_max)),
                step=1,
            )

            fig, err = make_plot(
                hist_df,
                fc_df,
                station,
                old_year,
                new_year,
                show_outer,
                show_inner,
                rotate_labels,
                show_bounds_lines,
            )
            if err:
                st.warning(err)
            else:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No plot can be generated for this station.")

    with tab2:
        st.dataframe(fc_df, use_container_width=True)

    with tab3:
        st.dataframe(mi_df, use_container_width=True)
