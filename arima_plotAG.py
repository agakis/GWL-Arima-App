import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

HIST_BASENAME = "timeseries_yearmax"
HIST_SHEET = "timeseries_yearmax"

FC_BASENAME = "arima_yearmax_forecast"
FC_SHEET = "forecast"

DEFAULT_OLD_YEAR = 1988
DEFAULT_ROTATE = True
DEFAULT_SHOW_INNER = True
DEFAULT_SHOW_OUTER = True


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


@st.cache_data
def load_hist(excel_path: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=HIST_SHEET, engine="openpyxl")
    required = {"station", "x", "y"}
    if not required.issubset(df.columns):
        raise KeyError(f"Historical sheet must contain columns: {required}")
    if not any(is_year_col(c) for c in df.columns):
        raise KeyError("No year columns found in historical file.")
    return df


@st.cache_data
def load_fc(excel_path: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=FC_SHEET, engine="openpyxl")
    required = {"station", "year", "forecast", "lo_68", "hi_68", "lo_30", "hi_30"}
    if not required.issubset(df.columns):
        raise KeyError(f"Forecast sheet must contain columns: {required}")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    return df


def build_hist_series(hist_df: pd.DataFrame, station: str) -> pd.Series:
    rows = hist_df[hist_df["station"].astype(str).str.strip() == str(station).strip()]
    if rows.empty:
        return pd.Series(dtype=float)

    row = rows.iloc[0]
    year_cols = sorted([c for c in hist_df.columns if is_year_col(c)], key=year_int)
    data = {year_int(c): row[c] for c in year_cols}
    s = pd.to_numeric(pd.Series(data), errors="coerce").dropna()
    return s.sort_index()


def main():
    st.set_page_config(page_title="ARIMA Forecast Viewer (Yearly Max)", layout="wide")
    st.title("ARIMA forecast viewer — yearly maximum groundwater levels")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    hist_path = find_excel_file(script_dir, HIST_BASENAME)
    fc_path = find_excel_file(script_dir, FC_BASENAME)

    hist = load_hist(hist_path)
    fc = load_fc(fc_path)

    stations = sorted(set(hist["station"].astype(str)) | set(fc["station"].astype(str)))
    station = st.selectbox("Choose station", stations)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        show_outer = st.checkbox("Show outer band (68%)", value=DEFAULT_SHOW_OUTER)
    with c2:
        show_inner = st.checkbox("Show inner band (30%)", value=DEFAULT_SHOW_INNER)
    with c3:
        rotate_labels = st.checkbox("Rotate x-axis labels", value=DEFAULT_ROTATE)
    with c4:
        show_bounds_lines = st.checkbox("Show bound lines", value=True)

    y_hist = build_hist_series(hist, station)
    if y_hist.empty:
        st.warning("No historical data for this station.")
        return

    fc_station = fc[fc["station"].astype(str).str.strip() == str(station).strip()].copy()
    if fc_station.empty:
        st.warning("No forecast results for this station.")
        return
    fc_station = fc_station.sort_values("year")

    train_end = int(fc_station["train_end"].iloc[0]) if "train_end" in fc_station.columns else int(y_hist.index.max())

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
        step=1
    )

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

        # ---- Outer band (68%) ----
        if show_outer:
            fig.add_trace(go.Scatter(
                x=x_fc, y=hi68,
                mode="lines",
                line=dict(width=0),
                name="",
                showlegend=False,
                hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=x_fc, y=lo68,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                name="Outer band (68%)"
            ))

        # ---- Inner band (30%) ----
        if show_inner:
            fig.add_trace(go.Scatter(
                x=x_fc, y=hi30,
                mode="lines",
                line=dict(width=0),
                name="",
                showlegend=False,
                hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=x_fc, y=lo30,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                name="Inner band (30%)"
            ))

        # Bound lines (as requested)
        if show_bounds_lines:
            fig.add_trace(go.Scatter(x=x_fc, y=hi68, mode="lines+markers", name="Upper 68"))
            fig.add_trace(go.Scatter(x=x_fc, y=hi30, mode="lines+markers", name="Upper 30"))
            fig.add_trace(go.Scatter(x=x_fc, y=lo30, mode="lines+markers", name="Lower 30"))
            fig.add_trace(go.Scatter(x=x_fc, y=lo68, mode="lines+markers", name="Lower 68"))

        # Forecast mean
        fig.add_trace(go.Scatter(x=x_fc, y=y_fc, mode="lines+markers", name="Forecast"))

    # Historical
    x_h = y_hist_plot.index.to_numpy(int)
    y_h = y_hist_plot.to_numpy(float)
    fig.add_trace(go.Scatter(x=x_h, y=y_h, mode="lines+markers", name="Historical"))

    # Train end marker
    if old_year <= train_end <= new_year:
        fig.add_vline(x=train_end, line_width=1, line_dash="dot")
        fig.add_annotation(
            x=train_end,
            y=float(np.nanmax(y_h)),
            text=f"Train end: {train_end}",
            showarrow=False,
            yanchor="bottom"
        )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Water level (m)",
        hovermode="x unified",
        height=580,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    fig.update_xaxes(
        type="linear",
        tickmode="linear",
        dtick=1,
        tickangle=45 if rotate_labels else 0
    )

    fig.update_yaxes(
        tickmode="linear",
        dtick=0.5,
        showgrid=True,
        minor=dict(tickmode="linear", dtick=0.1, showgrid=True)
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show forecast table"):
        st.dataframe(fc_station, use_container_width=True)


if __name__ == "__main__":
    main()