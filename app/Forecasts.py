"""Forecast viewer - the landing page.

Pick a station, a pollutant and a forecast origin; see what each model predicted
for the next 24 hours against what was actually measured.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import theme as T

T.setup("Forecast viewer")

CONTEXT_HOURS = 72


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
panel = T.load_panel()
fc = T.load_forecasts()

st.markdown("# Air quality forecasting · Bosnia and Herzegovina")
st.markdown(
    '<p class="lede">24-hour-ahead forecasts for six pollutants across 23 monitoring '
    'stations, scored on held-out 2024 data. Pick a station, a pollutant and a day to '
    'see what each model predicted against what was actually measured.</p>',
    unsafe_allow_html=True)

if fc is None:
    T.missing_forecasts_notice()
    st.stop()

MODELS = T.model_columns(fc)


# --------------------------------------------------------------------------
# controls
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Select a forecast")

    stations = sorted(fc.station.astype(str).unique())
    station = st.selectbox("Station", stations)

    sub = fc[fc.station.astype(str) == station]
    pollutants = sorted(sub.pollutant.astype(str).unique())
    pollutant = st.selectbox(
        "Pollutant", pollutants,
        format_func=lambda p: T.POLLUTANT_LABELS.get(p, p.upper()))

    sub = sub[sub.pollutant.astype(str) == pollutant]
    origins = sorted(sub.origin.unique())
    origin = st.select_slider(
        "Forecast day", options=origins, value=origins[len(origins) // 2],
        format_func=lambda d: pd.Timestamp(d).strftime("%d %b %Y"))

    st.markdown("### Show")
    shown = [m for m in MODELS
             if st.checkbox(T.MODEL_LABELS[m], value=m in ("chronos2", "ensemble"),
                            key=f"show_{m}")]
    band = st.checkbox("Uncertainty band (10th–90th pct)", value=True)
    st.caption("Bands are available for Chronos-2 and the ensemble.")

w = sub[sub.origin == origin].sort_values("hour")
if w.empty:
    st.warning("No forecast for that combination.")
    st.stop()

unit = T.UNITS[pollutant]
pname = T.POLLUTANT_LABELS.get(pollutant, pollutant.upper())
origin_ts = pd.Timestamp(origin)


# --------------------------------------------------------------------------
# scores for this window
# --------------------------------------------------------------------------
scored = w.scored.to_numpy(bool)
actual = w.actual.to_numpy(float)
scale = float(w.mase_scale.iloc[0])


def window_scores(col: str) -> dict[str, float]:
    pred = w[col].to_numpy(float)
    err = np.where(scored, pred - actual, np.nan)
    if not np.isfinite(err).any():
        return {"mae": np.nan, "mase": np.nan}
    mae = float(np.nanmean(np.abs(err)))
    return {"mae": mae,
            "mase": mae / scale if np.isfinite(scale) and scale > 0 else np.nan}


scores = {m: window_scores(m) for m in MODELS if w[m].notna().any()}
best = min(scores, key=lambda m: scores[m]["mase"]) if scores else None

card_items = [("Station", station, f"{w.city.iloc[0]}"),
              ("Pollutant", pname, unit),
              ("Forecast from", origin_ts.strftime("%d %b %Y"),
               f"{int(scored.sum())} of 24 h scorable")]
order = [m for m in MODELS if m in scores]
for m in order[:3]:
    s = scores[m]
    card_items.append((T.MODEL_LABELS[m], f"{s['mase']:.2f}",
                       f"MASE · MAE {s['mae']:.1f}"))
hi = next((i for i, c in enumerate(card_items)
           if best and c[0] == T.MODEL_LABELS[best]), None)
T.cards(card_items, highlight=hi)


# --------------------------------------------------------------------------
# the chart
# --------------------------------------------------------------------------
hist = panel[(panel.station.astype(str) == station)
             & (panel.datetime >= origin_ts - pd.Timedelta(hours=CONTEXT_HOURS))
             & (panel.datetime < origin_ts)]
future_ts = pd.date_range(origin_ts, periods=24, freq="h")

fig = go.Figure()

# context: what the model saw
fig.add_trace(go.Scatter(
    x=hist.datetime, y=hist[pollutant], name="Observed (context)",
    mode="lines", line=dict(color=T.MUTED, width=1.6),
    hovertemplate="%{x|%d %b %H:%M}<br>%{y:.1f} " + unit + "<extra></extra>"))

# uncertainty bands, drawn first so lines sit on top
if band:
    for m in shown:
        lo, hi_c = f"{m}_p10", f"{m}_p90"
        if lo not in w.columns or w[lo].isna().all():
            continue
        c = T.MODEL_COLORS[m]
        rgba = f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.13)"
        fig.add_trace(go.Scatter(
            x=list(future_ts) + list(future_ts[::-1]),
            y=list(w[hi_c]) + list(w[lo][::-1]),
            fill="toself", fillcolor=rgba, line=dict(width=0),
            name=f"{T.MODEL_LABELS[m]} 10–90%", hoverinfo="skip",
            showlegend=False))

# actuals over the forecast horizon, filled points excluded
fig.add_trace(go.Scatter(
    x=future_ts, y=np.where(scored, actual, np.nan), name="Actual",
    mode="lines+markers", line=dict(color=T.INK, width=2.4),
    marker=dict(size=5),
    hovertemplate="%{x|%d %b %H:%M}<br>actual %{y:.1f} " + unit + "<extra></extra>"))

for m in shown:
    if w[m].isna().all():
        continue
    fig.add_trace(go.Scatter(
        x=future_ts, y=w[m], name=T.MODEL_LABELS[m], mode="lines",
        line=dict(color=T.MODEL_COLORS[m], width=2.2, dash="dash"),
        hovertemplate="%{x|%d %b %H:%M}<br>%{y:.1f} " + unit + "<extra></extra>"))

fig.add_vline(x=origin_ts, line=dict(color=T.LINE, width=1.5, dash="dot"))
fig.add_annotation(x=origin_ts, y=1.02, yref="paper", text="forecast starts",
                   showarrow=False, font=dict(size=11, color=T.MUTED),
                   xanchor="left", xshift=6)

fig.update_layout(
    height=460, margin=dict(l=10, r=10, t=44, b=10),
    title=dict(text=f"{pname} at {station} — 24 hours from "
                    f"{origin_ts:%d %b %Y}", font=dict(size=15)),
    hovermode="x unified", plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0,
                bgcolor="rgba(0,0,0,0)"),
    yaxis=dict(title=f"{pname} ({unit})", gridcolor=T.LINE, zeroline=False),
    xaxis=dict(gridcolor=T.LINE, zeroline=False))
st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------
# detail
# --------------------------------------------------------------------------
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown("#### Error through the horizon")
    ef = go.Figure()
    for m in shown:
        if w[m].isna().all():
            continue
        err = np.where(scored, np.abs(w[m].to_numpy(float) - actual), np.nan)
        ef.add_trace(go.Scatter(
            x=np.arange(1, 25), y=err, name=T.MODEL_LABELS[m], mode="lines+markers",
            line=dict(color=T.MODEL_COLORS[m], width=2), marker=dict(size=4)))
    ef.update_layout(
        height=270, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False, hovermode="x unified",
        yaxis=dict(title=f"absolute error ({unit})", gridcolor=T.LINE),
        xaxis=dict(title="hours ahead", gridcolor=T.LINE, dtick=6))
    st.plotly_chart(ef, width="stretch")
    st.caption("Gaps are hours excluded from scoring — the value was missing or "
               "was interpolated when the dataset was built.")

with right:
    st.markdown("#### This window")
    tbl = pd.DataFrame([
        {"Model": T.MODEL_LABELS[m], "MASE": scores[m]["mase"],
         f"MAE ({unit})": scores[m]["mae"]}
        for m in order
    ]).sort_values("MASE")
    st.dataframe(
        tbl, hide_index=True, width="stretch",
        column_config={
            "MASE": st.column_config.ProgressColumn(
                "MASE", format="%.3f", min_value=0.0,
                max_value=float(max(1.2, tbl["MASE"].max()))),
            f"MAE ({unit})": st.column_config.NumberColumn(format="%.2f"),
        })
    st.caption("MASE below 1.0 beats the seasonal-naive forecast for this window. "
               "One window is noisy — the headline numbers average 31,688 of them.")


# --------------------------------------------------------------------------
# how this station scores overall
# --------------------------------------------------------------------------
st.markdown("#### How this series scores across the whole year")

series = fc[(fc.station.astype(str) == station)
            & (fc.pollutant.astype(str) == pollutant)]
rows = []
for m in MODELS:
    if series[m].isna().all():
        continue
    err = np.where(series.scored, np.abs(series[m] - series.actual), np.nan)
    per = pd.DataFrame({"origin": series.origin.to_numpy(), "err": err,
                        "scale": series.mase_scale.to_numpy()})
    g = per.groupby("origin").agg(mae=("err", "mean"), scale=("scale", "first"))
    rows.append({"model": T.MODEL_LABELS[m],
                 "MASE": float((g.mae / g.scale).mean()),
                 f"MAE ({unit})": float(g.mae.mean()),
                 "windows": int(len(g))})

if rows:
    summary = pd.DataFrame(rows).sort_values("MASE")
    bar = go.Figure(go.Bar(
        x=summary["MASE"], y=summary.model, orientation="h",
        marker_color=[T.MODEL_COLORS[k] for k in
                      [next(kk for kk, vv in T.MODEL_LABELS.items() if vv == n)
                       for n in summary.model]],
        text=[f"{v:.3f}" for v in summary["MASE"]], textposition="outside",
        hovertemplate="%{y}<br>MASE %{x:.3f}<extra></extra>"))
    bar.add_vline(x=1.0, line=dict(color="#d4803f", width=1.4, dash="dash"))
    bar.update_layout(
        height=60 + 42 * len(summary), margin=dict(l=10, r=60, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="MASE (lower is better · 1.0 = seasonal naive)",
                   gridcolor=T.LINE, zeroline=False),
        yaxis=dict(autorange="reversed"))
    st.plotly_chart(bar, width="stretch")
    st.caption(f"{pname} at {station}, averaged over "
               f"{summary.windows.iloc[0]:,} forecast windows in 2024.")
