"""Station map and explorer.

The 23 monitoring stations on a map, coloured by whichever measure you pick,
with a detail view for whichever one you click through to.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import theme as T

T.setup("Stations")

stations = T.load_stations()
panel = T.load_panel()
fc = T.load_forecasts()

st.markdown("# Monitoring stations")

POLLUTANTS = ["pm10", "pm25", "so2", "no2", "o3", "co"]


# --------------------------------------------------------------------------
# controls
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Colour the map by")
    metric = st.radio(
        "Measure", ["Coverage", "Forecast skill"],
        captions=["Share of 2024 hours with a real reading",
                  "Mean MASE — needs forecast data"],
        label_visibility="collapsed")
    pollutant = st.selectbox(
        "Pollutant", POLLUTANTS,
        format_func=lambda p: T.POLLUTANT_LABELS[p])
    if metric == "Forecast skill" and fc is not None:
        model = st.selectbox(
            "Model", T.model_columns(fc),
            format_func=lambda m: T.MODEL_LABELS[m])
    else:
        model = None


# --------------------------------------------------------------------------
# the value each station is coloured by
# --------------------------------------------------------------------------
df = stations.copy()
cov_col = f"{pollutant}_coverage"
df["coverage"] = df.get(cov_col, pd.Series(np.nan, index=df.index)).fillna(0.0)

if metric == "Forecast skill" and fc is not None and model is not None:
    sub = fc[fc.pollutant.astype(str) == pollutant]
    if not sub.empty:
        err = np.where(sub.scored, np.abs(sub[model] - sub.actual), np.nan)
        per = pd.DataFrame({"station": sub.station.astype(str).to_numpy(),
                            "origin": sub.origin.to_numpy(), "err": err,
                            "scale": sub.mase_scale.to_numpy()})
        g = (per.groupby(["station", "origin"])
                .agg(mae=("err", "mean"), scale=("scale", "first")))
        g["mase"] = g.mae / g.scale
        skill = g.groupby("station").mase.mean().rename("value")
        df = df.merge(skill, left_on="station", right_index=True, how="left")
    else:
        df["value"] = np.nan
    colorbar, scale_name, reverse = "MASE", "RdYlBu", True
    hover_extra = "MASE %{customdata[1]:.3f}"
else:
    df["value"] = df["coverage"]
    colorbar, scale_name, reverse = "Coverage %", "Blues", False
    hover_extra = "coverage %{customdata[1]:.0f}%"

has_value = df.value.notna() & (df.value > 0)
# Three distinct states, which the earlier version wrongly collapsed into two:
#   measured in 2024        -> coloured
#   has the instrument but
#   stopped before 2024     -> hollow marker, "no 2024 data"
#   never had the instrument -> grey
ever_col = f"{pollutant}_ever"
df["ever"] = df.get(ever_col, pd.Series(False, index=df.index)).fillna(False)
inactive = ~has_value & df.ever
never = ~has_value & ~df.ever


# --------------------------------------------------------------------------
# headline cards
# --------------------------------------------------------------------------
measured = int((df.coverage > 0).sum())

if metric == "Forecast skill" and has_value.any():
    beat = int((df.loc[has_value, "value"] < 1.0).sum())
    third = ("Beating the baseline", f"{beat} of {int(has_value.sum())}",
             "stations with MASE below 1.0")
elif measured:
    third = ("Mean coverage", f"{df.coverage[df.coverage > 0].mean():.0f}%",
             "where the pollutant is measured")
else:
    third = ("Mean coverage", "—", "not measured anywhere")

T.cards([
    ("Stations", "23", "12 cities, two networks"),
    (f"Measuring {T.POLLUTANT_LABELS[pollutant]}", str(measured), "of 23 stations"),
    third,
])


# --------------------------------------------------------------------------
# map
# --------------------------------------------------------------------------
plot_df = df[has_value].copy()
grey_df = df[never].copy()
off_df = df[inactive].copy()

fig = go.Figure()

if not grey_df.empty:
    fig.add_trace(go.Scattermap(
        lat=grey_df.lat, lon=grey_df.lon, mode="markers",
        marker=dict(size=11, color="#4a555f"),
        name="no instrument",
        customdata=np.stack([grey_df.station, grey_df.city], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}"
                      "<br>no instrument for this pollutant<extra></extra>"))

if not off_df.empty:
    last = off_df.last_seen.dt.strftime("%b %Y").fillna("unknown")
    fig.add_trace(go.Scattermap(
        lat=off_df.lat, lon=off_df.lon, mode="markers",
        marker=dict(size=13, color="#d4803f"),
        name="stopped before 2024",
        customdata=np.stack([off_df.station, off_df.city, last], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}"
                      "<br>measured this pollutant, but stopped "
                      "— last reading %{customdata[2]}<extra></extra>"))

if not plot_df.empty:
    fig.add_trace(go.Scattermap(
        lat=plot_df.lat, lon=plot_df.lon, mode="markers",
        marker=dict(size=17, color=plot_df.value, colorscale=scale_name,
                    reversescale=reverse, showscale=True,
                    colorbar=dict(title=colorbar, thickness=12, len=.7,
                                  outlinewidth=0)),
        name="stations",
        customdata=np.stack([plot_df.station, plot_df.value,
                             plot_df.city, plot_df.station_type], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[2]} · "
                      "%{customdata[3]}<br>" + hover_extra + "<extra></extra>"))

fig.update_layout(
    map=dict(style="carto-darkmatter", center=dict(lat=44.1, lon=17.8), zoom=6.5),
    height=520, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
st.plotly_chart(fig, width="stretch")

if metric == "Forecast skill" and fc is None:
    T.missing_forecasts_notice()


# --------------------------------------------------------------------------
# station detail
# --------------------------------------------------------------------------
st.markdown("### Station detail")

alive = stations[stations.active_2024].station
opts = sorted(stations.station)
# Default to a station that still reported in 2024 - one station (Ambasada)
# stopped before the evaluation year and would otherwise open on an empty view.
default = opts.index(sorted(alive)[0]) if len(alive) else 0

pick = st.selectbox("Station", opts, index=default,
                    label_visibility="collapsed")
row = stations[stations.station == pick].iloc[0]

if pick not in set(alive):
    ever_list = [T.POLLUTANT_LABELS[p] for p in POLLUTANTS
                 if bool(row.get(f"{p}_ever", False))]
    last_seen = row.get("last_seen")
    when = (pd.Timestamp(last_seen).strftime("%d %B %Y")
            if pd.notna(last_seen) else "an unknown date")
    st.markdown(
        f'<div class="note"><b>{pick} stopped reporting before the evaluation '
        f'year.</b><br>Last reading {when}. It measured '
        f'{", ".join(ever_list) if ever_list else "nothing"} while active, and '
        'its data contributed to training, but it has no 2024 measurements — so '
        'the coverage figures below, which describe 2024, are all zero.'
        '</div>', unsafe_allow_html=True)

c1, c2 = st.columns([2, 3], gap="large")

with c1:
    st.markdown(f"**{pick}** · {row.city}")
    meta = {"Network": str(row.source).upper(),
            "Type": row.station_type,
            "Latitude": f"{row.lat:.4f}",
            "Longitude": f"{row.lon:.4f}"}
    if pd.notna(row.get("elev_m")):
        meta["Elevation"] = f"{row.elev_m:.0f} m"
    if pd.notna(row.get("code")) and str(row.get("code")) != "nan":
        meta["EEA code"] = row.code
    st.dataframe(
        pd.DataFrame({"": list(meta.keys()), " ": list(meta.values())}),
        hide_index=True, width="stretch")

with c2:
    cov = {T.POLLUTANT_LABELS[p]: float(row.get(f"{p}_coverage", 0) or 0)
           for p in POLLUTANTS}
    cov_df = pd.DataFrame({"pollutant": list(cov), "coverage": list(cov.values())})
    bar = px.bar(cov_df, x="coverage", y="pollutant", orientation="h",
                 range_x=[0, 100], text=[f"{v:.0f}%" for v in cov_df.coverage])
    colours = []
    for p, v in zip(POLLUTANTS, cov_df.coverage):
        if v > 0:
            colours.append(T.ACCENT)
        elif bool(row.get(f"{p}_ever", False)):
            colours.append(T.ACCENT_WARM)      # had the instrument, not in 2024
        else:
            colours.append("#4a555f")      # never had it
    bar.update_traces(marker_color=colours, textposition="outside",
                      cliponaxis=False)
    bar.update_layout(
        height=260, margin=dict(l=0, r=30, t=24, b=0),
        title=dict(text="2024 coverage by pollutant", font=dict(size=13)),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="", gridcolor=T.LINE, ticksuffix="%"),
        yaxis=dict(title="", autorange="reversed"))
    st.plotly_chart(bar, width="stretch")
    st.caption("Grey means the station has no instrument for that pollutant. "
               "Amber means it measured it at some point, but not during 2024.")


# --------------------------------------------------------------------------
# the year at this station
# --------------------------------------------------------------------------
avail = [p for p in POLLUTANTS if float(row.get(f"{p}_coverage", 0) or 0) > 0]
if avail:
    st.markdown("### The measured year")
    show_p = st.multiselect(
        "Pollutants", avail, default=avail[:2],
        format_func=lambda p: T.POLLUTANT_LABELS[p])

    series = panel[(panel.station.astype(str) == pick)
                   & (panel.datetime.dt.year == 2024)]
    if show_p and not series.empty:
        # Daily means keep a year of hourly data readable without hiding episodes.
        daily = (series.set_index("datetime")[show_p]
                       .resample("D").mean().reset_index())
        ts = go.Figure()
        palette = ["#1f5c8b", "#c2703a", "#5b8c5a", "#a97fc4", "#cf6b5a", "#7fb8e0"]
        for i, p in enumerate(show_p):
            ts.add_trace(go.Scatter(
                x=daily.datetime, y=daily[p], name=T.POLLUTANT_LABELS[p],
                mode="lines", line=dict(width=1.7, color=palette[i % len(palette)]),
                hovertemplate="%{x|%d %b}<br>%{y:.1f}<extra></extra>"))
        ts.update_layout(
            height=300, margin=dict(l=10, r=10, t=10, b=10),
            hovermode="x unified", plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.05, x=0, bgcolor="rgba(0,0,0,0)"),
            yaxis=dict(title="daily mean concentration", gridcolor=T.LINE),
            xaxis=dict(gridcolor=T.LINE))
        st.plotly_chart(ts, width="stretch")
        st.caption("Daily means over 2024. Gaps are periods with no measurement.")
