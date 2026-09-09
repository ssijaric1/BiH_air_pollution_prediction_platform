"""Shared look and helpers for the app.

One place for the palette, the CSS and the data loading, so the two pages stay
visually identical and neither reads a parquet file twice.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

DATA = Path(__file__).resolve().parent / "data"

# The model palette is fixed here so a model is the same colour on every chart.
# The app is dark by default and stays dark - a single committed palette rather
# than two half-tuned ones, which is what produced black-on-charcoal text.
INK = "#e8edf2"          # primary text
MUTED = "#8d9aa8"        # secondary text, context lines
LINE = "#2b343d"         # borders and chart gridlines
SURFACE = "#1c232a"      # cards, sidebar, panels
CANVAS = "#141a20"       # page background
ACCENT = "#4a93cc"       # blue
ACCENT_WARM = "#d4803f"  # orange

MODEL_COLORS = {
    "chronos2": "#4a93cc",
    "chronos2_nbr": "#7fb8e0",
    "ensemble": "#d4803f",
    "gnn": "#6faa6d",
    "diurnal7": "#8d9aa8",
}
MODEL_LABELS = {
    "chronos2": "Chronos-2",
    "chronos2_nbr": "Chronos-2 + neighbours",
    "ensemble": "Ensemble",
    "gnn": "Spatial GNN",
    "diurnal7": "Diurnal reference",
}
POLLUTANT_LABELS = {
    "pm10": "PM10", "pm25": "PM2.5", "so2": "SO₂",
    "no2": "NO₂", "o3": "O₃", "co": "CO",
}
UNITS = {p: "µg/m³" for p in POLLUTANT_LABELS}
UNITS["co"] = "mg/m³"

CSS = f"""
<style>
  .stApp {{ background: {CANVAS}; color: {INK}; }}
  #MainMenu, footer, header {{ visibility: hidden; }}
  .block-container {{ padding-top: 2.2rem; max-width: 1250px; }}

  h1, h2, h3, h4, h5 {{ color: {INK}; letter-spacing: -.02em; font-weight: 650; }}
  h1 {{ font-size: 1.85rem; margin-bottom: .15rem; }}
  p, li, label, .stMarkdown {{ color: {INK}; }}

  .lede {{ color: {MUTED}; font-size: .95rem; margin: 0 0 1.4rem 0;
           max-width: 62ch; line-height: 1.55; }}

  .cards {{ display: flex; gap: .8rem; flex-wrap: wrap; margin: .2rem 0 1.3rem; }}
  .card {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 12px;
           padding: .85rem 1.1rem; flex: 1 1 150px; min-width: 140px; }}
  .card .k {{ color: {MUTED}; font-size: .72rem; text-transform: uppercase;
              letter-spacing: .07em; font-weight: 600; }}
  .card .v {{ color: {INK}; font-size: 1.5rem; font-weight: 660;
              line-height: 1.25; margin-top: .18rem; }}
  .card .s {{ color: {MUTED}; font-size: .78rem; }}
  .card.hi {{ border-color: {ACCENT}; box-shadow: 0 0 0 1px {ACCENT}33; }}

  .panel {{ background: {SURFACE}; border: 1px solid {LINE};
            border-radius: 14px; padding: 1.1rem 1.2rem; margin-bottom: 1rem; }}

  .note {{ background: #2a2318; border: 1px solid #4a3d22; color: #e8c98a;
           border-radius: 10px; padding: .8rem 1rem; font-size: .87rem;
           line-height: 1.5; }}
  .note b {{ color: #f2dda8; }}

  [data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {LINE}; }}
  [data-testid="stSidebar"] .block-container {{ padding-top: 1.4rem; }}
  [data-testid="stCaptionContainer"], .stCaption {{ color: {MUTED} !important; }}

  .stTabs [data-baseweb="tab-list"] {{ gap: .3rem; border-bottom: 1px solid {LINE}; }}
  .stTabs [data-baseweb="tab"] {{ font-size: .9rem; font-weight: 550; color: {MUTED}; }}
  .stTabs [aria-selected="true"] {{ color: {INK}; }}

  div[data-testid="stDataFrame"] {{ border: 1px solid {LINE}; border-radius: 10px; }}
</style>
"""


def _register_template() -> None:
    """One plotly template for the whole app, so no chart carries default
    (dark-on-dark) text or gridlines."""
    import plotly.graph_objects as go
    import plotly.io as pio

    pio.templates["bih"] = go.layout.Template(layout=dict(
        font=dict(color=INK, size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(font=dict(color=INK, size=15)),
        xaxis=dict(gridcolor=LINE, linecolor=LINE, zerolinecolor=LINE,
                   tickfont=dict(color=MUTED), title=dict(font=dict(color=MUTED))),
        yaxis=dict(gridcolor=LINE, linecolor=LINE, zerolinecolor=LINE,
                   tickfont=dict(color=MUTED), title=dict(font=dict(color=MUTED))),
        legend=dict(font=dict(color=MUTED), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=LINE,
                        font=dict(color=INK)),
        coloraxis=dict(colorbar=dict(tickfont=dict(color=MUTED),
                                     title=dict(font=dict(color=MUTED)))),
    ))
    pio.templates.default = "bih"


def setup(title: str) -> None:
    st.set_page_config(page_title=f"{title} · BiH Air Quality",
                       page_icon="🌫️", layout="wide")
    _register_template()
    st.markdown(CSS, unsafe_allow_html=True)


def cards(items: list[tuple[str, str, str]], highlight: int | None = None) -> None:
    """items: (label, value, sub). highlight: index to outline, or None."""
    html = ['<div class="cards">']
    for i, (k, v, s) in enumerate(items):
        cls = "card hi" if i == highlight else "card"
        html.append(f'<div class="{cls}"><div class="k">{k}</div>'
                    f'<div class="v">{v}</div><div class="s">{s}</div></div>')
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_panel() -> pd.DataFrame:
    return pd.read_parquet(DATA / "panel.parquet")


@st.cache_data(show_spinner=False)
def load_stations() -> pd.DataFrame:
    return pd.read_parquet(DATA / "stations.parquet")


@st.cache_data(show_spinner=False)
def load_forecasts() -> pd.DataFrame | None:
    f = DATA / "forecasts.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f)
    df["origin"] = pd.to_datetime(df["origin"])
    return df


def model_columns(fc: pd.DataFrame) -> list[str]:
    """Model columns actually present, in the order we want them drawn."""
    return [m for m in MODEL_COLORS if m in fc.columns]


def missing_forecasts_notice() -> None:
    st.markdown(
        '<div class="note"><b>No forecast data yet.</b><br>'
        'Run the final cell of <code>notebooks/07_ensemble/02_ensemble_full.ipynb</code> '
        'in Colab, put the <code>forecasts.parquet</code> it writes to Drive into '
        '<code>results/</code>, then run <code>python scripts/prepare_app_data.py</code>.'
        '</div>', unsafe_allow_html=True)
