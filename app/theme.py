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
INK = "#0f1720"
MUTED = "#6b7a8c"
LINE = "#dbe3ea"
SURFACE = "#ffffff"
CANVAS = "#f5f7f9"
ACCENT = "#1f5c8b"

MODEL_COLORS = {
    "chronos2": "#1f5c8b",
    "chronos2_nbr": "#3d86bd",
    "ensemble": "#c2703a",
    "gnn": "#5b8c5a",
    "diurnal7": "#9aa7b4",
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
  .stApp {{ background: {CANVAS}; }}
  #MainMenu, footer, header {{ visibility: hidden; }}
  .block-container {{ padding-top: 2.2rem; max-width: 1250px; }}

  h1, h2, h3 {{ color: {INK}; letter-spacing: -.02em; font-weight: 650; }}
  h1 {{ font-size: 1.85rem; margin-bottom: .15rem; }}

  .lede {{ color: {MUTED}; font-size: .95rem; margin: 0 0 1.4rem 0;
           max-width: 62ch; line-height: 1.55; }}

  /* metric cards */
  .cards {{ display: flex; gap: .8rem; flex-wrap: wrap; margin: .2rem 0 1.3rem; }}
  .card {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 12px;
           padding: .85rem 1.1rem; flex: 1 1 150px; min-width: 140px; }}
  .card .k {{ color: {MUTED}; font-size: .72rem; text-transform: uppercase;
              letter-spacing: .07em; font-weight: 600; }}
  .card .v {{ color: {INK}; font-size: 1.5rem; font-weight: 660;
              line-height: 1.25; margin-top: .18rem; }}
  .card .s {{ color: {MUTED}; font-size: .78rem; }}
  .card.hi {{ border-color: {ACCENT}; box-shadow: 0 0 0 1px {ACCENT}22; }}

  .panel {{ background: {SURFACE}; border: 1px solid {LINE};
            border-radius: 14px; padding: 1.1rem 1.2rem; margin-bottom: 1rem; }}

  .note {{ background: #fff8e6; border: 1px solid #f0dca8; color: #6b5316;
           border-radius: 10px; padding: .8rem 1rem; font-size: .87rem;
           line-height: 1.5; }}

  [data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {LINE}; }}
  [data-testid="stSidebar"] .block-container {{ padding-top: 1.4rem; }}

  .stTabs [data-baseweb="tab-list"] {{ gap: .3rem; border-bottom: 1px solid {LINE}; }}
  .stTabs [data-baseweb="tab"] {{ font-size: .9rem; font-weight: 550; }}

  div[data-testid="stDataFrame"] {{ border: 1px solid {LINE}; border-radius: 10px; }}

  @media (prefers-color-scheme: dark) {{
    .stApp {{ background: #10161c; }}
    h1, h2, h3 {{ color: #e8eef4; }}
    .lede, .card .k, .card .s {{ color: #93a3b3; }}
    .card, .panel {{ background: #18212a; border-color: #263340; }}
    .card .v {{ color: #e8eef4; }}
    [data-testid="stSidebar"] {{ background: #18212a; border-color: #263340; }}
  }}
</style>
"""


def setup(title: str) -> None:
    st.set_page_config(page_title=f"{title} · BiH Air Quality",
                       page_icon="🌫️", layout="wide")
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
