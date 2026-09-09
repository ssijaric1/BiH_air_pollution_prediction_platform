"""Build the small data files the Streamlit app reads.

The app has to run on Streamlit Community Cloud, which means everything it
reads must be committed to the repository. The full hourly panel is 160 MB and
the raw forecast export is tens of MB, so neither can go in as-is. This script
slims both to the 2024 evaluation year, casts to float32, and writes parquet.

    python scripts/prepare_app_data.py

Inputs (not tracked in git):
    dataset/bih_dataset/bih_hourly_clean.csv   the full panel
    results/forecasts.parquet                  exported by 02_ensemble_full.ipynb

Outputs (tracked, a few MB total):
    app/data/panel.parquet       measurements + weather, Dec 2023 onward
    app/data/forecasts.parquet   every model's 24 h forecasts on the 2024 windows
    app/data/stations.parquet    coordinates, type, and per-pollutant coverage
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "dataset" / "bih_dataset" / "bih_hourly_clean.csv"
COORDS = ROOT / "dataset" / "shared" / "station_coords.csv"
FORECASTS_IN = ROOT / "results" / "forecasts.parquet"
OUT = ROOT / "app" / "data"

POLLUTANTS = ["pm10", "pm25", "so2", "no2", "o3", "co"]
WEATHER = ["temperature", "wind_speed", "humidity", "pressure"]
# Windows start at midnight through 2024; the viewer draws several days of
# lead-in context, so the panel has to reach back into 2023.
PANEL_START = "2023-12-01"


def build_panel() -> pd.DataFrame:
    cols = (["city", "station", "datetime"] + POLLUTANTS + WEATHER
            + [f"{p}_filled" for p in POLLUTANTS])
    df = pd.read_csv(CLEAN, usecols=cols, parse_dates=["datetime"])
    df = df[df.datetime >= PANEL_START].copy()

    for p in POLLUTANTS:
        # One nullable-free column per pollutant: the value, plus a flag saying
        # whether it is a real reading. The app never plots filled points as
        # ground truth, matching the scoring rule.
        df[f"{p}_filled"] = df[f"{p}_filled"].fillna(False).astype(bool)

    for c in POLLUTANTS + WEATHER:
        df[c] = df[c].astype("float32")

    df["city"] = df.city.astype("category")
    df["station"] = df.station.astype("category")
    return df.sort_values(["station", "datetime"], ignore_index=True)


def build_stations(panel: pd.DataFrame) -> pd.DataFrame:
    st = pd.read_csv(COORDS)
    # Coverage over the evaluation year only - that is the period the app shows,
    # so a station's headline coverage should describe the same window.
    y2024 = panel[panel.datetime.dt.year == 2024]
    cov = (y2024.groupby("station", observed=True)[POLLUTANTS]
                .apply(lambda g: g.notna().mean() * 100)
                .round(1)
                .add_suffix("_coverage")
                .reset_index())
    out = st.merge(cov, on="station", how="left")
    out["station_type"] = out.station_type.fillna("unknown")
    return out


def slim_forecasts() -> pd.DataFrame | None:
    if not FORECASTS_IN.exists():
        return None
    fc = pd.read_parquet(FORECASTS_IN)
    fc["origin"] = pd.to_datetime(fc.origin)
    for c in fc.select_dtypes("float64").columns:
        fc[c] = fc[c].astype("float32")
    for c in ("station", "city", "pollutant", "season", "regime"):
        if c in fc.columns:
            fc[c] = fc[c].astype("category")
    return fc.sort_values(["station", "pollutant", "origin", "hour"],
                          ignore_index=True)


def report(name: str, path: Path, df: pd.DataFrame) -> None:
    mb = path.stat().st_size / 1e6
    print(f"  {name:<22} {len(df):>9,} rows  {mb:>6.1f} MB")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    if not CLEAN.exists():
        print(f"missing {CLEAN}\n"
              "Run notebooks/02_dataset/build_dataset.ipynb and "
              "notebooks/03_eda/eda.ipynb first, or copy the file in.")
        return 1

    print("writing app data")

    panel = build_panel()
    p = OUT / "panel.parquet"
    panel.to_parquet(p, index=False, compression="zstd")
    report("panel.parquet", p, panel)

    stations = build_stations(panel)
    p = OUT / "stations.parquet"
    stations.to_parquet(p, index=False)
    report("stations.parquet", p, stations)

    fc = slim_forecasts()
    if fc is None:
        print(f"\n  no {FORECASTS_IN.relative_to(ROOT)} - skipping forecasts.\n"
              "  Run the last cell of notebooks/07_ensemble/02_ensemble_full.ipynb\n"
              "  in Colab, then put the file it writes to Drive in results/.")
    else:
        p = OUT / "forecasts.parquet"
        fc.to_parquet(p, index=False, compression="zstd")
        report("forecasts.parquet", p, fc)

    total = sum(f.stat().st_size for f in OUT.glob("*.parquet")) / 1e6
    print(f"\n  total {total:.1f} MB in {OUT.relative_to(ROOT)}")
    if total > 90:
        print("  WARNING: over 90 MB. GitHub rejects single files above 100 MB;\n"
              "  consider subsampling the forecast windows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
