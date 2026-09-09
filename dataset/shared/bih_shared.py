"""Shared loading, masking and metrics for the BIH air-quality project.

Import this from the GNN, xLSTM and Chronos notebooks so all three use one
definition of the split, the mask and the metrics.

    import sys; sys.path.append(SHARED_DIR)
    import bih_shared as bs

    split  = bs.load_split(SHARED_DIR)
    panel  = bs.load_panel(CLEAN, SHARED_DIR)
    wins   = bs.load_windows(SHARED_DIR)
    truth  = bs.target(panel, station, pollutant, origin)   # y, mask
    scores = bs.score(pred, truth.y, truth.mask, scale)
"""
import json, os
import numpy as np
import pandas as pd

POLLUTANTS = ["pm10", "pm25", "so2", "no2", "o3", "co"]
HORIZON = 24


def load_split(shared_dir):
    with open(os.path.join(shared_dir, "split.json")) as f:
        return json.load(f)


def load_groups(shared_dir):
    with open(os.path.join(shared_dir, "station_groups.json")) as f:
        return json.load(f)


def load_windows(shared_dir):
    return pd.read_csv(os.path.join(shared_dir, "eval_windows.csv"),
                       parse_dates=["origin"])


def load_coords(shared_dir):
    return pd.read_csv(os.path.join(shared_dir, "station_coords.csv"))


def load_panel(clean_csv, shared_dir):
    """{(station, pollutant): DataFrame(y, filled)} on a gap-free hourly index.

    Reindexing to a complete hourly range matters: a missing hour has to exist
    as NaN, or every model silently treats the next reading as one hour later
    than it is.
    """
    groups = load_groups(shared_dir)
    cols = (["source", "city", "station", "datetime"] + POLLUTANTS
            + [f"{p}_filled" for p in POLLUTANTS])
    df = pd.read_csv(clean_csv, usecols=cols, parse_dates=["datetime"])
    full = pd.date_range(df.datetime.min(), df.datetime.max(), freq="h")
    panel = {}
    for st, g in df.groupby("station", sort=True):
        g = g.drop_duplicates("datetime").set_index("datetime").reindex(full)
        for p in POLLUTANTS:
            if groups["by_station"].get(st, {}).get(p) == "never":
                continue
            panel[(st, p)] = pd.DataFrame(
                {"y": g[p].to_numpy(float),
                 "filled": g[f"{p}_filled"].fillna(False).to_numpy(bool)},
                index=full)
    return panel


def target(panel, station, pollutant, origin, horizon=HORIZON):
    """The 24 hours to predict, and the mask of what may be scored.

    mask is False wherever the value is missing or was interpolated during
    dataset build - those points are not ground truth.
    """
    s = panel[(station, pollutant)]
    i = s.index.get_loc(pd.Timestamp(origin))
    y = s.y.to_numpy()[i:i + horizon]
    fl = s.filled.to_numpy()[i:i + horizon]
    return y, (~np.isnan(y) & ~fl)


def context(panel, station, pollutant, origin, length):
    """The `length` hours before the origin. Filled values are allowed here."""
    s = panel[(station, pollutant)]
    i = s.index.get_loc(pd.Timestamp(origin))
    return s.y.to_numpy()[max(0, i - length):i]


def mase_scale(ctx, season=24):
    """In-context seasonal-naive MAE - the MASE denominator."""
    if len(ctx) <= season:
        return np.nan
    d = np.abs(ctx[season:] - ctx[:-season])
    return float(np.nanmean(d)) if np.isfinite(d).any() else np.nan


def score(pred, y, mask, scale=np.nan):
    """MAE, RMSE and MASE over masked points only."""
    err = np.where(mask, np.asarray(pred, float) - y, np.nan)
    mae = float(np.nanmean(np.abs(err)))
    return {"mae": mae,
            "rmse": float(np.sqrt(np.nanmean(err ** 2))),
            "mase": mae / scale if np.isfinite(scale) and scale > 0 else np.nan,
            "n": int(mask.sum())}


def aggregate(rows):
    """Per-pollutant means. Never average MAE/RMSE across pollutants - CO is
    mg/m3 and the rest ug/m3. MASE is the only cross-pollutant summary."""
    r = pd.DataFrame(rows)
    return r.groupby(["model", "pollutant"])[["mae", "rmse", "mase"]].mean()
