"""R4K-1 shared definitions: the per-frame table, shots (PROTOCOL.md section 3) and the comparisons of section 4 (Q2)."""
from __future__ import annotations

import csv
import gzip
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SEED = 20261027
BOOTSTRAP = 2000

# Operating points: frames scoring below the value are accepted as sharp (Frame Blur Studio: score < threshold).
R3_POINTS = {"r98": 0.016399000000000018, "r95": 0.05490985000000001, "r90": 0.23241829999999963}
V9_POINTS = {"M=0": 1e-9, "M<0.05": 0.05, "M<0.25": 0.25}

NUMERIC = ["index", "t", "key", "coded_bytes", "r3_score", "r3_logit", "v9_mass", "v9_score", "r3_blur_cells",
           "r3_mean_logit", "v9_blur_cells", "v9_mean_logit", "luma", "laplacian", "flat_e3", "edge_e3", "flat_edge",
           "zero", "block8", "hcorr", "flow", "thumb_diff", "hist_diff"]


VARIANTS = ("protocol", "revised")


def load(label: str, variant: str = "protocol") -> dict[str, np.ndarray]:
    """variant "protocol": the definitions of PROTOCOL.md as fixed. "revised": deviations D1 and D2 of REPORT.md.

    D1 (black frame): mean luminance < 8 AND mean |Laplacian| < 1. On video A, mean luminance < 8 alone also caught
       4,622 frames of dark but structured scenes (median |Laplacian| 2.9; true black frames: 0.4).
    D2 (shots): the hue-histogram criterion only when both frames have mean luminance >= 20 (hue is unstable in dark
       frames), and no cut within 3 frames after a cut. The protocol rule produced 438 shots of <= 3 frames on video A.
    """
    with gzip.open(DATA / f"frames_{label}.csv.gz", "rt", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    table = {c: np.array([float(r[c]) if r[c] != "" else np.nan for r in rows]) for c in NUMERIC}
    table["type"] = np.array([r["type"] for r in rows])
    table["index"] = table["index"].astype(int)
    if variant == "protocol":
        table["black"] = table["luma"] < 8
    elif variant == "revised":
        table["black"] = (table["luma"] < 8) & (table["laplacian"] < 1.0)
    else:
        raise ValueError(variant)
    table["shot"], table["cut"] = shots(table, variant)
    return table


def _rolling_median_excluding_centre(x: np.ndarray, half: int = 6) -> np.ndarray:
    n = len(x)
    out = np.empty(n)
    for k in range(n):
        lo, hi = max(0, k - half), min(n, k + half + 1)
        window = np.concatenate((x[lo:k], x[k + 1:hi]))
        window = window[np.isfinite(window)]
        out[k] = np.median(window) if window.size else np.nan
    return out


def shots(table: dict, variant: str = "protocol") -> tuple[np.ndarray, np.ndarray]:
    """Cut at k when the thumbnail difference or the hue-histogram distance to frame k - 1 is more than 3x its local
    median (k - 6 .. k + 6, k excluded) and above its absolute floor (6 grey levels / 0.25). Pixels only: the coded
    frame type is never used. A black frame (mean luminance < 8) never starts a shot."""
    td, hd, luma = table["thumb_diff"], table["hist_diff"], table["luma"]
    cut = np.zeros(len(td), bool)
    with np.errstate(invalid="ignore"):
        cut |= (td > 3 * _rolling_median_excluding_centre(td)) & (td > 6)
        hue = (hd > 3 * _rolling_median_excluding_centre(hd)) & (hd > 0.25)
        if variant == "revised":
            hue &= (luma >= 20) & (np.r_[np.nan, luma[:-1]] >= 20)
        cut |= hue
    cut &= ~table["black"]
    cut[0] = True
    if variant == "revised":
        last = 0
        for k in np.flatnonzero(cut)[1:]:
            if k - last < 4:
                cut[k] = False
            else:
                last = k
    return np.cumsum(cut) - 1, cut


def comparisons(table: dict) -> list[dict]:
    """Q2 pairs (reference frame, other frame) inside one shot, neither black.

    mid_I: an I-frame whose two neighbours are in its shot, against each neighbour.
    cut_I: an I-frame that starts a shot, against the next two frames of the shot.
    P_vs_B: a P-frame whose two neighbours are B-frames of its shot, against each neighbour.
    """
    t, shot, cut, black = table["type"], table["shot"], table["cut"], table["black"]
    n = len(t)
    out = []

    def add(kind, ref, other):
        if 0 <= other < n and shot[other] == shot[ref] and not black[other] and not black[ref]:
            out.append({"kind": kind, "ref": ref, "other": other, "shot": int(shot[ref])})

    for k in range(n):
        if t[k] == "I":
            if cut[k]:
                add("cut_I", k, k + 1)
                add("cut_I", k, k + 2)
            elif 0 < k < n - 1 and shot[k - 1] == shot[k] == shot[k + 1]:
                add("mid_I", k, k - 1)
                add("mid_I", k, k + 1)
        elif t[k] == "P" and 0 < k < n - 1 and t[k - 1] == "B" and t[k + 1] == "B" \
                and shot[k - 1] == shot[k] == shot[k + 1]:
            add("P_vs_B", k, k - 1)
            add("P_vs_B", k, k + 1)
    return out


def cluster_bootstrap(clusters: np.ndarray, statistic, rng: np.random.Generator, n: int = BOOTSTRAP):
    """95% percentile interval of statistic(selected positions) under resampling of whole clusters."""
    ids = np.unique(clusters)
    members = {c: np.flatnonzero(clusters == c) for c in ids}
    values = []
    for _ in range(n):
        pick = rng.choice(ids, len(ids), replace=True)
        values.append(statistic(np.concatenate([members[c] for c in pick])))
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))] if values.size else [None, None]
