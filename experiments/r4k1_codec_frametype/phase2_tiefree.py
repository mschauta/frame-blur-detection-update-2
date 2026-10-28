"""R4K-1, exploratory (REPORT.md section 4.3): the Phase II simulation with tie-free readouts.

usage: python phase2_tiefree.py

The sharpest frame of each 1 / 5 / 20 s window by four scores: round3's logit, v9's blur mass M, and the mean cell logit
of each detector (no ties). Tied minima count as picks shared evenly, as in analyse.py. Black frames are not candidates.
Writes data/exploratory_phase2_tiefree.json.
"""
import json

import numpy as np

from common import load

out = {}
for L in "AB":
    T = load(L, "revised")
    elig = ~T["black"]
    res = {"flow_median_by_type": {t: float(np.nanmedian(T["flow"][elig & (T["type"] == t)])) for t in "IPB"}}
    for col in ("r3_logit", "v9_mass", "r3_mean_logit", "v9_mean_logit"):
        res[col] = {}
        for width in (1, 5, 20):
            window = np.floor((T["t"] - T["t"][0]) / width).astype(int)
            picked, expected, ties = [], [], []
            for w in np.unique(window[elig]):
                cand = np.flatnonzero((window == w) & elig)
                v = T[col][cand]
                best = cand[v == v.min()]
                picked.append(np.mean(T["type"][best] == "I")); expected.append(np.mean(T["type"][cand] == "I")); ties.append(len(best))
            res[col][f"{width}s"] = {"enrichment": float(np.mean(picked) / np.mean(expected)), "picks_I": float(np.mean(picked)), "mean_tied_minima": float(np.mean(ties))}
    out[L] = res
    print(L, "flow median by type:", {t: round(v, 2) for t, v in res["flow_median_by_type"].items()})
    for col in ("r3_logit", "v9_mass", "r3_mean_logit", "v9_mean_logit"):
        print(f"  {col:14s}", {w: (round(e["enrichment"], 1), round(e["mean_tied_minima"], 1)) for w, e in res[col].items()})
json.dump(out, open("data/exploratory_phase2_tiefree.json", "w"), indent=1)
