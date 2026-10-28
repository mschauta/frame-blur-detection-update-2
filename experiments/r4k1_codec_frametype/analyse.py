"""R4K-1 step 3: every analysis of PROTOCOL.md section 4 for one video; writes data/results_<label>.json.

usage: python analyse.py --label A

Inference unit: the shot. One set of 2,000 shot resamples (seed 20261027) serves every interval of a video, so the
intervals of different statistics come from the same resamples. Shares use per-shot sums; medians use the resampled
comparisons.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import spearmanr

from common import BOOTSTRAP, DATA, R3_POINTS, SEED, V9_POINTS, VARIANTS, comparisons, load

TYPES = ("I", "P", "B")


class Resampler:
    """Shot resamples as count weights: W[b, s] = how often shot s was drawn in resample b."""

    def __init__(self, shots: np.ndarray, rng: np.random.Generator):
        self.ids = np.unique(shots)
        self.S = len(self.ids)
        self.W = np.stack([np.bincount(rng.integers(0, self.S, self.S), minlength=self.S) for _ in range(BOOTSTRAP)])

    def _pos(self, shots):
        return np.searchsorted(self.ids, shots)

    def share(self, shots, numerator, denominator=None):
        """Share numerator / denominator (boolean arrays over observations) and its 95% interval."""
        denominator = np.ones(len(numerator), bool) if denominator is None else denominator
        if not denominator.any():
            return {"value": None, "ci95": [None, None], "n": 0}
        pos = self._pos(shots)
        num = np.bincount(pos, weights=(numerator & denominator).astype(float), minlength=self.S)
        den = np.bincount(pos, weights=denominator.astype(float), minlength=self.S)
        point = num.sum() / den.sum() if den.sum() else float("nan")
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = (self.W @ num) / (self.W @ den)
        boot = boot[np.isfinite(boot)]
        return {"value": float(point), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                "n": int(denominator.sum())}

    def median(self, shots, values):
        """Median of values (one per observation) and its 95% interval (weighted median per resample)."""
        ok = np.isfinite(values)
        if not ok.any():
            return {"value": None, "ci95": [None, None], "n": 0}
        values, pos = values[ok], self._pos(shots[ok])
        order = np.argsort(values, kind="mergesort")
        values, pos = values[order], pos[order]
        boot = []
        for w in self.W:
            weights = w[pos].astype(float)
            cum = np.cumsum(weights)
            if cum[-1] == 0:
                continue
            boot.append(values[np.searchsorted(cum, cum[-1] / 2)])
        return {"value": float(np.median(values)) if values.size else float("nan"),
                "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], "n": int(values.size)}

    def mean(self, shots, values):
        ok = np.isfinite(values)
        if not ok.any():
            return {"value": None, "ci95": [None, None], "n": 0}
        pos = self._pos(shots[ok])
        s = np.bincount(pos, weights=values[ok], minlength=self.S)
        c = np.bincount(pos, minlength=self.S).astype(float)
        boot = (self.W @ s) / (self.W @ c)
        return {"value": float(s.sum() / c.sum()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                "n": int(ok.sum())}


def accepted(table, model, point):
    """Accepted as sharp at an operating point: score below the threshold (round3 score; v9 blur mass)."""
    if model == "r3":
        return table["r3_score"] < R3_POINTS[point]
    return table["v9_mass"] < V9_POINTS[point]


def equal_share(values: np.ndarray, share: float, eligible: np.ndarray) -> np.ndarray:
    """Accept the lowest values up to floor(N * share) among eligible frames, never splitting equal values."""
    v = values[eligible]
    target = int(np.floor(len(v) * share))
    out = np.zeros(len(values), bool)
    if target == 0:
        return out
    cut = np.sort(v)[target]
    out[eligible] = v < cut
    return out


def describe(T, checks) -> dict:
    n = len(T["type"])
    fps = (n - 1) / float(T["t"][-1] - T["t"][0])          # timestamps are rounded to 1 ms; the mean period is exact
    duration = n / fps
    I = np.flatnonzero(T["type"] == "I")
    gaps = np.diff(I)
    cut_I = int((T["cut"] & (T["type"] == "I")).sum())
    shot_len = np.bincount(T["shot"])
    return {
        "frames": n, "duration_s": duration, "fps": fps,
        "video_bytes": checks["video_bytes"], "mean_bitrate_mbit_s": checks["video_bytes"] * 8 / duration / 1e6,
        "types": {t: int((T["type"] == t).sum()) for t in TYPES},
        "coded_kb_median": {t: float(np.median(T["coded_bytes"][T["type"] == t]) / 1024) for t in TYPES},
        "coded_bytes_share": {t: float(T["coded_bytes"][T["type"] == t].sum() / T["coded_bytes"].sum()) for t in TYPES},
        "i_frame_gap_frames": {"median": float(np.median(gaps)), "p10": float(np.percentile(gaps, 10)),
                               "p90": float(np.percentile(gaps, 90))},
        "shots": int(T["shot"].max() + 1), "shot_length_median_frames": float(np.median(shot_len)),
        "black_frames": int(T["black"].sum()),
        "i_frames_at_detected_cuts": cut_I, "i_frames_total": int(len(I)),
        "detected_cuts_that_are_i_frames": float(cut_I / max(1, int(T["cut"].sum()))),
    }


def q1(T, R) -> dict:
    out = {}
    for t in TYPES:
        sel = T["type"] == t
        entry = {"n": int(sel.sum()),
                 "r3_logit_quantiles_10_25_50_75_90": np.percentile(T["r3_logit"][sel], [10, 25, 50, 75, 90]).tolist(),
                 "v9_mass_quantiles_10_25_50_75_90": np.percentile(T["v9_mass"][sel], [10, 25, 50, 75, 90]).tolist()}
        for model, points in (("r3", R3_POINTS), ("v9", V9_POINTS)):
            for p in points:
                entry[f"{model}_accepted_{p}"] = R.share(T["shot"], accepted(T, model, p), sel & ~T["black"])
        out[t] = entry
    return out


def _pairs(T, kinds=None):
    comps = [c for c in comparisons(T) if kinds is None or c["kind"] in kinds]
    ref = np.array([c["ref"] for c in comps], int)
    other = np.array([c["other"] for c in comps], int)
    shot = np.array([c["shot"] for c in comps], int)
    kind = np.array([c["kind"] for c in comps])
    return ref, other, shot, kind


def q2_effect(T, R, ref, other, shot) -> dict:
    out = {"comparisons": int(len(ref)), "shots": int(len(np.unique(shot)))}
    if not len(ref):
        return out
    for model, col in (("r3", "r3_logit"), ("v9", "v9_mass")):
        d = T[col][other] - T[col][ref]                     # > 0: the reference frame scores sharper
        out[model] = {"ref_sharper": R.share(shot, d > 0), "tie": R.share(shot, d == 0),
                      "other_sharper": R.share(shot, d < 0), "median_difference_other_minus_ref": R.median(shot, d),
                      "net_ref_sharper_minus_other_sharper": float(np.mean(d > 0) - np.mean(d < 0))}
        flips = {}
        for p in (R3_POINTS if model == "r3" else V9_POINTS):
            acc = accepted(T, model, p)
            flips[p] = {"ref_sharp_other_blurred": R.share(shot, acc[ref] & ~acc[other]),
                        "ref_blurred_other_sharp": R.share(shot, ~acc[ref] & acc[other])}
        out[model]["decision_flips"] = flips
    return out


def q2(T, R) -> dict:
    ref, other, shot, kind = _pairs(T)
    return {k: q2_effect(T, R, ref[kind == k], other[kind == k], shot[kind == k]) for k in ("mid_I", "cut_I", "P_vs_B")}


def q3(T, R) -> dict:
    """Frames since the last I-frame of the same shot (0 = the I-frame), with shot-centred scores."""
    n = len(T["type"])
    pos = np.full(n, -1)
    last = -1
    for k in range(n):
        if T["type"][k] == "I":
            last = k
        if last >= 0 and T["shot"][last] == T["shot"][k]:
            pos[k] = k - last
    shot_median_r3 = np.array([np.median(T["r3_logit"][T["shot"] == s]) for s in range(T["shot"].max() + 1)])
    centred = T["r3_logit"] - shot_median_r3[T["shot"]]
    ok = (pos >= 0) & ~T["black"]
    out = {}
    for p in range(13):
        sel = ok & (pos == p)
        if sel.sum() < 20:
            continue
        out[str(p)] = {"n": int(sel.sum()), "types": {t: int((sel & (T["type"] == t)).sum()) for t in TYPES},
                       "r3_shot_centred_logit_median": float(np.median(centred[sel])),
                       "r3_accepted_r95": R.share(T["shot"], accepted(T, "r3", "r95"), sel),
                       "v9_accepted_M<0.05": R.share(T["shot"], accepted(T, "v9", "M<0.05"), sel),
                       "v9_mass_median": float(np.median(T["v9_mass"][sel]))}
    return out


def q4(T, R, variant) -> dict:
    """protocol: flow difference of the two frames below its 25th percentile and both flows below the shot median.
    revised (D3): the motion between the two compared frames (the mean flow of the frames after the earlier one, up
    to the later one) below its 25th percentile. The flow of a shot-start I-frame measures the cut, not motion."""
    ref, other, shot, kind = _pairs(T)
    shot_flow = np.array([np.nanmedian(T["flow"][T["shot"] == s]) for s in range(T["shot"].max() + 1)])
    out = {"rule": variant}
    for k in ("mid_I", "cut_I", "P_vs_B"):
        sel = kind == k
        if not sel.any():
            out[k] = {"comparisons": 0}
            continue
        r, o, s = ref[sel], other[sel], shot[sel]
        if variant == "protocol":
            dflow = np.abs(T["flow"][r] - T["flow"][o])
            calm = (dflow < np.nanpercentile(dflow, 25)) & (T["flow"][r] < shot_flow[s]) & (T["flow"][o] < shot_flow[s])
        else:
            lo, hi = np.minimum(r, o), np.maximum(r, o)
            dflow = np.array([np.nanmean(T["flow"][a + 1:b + 1]) for a, b in zip(lo, hi)])
            calm = dflow < np.nanpercentile(dflow, 25)
        full, matched = q2_effect(T, R, r, o, s), q2_effect(T, R, r[calm], o[calm], s[calm])
        out[k] = {"matched": matched, "flow_p25_px": float(np.nanpercentile(dflow, 25)),
                  "net_effect_remaining": {m: (matched[m]["net_ref_sharper_minus_other_sharper"]
                                               / full[m]["net_ref_sharper_minus_other_sharper"])
                                           if full[m]["net_ref_sharper_minus_other_sharper"] else None
                                           for m in ("r3", "v9")}}
    return out


def _rank_residual(y, x):
    ry, rx = np.argsort(np.argsort(y)).astype(float), np.argsort(np.argsort(x)).astype(float)
    slope, icept = np.polyfit(rx, ry, 1)
    return ry - (slope * rx + icept)


def q5(T, R) -> dict:
    rows = {"r3": [], "v9": [], "r3_partial": [], "v9_partial": []}
    shots_used = []
    for s in range(T["shot"].max() + 1):
        sel = (T["shot"] == s) & (T["type"] == "B") & ~T["black"] & np.isfinite(T["flow"])
        if sel.sum() < 10:
            continue
        b = T["coded_bytes"][sel]
        if np.ptp(b) == 0:
            continue
        shots_used.append(s)
        for model, col in (("r3", "r3_logit"), ("v9", "v9_mass")):
            y = T[col][sel]
            rows[model].append(spearmanr(b, y).statistic if np.ptp(y) > 0 else np.nan)
            ry, rb = _rank_residual(y, T["flow"][sel]), _rank_residual(b, T["flow"][sel])
            rows[f"{model}_partial"].append(float(np.corrcoef(rb, ry)[0, 1]) if ry.std() > 0 and rb.std() > 0 else np.nan)
    shots_used = np.array(shots_used)
    out = {"shots": int(len(shots_used)), "note": "Spearman between coded size and score among B-frames of a shot; "
           "negative = more bits, less blur evidence; partial = after removing the linear trend in flow rank"}
    for key, values in rows.items():
        v = np.array(values, float)
        out[key] = {"mean_rho": R.mean(shots_used, v), "median_rho": float(np.nanmedian(v)),
                    "share_negative": float(np.nanmean(v < 0))}
    return out


def q6(T, R) -> dict:
    eligible = ~T["black"]
    i_share = float(np.mean(T["type"][eligible] == "I"))
    out = {"i_frame_share": i_share, "operating_points": {}, "equal_shares": {}, "phase2": {}}
    for model, points in (("r3", R3_POINTS), ("v9", V9_POINTS)):
        for p in points:
            acc = accepted(T, model, p) & eligible
            out["operating_points"][f"{model}_{p}"] = {
                "accepted_share": float(acc.sum() / eligible.sum()),
                "i_share_of_accepted": R.share(T["shot"], T["type"] == "I", acc),
                "enrichment": float(np.mean(T["type"][acc] == "I") / i_share) if acc.any() else None}
    for share in (0.10, 0.25, 0.50, 0.75, 0.90):
        entry = {}
        for model, col in (("r3", "r3_logit"), ("v9", "v9_mass")):
            acc = equal_share(T[col], share, eligible)
            entry[model] = {"actual_share": float(acc.sum() / eligible.sum()),
                            "accepted_by_type": {t: float(acc[eligible & (T["type"] == t)].mean()) for t in TYPES},
                            "i_share_of_accepted": R.share(T["shot"], T["type"] == "I", acc),
                            "enrichment": float(np.mean(T["type"][acc] == "I") / i_share) if acc.any() else None}
        out["equal_shares"][f"{int(share * 100)}%"] = entry
    # Phase II: the sharpest frame of each time window; ties shared out evenly; black frames are not candidates.
    rng = np.random.default_rng(SEED)
    for width in (1, 5, 20):
        window = np.floor((T["t"] - T["t"][0]) / width).astype(int)
        entry = {}
        for model, col in (("r3", "r3_logit"), ("v9", "v9_mass")):
            picked, expected = [], []
            for w in np.unique(window[eligible]):
                cand = np.flatnonzero((window == w) & eligible)
                vals = T[col][cand]
                best = cand[vals == vals.min()]
                picked.append(np.mean(T["type"][best] == "I"))
                expected.append(np.mean(T["type"][cand] == "I"))
            picked, expected = np.array(picked), np.array(expected)
            boot = []
            for _ in range(BOOTSTRAP):
                b = rng.integers(0, len(picked), len(picked))
                boot.append(picked[b].mean() / expected[b].mean())
            entry[model] = {"windows": int(len(picked)), "i_share_of_picks": float(picked.mean()),
                            "i_share_expected": float(expected.mean()), "enrichment": float(picked.mean() / expected.mean()),
                            "enrichment_ci95_window_bootstrap": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
        out["phase2"][f"{width}s"] = entry
    return out


PIXEL = ("flat_e3", "edge_e3", "flat_edge", "zero", "block8", "hcorr", "laplacian", "luma")


def q7(T, R) -> dict:
    ref, other, shot, kind = _pairs(T)
    out = {}
    for k in ("mid_I", "cut_I", "P_vs_B"):
        sel = kind == k
        if not sel.any():
            out[k] = {"comparisons": 0}
            continue
        r, o, s = ref[sel], other[sel], shot[sel]
        entry = {"comparisons": int(sel.sum())}
        for stat in PIXEL:
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = T[stat][o] / T[stat][r] - 1
            entry[stat] = {"other_higher": R.share(s, T[stat][o] > T[stat][r]),
                           "median_relative_difference": R.median(s, rel)}
        d_flat = T["flat_e3"][o] - T["flat_e3"][r]
        for model, col in (("r3", "r3_logit"), ("v9", "v9_mass")):
            d_score = T[col][o] - T[col][r]
            ok = np.isfinite(d_flat) & np.isfinite(d_score)
            rho = spearmanr(d_flat[ok], d_score[ok]).statistic
            boot_rng = np.random.default_rng(SEED)
            ids = np.unique(s[ok])
            members = {i: np.flatnonzero(s[ok] == i) for i in ids}
            boot = []
            for _ in range(BOOTSTRAP):
                pick = np.concatenate([members[i] for i in boot_rng.choice(ids, len(ids))])
                boot.append(spearmanr(d_flat[ok][pick], d_score[ok][pick]).statistic)
            entry[f"spearman_delta_flat_e3_vs_delta_{model}"] = {
                "value": float(rho), "ci95": [float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))]}
        out[k] = entry
    return out


def q8(label, R, variant) -> dict:
    z = np.load(DATA / f"cells_{label}_{variant}.npz")
    values, kind, shot = z["values"], z["kind"], z["shot"]
    out = {}
    for k in ("mid_I", "cut_I", "all_I"):
        sel = np.ones(len(kind), bool) if k == "all_I" else kind == k
        entry = {"comparisons": int(sel.sum())}
        if not sel.any():
            out[k] = entry
            continue
        for m, model in enumerate(("r3", "v9")):
            rows = []
            for d in range(10):
                rows.append({name: R.mean(shot[sel], values[sel, m, d, i])
                             for i, name in enumerate(("mean_logit_difference", "crossing_to_blur", "crossing_to_sharp"))})
            entry[model] = rows
        out[k] = entry
    return out


def q9(T) -> dict:
    eligible = ~T["black"]
    out = {}
    for share in (0.10, 0.25, 0.50, 0.75, 0.90):
        a = equal_share(T["r3_logit"], share, eligible)
        b = equal_share(T["v9_mass"], share, eligible)
        out[f"{int(share * 100)}%"] = {t: float(np.mean(a[eligible & (T["type"] == t)] == b[eligible & (T["type"] == t)]))
                                      for t in TYPES}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--variant", choices=VARIANTS, default="revised")
    a = ap.parse_args()
    T = load(a.label, a.variant)
    checks = json.loads((DATA / f"checks_{a.label}.json").read_text(encoding="utf-8"))
    R = Resampler(T["shot"], np.random.default_rng(SEED))
    results = {"label": a.label, "variant": a.variant, "descriptive": describe(T, checks)}
    for name, fn in (("Q1", lambda: q1(T, R)), ("Q2", lambda: q2(T, R)), ("Q3", lambda: q3(T, R)),
                     ("Q4", lambda: q4(T, R, a.variant)), ("Q5", lambda: q5(T, R)), ("Q6", lambda: q6(T, R)),
                     ("Q7", lambda: q7(T, R)), ("Q8", lambda: q8(a.label, R, a.variant)), ("Q9", lambda: q9(T))):
        results[name] = fn()
        print(f"{a.label} {a.variant}: {name} done", flush=True)
    # Exploratory sensitivity: the same Q2 with 30 s time blocks instead of shots as the resampling unit.
    blocks = np.floor((T["t"] - T["t"][0]) / 30).astype(int)
    RB = Resampler(blocks, np.random.default_rng(SEED))
    TB = dict(T, shot=blocks)
    results["Q2_time_block_clusters_exploratory"] = {
        k: q2_effect(TB, RB, *[x[kind == k] for x in (ref, other)], blocks[ref[kind == k]])
        for ref, other, _, kind in [_pairs(T)] for k in ("mid_I", "cut_I", "P_vs_B")}
    (DATA / f"results_{a.label}_{a.variant}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
