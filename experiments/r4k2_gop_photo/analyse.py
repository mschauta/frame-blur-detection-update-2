"""R4K-2 analysis (PROTOCOL.md, measures 1-4): data/frames.csv.gz -> data/results.json.

Scores: round3 Base / Small: the image logit, blurred when sigmoid(logit) >= the file's threshold (r95, r98);
v9: the built-in M, blurred when M > its r95 point (0.007882) or M > 0. Every frame is 0 px sharp.
95% intervals: bootstrap over the photos (2000 resamples, seed 20261028), the same resamples for every statistic.

usage: python experiments/r4k2_gop_photo/analyse.py
"""
from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = HERE / "data"
REPS, SEED = 2000, 20261028
SCORES = {"base": "base_logit", "small": "small_logit", "v9": "v9_m"}


def thresholds() -> dict:
    """Per model: {call name: score threshold}; a frame is blurred when score >= (round3) / > (v9) the value."""
    out = {}
    files = json.loads((DATA / "thresholds.json").read_text(encoding="utf-8"))      # the weight files' own thresholds
    for m in ("base", "small"):
        out[m] = {r: math.log(t / (1 - t)) for r, t in files[m]["thresholds"].items() if r in ("r95", "r98")}
    pts = json.loads((REPO / "reproduction" / "heldout_films.json").read_text(encoding="utf-8"))
    out["v9"] = {"r95": pts["update2_recall_points"]["r95"]["threshold"], "M>0": 0.0}
    return out


def calls(m: str, s: np.ndarray, t: float) -> np.ndarray:
    return s > t if m == "v9" else s >= t


def load() -> dict:
    with gzip.open(DATA / "frames.csv.gz", "rt", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    n = {}
    for r in rows:
        n[r["photo"]] = n.get(r["photo"], 0) + 1
    complete = {p for p, c in n.items() if c == 2 * 4 * 3 * 48}      # a run in progress: whole photos only
    rows = [r for r in rows if r["photo"] in complete]
    photos = sorted(complete)
    pid = {p: i for i, p in enumerate(photos)}
    d = {"photo": np.array([pid[r["photo"]] for r in rows]), "n_photos": len(photos)}
    for c in ("motion", "rate", "struct", "type"):
        d[c] = np.array([r[c] for r in rows])
    d["k"] = np.array([int(r["k"]) for r in rows])
    for c in ("pkt_size", "psnr", "base_logit", "small_logit", "v9_m"):
        d[c] = np.array([float(r[c]) for r in rows])
    return d


class Boot:
    """Ratio-of-sums statistics per photo, bootstrapped by photo multiplicities (one matrix for everything)."""
    def __init__(self, n_photos):
        rng = np.random.default_rng(SEED)
        self.W = np.stack([np.bincount(rng.integers(n_photos, size=n_photos), minlength=n_photos) for _ in range(REPS)])
        self.n = n_photos

    def mean(self, photo, values) -> dict:
        s = np.bincount(photo, values, self.n)
        c = np.bincount(photo, None, self.n).astype(float)
        boot = (self.W @ s) / np.maximum(self.W @ c, 1e-12)
        return {"value": float(s.sum() / c.sum()), "ci95": np.percentile(boot, [2.5, 97.5]).tolist(), "n": int(c.sum())}

    def diff(self, pa, va, pb, vb) -> dict:
        """mean(a) - mean(b), both resampled with the same photos."""
        sa, ca = np.bincount(pa, va, self.n), np.bincount(pa, None, self.n).astype(float)
        sb, cb = np.bincount(pb, vb, self.n), np.bincount(pb, None, self.n).astype(float)
        boot = (self.W @ sa) / np.maximum(self.W @ ca, 1e-12) - (self.W @ sb) / np.maximum(self.W @ cb, 1e-12)
        return {"value": float(sa.sum() / ca.sum() - sb.sum() / cb.sum()),
                "ci95": np.percentile(boot, [2.5, 97.5]).tolist(), "n": [int(ca.sum()), int(cb.sum())]}


def main():
    d = load()
    th = thresholds()
    B = Boot(d["n_photos"])
    motions, rates, structs = ("static", "pan"), ("crf23", "crf28", "crf33", "2mbps"), ("I", "IP", "IPB")
    res = {"photos": d["n_photos"], "frames": int(len(d["k"])), "thresholds": th,
           "bootstrap": f"{REPS} resamples over the photos, seed {SEED}"}

    # quality and size by structure and type
    q = {}
    for mo in motions:
        for ra in rates:
            for st in structs:
                for ty in ("I", "P", "B"):
                    sel = (d["motion"] == mo) & (d["rate"] == ra) & (d["struct"] == st) & (d["type"] == ty)
                    if sel.any():
                        q[f"{mo}/{ra}/{st}/{ty}"] = {"n": int(sel.sum()), "psnr_mean": float(d["psnr"][sel].mean()),
                                                     "pkt_size_mean": float(d["pkt_size"][sel].mean())}
    res["quality"] = q

    # 1. false alarms by type
    fa = {}
    for m, col in SCORES.items():
        for cname, t in th[m].items():
            c = calls(m, d[col], t).astype(float)
            for mo in motions:
                for ra in rates:
                    for st in structs:
                        for ty in ("I", "P", "B"):
                            sel = (d["motion"] == mo) & (d["rate"] == ra) & (d["struct"] == st) & (d["type"] == ty)
                            if sel.any():
                                fa[f"{m}/{cname}/{mo}/{ra}/{st}/{ty}"] = B.mean(d["photo"][sel], c[sel])
    res["false_alarms"] = fa
    total = {}
    for m, col in SCORES.items():
        for cname, t in th[m].items():
            c = calls(m, d[col], t)
            total[f"{m}/{cname}"] = {"blurred": int(c.sum()), "of": int(len(c))}
    res["false_alarms_total"] = total

    # 2. paired difference against the I-only coding of the same source frame
    key = lambda sel: list(zip(d["photo"][sel], d["motion"][sel], d["rate"][sel], d["k"][sel]))      # noqa: E731
    ref_sel = d["struct"] == "I"
    ref = {k: i for k, i in zip(key(ref_sel), np.flatnonzero(ref_sel))}
    paired = {}
    for st in ("IP", "IPB"):
        sel = np.flatnonzero(d["struct"] == st)
        j = np.array([ref[k] for k in key(d["struct"] == st)])
        for m, col in SCORES.items():
            diff = d[col][sel] - d[col][j]
            for mo in motions:
                for ra in rates:
                    for ty in ("I", "P", "B"):
                        g = (d["motion"][sel] == mo) & (d["rate"][sel] == ra) & (d["type"][sel] == ty)
                        if not g.any():
                            continue
                        p = d["photo"][sel][g]
                        out = {"score_difference": B.mean(p, diff[g]),
                               "share_scored_higher_than_I_only": B.mean(p, (diff[g] > 0).astype(float)),
                               "share_tied": float(np.mean(diff[g] == 0))}
                        for cname, t in th[m].items():
                            ca = calls(m, d[col][sel][g], t).astype(float)
                            cb = calls(m, d[col][j][g], t).astype(float)
                            out[f"blurred_share_difference_{cname}"] = B.mean(p, ca - cb)
                        paired[f"{m}/{mo}/{ra}/{st}/{ty}"] = out
    res["paired_vs_I_only"] = paired

    # 3. R4K-1 repeated: the in-clip I-frame against its two neighbours
    rep = {}
    idx = {k: i for k, i in zip(zip(d["photo"], d["motion"], d["rate"], d["struct"], d["k"]), range(len(d["k"])))}
    for st in ("IP", "IPB"):
        for m, col in SCORES.items():
            for mo in motions:
                for ra in rates:
                    rows_g, rows_n = [], []
                    for ph in range(d["n_photos"]):
                        for k in (15, 30, 45):
                            g = [idx[(ph, mo, ra, st, kk)] for kk in (k - 1, k, k + 1)]
                            n = [idx[(ph, mo, ra, "I", kk)] for kk in (k - 1, k, k + 1)]
                            assert d["type"][g[1]] == "I" and d["type"][g[0]] != "I" and d["type"][g[2]] != "I"
                            rows_g.append((ph, d[col][g])); rows_n.append((ph, d[col][n]))

                    def shares(rows):
                        ph = np.array([r[0] for r in rows])
                        v = np.array([r[1] for r in rows])
                        lowest = v[:, 1] < np.minimum(v[:, 0], v[:, 2])
                        tie = (v[:, 1] == np.minimum(v[:, 0], v[:, 2]))
                        return ph, lowest.astype(float), tie.astype(float)
                    pg, lg, tg = shares(rows_g)
                    pn, ln, tn = shares(rows_n)
                    rep[f"{m}/{mo}/{ra}/{st}"] = {"i_frame_sharpest": B.mean(pg, lg), "tied": B.mean(pg, tg),
                                                   "null_I_only_same_positions": B.mean(pn, ln),
                                                   "null_tied": B.mean(pn, tn),
                                                   "difference": B.diff(pg, lg, pn, ln)}
    res["r4k1_repeat"] = rep

    # 4. quality-matched: P/B frames of IP / IPB against the I-only frames in the same 1 dB PSNR bin
    # Within each bin the P/B mean minus the I-only mean, weighted by the bin's P/B count; bins with >= 30 frames on
    # both sides. A resample where a bin loses one side drops that bin for that resample.
    qm = {}
    bins = np.floor(d["psnr"]).astype(int)
    per_photo = lambda ix, v: (np.bincount(d["photo"][ix], v[ix], d["n_photos"]),                  # noqa: E731
                               np.bincount(d["photo"][ix], None, d["n_photos"]).astype(float))
    for m, col in SCORES.items():
        for mo in motions:
            ref_sel = (d["motion"] == mo) & (d["struct"] == "I")
            pb_sel = (d["motion"] == mo) & (d["struct"] != "I") & (d["type"] != "I")
            rows_bins, num, den, val, tot = [], np.zeros(REPS), np.zeros(REPS), 0.0, 0
            for b_ in np.unique(bins[pb_sel]):
                a_ = np.flatnonzero(pb_sel & (bins == b_))
                c_ = np.flatnonzero(ref_sel & (bins == b_))
                if len(a_) < 30 or len(c_) < 30:
                    continue
                ma_, mc_ = float(d[col][a_].mean()), float(d[col][c_].mean())
                rows_bins.append({"psnr_bin": int(b_), "n_pb": len(a_), "n_I_only": len(c_), "mean_pb": ma_,
                                  "mean_I_only": mc_})
                val += len(a_) * (ma_ - mc_); tot += len(a_)
                (s_a, c_a), (s_c, c_c) = per_photo(a_, d[col]), per_photo(c_, d[col])
                wa, wc = B.W @ c_a, B.W @ c_c
                ok = (wa > 0) & (wc > 0)
                diff = np.where(ok, (B.W @ s_a) / np.maximum(wa, 1e-12) - (B.W @ s_c) / np.maximum(wc, 1e-12), 0.0)
                num += np.where(ok, wa, 0.0) * diff
                den += np.where(ok, wa, 0.0)
            if rows_bins:
                qm[f"{m}/{mo}"] = {"bins": rows_bins, "weighted_difference_pb_minus_I_only": {
                    "value": val / tot, "ci95": np.percentile(num / np.maximum(den, 1e-12), [2.5, 97.5]).tolist(),
                    "pb_frames_in_matched_bins": tot, "pb_frames": int(pb_sel.sum())}}
    res["quality_matched"] = qm
    (DATA / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")
    print(f"{DATA / 'results.json'}: {res['frames']} frames, {res['photos']} photos")
    print("false alarms total:", json.dumps(res["false_alarms_total"]))


if __name__ == "__main__":
    main()
