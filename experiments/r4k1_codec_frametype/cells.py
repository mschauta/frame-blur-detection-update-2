"""R4K-1 step 2 (Q8): where in the picture the P/B frame gains blur evidence over the I-frame of its shot.

usage: python cells.py --label A --r3 SESSION_DIR --v9 SESSION_DIR [--workers 24]

For every mid-shot and cut I-frame comparison (common.comparisons): the I-frame's per-cell residual level (mean |d| of
the published codec residual in each 16 px cell), split into deciles within the frame (flat -> textured, active
picture only), and per decile and detector: the mean cell-logit difference (other - I), the share of cells crossing
from p < 0.5 to p >= 0.5, and the reverse share. Writes data/cells_<label>.npz.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from common import DATA, VARIANTS, comparisons, load
from extract import residual

ROWS, COLS = 67, 120          # whole 16 px cells of a 1920 x 1080 frame, the grid both detectors share


def _cell_residual(path: Path) -> np.ndarray:
    rgb = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    a = np.abs(residual(rgb))[:ROWS * 16, :COLS * 16]
    return a.reshape(ROWS, 16, COLS, 16).mean((1, 3))


def _logits(session: Path, k: int) -> np.ndarray:
    with np.load(session / "cells" / f"frame_{k:09d}.npz", allow_pickle=False) as m:
        return np.asarray(m["logits"], np.float32)[:ROWS, :COLS]


def _work(job: dict) -> list[tuple]:
    cv2.setNumThreads(1)
    r3, v9, active = Path(job["r3"]), Path(job["v9"]), np.asarray(job["active"], bool)
    out = []
    for ref, others in job["refs"]:
        level = _cell_residual(r3 / "frames" / f"frame_{ref:09d}.png")
        edges = np.quantile(level[active], np.linspace(0, 1, 11))
        decile = np.clip(np.searchsorted(edges, level, side="right") - 1, 0, 9)
        base = {name: _logits(s, ref) for name, s in (("r3", r3), ("v9", v9))}
        for other in others:
            values = np.full((2, 10, 3), np.nan)
            for m, (name, session) in enumerate((("r3", r3), ("v9", v9))):
                a, b = base[name], _logits(session, other)
                for d in range(10):
                    sel = active & (decile == d)
                    if sel.any():
                        values[m, d] = [float((b[sel] - a[sel]).mean()), float(((a[sel] < 0) & (b[sel] >= 0)).mean()),
                                        float(((a[sel] >= 0) & (b[sel] < 0)).mean())]
            out.append((ref, other, values))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--r3", type=Path, required=True)
    ap.add_argument("--v9", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--variant", choices=VARIANTS, default="revised")
    a = ap.parse_args()
    table = load(a.label, a.variant)
    checks = json.loads((DATA / f"checks_{a.label}.json").read_text(encoding="utf-8"))
    top, bottom, left, right = checks["letterbox_active_box_top_bottom_left_right"]
    centres_y, centres_x = np.arange(ROWS) * 16 + 8, np.arange(COLS) * 16 + 8
    active = ((centres_y >= top) & (centres_y < bottom))[:, None] & ((centres_x >= left) & (centres_x < right))[None]
    pairs = [c for c in comparisons(table) if c["kind"] in ("mid_I", "cut_I")]
    kind = {(c["ref"], c["other"]): c["kind"] for c in pairs}
    shot = {(c["ref"], c["other"]): c["shot"] for c in pairs}
    by_ref: dict[int, list[int]] = {}
    for c in pairs:
        by_ref.setdefault(c["ref"], []).append(c["other"])
    refs = sorted(by_ref.items())
    jobs = [{"r3": str(a.r3), "v9": str(a.v9), "active": active.tolist(), "refs": refs[i:i + 20]}
            for i in range(0, len(refs), 20)]
    results = []
    with ProcessPoolExecutor(a.workers) as pool:
        for part in pool.map(_work, jobs):
            results.extend(part)
    results.sort(key=lambda r: (r[0], r[1]))
    np.savez_compressed(DATA / f"cells_{a.label}_{a.variant}.npz",
                        ref=np.array([r[0] for r in results]), other=np.array([r[1] for r in results]),
                        kind=np.array([kind[(r[0], r[1])] for r in results]),
                        shot=np.array([shot[(r[0], r[1])] for r in results]),
                        values=np.stack([r[2] for r in results]),
                        layout=np.array(["detector: r3, v9", "decile: flat .. textured",
                                         "metric: mean logit diff, crossing to blur, crossing to sharp"]))
    print(f"{a.label} {a.variant}: {len(results)} comparisons, {int(active.sum())} active cells")


if __name__ == "__main__":
    main()
