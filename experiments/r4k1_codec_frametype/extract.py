"""R4K-1 step 1: one row per frame of a video, from the video stream, the decoded frames and two detector sessions.

usage: python extract.py --label A --video VIDEO --r3 SESSION_DIR --v9 SESSION_DIR --out DIR [--workers 24]

Writes DIR/frames_<label>.csv.gz (per-frame variables, PROTOCOL.md section 3) and DIR/checks_<label>.json (input
digests, alignment checks, letterbox). Shots are added by analyse.py from the thumbnail differences stored here.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import correlate

CELL = 16


def valid_cells(height: int, width: int, pooling: str = "lse") -> np.ndarray:
    """The cells a session's frame score reads (Frame Blur Studio's maps.valid_cells, copied unchanged for this
    publication). lse: the image padded to whole cells, a cell counts when at least half its pixels are inside the
    image. mass: the image cut to whole cells, every cell counts (the outer ring included)."""
    if pooling == "mass":
        return np.ones((height // CELL, width // CELL), dtype=bool)
    shape = ((height + CELL - 1) // CELL, (width + CELL - 1) // CELL)
    rows = np.minimum(CELL, height - np.arange(shape[0]) * CELL)
    cols = np.minimum(CELL, width - np.arange(shape[1]) * CELL)
    return rows[:, None] * cols[None, :] >= CELL * CELL / 2


KERNEL = np.array([[3, 10, 3], [10, 92, 10], [3, 10, 3]], float) / 144      # the published codec comparison
LUMA = np.array([0.2126729, 0.7151522, 0.0721750])
_LIN = np.where(np.arange(256) / 255.0 <= 0.04045, np.arange(256) / 255.0 / 12.92,
                ((np.arange(256) / 255.0 + 0.055) / 1.055) ** 2.4)
FLOW_SIZE = (480, 270)
THUMB = (64, 36)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def residual(rgb8: np.ndarray) -> np.ndarray:
    """d = Y - K*Y on linear-light luminance, float64, replicated borders (meshlab/tools/codec_compare.py fp)."""
    y = _LIN[rgb8] @ LUMA
    return y - correlate(y, KERNEL, mode="nearest")


def codec_stats(d: np.ndarray) -> dict:
    """The statistics of the published codec comparison (codec_compare.py stats), unchanged."""
    h, w = d.shape
    h, w = h - h % 16, w - w % 16
    d = d[:h, :w]
    a = np.abs(d)
    c = a.reshape(h // 16, 16, w // 16, 16).mean((1, 3))
    flat_c, edge_c = c <= np.quantile(c, 0.3), c >= np.quantile(c, 0.9)
    up = lambda m: np.repeat(np.repeat(m, 16, 0), 16, 1)  # noqa: E731
    fm, em = up(flat_c), up(edge_c)
    flat, edge = a[fm].mean(), a[em].mean()
    dx, dy = np.abs(np.diff(d, axis=1)), np.abs(np.diff(d, axis=0))
    bx, by = (np.arange(w - 1) % 8) == 7, (np.arange(h - 1) % 8) == 7
    block = (dx[:, bx].mean() + dy[by].mean()) / (dx[:, ~bx].mean() + dy[~by].mean())
    x, y = d[:, :-1][fm[:, :-1]], d[:, 1:][fm[:, :-1]]
    hc = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else 0.0
    return {"flat_e3": flat * 1e3, "zero": float((a < 2e-4).mean()), "edge_e3": edge * 1e3,
            "flat_edge": flat / max(edge, 1e-12), "block8": float(block), "hcorr": hc}


def ffprobe_frames(video: Path) -> list[dict]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "frame=key_frame,pict_type,pkt_size,best_effort_timestamp_time", "-of", "json", str(video)],
                         capture_output=True, check=True).stdout
    return json.loads(out)["frames"]


def records(session: Path) -> dict[int, dict]:
    with (session / "records.jsonl").open(encoding="utf-8") as f:
        return {r["index"]: r for r in map(json.loads, f)}


def letterbox(paths: list[Path]) -> tuple[int, int, int, int]:
    """Active picture box (top, bottom, left, right) from the median of spread frames: rows/columns darker than 6."""
    stack = np.stack([cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths])
    med = np.median(stack, 0)
    rows, cols = np.flatnonzero(med.mean(1) >= 6), np.flatnonzero(med.mean(0) >= 6)
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def _cells(path: Path, mask: np.ndarray) -> tuple[float, float]:
    with np.load(path, allow_pickle=False) as m:
        logits = np.asarray(m["logits"], np.float32)
    v = logits[mask] if logits.shape == mask.shape else logits[:mask.shape[0], :mask.shape[1]][mask]
    return float((v >= 0).mean()), float(v.mean())


def _work(job: dict) -> list[dict]:
    """A contiguous range of frames: pixel statistics, motion, thumbnails and both detectors' cell summaries."""
    cv2.setNumThreads(1)
    r3, v9, box = Path(job["r3"]), Path(job["v9"]), job["box"]
    top, bottom, left, right = box
    rows, prev_small, prev_thumb, prev_hist = [], None, None, None
    start = job["start"]
    for k in range(max(0, start - 1), job["stop"]):
        bgr = cv2.imread(str(r3 / "frames" / f"frame_{k:09d}.png"), cv2.IMREAD_COLOR)
        grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(grey, FLOW_SIZE, interpolation=cv2.INTER_AREA)
        thumb = cv2.resize(grey, THUMB, interpolation=cv2.INTER_AREA).astype(np.float32)
        hue = cv2.cvtColor(cv2.resize(bgr, THUMB, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)[..., 0]
        hist = np.histogram(hue, bins=16, range=(0, 180))[0] / hue.size
        if k >= start:
            row = {"index": k}
            active = grey[top:bottom, left:right]
            row["luma"] = float(active.mean())
            row["laplacian"] = float(np.abs(cv2.Laplacian(active.astype(np.float32), cv2.CV_32F)).mean())
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)[top:bottom, left:right]
            row.update(codec_stats(residual(rgb)))
            if prev_small is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_small, small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                row["flow"] = float(np.hypot(flow[..., 0], flow[..., 1]).mean() * grey.shape[1] / FLOW_SIZE[0])
                row["thumb_diff"] = float(np.abs(thumb - prev_thumb).mean())
                row["hist_diff"] = float(np.abs(hist - prev_hist).sum())
            else:
                row["flow"] = row["thumb_diff"] = row["hist_diff"] = float("nan")
            for name, session, mask in (("r3", r3, job["mask_r3"]), ("v9", v9, job["mask_v9"])):
                share, mean = _cells(session / "cells" / f"frame_{k:09d}.npz", np.asarray(mask, bool))
                row[f"{name}_blur_cells"], row[f"{name}_mean_logit"] = share, mean
            rows.append(row)
        prev_small, prev_thumb, prev_hist = small, thumb, hist
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--r3", type=Path, required=True)
    ap.add_argument("--v9", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--chunk", type=int, default=400)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    checks: dict = {"label": a.label}

    probe = ffprobe_frames(a.video)
    rec3, rec9 = records(a.r3), records(a.v9)
    n = len(probe)
    checks["frames"] = {"ffprobe": n, "round3_session": len(rec3), "v9_session": len(rec9)}
    assert len(rec3) == len(rec9) == n and set(rec3) == set(rec9) == set(range(n)), checks["frames"]
    ts_probe = np.array([float(f["best_effort_timestamp_time"]) for f in probe])
    ts_sess = np.array([rec3[k]["timestamp"] for k in range(n)])
    offset = float(np.median(ts_sess - ts_probe))
    checks["timestamps"] = {"median_offset_s": offset, "max_abs_error_after_offset_s": float(np.abs(ts_sess - ts_probe - offset).max()),
                            "frame_period_s": float(np.median(np.diff(ts_probe)))}
    for name, session in (("round3", a.r3), ("v9", a.v9)):
        manifest = json.loads((session / "session.json").read_text(encoding="utf-8"))
        checks[f"{name}_session"] = {"dir": session.name, "model": manifest.get("model"),
                                     "status": manifest.get("status"), "decode_warning": manifest.get("decode_warning")}

    rng = np.random.default_rng(20261027)
    sample = sorted(rng.choice(n, 100, replace=False).tolist())
    same = sum(sha256(a.r3 / rec3[k]["frame_path"]) == sha256(a.v9 / rec9[k]["frame_path"]) for k in sample)
    checks["identical_png_in_both_sessions"] = {"sampled": len(sample), "identical": same}

    # Index alignment: the session PNG matches ffmpeg's decoded frame k better than frames k - 1 and k + 1.
    probe_idx = sorted(rng.choice(np.arange(2, n - 2), 20, replace=False).tolist())
    h, w = cv2.imread(str(a.r3 / rec3[0]["frame_path"]), cv2.IMREAD_GRAYSCALE).shape
    align = []
    period = checks["timestamps"]["frame_period_s"]
    for k in probe_idx:
        # Accurate input seek to frame k - 1 (by its ffprobe timestamp), then three decoded frames.
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{ts_probe[k - 1] - 0.25 * period:.6f}", "-i", str(a.video),
                              "-frames:v", "3", "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
        frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(np.float32)
        png = cv2.imread(str(a.r3 / rec3[k]["frame_path"]), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        mae = [float(np.abs(f - png).mean()) for f in frames]
        align.append({"index": k, "mae_k-1_k_k+1": mae, "best_is_k": int(np.argmin(mae)) == 1})
    checks["index_alignment"] = {"probes": len(align), "best_is_k": sum(x["best_is_k"] for x in align), "detail": align}

    spread = np.linspace(0, n - 1, 200).round().astype(int)
    box = letterbox([a.r3 / rec3[k]["frame_path"] for k in spread])
    checks["letterbox_active_box_top_bottom_left_right"] = box
    mask_r3, mask_v9 = valid_cells(h, w, "lse"), valid_cells(h, w, "mass")
    checks["cells"] = {"round3_grid": list(mask_r3.shape), "round3_counted": int(mask_r3.sum()),
                       "v9_grid": list(mask_v9.shape), "v9_counted": int(mask_v9.sum())}
    checks["video_sha256"] = sha256(a.video)
    checks["video_bytes"] = a.video.stat().st_size

    jobs = [{"r3": str(a.r3), "v9": str(a.v9), "box": box, "start": s, "stop": min(n, s + a.chunk),
             "mask_r3": mask_r3.tolist(), "mask_v9": mask_v9.tolist()} for s in range(0, n, a.chunk)]
    rows: list[dict] = []
    with ProcessPoolExecutor(a.workers) as pool:
        for done, part in enumerate(pool.map(_work, jobs), 1):
            rows.extend(part)
            if done % 10 == 0 or done == len(jobs):
                print(f"{a.label}: {done}/{len(jobs)} chunks", flush=True)
    rows.sort(key=lambda r: r["index"])
    assert [r["index"] for r in rows] == list(range(n))
    for r, f in zip(rows, probe):
        k = r["index"]
        r.update(type=f["pict_type"], key=int(f["key_frame"]), coded_bytes=int(f["pkt_size"]),
                 t=float(f["best_effort_timestamp_time"]),
                 r3_score=rec3[k]["probability"], r3_logit=rec3[k]["image_logit"],
                 v9_mass=rec9[k]["blur_mass"], v9_score=rec9[k]["probability"])
    columns = ["index", "t", "type", "key", "coded_bytes", "r3_score", "r3_logit", "v9_mass", "v9_score",
               "r3_blur_cells", "r3_mean_logit", "v9_blur_cells", "v9_mean_logit", "luma", "laplacian",
               "flat_e3", "edge_e3", "flat_edge", "zero", "block8", "hcorr", "flow", "thumb_diff", "hist_diff"]
    with gzip.open(a.out / f"frames_{a.label}.csv.gz", "wt", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        writer.writerows({c: r[c] for c in columns} for r in rows)
    (a.out / f"checks_{a.label}.json").write_text(json.dumps(checks, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in checks.items() if k not in ("index_alignment",)}, indent=1)[:2000])
    print("alignment:", checks["index_alignment"]["best_is_k"], "/", checks["index_alignment"]["probes"])


if __name__ == "__main__":
    main()
