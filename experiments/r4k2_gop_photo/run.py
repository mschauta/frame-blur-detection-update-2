"""R4K-2: the coded frame type on sharp etalon photos (PROTOCOL.md).

As run on 2026-10-28/29 (blurtrainer/experiments/r4k2_gop_foto/run.py), with the publication edits listed in
MANIFEST.json: the etalon albums' names are withheld and the training's x264 settings are copied in. It needs the
non-public photos and two unpublished modules of the training code (probe_round3: the photo ladder's order and photo
loading; blurtrainer.data.read_index: the photo index), so it does not rerun outside the author's machine. The coding
chain (code()) is complete here; analyse.py reruns from the published table.

100 landscape etalon photos (the photo ladder's order, never trained on), a 1440 x 1080 crop, two 48-frame clips per
photo (static; a pan of 1 integer pixel per frame), each coded three ways (I only; IP and IPB with an I-frame every 15
frames) at four rates, through the training's coding chain. Every decoded frame is scored by the round3 Base, the
round3 Small (Update 1 loader) and v9 (Update 2 loader). Every frame is 0 px sharp, so every blurred call is false.

usage: python experiments/r4k2_gop_foto/run.py [--photos 100] [--check]     (resumable; --check: one photo, no CSV)
Output: experiments/r4k2_gop_foto/data/frames.csv (one row per frame, appended per clip), data/photos.json
(published as data/frames.csv.gz with the photo ids replaced by p000 ... p099; data/photos.json is not published)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import probe_round3 as P                                     # noqa: E402  the ladder's photos and their order
from blurtrainer.data import read_index                      # noqa: E402
# the training's x264 settings (blurtrainer.synth.ENCODER), copied in for publication
ENCODER = ["-c:v", "libx264", "-profile:v", "high", "-preset", "medium", "-threads", "1",
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
ETALON_ALBUMS: tuple[str, ...] = ()           # publication edit: the two etalon albums' bin names, withheld

DATA = HERE / "data"
FRAMES = DATA / "frames.csv"
W, H, N, PAN_STEP = 1440, 1080, 48, 1
STRUCT = {"I": "keyint=1:min-keyint=1:scenecut=0:bframes=0",
          "IP": "keyint=15:min-keyint=15:scenecut=0:bframes=0:open-gop=0",
          "IPB": "keyint=15:min-keyint=15:scenecut=0:bframes=3:open-gop=0"}
RATES = {"crf23": ["-crf", "23"], "crf28": ["-crf", "28"], "crf33": ["-crf", "33"],
         "2mbps": ["-b:v", "2M", "-maxrate", "2M", "-bufsize", "4M"]}
MOTIONS = ("static", "pan")
UP = ROOT.parent
MODELS = {"base": ("update1", UP / "frame-blur-detection-update-1" / "weights" / "round3_base_p99_ep029.pt",
                   "9840160f0c337fc1312c2566abf4231be0bd7ee709f65f489fd8f81c65b8d0e6"),
          "small": ("update1", UP / "frame-blur-detection-update-1" / "weights" / "round3_small_p99_ep029.pt",
                    "b715a163f9c6927464d97b7002967c8361c6a0de2b87e3103dc72a7ba733d48b"),
          "v9": ("update2", ROOT / "weights" / "v9_base_selfmap_v2_ep019.pt",
                 "10d6aa1ebe8cb4bf6bb3cd048aaf75edb07fc1bc23a96d1e8a645a87d0f70ba3")}
COLS = ["photo", "motion", "rate", "struct", "k", "type", "pkt_size", "psnr", "base_logit", "small_logit", "v9_m"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def loader(repo: str):
    spec = importlib.util.spec_from_file_location(f"blurdetect_{repo}", UP / f"frame-blur-detection-{repo.replace('update', 'update-')}"
                                                  / "blurdetect" / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_models() -> dict:
    mods = {r: loader(r) for r in ("update1", "update2")}
    out = {}
    for name, (repo, path, sha) in MODELS.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha, name
        model, meta = mods[repo].load_detector(path, "cuda")
        assert (meta.get("pooling", "lse") == "mass") == (name == "v9"), meta
        out[name] = model
    return out


def photos(n: int) -> list[dict]:
    """The first n photos, in the photo ladder's order, that hold the pan (1080 px high after the banner crop, at
    least W + N * PAN_STEP wide). Deviation D1: the ladder's 300 photos hold only 14 such photos (one album's photos are
    about 752 px high after their banner crop), so the order continues over the whole validation / test pool."""
    from PIL import Image
    rows = [r for r in read_index(ROOT / "data" / "round3_index.csv") if r["source"] == "photo" and r["synth"] == "sharp"
            and r["split"] in ("val", "test") and r["shot_bin"] in ETALON_ALBUMS]
    rows.sort(key=lambda r: P.h("ladder", r["sample_id"]))
    picked = []
    for r in rows:
        p = Path(r["png"]) if Path(r["png"]).is_absolute() else P.DATASET / r["png"]
        with Image.open(p) as im:
            w, h = im.size
        h -= int(r["crop_bottom"]) if r.get("crop_bottom") else 0
        if h // 2 * 2 == H and w // 2 * 2 >= W + N * PAN_STEP and w > h:
            assert P.load_photo(r).shape[:2] == (H, w // 2 * 2)
            picked.append(r)
        if len(picked) == n:
            break
    assert len(picked) == n, len(picked)
    return picked


def clip(rgb: np.ndarray, motion: str) -> np.ndarray:
    w = rgb.shape[1]
    if motion == "static":
        x0 = (w - W) // 2
        return np.ascontiguousarray(np.broadcast_to(rgb[:H, x0:x0 + W], (N, H, W, 3)))
    x0 = (w - W - N * PAN_STEP) // 2
    return np.stack([rgb[:H, x0 + k * PAN_STEP:x0 + k * PAN_STEP + W] for k in range(N)])


def _run(cmd, data=None) -> bytes:
    p = subprocess.run(cmd, input=data, capture_output=True)
    if p.returncode:
        raise RuntimeError(f"{cmd[0]} failed: {p.stderr.decode(errors='replace')[-800:]}")
    return p.stdout


def code(src: np.ndarray, struct: str, rate: str) -> tuple[np.ndarray, list[str], list[int]]:
    """48 frames through x264 (the training's chain, synth.h264, with a GOP) and back; frame types from the stream."""
    data, pix = src.tobytes(), "rgb24"
    if data[:3] == b"ID3":                                    # see synth.h264: ffmpeg would read it as an ID3 tag
        data, pix = np.ascontiguousarray(src[..., ::-1]).tobytes(), "bgr24"
    enc = _run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", pix, "-s", f"{W}x{H}", "-r", "30", "-i", "-",
                "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p", *ENCODER, *RATES[rate],
                "-x264-params", STRUCT[struct], "-f", "h264", "-"], data)
    with tempfile.TemporaryDirectory(dir=DATA) as tmp:
        f = Path(tmp) / "clip.h264"
        f.write_bytes(enc)
        probe = json.loads(_run(["ffprobe", "-v", "error", "-f", "h264", "-show_entries", "frame=pict_type,pkt_size",
                                 "-of", "json", str(f)]))["frames"]
        dec = _run(["ffmpeg", "-v", "error", "-threads", "1", "-f", "h264", "-i", str(f), "-fps_mode", "passthrough",
                    "-vf", "scale=in_color_matrix=bt709:in_range=tv,format=rgb24", "-f", "rawvideo", "-"])
    out = np.frombuffer(dec, np.uint8).reshape(-1, H, W, 3)
    assert len(out) == len(probe) == N, (len(out), len(probe))
    return out, [p["pict_type"] for p in probe], [int(p["pkt_size"]) for p in probe]


def psnr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    mse = ((a.astype(np.float32) - b.astype(np.float32)) ** 2).reshape(len(a), -1).mean(1)
    return 10 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-12))


def job(row, motion, rate, struct):
    src = clip(P.load_photo(row), motion)
    dec, types, sizes = code(src, struct, rate)
    q = psnr(dec, src)
    if motion == "pan":                                       # alignment: frame k matches source k better than k + 1
        assert q[N // 2] > psnr(dec[N // 2:N // 2 + 1], src[N // 2 + 1:N // 2 + 2])[0]
    return row["sample_id"], motion, rate, struct, dec, types, sizes, q


def score(models, frames: np.ndarray) -> dict:
    out = {m: [] for m in models}
    for j in range(0, len(frames), 8):
        x = torch.from_numpy(frames[j:j + 8]).permute(0, 3, 1, 2).contiguous().cuda()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for m, model in models.items():
                out[m].append(model(x)["image"].float().cpu().numpy())
    return {m: np.concatenate(v) for m, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", type=int, default=100)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--check", action="store_true", help="one photo, every clip, printed, nothing written")
    a = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    rows = photos(1 if a.check else a.photos)
    jobs = [(r, m, rt, s) for r in rows for m in MOTIONS for rt in RATES for s in STRUCT]
    done = set()
    if FRAMES.exists() and not a.check:
        with open(FRAMES, newline="", encoding="utf-8") as fh:
            done = {(r["photo"], r["motion"], r["rate"], r["struct"]) for r in csv.DictReader(fh)}
    todo = [j for j in jobs if (j[0]["sample_id"], j[1], j[2], j[3]) not in done]
    log(f"R4K-2: {len(rows)} fotó, {len(jobs)} klip ({len(todo)} hátra) × {N} frame, modellek {', '.join(MODELS)}")
    models = load_models()
    if not a.check:
        (DATA / "photos.json").write_text(json.dumps([r["sample_id"] for r in rows], indent=1), encoding="utf-8")
    fh = None if a.check else open(FRAMES, "a", newline="", encoding="utf-8")
    w = None if a.check else csv.DictWriter(fh, COLS)
    if w and not done:
        w.writeheader()
    t0 = time.time()
    with ThreadPoolExecutor(a.threads) as pool:
        ahead = [pool.submit(job, *j) for j in todo[:a.threads + 2]]
        for n in range(len(todo)):
            pid, motion, rate, struct, dec, types, sizes, q = ahead.pop(0).result()
            if n + a.threads + 2 < len(todo):
                ahead.append(pool.submit(job, *todo[n + a.threads + 2]))
            s = score(models, dec)
            if a.check:
                log(f"{motion:6s} {rate:5s} {struct:3s} {''.join(types)}  PSNR {q.min():.1f}-{q.max():.1f}  "
                    f"base {s['base'].min():.1f}..{s['base'].max():.1f}  small {s['small'].min():.1f}..{s['small'].max():.1f}"
                    f"  v9 M {s['v9'].min():.3f}..{s['v9'].max():.3f}")
                continue
            for k in range(N):
                w.writerow({"photo": pid, "motion": motion, "rate": rate, "struct": struct, "k": k, "type": types[k],
                            "pkt_size": sizes[k], "psnr": f"{q[k]:.3f}", "base_logit": f"{s['base'][k]:.6g}",
                            "small_logit": f"{s['small'][k]:.6g}", "v9_m": f"{s['v9'][k]:.6g}"})
            fh.flush()
            if (n + 1) % 24 == 0:
                left = (time.time() - t0) / (n + 1) * (len(todo) - n - 1)
                log(f"  {n + 1} / {len(todo)} klip, hátra kb. {left / 60:.0f} perc")
    if fh:
        fh.close()
    log("kész")


if __name__ == "__main__":
    main()
