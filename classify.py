"""Sort images into sharp / blur with a published weight file.

    python classify.py --weights weights/v9_base_selfmap_v2_ep019.pt --input frames --output sorted --mass 0.05
    python classify.py --weights weights/v9_base_selfmap_v2_ep019.pt --input frames --output sorted --mass 0 --move
    python classify.py --weights weights/v9_base_selfmap_v2_ep019.pt --input frames --output sorted --mass 0.05 --dry-run
    python classify.py --weights weights/v9_base_selfmap_v2_ep019.pt --input frames --output sorted --mass 0.05 --resume
    python classify.py --weights weights/round3_base_p99_ep029.pt --input frames --output sorted --recall 95

(the weight files are release assets: v9_base_selfmap_v2_ep019.pt of this repository, round3_base_p99_ep029.pt of
Update 1; put them into weights/, see README.md)

Every image of --input (png, jpg, jpeg, bmp, tif, tiff, webp; not recursive) is scored and copied (or moved) to
<output>/sharp or <output>/blur. <output>/results.csv lists every image with its score and decision.

The decision depends on the weight file:
  blur mass (Update 2, "pooling": "mass")  the score is the blur mass M (0 = no cell says blur; not a probability).
      There is no stored threshold: an image is called blurred when M > --mass. --mass 0 rejects every image with any
      blur cell, the strictest choice; README.md, "Weights", gives reference points.
  log-sum-exp (Update 1 and earlier)  the score is the sigmoid output, the threshold the one stored with the weights:
      it catches 90, 95 or 98% of the blurred frames of the validation split (--recall).

results.csv is written as the run goes: a row is added right after its image was copied or moved, so an interrupted
run loses at most the image in hand. --resume continues such a run: images already in results.csv are skipped, and an
image found in <output>/sharp or <output>/blur without a row (moved just before the interruption) is scored again in
place and gets its row. <output>/run.json records the run's settings (the weight file's sha256, the score, the
threshold, the input folder, copy or move); --resume continues only with the same settings, so one results.csv never
mixes two decision rules. --dry-run only scores, into results_dry_run.csv (settings in run_dry_run.json), and copies
or moves nothing, also with --resume; a real run in the same output folder afterwards starts from scratch.

Images are scored one at a time, as for the published numbers: batching and precision move the scores by a little.

The models were trained for video frames (1080p, decoded from H.264, stored losslessly as PNG); see METHOD.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from blurdetect import load_detector

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
# The release assets (README.md, "Weights").
PUBLISHED_SHA256 = {
    "10d6aa1ebe8cb4bf6bb3cd048aaf75edb07fc1bc23a96d1e8a645a87d0f70ba3": "v9_base_selfmap_v2_ep019.pt (Update 2)",
    "9840160f0c337fc1312c2566abf4231be0bd7ee709f65f489fd8f81c65b8d0e6": "round3_base_p99_ep029.pt (Update 1)",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def score(model, path: Path, device: str, mass: bool) -> float:
    """The blur mass M (mass models) or the sigmoid of the image logit."""
    x = torch.from_numpy(np.asarray(Image.open(path).convert("RGB")).copy()).permute(2, 0, 1)[None].to(device)
    with torch.no_grad():
        if device.startswith("cuda"):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                value = model(x)["image"]
        else:
            value = model(x)["image"]
    value = value.float()
    return float(value) if mass else float(torch.sigmoid(value))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--mass", type=float, help="blur mass models: blurred when M > this value (>= 0)")
    ap.add_argument("--recall", type=int, choices=(90, 95, 98),
                    help="log-sum-exp models: the stored threshold that catches this share of blurred frames (default 95)")
    ap.add_argument("--move", action="store_true", help="move the files instead of copying them")
    ap.add_argument("--dry-run", action="store_true", help="only score, into results_dry_run.csv; copy or move nothing")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run into the same --output")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)

    digest = sha256(a.weights)
    known = PUBLISHED_SHA256.get(digest)
    print(f"weights sha256 {digest}: " + (f"the published {known}" if known else "not a published release asset"))
    model, meta = load_detector(a.weights, a.device)
    is_mass = meta.get("pooling", "lse") == "mass"
    if is_mass:
        if a.recall is not None:
            raise SystemExit("this weight file gives a blur mass M and stores no recall thresholds; use --mass "
                             "(README.md, Weights, gives reference points)")
        # NaN or infinity would call every image sharp
        if a.mass is None or not math.isfinite(a.mass) or a.mass < 0:
            raise SystemExit("this weight file gives a blur mass M: choose the threshold with --mass (a finite "
                             "number >= 0; README.md, Weights, gives reference points)")
        thr, field = a.mass, "blur_mass"
        rule = f"blurred when M > {thr:g}"
    else:
        if a.mass is not None:
            raise SystemExit("this weight file gives a log-sum-exp score with stored thresholds; use --recall")
        recall = a.recall or 95
        thr, field = meta["thresholds"][f"r{recall}"], "blur_score"
        rule = f"threshold {thr:.6f} (catches {recall}% of the blurred validation frames)"
    fields = ["file", field, "decision"]
    files = sorted(p for p in a.input.iterdir() if p.is_file() and p.suffix.lower() in EXTS)
    a.output.mkdir(parents=True, exist_ok=True)
    results = a.output / ("results_dry_run.csv" if a.dry_run else "results.csv")
    run_file = a.output / ("run_dry_run.json" if a.dry_run else "run.json")
    # What a resumed run must share with the run it continues.
    run = {"weights_sha256": digest, "score": field, "threshold": thr, "input": str(a.input.resolve())}
    if not a.dry_run:
        run["move"] = a.move
    folders = {label: a.output / label for label in ("sharp", "blur")}
    done: set[str] = set()
    if a.resume:
        if results.exists() and not run_file.exists():
            raise SystemExit(f"{results} has no {run_file.name} (an earlier classify.py wrote it), so its settings "
                             "cannot be checked; use a new output folder")
        if run_file.exists():
            saved = json.loads(run_file.read_text(encoding="utf-8"))
            changed = [key for key in run.keys() | saved.keys() if saved.get(key) != run.get(key)]
            if changed:
                raise SystemExit(f"{run_file} records another {', '.join(sorted(changed))}; resume with the same "
                                 "settings, or use a new output folder")
        if results.exists():
            with open(results, newline="", encoding="utf-8") as fh:
                done = {row["file"] for row in csv.DictReader(fh)}
    else:
        if results.exists():
            raise SystemExit(f"{results} exists; use --resume to continue that run, or a new output folder")
        if not a.dry_run:
            for folder in folders.values():
                if folder.exists() and any(folder.iterdir()):
                    raise SystemExit(f"{folder} is not empty; use a new or empty output folder, or --resume")
    if not a.dry_run:
        for folder in folders.values():
            folder.mkdir(exist_ok=True)
    run_file.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    # Moved just before an interruption: in an output folder, but without a row. A dry run leaves the folders alone.
    orphans = [p for folder in folders.values() if folder.exists() for p in sorted(folder.iterdir())
               if a.resume and not a.dry_run and p.is_file() and p.suffix.lower() in EXTS and p.name not in done]
    todo = [f for f in files if f.name not in done and f.name not in {p.name for p in orphans}]
    print(f"{len(todo) + len(orphans)} images to score ({len(done)} already in results.csv), model input: "
          f"{meta['input_description']}, epoch {meta['epoch']}, {rule}, device {a.device}"
          + (", dry run" if a.dry_run else ""))
    new_file = not results.exists()
    counts = {"sharp": 0, "blur": 0}
    with open(results, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fields)
        if new_file:
            writer.writeheader()
            fh.flush()
        for i, f in enumerate(orphans + todo, 1):
            value = score(model, f, a.device, is_mass)
            label = "blur" if (value > thr if is_mass else value >= thr) else "sharp"
            if not a.dry_run:
                if f in orphans:
                    if f.parent != folders[label]:
                        shutil.move(f, folders[label] / f.name)
                else:
                    (shutil.move if a.move else shutil.copy2)(f, folders[label] / f.name)
            # M keeps its significant digits: a tiny positive mass must not print as 0 next to "blur"
            writer.writerow({"file": f.name, field: f"{value:.7g}" if is_mass else f"{value:.6f}", "decision": label})
            fh.flush()
            counts[label] += 1
            print(f"[{i}/{len(orphans) + len(todo)}] {f.name}: {label} ({'M' if is_mass else 'score'} {value:.4f})")
    print(f"sharp {counts['sharp']}, blur {counts['blur']} in this run -> {a.output}")


if __name__ == "__main__":
    main()
