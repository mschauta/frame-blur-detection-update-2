# R4K-2: the coded frame type on sharp photos. A controlled GOP test, three detectors

**Summary.** 100 sharp (0 px) etalon photos were made into 48-frame clips, coded three ways at four rates with the
x264 chain used in training, and every decoded frame was scored by three published detectors. Every frame is sharp, so
every blur decision is a false alarm.

- **Coded with I-frames only, a sharp photo is practically never called blurred:** 0.0–0.2% of frames at the
  decision thresholds, by every detector at every rate (v9 at M > 0: up to 0.8%). This is how all training photos were
  coded.
- **Coded as a P- or B-frame, the same source frame scores higher (blurrier).** For the round3 detectors, by 7 to 14
  logits at CRF 23 and 2 Mbit/s, and at every rate in the pan clips. The paired intervals exclude 0. The share called
  blurred at r95 rises by up to 8.1 points (round3 Small, static, 2 Mbit/s); for the Base by up to 7.1 (pan, CRF 33,
  P-frames: +7.1 [+3.5, +11.3]), and by up to 15.1 at r98.
- **At equal quality the effect remains in the pan clips:** P/B frames score about 10 logits higher than I-only frames
  in the same 1 dB PSNR bin, in every bin from 30 to 40 dB.
- **It is not the frame type alone.** At CRF 23 and 2 Mbit/s the group-of-pictures (GOP) coding also makes its
  I-frames much better (about 40 dB against 37 dB or 31.5 dB). The round3 detectors score these very clean I-frames
  about 6–8 logits blurrier than the I-only coding of the same frame.
- **R4K-1 is reproduced with an exact signal.** In the pan clips at CRF 28 and 33 the in-clip I-frame scores lower
  (sharper) than both its neighbours in 98–100% of triples, against about 33% for the same positions coded I-only.
- **v9** gives most of these frames exactly M = 0, so its scores barely move, but its decisions at r95 shift by a
  similar amount as round3's.
- **The false alarms gather on a minority of photos:** at r95, 22–25 of the 100 photos carry all of them.

By the rule fixed in advance, **the codec effect is confirmed** for all three detectors.

## 1. Question and status

Protocol: [PROTOCOL.md](PROTOCOL.md) (English translation) and [PROTOCOL_hu.md](PROTOCOL_hu.md) (the registered
text, with the album names redacted; both hashes in [PROTOCOL.sha256](PROTOCOL.sha256)). It was fixed on 2026-10-28,
before any result. Every measure it lists is reported below. Deviations and additions are listed in section 3.

[R4K-1](../r4k1_codec_frametype/REPORT.md) observed on two films that the detectors score the I-frame inside a shot as
sharper than its P/B neighbours. There the true sharpness was unknown. Here it is known: every frame comes from a
0 px photo. And the frame type is set by the coder, with the same source frame also coded I-only as a paired control.

## 2. Material

- **Photos:** 100 landscape etalon photos from the validation and test part of the photo pool (never trained on), in
  the photo ladder's hash order (D1 below). All 100 are from one of the two albums. Crop 1440 × 1080. The photos are
  not published; in the table they are `p000` … `p099`.
- **Clips:** per photo a static clip (the same crop 48 times) and a pan (the crop moves 1 whole pixel per frame: no
  resampling, every frame stays 0 px). 30 fps.
- **Coding** (`run.py`, `code()`): bt709, tv range, yuv420p, libx264 high, preset medium, 1 thread, scenecut 0,
  closed GOP:
  - `I`: I-frames only;
  - `IP`: an I-frame every 15 frames, no B-frames;
  - `IPB`: the same with 3 B-frames.

  The I-frames of `IP` and `IPB` are at frames 0, 15, 30, 45. Rates: CRF 23, 28, 33 and 2 Mbit/s (maxrate 2 Mbit/s,
  buffer 4 Mbit). The frame type is read from the stream (ffprobe `pict_type`), and the quality per frame (PSNR against
  the source, RGB 8 bit) and the coded size (`pkt_size`) are recorded.
- **Detectors** (GPU, bfloat16 autocast, fingerprint float32):
  - round3 Base (Update 1 v1.0, `9840160f…d0e6`) and round3 Small (Update 1 v1.1, `b715a163…d48b`): image logit,
    blurred at the file's r95 / r98 threshold ([data/thresholds.json](data/thresholds.json));
  - v9 (this repository, `10d6aa1e…0ba3`): the built-in M, blurred at M > 0.007882 (r95, from
    `reproduction/heldout_films.json`) or M > 0.
- In all 115,200 frames. A check run on one photo (not part of the result) confirmed the frame-type pattern (`I` only
  I; `IP` and `IPB` I-frames at 0, 15, 30, 45, `IBBBP…`), 48 decoded frames per clip and the pan alignment.

**Quality.** At the same CRF the GOP coding gives better frames: at CRF 23 the I-only frames have 37.1 dB, the `IP` /
`IPB` frames 40.0–40.4 dB; at 2 Mbit/s 31.5 dB against 39.6–40.3 dB. At the same CRF an I-frame inside a GOP is
about three times the size of an I-only frame (at 2 Mbit/s 14 times). The static P/B frames are nearly empty (60–340 bytes): copies of their I-frame. Full tables:
[tables.md](tables.md).

## 3. Deviations and additions

- **D1 (selection), 2026-10-28 23:25.** The protocol takes the first 100 suitable photos among the photo ladder's 300.
  Only 14 of the 300 fit (landscape, 1080 px high after the banner crop, at least 1488 px wide): one album's landscape
  photos are about 752 px high after its watermark-band crop. The selection therefore continues in the same hash
  order over the whole validation and test pool (3,981 photos): 228 fit, the first 100 are used. All 100 are from the
  other album. The crop and everything else are unchanged.
- **D2 (measure 4), 2026-10-29.** Measure 4 is not paired. The same source frame has a different PSNR in the two
  codings, so at equal quality the P/B frames are compared with the I-only frames that fall into the same 1 dB bin.
  Only bins with at least 30 frames on both sides count. The differences are weighted by the P/B count, with the
  bootstrap over the photos.
- **Addition (measure 3).** The protocol names the `IPB` clips. The analysis also reports the `IP` clips; those rows
  are marked as an addition.
- **Addition (descriptive).** How the false alarms spread over the photos (section 4.5) is not a measure of the
  protocol.

## 4. Results

### 4.1 False alarms by frame type (measure 1)

At r95, % of frames (round3 Base; Small and v9 and r98 in [tables.md](tables.md)):

| motion, rate | I only | IP: I | IP: P | IPB: I | IPB: P | IPB: B |
|---|---|---|---|---|---|---|
| static, CRF 23 | 0.0 | 0.8 | 1.7 | 2.0 | 2.5 | 2.6 |
| static, CRF 28 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| static, CRF 33 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| static, 2 Mbit/s | 0.0 | 3.8 | 4.8 | 3.5 | 4.3 | 4.5 |
| pan, CRF 23 | 0.0 | 1.0 | 2.2 | 2.0 | 3.3 | 2.7 |
| pan, CRF 28 | 0.0 | 0.0 | 3.1 | 0.0 | 2.5 | 2.5 |
| pan, CRF 33 | 0.0 | 0.0 | 7.1 | 0.0 | 4.5 | 4.4 |
| pan, 2 Mbit/s | 0.0 | 2.0 | 3.0 | 3.8 | 4.3 | 4.3 |

In all, at r95:

| | round3 Base | round3 Small | v9 |
|---|---|---|---|
| false alarms (of 115,200 frames) | 1,969 (1.7%) | 2,746 (2.4%) | 1,697 (1.5%) |

At r98 the Base gives 5,381 and the Small 5,955; v9 at M > 0 gives 12,522. Under I-only coding the highest share in any
cell is 0.2% (round3 Small, r98) and 0.8% for v9 at M > 0.

### 4.2 Paired difference against the I-only coding (measure 2, primary)

The same source frame (photo, clip, rate, frame number) in `IP` / `IPB` minus in `I` only, 95% interval over the
photos. round3 Base, logit:

| motion, rate | IP: I | IP: P | IPB: B |
|---|---|---|---|
| static, CRF 23 | +8.2 [+7.8, +8.6] | +12.6 [+12.0, +13.1] | +12.8 [+12.3, +13.4] |
| static, CRF 28 | +2.6 [+2.2, +3.0] | +4.7 [+4.1, +5.4] | +5.1 [+4.4, +5.7] |
| static, CRF 33 | −1.4 [−1.8, −1.0] | −1.6 [−2.1, −1.1] | −1.7 [−2.2, −1.2] |
| static, 2 Mbit/s | +7.6 [+6.9, +8.2] | +9.1 [+8.4, +9.8] | +9.1 [+8.4, +9.8] |
| pan, CRF 23 | +7.6 [+7.1, +8.0] | +13.5 [+13.0, +14.1] | +13.7 [+13.2, +14.3] |
| pan, CRF 28 | +2.3 [+1.9, +2.7] | +12.7 [+12.1, +13.2] | +12.4 [+11.8, +13.0] |
| pan, CRF 33 | −1.3 [−1.7, −1.0] | +11.2 [+10.5, +11.8] | +10.0 [+9.4, +10.7] |
| pan, 2 Mbit/s | +6.0 [+5.4, +6.7] | +10.0 [+9.4, +10.7] | +10.3 [+9.7, +10.9] |

The Small's differences are within 3 logits of these, with the same signs. The P-frames of `IPB` behave like
its B-frames.

In decisions, P-frames of `IP`, share called blurred at r95, percentage points:

| | round3 Base | round3 Small | v9 |
|---|---|---|---|
| pan, CRF 33 | +7.1 [+3.5, +11.3] | +5.3 [+2.5, +8.6] | +4.6 [+2.0, +7.9] |
| pan, CRF 28 | +3.1 [+0.8, +6.3] | +3.0 [+0.9, +5.3] | +2.2 [+0.7, +4.1] |
| static, 2 Mbit/s | +4.8 [+1.6, +8.9] | +8.0 [+3.5, +13.2] | +3.9 [+0.9, +7.9] |

At r98, pan CRF 33: round3 Base +15.1 [+9.5, +21.1], Small +12.2 [+7.6, +17.0]. On the static clips at CRF 28 and 33
no detector makes a single false alarm in any coding.

v9's M is exactly 0 on 75–100% of these frames in both codings (ties), so its score differences are tiny (+0.000 to
+0.003). Its decision differences at r95 are of a similar size as the round3 detectors'.

### 4.3 R4K-1 repeated (measure 3)

How often the in-clip I-frame (frames 15, 30, 45) is the lowest-scoring of the triple with its two neighbours; null:
the same positions coded I-only. `IPB` clips, the protocol's measure:

| | static (CRF 23 / 28 / 33 / 2 Mbit/s) | pan, CRF 23 | pan, CRF 28 | pan, CRF 33 | pan, 2 Mbit/s |
|---|---|---|---|---|---|
| round3 Base | 42.3 / 44.0 / 38.7 / 51.7% | 64.3% | 99.3% | 100.0% | 73.7% |
| round3 Small | 44.7 / 39.7 / 35.7 / 50.3% | 52.7% | 98.0% | 100.0% | 59.7% |
| null (I-only) | 0% (identical frames: ties) / 2 Mbit/s 31% | 28–30% | 33% | 32% | 30–32% |

Every round3 difference from the null has a 95% interval above 0 (from +20 points on the static clip at 2 Mbit/s to
+68 on the pan at CRF 33). v9 ties in 73–100% of the triples (M = 0), and the I-frame is the lowest in 13–17% of the pan
triples, against 0–0.3% under the null. The `IP` clips (an addition) give the same picture, with the Base at 80.7% on
the pan at CRF 23.

### 4.4 At equal quality (measure 4, D2)

P/B frames minus I-only frames in the same 1 dB PSNR bin, weighted by the P/B count:

| | static | pan |
|---|---|---|
| round3 Base (logit) | +3.6 [+2.8, +4.4] | +10.3 [+9.6, +11.1] |
| round3 Small (logit) | +4.0 [+3.2, +4.7] | +9.6 [+8.8, +10.3] |
| v9 (M) | +0.0008 [+0.0000, +0.0025] | +0.0018 [+0.0004, +0.0045] |

The matched bins hold 27,827 of the 35,200 static and 28,764 of the 35,200 pan P/B frames. Per bin (round3 Base):

- **pan:** in every bin from 30 to 40 dB the P/B frames sit at −7.5 to −13.4, the I-only frames at −17 to −21;
- **static:** up to 36 dB the two agree within about 2 logits. Above 37 dB the P/B frames rise (at 40 dB −10.4 against
  −17.0). These are the copies of the very clean GOP I-frames (4.2, column `IP: I`).

### 4.5 How the false alarms spread over the photos (descriptive)

| | false alarms | photos with any (of 100) | share of the 5 photos with most |
|---|---|---|---|
| round3 Base, r95 | 1,969 | 23 | 77% |
| round3 Small, r95 | 2,746 | 25 | 63% |
| v9, r95 | 1,697 | 22 | 80% |
| round3 Base, r98 | 5,381 | 47 | 46% |

The logit shift of P/B frames is large everywhere (4.2), but the decision threshold (about −2.8 for the Base at r95)
is crossed only by photos that already score near it. The median I-only logit is about −20.

## 5. Interpretation

- **By the rule fixed in advance the codec effect is confirmed** for all three detectors. At several rates the P/B
  frames have a higher score and a larger share called blurred than the same source frames coded I-only, and the
  paired intervals exclude 0.
- **The effect has two parts.**
  1. **Motion-compensated P/B frames look blurrier at any quality** (pan, about +10 logits in every PSNR bin).
  2. **Very clean GOP coding makes even its I-frames look blurrier.** At CRF 23 and 2 Mbit/s the GOP's I-frames get
     3 dB and 8 dB more than the I-only coding and score about 6–8 logits higher. At CRF 33, where they also gain about
     3 dB but from a lower start, they score about 1.4 lower. So the response to quality is not monotone.
- **What the detectors treat as sharp is, in part, how an I-only coding at the training's rates looks.** All training
  photos were coded as single frames. The pixel-level cause (which residual or block pattern the detectors respond to)
  is not tested here; R4K-1 section 4 describes pixel statistics on its two films.
- **For ranking, the shift dominates.** Sharp photos sit near −20 logits, so a shift of 10 changes the order of frames
  far more than it changes decisions. This is why Phase II (the sharpest frame per window) picks I-frames
  (R4K-1: 76× and 32×).

Consequences, as fixed in the protocol (4b of the training plan): training photos should also pass through multi-frame
GOP coding; film evaluations should be split by frame type; and on such material, Phase II as it stands would pick
I-frames.

## 6. Limitations

- **One coder** (x264, preset medium, the training's settings), one size (1440 × 1080), whole-pixel pans and static
  clips. Other coders, fractional motion, real motion blur and real film content are not tested.
- **One album.** By D1 all 100 photos come from one of the two photo albums.
- **The intervals describe the photos.** The bootstrap resamples these 100 photos. Most false alarms come from about
  a quarter of them, so the decision differences have wide intervals. The intervals do not describe other albums or
  other kinds of image.
- **Quality and frame type change together** in GOP coding: the GOP's I-frames are also coded at a different quantiser
  than the I-only ones. Measure 4 compares at equal PSNR, which is one measure of quality, and it is not paired (D2).
- **Not an R4K-1 replication on film:** this shows the mechanism exists under control. How much of R4K-1's film
  effect it explains is not measured.

## 7. Files and reproduction

| file | what |
|---|---|
| `PROTOCOL_hu.md`, `PROTOCOL.sha256` | the registered protocol (Hungarian, album names redacted) and the hashes |
| `PROTOCOL.md` | its English translation |
| `run.py` | the coding and scoring as run, with the publication edits in `MANIFEST.json`; needs the non-public photos |
| `analyse.py` | measures 1–4: `data/frames.csv.gz` → `data/results.json` |
| `report_tables.py` | `data/results.json` and `data/frames.csv.gz` → `tables.md` (standard library only) |
| `data/frames.csv.gz` | one row per frame (115,200): photo (`p000` … `p099`), motion, rate, structure, frame number, frame type, coded size, PSNR, and the three detectors' scores |
| `data/thresholds.json` | the round3 files' identity and thresholds |
| `data/results.json` | every measure, with intervals |
| `data/run.log` | the run's log (Hungarian messages) |
| `tables.md` | the full tables |
| `MANIFEST.json` | SHA-256 of every file, the publication edits |

```
pip install -r requirements-experiments.txt
python experiments/r4k2_gop_photo/analyse.py
python experiments/r4k2_gop_photo/report_tables.py
```

`analyse.py` needs numpy; `report_tables.py` only the standard library. From `data/frames.csv.gz` they reproduce
`data/results.json` and `tables.md` byte for byte. `run.py` additionally needs torch, both repositories' loaders and
weights, and ffmpeg / ffprobe on the PATH.

## 8. Log

Local time (CET), translated from the Hungarian log.

- 2026-10-28 23:20: the pipeline check on one photo (`run.py --check`, not part of the result): frame types as
  designed, 48 decoded frames, pan alignment correct.
- 2026-10-28 23:21: the protocol's hash recorded.
- 2026-10-28 23:25: D1 (section 3).
- 2026-10-29 01:20: the run complete (115,200 frames, 100 photos, all 2,400 clips). Analysis: `analyse.py`.
- 2026-10-29: D2 and the results written to the log; the measure-3 table of the first log entry showed the `IP`
  clips and was corrected to the protocol's `IPB` before publication (same conclusion).
