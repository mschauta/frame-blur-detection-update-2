# R4K-2: the effect of the coded frame type on sharp etalon photos (controlled GOP test)

English translation of the registered protocol, [PROTOCOL_hu.md](PROTOCOL_hu.md). The Hungarian text was fixed on
2026-10-28, before any result existed. Where the two differ, the Hungarian text governs.

**One redaction.** The registered text names the two photo albums in one place (§ Photos). Both copies here replace the
names with a placeholder. The registered SHA-256 is of the unredacted text, so it cannot be checked against
`PROTOCOL_hu.md`; [PROTOCOL.sha256](PROTOCOL.sha256) records both hashes, and the placeholder is the only difference.

File names refer to the training code as run; here the scripts are `run.py` and `analyse.py`.

## Protocol (fixed on 2026-10-28, before any result existed)

**Background.** R4K-1 observed on two videos that the models score the I-frame inside a shot as sharper than its P/B
neighbours. That was an observation: the true sharpness was not known there. The 4b test of `round4_terv.md` repeats it
with an exact signal: on clips made from 0 px etalon photos every frame is sharp, so every blur decision is a false
alarm. Two additions from a ChatGPT conversation (2026-10-28) are adopted: the I-frame-only coding as a paired control,
and recording the quality per frame (the same CRF does not mean the same quality).

**Question.** Does the same sharp source frame get more blur decisions or a higher score merely because it was coded as
a P- or B-frame rather than an I-frame?

**Models** (published files, published loaders):
- round3 Base (`round3_base_p99_ep029.pt`) and round3 Small (`round3_small_p99_ep029.pt`), Update 1 loader;
- v9 Base (`v9_base_selfmap_v2_ep019.pt`), Update 2 loader, built-in M (inner cells).

**Photos.** In the order of the photo ladder's 300 etalon photos (validation and test part, [two albums, names
withheld], never trained on), the first 100 landscape photos 1620 px wide. Crop: 1440 × 1080.

**Clips** (two per photo, 48 frames, 30 fps):
- static: the same crop 48 times;
- pan: the crop moves 1 px to the right per frame (whole pixels: no resampling, every frame is 0 px).

**Coding** (with the training's chain: bt709, tv range, yuv420p, libx264 high, preset medium, 1 thread):
- three structures:
  - `I`: I-frames only (keyint 1);
  - `IP`: keyint 15, no B-frames;
  - `IPB`: keyint 15, 3 B-frames.
- Common: scenecut 0, closed GOP, so the I-frames are at frames 0, 15, 30 and 45, in the `IP` and the `IPB` clip
  alike.
- four rates: CRF 23 (the typical value of the photo ladder and the training), CRF 28, CRF 33, and a 2 Mbit/s mean
  bitrate (maxrate 2 Mbit/s, buffer 4 Mbit; at 1440 × 1080 this corresponds roughly to the 2.6 Mbit/s 1080p film).
- ffprobe reads the frame type from the coded stream (`pict_type`); the decoded frames come in display order, and
  their number is checked (48).
- Quality: per frame, the PSNR against the source (RGB, 8 bit) and the coded size (`pkt_size`).

In all, 100 photos × 2 clips × 3 structures × 4 rates × 48 frames = 115,200 frames, scored by all three models (GPU,
bfloat16 autocast, fingerprint float32).

### Measures

1. **False alarms by frame type.** The share called blurred by structure, rate, motion and frame type: the round3
   models at their own r95 (and r98) thresholds, v9 at M > 0.007882 (r95) and M > 0.
2. **Paired difference against the I-only control (primary).** The same source frame (photo, clip, rate, index) in the
   `IP` or `IPB` coding and in the I-only coding: the difference of the score (round3: logit, v9: M) and of the share
   called blurred, split by the frame type in `IP` / `IPB`. 95% interval from a bootstrap over the photos (2000
   resamples, seed 20261028).
3. **R4K-1 repeated.** The in-clip I-frames of the `IPB` clips (frames 15, 30, 45) with their two neighbours: how often
   the sharpest of the triple (the lowest score) is the I-frame. Null: the same positions in the I-only coding (there
   all three are I-frames). Ties counted separately.
4. **At equal quality.** The paired difference by PSNR bin (1 dB): if the effect of the P/B frames comes only from the
   lower quality, it vanishes at equal PSNR.

### Interpretation (in advance)

- **The codec effect is confirmed** if at one rate or more the P/B frames' share called blurred (or score) is larger
  than that of the same source frames in the I-only coding, and the paired interval does not contain 0. The size is
  reported per rate.
- If confirmed, the consequences per the 4b test of `round4_terv.md`: the photos should also pass through multi-frame
  coding in training; the evaluation on film should be split by frame type; on such coding Phase II would pick
  I-frames.
- If it is shown at no rate, the film phenomenon (R4K-1) is not the frame type by itself: the real film content
  (motion, noise, light) or differences between encoders remain as explanations.
- The static clip is the most favourable case for P/B frames (they copy almost everything from the reference); the pan
  also tests the motion estimation. Neither has real motion blur.

### Not included

- Real film material, other encoders (x265, hardware), other resolutions, fractional-pixel motion: a separate test if
  the result warrants it.

## Log

The log follows in [REPORT.md](REPORT.md).
