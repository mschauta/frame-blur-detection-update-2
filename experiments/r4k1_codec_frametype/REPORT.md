# R4K-1: The coded frame type changes blur scores. Two videos, two detectors

**Summary.** Two full-resolution blur detectors scored every frame of two H.264 videos. Inside a shot, both detectors
score the I-frame, the frame coded without reference to other frames, as sharper than its P/B neighbours, although the
neighbours show the same picture.

- **Video A (2.6 Mbit/s; its I-frames mostly start shots):**
  - **Same-shot comparisons:** round3 scores the shot-start I-frame sharper than the next two frames of the shot in
    93.4% of comparisons [91.5, 95.3]. The median difference is 14.4 logits.
  - **Decision flips:** at round3's r95 operating point, the I-frame is accepted as sharp and its next frame is rejected
    in 74% of comparisons.
- **Video B (5.7 Mbit/s; fixed 90-frame GOP, every I-frame inside a shot):** round3 scores the I-frame sharper than its
  same-shot neighbours in 68.1% [64.5, 72.3], and the reverse holds in 31.9%.
- **Selecting the sharpest frame per 20 s window:** with round3, the pick is an I-frame 76× (A) and 32× (B) more often
  than the I-frame share predicts.
- **v9 is less affected in its decisions:** 41× and 1.7× in the same test.
  - This is mostly because v9 gives many frames exactly M = 0 and cannot rank them. In a 20 s window of video B, 253
    frames tie on average.
  - Its tie-free cell evidence shows almost the same enrichment as round3 (exploratory: 76× and 13×).
- **In the decoded pixels:**
  - In video A, the flat regions of the P/B neighbours carry 21–28% more fine residual than the I-frame, and that
    residual is whiter: its horizontal correlation is 92% lower.
  - In both videos the 8 px block pattern is weaker on the neighbours.
- **Not explained by motion:** the effect remains in comparisons with little motion between the two frames.
- **Not explained by coded size:** within a frame type, the coded size barely matters.

The coded frame type therefore matters for both detectors, and most strongly where frames are ranked or selected. This
is a case study of two videos with a fixed analysis plan. The hypotheses were formed on the same videos.

## 1. Question and status

Protocol: [PROTOCOL.md](PROTOCOL.md), fixed before the analyses (SHA-256 in `PROTOCOL.sha256`). Every analysis it lists
is reported below. Deviations are listed in section 3, and analyses added afterwards are marked *exploratory*.

The question arose when two detectors were compared frame by frame on video A. Their largest disagreements were
isolated I-frames, and a manual look found that P/B neighbours of an I-frame looked alike but scored differently.

An earlier codec comparison of this project matched single-frame (intra-only) H.264 encodes of photos to video
statistics, so it could not see this. The photo anchors used to train both detectors were encoded the same way: every
training photo is an I-frame.

## 2. Material and checks

| | Video A | Video B |
|---|---|---|
| content | drama series episode (third-party broadcast-style encode) | studio web release |
| container, codec | Matroska, H.264 High 4:2:0 | MP4, H.264 High 4:2:0 |
| frames, rate, length | 62,674 · 24 fps · 43.5 min | 29,833 · 29.97 fps · 16.6 min |
| mean bitrate | 2.59 Mbit/s | 5.71 Mbit/s |
| I / P / B frames | 760 / 11,656 / 50,258 | 332 / 3,101 / 26,400 |
| median coded size I / P / B | 45.1 / 25.4 / 5.3 kB | 117.8 / 29.3 / 10.8 kB |
| I-frame placement | at shot starts (604 of 760), and at least every 240 frames (10 s) | fixed every 90 frames, never at shot starts |
| letterbox | 60 px bars top and bottom | none |
| in either detector's training data | no | no (content SHA-256 checked) |

The detectors share the ConvNeXt-Base trunk, the edge-fingerprint input and the 16 px cell grid. They differ in
training and in how the cell map becomes a frame score:

- **round3** (weights `9840160f…d0e6`) uses a log-sum-exp over the cells, a soft maximum: one strongly blurred region
  can decide.
- **v9** (weights `10d6aa1e…0ba3`) uses the blur mass M, the mean of the positive cell logits over all whole cells: a
  region counts in proportion to its area. This is the desktop tool's readout, with the outer cell ring included. The
  published loader's built-in M leaves the ring out, as in training (repository README, "Weights").

Both ran in Frame Blur Studio, the author's desktop application (not published), on every frame, at native
resolution, in CUDA bfloat16.

Checks (`data/checks_*.json`):

- **Frame counts:** equal in ffprobe and in both sessions.
- **Timestamps:** ffprobe and the session agree to < 1 µs.
- **Identical input:** the two detectors' sessions hold byte-identical decoded frames (100 of 100 sampled per video).
- **Index alignment:** for 20 sampled indices per video, the session frame matches ffmpeg's decoded frame k better than
  frames k ± 1. This held in 20/20 (B) and 19/20 (A). The 20th probe of A fell on a static frame whose three frames are
  identical, so it cannot discriminate.

## 3. Methods in brief, and deviations

Per frame: coded type and size, shot (from pixels only), position in the group of pictures, pixel statistics on the
active picture, optical-flow motion, and both detectors' scores and cell summaries.

- **Pixel statistics:** the definitions of the published codec comparison, unchanged (flat_e3, edge_e3, flat_edge,
  zero, block8, hcorr on the m09 residual).
- **Main analysis:** pairs of frames inside one shot:
  - a mid-shot I-frame against its two neighbours;
  - a shot-start I-frame against the next two frames;
  - a P-frame against its two B neighbours.
- **Intervals:** 95% percentile intervals from a shot-level cluster bootstrap (2,000 resamples). One set of resamples
  serves all statistics of a video.

**Deviations from the protocol**, found in the validation of video A before the results were interpreted. The
protocol-as-written results are kept: table T2a shows them for the main analysis, and `data/results_*_protocol.json`
holds them in full. The main conclusions are the same under both.

- **D1, black frames.**
  - Protocol: mean luminance < 8. On video A this also caught 4,622 frames of dark but structured scenes (median
    |Laplacian| 2.9; true black frames: 0.4), which then left the comparisons.
  - Revised: mean luminance < 8 and mean |Laplacian| < 1.
- **D2, shots.**
  - The hue-histogram cut criterion fired on the unstable hue of dark frames, giving 438 shots of ≤ 3 frames on video A.
  - Revised: the hue criterion only when both frames have mean luminance ≥ 20, and no cut within 3 frames after a cut.
  - Result on video A: 1,064 instead of 1,491 shots, median length 28 instead of 9 frames. Video B changes little
    (92 instead of 102 shots).
- **D3, motion control (Q4).**
  - The protocol rule used each frame's flow against its predecessor. For a shot-start I-frame that flow measures the
    cut, which left one matched comparison.
  - Revised: the mean flow between the two compared frames, below its 25th percentile.

Implementation choices the protocol did not fix:

- Black frames are not candidates in the acceptance shares and in the Phase II simulation. They carry no blur
  information and would inflate the sharp share.
- Equal acceptance shares never split equal scores. v9 cannot reach 10% or 25% on video B, where 42.5% of its frames
  have M = 0.
- Phase II ties are shared evenly among the tied frames. Its intervals resample windows instead of shots.

An exploratory sensitivity check resamples 30 s time blocks instead of shots. It gives the same main intervals
(table T2a).

## 4. Results

### 4.1 Inside a shot, the I-frame scores sharper (Q2)

![Same-shot comparisons](figures/fig1_same_shot_comparisons.svg)

| | round3: I-frame sharper / neighbour sharper | v9: I-frame sharper / equal / neighbour sharper |
|---|---|---|
| A, shot-start I-frame vs next 2 frames (1,206 comparisons, 603 shots) | 93.4% / 6.6%; median +14.4 logits | 86.2% / 5.4% / 8.4% |
| A, mid-shot I-frame vs neighbours (292, 43) | 73.6% / 26.4%; median +1.2 logits | 57.5% / 16.1% / 26.4% |
| B, mid-shot I-frame vs neighbours (662, 83) | 68.1% / 31.9%; median +0.6 logits | 39.0% / 37.0% / 24.0% |

- **The effect decides frames on video A.** The shot-start I-frame is accepted as sharp and its neighbour rejected in
  74.1% of comparisons at round3 r95, and in 62.2% at v9 M < 0.05 (table T3).
- **On video B the flips are rare:** 6.3% and 3.8%. Most frames of B are accepted anyway (70–78% at these points).
- **P- vs B-frames:**
  - On video A, both detectors score the P-frame blurrier than its B neighbours: round3 says the B neighbour is sharper
    in 69.7% of comparisons, v9 in 72.3%. The difference is small (median 0.35 logit) and rarely crosses a threshold.
  - On video B the two types do not differ (52.7% / 47.3%).

### 4.2 Only the I-frame itself (Q3)

![Distance from the I-frame](figures/fig2_distance_from_i_frame.svg)

The effect is confined to the I-frame:
- **Video A:** round3 at r95 accepts 85.2% of I-frames but only 19.5–22.0% of the 1st to 12th frame after them; v9
  (M < 0.05) accepts 83.9% against 27.9–31.4%. There is no gradual drift across the group of pictures.
- **Video B:** 72.5% against 67.0–72.0% (round3).

### 4.3 What this does to decisions (Q6)

![I-frame enrichment at equal acceptance](figures/fig4_i_frame_enrichment.svg)

**Thresholds and shares.** At the operating points, I-frames are over-represented among accepted frames on video A: by
3.9× at round3 r95 and 2.9× at v9 M < 0.05. On video B there is no over-representation (1.0×; table T7). At equal
acceptance shares the enrichment grows as the share shrinks:

| enrichment of I-frames | 10% | 25% | 50% |
|---|---|---|---|
| video A, round3 / v9 | 7.5× / 6.8× | 3.5× / 3.3× | 1.9× / 1.8× |
| video B, round3 / v9 | 2.8× / – | 1.6× / – | 1.18× / 1.15× |

![Phase II](figures/fig5_phase2_windows.svg)

**Selecting the sharpest frame per window (Phase II).** This is where the frame type dominates:

| 20 s windows | picks that are I-frames | expected | enrichment |
|---|---|---|---|
| A, round3 | 91.6% | 1.21% | 76× [69, 83] |
| A, v9 | 49.2% | 1.21% | 41× [34, 48] |
| B, round3 | 36.0% | 1.11% | 32× [21, 45] |
| B, v9 | 1.9% | 1.11% | 1.7× [1.1, 2.5] |

*Exploratory: why v9 picks fewer I-frames.* v9 ranks no frame within its M = 0 block, so a window's minimum is often a
large tie that is shared out evenly: 47 frames on average in a 20 s window of A, 253 in B. Ranked instead by its
tie-free mean cell logit, v9's picks are I-frames 76× (A) and 13× (B) more often than expected, close to round3
(`data/exploratory_phase2_tiefree.json`). v9's smaller effect in decisions therefore comes mostly from its readout's
coarse resolution among sharp frames. Its cell evidence is not insensitive to the frame type.

### 4.4 Motion, coded size and frame placement (Q4, Q5)

**Motion (table T5).** Where little motion separates the two compared frames, the effect remains:
- **A, shot-start:** 302 comparisons; round3 96.7%, v9 85.8%.
- **B, mid-shot:** 166 comparisons; round3 78.9%. v9: 22.9% I-frame sharper against 12.7% neighbour sharper, with
  64.5% ties.
- **A, mid-shot:** the low-motion subset is 2 shots, so it cannot be judged.

**Coded size (table T6).** Within the B-frames of a shot, more coded bytes go with slightly more blur evidence. Most of
this is motion: the mean Spearman is +0.13 on A and +0.09 on B, and +0.06 and +0.02 once motion is removed. Within a
frame type, coded size is not what matters.

*Exploratory, placement.* On video B the P-frames are accepted far less often than the I- and B-frames (36% vs 73–74%
at round3 r95), yet inside a shot they do not differ from their B neighbours. B's encoder inserts P-frames where motion
is high: median flow 11.7 px against 2.4–2.8 px for B- and I-frames. Unconditional comparisons by frame type would
mistake this placement for a codec effect. That is why the main analysis compares frames of the same shot.

### 4.5 What differs in the decoded pixels (Q7)

![Pixel statistics](figures/fig6_pixel_statistics.svg)

Neighbours compared with the I-frame of their shot, median relative difference:

| | flat-region residual (flat_e3) | flat-region correlation (hcorr) | 8 px block pattern (block8) | near-zero residual share | mean \|Laplacian\| |
|---|---|---|---|---|---|
| A, shot-start | +21.0% [19.8, 22.2] | −92% | −7.7% | −2.9% | +8.9% |
| A, mid-shot | +27.5% [20.8, 40.8] | −62% | −2.5% | −0.2% | +0.3% |
| B, mid-shot | +1.0% [0.1, 2.0] | −16% | −14.0% | −0.5% | −1.1% |

**What differs.**
- **I-frame:** smoother, more correlated flat regions and a stronger 8 × 8 block pattern.
- **Inter-coded neighbours:** more, whiter fine residual in flat regions (video A) and a weaker block pattern (both
  videos).
- **Visually** the frames look alike at 100% zoom (inspected locally; no image is published).

**Association with the score difference.** Across mid-shot comparisons on A, the flat-residual difference goes with
the score difference (Spearman +0.66 for round3, +0.30 for v9). It does not on A's shot-start comparisons (≈ 0) or on
B (−0.11). The flat-region residual is therefore at most part of the mechanism.

### 4.6 Where in the picture (Q8)

![Cells by texture](figures/fig3_cells_by_texture.svg)

**Video B.** The neighbour gains blur evidence in flat cells. In the flattest decile of the I-frame's cells, round3's
cell logit rises by +1.01 [0.68, 1.37]; in the most textured, by +0.05. v9 rises by +0.46 in flat cells and falls
slightly in textured ones.

**Video A.** The shift is global: +13 to +15 logits (round3) and +8 to +10.5 (v9) in every decile. The shot-start
I-frame's map is uniformly far on the sharp side, so 20% (flattest) to 33% (textured) of the cells cross to p ≥ 0.5 on
the neighbour.

### 4.7 The two detectors (Q9)

- **Agreement:** at equal acceptance shares the detectors agree on 88–98% of frames of each type (table T12).
- **Same direction:** on the same comparisons the frame type moves both of them the same way.
- **Where they differ:**
  - v9's readout absorbs part of the effect into its M = 0 block, both in the same-shot comparisons (more ties) and in
    Phase II (section 4.3).
  - On video B, v9's cell evidence moves less in flat cells and the other way in textured cells (section 4.6).

## 5. Hypotheses

| | statement | verdict |
|---|---|---|
| H1 | both detectors score the I-frame sharper than its same-shot neighbours more often than the reverse | **supported** in both videos (intervals separated). On B, v9's margin is smaller and 37% of its comparisons are ties |
| H2 | larger on A (2.6 Mbit/s) than on B (5.7 Mbit/s) | **supported in size and in decisions, not attributable to bitrate**: the videos also differ in encoder settings, I-frame placement, content and frame rate |
| H3 | the increase is largest in flat cells | **supported on B; not on A**, where the shift is global |
| H4 | P/B neighbours have higher flat_e3, associated with the score difference | **partly**: +21–28% on A, +1% on B; associated only on A's mid-shot comparisons |
| H5 | the effect remains in motion-matched comparisons | **supported where testable** (A shot-start; B mid-shot for round3, weaker for v9); A mid-shot not testable |
| H6 | I-frames over-represented among accepted frames and Phase II picks on A | **supported**: 3.9× at round3 r95; 76× in 20 s Phase II windows. Also on B in Phase II (32× round3) |
| H7 | v9 less affected than round3 | **in decisions, yes; in its evidence, hardly**: the difference comes mostly from v9's M = 0 tie block (exploratory, section 4.3) |

## 6. Interpretation

Both detectors respond to how a frame was coded, and not only to the optical blur in it.

- **What the detectors see:** the I-frame, coded alone and with several times the bits, has cleaner flat regions. The
  inter-coded frames that follow carry different fine structure, which both detectors read as more blur evidence. The
  frames look alike to the eye.
- **Why the effect is so much stronger on A:** A's encoder spends its I-frames at shot starts and gives them about 8×
  the bits of a B-frame at a low bitrate. B's I-frames are periodic, mid-shot, and at more than twice the bitrate.
- **The likely origin:** the detectors' sharp photo anchors were all intra-coded single frames. The detectors never saw
  a sharp inter-coded frame with a known label.

## 7. Limitations

- Two videos; the hypotheses come from a look at the same data. The videos differ in more than bitrate, so no
  dose-response on bitrate can be claimed.
- **No ground truth** for which of two neighbouring frames is sharper. "Look alike" rests on local inspection, not on
  a measurement.
- **Shots come from a pixel heuristic** (validated by its statistics, not by annotation). On A the mid-shot I-frames
  fall in only 43 shots, and their intervals are wide.
- **Motion is a mean optical-flow proxy.** Local motion (a hand) can be missed.
- **Shot-start I-frames (A)** also differ in content position (the first frame after a cut). The mid-shot comparisons,
  the absence of drift after the I-frame (4.2), and video B, where no I-frame starts a shot, argue against content as
  the explanation, but do not exclude every content effect.

## 8. Consequences for this project

- **Phase II** (the sharpest frame per time window) must not rank frames of different coded types against each other
  without correction. On encodes like A it mostly returns I-frames.
- **Per-video thresholds and ratios** inherit the encoder's frame structure; low-bitrate films are the worst case.
- **Training:** photo anchors should also pass through a multi-frame encode, so that inter-coded sharp frames get a
  known label. A controlled test (etalon photos coded with a real group of pictures, scored by frame type) can confirm
  the mechanism directly.
- **Deblocking:** a restoration step (planned RGB-mesh deblocking) can use the I-frames of a shot as a reference.
- **Evaluation:** any detector evaluation on film material should split results by coded frame type.

## 9. Reproduction

Requires the two videos (not published; SHA-256 in `data/checks_*.json`), the four Frame Blur Studio sessions (not
published), Python 3.13 with numpy, scipy, opencv-python, Pillow, and ffmpeg / ffprobe 7.1. Run the commands in this
folder. Without the videos and sessions, `analyse.py`, `figures.py` and `report_tables.py` still rerun every analysis
from the per-frame data in `data/`.

```
python extract.py --label A --video <video A> --r3 <round3 session A> --v9 <v9 session A> --out data
python extract.py --label B --video <video B> --r3 <round3 session B> --v9 <v9 session B> --out data
python cells.py --label A --variant revised --r3 <round3 session A> --v9 <v9 session A>   # and --variant protocol, and B
python analyse.py --label A --variant revised                                              # and protocol, and B
python figures.py
python report_tables.py > tables.md
```

`data/frames_*.csv.gz` holds every per-frame number (no image content). `data/results_*_{revised,protocol}.json`
holds every result with its interval. `MANIFEST.json` lists the SHA-256 of every file of this folder.

## Appendix: full tables (generated by `report_tables.py` from the result files)

### T1. Material

| video | frames | fps | duration | Mbit/s | I / P / B frames | median coded kB I / P / B | share of bytes I / P / B | frames between I-frames, median (p10–p90) | I-frames at shot starts | shots (median length, frames) | black frames |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 62,674 | 24.000 | 43.5 min | 2.59 | 760 / 11,656 / 50,258 | 45.1 / 25.4 / 5.3 | 6% / 45% / 48% | 67 (29–152) | 604 of 760 | 1,064 (28) | 544 |
| B | 29,833 | 29.970 | 16.6 min | 5.71 | 332 / 3,101 / 26,400 | 117.8 / 29.3 / 10.8 | 7% / 17% / 76% | 90 (90–90) | 1 of 332 | 92 (218) | 35 |

### T2. Same-shot comparisons (Q2): share of comparisons in which the reference frame scores sharper

| video | comparison | comparisons / shots | round3: reference sharper | round3: neighbour sharper | round3: median logit difference (neighbour − reference) | v9: reference sharper | v9: equal | v9: neighbour sharper |
|---|---|---|---|---|---|---|---|---|
| A | mid-shot I-frame vs neighbours | 292 / 43 | 73.6% [59.4, 97.3] | 26.4% [2.7, 40.6] | +1.17 [+0.00, +13.68] | 57.5% [49.0, 83.6] | 16.1% | 26.4% [5.4, 40.8] |
| A | shot-start I-frame vs next 2 frames | 1,206 / 603 | 93.4% [91.5, 95.3] | 6.6% [4.7, 8.5] | +14.37 [+13.50, +15.02] | 86.2% [83.7, 88.9] | 5.4% | 8.4% [6.2, 10.4] |
| A | P-frame vs B neighbours | 19,777 / 906 | 30.2% [27.8, 32.9] | 69.7% [67.1, 72.2] | -0.35 [-0.42, -0.29] | 20.9% [19.4, 22.5] | 6.8% | 72.3% [66.6, 76.8] |
| B | mid-shot I-frame vs neighbours | 662 / 83 | 68.1% [64.5, 72.3] | 31.9% [27.7, 35.5] | +0.60 [+0.47, +0.82] | 39.0% [33.2, 44.8] | 37.0% | 24.0% [19.1, 28.2] |
| B | shot-start I-frame vs next 2 frames | 0 | – | – | – | – | – | – |
| B | P-frame vs B neighbours | 774 / 81 | 52.7% [48.4, 56.8] | 47.3% [43.2, 51.6] | +0.05 [-0.04, +0.12] | 34.4% [28.7, 39.3] | 34.8% | 30.9% [25.9, 35.5] |

### T2a. The same, under the protocol definitions and with 30 s time blocks as the resampling unit

| video | comparison | protocol: comparisons / shots | protocol: round3 reference sharper | protocol: v9 reference sharper | revised, 30 s blocks: round3 | revised, 30 s blocks: v9 |
|---|---|---|---|---|---|---|
| A | mid-shot I-frame vs neighbours | 108 / 22 | 93.5% [72.0, 99.2] | 49.1% [39.5, 82.5] | 73.6% [64.2, 87.7] | 57.5% [49.1, 68.3] |
| A | shot-start I-frame vs next 2 frames | 1,101 / 554 | 93.9% [91.9, 95.7] | 86.6% [83.7, 89.4] | 93.4% [91.5, 95.5] | 86.2% [83.4, 89.2] |
| A | P-frame vs B neighbours | 19,273 / 920 | 30.0% [27.6, 32.5] | 20.2% [18.8, 21.7] | 30.2% [27.9, 32.7] | 20.9% [19.1, 22.7] |
| B | mid-shot I-frame vs neighbours | 662 / 83 | 68.1% [64.5, 72.1] | 39.0% [33.3, 44.8] | 68.1% [63.6, 72.5] | 39.0% [32.0, 46.0] |
| B | P-frame vs B neighbours | 774 / 81 | 52.7% [48.6, 56.8] | 34.4% [28.6, 39.6] | 52.7% [48.3, 57.0] | 34.4% [28.0, 40.9] |

### T3. Decision flips inside the comparisons (Q2): reference accepted as sharp, neighbour not

| video | comparison | round3 r98 | round3 r95 | round3 r90 | v9 M = 0 | v9 M < 0.05 | v9 M < 0.25 |
|---|---|---|---|---|---|---|---|
| A | mid-shot I-frame vs neighbours | 26.7% [12.8, 76.8] | 25.0% [14.3, 75.0] | 22.3% [11.4, 72.9] | 25.3% [12.9, 57.2] | 19.2% [9.0, 65.9] | 18.2% [8.5, 62.0] |
| A | shot-start I-frame vs next 2 frames | 78.2% [75.0, 81.4] | 74.1% [70.7, 77.6] | 69.9% [66.3, 73.5] | 68.4% [64.8, 72.0] | 62.2% [58.5, 65.9] | 54.9% [51.1, 58.8] |
| A | P-frame vs B neighbours | 0.8% [0.4, 1.3] | 0.8% [0.6, 1.0] | 1.0% [0.8, 1.3] | 1.0% [0.5, 1.8] | 0.7% [0.5, 0.9] | 1.0% [0.7, 1.3] |
| B | mid-shot I-frame vs neighbours | 8.9% [5.8, 12.3] | 6.3% [4.1, 8.9] | 3.3% [1.9, 4.9] | 13.7% [9.5, 18.7] | 3.8% [2.2, 5.3] | 2.7% [1.5, 4.1] |
| B | P-frame vs B neighbours | 5.6% [3.2, 8.2] | 5.2% [2.8, 7.6] | 5.8% [3.7, 7.7] | 5.6% [3.3, 8.6] | 6.6% [4.1, 9.0] | 4.0% [2.1, 5.9] |

### T4. Distance from the I-frame (Q3): accepted as sharp at round3 r95 / v9 M < 0.05

| video | I-frames | round3: I-frame | v9: I-frame | round3: frames 1–12 after | v9: frames 1–12 after | round3 shot-centred logit, I-frame | … frames 1–12 after |
|---|---|---|---|---|---|---|---|
| A | 750 | 85.2% [80.3, 89.5] | 83.9% [78.0, 88.2] | 19.5–22.0% | 27.9–31.4% | -13.49 | -0.21 … -0.01 |
| B | 331 | 72.5% [66.9, 79.2] | 78.5% [73.3, 85.0] | 67.0–72.0% | 74.6–78.4% | -0.51 | -0.05 … +0.14 |

### T5. Motion control (Q4, revised rule D3): comparisons with little motion between the two frames

| video | comparison | comparisons / shots | motion below (px/frame) | round3: reference sharper | v9: reference sharper | round3: net effect kept | v9: net effect kept |
|---|---|---|---|---|---|---|---|
| A | mid-shot I-frame vs neighbours | 73 / 2 | 0.00 | 58.9% [57.9, 62.5] | 54.8% [50.9, 68.8] | 0.38 | 0.53 |
| A | shot-start I-frame vs next 2 frames | 302 / 160 | 1.46 | 96.7% [93.6, 99.3] | 85.8% [80.4, 90.8] | 1.07 | 1.07 |
| A | P-frame vs B neighbours | 4944 / 350 | 1.15 | 40.4% [34.3, 45.1] | 26.4% [22.4, 31.6] | 0.49 | 0.53 |
| B | mid-shot I-frame vs neighbours | 166 / 52 | 1.15 | 78.9% [70.8, 86.3] | 22.9% [14.0, 32.3] | 1.60 | 0.68 |
| B | P-frame vs B neighbours | 194 / 60 | 1.28 | 59.8% [50.0, 68.8] | 17.0% [8.5, 26.7] | 3.61 | -0.74 |

### T6. Coded size within shots (Q5): Spearman between coded bytes and blur score among B-frames

| video | shots | round3 | round3, motion removed | v9 | v9, motion removed |
|---|---|---|---|---|---|
| A | 648 | +0.133 [+0.112, +0.154] | +0.060 [+0.041, +0.078] | +0.158 [+0.138, +0.179] | +0.075 [+0.056, +0.093] |
| B | 88 | +0.085 [+0.041, +0.131] | +0.015 [-0.023, +0.053] | +0.108 [+0.070, +0.145] | +0.032 [+0.004, +0.060] |

### T7. Accepted as sharp by frame type at the operating points (Q1/Q6; black frames excluded)

| video | operating point | accepted (all) | accepted I-frames | accepted P-frames | accepted B-frames | I-frame enrichment |
|---|---|---|---|---|---|---|
| A | round3 r98 | 14.1% | 78.7% [66.3, 88.3] | 11.3% [5.5, 20.1] | 13.7% [7.1, 23.6] | 5.59× |
| A | round3 r95 | 22.0% | 85.2% [80.3, 89.5] | 16.3% [9.3, 26.8] | 22.4% [12.8, 33.6] | 3.86× |
| A | round3 r90 | 27.2% | 87.7% [82.8, 91.5] | 20.2% [13.0, 30.5] | 27.9% [17.3, 39.1] | 3.22× |
| A | v9 M=0 | 9.8% | 67.5% [57.4, 75.9] | 7.3% [3.2, 13.4] | 9.5% [4.6, 16.9] | 6.87× |
| A | v9 M<0.05 | 29.1% | 83.9% [78.0, 88.2] | 21.3% [14.0, 31.6] | 30.1% [19.4, 41.2] | 2.88× |
| A | v9 M<0.25 | 39.8% | 86.9% [82.7, 90.4] | 29.8% [22.7, 38.8] | 41.4% [32.3, 50.7] | 2.19× |
| B | round3 r98 | 61.0% | 64.4% [57.8, 71.8] | 24.5% [17.3, 35.8] | 65.2% [58.5, 73.1] | 1.05× |
| B | round3 r95 | 70.2% | 72.5% [66.9, 79.2] | 35.8% [25.7, 50.8] | 74.2% [68.2, 80.9] | 1.03× |
| B | round3 r90 | 76.6% | 76.7% [71.0, 83.5] | 46.4% [34.8, 62.3] | 80.1% [74.8, 86.0] | 1.00× |
| B | v9 M=0 | 42.5% | 50.8% [43.8, 59.4] | 12.9% [9.0, 19.7] | 45.8% [39.5, 53.5] | 1.20× |
| B | v9 M<0.05 | 77.9% | 78.5% [73.3, 85.0] | 48.7% [37.8, 63.9] | 81.3% [76.5, 86.7] | 1.01× |
| B | v9 M<0.25 | 85.2% | 85.5% [80.9, 90.4] | 58.0% [47.6, 72.2] | 88.4% [84.7, 92.1] | 1.00× |

### T8. Equal acceptance shares (Q6): I-frame enrichment and acceptance by type

| video | target | detector | actual | I accepted | P accepted | B accepted | I-frame enrichment |
|---|---|---|---|---|---|---|---|
| A | 10% | round3 | 10.0% | 75.2% | 7.8% | 9.5% | 7.52× |
| A | 10% | v9 | 10.0% | 67.6% | 7.4% | 9.7% | 6.76× |
| A | 25% | round3 | 25.0% | 86.8% | 18.3% | 25.6% | 3.47× |
| A | 25% | v9 | 25.0% | 81.7% | 18.2% | 25.7% | 3.27× |
| A | 50% | round3 | 50.0% | 93.6% | 39.5% | 51.8% | 1.87× |
| A | 50% | v9 | 50.0% | 90.0% | 38.5% | 52.1% | 1.80× |
| A | 75% | round3 | 75.0% | 95.1% | 64.3% | 77.2% | 1.27× |
| A | 75% | v9 | 75.0% | 92.8% | 62.3% | 77.7% | 1.24× |
| A | 90% | round3 | 90.0% | 96.1% | 82.3% | 91.7% | 1.07× |
| A | 90% | v9 | 90.0% | 94.0% | 81.4% | 91.9% | 1.04× |
| B | 10% | round3 | 10.0% | 27.8% | 2.3% | 10.7% | 2.78× |
| B | 10% | v9 | 0% (target inside the M = 0 block) | – | – | – | – |
| B | 25% | round3 | 25.0% | 39.3% | 6.3% | 27.0% | 1.57× |
| B | 25% | v9 | 0% (target inside the M = 0 block) | – | – | – | – |
| B | 50% | round3 | 50.0% | 58.9% | 15.8% | 53.9% | 1.18× |
| B | 50% | v9 | 50.0% | 57.7% | 17.3% | 53.7% | 1.15× |
| B | 75% | round3 | 75.0% | 76.1% | 44.0% | 78.6% | 1.02× |
| B | 75% | v9 | 75.0% | 76.1% | 44.8% | 78.5% | 1.02× |
| B | 90% | round3 | 90.0% | 89.7% | 70.1% | 92.3% | 1.00× |
| B | 90% | v9 | 90.0% | 90.0% | 66.8% | 92.7% | 1.00× |

### T9. Phase II simulation (Q6): the sharpest frame of each window

| video | window | detector | windows | picks that are I-frames | expected | enrichment |
|---|---|---|---|---|---|---|
| A | 1s | round3 | 2599 | 23.5% | 1.21% | 19.5× [18.8, 20.1] |
| A | 1s | v9 | 2599 | 18.8% | 1.21% | 15.6× [14.8, 16.4] |
| A | 5s | round3 | 522 | 76.8% | 1.21% | 63.6× [59.8, 67.7] |
| A | 5s | v9 | 522 | 54.0% | 1.21% | 44.7× [41.1, 48.7] |
| A | 20s | round3 | 131 | 91.6% | 1.21% | 75.8× [68.8, 83.3] |
| A | 20s | v9 | 131 | 49.2% | 1.21% | 40.7× [34.3, 47.7] |
| B | 1s | round3 | 995 | 11.0% | 1.11% | 9.9× [8.4, 11.4] |
| B | 1s | v9 | 995 | 3.2% | 1.11% | 2.9× [2.2, 3.7] |
| B | 5s | round3 | 199 | 29.1% | 1.11% | 26.2× [20.7, 32.4] |
| B | 5s | v9 | 199 | 2.3% | 1.11% | 2.1× [1.2, 3.2] |
| B | 20s | round3 | 50 | 36.0% | 1.11% | 32.4× [21.3, 44.9] |
| B | 20s | v9 | 50 | 1.9% | 1.11% | 1.7× [1.1, 2.5] |

### T10. Decoded pixels (Q7): median relative difference, neighbour vs reference frame

| video | comparison | flat_e3 | edge_e3 | flat_edge | zero | block8 | hcorr | laplacian |
|---|---|---|---|---|---|---|---|---|
| A | mid-shot I-frame vs neighbours | +27.5 [+20.8, +40.8]% | +0.0 [-3.8, +0.0]% | +29.0 [+22.1, +43.5]% | -0.2 [-3.4, -0.0]% | -2.5 [-8.9, -0.0]% | -61.7 [-113.3, -21.7]% | +0.3 [+0.0, +14.6]% |
| A | shot-start I-frame vs next 2 frames | +21.0 [+19.8, +22.2]% | -1.6 [-2.1, -0.8]% | +21.4 [+20.1, +22.6]% | -2.9 [-3.2, -2.8]% | -7.7 [-8.0, -7.3]% | -92.0 [-99.3, -86.4]% | +8.9 [+8.0, +9.6]% |
| A | P-frame vs B neighbours | +2.2 [+1.5, +2.9]% | -3.1 [-3.4, -2.8]% | +6.1 [+5.1, +7.0]% | +0.1 [+0.0, +0.2]% | -3.0 [-3.2, -2.9]% | -3.7 [-6.4, -0.9]% | -1.2 [-1.7, -0.7]% |
| B | mid-shot I-frame vs neighbours | +1.0 [+0.1, +2.0]% | -1.6 [-2.0, -1.2]% | +2.7 [+1.5, +3.7]% | -0.5 [-0.7, -0.2]% | -14.0 [-15.0, -12.9]% | -16.3 [-21.1, -12.2]% | -1.1 [-1.4, -0.5]% |
| B | P-frame vs B neighbours | +2.9 [+2.1, +3.6]% | -3.7 [-4.0, -3.4]% | +7.1 [+6.2, +8.1]% | +0.2 [+0.1, +0.5]% | -9.4 [-10.0, -8.6]% | -24.5 [-31.1, -18.6]% | -3.6 [-4.1, -2.9]% |

Association across comparisons between the flat_e3 difference and the score difference (Spearman):

| video | comparison | round3 logit | v9 M |
|---|---|---|---|
| A | mid-shot I-frame vs neighbours | +0.655 [+0.147, +0.751] | +0.304 [+0.033, +0.544] |
| A | shot-start I-frame vs next 2 frames | -0.022 [-0.105, +0.060] | +0.022 [-0.059, +0.105] |
| B | mid-shot I-frame vs neighbours | -0.109 [-0.202, -0.016] | -0.138 [-0.235, -0.051] |

### T11. Cells (Q8): neighbour minus I-frame, by the I-frame's cell residual decile (1 = flattest)

| video | detector | decile | mean logit difference | cells crossing to p ≥ 0.5 | cells crossing to p < 0.5 |
|---|---|---|---|---|---|
| A | round3 | 1 | +13.31 [+12.68, +13.94] | 20.4% [17.8, 22.9] | 0.6% [0.3, 0.9] |
| A | round3 | 3 | +14.49 [+13.77, +15.19] | 25.9% [22.8, 28.5] | 0.6% [0.3, 0.9] |
| A | round3 | 5 | +14.88 [+14.13, +15.59] | 28.5% [25.1, 31.3] | 0.6% [0.3, 1.0] |
| A | round3 | 7 | +15.00 [+14.23, +15.71] | 30.7% [27.1, 33.7] | 0.7% [0.4, 1.0] |
| A | round3 | 9 | +14.67 [+13.72, +15.54] | 32.7% [28.8, 36.1] | 0.6% [0.3, 0.8] |
| A | round3 | 10 | +12.96 [+11.03, +14.69] | 28.9% [24.1, 33.8] | 0.6% [0.3, 0.9] |
| A | v9 | 1 | +8.20 [+7.64, +8.80] | 17.2% [14.9, 19.5] | 3.4% [2.4, 4.5] |
| A | v9 | 3 | +9.39 [+8.82, +10.00] | 22.4% [19.6, 24.8] | 3.0% [2.2, 4.0] |
| A | v9 | 5 | +9.93 [+9.34, +10.56] | 24.9% [21.8, 27.6] | 2.7% [1.9, 3.7] |
| A | v9 | 7 | +10.31 [+9.70, +10.92] | 26.9% [23.6, 29.7] | 2.5% [1.8, 3.4] |
| A | v9 | 9 | +10.50 [+9.74, +11.21] | 28.9% [25.2, 32.1] | 2.1% [1.5, 2.9] |
| A | v9 | 10 | +9.50 [+8.01, +10.86] | 25.1% [20.8, 29.6] | 1.7% [1.1, 2.5] |
| B | round3 | 1 | +1.01 [+0.68, +1.37] | 1.6% [1.1, 2.2] | 1.7% [1.0, 2.4] |
| B | round3 | 3 | +0.83 [+0.54, +1.18] | 2.2% [1.6, 2.8] | 1.5% [1.0, 1.9] |
| B | round3 | 5 | +0.59 [+0.34, +0.86] | 2.1% [1.5, 2.6] | 1.7% [1.2, 2.3] |
| B | round3 | 7 | +0.18 [-0.09, +0.47] | 1.7% [1.2, 2.1] | 1.8% [1.3, 2.4] |
| B | round3 | 9 | +0.03 [-0.23, +0.32] | 1.4% [1.1, 1.8] | 2.2% [1.4, 2.9] |
| B | round3 | 10 | +0.05 [-0.22, +0.34] | 1.3% [1.0, 1.7] | 2.2% [1.5, 2.9] |
| B | v9 | 1 | +0.46 [+0.14, +0.81] | 1.4% [0.9, 2.0] | 1.3% [0.7, 1.8] |
| B | v9 | 3 | +0.06 [-0.24, +0.41] | 1.7% [1.1, 2.4] | 1.3% [0.8, 1.7] |
| B | v9 | 5 | -0.14 [-0.38, +0.13] | 1.7% [1.2, 2.2] | 1.6% [1.0, 2.2] |
| B | v9 | 7 | -0.34 [-0.59, -0.05] | 1.4% [1.0, 1.9] | 1.8% [1.1, 2.4] |
| B | v9 | 9 | -0.27 [-0.50, -0.00] | 1.2% [0.9, 1.6] | 1.9% [1.3, 2.6] |
| B | v9 | 10 | -0.14 [-0.34, +0.09] | 1.1% [0.8, 1.5] | 2.0% [1.3, 2.7] |

### T12. Detector agreement by frame type at equal acceptance shares (Q9)

| video | target share | I-frames | P-frames | B-frames |
|---|---|---|---|---|
| A | 10% | 87.6% | 96.6% | 95.6% |
| A | 25% | 91.7% | 97.5% | 96.3% |
| A | 50% | 94.3% | 94.8% | 95.2% |
| A | 75% | 96.9% | 89.9% | 91.0% |
| A | 90% | 96.3% | 90.7% | 94.1% |
| B | 10% | not comparable: v9 cannot reach the target without splitting its M = 0 block |  |  |
| B | 25% | not comparable: v9 cannot reach the target without splitting its M = 0 block |  |  |
| B | 50% | 91.5% | 94.7% | 91.1% |
| B | 75% | 97.0% | 95.6% | 96.7% |
| B | 90% | 97.3% | 95.5% | 97.9% |
