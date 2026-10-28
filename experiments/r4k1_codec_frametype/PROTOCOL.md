# R4K-1: Does the coded frame type change blur scores? Two videos, two detectors

Protocol, fixed on 2026-10-27 before the analyses below were run. Results are reported in `REPORT.md`; every analysis
listed here is reported whatever its outcome. Analyses added later are marked as exploratory.

## 1. Origin and status

The question comes from an exploratory look at the same two videos (2026-10-27). A frame-by-frame comparison of two
blur detectors on a drama episode showed that the largest disagreements were isolated I-frames: frames the encoder had
coded without reference to other frames. A first paired check then found that, inside a shot, the I-frame was scored
sharper than its P/B neighbours although the frames look alike.

The hypotheses below were therefore formed on the same data they are tested on. This is a descriptive case study with a
fixed analysis plan, not an independent confirmation. Its claim is limited to these two videos and these two
detectors: it shows *whether* the coded frame type can matter and *how* the two detectors differ, not how often it
matters in general.

An earlier codec comparison of this project (published METHOD, "Why the video path") matched single-frame H.264 encodes
of photos to video statistics. A single encoded frame is intra-coded; that comparison could not see differences
between I-frames and the inter-coded P/B frames of a real group of pictures, as its own limitations state. This
experiment measures exactly that difference.

## 2. Material

| | Video A | Video B |
|---|---|---|
| Content | drama series episode (third-party broadcast-style encode) | studio web release |
| Container / codec | Matroska, H.264 High, 4:2:0 | MP4, H.264 High, 4:2:0 |
| Frames, rate | 62,674 at 24 fps (decoded) | 29,833 at 29.97 fps |
| Mean bitrate | 2.6 Mbit/s | 5.7 Mbit/s |
| In any detector's training data | no (checked by name and size against the training corpus) | no (checked by content SHA-256) |

The videos are not published and not named. Their SHA-256 digests are recorded in the results.

Detectors (both full-resolution, 16 px cell maps, the same edge-fingerprint input):
- **round3**: ConvNeXt-Base, image score = log-sum-exp over cell logits (a soft maximum: one strongly blurred region can
  decide). Weight file SHA-256 `9840160f…d0e6`.
- **v9**: the same trunk retrained on soft cell targets; image score = blur mass M = mean of max(0, cell logit) over all
  whole cells (an average: a small region contributes in proportion to its area). Weight file SHA-256 `10d6aa1e…0ba3`.

Sessions: four Frame Blur Studio analyses (each video with each detector, every frame, CUDA bfloat16), which store per
frame the score, the logit or blur mass, the full cell-logit map, and the decoded frame as a lossless PNG.

## 3. Per-frame variables

All frames of both videos, indexed in decode-output (presentation) order.

- **Coded frame type** (I / P / B), **coded size** (bytes) and **key-frame flag**, from `ffprobe -show_frames`.
  - Alignment check: equal frame counts; the presentation timestamps of ffprobe and of the session agree.
  - For 20 indices per video, the session PNG must match the frame that ffmpeg decodes at that index better than the
    frames at index ±1.
- **Shot boundaries**, from pixels only (never from frame types):
  - d(k): the mean absolute difference of 64 × 36 grey thumbnails between frames k − 1 and k, plus the
    16-bin hue-histogram L1 distance.
  - A cut at k when d(k) > 3 × the median of d over k − 6 … k + 6 (k excluded) and d(k) exceeds an absolute floor
    (thumbnail difference 6 grey levels or histogram distance 0.25).
  - Black frames (mean luminance < 8) never start a shot.
- **Position in the group of pictures:** frames since the last I-frame and until the next one. An I-frame is a
  **cut I-frame** when a shot starts at it, otherwise a **mid-shot I-frame**.
- **Pixel statistics**, on the active picture (letterbox bars found once per video from the median of 200 spread
  frames and excluded):
  - mean luminance;
  - mean |Laplacian| of the grey image;
  - the codec statistics of the published comparison, unchanged: flat_e3, edge_e3, flat_edge, zero, block8, hcorr on
    d = Y − K∗Y (K = [[3,10,3],[10,92,10],[3,10,3]] / 144, linear-light luminance, replicated borders, float64,
    16 px cells, the 30% / 90% residual cell quantiles).
- **Motion:** mean Farnebäck optical-flow magnitude between frames k − 1 and k on 480 × 270 grey frames, in
  full-resolution pixels.
- **Detector outputs** (both detectors):
  - image score (round3: sigmoid score and logit; v9: M);
  - share of cells with p ≥ 0.5 (round3: cells counted by its pooling; v9: all whole cells);
  - mean cell logit.

## 4. Questions and analyses

Inference unit: the shot. Confidence intervals are 95% percentile intervals of a cluster bootstrap over shots
(2,000 resamples, seed 20261027). Frame-level tests that treat frames as independent are not used.

**Q1 – Frame type, unconditional.** Score distributions by frame type, per video and detector. This is descriptive
only, because frame type is confounded with content (I-frames sit at shot starts).

**Q2 – Frame type within a shot (main analysis).**
- Mid-shot I-frames: each I-frame against its two same-shot neighbours (k − 1, k + 1).
- Cut I-frames: against the following frames k + 1 and k + 2.
- P-frames against their B neighbours, the same way.
- Effect measures, per detector:
  - the share of comparisons in which the I-frame scores sharper (ties counted separately: v9 has a block of M = 0);
  - the median difference in round3 logit and in v9 M;
  - the share of comparisons in which the decision flips at the operating points of Q6.

**Q3 – Distance from the I-frame.** Within shots, the median score and the share of frames scored sharp, as a function
of the frames since the last I-frame (0 … 12) and of frame type.

**Q4 – Motion control.**
- Q2 repeated on comparisons whose two frames differ little in motion: flow difference below the 25th percentile of all
  comparisons, and both flows below the shot median.
- The share of the Q2 effect that remains.

**Q5 – Coded size.**
- Within shots (shots with ≥ 10 B-frames), the Spearman correlation between coded size and score among B-frames.
- Raw, and partial on motion (rank residuals after removing a linear trend in flow rank).
- Summarised over shots, with bootstrap CIs.

**Q6 – Decision impact.**
- Share of frames accepted as sharp by frame type at the operating points:
  - round3 at r98 / r95 / r90 (scores 0.0164 / 0.0549 / 0.2324);
  - v9 at M = 0 / M < 0.05 / M < 0.25;
  - equal acceptance shares 10 / 25 / 50 / 75 / 90%.
- I-frame enrichment among accepted frames: the share of accepted frames that are I-frames, divided by the I-frame
  share.
- **Phase II simulation:**
  - The sharpest frame in each time window of 1, 5 and 20 s, with ties broken uniformly at random.
  - The I-frame share of the picks against the I-frame share expected without any frame-type effect (the mean over
    windows of the I-frame share of each window).

**Q7 – Pixel statistics by frame type.**
- The Q2 comparisons repeated for the pixel statistics of §3, to describe what differs in the decoded pixels.
- The association across comparisons between the difference in flat_e3 and the difference in score (Spearman).

**Q8 – Where in the picture (cells).**
- For mid-shot and cut I-frame comparisons: cell-wise difference in logit (P/B frame minus I-frame), both detectors.
- Grouped by the I-frame's cell residual level (deciles of the per-cell mean |d| within the frame: flat → textured).
- Reported per decile: the mean logit difference and the share of cells that cross from p < 0.5 to p ≥ 0.5.

**Q9 – Detectors compared.** All the above side by side, plus the agreement of the two detectors by frame type at equal
acceptance shares.

## 5. Hypotheses (from the exploratory look; stated before the analyses)

- **H1:** in Q2 both detectors score the I-frame sharper than its same-shot neighbours more often than the reverse.
- **H2:** the effect is larger in video A (2.6 Mbit/s) than in video B (5.7 Mbit/s). With two videos this is a
  consistency check, not a test of bitrate.
- **H3:** the P/B-minus-I logit increase is largest in the low-residual (flat) cell deciles.
- **H4:** P/B frames have higher flat_e3 than the I-frame of the same shot, and the difference is associated with the
  score difference.
- **H5:** the Q2 effect remains in the motion-matched subset (Q4).
- **H6:** I-frames are over-represented among accepted frames and among Phase II picks, in video A.
- **H7:** v9 is less affected than round3: smaller share of flips at equal acceptance.

## 6. Publication rules

- No frame, crop or cell-map image from either video is published; the videos are referred to as A and B.
- Published: this protocol, the analysis code, the aggregate results (JSON), figures drawn from them, the per-frame
  numeric table (no image content), and the input SHA-256 digests.
- Deviations from this protocol are listed in the report with their reason.
