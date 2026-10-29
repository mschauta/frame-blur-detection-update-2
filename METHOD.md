# Method — changes in Update 2

This file lists what differs from [Update 1](https://github.com/mschauta/frame-blur-detection-update-1/blob/main/METHOD.md),
which in turn lists its changes against the original
[METHOD.md](https://github.com/mschauta/frame-blur-detection/blob/main/METHOD.md); the section numbers follow the
original. Everything not mentioned here is as described there.

## 1. Task and scope

The task is the same: is a video frame usable, or is it blurred? The cost of a blurred frame accepted as sharp is
still the expensive error. What changed is the output. Update 2 gives a **blur mass M** per frame and a cell map, not a
thresholded class. The threshold is chosen when filtering (Section 8, Readout). Out of scope as before: captions,
overlaid graphics and animation.

## 2. Material

| | Update 1 | Update 2 |
|---|---|---|
| videos | 173 videos, all 1920 × 1080, H.264, 4:2:0 | the same 173 videos |
| frames for training | 589,663 selected, VLM-labelled frames | **every frame of the 167 training videos: 4,850,458**, with no thinning and no class balancing |
| held out | 3 validation + 3 test films | the same six films (160,142 frames). None of their frames was trained on, and neither was any release of them |
| left out | — | 2 of the 5,010,602 frames (overlay text, not visually processable) |
| photos | 48,797 originals (33,484 train / 7,656 validation / 7,657 test) | unchanged |

## 4. Labelling

**Update 2 uses no VLM label in training.** Each frame's target is the cell map of the Update 1 model
(`round3_base_p99_ep029.pt`), computed once over the whole corpus (Section 5). The VLM labels of the held-out films are
used only in the evaluation (Section 9).

## 5. Frame targets (replaces the v8 selection)

- **Soft cell targets.** Every 16 px cell of a frame gets the Update 1 model's blur probability as its target,
  σ(cell logit), so the strength of the evidence is kept. The logits were stored as 8-bit integers at 1/4-logit
  resolution, from −32 to +31.75.
- **The grid.** Frames are cut to 1920 × 1072, which is 67 whole cell rows. 1080 is 67.5 cells: with the half row,
  a vertical flip or a quarter turn would move the target grid 8 px off the model's cells.
- **Not taught:**
  - **The outer cell ring.** It was left out at the user's decision, after sharp photos' Update 1 maps showed a yellow
    border. On frames, Update 1 marks the border no more than its neighbours: on 30,000 random frames the outermost
    row or column has a mean p of 0.06–0.07 against 0.05–0.11 for the next cells.
  - **Measurement-vetoed blocks.** The non-learned motion measurement works in 160 px blocks. A block gets no target
    when its measured motion is at least 20 px but the Update 1 map is sharp there (every cell logit < 0). The motion
    is uncompensated region motion, p90, with at least 500 structured pixels; 20 px is about 6 px of blur at the
    typical exposure factor of 0.3. 11.5% of the training frames hold at least one such block (939,924 blocks in all).
    Nothing is relabelled: the cells are simply not taught.
- **Every other cell is taught, flat ones included.** A first run excluded flat cells (low fingerprint energy) and was
  stopped in its third epoch. Below that threshold were 76% of the inner frame cells and 90% of Update 1's strong blur
  cells (logit ≥ 2), because blur removes edges: a blurred cell looks flat. Large smooth moving surfaces such as skin
  would never have been taught.
- **No validity masks on frames.** Watermarks are visible on the frames, and their targets come from Update 1's
  unmasked maps.

The Update 1 maps of the training videos are in-sample for Update 1, since it was trained on labelled frames of those
videos. Its maps of the six held-out films are not.

## 6. Photos

- **The pool, the split and the coding chain are unchanged.** Blur is applied in linear light, followed by a
  single-frame H.264 encode at CRF 20 / 23 / 26 / 28.
- **Sharp:** 0, 1, 2, 3 px, as before.
- **Blurred:** 10, 12, 16, 20, 24, 30 px, instead of 16–30. The untrained gap shrinks from 4–15 px to 4–9 px.
  Update 1's photo ladder flips at about 9 px, so the stricter class matches where it already decides.
- **Grids:** 192 points (length × 12 directions × 4 CRF) for the sharp class and 288 for the blurred class, walked
  differently for every photo and epoch.
- **Mixing:** one photo sample per three frames, half sharp and half blurred. Every cell of a photo carries its
  class (0 or 1), except the outer ring, which is not taught here either. Half of the photo samples carry a frame
  watermark validity mask, as before.
- **What the photos do.** By about epoch 10 the photo term is almost always met: its median loss was 0.00017,
  against 0.071 for frames. From then on it acts as an anchor at the two ends of the scale. The transition between
  them (4–9 px on photos) is taught by the frame targets, not by the photos.
- **Every training photo is an I-frame**, a single intra-coded frame. Video frames are mostly P- and B-frames. The
  two-video experiment ([experiments/r4k1_codec_frametype/](experiments/r4k1_codec_frametype/)) shows what this does.
  Its controlled follow-up on sharp photos ([experiments/r4k2_gop_photo/](experiments/r4k2_gop_photo/)) confirms
  it.

## 7. Model input

Unchanged from Update 1: the p99-normalised signed `m09` fingerprint in float32. Frames and photos are cut to whole
16 px cells before the model; the loader does the same (`"crop": "cells"` in the weight file).

## 8. Model, loss and sampling

| | Update 2 |
|---|---|
| trunk | ConvNeXt-Base to stride 16, as in Update 1: the original ConvNeXt architecture as implemented in torchvision, not ConvNeXt V2. Initialised from torchvision's supervised ImageNet-1K weights `ConvNeXt_Base_Weights.IMAGENET1K_V1`, **not** from Update 1's weights, so the cell map is learned again rather than inherited |
| head | as before: one logit per 16 px cell (60.2 M parameters in all) |
| readout | **blur mass M** = the mean of max(0, cell logit) over the valid inner cells. It is fixed, not learned, and replaces the learnable log-sum-exp; the unused log-sum-exp parameter stays in the state dict. M = 0 means no inner cell says blur. Only cells that say blur (logit > 0) count, so a sharp background cannot average away a small blurred region |
| loss | soft binary cross-entropy per cell, over the taught cells only. There is no image-level loss: M is a readout of the cells |
| sampling | 20 epochs make **one pass** over the training frames (242,523 per epoch, each frame seen once) plus 80,841 photo samples per epoch. The order is a seeded permutation chain and resumes exactly |
| augmentation | a random quarter turn per batch and random horizontal / vertical flips per sample, exact on the cell grid |
| optimisation | AdamW, learning rate 1e-4 (trunk) / 1e-3 (head), weight decay 0.05, gradient clip 1.0, batch 8 × 2 accumulation. Schedule: 0.25-epoch warmup, a constant rate, then cosine decay over the last 20% of the steps to 1%. An exponential moving average (EMA) of the weights, decay 0.9998, is evaluated and published |
| length | 404,585 optimizer steps |
| checkpoint | the last epoch (19), EMA weights |

**Readout in use.** The weight file's built-in M leaves the outer ring out, as in training. A tool may let every cell
count. The author's desktop tool does that, and so did the two-video experiment. The ring then adds its own,
untaught evidence. `classify.py` uses the built-in readout.

## 9. Evaluation

- **Per epoch, during training (EMA, CUDA bfloat16).** Three sets:
  - 16,056 held-out frames, a fixed hash-chosen ~10% of the six held-out films' frames;
  - 7,899 training frames;
  - the 4,000-sample photo validation set.

  What is measured:
  - the fit to the Update 1 map, as soft cross-entropy over the taught cells, also relative to the targets' own
    entropy, which is the lowest fit any model can reach;
  - the Spearman correlation of M with Update 1's M and with the measured region motion;
  - the share of frames with M = 0;
  - the photo AUC.

  Source: [reproduction/training_eval.json](reproduction/training_eval.json).
- **VLM labels on the held-out films** (README, Results). After the checkpoint was fixed, every labelled
  frame of the six held-out films is scored once with the published file and loader: 114,687 frames, the frames of
  Update 1's `heldout_films.json`.
  - **Classes, as in Update 1:** sharp = `sss` and blurred = `bbb` + `bsb`. One of the six films is `ssb`-dominant
    (`sss / (sss + ssb)` = 0.40 < 0.46), and there `ssb` counts as sharp too. The evaluation code checks the counts and
    Update 1's AUCs against Update 1's published file.
  - **The comparison:** Update 1's corpus scores of the same frames.
  - **Intervals:** a cluster bootstrap over shots (2,000 resamples, shots drawn within each film), with one set of
    resamples for both models, so the difference between them is paired.
  - **Recall points:** Update 2 has no stored threshold. Its r90 / r95 / r98 points are the M thresholds that catch
    90 / 95 / 98% of the blurred frames of the three validation films. They are then applied to the test films.
    A point inside the M = 0 block could not be reached; none of the three falls there.

  Source: [reproduction/heldout_films.json](reproduction/heldout_films.json).
- **Photo ladder.** The ladder of Update 1, unchanged: the same 300 unseen photos, each with one fixed direction,
  L = 0 … 30 px, through H.264 CRF 23. Scored with the published file and loader. Source:
  [reproduction/photo_ladder_epoch19.json](reproduction/photo_ladder_epoch19.json).
- **Two videos outside the training data, every frame.** This is a separate experiment with its own fixed protocol:
  [experiments/r4k1_codec_frametype/](experiments/r4k1_codec_frametype/).
- **Sharp photos under three GOP structures and four rates**, every frame scored by Update 1 (Base and Small) and
  Update 2. This is also a separate experiment with its own fixed protocol:
  [experiments/r4k2_gop_photo/](experiments/r4k2_gop_photo/).
- **Not done yet:** scoring the whole corpus, the two-release agreement and Update 1's scene-consistency check. They
  need a full pass over the corpus on the GPU.

## 10. Training environment

Same as Update 1: NVIDIA GeForce RTX 5090 (32 GB), Windows 11, Python 3.13, PyTorch 2.14 with CUDA 13.0, FFmpeg 7.1.1
(x264). Training ran from 22 to 27 October 2026, 113.7 hours of wall-clock time (4.7 days), including the per-epoch
evaluation. It was resumed once, 17 minutes after the start, with the same code.

## 12. Data availability

Unchanged: the training data is not published.
