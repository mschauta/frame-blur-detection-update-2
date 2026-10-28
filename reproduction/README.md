# Evidence

Aggregate numbers behind the README. No images, frames or per-frame records of the non-public training data are
included.

| file | what |
|---|---|
| `training_eval.json` | The per-epoch evaluation of the run (epochs 0–19, EMA weights). Held-out frames, training frames and validation photos: the fit to Update 1's maps and the targets' own entropy, the correlations with Update 1's M and with the measured motion, the M = 0 shares, the photo AUC. Also the Update 1 reference values on the same frames |
| `photo_ladder_epoch19.json` | The photo ladder with the published checkpoint: share called blurred at M > 0 / 0.05 / 0.25, medians of M and of the cell logits per added blur length, the flip points, and the order of pairs of lengths for three readouts |
| `photo_ladder_update1_epoch29.json` | The same ladder for Update 1, copied unchanged from the Update 1 repository for the comparison |
| `heldout_films.json` | Every VLM-labelled frame of the six held-out films (114,687), scored once with the published checkpoint, against Update 1's scores of the same frames: AUC per film, per role and overall with paired shot-bootstrap intervals, the AUCs of four readouts (M, M over every cell, mean and strongest inner cell logit), the M = 0 shares, and the r90 / r95 / r98 thresholds from the validation films with both models' operating points. Film names withheld |
| `render_photo_ladder.py` | Redraws `figures/photo_ladder.svg` from the two ladder files (standard library only) |

The files are strict JSON. `tests/test_reproduction.py` checks that the numbers quoted in the README are the ones in
these files.
