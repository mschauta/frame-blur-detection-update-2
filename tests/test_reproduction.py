"""The evidence files are strict JSON, and the numbers quoted in README.md / METHOD.md are the ones in them."""
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = sorted((ROOT / "reproduction").glob("*.json"))
EXPERIMENT = ROOT / "experiments" / "r4k1_codec_frametype"


def strict(path: Path):
    def refuse(token):
        raise ValueError(f"{path.name}: {token} is not JSON")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=refuse)


@pytest.mark.parametrize("path", EVIDENCE, ids=lambda p: p.name)
def test_evidence_is_strict_json(path):
    strict(path)


def flat(path: Path) -> str:
    """The text with every run of whitespace (line wraps) as one space."""
    return " ".join(path.read_text(encoding="utf-8").split())


def test_readme_quotes_the_training_evaluation():
    readme = flat(ROOT / "README.md")
    ev = strict(ROOT / "reproduction" / "training_eval.json")
    last = ev["epochs"][-1]
    assert last["epoch"] == 19 and len(ev["epochs"]) == 20
    h, ph, ref = last["heldout_frames"], last["validation_photos"], ev["reference"]["heldout_frames"]
    assert ref["n"] == 16_056 and "16,056 frames of the six held-out films" in readme
    assert f"| {h['bce_vs_round3']:.4f}, of which {ref['target_entropy']:.4f} is the targets' own entropy |" in readme
    assert f"| {h['spearman_m_r3']:.3f} |" in readme
    assert f"| {h['spearman_region_motion']:.3f} | {ref['round3_spearman_region_motion']:.3f} |" in readme
    assert f"| {h['m_zero_share']:.1%} | {ref['round3_m_zero_share']:.1%} |" in readme
    assert f"| {ph['auc']:.3f} |" in readme
    assert f"| {ph['m_zero_share_sharp']:.2%} / {ph['m_median_blurred']:.1f} |" in readme
    assert f"Update 2 is at {h['fit_above_target_entropy']:.4f}" in readme
    ep = {e["epoch"]: e["heldout_frames"]["fit_above_target_entropy"] for e in ev["epochs"]}
    assert f"{ep[15]:.4f} → {ep[19]:.4f} ({(ep[19] - ep[15]) / ep[15]:.1%})".replace("-", "−") in readme
    assert f"{ep[5]:.4f} → {ep[15]:.4f} ({(ep[15] - ep[5]) / ep[5]:.0%})".replace("-", "−") in readme
    tr = {e["epoch"]: e for e in ev["epochs"]}[19]
    raw = tr["heldout_frames"]["bce_vs_round3"] - tr["train_eval_frames"]["bce_vs_round3"]
    net = tr["heldout_frames"]["fit_above_target_entropy"] - tr["train_eval_frames"]["fit_above_target_entropy"]
    assert f"{raw:.4f} raw, and only {net:.4f}" in readme
    assert f"at most about +0.5 logit ({ref['round3_strongest_inner_cell_at_most']['0.5']:.1%} of the" in readme


def test_readme_quotes_the_photo_ladder():
    readme = flat(ROOT / "README.md")
    lad = strict(ROOT / "reproduction" / "photo_ladder_epoch19.json")
    up1 = strict(ROOT / "reproduction" / "photo_ladder_update1_epoch29.json")
    assert lad["photos"] == up1["photos"] == 300 and lad["L_px"] == up1["L_px"]
    flip = lad["flip_L_50pct"]
    assert (f"(M > 0: {flip['M > 0']:.2f} px, M > 0.05: {flip['M > 0.05']:.2f} px, M > 0.25: "
            f"{flip['M > 0.25']:.2f} px)") in readme
    assert all(abs(v - 9.1) < 0.15 for v in up1["flip_L_50pct"].values()) and "about 9.1 px for Update 1" in readme
    count = {r["L"]: round(r["blurred_M > 0"] * 300) for r in lad["per_L"]}
    assert all(count[L] == 0 for L in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)) and count[3.0] == count[4.0] == 1
    weak = {r["L"]: round(r["blurred_M > 0.05"] * 300) for r in lad["per_L"]}
    assert weak[3.0] == weak[4.0] == 0
    assert f"At 5 px {count[5.0]} photos have M > 0, at 6 px {count[6.0]} ({count[6.0] / 300:.0%}), at 8 px {count[8.0]}" in readme
    m0 = lad["order"]["M"]["0_3_px"]
    assert f"{m0['tied']:.1%} of the pairs" in readme
    t = lad["order"]["mean_inner_logit"]
    assert f"{t['0_3_px']['ordered']:.1%} of those pairs ({t['0_3_px']['reversed']:.1%} reversed)" in readme
    assert f"on {t['strictly_increasing_0_1_2_3']:.0%} of the photos" in readme
    assert f"managed {up1['pair_order_accuracy_0_3_px']:.1%} and {up1['strictly_increasing_0_1_2_3']:.1%}" in readme
    assert (f"the two are {t['3_16_px']['ordered']:.1%} and {up1['pair_order_accuracy_3_16_px']:.1%}") in readme
    med = {r["L"]: r["median_m"] for r in lad["per_L"]}
    assert (f"median {med[8.0]:.1f} at 8 px, {med[10.0]:.1f} at 10 px, {med[16.0]:.1f} at 16–20 px, "
            f"{med[30.0]:.1f} at 30 px") in readme and round(med[20.0], 1) == round(med[16.0], 1)
    ring = {r["L"]: round(r["any_cell_positive_ring_included"] * 300) for r in lad["per_L"]}
    assert ring[0.5] - count[0.5] == 1 and ring[6.0] - count[6.0] == 12 and f"({ring[6.0]} instead of {count[6.0]})" in readme


def test_readme_quotes_the_heldout_evaluation():
    readme = flat(ROOT / "README.md")
    ev = strict(ROOT / "reproduction" / "heldout_films.json")
    groups = {g["group"]: g for g in ev["groups"]}
    val, test, both = groups["validation films (3)"], groups["test films (3)"], groups["validation + test (6 films)"]
    assert (both["n_sharp"], both["n_blur"]) == (71_010, 43_677) and "114,687 frames (71,010 sharp, 43,677 blurred)" in readme

    def signed(v):
        return f"{v:+.4f}".replace("-", "−")

    def diff(g):
        d = g["auc_difference_update2_M_minus_update1"]
        return f"{signed(d['value'])} [{signed(d['ci95'][0])}, {signed(d['ci95'][1])}]"

    for name, g in (("validation films (3)", val), ("test films (3)", test)):
        row = f"| {name} | {g['auc']['update2_M']:.3f} | {g['auc']['update1']:.3f} | {diff(g)} |"
        assert row in readme, row
    a, ci = both["auc"], both["auc_ci95"]
    assert (f"| all six | {a['update2_M']:.3f} [{ci['update2_M'][0]:.3f}, {ci['update2_M'][1]:.3f}] | "
            f"{a['update1']:.3f} [{ci['update1'][0]:.3f}, {ci['update1'][1]:.3f}] | {diff(both)} |") in readme
    per_film = [f["auc"]["update2_M"] - f["auc"]["update1"] for f in ev["films"]]
    lower = sorted(-d for d in per_film if d < 0)
    assert len(lower) == 5 and f"by {lower[0]:.4f} to {lower[-1]:.4f}, and higher on one, by {max(per_film):.4f}" in readme
    points = ev["update2_recall_points"]
    ops = test["operating_points"]
    for r in ("r90", "r95", "r98"):
        u2, u1 = ops[f"update2_{r}"], ops[f"update1_{r}"]
        assert (f"| {r} | M > {points[r]['threshold']:g} | {u2['sharp_kept']:.1%} / {u2['blur_caught']:.1%} | "
                f"{u1['sharp_kept']:.1%} / {u1['blur_caught']:.1%} |") in readme, r
        assert f"{r}: M > {points[r]['threshold']:g}" in readme
    m0 = ops["update2_M>0"]
    assert f"| M > 0 | | {m0['sharp_kept']:.1%} / {m0['blur_caught']:.1%} | |" in readme
    assert (f"same AUC as the built-in M ({a['update2_M_all_cells']:.4f} and {a['update2_M']:.4f})") in readme
    assert (f"mean inner cell logit reaches AUC {a['update2_mean_inner_logit']:.3f}, the strongest inner cell "
            f"{a['update2_strongest_inner_logit']:.3f}") in readme
    z = both["m_zero_share"]
    assert f"**M = 0** on {z['sharp']:.1%} of the sharp and {z['blurred']:.1%} of the blurred labelled frames" in readme


def test_method_quotes_the_counts():
    method = flat(ROOT / "METHOD.md")
    for number in ("4,850,458", "160,142", "242,523", "80,841", "404,585", "114,687", "939,924"):
        assert number in method, number


def test_experiment_manifest_matches_the_files():
    manifest = strict(EXPERIMENT / "MANIFEST.json")
    on_disk = {p.relative_to(EXPERIMENT).as_posix() for p in EXPERIMENT.rglob("*")
               if p.is_file() and p.name != "MANIFEST.json" and "__pycache__" not in p.parts}
    assert set(manifest["files"]) == on_disk
    for name, digest in manifest["files"].items():
        data = (EXPERIMENT / name).read_bytes()
        if name.endswith((".md", ".py", ".json", ".log", ".svg", ".sha256")):
            # git may check text files out with CRLF line ends (core.autocrlf); the hashes are of the LF files
            data = data.replace(b"\r\n", b"\n")
        assert hashlib.sha256(data).hexdigest() == digest, name


def test_experiment_protocol_is_the_registered_one():
    digest = hashlib.sha256((EXPERIMENT / "PROTOCOL.md").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert digest in (EXPERIMENT / "PROTOCOL.sha256").read_text(encoding="utf-8")
