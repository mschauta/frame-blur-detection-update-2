"""R4K-1 step 5: the Markdown tables of REPORT.md, generated from data/results_*.json (standard library only).

usage: python report_tables.py > tables.md
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
KINDS = {"mid_I": "mid-shot I-frame vs neighbours", "cut_I": "shot-start I-frame vs next 2 frames",
         "P_vs_B": "P-frame vs B neighbours"}


def load(label, variant):
    return json.loads((HERE / "data" / f"results_{label}_{variant}.json").read_text(encoding="utf-8"))


def pct(e, digits=1):
    if not e or e.get("value") is None:
        return "–"
    lo, hi = e["ci95"]
    return f"{e['value'] * 100:.{digits}f}% [{lo * 100:.{digits}f}, {hi * 100:.{digits}f}]"


def num(e, digits=2, sign=False):
    if not e or e.get("value") is None:
        return "–"
    f = f"{{:{'+' if sign else ''}.{digits}f}}"
    return f"{f.format(e['value'])} [{f.format(e['ci95'][0])}, {f.format(e['ci95'][1])}]"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    R = {(v, l): load(l, v) for v in ("revised", "protocol") for l in "AB"}
    rv = {l: R[("revised", l)] for l in "AB"}
    print("### T1. Material\n")
    rows = []
    for l in "AB":
        d = rv[l]["descriptive"]
        rows.append([l, f"{d['frames']:,}", f"{d['fps']:.3f}", f"{d['duration_s'] / 60:.1f} min", f"{d['mean_bitrate_mbit_s']:.2f}",
                     " / ".join(f"{d['types'][t]:,}" for t in "IPB"),
                     " / ".join(f"{d['coded_kb_median'][t]:.1f}" for t in "IPB"),
                     " / ".join(f"{d['coded_bytes_share'][t] * 100:.0f}%" for t in "IPB"),
                     f"{d['i_frame_gap_frames']['median']:.0f} ({d['i_frame_gap_frames']['p10']:.0f}–{d['i_frame_gap_frames']['p90']:.0f})",
                     f"{d['i_frames_at_detected_cuts']} of {d['i_frames_total']}", f"{d['shots']:,} ({d['shot_length_median_frames']:.0f})",
                     f"{d['black_frames']:,}"])
    print(table(["video", "frames", "fps", "duration", "Mbit/s", "I / P / B frames", "median coded kB I / P / B",
                 "share of bytes I / P / B", "frames between I-frames, median (p10–p90)", "I-frames at shot starts",
                 "shots (median length, frames)", "black frames"], rows))

    print("\n### T2. Same-shot comparisons (Q2): share of comparisons in which the reference frame scores sharper\n")
    rows = []
    for l in "AB":
        for k, name in KINDS.items():
            q = rv[l]["Q2"][k]
            if not q.get("comparisons"):
                rows.append([l, name, "0", "–", "–", "–", "–", "–", "–"])
                continue
            rows.append([l, name, f"{q['comparisons']:,} / {q['shots']}",
                         pct(q["r3"]["ref_sharper"]), pct(q["r3"]["other_sharper"]),
                         num(q["r3"]["median_difference_other_minus_ref"], 2, True),
                         pct(q["v9"]["ref_sharper"]), f"{q['v9']['tie']['value'] * 100:.1f}%", pct(q["v9"]["other_sharper"])])
    print(table(["video", "comparison", "comparisons / shots", "round3: reference sharper", "round3: neighbour sharper",
                 "round3: median logit difference (neighbour − reference)", "v9: reference sharper", "v9: equal",
                 "v9: neighbour sharper"], rows))

    print("\n### T2a. The same, under the protocol definitions and with 30 s time blocks as the resampling unit\n")
    rows = []
    for l in "AB":
        for k, name in KINDS.items():
            p, tb = R[("protocol", l)]["Q2"][k], rv[l]["Q2_time_block_clusters_exploratory"][k]
            if not p.get("comparisons"):
                continue
            rows.append([l, name, f"{p['comparisons']:,} / {p['shots']}", pct(p["r3"]["ref_sharper"]), pct(p["v9"]["ref_sharper"]),
                         pct(tb["r3"]["ref_sharper"]), pct(tb["v9"]["ref_sharper"])])
    print(table(["video", "comparison", "protocol: comparisons / shots", "protocol: round3 reference sharper",
                 "protocol: v9 reference sharper", "revised, 30 s blocks: round3", "revised, 30 s blocks: v9"], rows))

    print("\n### T3. Decision flips inside the comparisons (Q2): reference accepted as sharp, neighbour not\n")
    rows = []
    for l in "AB":
        for k, name in KINDS.items():
            q = rv[l]["Q2"][k]
            if not q.get("comparisons"):
                continue
            rows.append([l, name] + [pct(q["r3"]["decision_flips"][p]["ref_sharp_other_blurred"]) for p in ("r98", "r95", "r90")]
                        + [pct(q["v9"]["decision_flips"][p]["ref_sharp_other_blurred"]) for p in ("M=0", "M<0.05", "M<0.25")])
    print(table(["video", "comparison", "round3 r98", "round3 r95", "round3 r90", "v9 M = 0", "v9 M < 0.05", "v9 M < 0.25"], rows))

    print("\n### T4. Distance from the I-frame (Q3): accepted as sharp at round3 r95 / v9 M < 0.05\n")
    rows = []
    for l in "AB":
        q = rv[l]["Q3"]
        later = [q[str(p)] for p in range(1, 13) if str(p) in q]
        rows.append([l, f"{q['0']['n']}", pct(q["0"]["r3_accepted_r95"]), pct(q["0"]["v9_accepted_M<0.05"]),
                     f"{min(e['r3_accepted_r95']['value'] for e in later) * 100:.1f}–{max(e['r3_accepted_r95']['value'] for e in later) * 100:.1f}%",
                     f"{min(e['v9_accepted_M<0.05']['value'] for e in later) * 100:.1f}–{max(e['v9_accepted_M<0.05']['value'] for e in later) * 100:.1f}%",
                     f"{q['0']['r3_shot_centred_logit_median']:+.2f}",
                     f"{min(e['r3_shot_centred_logit_median'] for e in later):+.2f} … {max(e['r3_shot_centred_logit_median'] for e in later):+.2f}"])
    print(table(["video", "I-frames", "round3: I-frame", "v9: I-frame", "round3: frames 1–12 after", "v9: frames 1–12 after",
                 "round3 shot-centred logit, I-frame", "… frames 1–12 after"], rows))

    print("\n### T5. Motion control (Q4, revised rule D3): comparisons with little motion between the two frames\n")
    rows = []
    for l in "AB":
        q = rv[l]["Q4"]
        for k, name in KINDS.items():
            e = q.get(k, {})
            if not e.get("matched"):
                continue
            m = e["matched"]
            rows.append([l, name, f"{m['comparisons']} / {m['shots']}", f"{e['flow_p25_px']:.2f}", pct(m["r3"]["ref_sharper"]),
                         pct(m["v9"]["ref_sharper"]), f"{e['net_effect_remaining']['r3']:.2f}" if e["net_effect_remaining"]["r3"] is not None else "–",
                         f"{e['net_effect_remaining']['v9']:.2f}" if e["net_effect_remaining"]["v9"] is not None else "–"])
    print(table(["video", "comparison", "comparisons / shots", "motion below (px/frame)", "round3: reference sharper",
                 "v9: reference sharper", "round3: net effect kept", "v9: net effect kept"], rows))

    print("\n### T6. Coded size within shots (Q5): Spearman between coded bytes and blur score among B-frames\n")
    rows = []
    for l in "AB":
        q = rv[l]["Q5"]
        rows.append([l, q["shots"], num(q["r3"]["mean_rho"], 3, True), num(q["r3_partial"]["mean_rho"], 3, True),
                     num(q["v9"]["mean_rho"], 3, True), num(q["v9_partial"]["mean_rho"], 3, True)])
    print(table(["video", "shots", "round3", "round3, motion removed", "v9", "v9, motion removed"], rows))

    print("\n### T7. Accepted as sharp by frame type at the operating points (Q1/Q6; black frames excluded)\n")
    rows = []
    for l in "AB":
        q1, op = rv[l]["Q1"], rv[l]["Q6"]["operating_points"]
        for model, points in (("r3", ("r98", "r95", "r90")), ("v9", ("M=0", "M<0.05", "M<0.25"))):
            for p in points:
                rows.append([l, f"{'round3' if model == 'r3' else 'v9'} {p}", f"{op[f'{model}_{p}']['accepted_share'] * 100:.1f}%"]
                            + [pct(q1[t][f"{model}_accepted_{p}"]) for t in "IPB"] + [f"{op[f'{model}_{p}']['enrichment']:.2f}×"])
    print(table(["video", "operating point", "accepted (all)", "accepted I-frames", "accepted P-frames", "accepted B-frames",
                 "I-frame enrichment"], rows))

    print("\n### T8. Equal acceptance shares (Q6): I-frame enrichment and acceptance by type\n")
    rows = []
    for l in "AB":
        for s, e in rv[l]["Q6"]["equal_shares"].items():
            for model in ("r3", "v9"):
                m = e[model]
                if not m["actual_share"]:
                    rows.append([l, s, "round3" if model == "r3" else "v9", "0% (target inside the M = 0 block)", "–", "–", "–", "–"])
                    continue
                rows.append([l, s, "round3" if model == "r3" else "v9", f"{m['actual_share'] * 100:.1f}%"]
                            + [f"{m['accepted_by_type'][t] * 100:.1f}%" for t in "IPB"] + [f"{m['enrichment']:.2f}×"])
    print(table(["video", "target", "detector", "actual", "I accepted", "P accepted", "B accepted", "I-frame enrichment"], rows))

    print("\n### T9. Phase II simulation (Q6): the sharpest frame of each window\n")
    rows = []
    for l in "AB":
        for w, e in rv[l]["Q6"]["phase2"].items():
            for model in ("r3", "v9"):
                m = e[model]
                rows.append([l, w, "round3" if model == "r3" else "v9", m["windows"], f"{m['i_share_of_picks'] * 100:.1f}%",
                             f"{m['i_share_expected'] * 100:.2f}%",
                             f"{m['enrichment']:.1f}× [{m['enrichment_ci95_window_bootstrap'][0]:.1f}, {m['enrichment_ci95_window_bootstrap'][1]:.1f}]"])
    print(table(["video", "window", "detector", "windows", "picks that are I-frames", "expected", "enrichment"], rows))

    print("\n### T10. Decoded pixels (Q7): median relative difference, neighbour vs reference frame\n")
    stats = ("flat_e3", "edge_e3", "flat_edge", "zero", "block8", "hcorr", "laplacian")
    rows = []
    for l in "AB":
        for k, name in KINDS.items():
            q = rv[l]["Q7"][k]
            if not q.get("comparisons"):
                continue
            rows.append([l, name] + [num({kk: (v * 100 if v is not None else None) if kk == "value" else [x * 100 for x in v]
                                          for kk, v in q[s]["median_relative_difference"].items() if kk in ("value", "ci95")}, 1, True) + "%"
                                     for s in stats])
    print(table(["video", "comparison"] + list(stats), rows))
    rows = []
    for l in "AB":
        for k in ("mid_I", "cut_I"):
            q = rv[l]["Q7"][k]
            if not q.get("comparisons"):
                continue
            rows.append([l, KINDS[k], num(q["spearman_delta_flat_e3_vs_delta_r3"], 3, True), num(q["spearman_delta_flat_e3_vs_delta_v9"], 3, True)])
    print("\nAssociation across comparisons between the flat_e3 difference and the score difference (Spearman):\n")
    print(table(["video", "comparison", "round3 logit", "v9 M"], rows))

    print("\n### T11. Cells (Q8): neighbour minus I-frame, by the I-frame's cell residual decile (1 = flattest)\n")
    rows = []
    for l in "AB":
        q = rv[l]["Q8"]["all_I"]
        for model in ("r3", "v9"):
            for d in (0, 2, 4, 6, 8, 9):
                e = q[model][d]
                rows.append([l, "round3" if model == "r3" else "v9", d + 1, num(e["mean_logit_difference"], 2, True),
                             pct(e["crossing_to_blur"]), pct(e["crossing_to_sharp"])])
    print(table(["video", "detector", "decile", "mean logit difference", "cells crossing to p ≥ 0.5", "cells crossing to p < 0.5"], rows))

    print("\n### T12. Detector agreement by frame type at equal acceptance shares (Q9)\n")
    rows = []
    for l in "AB":
        for s, e in rv[l]["Q9"].items():
            if not rv[l]["Q6"]["equal_shares"][s]["v9"]["actual_share"]:
                rows.append([l, s, "not comparable: v9 cannot reach the target without splitting its M = 0 block", "", ""])
                continue
            rows.append([l, s] + [f"{e[t] * 100:.1f}%" for t in "IPB"])
    print(table(["video", "target share", "I-frames", "P-frames", "B-frames"], rows))


if __name__ == "__main__":
    main()
