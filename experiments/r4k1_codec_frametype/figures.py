"""R4K-1 step 4: SVG figures from data/results_A.json and data/results_B.json (standard library only).

usage: python figures.py

Colour = detector (round3 blue, v9 orange: the first two slots of a palette validated for colour-vision deficiency,
light and dark); videos are separate panels, never colours. Every mark carries a <title> tooltip with its numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
DET = {"r3": ("round3 (log-sum-exp)", "--series-1"), "v9": ("v9 (blur mass)", "--series-2")}
VIDEO = {"A": "Video A · 2.6 Mbit/s", "B": "Video B · 5.7 Mbit/s"}
STYLE = """<style>
svg { --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --grid:#e4e3df; --series-1:#2a78d6; --series-2:#eb6834; }
@media (prefers-color-scheme: dark) {
  svg { --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --grid:#3a3936; --series-1:#3987e5; --series-2:#d95926; }
}
text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; fill: var(--ink2); font-size: 12px; }
.title { fill: var(--ink); font-size: 15px; font-weight: 600; }
.panel { fill: var(--ink); font-size: 13px; font-weight: 600; }
.axis { stroke: var(--ink2); stroke-width: 1; }
.grid { stroke: var(--grid); stroke-width: 1; }
.ref { stroke: var(--ink2); stroke-width: 1; stroke-dasharray: 4 3; }
</style>"""


def esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Svg:
    def __init__(self, width: int, height: int, title: str, subtitle: str = ""):
        self.w, self.h = width, height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
                      f'viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">', STYLE,
                      f'<rect width="{width}" height="{height}" fill="var(--surface)"/>',
                      f'<text class="title" x="16" y="26">{esc(title)}</text>']
        if subtitle:
            self.parts.append(f'<text x="16" y="46">{esc(subtitle)}</text>')

    def add(self, s: str) -> None:
        self.parts.append(s)

    def legend(self, x: int, y: int, keys=("r3", "v9")) -> None:
        for k in keys:
            name, var = DET[k]
            self.add(f'<circle cx="{x + 5}" cy="{y - 4}" r="5" fill="var({var})"/>'
                     f'<text x="{x + 15}" y="{y}" style="fill:var(--ink)">{esc(name)}</text>')
            x += 22 + 7 * len(name)

    def save(self, name: str) -> None:
        OUT.mkdir(exist_ok=True)
        (OUT / name).write_text("\n".join(self.parts + ["</svg>"]), encoding="utf-8")


class Panel:
    """A plotting area with linear x / y scales."""

    def __init__(self, svg: Svg, x0, y0, w, h, xlim, ylim, title="", xlabel="", ylabel="", xticks=(), yticks=(),
                 xfmt=str, yfmt=str):
        self.s, self.x0, self.y0, self.w, self.h, self.xl, self.yl = svg, x0, y0, w, h, xlim, ylim
        if title:
            svg.add(f'<text class="panel" x="{x0}" y="{y0 - 10}">{esc(title)}</text>')
        for t in yticks:
            y = self.Y(t)
            svg.add(f'<line class="grid" x1="{x0}" x2="{x0 + w}" y1="{y:.1f}" y2="{y:.1f}"/>'
                    f'<text x="{x0 - 6}" y="{y + 4:.1f}" text-anchor="end">{esc(yfmt(t))}</text>')
        for t in xticks:
            x = self.X(t)
            svg.add(f'<text x="{x:.1f}" y="{y0 + h + 16}" text-anchor="middle">{esc(xfmt(t))}</text>')
        svg.add(f'<line class="axis" x1="{x0}" x2="{x0 + w}" y1="{y0 + h}" y2="{y0 + h}"/>')
        if xlabel:
            svg.add(f'<text x="{x0 + w / 2}" y="{y0 + h + 34}" text-anchor="middle">{esc(xlabel)}</text>')
        if ylabel:
            svg.add(f'<text transform="translate({x0 - 44},{y0 + h / 2}) rotate(-90)" text-anchor="middle">{esc(ylabel)}</text>')

    def X(self, v):
        return self.x0 + (v - self.xl[0]) / (self.xl[1] - self.xl[0]) * self.w

    def Y(self, v):
        return self.y0 + self.h - (v - self.yl[0]) / (self.yl[1] - self.yl[0]) * self.h

    def hline(self, v):
        self.s.add(f'<line class="ref" x1="{self.x0}" x2="{self.x0 + self.w}" y1="{self.Y(v):.1f}" y2="{self.Y(v):.1f}"/>')

    def vline(self, v):
        self.s.add(f'<line class="ref" x1="{self.X(v):.1f}" x2="{self.X(v):.1f}" y1="{self.y0}" y2="{self.y0 + self.h}"/>')

    def series(self, xs, ys, lo, hi, var, tips):
        clip = lambda v: min(max(v, self.yl[0]), self.yl[1])  # noqa: E731
        lo, hi = [clip(v) for v in lo], [clip(v) for v in hi]
        band = [f"{self.X(x):.1f},{self.Y(v):.1f}" for x, v in zip(xs, hi)] + \
               [f"{self.X(x):.1f},{self.Y(v):.1f}" for x, v in zip(reversed(xs), reversed(lo))]
        self.s.add(f'<polygon points="{" ".join(band)}" fill="var({var})" fill-opacity="0.16"/>')
        self.s.add(f'<polyline points="{" ".join(f"{self.X(x):.1f},{self.Y(y):.1f}" for x, y in zip(xs, ys))}" '
                   f'fill="none" stroke="var({var})" stroke-width="2"/>')
        for x, y, tip in zip(xs, ys, tips):
            self.s.add(f'<circle cx="{self.X(x):.1f}" cy="{self.Y(y):.1f}" r="4" fill="var({var})" '
                       f'stroke="var(--surface)" stroke-width="2"><title>{esc(tip)}</title></circle>')

    def dot_h(self, value, lo, hi, y, var, tip, hollow=False):
        if hollow:
            self.s.add(f'<circle cx="{self.X(value):.1f}" cy="{y:.1f}" r="4" fill="var(--surface)" stroke="var({var})" '
                       f'stroke-width="2"><title>{esc(tip)}</title></circle>')
            return
        self.s.add(f'<line x1="{self.X(lo):.1f}" x2="{self.X(hi):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="var({var})" stroke-width="2"/>'
                   f'<circle cx="{self.X(value):.1f}" cy="{y:.1f}" r="5" fill="var({var})" stroke="var(--surface)" '
                   f'stroke-width="2"><title>{esc(tip)}</title></circle>')


def pct(v):
    return f"{v * 100:.0f}%"


def nice_ticks(lo, hi, count=5):
    """Round tick values covering [lo, hi]."""
    import math
    raw = (hi - lo) / max(1, count - 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=10 * mag)
    start = math.floor(lo / step) * step
    ticks, v = [], start
    while v <= hi + step * 1e-9:
        ticks.append(round(v, 10))
        v += step
    return ticks


def load(variant: str = "revised"):
    return {k: json.loads((HERE / "data" / f"results_{k}_{variant}.json").read_text(encoding="utf-8")) for k in ("A", "B")}


def fig_q2(R):
    """Main result: how often the I-frame (or P-frame) is scored sharper than its same-shot neighbour."""
    kinds = [("mid_I", "mid-shot I-frame vs its neighbours"), ("cut_I", "shot-start I-frame vs next 2 frames"),
             ("P_vs_B", "P-frame vs its B neighbours")]
    svg = Svg(860, 470, "Inside a shot: how often is the reference frame scored sharper than its neighbour?",
              "Share of same-shot comparisons, 95% intervals (shot bootstrap). Ties (equal scores) are neither.")
    svg.legend(16, 70)
    svg.add('<circle cx="491" cy="66" r="5" fill="var(--ink2)"/><text x="501" y="70">reference sharper</text>'
            '<circle cx="631" cy="66" r="4" fill="var(--surface)" stroke="var(--ink2)" stroke-width="2"/>'
            '<text x="641" y="70">neighbour sharper</text>')
    for col, video in enumerate(("A", "B")):
        p = Panel(svg, 260 + col * 300, 110, 260, 300, (0, 1), (0, 1), VIDEO[video], "share of comparisons",
                  xticks=(0, 0.25, 0.5, 0.75, 1), xfmt=pct)
        p.vline(0.5)
        for row, (kind, label) in enumerate(kinds):
            yc = 110 + 50 + row * 100
            if col == 0:
                svg.add(f'<text x="16" y="{yc + 4}" style="fill:var(--ink)">{esc(label)}</text>')
            q = R[video]["Q2"][kind]
            if not q.get("comparisons"):
                svg.add(f'<text x="{p.x0 + 8}" y="{yc + 4}">none: this encoder puts no I-frame at shot starts</text>')
                continue
            for j, det in enumerate(("r3", "v9")):
                e = q[det]["ref_sharper"]
                tip = (f"{VIDEO[video]} · {label} · {DET[det][0]}: reference sharper {pct(e['value'])} "
                       f"[{pct(e['ci95'][0])}–{pct(e['ci95'][1])}], tie {pct(q[det]['tie']['value'])}, "
                       f"neighbour sharper {pct(q[det]['other_sharper']['value'])}; {q['comparisons']} comparisons, {q['shots']} shots")
                o = q[det]["other_sharper"]
                p.dot_h(o["value"], 0, 0, yc - 12 + j * 24, DET[det][1], tip, hollow=True)
                p.dot_h(e["value"], e["ci95"][0], e["ci95"][1], yc - 12 + j * 24, DET[det][1], tip)
            svg.add(f'<text x="{p.x0 + p.w}" y="{yc + 34}" text-anchor="end" style="font-size:11px">'
                    f'{q["comparisons"]:,} comparisons · {q["shots"]} shots</text>')
    svg.save("fig1_same_shot_comparisons.svg")


def fig_q3(R):
    svg = Svg(860, 420, "Accepted as sharp, by distance from the last I-frame of the shot",
              "round3 at r95 (score < 0.0549), v9 at M < 0.05; 95% intervals (shot bootstrap). 0 = the I-frame itself.")
    svg.legend(16, 70)
    for col, video in enumerate(("A", "B")):
        q = R[video]["Q3"]
        xs = sorted(int(k) for k in q)
        p = Panel(svg, 80 + col * 400, 110, 330, 230, (0, 12), (0, 1), VIDEO[video], "frames since the last I-frame",
                  "accepted as sharp", xticks=range(0, 13, 2), yticks=(0, 0.25, 0.5, 0.75, 1), yfmt=pct)
        for det, key in (("r3", "r3_accepted_r95"), ("v9", "v9_accepted_M<0.05")):
            ys = [q[str(x)][key]["value"] for x in xs]
            lo = [q[str(x)][key]["ci95"][0] for x in xs]
            hi = [q[str(x)][key]["ci95"][1] for x in xs]
            tips = [f"{VIDEO[video]} · {DET[det][0]} · {x} frames after I: {pct(y)} accepted (n = {q[str(x)]['n']})"
                    for x, y in zip(xs, ys)]
            p.series(xs, ys, lo, hi, DET[det][1], tips)
    svg.save("fig2_distance_from_i_frame.svg")


def fig_q8(R):
    svg = Svg(860, 420, "Where the neighbour gains blur evidence over the I-frame",
              "Mean cell-logit difference (neighbour − I-frame) by the I-frame's cell residual decile; 95% intervals.")
    svg.legend(16, 70)
    for col, video in enumerate(("A", "B")):
        q = R[video]["Q8"]["all_I"]
        vals = [q[d][k]["mean_logit_difference"][s] for d in ("r3", "v9") for k in range(10) for s in ("value",)]
        lows = [q[d][k]["mean_logit_difference"]["ci95"][0] for d in ("r3", "v9") for k in range(10)]
        highs = [q[d][k]["mean_logit_difference"]["ci95"][1] for d in ("r3", "v9") for k in range(10)]
        ticks = nice_ticks(min(0, min(lows)), max(highs + vals))
        p = Panel(svg, 80 + col * 400, 110, 330, 230, (1, 10), (ticks[0], ticks[-1]), VIDEO[video],
                  "cell residual decile of the I-frame (1 = flattest)", "logit difference",
                  xticks=range(1, 11), yticks=ticks, yfmt=lambda v: f"{v:g}")
        p.hline(0)
        for det in ("r3", "v9"):
            rows = q[det]
            xs = list(range(1, 11))
            ys = [r["mean_logit_difference"]["value"] for r in rows]
            tips = [f"{VIDEO[video]} · {DET[det][0]} · decile {x}: {y:+.3f} logit, cells crossing to blur "
                    f"{pct(r['crossing_to_blur']['value'])}" for x, y, r in zip(xs, ys, rows)]
            p.series(xs, ys, [r["mean_logit_difference"]["ci95"][0] for r in rows],
                     [r["mean_logit_difference"]["ci95"][1] for r in rows], DET[det][1], tips)
    svg.save("fig3_cells_by_texture.svg")


def fig_q6(R):
    svg = Svg(860, 420, "I-frames among the frames accepted as sharp, at equal acceptance shares",
              "Enrichment = I-frame share of accepted frames ÷ I-frame share of all frames; 1 = no frame-type effect.")
    svg.legend(16, 70)
    shares = ["10%", "25%", "50%", "75%", "90%"]
    xs = [10, 25, 50, 75, 90]
    top = max(R[v]["Q6"]["equal_shares"][s][d]["enrichment"] or 0 for v in "AB" for s in shares for d in ("r3", "v9"))
    ylim = (0, max(2, top * 1.1))
    for col, video in enumerate(("A", "B")):
        q = R[video]["Q6"]["equal_shares"]
        ticks = [0, 1] + [v for v in (2, 4, 6, 8, 10) if v <= ylim[1]]
        p = Panel(svg, 80 + col * 400, 110, 330, 230, (0, 100), ylim, VIDEO[video], "frames accepted as sharp",
                  "I-frame enrichment", xticks=xs, xfmt=lambda v: f"{v}%", yticks=ticks, yfmt=lambda v: f"{v}×")
        p.hline(1)
        base = R[video]["Q6"]["i_frame_share"]
        for det in ("r3", "v9"):
            # A target inside v9's block of equal M = 0 cannot be met without splitting ties: no point there.
            keep = [(x, s) for x, s in zip(xs, shares) if q[s][det]["enrichment"] is not None
                    and q[s][det]["actual_share"] >= 0.5 * x / 100]
            ys = [q[s][det]["enrichment"] for _, s in keep]
            lo = [q[s][det]["i_share_of_accepted"]["ci95"][0] / base for _, s in keep]
            hi = [q[s][det]["i_share_of_accepted"]["ci95"][1] / base for _, s in keep]
            tips = [f"{VIDEO[video]} · {DET[det][0]} · {s} target (actual {pct(q[s][det]['actual_share'])}): "
                    f"{y:.2f}× ; accepted I {pct(q[s][det]['accepted_by_type']['I'])}, P {pct(q[s][det]['accepted_by_type']['P'])}, "
                    f"B {pct(q[s][det]['accepted_by_type']['B'])}" for (_, s), y in zip(keep, ys)]
            if keep:
                p.series([x for x, _ in keep], ys, lo, hi, DET[det][1], tips)
            missing = [s for x, s in zip(xs, shares) if (x, s) not in keep]
            if missing:
                svg.add(f'<text x="{p.x0}" y="{p.y0 + p.h + 52}" style="font-size:11px">v9: no point at {", ".join(missing)} '
                        f'(inside its block of equal M = 0)</text>')
    svg.save("fig4_i_frame_enrichment.svg")


def fig_phase2(R):
    svg = Svg(860, 360, "Sharpest frame per time window: how often is it an I-frame?",
              "Enrichment of I-frames among the picks over the I-frame share of the windows; 95% intervals (window bootstrap).")
    svg.legend(16, 70)
    widths = ["1s", "5s", "20s"]
    import math
    L = math.log10
    for col, video in enumerate(("A", "B")):
        p = Panel(svg, 130 + col * 370, 110, 300, 190, (L(0.8), L(150)), (0, 1), VIDEO[video],
                  "I-frame enrichment of the picks (log scale)", xticks=[L(v) for v in (1, 2, 5, 10, 20, 50, 100)],
                  xfmt=lambda v: f"{10 ** v:.0f}×")
        p.vline(0)
        for row, wkey in enumerate(widths):
            yc = 110 + 35 + row * 60
            if col == 0:
                svg.add(f'<text x="16" y="{yc + 4}" style="fill:var(--ink)">{wkey} windows</text>')
            for j, det in enumerate(("r3", "v9")):
                e = R[video]["Q6"]["phase2"][wkey][det]
                tip = (f"{VIDEO[video]} · {wkey} · {DET[det][0]}: picks are I-frames {pct(e['i_share_of_picks'])} vs "
                       f"{pct(e['i_share_expected'])} expected → {e['enrichment']:.2f}× ({e['windows']} windows)")
                lo, hi = e["enrichment_ci95_window_bootstrap"]
                p.dot_h(L(e["enrichment"]), L(lo), L(hi), yc - 10 + j * 20, DET[det][1], tip)
    svg.save("fig5_phase2_windows.svg")


def fig_q7(R):
    stats = [("flat_e3", "flat-area residual"), ("flat_edge", "flat / strong residual"), ("zero", "near-zero residual share"),
             ("block8", "8 px block pattern"), ("edge_e3", "strong-cell residual"), ("laplacian", "mean |Laplacian|")]
    svg = Svg(860, 400, "What differs in the decoded pixels: neighbour vs I-frame of the same shot",
              "Median relative difference (neighbour ÷ I-frame − 1), mid-shot and shot-start comparisons; 95% intervals.")
    for col, video in enumerate(("A", "B")):
        kinds = [k for k in ("mid_I", "cut_I") if R[video]["Q7"][k].get("comparisons")]
        vals = [R[video]["Q7"][k][s]["median_relative_difference"] for k in kinds for s, _ in stats]
        span = max(0.05, max(max(abs(v["ci95"][0]), abs(v["ci95"][1])) for v in vals) * 1.1)
        p = Panel(svg, 240 + col * 310, 90, 270, 270, (-span, span), (0, 1), VIDEO[video], "relative difference",
                  xticks=(-span, 0, span), xfmt=lambda v: f"{v * 100:+.0f}%")
        p.vline(0)
        for row, (stat, label) in enumerate(stats):
            yc = 90 + 25 + row * 42
            if col == 0:
                svg.add(f'<text x="16" y="{yc + 8}" style="fill:var(--ink)">{esc(label)}</text>')
            for j, (kind, shade) in enumerate((("mid_I", "--ink"), ("cut_I", "--ink2"))):
                if kind not in kinds:
                    continue
                e = R[video]["Q7"][kind][stat]["median_relative_difference"]
                tip = (f"{VIDEO[video]} · {label} · {'mid-shot' if kind == 'mid_I' else 'shot-start'} I: "
                       f"{e['value'] * 100:+.1f}% [{e['ci95'][0] * 100:+.1f}, {e['ci95'][1] * 100:+.1f}]; neighbour higher in "
                       f"{pct(R[video]['Q7'][kind][stat]['other_higher']['value'])}")
                p.dot_h(e["value"], e["ci95"][0], e["ci95"][1], yc + j * 14, shade, tip)
    svg.add('<circle cx="21" cy="382" r="5" fill="var(--ink)"/><text x="31" y="386">mid-shot I-frame</text>'
            '<circle cx="161" cy="382" r="5" fill="var(--ink2)"/><text x="171" y="386">shot-start I-frame</text>')
    svg.save("fig6_pixel_statistics.svg")


def main() -> None:
    R = load()
    for fn in (fig_q2, fig_q3, fig_q8, fig_q6, fig_phase2, fig_q7):
        fn(R)
    print("figures:", sorted(p.name for p in OUT.glob("*.svg")))


if __name__ == "__main__":
    main()
