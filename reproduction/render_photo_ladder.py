"""Render figures/photo_ladder.svg from photo_ladder_epoch19.json and photo_ladder_update1_epoch29.json (standard
library only).

    python reproduction/render_photo_ladder.py

Colour = model (Update 1 blue, Update 2 orange: the first two slots of a palette validated for colour-vision
deficiency, light and dark, as in the two-video experiment). Lines are the main operating point, bands the range of
the other two; every mark carries a <title> tooltip with its numbers.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
v2 = json.loads((HERE / "photo_ladder_epoch19.json").read_text(encoding="utf-8"))
v1 = json.loads((HERE / "photo_ladder_update1_epoch29.json").read_text(encoding="utf-8"))
W, H = 920, 400
STYLE = """<style>
svg { --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --grid:#e4e3df; --series-1:#2a78d6; --series-2:#eb6834; }
@media (prefers-color-scheme: dark) {
  svg { --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --grid:#3a3936; --series-1:#3987e5; --series-2:#d95926; }
}
text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; fill: var(--ink2); font-size: 12px; }
.title { fill: var(--ink); font-size: 15px; font-weight: 600; }
.panel { fill: var(--ink); font-size: 13px; font-weight: 600; }
.key { fill: var(--ink); }
.axis { stroke: var(--ink2); stroke-width: 1; }
.grid { stroke: var(--grid); stroke-width: 1; }
</style>"""
X_TICKS = (0, 3, 6, 10, 16, 20, 24, 30)


class Panel:
    def __init__(self, parts, x0, y0, w, h, ylim, yticks, title, ylabel, yfmt):
        self.p, self.x0, self.y0, self.w, self.h, self.yl = parts, x0, y0, w, h, ylim
        parts.append(f'<text class="panel" x="{x0}" y="{y0 - 12}">{title}</text>')
        for t in yticks:
            y = self.Y(t)
            parts.append(f'<line class="grid" x1="{x0}" x2="{x0 + w}" y1="{y:.1f}" y2="{y:.1f}"/>'
                         f'<text x="{x0 - 6}" y="{y + 4:.1f}" text-anchor="end">{yfmt(t)}</text>')
        for t in X_TICKS:
            parts.append(f'<text x="{self.X(t):.1f}" y="{y0 + h + 16}" text-anchor="middle">{t}</text>')
        parts.append(f'<line class="axis" x1="{x0}" x2="{x0 + w}" y1="{y0 + h}" y2="{y0 + h}"/>')
        parts.append(f'<text x="{x0 + w / 2}" y="{y0 + h + 34}" text-anchor="middle">added camera motion L (px)</text>')
        parts.append(f'<text transform="translate({x0 - 44},{y0 + h / 2}) rotate(-90)" text-anchor="middle">{ylabel}</text>')

    def X(self, v):
        return self.x0 + v / 30 * self.w

    def Y(self, v):
        v = min(max(v, self.yl[0]), self.yl[1])
        return self.y0 + self.h - (v - self.yl[0]) / (self.yl[1] - self.yl[0]) * self.h

    def series(self, xs, ys, lo, hi, var, tips):
        band = [f"{self.X(x):.1f},{self.Y(v):.1f}" for x, v in zip(xs, hi)] + \
               [f"{self.X(x):.1f},{self.Y(v):.1f}" for x, v in zip(reversed(xs), reversed(lo))]
        self.p.append(f'<polygon points="{" ".join(band)}" fill="var({var})" fill-opacity="0.16"/>')
        self.p.append(f'<polyline points="{" ".join(f"{self.X(x):.1f},{self.Y(y):.1f}" for x, y in zip(xs, ys))}" '
                      f'fill="none" stroke="var({var})" stroke-width="2"/>')
        for x, y, tip in zip(xs, ys, tips):
            self.p.append(f'<circle cx="{self.X(x):.1f}" cy="{self.Y(y):.1f}" r="4" fill="var({var})" '
                          f'stroke="var(--surface)" stroke-width="2"><title>{tip}</title></circle>')


def key(parts, x, y, var, text):
    parts.append(f'<circle cx="{x + 5}" cy="{y - 4}" r="5" fill="var({var})"/><text class="key" x="{x + 15}" y="{y}">{text}</text>')


parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
         f'aria-label="Photo ladder: Update 1 and Update 2">', STYLE, f'<rect width="{W}" height="{H}" fill="var(--surface)"/>',
         f'<text class="title" x="16" y="26">Photo ladder: {v2["photos"]} unseen photos, known camera motion, H.264 CRF 23</text>']
key(parts, 16, 52, "--series-1", "Update 1: threshold r95 (band r90 … r98)")
key(parts, 330, 52, "--series-2", "Update 2: M &gt; 0.05 (band M &gt; 0.25 … M &gt; 0)")
pct = lambda t: f"{t * 100:.0f}%"                                          # noqa: E731
a = Panel(parts, 70, 100, 380, 236, (0, 1), (0, 0.25, 0.5, 0.75, 1), "Called blurred", "share of the photos", pct)
r1 = v1["per_L"]
a.series([r["L"] for r in r1], [r["blur_r95"] for r in r1], [r["blur_r90"] for r in r1], [r["blur_r98"] for r in r1],
         "--series-1", [f"Update 1, {r['L']:g} px: r90 {pct(r['blur_r90'])}, r95 {pct(r['blur_r95'])}, r98 {pct(r['blur_r98'])}"
                        for r in r1])
r2 = v2["per_L"]
a.series([r["L"] for r in r2], [r["blurred_M > 0.05"] for r in r2], [r["blurred_M > 0.25"] for r in r2],
         [r["blurred_M > 0"] for r in r2], "--series-2",
         [f"Update 2, {r['L']:g} px: M &gt; 0 {pct(r['blurred_M > 0'])}, M &gt; 0.05 {pct(r['blurred_M > 0.05'])}, "
          f"M &gt; 0.25 {pct(r['blurred_M > 0.25'])}" for r in r2])
top = max(r["quartiles_m"][1] for r in r2)
step = 5 if top > 12 else 2
ticks = [step * i for i in range(int(top // step) + 2)]
b = Panel(parts, 540, 100, 360, 236, (0, ticks[-1]), ticks, "Update 2: blur mass M (median, quartiles)", "M",
          lambda t: f"{t:g}")
b.series([r["L"] for r in r2], [r["median_m"] for r in r2], [r["quartiles_m"][0] for r in r2],
         [r["quartiles_m"][1] for r in r2], "--series-2",
         [f"{r['L']:g} px: median M {r['median_m']:.3g}, quartiles {r['quartiles_m'][0]:.3g} … {r['quartiles_m'][1]:.3g}"
          for r in r2])
parts.append("</svg>")
out = HERE.parent / "figures" / "photo_ladder.svg"
out.parent.mkdir(exist_ok=True)
out.write_bytes("\n".join(parts).encode("utf-8"))
print(out)
