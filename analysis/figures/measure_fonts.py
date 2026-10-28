"""Measure every figure PDF: effective font sizes, stroke widths and page size.

    python analysis/figures/measure_fonts.py [files...]      (default: every PDF in figures/out)

The sizes are read from the PDF content streams themselves (no PDF library is installed): each
text run's size is the Tf size times the scale of its text matrix (Tm) and of the current
transformation (cm, with the q/Q stack). A stroke width is the w operator times the CTM scale.
Writes figures/out/font_report.csv and exits 1 if any non-watermark text is below 7 pt, any
stroke is below 1 pt, or a page is not 89 or 183 mm wide.
"""

from __future__ import annotations

import csv
import re
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _common as C  # noqa: E402

MIN_PT, MIN_LW = 7.0, 1.0
WATERMARK_PT = 14.0
WIDTHS_MM = (89.0, 183.0)
NUM = rb"[-+]?(?:\d+\.?\d*|\.\d+)"
TOKEN = re.compile(rb"/[^\s/\[\]()<>]+|\((?:\\.|[^\\)])*\)|<[0-9A-Fa-f\s]*>|\[|\]|" + NUM + rb"|[A-Za-z'\"*]+")


def streams(data: bytes):
    for m in re.finditer(rb"<<(.*?)>>\s*stream\r?\n", data, re.S):
        start = m.end()
        end = data.find(b"endstream", start)
        raw = data[start:end]
        if b"/Length1" in m.group(1) or b"/FontFile" in m.group(1):
            continue                      # an embedded font program, not page content
        if b"FlateDecode" in m.group(1):
            try:
                yield zlib.decompress(raw)
            except zlib.error:
                try:
                    yield zlib.decompressobj().decompress(raw)
                except zlib.error:
                    continue
        elif b"/Image" not in m.group(1):
            yield raw


def mul(a, b):
    return [a[0] * b[0] + a[1] * b[2], a[0] * b[1] + a[1] * b[3],
            a[2] * b[0] + a[3] * b[2], a[2] * b[1] + a[3] * b[3],
            a[4] * b[0] + a[5] * b[2] + b[4], a[4] * b[1] + a[5] * b[3] + b[5]]


def scale(m):
    return abs(m[0] * m[3] - m[1] * m[2]) ** 0.5


def measure(path: Path) -> dict:
    data = path.read_bytes()
    mb = re.search(rb"/MediaBox\s*\[\s*(" + NUM + rb")\s+(" + NUM + rb")\s+(" + NUM + rb")\s+(" + NUM + rb")", data)
    width_mm = (float(mb.group(3)) - float(mb.group(1))) / 72 * 25.4 if mb else float("nan")
    sizes, widths = [], []
    for s in streams(data):
        if b"Tf" not in s and b" w" not in s and b"\nw" not in s:
            continue
        ctm, stack, ops, tf, tm = [1, 0, 0, 1, 0, 0], [], [], 0.0, [1, 0, 0, 1, 0, 0]
        for tok in TOKEN.findall(s):
            if re.fullmatch(NUM, tok) or tok.startswith((b"/", b"(", b"<", b"[", b"]")):
                ops.append(tok)
                continue
            op = tok
            nums = [float(x) for x in ops if re.fullmatch(NUM, x)]
            if op == b"q":
                stack.append(list(ctm))
            elif op == b"Q":
                ctm = stack.pop() if stack else [1, 0, 0, 1, 0, 0]
            elif op == b"cm" and len(nums) >= 6:
                ctm = mul(nums[-6:], ctm)
            elif op == b"BT":
                tm = [1, 0, 0, 1, 0, 0]
            elif op == b"Tf" and nums:
                tf = nums[-1]
            elif op == b"Tm" and len(nums) >= 6:
                tm = nums[-6:]
            elif op in (b"Tj", b"TJ", b"'", b'"'):
                sizes.append(round(tf * scale(mul(tm, ctm)), 2))
            elif op == b"w" and nums:
                widths.append(round(nums[-1] * scale(ctm), 3))
            ops = []
    text = [x for x in sizes if abs(x - WATERMARK_PT) > 0.05]
    strokes = [w for w in widths if w > 0]
    return dict(file=path.name, width_mm=round(width_mm, 1), n_text=len(text),
                min_pt=min(text) if text else None, max_pt=max(text) if text else None,
                sizes=" ".join(f"{v:g}" for v in sorted(set(text))),
                watermark=any(abs(x - WATERMARK_PT) <= 0.05 for x in sizes),
                min_stroke_pt=min(strokes) if strokes else None)


def main(argv=None) -> int:
    files = [Path(f) for f in (argv or sys.argv[1:])] or sorted(C.FIG_OUT.glob("*.pdf"))
    rows = [measure(f) for f in files]
    bad = 0
    for r in rows:
        ok_font = r["min_pt"] is not None and r["min_pt"] >= MIN_PT - 1e-6
        ok_lw = r["min_stroke_pt"] is None or r["min_stroke_pt"] >= MIN_LW - 1e-6
        ok_w = any(abs(r["width_mm"] - w) < 0.6 for w in WIDTHS_MM)
        r["pass"] = ok_font and ok_lw and ok_w
        bad += not r["pass"]
        print(f"{'ok  ' if r['pass'] else 'FAIL'} {r['file']:<44} {r['width_mm']:6.1f} mm  text {r['min_pt']}-{r['max_pt']} pt "
              f"[{r['sizes']}]  min stroke {r['min_stroke_pt']} pt  watermark {r['watermark']}")
    out = C.assert_writable(C.FIG_OUT / "font_report.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["file"])
        w.writeheader()
        w.writerows(rows)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
