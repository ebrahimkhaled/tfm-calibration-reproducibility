# Builds preregistration.tex and preregistration.pdf from preregistration.md.
# Needs pandoc and xelatex on the PATH. Run from any folder.
#
# Pandoc sizes the columns of a wide pipe table by the length of the dashes in
# its separator line, so a temporary copy of the markdown gets separators that
# follow the longest cell of each column. The .md file itself is not touched.
import re
import subprocess
from pathlib import Path

here = Path(__file__).resolve().parent
src = here / "preregistration.md"
tmp = here / "_build_tmp.md"
tex = here / "preregistration.tex"

SEP = re.compile(r"^\|(\s*:?-+:?\s*\|)+\s*$")


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def resize_tables(lines):
    out = list(lines)
    i = 0
    while i < len(out):
        if SEP.match(out[i]) and i > 0 and out[i - 1].lstrip().startswith("|"):
            start = i - 1
            end = i + 1
            while end < len(out) and out[end].lstrip().startswith("|"):
                end += 1
            rows = [cells(out[k]) for k in range(start, end) if k != i]
            ncol = len(cells(out[i]))
            longest = [0] * ncol
            for r in rows:
                for j, c in enumerate(r[:ncol]):
                    longest[j] = max(longest[j], len(c))
            # square-root weights stop one long column from squeezing the short ones; a wide table
            # of short cells (the scenario appendix) is better served by plain length weights
            power = 1.0 if ncol > 7 else 0.5
            root = [max(w, 2) ** power for w in longest]
            dashes = [max(8, round(60 * r / sum(root))) for r in root]
            out[i] = "|" + "|".join("-" * d for d in dashes) + "|"
            i = end
        else:
            i += 1
    return out


text = src.read_text(encoding="utf-8").splitlines()
tmp.write_text("\n".join(resize_tables(text)) + "\n", encoding="utf-8")

subprocess.run([
    "pandoc", str(tmp), "-s", "-o", str(tex),
    "--from", "markdown",
    "-V", "geometry:a4paper,margin=2cm",
    "-V", "fontsize=10pt",
    "-V", "colorlinks=true",
    "-V", "linkcolor=black", "-V", "urlcolor=black",
    "--columns=60",
], check=True)
tmp.unlink()

# two passes so the cross-references settle
for _ in range(2):
    subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
                   cwd=here, check=True, stdout=subprocess.DEVNULL)

for ext in (".aux", ".log", ".out"):
    p = here / ("preregistration" + ext)
    if p.exists():
        p.unlink()
print("wrote", tex.name, "and preregistration.pdf")
