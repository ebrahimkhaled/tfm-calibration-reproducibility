"""Fills the {{table.md}} placeholders of PILOT_REPORT.template.md with the generated tables.

    python pilot_build_report.py

Tables come from results/pilot/tables/ (pilot_analyse.py, pilot_budget.py, pilot_report_tables.py),
so no number in the tables is typed by hand.
"""

import re
import sys
from pathlib import Path

REBUILD = Path(__file__).resolve().parent.parent
TAB = REBUILD / "results" / "pilot" / "tables"


def main() -> int:
    src = (REBUILD / "code" / "PILOT_REPORT.template.md").read_text(encoding="utf-8")

    def fill(m):
        return (TAB / m.group(1)).read_text(encoding="utf-8").strip()

    out = re.sub(r"\{\{([A-Za-z0-9_.]+)\}\}", fill, src)
    (REBUILD / "PILOT_REPORT.md").write_text(out, encoding="utf-8")
    print("wrote", REBUILD / "PILOT_REPORT.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
