"""Run every test_* function of the test modules without pytest (the project venv has none).

    python tests/run_tests.py [name-filter]
"""

import importlib
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))


def main() -> int:
    filt = sys.argv[1] if len(sys.argv) > 1 else ""
    failed = 0
    total = 0
    for mod_file in sorted(HERE.glob("test_*.py")):
        mod = importlib.import_module(mod_file.stem)
        for name in sorted(n for n in dir(mod) if n.startswith("test_")):
            if filt not in name:
                continue
            total += 1
            t0 = time.perf_counter()
            try:
                getattr(mod, name)()
                print(f"ok    {mod_file.stem}.{name} ({time.perf_counter() - t0:.1f}s)", flush=True)
            except Exception:  # noqa: BLE001
                failed += 1
                print(f"FAIL  {mod_file.stem}.{name}", flush=True)
                traceback.print_exc()
    print(f"{total - failed}/{total} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
