"""Scenario groups of the main run (schedule P4, pre-registration 9.3; RUN_MAIN.md).

    python code/main_groups.py <group> [--n small|3000]     prints a comma-separated scenario list
    python code/main_groups.py --table                       every group, for the pre-registration

Each scenario's TabPFN-3.5 fits run on one device only, so the groups below never overlap:
    tabpfn_gpu    core pi = 3 % (16) and T2 n = 500                        GPU process
    tabpfn_cpu_a  T2 n = 3000 (does not fit on the GPU), core pi = 30 % (16), T1 and T3 (4)  CPU worker A
    tabpfn_cpu_b  core pi = 10 % (16), T4a and T4b (4)                                      CPU worker B
The stress scenarios are split so that the three streams finish at about the same time (pilot
budget: worker A about 63-77 h, worker B 57-71 h, GPU about 79 h with the temperature arms).
Other groups:
    all, small (n <= 1000, R = 200 for the TFM arms), n3000 (R = 150)
    legacy        Real-TabPFN-2 (V2) on the 16 core pi = 10 % scenarios, R = 100
    temperature   tabpfn_t1 and tabicl_t1: S1-S4 at pi = 10 %, n in {200, 3000}, R = 100
    pi30          the scenarios that have the e200 validation subsample (review D13)
Random-design tier (family R, pre-registration 3.9; never part of 'all'):
    random                every design R0001..R{D}, one replicate each
    random_tabpfn_gpu     designs whose n x p fits TabPFN-3.5 on the 6 GB card   GPU process
    random_tabpfn_cpu     the rest                                                CPU workers
    random_tabpfn_cpu_a, random_tabpfn_cpu_b   the same designs split by turns between the two workers
    random_mitra_not_run  designs where Mitra-v2 is recorded as "does not fit in 6 GB"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dgp  # noqa: E402


def groups() -> dict:
    t = dgp.load_table()
    core = t[t.kind == "core"]
    name = lambda d: list(d.scenario)  # noqa: E731
    pi = lambda d, v: d[(d.pi - v).abs() < 1e-9]  # noqa: E731
    fam = lambda *f: name(t[t.family.isin(f)])  # noqa: E731
    g = {
        "all": name(t),
        "tabpfn_gpu": name(pi(core, 0.03)) + ["T2_pi10_n500"],
        "tabpfn_cpu_a": ["T2_pi10_n3000"] + name(pi(core, 0.30)) + fam("T1", "T3"),
        "tabpfn_cpu_b": name(pi(core, 0.10)) + fam("T4a", "T4b"),
        "legacy": name(pi(core, 0.10)),
        "temperature": [f"S{k}_pi10_n{n}" for k in (1, 2, 3, 4) for n in (200, 3000)],
        "pi30": name(pi(t, 0.30)),
    }
    tab = set(g["tabpfn_gpu"]) | set(g["tabpfn_cpu_a"]) | set(g["tabpfn_cpu_b"])
    assert len(tab) == len(t) == len(g["tabpfn_gpu"]) + len(g["tabpfn_cpu_a"]) + len(g["tabpfn_cpu_b"]), \
        "TabPFN-3.5 device groups must cover every scenario exactly once"
    return g


def random_groups() -> dict:
    """The random-design tier (scenario family R, pre-registration 3.9): every design once, and
    TabPFN-3.5 on the GPU or on a CPU worker by the n x p rule of dgp_random.tabpfn_on_gpu."""
    import dgp_random
    t = dgp_random.load_table()
    gpu = [s for s in t.scenario if dgp_random.tabpfn_on_gpu(dgp_random.get_design(s))]
    cpu = [s for s in t.scenario if s not in set(gpu)]
    g = {"random": list(t.scenario), "random_tabpfn_gpu": gpu, "random_tabpfn_cpu": cpu,
         # the two CPU workers split the CPU designs by turns, so that each gets large and small ones
         "random_tabpfn_cpu_a": cpu[0::2], "random_tabpfn_cpu_b": cpu[1::2],
         "random_mitra_not_run": [s for s in t.scenario if not dgp_random.mitra_fits(dgp_random.get_design(s))]}
    assert sorted(gpu + cpu) == sorted(t.scenario) and not set(gpu) & set(cpu), \
        "TabPFN-3.5 device groups of the random designs must cover every design exactly once"
    return g


def select(group: str, n: str | None) -> list:
    if group.startswith("random"):
        return random_groups()[group]
    t = dgp.load_table().set_index("scenario")
    out = groups()[group]
    if n == "small":
        out = [s for s in out if t.loc[s, "n"] <= 1000]
    elif n == "3000":
        out = [s for s in out if t.loc[s, "n"] == 3000]
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--table":
        for k, v in groups().items():
            print(f"{k} ({len(v)}): {', '.join(v)}")
        for k, v in random_groups().items():
            print(f"{k} ({len(v)}): {', '.join(v[:6])}{' ...' if len(v) > 6 else ''}")
        sys.exit(0)
    n = a[a.index("--n") + 1] if "--n" in a else None
    sel = select(a[0], n)
    if not sel:
        sys.exit(f"group {a[0]} with --n {n} is empty")
    print(",".join(sel))
