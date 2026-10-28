"""The random-design tier (scenario family R, pre-registration 3.9), added 2026-10-07 (b).
Run with `python tests/run_tests.py`."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dgp  # noqa: E402
import dgp_random as R  # noqa: E402
import main_groups  # noqa: E402
import run  # noqa: E402

REB = Path(__file__).resolve().parents[2]


def test_random_settings_in_range():
    t = R.load_table()
    assert t.pi.between(0.02, 0.40).all() and t.n.between(150, 4000).all() and t.p.between(4, 30).all()
    assert t.rho.between(0, 0.6).all() and t.auc_target.between(0.65, 0.90).all()
    for c in ("w_nl", "w_int", "w_link"):
        assert t[c].between(0, 1).all()
    assert t.frac_disc.between(0, 0.5).all()
    assert (t.n_eval == [max(1000, math.ceil(round(100 / x, 9))) for x in t.pi]).all()   # >= 100 events
    assert (t.solve_status == "ok").all()
    assert np.max(np.abs(t.realized_prevalence - t.pi)) < 1e-9
    assert np.max(np.abs(t.oracle_auc - t.auc_target)) < 1e-6
    assert list(t.scenario_id) == [1000 + d for d in t.design] and t.scenario.is_unique


def test_random_setting_is_reproducible_and_own_stream():
    a, b = R.draw_setting(7), R.draw_setting(7)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert R.draw_setting(8)["pi"] != a["pi"]
    s = R.get_design("R0007")
    assert s["family"] == "R" and s["n_recal"] == 0 and s["n_probe"] == 0
    assert R.entropy(s, 1, False) == [4242, 2, 7, 1] and R.entropy(s, 1, True) == [4242, 2, 7, 1, 777]


def test_random_replicate_shapes_and_fingerprint():
    r1 = dgp.make_replicate("R0005", 1, smoke=True)
    r2 = dgp.make_replicate(1005, 1, smoke=True)
    sc = r1.scenario
    assert r1.dev.X.shape == (sc["n"], sc["p"]) and len(r1.eval.y) == sc["n_eval"] and len(r1.recal.y) == 0
    assert r1.fingerprints == r2.fingerprints                 # by name or by id, the same data
    assert dgp.make_replicate("R0005", 1).fingerprints["all"] != r1.fingerprints["all"]   # main vs smoke
    # discrete columns hold codes; continuous ones do not
    for j, (c, k) in enumerate(zip(sc["columns"], sc["kinds"])):
        v = np.unique(r1.eval.X[:, j])
        if k == "bin":
            assert set(v) <= {0.0, 1.0}
        elif k == "cat":
            assert set(v) <= set(range(sc["cat_levels"][c]))
        else:
            assert len(v) > 100
    assert np.all((r1.eval.p_star > 0) & (r1.eval.p_star < 1))


def test_core_replicates_unchanged_by_the_hook():
    """The family-R branches must not move a single bit of the core design's data."""
    f = REB / "results" / "smoke_freeze" / "preds" / "S1_pi03_n200" / "0001.parquet"
    m = json.loads(pq.read_schema(f).metadata[b"tfmcal"])
    r = dgp.make_replicate("S1_pi03_n200", 1, smoke=True)
    assert m["fingerprints"] == r.fingerprints


def test_random_shape_built_from_core_generators():
    s = R.get_design("R0001")
    rng = np.random.default_rng(3)
    X = R.draw_covariates(s, 2000, rng)
    nonlin, inter = R.shape_parts(X, s)
    r = R.roles(s)
    ix = {c: j for j, c in enumerate(s["columns"])}
    x1, x3, x4 = X[:, ix[r["x1"]]], X[:, ix[r["x3"]]], X[:, ix[r["x4"]]]
    x7 = X[:, ix[r["x7"]]] if r["x7"] else (X[:, ix[r["x7_from"]]] > 0).astype(float)
    assert np.allclose(nonlin, (x1 ** 2 - 1) / math.sqrt(2) + (x3 ** 3 - 3 * x3) / math.sqrt(6), atol=1e-12)
    assert np.allclose(inter, 0.5 * x1 * x7 + 0.5 * x3 * x4, atol=1e-12)
    # the link mixture: 0 is the logit, 1 is the complementary log-log of S4
    eta = np.linspace(-6, 3, 50)
    assert np.allclose(R.inv_link(eta, 0.0), dgp.inv_link(eta, "logit"))
    assert np.allclose(R.inv_link(eta, 1.0), dgp.inv_link(eta, "cloglog"))
    assert np.all(np.diff(R.inv_link(eta, 0.4)) > 0)


def test_device_rules_match_the_measured_cells():
    cell = lambda n, ne, p: dict(n=n, n_eval=ne, p=p)  # noqa: E731
    # TabPFN-3.5: (n + N_eval) x p <= 80,000 on the GPU
    assert R.tabpfn_on_gpu(cell(500, 2000, 30)) and R.tabpfn_on_gpu(cell(4000, 2000, 8))
    assert not R.tabpfn_on_gpu(cell(1500, 5000, 20)) and not R.tabpfn_on_gpu(cell(1300, 2000, 30))
    # Mitra-v2: n^2 x p <= 8e7
    assert R.mitra_fits(cell(2000, 0, 20)) and R.mitra_fits(cell(3000, 0, 8)) and R.mitra_fits(cell(1500, 0, 30))
    for n, p in ((2500, 15), (3500, 8), (4000, 6), (3000, 30)):
        assert not R.mitra_fits(cell(n, 0, p))


def test_random_scenarios_parse_and_groups():
    assert run.parse_scenarios("R0001-R0003") == ["R0001", "R0002", "R0003"]
    assert run.parse_scenarios("R0009") == ["R0009"]
    assert not any(s.startswith("R") for s in run.parse_scenarios("all"))
    g = main_groups.random_groups()
    assert sorted(g["random_tabpfn_gpu"] + g["random_tabpfn_cpu"]) == sorted(g["random"])
    assert sorted(g["random_tabpfn_cpu_a"] + g["random_tabpfn_cpu_b"]) == sorted(g["random_tabpfn_cpu"])
    assert len(main_groups.groups()["all"]) == 58                 # the core groups are untouched
    nf = [s for s in g["random"] if s in set(g["random_mitra_not_run"])][0]
    assert run.not_run_reason("mitra", dgp.get_scenario(nf)).startswith("does not fit in 6 GB")
    assert run.not_run_reason("tabicl", dgp.get_scenario(nf)) is None
    assert run.not_run_reason("mitra", dgp.get_scenario("T2_pi10_n3000")).startswith("does not fit in 6 GB")
    assert run.not_run_reason("mitra", dgp.get_scenario("T2_pi10_n500")) is None
