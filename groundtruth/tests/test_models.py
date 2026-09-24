"""Model ladder, runtime parity, predict template, package + check round trip."""
from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth  # noqa: E402

from gtlab import metric, package  # noqa: E402
from gtlab import systems as S  # noqa: E402
from gtlab.check import check  # noqa: E402
from gtlab.models.ensemble import Ensemble, PerObs  # noqa: E402
from gtlab.models.l0 import L0a, L0b  # noqa: E402
from gtlab.models.l1 import L1  # noqa: E402
from gtlab.models.l2 import L2  # noqa: E402
from gtlab.runtime import infer as rt  # noqa: E402


@pytest.fixture(scope="module")
def data():
    spec, tr = synth.make_runs(6, 200, 0)
    _, va = synth.make_runs(2, 400, 1, split="val")
    sigma = metric.sigma_proxy(tr + va)
    return spec, tr, va, sigma


@pytest.fixture(scope="module")
def fitted(data):
    spec, tr, va, sigma = data
    l0a = L0a(spec, sigma=sigma).fit(tr)
    l0b = L0b(spec, sigma=sigma).fit(tr)
    l1 = L1(spec, sigma=sigma, K=2, max_nfev=15, delay_grid=(0, 2)).fit(tr)
    l2 = L2(spec, sigma=sigma, l1=l1, n_cplx=1, max_nfev=3).fit(tr)
    return {"l0a": l0a, "l0b": l0b, "l1": l1, "l2": l2}


def _parity(model, runs):
    doc = json.loads(json.dumps(model.doc()))        # through JSON, as shipped
    err = 0.0
    for r in runs:
        U = np.vstack([r.U, r.U[::-1]])              # longer than training
        err = max(err, float(np.abs(rt.rollout_from_blob(doc, r.y0, U) - model.rollout(r.y0, U)).max()))
    return err


@pytest.mark.parametrize("kind", ["l0a", "l0b", "l1", "l2"])
def test_parity(fitted, data, kind):
    _, _, va, _ = data
    m = fitted[kind]
    Y = m.rollout(va[0].y0, va[0].U)
    assert Y.shape == va[0].Y.shape and np.all(np.isfinite(Y))
    assert _parity(m, va) < 1e-8


def test_l1_beats_persistence(fitted, data):
    _, _, va, sigma = data

    def sc(m):
        return np.mean([metric.robust_score(m.rollout(r.y0, r.U), r.Ytrue, sigma) for r in va])
    assert sc(fitted["l1"]) > sc(fitted["l0a"]) + 0.1
    assert sc(fitted["l0b"]) > sc(fitted["l0a"])


def test_l1_stable_and_clipped(fitted, data):
    spec, _, va, _ = data
    m = fitted["l1"]
    assert np.all((m.a > 0) & (m.a < 1))
    lo, hi = rt.clip_vectors(m.meta)
    U = np.tile(np.array(spec.hi()), (4000, 1))
    Y = rt.rollout_from_blob(m.doc(), va[0].y0, U)
    assert np.all(np.isfinite(Y)) and np.all(Y >= lo) and np.all(Y <= hi)


def test_l2_stable(fitted):
    m = fitted["l2"]
    A = np.array(m.export()["A"])
    assert np.max(np.abs(np.linalg.eigvals(A))) < 1.0


def test_ensemble_perobs_parity(fitted, data):
    spec, _, va, _ = data
    ens = Ensemble(spec, [fitted["l0b"], fitted["l1"], fitted["l2"]])
    po = PerObs(spec, [fitted["l1"], fitted["l0b"]], [0, 1, 0])
    assert _parity(ens, va) < 1e-8
    assert _parity(po, va) < 1e-8


def test_ode_wrapper_parity(data):
    from gtlab.models.ode_model import ODEModel, has_family
    spec, tr, va, sigma = data
    if not has_family(spec.id):
        pytest.skip("no ODE family module yet")
    m = ODEModel(spec, sigma=sigma)
    m._prepare(tr)
    assert _parity(m, va[:1]) < 1e-8


def test_nonfinite_guard():
    spec = S.get("power_grid")
    from gtlab.models.common import make_doc
    doc = make_doc(spec, {"kind": "l0a"})
    lo, hi = rt.clip_vectors(doc)
    Y = np.array([[np.nan, np.inf, -5.0], [1.0, 2.0, 3.0]])
    out = rt.finalize(Y, np.array([10.0, 50.0, 0.3]), lo, hi)
    assert np.all(np.isfinite(out))
    assert out[0, 0] == 10.0 and out[0, 1] == 50.0 and out[0, 2] == 0.0 and out[1, 2] == 1.0


# ----------------------------------------------------------------------------- predict template
def _load_predict(folder, name):
    spec = importlib.util.spec_from_file_location(name, str(Path(folder) / "predict.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _episode(spec, T=4000, seed=0):
    from gtlab.check import make_episodes
    eps, _ = make_episodes(spec, 3, T, seed)
    return eps


def test_predict_template(fitted, tmp_path):
    spec = S.get("power_grid")
    d = package.build(["power_grid"], "t", {"power_grid": fitted["l1"].doc()}, stamp="x", out_root=tmp_path)
    folder = d / "power_grid"
    mod = _load_predict(folder, "pred_ok")
    ctx = spec.context()
    eps = _episode(spec)
    a1 = mod.predict(eps[0]["initial"], eps[0]["interventions"], ctx)
    b = mod.predict(eps[1]["initial"], eps[1]["interventions"], ctx)
    a2 = mod.predict(eps[0]["initial"], eps[0]["interventions"], ctx)
    assert len(a1) == 4000 and a1 == a2 and a1 != b
    assert all(isinstance(v, float) and np.isfinite(v) for row in a1 for v in row.values())
    assert set(a1[0]) == set(spec.observables)
    # the model path was used (not the persistence fallback)
    y0 = np.array([eps[0]["initial"][o] for o in spec.observables])
    U = np.array([[a[c] for c in spec.controls] for a in eps[0]["interventions"]])
    ref = rt.rollout_from_blob(fitted["l1"].doc(), y0, U)
    assert np.allclose([[r[o] for o in spec.observables] for r in a1], ref, rtol=0, atol=1e-9)


def test_predict_fallback_corrupt(fitted, tmp_path):
    spec = S.get("power_grid")
    d = package.build(["power_grid"], "t", {"power_grid": fitted["l1"].doc()}, stamp="y", out_root=tmp_path)
    folder = d / "power_grid"
    (folder / "model.json").write_text("{not json")
    mod = _load_predict(folder, "pred_bad")
    eps = _episode(spec)
    init = dict(eps[0]["initial"], renewable_share=1.7)
    out = mod.predict(init, eps[0]["interventions"], spec.context())
    assert len(out) == 4000
    assert out[0]["renewable_share"] == 1.0                      # hard-clipped persistence
    assert out[-1]["load"] == float(init["load"])
    # wrong family in context -> fallback, never raises
    mod2 = _load_predict(d / "power_grid", "pred_bad2")
    out2 = mod2.predict(eps[0]["initial"], eps[0]["interventions"][:10], dict(spec.context(), family="market"))
    assert len(out2) == 10


# ----------------------------------------------------------------------------- package + check
def test_persistence_package_check(tmp_path):
    d = package.build_persistence_all("persist", stamp="z", out_root=tmp_path)
    z = d / "submission.zip"
    with zipfile.ZipFile(z) as zf:
        roots = {n.split("/")[0] for n in zf.namelist()}
        assert roots == set(S.SYSTEM_IDS)
        assert not any("__pycache__" in n for n in zf.namelist())
    # deterministic zip bytes
    d2 = package.build_persistence_all("persist", stamp="z2", out_root=tmp_path)
    assert (d2 / "submission.zip").read_bytes() == z.read_bytes()
    rep = check(z, episodes=4, T=4000, verbose=False)
    assert rep["pass"], json.dumps(rep, indent=1)[:3000]
    assert set(rep["systems"]) == set(S.SYSTEM_IDS)
    assert Path(rep["report_path"]).exists()


def test_check_catches_violations(tmp_path):
    d = package.build(["market"], "bad", {"market": package.make_doc(S.get("market"), {"kind": "l0a"})},
                      stamp="b", out_root=tmp_path)
    folder = d / "market"
    p = folder / "predict.py"
    p.write_text(p.read_text() + "\nimport socket\nKEY = 'AIzaFAKE'\n")
    zp = package.write_zip(d, ["market"], d / "submission.zip")
    rep = check(zp, episodes=2, T=50, verbose=False)
    assert not rep["pass"]
    errs = " ".join(rep["static_errors"])
    assert "socket" in errs and "AIza" in errs
    # a crashing predict fails the sandbox run
    p.write_text("def predict(initial, interventions, context):\n    raise RuntimeError('x')\n")
    zp = package.write_zip(d, ["market"], d / "submission.zip")
    rep = check(zp, episodes=2, T=50, verbose=False)
    assert not rep["pass"] and not rep["systems"]["market"]["pass"]
