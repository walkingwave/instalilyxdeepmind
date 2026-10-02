"""ens9 stage 1: honest leave-one-run-out predictions for every ensemble member (no credits).

    python scripts/lab_ens9_loo.py [--systems a,b] [--workers 4] [--budget 60] [--starts 6] [--nfev 50]

Every member is refit from its family's default init (cold, lab protocol: multistart LHS,
spread 0.5, no early phase) on all runs but one, at the calibrated organizer sigma (1.0x), and
the held-out run is predicted. l0b_lin is refit per fold (clip margin 1). Predictions (raw,
not finalized) are cached as JSON in ENS9_SCRATCH/<system>/<label>__<held>.json.
Every run is a fold, exam and long holds included.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import importlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gtlab import metric, select as SEL, systems as S  # noqa: E402
from gtlab.ledger import Ledger, data_dir, load_runs  # noqa: E402
from gtlab.models import common as C  # noqa: E402
from gtlab.ode import core, fit as F  # noqa: E402

SCRATCH = Path(os.environ.get("ENS9_SCRATCH", "C:/Users/DYLANH~1/AppData/Local/Temp/gtscratch/ens9"))
PLANS = ROOT / "plans"
FINAL = ROOT / "submissions" / "20260928-1440-final1"

# member label -> doc under plans/ (theta for the full-data prediction, family + mech for refits)
MEMBERS = {
    "ad_auction": {"v8b": "ad_auction_ad_auction_v8b_loo", "s1": "ad_auction_ad_auction_s1_cal4",
                   "min": "ad_auction_ad_auction_min_v8ref", "v8c": "ad_auction_ad_auction_v8c_loo"},
    "epidemic": {"y2": "epidemic_epidemic_y2_cabase", "c": "epidemic_epidemic_c_ca",
                 "y11": "epidemic_epidemic_y11_AC", "y6": "epidemic_epidemic_y6_AC"},
    "hospital_queue": {"p3": "hospital_queue_hospital_queue_p3_cabase", "c2": "hospital_queue_hospital_queue_c2_ca",
                       "y5": "hospital_queue_hospital_queue_y5_y5AB", "y3": "hospital_queue_hospital_queue_y3_y3AB12"},
    "market": {"y3": "market_market_y3_r9", "d3": "market_market_d3_AB", "z12": "market_market_z12_r9",
               "z8": "market_market_z8_AB"},
    "power_grid": {"v9c": "power_grid_power_grid_v9c_cabase", "v8k": "power_grid_power_grid_v8k_v8k_lab",
                   "min": "power_grid_power_grid_min_v8ref", "w5": "power_grid_power_grid_w5_w5",
                   "c": "power_grid_power_grid_c_ca"},
    "reservoir": {"v8": "reservoir_reservoir_v8_v8lab", "min2": "reservoir_reservoir_min2",
                  "min": "reservoir_reservoir_min", "s1": "reservoir_reservoir_s1_cal4"},
    "social_contagion": {"z20": "social_contagion_social_contagion_z20_zr6BC",
                         "z21": "social_contagion_social_contagion_z21_zr6BC",
                         "z15": "social_contagion_social_contagion_z15_zr6BC",
                         "c": "social_contagion_social_contagion_c_ca"},
    "supply_chain": {"v8b": "supply_chain_supply_chain_v8b_a", "min": "supply_chain_supply_chain_min_v7b",
                     "min2": "supply_chain_supply_chain_min2_v7b", "v8c": "supply_chain_supply_chain_v8c_a"},
    "traffic": {"z8": "traffic_traffic_z8_cabase", "c2": "traffic_traffic_c2_ca", "c": "traffic_traffic_c_ca",
                "v8d": "traffic_traffic_v8d_v8dall7"},
    "wildlife": {"v8h": "wildlife_wildlife_v8h_p7", "w2": "wildlife_wildlife_w2_p7", "w4": "wildlife_wildlife_w4_p7",
                 "z9": "wildlife_wildlife_z9_p7", "c": "wildlife_wildlife_c_ca"},
}
# the current pick (Final slot 1) as an option over member labels, per observable
PICK = {
    "ad_auction": [("v8b",), ("v8b", "s1"), ("v8b", "min")],
    "power_grid": None,  # filled from the shipped perobs map at selection time
}
SYSTEMS = list(MEMBERS)


def get_runs(system):
    spec = S.get(system)
    return spec, load_runs(spec, Ledger(data_dir(system, False), system, False))


def cal_sigma(system):
    return np.asarray(json.loads((PLANS / "sigma_calibrated.json").read_text())[system]["sigma"], float)


def member_blob(system, label):
    return json.loads((PLANS / f"{MEMBERS[system][label]}_doc.json").read_text())["model"]


def _job(system, label, held, budget, starts, nfev):
    spec, runs = get_runs(system)
    sigma = cal_sigma(system)
    tr = [r for r in runs if r.exp != held]
    te = next(r for r in runs if r.exp == held)
    t0 = time.time()
    pay = {"train": [r.exp for r in tr]}
    with C.single_thread():
        if label == "l0b":
            clip = C.soft_clip(spec, tr, margin=1.0)
            m = SEL.make_model("l0b", spec, clip, sigma, cfg={"sq": False, "pairs": None}).fit(tr)
            Y = m.rollout(te.y0, te.U)
        else:
            blob = member_blob(system, label)
            mod = importlib.import_module(f"gtlab.ode.{blob['family']}")
            mech = frozenset(blob["mech"])
            n_sub = int(blob.get("n_sub", getattr(mod, "N_SUB", 2)))
            th, info = F.fit_ode(mod, tr, sigma, mech, n_starts=starts, max_nfev=nfev, n_sub=n_sub, early_T=None,
                                 time_budget=budget, theta0=None, spread=0.5)
            pay["theta"] = [float(v) for v in th]
            pay["cost"] = float(info["cost"])
            Y = core.rollout(mod, te.y0, te.U, core.theta_dict(mod, th), mech, n_sub=n_sub)
    Y = np.asarray(Y, float)
    pay["Y"] = Y.tolist()
    pay["score"] = metric.score_per_obs(Y, te.Y, sigma).tolist()
    pay["sec"] = time.time() - t0
    return pay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--budget", type=float, default=60)
    ap.add_argument("--starts", type=int, default=6)
    ap.add_argument("--nfev", type=int, default=50)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    systems = a.systems.split(",") if a.systems else SYSTEMS
    jobs = []
    for s in systems:
        _, runs = get_runs(s)
        (SCRATCH / s).mkdir(parents=True, exist_ok=True)
        labels = a.labels.split(",") if a.labels else list(MEMBERS[s]) + ["l0b"]
        for label in labels:
            for r in runs:
                f = SCRATCH / s / f"{label}__{r.exp}.json"
                if a.force or not f.exists():
                    jobs.append((s, label, r.exp))
    jobs.sort(key=lambda j: j[1] != "l0b")
    print(f"{len(jobs)} jobs on {a.workers} workers", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_job, s, l, h, a.budget, a.starts, a.nfev): (s, l, h) for s, l, h in jobs}
        for n, fu in enumerate(as_completed(futs)):
            s, l, h = futs[fu]
            try:
                pay = fu.result()
            except Exception as e:
                pay = {"error": repr(e)}
            (SCRATCH / s / f"{l}__{h}.json").write_text(json.dumps(pay))
            msg = ("ERR " + pay["error"]) if "error" in pay else f"{np.mean(pay['score']):.3f} {pay['sec']:.0f}s"
            print(f"[{n + 1}/{len(jobs)} {time.time() - t0:.0f}s] {s} {l} held {h} {msg}", flush=True)


if __name__ == "__main__":
    main()
