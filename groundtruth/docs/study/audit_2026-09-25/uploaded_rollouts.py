import numpy as np, json, importlib.util, sys, warnings, time
warnings.filterwarnings("ignore")
from gtlab import systems as S, design as D
from gtlab.ledger import Ledger, load_runs, data_dir
np.set_printoptions(precision=2, suppress=True, linewidth=220)
best = {"supply_chain":"u003","ad_auction":"u003","traffic":"u003","power_grid":"u003","wildlife":"u004","reservoir":"u003",
        "hospital_queue":"u005","market":"u003","social_contagion":"u003","epidemic":"u004"}
dirs = {"u003":"submissions/20260924-2301-u003-screen","u004":"submissions/20260924-2305-u004-test","u005":"submissions/20260924-2312-u005-lam4"}
for sid in S.SYSTEM_IDS:
    spec = S.get(sid)
    runs = load_runs(spec, Ledger(data_dir(sid, False), sid, False))
    folder = f"{dirs[best[sid]]}/{sid}"
    sp = importlib.util.spec_from_file_location(f"pred_{sid}", f"{folder}/predict.py"); mod = importlib.util.module_from_spec(sp); sp.loader.exec_module(mod)
    doc = json.load(open(f"{folder}/model.json"))
    Ymax = np.max(np.concatenate([r.Y for r in runs]), axis=0); Ymin = np.min(np.concatenate([r.Y for r in runs]), axis=0)
    print(f"\n== {sid} [{best[sid]}] kind={doc['model'].get('kind')} lam={doc['model'].get('lam')} clip_hi={np.round(doc['clip_hi'],1)} data_max={np.round(Ymax,1)} data_min={np.round(Ymin,1)}")
    ctx = spec.context()
    rng = np.random.default_rng(7)
    for ep in range(8):
        cat = ["sustained","order","recovery","composition"][ep % 4]
        U = D.eval_like(spec, cat, 4000, rng)
        y0 = runs[ep % len(runs)].y0
        init = {o: float(v) for o, v in zip(spec.observables, y0)}
        ivs = [{c: float(U[t, i]) for i, c in enumerate(spec.controls)} for t in range(4000)]
        t0 = time.time(); out = mod.predict(init, ivs, ctx); dt = time.time() - t0
        Y = np.array([[d[o] for o in spec.observables] for d in out])
        above = np.mean(Y > Ymax[None, :] * 1.05 + 1e-9, axis=0); below = np.mean(Y < Ymin[None, :] - 0.05 * (Ymax - Ymin)[None, :] - 1e-9, axis=0)
        print(f"  {cat:<11} {dt:4.1f}s  end={np.round(Y[-1],1)} max={np.round(Y.max(0),1)}  frac>data_max={np.round(above,2)} frac<data_min={np.round(below,2)}")
