import numpy as np, json, sys
from gtlab import systems as S
from gtlab.ledger import Ledger, load_runs
np.set_printoptions(precision=2, suppress=True, linewidth=200)
for sid in S.SYSTEM_IDS:
    spec = S.get(sid)
    from gtlab.ledger import data_dir; led = Ledger(data_dir(sid, False), sid, False)
    runs = load_runs(spec, led)
    print(f"\n===== {sid}  obs={spec.observables}  ctrl={spec.controls}  runs={len(runs)} ticks={sum(r.T for r in runs)}")
    for r in runs:
        U = r.U; Y = r.Y
        # distinct control vectors
        uniq = np.unique(np.round(U, 4), axis=0)
        # switches
        sw = int(np.sum(np.any(np.diff(U, axis=0) != 0, axis=1)))
        print(f"-- {r.exp:<22} T={r.T:<4} uniq_u={len(uniq):<3} switches={sw:<3} y0={np.round(r.y0,2)}")
        idx = [0, 5, 20, 60, 119, 199, 299, 399] 
        idx = [i for i in idx if i < r.T]
        for i in idx:
            print(f"     t={i:<4} u={np.round(U[i],3)} y={np.round(Y[i],2)}")
        # noise estimate from last 30 ticks if constant control
        tail = Y[-30:]
        d = np.diff(tail, axis=0)
        print(f"     tail30 mean={np.round(tail.mean(0),2)} sd={np.round(tail.std(0),2)}  diff_sd/sqrt2={np.round(d.std(0)/np.sqrt(2),2)}")
