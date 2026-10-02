"""epi9: fold table from ode_lab logs (scratch logs given as args)."""
import re, sys
for fn in sys.argv[1:]:
    rows = re.findall(r"LOO (\S+)\s+ode \[([\d. ]+)\] mean ([\d.]+)", open(fn).read())
    ms = [float(m) for _, _, m in rows]
    v = re.search(r"VERDICT.*", open(fn).read())
    print(f"{fn.split('/')[-1]:<14}", " ".join(f"{n.split('.')[0]}={float(m):.3f}" for n, _, m in rows),
          f"| mean {sum(ms)/len(ms):.3f} (n={len(ms)})" if ms else "", "|", (v.group(0)[:60] if v else "running"))
