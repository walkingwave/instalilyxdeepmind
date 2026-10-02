"""Write the wildlife_tm switch variants from gtlab/ode/wildlife_tm.py (same equations, other FLAGS).

    python scripts/lab_tm_wild_make.py
"""
from pathlib import Path

ODE = Path(__file__).resolve().parent.parent / "gtlab" / "ode"
base = (ODE / "wildlife_tm.py").read_text()
VARIANTS = {"tmq": ["cq"], "tmp": ["pd"], "tmqp": ["cq", "pd"], "tms": ["sh"], "tmsp": ["sh", "pd"], "tmqs": ["cq", "sh"]}
for tag, flags in VARIANTS.items():
    src = base.replace('FAMILY = "wildlife_tm"', f'FAMILY = "wildlife_{tag}"').replace("FLAGS = []", f"FLAGS = {flags!r}")
    src = src.replace("Grey-box ODE: wildlife_tm =", f"Grey-box ODE: wildlife_{tag} (flags {flags}) =")
    (ODE / f"wildlife_{tag}.py").write_text(src)
    print("wrote", f"wildlife_{tag}.py")
