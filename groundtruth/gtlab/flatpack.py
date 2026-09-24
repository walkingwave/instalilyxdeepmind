"""Build a clean, self-contained predict.py per system (plus model.json).

Every submission folder holds exactly two files. The inference runtime (and any ODE modules) is
inlined into predict.py through the AST: no docstrings, no comments, no provenance strings.
model.json keeps only the numbers the forecaster needs.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime" / "infer.py"
ODE_DIR = ROOT / "ode"

KEEP_DOC_KEYS = ("format", "system", "observables", "controls", "bounds", "recovery",
                 "hard_lo", "hard_hi", "clip_lo", "clip_hi", "model")


class _Strip(ast.NodeTransformer):
    def _body(self, body):
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                and isinstance(body[0].value.value, str):
            body = body[1:] or [ast.Pass()]
        return body

    def visit_Module(self, node):
        self.generic_visit(node)
        node.body = [n for n in self._body(node.body)
                     if not (isinstance(n, ast.ImportFrom) and n.module == "__future__")]
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        node.body = self._body(node.body)
        node.returns = None
        for a in node.args.args + node.args.kwonlyargs + node.args.posonlyargs:
            a.annotation = None
        return node

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        node.body = self._body(node.body)
        return node

    def visit_AnnAssign(self, node):
        if node.value is None:
            return None
        return ast.copy_location(ast.Assign(targets=[node.target], value=node.value), node)


DROP_TOP = {"ode_modules", "_load_file_module", "_MOD_CACHE"}


def _drop(tree, names):
    keep = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names:
            continue
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets):
            continue
        if isinstance(n, ast.Import) and all(a.name == "importlib.util" for a in n.names) and "importlib" in names:
            continue
        keep.append(n)
    tree.body = keep
    return tree


def clean_source(src: str, drop=()) -> str:
    tree = _Strip().visit(ast.parse(src))
    if drop:
        tree = _drop(tree, set(drop))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _as_namespace(fn_name: str, src: str) -> str:
    """Wrap a module's source in a function returning an object with its globals."""
    body = clean_source(src)
    lines = [f"def {fn_name}():"]
    lines += ["    " + ln if ln.strip() else "" for ln in body.splitlines()]
    lines += ["    class _Namespace_q7:", "        pass", "    _obj_q7 = _Namespace_q7()",
              "    _obj_q7.__dict__.update(locals())", "    return _obj_q7"]
    return "\n".join(lines)


TAIL = '''
SYSTEM = {system!r}
HARD = {hard!r}
_HERE = Path(__file__).resolve().parent
_DOC = {{}}


def _get_doc():
    if "doc" not in _DOC:
        _DOC["doc"] = load_doc(_HERE)
    return _DOC["doc"]


def _fnum(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return f if math.isfinite(f) else 0.0


def _persistence(initial, n, names):
    row = {{}}
    for k in names:
        v = _fnum(initial.get(k)) if isinstance(initial, dict) else 0.0
        lo, hi = HARD.get(k, (0.0, None))
        if lo is not None and v < lo:
            v = float(lo)
        if hi is not None and v > hi:
            v = float(hi)
        row[k] = v
    return [dict(row) for _ in range(n)]


def _names(initial, context):
    names = list(initial.keys()) if isinstance(initial, dict) else []
    try:
        for k in context.get("observables", []) or []:
            if k not in names:
                names.append(k)
    except Exception:
        pass
    return names


def _valid(out, n, names):
    if not isinstance(out, list) or len(out) != n:
        return False
    for row in out:
        if not isinstance(row, dict) or len(row) != len(names):
            return False
        for k in names:
            v = row.get(k)
            if not isinstance(v, float) or not math.isfinite(v):
                return False
    return True


def predict(initial, interventions, context):
    try:
        n = len(interventions)
    except Exception:
        n = 4000
    try:
        names = _names(initial, context)
    except Exception:
        names = []
    try:
        doc = _get_doc()
        init = {{k: initial.get(k, 0.0) for k in names}}
        out = predict_episode(doc, init, interventions, context, base_dir=_HERE, tag=SYSTEM)
        if _valid(out, n, names):
            return out
    except Exception:
        pass
    return _persistence(initial, n, names)
'''


def kinds_used(blob):
    out = set()
    if isinstance(blob, dict) and "kind" in blob:
        out.add(blob["kind"])
        for m in blob.get("members", []) or []:
            out |= kinds_used(m)
        if isinstance(blob.get("member"), dict):
            out |= kinds_used(blob["member"])
    return out


def _prune(src, kinds, roots=("predict",)):
    """Keep only top-level definitions reachable from `roots`; restrict _KINDS to `kinds`."""
    tree = ast.parse(src)
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_KINDS" for t in n.targets) \
                and isinstance(n.value, ast.Dict):
            pairs = [(k, v) for k, v in zip(n.value.keys, n.value.values)
                     if isinstance(k, ast.Constant) and k.value in kinds]
            n.value.keys = [k for k, _ in pairs]
            n.value.values = [v for _, v in pairs]

    def defined(n):
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            return {n.name}
        if isinstance(n, ast.Assign):
            return {t.id for t in n.targets if isinstance(t, ast.Name)}
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            return {(a.asname or a.name).split(".")[0] for a in n.names}
        return set()

    by_name = {}
    for n in tree.body:
        for d in defined(n):
            by_name.setdefault(d, []).append(n)
    keep, todo = set(), list(roots)
    while todo:
        name = todo.pop()
        if name in keep or name not in by_name:
            continue
        keep.add(name)
        for n in by_name[name]:
            for sub in ast.walk(n):
                if isinstance(sub, ast.Name):
                    todo.append(sub.id)
    tree.body = [n for n in tree.body if (defined(n) & keep) or not defined(n)]
    return ast.unparse(tree)


def render(spec, families=(), kinds=None):
    parts = [clean_source(RUNTIME.read_text(), drop=DROP_TOP | {"importlib"})]
    if families:
        parts.append(_as_namespace("_ode_core", (ODE_DIR / "core.py").read_text()))
        for fam in sorted(families):
            parts.append(_as_namespace(f"_ode_{fam}", (ODE_DIR / f"{fam}.py").read_text()))
        mapping = ", ".join(f"{f!r}: _ode_{f}" for f in sorted(families))
        parts.append(
            "_ODE_BUILD = {" + mapping + "}\n_ODE_MODS = {}\n\n\n"
            "def ode_modules(family, base_dir=None, tag=''):\n"
            "    if family not in _ODE_MODS:\n"
            "        _ODE_MODS[family] = (_ode_core(), _ODE_BUILD[family]())\n"
            "    return _ODE_MODS[family]")
    else:
        parts.append("def ode_modules(family, base_dir=None, tag=''):\n    raise KeyError(family)")
    hard = {o: list(spec.output_bounds(o)) for o in spec.observables}
    parts.append(TAIL.format(system=spec.id, hard=hard).strip())
    src = "\n\n\n".join(parts) + "\n"
    if kinds is not None:
        src = _prune(src, set(kinds)) + "\n"
    compile(src, "predict.py", "exec")
    return src


def clean_doc(doc: dict) -> dict:
    return {k: doc[k] for k in KEEP_DOC_KEYS if k in doc}


def write_folder(folder: Path, spec, doc, families=()):
    folder.mkdir(parents=True, exist_ok=True)
    for p in folder.iterdir():
        if p.is_file():
            p.unlink()
    (folder / "predict.py").write_text(render(spec, families, kinds=kinds_used(doc["model"])))
    (folder / "model.json").write_text(json.dumps(clean_doc(doc), sort_keys=True, separators=(",", ":"),
                                                  allow_nan=False))
    smoke(folder, spec)


def smoke(folder: Path, spec, T=300):
    """Run the model path directly (bypassing predict()'s persistence fallback) so a broken
    model fails the build instead of silently shipping persistence."""
    import importlib.util
    import sys
    import numpy as np
    prev, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    sp = importlib.util.spec_from_file_location(f"smoke_{spec.id}_{id(folder)}", str(folder / "predict.py"))
    m = importlib.util.module_from_spec(sp)
    try:
        sp.loader.exec_module(m)
    finally:
        sys.dont_write_bytecode = prev
    lo = np.array([spec.bounds[c][0] for c in spec.controls])
    hi = np.array([spec.bounds[c][1] for c in spec.controls])
    U = lo + (hi - lo) * np.random.default_rng(0).random((T, spec.m))
    init = {o: 1.0 for o in spec.observables}
    out = m.predict_episode(m._get_doc(), init, [dict(zip(spec.controls, map(float, r))) for r in U],
                            spec.context(), base_dir=folder, tag=spec.id)
    if len(out) != T or not all(np.isfinite(list(r.values())).all() for r in out):
        raise RuntimeError(f"{spec.id}: smoke test produced invalid output")
