"""`gt` command line. Everything defaults to the mock; real calls need explicit flags + env.

  gt init [--mock]                       systems.json + budget.json per system
  gt plan SYS|--all --phase p1 [--real-dir]  materialize plan.json (data_mock/ unless --real-dir)
  gt collect SYS|--all --phase p1 [--dry-run] [--spend --max-steps N --yes]
  gt mock-collect SYS|--all --phase p1   same as collect without --spend (mock, data_mock/)
  gt resets SYS|--all --n 40 [--real]    free resets only (real needs GT_ALLOW_REAL=1)
  gt docs SYS|--all [--real]             free brief + documents (real needs GT_ALLOW_REAL=1)
  gt status [--mock]
  gt fit|validate|package|check|pipeline|mock-bench ...   (hooks into gtlab.select/package/check)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from gtlab import budget as budgetmod
from gtlab import design, systems
from gtlab.ledger import ROOT, Ledger, data_dir, load_runs, req_id


class CliError(SystemExit):
    pass


def _die(msg):
    print(f"error: {msg}", file=sys.stderr)
    raise CliError(2)


def _targets(a):
    if getattr(a, "all", False):
        return list(systems.SYSTEM_IDS)
    if not getattr(a, "system", None):
        _die("give a SYSTEM or --all")
    if a.system not in systems.SYSTEM_IDS:
        _die(f"unknown system {a.system}; one of {systems.SYSTEM_IDS}")
    return [a.system]


def _base(a):
    return Path(a.data_root) if getattr(a, "data_root", None) else None


def _is_mock(a):
    return not (getattr(a, "spend", False) or getattr(a, "real", False))


def _ledger(sys_id, mock, a):
    return Ledger(data_dir(sys_id, mock, _base(a)), sys_id, mock=mock)


def _mech(a):
    m = (getattr(a, "mech", None) or "AB").upper()
    if len(m) != 2 or not set(m) <= set("ABC") or m[0] == m[1]:
        _die("--mech must be two of A/B/C, e.g. AB")
    return tuple(m)


def _mock_gateway(sys_id, a, directory):
    from gtlab.gateway import MockGateway
    p = Path(directory) / "mock_server.json"
    if p.exists():
        return MockGateway.load(p), p
    return MockGateway(sys_id, mech=_mech(a), seed=getattr(a, "mock_seed", 0) or 0), p


def _real_gateway(spend):
    from gtlab.gateway import RealGateway
    return RealGateway(spend=spend)


def _confirm(a, what):
    if a.yes:
        return
    if not sys.stdin.isatty():
        _die(f"{what}: pass --yes (non-interactive)")
    ans = input(f"{what}. Type SPEND to continue: ").strip()
    if ans != "SPEND":
        _die("aborted")


# --------------------------------------------------------------------------- commands
def cmd_init(a):
    base = _base(a) or ROOT
    out = Path(base) / "systems.json"
    out.write_text(json.dumps({k: v.to_json() for k, v in systems.load_all().items()}, indent=1))
    print(f"wrote {out}")
    for s in systems.SYSTEM_IDS:
        d = data_dir(s, a.mock, _base(a))
        d.mkdir(parents=True, exist_ok=True)
        p = budgetmod.write(d, budgetmod.default_plan(s))
        print(f"  {p}")
    return 0


def cmd_plan(a):
    mock = not a.real_dir
    for s in _targets(a):
        spec = systems.get(s)
        d = data_dir(s, mock, _base(a))
        led = _ledger(s, mock, a)
        ex = None
        if a.experiments:
            ex = design.experiments_from_file(spec, a.experiments, a.phase)
            if not a.force and a.phase in design.load_plan_file(d).get("phases", {}):
                _die(f"{s} {a.phase} already materialized; add --force to replace it (refused if started)")
        ph = design.materialize(spec, d, a.phase, seed=a.seed, experiments=ex, force=a.force, ledger=led)
        from gtlab.collect import format_cost, phase_cost
        print(format_cost(s, a.phase, phase_cost(led, ph)))
        print(f"  plan hash {ph['hash']} seed {ph['seed']} -> {design.plan_path(d)}")
    return 0


def _get_phase(spec, d, phase, seed, led):
    doc = design.load_plan_file(d)
    ph = doc.get("phases", {}).get(phase)
    if ph is None:
        ph = design.materialize(spec, d, phase, seed=seed, ledger=led)
        print(f"  materialized {phase} into {design.plan_path(d)} (hash {ph['hash']})")
    return ph


def cmd_collect(a):
    spend = bool(a.spend)
    if spend and not a.dry_run:          # a dry run calls nothing, so it needs neither
        if a.max_steps is None:
            _die("--spend requires --max-steps N (no default)")
        if os.environ.get("GT_ALLOW_SPEND") != "1":
            _die("--spend requires GT_ALLOW_SPEND=1 in the environment")
    mock = not spend
    total, failed = 0, []
    for s in _targets(a):
        try:
            total += _collect_one(a, s, spend, mock)
        except (KeyboardInterrupt, CliError):
            raise
        except Exception as e:          # one system failing must not stop the others
            print(f"{s}: FAILED {type(e).__name__}: {e}", file=sys.stderr)
            failed.append(s)
    print(f"bought {total} {'REAL' if spend else 'mock'} steps"
          + (f"; failed: {', '.join(failed)}" if failed else ""))
    a._failed = failed
    return 1 if failed else 0


def _collect_one(a, s, spend, mock):
    from gtlab.collect import collect_phase, phase_cost
    spec = systems.get(s)
    d = data_dir(s, mock, _base(a))
    led = _ledger(s, mock, a)
    bplan = budgetmod.load(d, s)
    budgetmod.write(d, bplan)
    ph = _get_phase(spec, d, a.phase, a.seed, led)
    only = [x.strip() for x in (getattr(a, "only", None) or "").split(",") if x.strip()]
    if only:
        keep = [e for e in ph["experiments"] if e["exp_id"].split(".", 1)[-1] in only]
        if not keep:
            _die(f"--only matched no experiments in {a.phase}: {only}")
        ph = dict(ph, experiments=keep, steps=sum(e["T"] for e in keep))
    if a.dry_run:
        collect_phase(spec, None, led, ph, bplan, dry_run=True)
        return 0
    need = sum(x["todo"] for x in phase_cost(led, ph))
    if spend:
        if need == 0:
            print(f"{s} {a.phase}: nothing to buy")
            return 0
        _confirm(a, f"{s}: about to BUY {need} real simulator steps (cap {a.max_steps})")
        gw = _real_gateway(True)
        try:
            r = collect_phase(spec, gw, led, ph, bplan, max_steps=a.max_steps, reconcile=a.reconcile)
        finally:
            gw.close()
    else:
        gw, pkl = _mock_gateway(s, a, d)
        try:
            r = collect_phase(spec, gw, led, ph, bplan,
                              max_steps=a.max_steps if a.max_steps is not None else need,
                              reconcile=a.reconcile)
        finally:
            gw.save(pkl)
    return r["bought"]


def cmd_mock_collect(a):
    a.spend = False
    if a.fresh:
        import shutil
        for s in _targets(a):
            d = data_dir(s, True, _base(a))
            if d.exists():
                shutil.rmtree(d)
    return cmd_collect(a)


def _free_gateway(a, s, d):
    if a.real:
        if os.environ.get("GT_ALLOW_REAL") != "1":
            _die("--real needs GT_ALLOW_REAL=1 (free endpoints only; never charges steps)")
        return _real_gateway(False), None
    return _mock_gateway(s, a, d)


def cmd_resets(a):
    mock = not a.real
    for s in _targets(a):
        spec = systems.get(s)
        d = data_dir(s, mock, _base(a))
        led = _ledger(s, mock, a)
        gw, pkl = _free_gateway(a, s, d)
        try:
            k0 = len(led.reset_pool())
            for k in range(a.n):
                rq = req_id(s, "pool", 0, f"reset{k0 + k}")
                resp = gw.reset(s, request_id=rq)
                obs = {o: float(resp["observation"][o]) for o in spec.observables}
                led.log_reset_candidate({"exp": "pool", "k": k0 + k, "req": rq, "run_id": resp.get("run_id"),
                                         "obs": obs, "raw": resp})
            print(f"{s}: {a.n} resets logged to {led.resets_path}")
        finally:
            if pkl is not None:
                gw.save(pkl)
            elif hasattr(gw, "close"):
                gw.close()
    return 0


def cmd_docs(a):
    mock = not a.real
    for s in _targets(a):
        d = data_dir(s, mock, _base(a))
        d.mkdir(parents=True, exist_ok=True)
        gw, pkl = _free_gateway(a, s, d)
        try:
            doc = {"system": s, "fetched": time.time(), "brief": gw.brief(s),
                   "documents": gw.documents(s), "budget": gw.budget(s)}
        finally:
            if pkl is None and hasattr(gw, "close"):
                gw.close()
        (d / "docs.json").write_text(json.dumps(doc, indent=1))
        print(f"{s}: wrote {d / 'docs.json'}")
    return 0


def cmd_budget(a):
    """Free: print server-side remaining steps per system (no steps spent)."""
    os.environ.setdefault("GT_ALLOW_REAL", "1")
    gw = _real_gateway(False)
    try:
        tot = 0
        for s in _targets(a) if (a.system or a.all) else systems.SYSTEM_IDS:
            b = gw.budget(s)
            n = b.get("simulator_steps_remaining", b.get("remaining"))
            tot += int(n)
            print(f"{s:17s} {n}")
        print(f"{'total':17s} {tot}")
    finally:
        gw.close()
    return 0


def cmd_status(a):
    mock = a.mock
    print(f"{'system':<18}{'spent':>6}{'p1':>10}{'p2':>10}{'val':>10}{'res':>10}{'server':>8}  best_val")
    for s in systems.SYSTEM_IDS:
        d = data_dir(s, mock, _base(a))
        if not (d / "ledger.jsonl").exists():
            print(f"{s:<18}{'-':>6}")
            continue
        led = Ledger(d, s, mock=mock)
        bp = budgetmod.load(d, s)
        cols = "".join(f"{str(led.phase_charged(p)) + '/' + str(bp['phases'].get(p, 0)):>10}"
                       for p in ("p1", "p2", "val", "reserve"))
        lb = led.last_budget()
        best = "-"
        art = (Path(_base(a) or ROOT) / "artifacts" / ("mock" if mock else "") / s / "leaderboard.json")
        if art.exists():
            try:
                j = json.loads(art.read_text())
                best = str(j.get("best", j)[:1] if isinstance(j, list) else j.get("best", "?"))
            except Exception:
                best = "?"
        pend = len(led.pending_intents())
        print(f"{s:<18}{led.steps_charged():>6}{cols}{(lb or {}).get('remaining', '-'):>8}  {best}"
              + (f"  ({pend} dangling intents)" if pend else ""))
    return 0


# --------------------------------------------------------------------------- hooks (other owners)
def _lazy(modname, fn):
    import importlib
    try:
        mod = importlib.import_module(modname)
        return getattr(mod, fn)
    except (ImportError, AttributeError) as e:
        _die(f"{modname}.{fn} not available yet ({e})")


def _art_dir(a, s):
    mock = not getattr(a, "real_dir", False)
    return Path(_base(a) or ROOT) / "artifacts" / ("mock" if mock else "") / s


def _runs(a, s):
    mock = not getattr(a, "real_dir", False)
    d = data_dir(s, mock, _base(a))
    if not (d / "ledger.jsonl").exists():
        _die(f"no ledger at {d}")
    return load_runs(systems.get(s), Ledger(d, s, mock=mock))


def _as_pick(result, out_dir):
    """fit_and_select may return a doc, a path, or a report; fall back to the artifact dir."""
    if isinstance(result, (str, Path)) or (isinstance(result, dict) and "model" in result):
        return result
    return str(out_dir)


def cmd_fit(a):
    f = _lazy("gtlab.select", "fit_and_select")
    picks = {}
    for s in _targets(a):
        out = _art_dir(a, s)
        out.mkdir(parents=True, exist_ok=True)
        r = f(s, _runs(a, s), out)
        picks[s] = _as_pick(r, out)
        print(f"{s}: {r if not isinstance(r, dict) else {k: v for k, v in r.items() if k != 'model'}}")
    return picks


def cmd_validate(a):
    # validation is part of fit_and_select (it scores on the val split and writes leaderboard.json)
    return cmd_fit(a)


def cmd_package(a):
    if a.persistence:
        path = _lazy("gtlab.package", "build_persistence_all")(tag=a.tag)
        print(f"built {path}")
        return 0
    f = _lazy("gtlab.package", "build")
    targets = _targets(a)
    # default pick: artifacts/[mock/]<sys>/ (a dir holding model.json, written by fit)
    picks = json.loads(a.picks) if a.picks else {s: str(_art_dir(a, s)) for s in targets}
    path = f(targets, a.tag, picks)
    print(f"built {path}")
    return path


def _zip_of(path):
    p = Path(path)
    return p / "submission.zip" if p.is_dir() else p


def _passed(rep):
    return bool(rep.get("pass", rep.get("ok"))) if isinstance(rep, dict) else bool(rep)


def cmd_check(a):
    f = _lazy("gtlab.check", "check")
    rep = f(_zip_of(a.path))
    return 0 if _passed(rep) else 1


def cmd_pipeline(a):
    if a.dry_run:
        return cmd_collect(a)
    cmd_collect(a)
    a.real_dir = bool(a.spend)
    failed = set(getattr(a, "_failed", []))
    targets = [s for s in _targets(a) if s not in failed]
    fit = _lazy("gtlab.select", "fit_and_select")
    picks = {}
    for s in targets:
        out = _art_dir(a, s)
        out.mkdir(parents=True, exist_ok=True)
        picks[s] = _as_pick(fit(s, _runs(a, s), out), out)
    build = _lazy("gtlab.package", "build")
    path = build(targets, a.tag, picks)
    rep = _lazy("gtlab.check", "check")(_zip_of(path))
    print(f"package {path}; check {'PASS' if _passed(rep) else 'FAIL'}")
    return 0 if _passed(rep) and not failed else 1


def cmd_mock_bench(a):
    f = _lazy("gtlab.select", "mock_bench")
    for s in _targets(a):
        print(s, f(s))
    return 0


# --------------------------------------------------------------------------- parser
def build_parser():
    p = argparse.ArgumentParser(prog="gt", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", help="base dir holding data/ and data_mock/ (default: repo root)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def sysargs(sp):
        sp.add_argument("system", nargs="?")
        sp.add_argument("--all", action="store_true")

    def spendargs(sp):
        sp.add_argument("--phase", required=True, choices=design.PHASES)
        sp.add_argument("--seed", type=int, default=0, help="plan seed (materialized once)")
        sp.add_argument("--dry-run", action="store_true", help="print exact step cost, call nothing")
        sp.add_argument("--spend", action="store_true", help="REAL paid steps (needs GT_ALLOW_SPEND=1)")
        sp.add_argument("--max-steps", type=int, default=None, help="hard cap per system (required with --spend)")
        sp.add_argument("--yes", action="store_true", help="skip the interactive SPEND prompt")
        sp.add_argument("--reconcile", action="store_true", help="accept server/ledger drift as external spend")
        sp.add_argument("--only", help="comma list of experiment names in the phase, e.g. hold_rec,hold_pulse")
        sp.add_argument("--mech", default="AB", help="mock mechanism pair (fixed once data_mock/<sys>/mock_server.json exists)")
        sp.add_argument("--mock-seed", type=int, default=0)

    sp = sub.add_parser("init"); sp.add_argument("--mock", action="store_true"); sp.set_defaults(fn=cmd_init)
    sp = sub.add_parser("plan"); sysargs(sp)
    sp.add_argument("--phase", required=True, choices=design.PHASES)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--force", action="store_true", help="re-materialize (refused if a started experiment changes)")
    sp.add_argument("--experiments", help="JSON file: list of {exp_id, U:[[..]], category?, shop?, note?} "
                    "(or {system: [...]}) to use instead of the default phase design")
    sp.add_argument("--real-dir", action="store_true", help="plan into data/ (default data_mock/)")
    sp.set_defaults(fn=cmd_plan)
    sp = sub.add_parser("collect"); sysargs(sp); spendargs(sp); sp.set_defaults(fn=cmd_collect)
    sp = sub.add_parser("mock-collect"); sysargs(sp); spendargs(sp)
    sp.add_argument("--fresh", action="store_true", help="wipe data_mock/<sys> first")
    sp.set_defaults(fn=cmd_mock_collect)
    for name, fn in (("resets", cmd_resets), ("docs", cmd_docs)):
        sp = sub.add_parser(name); sysargs(sp)
        sp.add_argument("--real", action="store_true", help="free real endpoints; needs GT_ALLOW_REAL=1")
        sp.add_argument("--mech", default="AB"); sp.add_argument("--mock-seed", type=int, default=0)
        if name == "resets":
            sp.add_argument("--n", type=int, default=40)
        sp.set_defaults(fn=fn)
    sp = sub.add_parser("status"); sp.add_argument("--mock", action="store_true"); sp.set_defaults(fn=cmd_status)
    sp = sub.add_parser("budget"); sysargs(sp); sp.set_defaults(fn=cmd_budget)
    for name, fn in (("fit", cmd_fit), ("validate", cmd_validate), ("mock-bench", cmd_mock_bench)):
        sp = sub.add_parser(name); sysargs(sp)
        sp.add_argument("--real-dir", action="store_true", help="use data/ (default data_mock/)")
        sp.set_defaults(fn=fn)
    sp = sub.add_parser("package"); sysargs(sp)
    sp.add_argument("--tag", default="dev")
    sp.add_argument("--picks", help='JSON {sys: path-to-model.json-or-dir}; default artifacts/[mock/]<sys>/')
    sp.add_argument("--real-dir", action="store_true", help="use artifacts/<sys> (default artifacts/mock/<sys>)")
    sp.add_argument("--persistence", action="store_true", help="zero-training L0a build for all 10")
    sp.set_defaults(fn=cmd_package)
    sp = sub.add_parser("check"); sp.add_argument("path"); sp.set_defaults(fn=cmd_check)
    sp = sub.add_parser("pipeline"); sysargs(sp); spendargs(sp); sp.add_argument("--tag", default="pipeline")
    sp.set_defaults(fn=cmd_pipeline)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        r = a.fn(a)
    except CliError as e:
        return e.code
    return r if isinstance(r, int) else 0


if __name__ == "__main__":
    sys.exit(main())
