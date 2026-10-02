"""Generate soc9 family variants from social_contagion_z20 by text substitution."""
from pathlib import Path

ODE = Path(__file__).resolve().parent.parent / "gtlab" / "ode"
src = (ODE / "social_contagion_z20.py").read_text()


def sub(s, a, b):
    assert a in s, a
    return s.replace(a, b)


def d_chain(s, stages):
    extra = [f"D{i}" for i in range(2, stages + 1)]
    s = sub(s, '_PER = ["L", "M", "X", "D", "W1", "W2", "W3", "V1", "V2", "V3", "K"]',
            '_PER = ["L", "M", "X", "D", "W1", "W2", "W3", "V1", "V2", "V3", "K"' + "".join(f', "{e}"' for e in extra) + "]")
    s = sub(s, 'STATE_HI = np.array(([1e4] * 10 + [1.0]) * 2 + [2.0, 1.0])',
            f'STATE_HI = np.array(([1e4] * 10 + [1.0] + [1e4] * {len(extra)}) * 2 + [2.0, 1.0])')
    s = sub(s, '    L, Mm, X, D, W1, W2, W3, V1, V2, V3, K = s\n',
            '    L, Mm, X, D, W1, W2, W3, V1, V2, V3, K' + "".join(f", {e}" for e in extra) + ' = s\n')
    s = sub(s, 'pool = _pos(M, N - A - W - D, 1.0)', 'pool = _pos(M, N - A - W - D' + "".join(f" - {e}" for e in extra) + ', 1.0)')
    kd = f'{float(stages)} / th["tau_D"]'
    new = f'    dD = outL + outM + outX - {kd} * D\n'
    prev = "D"
    for e in extra:
        new += f'    d{e} = {kd} * ({prev} - {e})\n'
        prev = e
    s = sub(s, '    dD = outL + outM + outX - D / th["tau_D"]\n', new)
    s = sub(s, 'return [dL, dM, dX, dD, dW1, dW2, dW3, dV1, dV2, dV3, dK]',
            'return [dL, dM, dX, dD, dW1, dW2, dW3, dV1, dV2, dV3, dK' + "".join(f", d{e}" for e in extra) + "]")
    zz = "".join(", z" for _ in extra)
    s = sub(s, """    return _pack(M, [(1.0 - m0) * a, z, m0 * a, z, z, z, z, z, z, z, one,
                     (1.0 - m0) * b, z, m0 * b, z, z, z, z, z, z, z, one, z, z])""",
            f"""    return _pack(M, [(1.0 - m0) * a, z, m0 * a, z, z, z, z, z, z, z, one{zz},
                     (1.0 - m0) * b, z, m0 * b, z, z, z, z, z, z, z, one{zz}, z, z])""")
    return s


def head(s, fam, doc):
    s = sub(s, 'FAMILY = "social_contagion_z20"', f'FAMILY = "{fam}"')
    s = sub(s, '"""Grey-box ODE: social_contagion z20 (19 parameters) = z16 with onboarding time per community. NUMPY + math only.',
            f'"""Grey-box ODE: {fam}: {doc} NUMPY + math only.\n\nBase: social_contagion_z20.')
    return s


a = head(d_chain(src, 2), "social_contagion_soc9a",
         "z20 with a two-stage disappointed pool D -> D2 -> pool (stage rate 2/tau_D): a waiting time before reconsidering.")
(ODE / "social_contagion_soc9a.py").write_text(a)
b = head(d_chain(src, 3), "social_contagion_soc9b",
         "z20 with a three-stage disappointed pool (stage rate 3/tau_D).")
(ODE / "social_contagion_soc9b.py").write_text(b)
print("ok")

a2 = (ODE / "social_contagion_soc9a.py").read_text()
c = sub(a2, 'FAMILY = "social_contagion_soc9a"', 'FAMILY = "social_contagion_soc9c"')
c = sub(c, '    ("m0", 0.33, 0.0, 0.9, False),', '    ("m0", 0.33, 0.0, 1.0, False),')
c = sub(c, "soc9a: z20 with", "soc9c: soc9a with the initial-leaver share m0 allowed up to 1; z20 with")
(ODE / "social_contagion_soc9c.py").write_text(c)
d = sub(c, 'FAMILY = "social_contagion_soc9c"', 'FAMILY = "social_contagion_soc9d"')
d = sub(d, "soc9c: soc9a with the initial-leaver share m0 allowed up to 1;",
        "soc9d: soc9c with its own initial-leaver share in b (m0_b);")
d = sub(d, '    ("tau_on_b", 9.0, 1.0, 60.0, True),\n]', '    ("tau_on_b", 9.0, 1.0, 60.0, True),\n    ("m0_b", 0.33, 0.0, 1.0, False),\n]')
d = sub(d, "    m0 = th[\"m0\"]\n", "    m0 = th[\"m0\"]\n    m0b = th[\"m0_b\"]\n")
d = sub(d, "(1.0 - m0) * b, z, m0 * b,", "(1.0 - m0b) * b, z, m0b * b,")
(ODE / "social_contagion_soc9d.py").write_text(d)
print("ok2")

e = sub(a2, 'FAMILY = "social_contagion_soc9a"', 'FAMILY = "social_contagion_soc9e"')
e = sub(e, "soc9a: z20 with", "soc9e: soc9a where waiting former members come back through the offer (D2 -> onboarding at k_ret phi); z20 with")
e = sub(e, '    ("tau_on_b", 9.0, 1.0, 60.0, True),\n]', '    ("tau_on_b", 9.0, 1.0, 60.0, True),\n    ("k_ret", 0.02, 1e-4, 1.0, True),\n]')
e = sub(e, "    dV1 = phi * new - k * V1\n", "    ret = th[\"k_ret\"] * phi * D2\n    dV1 = phi * new + ret - k * V1\n")
e = sub(e, '    dD2 = 2.0 / th["tau_D"] * (D - D2)\n', '    dD2 = 2.0 / th["tau_D"] * (D - D2) - ret\n')
(ODE / "social_contagion_soc9e.py").write_text(e)
print("ok3")
