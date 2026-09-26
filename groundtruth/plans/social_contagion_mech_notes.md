# social_contagion: mechanism-pair search (`gtlab/ode/social_contagion_mech.py`)

Date: 2026-09-26. Data: 3 runs, 811 ticks (hold_rec 120, pulse200_200 400, compose 291),
`sigma_proxy` = (59.6, 32.4). Lab: `scripts/ode_lab.py --budget 300 --starts 10 --nfev 60`,
tags AB, AC, BC (15 parameters fitted in every pair; inactive terms have no effect).

## Base (always on)
Per community $i$: loyal $L_i$, incentive-led $M_i$, queue $W_i$, initial leavers $X_i$,
pool $P_i = (N_i - L_i - M_i - W_i - X_i)_+$, observed $A_i = L_i + M_i + X_i$.
$\rho_i = s_i s + q A_i/N_i\ (+\text{cross}_i)$, $r_i = \rho_i/(1 + \rho_i/1.5)$,
$\dot W_i = r_i P_i - W_i/\tau_{on}$, $\phi = c/(c + 0.7)$,
$\dot L_i = (1-\phi)W_i/\tau_{on} - c_L L_i - k_{conv} c L_i$,
$\dot M_i = \phi W_i/\tau_{on} + k_{conv} c L_i - \kappa_M M_i$, $\dot X_i = -k_X X_i$.
Without B, $\kappa_M = c_L$: the incentive changes nobody's churn.

## Mechanisms
- A credibility: $\dot K_i = (1 - K_i)/\tau_K - k_A K_i W_i/N_i$, $K_i(0) = 1$; $\rho_i \to K_i \rho_i$.
- B incentive expectations: $\dot E = (c - E)/\tau_E$, $E(0) = 0$; $\kappa_M = c_L + k_B (E - c)_+$.
- C cross ties: $\dot T = (\beta - T)/30$; $\text{cross}_a = k_C T A_b/N_b$, $\text{cross}_b = k_C T A_a/N_a$.

## Results (first round, $k_{conv} \le 0.2$)

| model | LOO hold | LOO pulse | LOO compose | LOO mean | in-sample hold / pulse / compose | in-sample mean | cost | at bound |
|---|---|---|---|---:|---|---:|---:|---|
| min v2 (current) | 0.745 | 0.648 | 0.701 | 0.698 | 0.870 / 0.914 / 0.901 | 0.895 | 30.7 | k_br |
| AB | 0.812 (0.876, 0.748) | 0.753 (0.715, 0.791) | 0.793 (0.795, 0.792) | 0.786 | 0.903 / 0.926 / 0.896 | 0.908 | 28.4 | k_conv |
| AC | 0.788 (0.851, 0.725) | 0.567 (0.605, 0.530) | 0.677 (0.695, 0.659) | 0.677 | 0.886 / 0.735 / 0.753 | 0.791 | 256.9 | tau_E, k_C |
| **BC** | 0.882 (0.974, 0.791) | 0.823 (0.839, 0.808) | 0.778 (0.790, 0.766) | **0.828** | 0.881 / 0.898 / 0.898 | 0.892 | 34.9 | k_conv, k_C |
| l0b_lin | 0.712 | 0.535 | 0.707 | 0.651 | | | | |

(per-observable a, b in brackets.)

BC theta: $N_a$ 234.6, $N_b$ 186.0, $s_a$ 0.0069, $s_b$ 0.00153, $q$ 0.0198, $\tau_{on}$ 12.2,
$c_L$ 0.0080, $k_X$ 0.393, $k_{conv}$ 0.2 (bound), $m_0$ 0.145, $k_B$ 0.0479, $\tau_E$ 47.7,
$k_C$ 0 (bound); $k_A$, $\tau_K$ inert.
AB theta: $k_A$ 0.189, $\tau_K$ 65.5, $k_B$ 0.0481, $\tau_E$ 55.5, $k_{conv}$ 0.2 (bound), $\tau_{on}$ 19.5.

Eval-shaped 4,000-tick rollouts: all finite, 0.6 s each. Outside observed range $\pm5\%$:
AB 0.00 / 0.12 / 0.00 / 0.03, AC 0 everywhere, BC 0.00 everywhere (sustained/order/recovery/composition),
against 0.00 / 0.29 / 0.12 / 0.26 for min v2.

## Reading
- B is required: without it (AC) the fit cannot produce the post-incentive collapse; cost 9x,
  pulse fold 0.57. The memory form ($\tau_E \approx 50$, $k_B \approx 0.048$, so $0.096$/tick at
  $E - c = 2$) matches the 0.09/tick log-excess slope read off the raw data.
- C fits $k_C = 0$ in both pairs that carry it: no bridge effect between 0 and 0.6, same as
  $k_{br} = 0$ in v2. So BC is in effect "B alone".
- A buys a little in-sample (0.908 vs 0.892, cost 28.4 vs 34.9) but loses 0.04 LOO; the data
  cannot separate "A active" from "C active and invisible". We take BC (B-only dynamics) as the
  ship candidate: +0.13 LOO over min v2, clean eval.
- $k_{conv}$ at 0.2 in both B pairs: under B, fast conversion is harmless while paid, it only
  sets how many leave after the stop. Revision in flight: bound 0.7, tags AB2 / BC2.
