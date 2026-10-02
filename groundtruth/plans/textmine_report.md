# Text mining the briefs for code constructs (Mon Sep 28, no credits)

Question: the briefs read like descriptions written from simulator code. If so, every verb
("rejects", "retain", "shared", "only when", "after a delay") is a literal term: a min/max cap, a
cohort that keeps its parameters, a shared-resource budget, a lag. We translate each phrase into the
term it implies and check it against the families in `submissions/20260928-1810-final2`.

## Sources and gaps

- Text used: `kit/briefs.md` (identical to the organizer kit copy), `kit/PROMPT.md`, `README.md`,
  `FAQ.md`, `handout.md`.
- **The `documents` endpoint was never saved.** No file under `data/`, `tune/` or `plans/` holds a
  `documents` response (only the mock's placeholder). PROMPT.md says `context["documents"]` =
  "public static documents, each {id, title, text}", passed to `predict()` at runtime and readable
  for free (`client.documents(system)`). These may be the "operational documents" Gemma is meant
  to read, possibly with more specific wording than the briefs. **Action: one free `documents`
  call per system (no steps), save to `data/<sys>/documents.json`, re-run this mining on it.**
- Shipped families (from each `model.json`, not `manifest.json`, which is stale: it still lists
  hospital_queue_p3, power_grid perobs and reservoir_v8):

| system | family (mech) |
|---|---|
| epidemic | epidemic_y2 (A fatigue, C postponed gatherings) |
| market | market_y3 (A settlement tie-up, B risk capacity) |
| traffic | traffic_z8 (A route learning, B crew fatigue); **C spillback off** |
| power_grid | power_grid_w5 (A, B always on; mechanisms not named in the brief) |
| supply_chain | supply_chain_v8b (adaptive commitment, heat/wear, congested transport as base) |
| wildlife | median of v8h, wild9p, wild9m (A nursery, B food renewal) |
| reservoir | reservoir_str9 (all three return paths on) |
| ad_auction | perobs over v8b, v8b+s1, v8b+min (all three mechanisms always on) |
| social_contagion | median z20/z21 (B incentive expectations, C relationships) |
| hospital_queue | hospital_queue_hosp9 (p3 base + wait law; "A fatigue, B finite chairs") |

Legend: **Yes** = present as written; **Diff** = present but in a different form; **No** = absent;
**Tried** = in an older family, dropped (reference given).

## Preamble (applies to all ten)

| phrase | implied term | ours | note |
|---|---|---|---|
| "Reset randomizes only observables; unobserved quantities use a fixed deterministic initialization rule, which can distribute observed initial totals among hidden compartments" | $x_0$ = fixed split of each observed total; hidden stocks at fixed reference values | Yes (all `x0`) | one exception, supply_chain below |
| "There is no random prehistory" | all lags/memories start at the reference value | Yes | |
| "Exactly two of three candidate mechanisms operate" | exactly two switches on | **Diff** on ad_auction, reservoir, supply_chain (all three effectively on) | see rank 6 |

## epidemic (epidemic_y2)

| phrase | implied term | ours | suggested test |
|---|---|---|---|
| "three interacting age groups", "differ in contacts, severity and recovery" | 3x3 contact matrix, per-age $\gamma_i$, severity $s_i$ | Yes (CBASE fixed, $\gamma$, sev) | none |
| "Daily infection onsets" | observable = flow E→I, not prevalence | Yes (onset flow through 2-tick report delay) | none |
| "School closure changes where contacts occur" | contacts moved, not removed: $c_{00}(1-r)$, $c_{01} + 0.3 c_{00} r$ | Yes | fraction moved (0.3) is a fixed constant; free it |
| "masks reduce exposure" | $\beta (1 - m_e \cdot mask)$ | Yes | none |
| "vaccination uses a shared clinic workforce" | **dose budget**: doses/tick $= v \cdot N \cdot clinic$, split across groups by eligible $S_i$, capped by $S_i$; not a per-capita rate on $S_i$ | **Diff**: $v \cdot k_v \cdot S_i$ (per capita, $k_v$ = vac_eff 1.5 free) | refit y2 with $\dot S_i = -\,v N\,clinic\, w_i S_i/\sum_j w_j N_j S_j$ (capped at $S_i$); check p8 long hold and every vaccination hold > 150 ticks. Per-capita decays $S$ exponentially, a dose budget drains it linearly and then stops: different sustained tail |
| "hospital pressure reduces clinic availability" | $clinic = 1/(1 + k H/H_{cap})$ | Yes | MATH_LOG Fri: fitted $k$ once went to 0; check current value |
| "Clinical referrals can wait for beds" | waiting list WL, admission = min(demand, free beds) | Yes (smooth min) | hard min vs smooth: trivial |
| "Behavior, developing immunity and postponed gatherings may retain intervention history" | 3 switches | Yes (A, C on; B off) | none |
| "waiting lists and intervention histories start empty" | WL=0, fatigue=0, debt=0 at reset | Yes | none |
| vaccination_rate max **0.003** | rate is a fraction of population per tick | Diff (scaled by vac_eff 1.5) | pin vac_eff = 1 under the dose-budget form |

## market (market_y3)

| phrase | implied term | ours | suggested test |
|---|---|---|---|
| "Interest rate and transaction tax are fractions per trading period" | carrying cost $r \cdot$ (cash borrowed or value held) per tick; tax $x \cdot$ trade value per trade; linear in $r$ | Diff: cash drain $a_r (r/0.1)^{n_r}$, $n_r$ free (2.0) | pin $n_r = 1$ and compare LOO; a literal fraction is linear |
| "different reservation values" | trade only when buyer reservation − seller reservation > tax: hard gate at a tax threshold; price bounded by the two reservation values | Yes (sharp gate at $x_c$ ≈ 0.0444, width 0.0005); price floor $p_{lo}$ | the gate is a step: good sign for the hypothesis |
| "storage capacities" (per group) | producer and consumer warehouses each with a cap; production stops when producer storage is full | Diff: consumer warehouse $I_w$ only | add producer stock with cap; test on long zero-control holds (price hump) |
| "placement speeds" (per group) | two order-placement lags, one per group | Diff: one $\tau_v$ | low priority |
| "Production consumes working cash; final consumption returns revenue" | $\dot C = rev - cost$ | Yes | none |
| "a new policy does not cancel commitments already made" | orders in the pipeline keep executing after a control change | Yes ($v$ state) | none |
| "finite dealer books" | hard cap on dealer inventory | Diff (H store, soft) | low |
| 3 mechanisms (funding tie-up / risk capacity / momentum) | switches | Yes (A, B on) | none |
| "warehouses start half full with full working cash, dealer books neutral, orders and settlement empty" | $I_w = 0.5$, $C = 1$ | Yes | none |

## traffic (traffic_z8, mech A+B)

| phrase | implied term | ours | suggested test |
|---|---|---|---|
| "Finite approach, junction and exit buffers reject excess arrivals" | three hard caps, **base, not a mechanism**: admit = min(arrivals, room) at each stage | **Diff/No**: approach cap on the route total only ($1 - n/n_{max}$); exit buffer $K_3$ exists only in mechanism C, which is **off** in the shipped fit; no junction cap | rank 1 |
| "Vehicles already crossing keep occupying the shared junction until their committed work finishes and exit space opens. Thus one route can obstruct the other even under unchanged signals" | junction transfer $p_2 \to p_3$ blocked when the exit buffer is full, and blocked vehicles consume the **shared** junction capacity for both routes: $J_{free} = J - (p_{2a} + p_{2b})$ | **No** in base (only via C's fronts) | rank 1 |
| "Signal timing divides new crossing admissions" | share $\sigma$ of junction admissions to route a | Yes | none |
| "freight priority changes which waiting class gets service" | two classes (light, heavy); priority reorders service; heavy takes more junction work | **No** (freight_priority read but unused) | rank 10 |
| "Toll changes the arriving mix" | toll moves the light/heavy share, not only the total | Diff: toll scales total demand $e^{-k(toll-2.5)}$ | fold into rank 10 |
| "ramp metering changes admitted demand" | admitted $= ramp \cdot demand$ | Yes | none |
| "clearance effort shifts a shared crew from intersection operation toward downstream exits" | **transfer**: junction capacity $\times (1 - k_j\,clr)$, exit capacity $\times (1 + k_x\,clr)$ | **Diff**: exit boost only; fit sends the boost toward 0 (traffic_z notes) | rank 2 |
| "Lane closure affects different sections unequally" | per-section (approach/junction/exit) closure sensitivity | Diff: per-route exit only ($L_a$, $L_b$, $n_l$) | test lane on junction and approach too |
| "waiting approach drivers may divert" | abandonment $-d\,p_1^2/n_{max}$ or $-d\,p_1$ | **Tried** (traffic_z6) | re-test only after rank 1 |
| "Route learning, crew fatigue/switching costs and persistent spillback fronts" | 3 switches | Yes (A, B on) | once exit buffer is base, recheck A+C and B+C |
| "Reported speed combines observed completed journey times with current stopped and moving class mix" | speed = blend of (lagged) completed trip time and an occupancy/class-mix term | Diff: occupancy law with a lag | low |
| "Roads and crossings start empty" | $p = 0$ at reset | Yes | none |

## power_grid (power_grid_w5)

| phrase | implied term | ours | suggested test |
|---|---|---|---|
| "reserve dispatch requests additional supply **in power units**" | requested MW = $u$ (coefficient 1) | Diff: $c_r = 1.36$ free | pin $c_r = 1$, let the limit $p_l + p_r ic$ absorb the rest |
| "Supply requests can be limited by available physical resources" | delivered = min(request, available) | Yes (smooth min, w5) | none |
| "Reserve resources differ in power, duration and thermal response, and share a charging connection" | ≥2 reserve units: fast/energy-limited (battery: power cap, energy stock) + slow thermal (ramp lag); one shared charging limit | Diff: one lag $R_s$ + one stock $E$ | rank 8 |
| "Dispatch allocates supply according to operating cost" | merit order: cheapest unit first, then next | No | part of rank 8 |
| "each load warms while off, cools while on, and switches at separate upper and lower temperature limits" | hysteresis thermostat population; price shifts both limits | Diff: damped oscillator ($w$, $\zeta$, kick) | **Tried** (pg9 temperature grid; not shipped) |
| "Every reset starts with the same asynchronous population at reference price **0.8**" | reset state = steady population at $p = 0.8$ | Yes ($D_{ref} = D(0.8)$) | none |
| "Interconnector setting opens remote delivery capacity, whose temperature depends on recent flows" | line temperature $\dot T_{ic} = (flow_{ic} - T_{ic})/\tau$; capacity derated when hot | Diff: $Th$ = lag of **load**, used only in renewable share | rank 7 |
| "Renewables can be curtailed at that connection" | share $\le$ connection headroom | Diff (share scales with $ic$) | part of rank 7 |
| "governors respond to frequency with finite response times and output limits" | $\dot G = a(\text{clip}(-k f_{dev}, \pm G_{lim}) - G)$ | Yes ($G_{lim} = 40$ fixed) | free $G_{lim}$; the p8 frequency leak sits near a limit |
| "Charging allowance limits grid power available for refilling reserves" | $P_{ch} = p_{ch}\cdot chg$ | Yes | none |
| frequency "in Hz" | nominal 50 | Yes | none |

## supply_chain (supply_chain_v8b)

| phrase | implied term | ours | suggested test |
|---|---|---|---|
| "Orders withdraw only available stock" | dispatch = min(order, stock) | Yes | none |
| "product mix selects new production and requested dispatch, without rewriting old batches" | class split applies to new flow only | Yes (class-resolved conveyors) | none |
| "Conveyors retain their destination and travel commitment after a rush change" | **cohort delay**: travel time fixed at departure; a rush/lead change affects new departures only | **Diff**: $k_q = k_{q0}/(1 + k_l\,lead)$ acts on everything already on the conveyor | rank 9 |
| "Rush handling bypasses treatment for new primary-line departures" | rush → skip a treatment stage (shorter path, no treatment capacity) for new departures | No (no treatment stage) | part of rank 9 |
| "The two intakes share forward transport" | one transport capacity for both classes | Yes (congested transport $q_m$) | none |
| "the second intake and treatment share cooling, while the primary intake, receiving and maintenance share drive service" | shared-resource budgets: receiving + maintenance + primary intake $\le$ drive capacity | No (maintenance costs nothing) | combine with next row |
| "Maintenance restores treatment activity but takes utility and productive treatment time" | maintenance lowers throughput while on; resets wear | Diff: resets wear only | **Tried** ("production maintenance loss, fit 0", supply_chain_min notes) |
| "return-space overflow leaves as secondary-grade goods counted in shipments" | rework loop with finite return buffer; overflow adds to shipments | No | low; mechanism A (rework) only |
| 3 mechanisms (congested transport+rework / heat-wear / adaptive commitments) | exactly two | **Diff**: all three effectively on | rank 6 |
| "Internal buffers and conveyors start empty" | $Q = 0$ at reset | **Diff**: $x_0$ puts shipments/2 on each class conveyor | compare first 10 ticks of shipments on every run with $Q_0 = 0$ (free) |

## wildlife (median v8h / wild9p / wild9m)

| phrase | implied term | ours | suggested test |
|---|---|---|---|
| "Hunting requests prey harvest" | harvest = min(quota, available prey); quota in animals per tick (coefficient 1) | Diff: $H\,q\,P^2/(P^2+P_h^2)$, $H$ free (≈2) | pin $H = 1$ with a smooth min on availability |
| "habitat protection changes shelter and resource renewal, **especially in the north**" | habitat weight per region, larger north | Diff: renewal weight $b_h$ shared; exposure death per region | **Tried** in part (bhr switch); retest with north weight > south |
| "Corridor access controls new interregional journeys; already travelling animals can still arrive" | transit pools; corridor gates departures only | Yes | none |
| "open pasture, mixed cover and sheltered browse differ in feeding, hunting exposure and predation" | 3 patches per region with patch-specific harvest exposure and predation | No (single pool) | low identifiability from totals |
| "young animals compete for nursery food" | juvenile stage with density-dependent survival (a delay) | Diff: instantaneous crowding | **Tried** (MATH_LOG: juvenile stage collapses to instant crowding) |
| "Food renewal shares a finite resource" | resource stock $R$ | Yes (B) | none |
| "arrivals compete for settlement space" | arrival survival $s_v (1 - P/(K s_i))^+$ | **No** (C "not modelled") | rank 5 |
| predation (prey deaths from predators) | $-a P Q/(P + h)$ | No | **Tried**, invisible in data (wildlife_min notes) |
| "Regional totals alone do not identify patch occupancy, juvenile condition or animals in transit" | the three hidden stores are patch shares, juveniles, transit pools | Diff (transit only) | none |

## reservoir (reservoir_str9)

| phrase | implied term | ours | suggested test |
|---|---|---|---|
| "Release and irrigation **request** water per tick" | delivered = min(release + irrigation, available) | Yes | none |
| "Withdrawal depth selects shallow versus deep release; irrigation draws near the surface" | outlet quality = mix of layer qualities weighted by depth; irrigation takes surface water | Diff: depth drives a single deficit $D$ | low |
| "Aeration increases oxygen transfer and vertical mixing" | two effects: DO up, layers mixed | Diff (D relaxes with $1-aer$; $q_{aer}$ ±0.03) | none |
| "Seasonal river supply brings nutrients" | nutrient load $\propto$ inflow(t) (period 67.75) → quality seasonal | No | residual amplitude ≤ 0.003 (str9 notes): skip |
| "Light, nutrient availability, decomposition and oxygen interact" | algae $\dot A = \mu\,light\cdot N/(N+K) A - \ldots$ | No | low (quality weak observable but small amplitude) |
| "Biomass can foul withdrawal screens, restricting actual discharge; flushing and aeration remove some of that attached material" | fouling stock $F$: grows with biomass, removed by high release (flushing) and aeration; outlet cap $\times (1 - k F)$ | **No** (in the original `reservoir.py` template only) | below the cut; test on a low-aeration long hold |
| "Groundwater, irrigated land and deposited material may return water or contaminants after a delay" | 3 candidate return paths (2 active) | **Diff**: all three present (gw inflow, irrigation return $R$, slow stock $N$) | rank 6 |
| "internal layers start from the same reference profile" | $D_0$, $N_0$ fixed | Yes ($d_0$, $n_0$) | none |
| "quality index from zero to one", "A tick is one operating day" | bounds [0,1] | Yes | none |

## ad_auction (perobs over v8b variants)

| phrase | implied term | ours | suggested test |
|---|---|---|---|
| "per-tick budget" | spend $\le$ cap exactly | Yes (+ post rule) | none |
| "Broader targeting contains narrower audiences" | nested segments: breadth adds lower-value outer segments | Diff: $A = N\,breadth$, win $\div (1 + k_w (b - 0.1))$ | low |
| "members can be in different stages of attention and purchase" | funnel stages | Yes (P preparation, Q1, Q2) | none |
| "started purchases remain committed and compete for limited fulfillment work" | fulfilment queue with capacity | Yes ($F$, idle capacity $K$) | none |
| "Different audiences require different amounts of fulfillment work" | work per conversion depends on breadth: completions $= F/w(b)$ | No ($F$ fixed in conversions) | test $F(b) = F_0/(1 + k_f (b - 0.1))$ on existing breadth changes |
| "Converted customers take time to become available again" | conversions remove people from the reachable pool for $\tau_c$ | No (only exposure removal per impression) | add $\dot Y = conv/N - Y/\tau_c$, $A \propto (1 - X - Y)$ |
| "Rival campaigns may move capital / repeated exposure may remove reachable people / broad introduction may change later follow-up" | exactly two of: $k_r$, $k_d X$, $g_0 + (1-g_0) P$ | **Diff**: all three always on | rank 6 |
| "people are available and unprepared, with no pending purchases or exposure recovery" | $X = P = Q = 0$ at reset | Yes | none |

## social_contagion (median z20/z21, mech B+C)

| phrase | implied term | ours | suggested test |
|---|---|---|---|
| "Seeding funds outreach" | recruitment effort $\propto$ seeding | Yes | none |
| "bridge outreach allocates effort between local recruitment and introductions" | local $= s(1-\beta)$, bridge $= s\beta$ | Yes | none |
| "Relationship-led, incentive-led and deliberative audiences mix differently" | 3 audience types, mix per community | Yes (L, M, X; $k_{inc}$ per community) | none |
| "Interested people must complete onboarding through a workforce **also needed by existing members**" | onboarding capacity $= W_f - c\,A$ (members consume staff): onboarding slows as membership grows | **No** ("no workforce ceiling": a fixed ceiling parked at its bound) | test the membership-loaded form $k_{on}(1 - A/A_w)^+$, not a fixed cap |
| "promises accompany waiting cohorts" | credibility cost per waiting cohort | Yes (A, off in the shipped pair) | none |
| "Disappointed former members need time before reconsidering" | refractory pool $D$ with $\tau_D$ | Yes | none |
| "Credibility, incentive expectations and cross-community relationships" | 3 switches | Yes (B, C) | none |
| "no paid promises; other queues begin empty" | V = 0 at reset | Yes | none |

## hospital_queue (hospital_queue_hosp9)

| phrase | implied term | ours | suggested test |
|---|---|---|---|
| "Diagnostic allocation divides shared staff" | assessment work $\propto D$, treatment work $\propto 1 - D$ | Yes | none |
| "urgent priority changes new service admissions" | admission order by class; urgent cases carry more work | **No** (U read, unused) | part of rank 4 |
| "Routine, urgent and elective cases need different assessment and treatment work" | per-class work; elective scheduling adds low-work cases | Diff (one class) | part of rank 4 |
| "occupy finite chairs or beds until completion"; "A completed assessment may hold its chair when the treatment queue is full" | blocking after service | Yes (A2 waits for beds on chairs) | none; note hosp9's "mechanism B = finite chairs" is base per the text |
| "Staffing and overtime change work delivered, not completed patient counts directly" | staff act on rates, not on counts | Yes | none |
| "Overtime can create later fatigue" | fatigue stock driven by overtime | Yes (A) | none |
| "staff changes can require orientation before staff are fully effective" | **handover**: effective staff lags staffing increases, $\dot S_e = (S - S_e)/\tau_o$ when $S > S_e$ | **No** in hosp9; **Yes** in hospital_queue_y3 (MATH_LOG: "orientation is active, staffing 7 → 20 gives 9–12/tick, not ≥ 18") | rank 3 |
| "Follow-up capacity diverts shared staff to a finite outside program that can prevent delayed returns after discharge" | $eff \times (1 - k_f\,Fu)$; returns $= \rho \cdot$ lagged discharges $\times (1 - Fu\cdot cap)$ | **No** (Fu unused) | rank 4 (**Tried** once: +0.013, lost exam) |
| "Patients waiting can deteriorate or leave" | leave $k_l W$; deterioration routine → urgent (more work) | Diff (leave only) | low |
| "overflow is referred elsewhere rather than stored in an invisible queue" | hard cap on the queue, excess dropped | Yes ($q_{cap}$ gate, post rule 333) | none |
| "Fatigue, handover and returning case mix are three possible mechanisms; exactly two apply" | two of three on | **Diff**: only fatigue present | ranks 3–4 |
| "services and the follow-up program start empty" | A, T, program = 0 | Yes | none |

## Numbers in the text that can pin parameters

| system | number | pin |
|---|---|---|
| epidemic | vaccination_rate ≤ 0.003, a fraction per tick | vac_eff = 1 under a dose budget |
| market | rate, tax "fractions per trading period" | linear carrying cost ($n_r = 1$) |
| market | warehouses half full, full cash, books neutral | already used |
| power_grid | reference price 0.8; reserve in power units; 50 Hz | $c_r = 1$; $D_{ref} = D(0.8)$ (used) |
| wildlife | hunting_quota "requests prey harvest" (0–8, animals/tick) | $H = 1$ |
| hospital | elective_scheduling 0–20 added to arrivals | coefficient 1 (used) |
| reservoir | release/irrigation "request water per tick" | coefficient 1 (used) |
| supply_chain | order_quantity 0–80 per tick; efforts 0–1.5 with 1 = nominal | nominal capacity at effort 1 |
| all | pulse = recovery + α(pulse_ref − recovery), α ∈ [0.7, 1] | eval corners; already in design |

## Top 10 missing terms, ranked

Score = strength of the wording (base statement > "may") × absence from the shipped family ×
likely effect on the sequence categories. All ten can be screened on data we already own.

| # | system | term the text implies | why ranked here | first test (free) |
|---|---|---|---|---|
| 1 | traffic | **Finite exit and junction buffers as base**: $p_2 \to p_3$ blocked when the exit is full, and blocked vehicles hold the **shared** junction capacity $J - (p_{2a}+p_{2b})$ for both routes | Stated twice as base physics ("Finite ... buffers reject", "one route can obstruct the other even under unchanged signals"); the shipped fit runs A+B, so there is no exit buffer at all | z8 with $K_3$ blocking + shared junction occupancy always on, fronts ($r_{up}$, $r_{dn}$) still the C switch; LOO + p7 long-hold fold |
| 2 | traffic | **Clearance is a transfer**: junction $\times(1 - k_j\,clr)$, exits $\times(1 + k_x\,clr)$ | "shifts a shared crew from intersection operation toward downstream exits"; a boost-only term fits to ≈ 0 exactly when the two effects cancel on the data | add $k_j$ to z8 (with rank 1); check clearance authority in `control_audit` afterwards |
| 3 | hospital | **Orientation lag on staffing increases** (handover mechanism) | Brief mechanism; MATH_LOG Mon says orientation is active (staffing 7 → 20 without overtime gives 9–12/tick, not ≥ 18); shipped hosp9 has no handover at all | port y3's orientation state into hosp9 (wait law unchanged); LOO + exam |
| 4 | hospital | **Follow-up diverts staff; returns after discharge; urgent_priority** | hosp9 carries one of the three mechanisms, two are active; Fu and U are read but unused | returns = $\rho$ · discharges lagged by $\tau_r$ · $(1 - Fu)$, staff · $(1 - k_f Fu)$; fit together with rank 3 as pairs (fatigue+handover, fatigue+returns) |
| 5 | wildlife | **Settlement competition on arrivals**: arrivals survive with $s_v(1 - P/(K s_i))^+$ | Mechanism C is not modelled; pairs AC and BC never fitted; only acts when the corridor is open (composition runs) | add C to v8h, fit AB, AC, BC on all runs; LOO |
| 6 | ad_auction, reservoir, supply_chain | **Exactly two of three**: one mechanism in each shipped family must be zero | The text says two apply; carrying a third lets it absorb the wrong response and extrapolate it in unseen schedules | refit each family three times with one mechanism removed (ad: $k_r$ / exposure $X$ / preparation $P$; reservoir: gw / irrigation return / slow stock; supply: congested transport / heat / commitment); keep a pair only if it ties or wins LOO |
| 7 | power_grid | **Interconnector temperature from recent interconnector flow**, derating remote capacity and curtailing renewables | Stated as base; w5's thermal state lags the load, not the line flow | $\dot T_{ic} = (flow_{ic} - T_{ic})/\tau$, remote cap $\times(1 - k(T_{ic} - T_0)^+)$, share capped by headroom; refit on p8 (the frequency leak) |
| 8 | power_grid | **Reserve in power units ($c_r = 1$) + two reserve units** (energy-limited fast unit, ramp-limited thermal unit) with merit order | "in power units" pins a coefficient the fit put at 1.36; "differ in power, duration and thermal response" names two units | pin $c_r = 1$ first (one refit); then a second reserve state with its own lag and energy stock |
| 9 | supply_chain | **Conveyor cohorts keep their travel time**: lead/rush changes act on new departures only | "retain their destination and travel commitment after a rush change"; ours retimes goods already in transit; lead_time_buy currently has ≈ 0 authority | replace $k_q(lead)$ on stock with an age-structured conveyor (speed set at entry: two parallel lanes at the old and new speed) |
| 10 | traffic | **Two vehicle classes**: toll moves the light/heavy mix, freight priority orders service, heavy vehicles take more junction work and lower speed | freight_priority is unused; toll works on the total only | split arrivals by a toll-dependent heavy share $h(toll)$; junction work $1 + w_h h$; freight priority weights service; only after 1–2 |

Worth one line each but below the cut: epidemic dose-budget vaccination (strong wording,
"shared clinic workforce"; matters on vaccination holds > 150 ticks, the sustained category, and
is a cheap refit: it would sit at rank 6–7 if the p8 epidemic run shows a vaccination tail
residual); social onboarding slowed by membership load; ad_auction conversions leaving the pool
plus breadth-dependent fulfilment work; reservoir screen fouling (the head law already explains the
pulse; keep for a low-aeration long hold); supply_chain $x_0$ with empty conveyors (text says
empty, ours seeds them).

## Optional paid checks (need Dylan's go; none bought)

| system | run | credits | separates |
|---|---|---|---|
| traffic | recovery hold 40, then clearance 1 → 0 at everything else fixed, 40 + 40 | 120 | rank 2 (junction slowdown vs exit boost) |
| hospital | staffing 7 → 20 with overtime 0, follow-up 0, hold 80; then follow-up 1 for 80 | 160 | ranks 3–4 |
| wildlife | corridor 1 at recovery habitat, 150 | 150 | rank 5 |

Budget check first (free): the last `remaining` readings in the ledgers are 160–780 per system
and p8 was bought after some of them.
