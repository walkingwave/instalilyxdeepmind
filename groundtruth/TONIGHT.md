# Tonight (Sep 23) - 3 uploads per system

Budget use tonight: **240 steps per system** (hold_rec 120 + hold_pulse 120). Both are part of the
planned p1 phase, so nothing gets bought twice later. 1,760 per system stays untouched.

## Upload 1 - persistence (0 credits)
Upload `groundtruth\submissions\20260923-2157-persist\submission.zip` in the **Public development** tab.

## Setup (PowerShell, once)
```powershell
cd "$HOME\Documents\Projects\instalily-hackathon\instalilyxdeepmind"
git push -u origin main          # the groundtruth/ commit is already made locally
cd groundtruth
py -3.12 -m venv .venv           # any python >= 3.10 works
.\.venv\Scripts\Activate.ps1
pip install numpy scipy httpx
$env:GROUNDTRUTH_GATEWAY_URL = "https://gt-gateway-wavddee32q-uc.a.run.app"
$env:GROUNDTRUTH_KEY = "<your learning key>"      # never put it in a file
```

## Probe (costs 240 steps per system, 2,400 total)
```powershell
# 1. free: see the exact cost, touches nothing
python -m gtlab.cli collect --all --phase p1 --only hold_rec,hold_pulse --dry-run

# 2. free: 20 extra resets per system (initial-condition spread, no steps charged)
$env:GT_ALLOW_REAL = "1"
python -m gtlab.cli resets --all --n 20 --real

# 3. PAID: 240 steps per system, hard-capped
$env:GT_ALLOW_SPEND = "1"
python -m gtlab.cli collect --all --phase p1 --only hold_rec,hold_pulse --spend --max-steps 240 --yes
Remove-Item Env:GT_ALLOW_SPEND
```
If it crashes, rerun the same command. It resumes and never re-buys a step.

## Send the data back
Zip the `data` folder and attach it in the chat:
```powershell
Compress-Archive -Path data -DestinationPath data-probe.zip -Force
```
Also commit it: `git add data; git commit -m "probe data"; git push`. That data can't be bought again.

## Upload 2 and 3
Fit the models, run the contract checker, and build the ZIPs:
- **Upload 2:** lag model fit on the probe (relax-to-equilibrium, linear features).
- **Upload 3:** after upload 2 scores show up, per system: keep whichever of upload 1 or 2 scored higher.

Last upload before **23:30** counts for today's sweep. Slots reset at midnight.
