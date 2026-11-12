# ME401 Rotary Hybrid Range-Extender Simulation

0D thermodynamic simulation of an epitrochoidal Wankel rotary hybrid range-extender (`gasoline + H2`), compared against an equivalent Atkinson-cycle reciprocating piston baseline across 2,000–7,000 RPM.

## Quick Start (Any Computer with Python 3.11+)

### Windows
Double-click **`run_windows.bat`** inside `Project Code\`, or run from PowerShell:
```powershell
.\run_windows.bat
```
On first launch, the script automatically creates a local `.venv`, installs `requirements.txt`, and launches `main.py`.

### macOS / Linux
Open a terminal inside `Project Code/`:
```bash
chmod +x run_mac_linux.sh
./run_mac_linux.sh
```

### Manual Run (if you manage your own Python environment)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Interactive Inputs & CLI Options

Running `main.py` in a terminal prompts you for the three primary inputs (press **Enter** to accept any default shown in brackets):
1. **Rotor radius `R` [mm]** (default `89.00`)
2. **Fuel mixture — Hydrogen mass fraction `x_H2`** in `0.00–0.30`, balance gasoline (default `0.10`)
3. **Power demand — Target generator electrical load `P_req` [kW]** (default `15.0`)

To skip prompts and run with defaults or CLI flags:
```bash
python main.py --batch
python main.py --batch --r-mm 92.0 --h2 0.20 --target-kw 18.0
```

## Outputs
All generated files are saved inside `Project Code/`:
- `output/speed_sweep_results.csv`, `parametric_cr_recess.csv`, `parametric_h2_blend.csv`, `optimal_cycle_trajectory.csv`, `simulation_summary.json`
- `figures/*.png` and `figures/*.pdf` comparative plots

## How to Share This Folder
Zip the `Project Code/` folder (or push it to Git) **excluding** `.venv/`, `__pycache__/`, `output/`, and `figures/`. The `.gitignore` file already excludes these automatically for Git. Anyone with Python 3.11+ can extract the folder and run `run_windows.bat` or `run_mac_linux.sh`.

