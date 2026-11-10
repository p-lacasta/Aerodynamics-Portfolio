# Advanced Vehicle Aerodynamics – Portfolio code

Python code for my coursework portfolio in **Advanced Vehicle Aerodynamics** at **Oxford Brookes University**. Each folder holds one section of the portfolio, and the scripts produce the figures and tables used in it.

---

## Repository layout

```
Aerodynamics-Portfolio/
├── S1 - Vehicle performance evaluation/
│   ├── src/
│   │   ├── params.py      # all vehicle, aero, tyre, powertrain and crosswind inputs
│   │   ├── model.py       # force models: engine, drag, lift, rolling resistance, traction, dv/dt
│   │   ├── simulate.py    # studies: power balance, 0–200 run, 80–120 elasticity, Cd/CL and yaw sensitivity
│   │   └── plots.py       # makes every Section 1 figure
│   └── figures/           # output (git-ignored)
├── S2 - Boundary layer evaluation/
│   ├── Boundary_Layer_v4.py
│   ├── Armfield Wind Tunnel Inlet Contraction and Working Section Profile.xlsx   # tunnel geometry
│   └── plots/             # output (git-ignored)
├── S3 - Body shape evaluation/
│   ├── body_shape.py
│   ├── datasets/DrivAer experimental/   # input force data (git-ignored, see below)
│   ├── figures/           # output (git-ignored)
│   └── Body_shape_results.xlsx          # tables exported by body_shape.py
└── .gitignore
```

---

## Setup

Written and run with Python 3.14 on Windows. Install the packages with:

```bash
pip install numpy scipy pandas matplotlib openpyxl
```

`openpyxl` is needed because pandas reads and writes the `.xlsx` files.

---

## Section 1 – Vehicle performance

**Vehicle:** CUPRA Formentor VZ5 (2.5 TSI, 7-speed dual-clutch). Every input is in `src/params.py` with its unit and source.

**Method:** a time-domain longitudinal model integrated with explicit Euler (`DT = 0.01 s`).

- Tractive force = engine torque (interpolated from the datasheet map) × gear ratio × final drive × driveline efficiency ÷ wheel radius, capped at μ·N for tyre grip.
- Resistance = aero drag ½ρC<sub>d</sub>A·V² + rolling resistance C<sub>rr</sub>·(mg − lift).
- Equation of motion: dv/dt = (F<sub>t</sub> − F<sub>res</sub>) / (k<sub>m</sub>·m), with k<sub>m</sub> the rotating-inertia factor.
- Upshift when the next gear gives at least as much tractive force, with the rev limit as a hard cap. Each shift has a torque gap of `SHIFT_TIME`.
- **Yaw:** a crosswind `w` perpendicular to the direction of travel gives airspeed √(v² + w²) and yaw angle ψ = atan(w/v). C<sub>d</sub>(ψ) and C<sub>L</sub>(ψ) are interpolated from the `PSI`, `CD_PSI` and `CL_PSI` tables.

**Run** (from the repo root):

```bash
python "S1 - Vehicle performance evaluation/src/simulate.py"   # prints the results tables only
python "S1 - Vehicle performance evaluation/src/plots.py"      # prints the tables and saves all figures
```

**Outputs:**
- Console: v<sub>max</sub> per gear (and whether power or the rev limit sets it), unrestricted and limited v<sub>max</sub>, the speed above which drag dominates, 0–100 and 0–200 km/h times, 80–120 km/h time in each usable gear, and sensitivity tables for C<sub>d</sub>, C<sub>L</sub> and crosswind (each as % change from the baseline).
- `figures/`: `engine_curves`, `resistance_forces`, `power_curves`, `accel_0_200`, `accel_baseline`, `elasticity_80_120`, `yaw_coefficients`, `cd_sensitivity`, `cl_sensitivity`, `yaw_sensitivity`.

**Main assumptions:** level road; ISA sea-level air density; constant C<sub>rr</sub>; constant friction coefficient μ; no wheel slip below the grip limit; lumped rotating-inertia factor; 250 km/h limiter; C<sub>d</sub> taken in body axes along the direction of travel. See `params.py` for the values and which ones are assumed.

---

## Section 2 – Boundary layer in the OBU wind tunnel

**Script:** `Boundary_Layer_v4.py`. Runs U<sub>∞</sub> = 10, 20, 30 and 40 m/s in the working section.

**Method:**
1. Read the contraction and working-section profile from the Armfield spreadsheet and resample it on a ~1 mm grid along the wall (PCHIP interpolation).
2. Core velocity from continuity: U(l) = U<sub>ws</sub>·A<sub>ws</sub>/A(l).
3. **Laminar:** Thwaites' method from the inlet, θ² = 0.45ν/U⁶ ∫U⁵ dl. The shape factor H(λ) comes from a curve fit to Thwaites' table (White, *Viscous Fluid Flow*), and δ uses the Blasius ratio δ/θ ≈ 7.4.
4. **Transition:** where Re<sub>θ</sub> reaches the value equivalent to flat-plate Re<sub>x</sub> = 5×10⁵ (Re<sub>θ</sub> ≈ 470).
5. **Turbulent:** 1/7th power law (δ = 0.16x/Re<sub>x</sub><sup>1/7</sup>, δ* = δ/8, θ = 7δ/72, H = 9/7) from a virtual origin that matches θ at transition.
6. Displacement effect: the loss in effective working-section area at the exit, (1 − (s − 2δ*)²/s²).

**Run:**

```bash
python "S2 - Boundary layer evaluation/Boundary_Layer_v4.py"
```

**Outputs:** a console table (contraction ratio, transition position, θ at the start of the working section, δ and δ* at its end, % area loss), and `plots/boundary_layer_summary.png`: U, λ, δ, δ* and θ along the wall, with the contraction and working section shaded.

---

## Section 3 – Body shape, aero balance and stability

**Script:** `body_shape.py`

**Input data:** DrivAer experimental force and moment data for the fastback, notchback and estateback (squareback) variants, one CSV per body in `datasets/DrivAer experimental/` (`FASTBACK_Force.csv`, `NOTCHBACK_Force.csv`, `ESTATEBACK_Force.csv`). The `datasets/` folder is git-ignored, so these files must be added locally before running. The data are from a 1:4 scale model (`SCALE = 4`); frontal area and wheelbase are scaled to full size.

**Method:**
- Mean C<sub>d</sub>, C<sub>L</sub> and pitching-moment coefficient C<sub>m</sub> per body; axle lift coefficients C<sub>L,f</sub> = C<sub>L</sub>/2 + C<sub>m</sub> and C<sub>L,r</sub> = C<sub>L</sub>/2 − C<sub>m</sub>; aero balance = C<sub>L,f</sub>/(C<sub>L,f</sub> + C<sub>L,r</sub>).
- Axle normal loads with speed from the static weight distribution minus aero lift on each axle.
- Change in rear lift needed for the aero lift split to match the static weight split.
- Fastback + spoiler variant: coefficient changes set in `SPOILER_DELTA`, applied to the measured fastback values.

**Run:**

```bash
python "S3 - Body shape evaluation/body_shape.py"
```

**Outputs:**
- `figures/`: `A_coefficients_bar`, `B1_axle_load_change_vs_speed`, `B2_front_load_share_vs_speed`, `E_cl_r_correction`, `F_spoiler_coefficients`, `G1_spoiler_axle_load_change_vs_speed`, `G2_spoiler_front_load_share_vs_speed`.
- `Body_shape_results.xlsx`: one sheet per table or figure (aero balance, dynamic loads, plotted data).

---

## Section 4 – NACA0015 aerofoil

To be added: comparison of wind tunnel and CFD results.

---

## Conventions

- SI units throughout. Speeds are in m/s inside the code and plotted in km/h where that reads better.
- Figures share one style (viridis colours, legends below the axes, 300 dpi PNG) so they look consistent across sections of the portfolio.
- Generated outputs (`figures/`, `plots/`) and input datasets are not tracked by git. Rerun the scripts to regenerate them.

## Author

Pau Lacasta, Oxford Brookes University
