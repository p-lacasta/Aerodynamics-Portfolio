import numpy as np
import params as p
from model import (engine_rpm, drag_force, rolling_force, grade_force,
                   resistance_force, tractive_force, acceleration)

# --- SETTINGS ---
DT         = 0.01          # [s] time step (explicit Euler)
T_MAX      = 60            # [s] safety stop, in case a target speed is never reached
DV_KMH     = 0.01          # [km/h] speed sweep resolution for power_curves()
N_GEARS    = len(p.GEAR_RATIOS)


# --- HELPERS ---

def time_to_speed(log, v_kmh):
    """First time the run reaches v_kmh [s], interpolated between log points."""
    v_ms = v_kmh/3.6
    if log["v"].max() < v_ms:
        return np.nan                 # speed never reached
    return np.interp(v_ms, log["v"], log["t"])    # v only increases, so this is safe

def should_upshift(v, gear):
    """Shift at the optimal point: as soon as the next gear gives at least as much tractive force.
    In gears 1-4 that point lies beyond the rev limit, so the rev limit is kept as a hard cap."""
    if gear == N_GEARS:                              # already in top gear
        return False
    if engine_rpm(v, gear) >= p.REV_LIMIT:           # engine cannot go past the rev limit
        return True
    return tractive_force(v, gear+1) >= tractive_force(v, gear)


# --- 1. POWER CURVES (steady state, no time integration) ---

def wheel_power(v, gear):
    """Power at the wheels [W] at full throttle. NaN outside the torque map:
    steady state, so no clutch slip below 2000 rpm."""
    rpm = engine_rpm(v, gear)
    P = tractive_force(v, gear) * v
    return np.where((rpm >= p.RPM[0]) & (rpm <= p.REV_LIMIT), P, np.nan)

def vmax_in_gear(v, P_avail, P_req):
    """Highest speed where available power covers the requirement [m/s], and why it stops:
    'power' if the curves cross, 'rev limit' if the gear runs out first."""
    ok = P_avail >= P_req                       # NaN compares as False
    if not ok.any():
        return np.nan, "-"
    i = np.where(ok)[0].max()
    if i+1 < len(v) and not np.isnan(P_avail[i+1]):
        e0, e1 = P_avail[i] - P_req[i], P_avail[i+1] - P_req[i+1]
        return v[i] + (v[i+1] - v[i]) * e0/(e0 - e1), "power"
    return v[i], "rev limit"

def power_curves():
    # Sweep up to the speed where top gear hits the rev limit
    v_top = p.REV_LIMIT * 2*np.pi/60 * p.WHEEL_RADIUS / (p.GEAR_RATIOS[-1]*p.FINAL_DRIVE[-1])
    v = np.arange(0, v_top*3.6 + DV_KMH, DV_KMH) / 3.6

    res = {"v": v,
           "P_drag": drag_force(v)*v,
           "P_roll": rolling_force(v)*v,
           "P_grad": grade_force(v)*v,
           "P_tot":  resistance_force(v)*v,
           "P_gear": {g: wheel_power(v, g) for g in range(1, N_GEARS+1)}}

    # V max per gear, overall (unrestricted) and with the limiter
    res["vmax_gear"] = {g: vmax_in_gear(v, res["P_gear"][g], res["P_tot"]) for g in res["P_gear"]}
    res["vmax_best_gear"] = max(res["vmax_gear"], key=lambda g: np.nan_to_num(res["vmax_gear"][g][0]))
    res["v_max"] = res["vmax_gear"][res["vmax_best_gear"]][0]
    res["v_max_lim"] = min(res["v_max"], p.V_LIMIT)

    # Drag share: first speed where P_drag > 50% of P_total
    share = np.divide(res["P_drag"], res["P_tot"], out=np.zeros_like(v), where=res["P_tot"] > 0)
    i = np.argmax(share >= 0.5)
    res["drag_share"] = share
    res["v_aero"] = np.interp(0.5, share[i-1:i+1], v[i-1:i+1])
    return res


# --- 2. ACCELERATION 0-100 (and on to 200) WITH SHIFTS ---

def accel_run(v_target=200/3.6):
    t, v, x, gear = 0.0, 0.0, 0.0, 1
    shift_timer = 0.0      # > 0 while a gear change is in progress
    log = {"t": [], "v": [], "x": [], "gear": [], "rpm": [], "F_t": [], "F_drag": []}

    while t < T_MAX:
        # Shift logic
        if shift_timer <= 0 and should_upshift(v, gear):
            gear += 1
            shift_timer = p.SHIFT_TIME
        drive = shift_timer <= 0
        if not drive:
            shift_timer -= DT

        log["t"].append(t)
        log["v"].append(v)
        log["x"].append(x)
        log["gear"].append(gear)
        log["rpm"].append(engine_rpm(v, gear))
        log["F_t"].append(tractive_force(v, gear) if drive else 0.0)
        log["F_drag"].append(drag_force(v))

        # Stop condition (after logging, so the final point is in the log)
        if v >= v_target:
            break

        v = max(v + acceleration(v, gear, drive)*DT, 0.0)   # don't roll back during a launch shift
        x = x + v*DT
        t = t + DT

    return {key: np.array(values) for key, values in log.items()}


# --- 3. ELASTICITY 80-120 IN A FIXED GEAR ---

def gear_is_valid(gear, v_start=80/3.6, v_end=120/3.6):
    """Usable if the engine stays inside its torque map for the whole run."""
    return engine_rpm(v_start, gear) >= p.RPM[0] and engine_rpm(v_end, gear) <= p.REV_LIMIT

def elasticity_run(gear, v_start=80/3.6, v_end=120/3.6):
    t, v, x = 0.0, v_start, 0.0
    log = {"t": [], "v": [], "x": [], "rpm": [], "F_t": [], "F_drag": []}

    while t < T_MAX:
        log["t"].append(t)
        log["v"].append(v)
        log["x"].append(x)
        log["rpm"].append(engine_rpm(v, gear))
        log["F_t"].append(tractive_force(v, gear))
        log["F_drag"].append(drag_force(v))

        if v >= v_end:
            break

        v = v + acceleration(v, gear)*DT
        x = x + v*DT
        t = t + DT

    log = {key: np.array(values) for key, values in log.items()}
    log["t_run"] = time_to_speed(log, v_end*3.6)
    return log


# --- 4. AERODYNAMIC COEFFICIENT SENSITIVITY ---
# One coefficient changes at a time; everything else stays at its params.py value.

CD_VALUES = p.CD*(1 + np.array([-0.2, -0.1, 0.0, 0.1, 0.2]))   # [-] baseline +/-20%
CL_VALUES = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])              # [-] downforce to lift
# C_L uses a wider, absolute range: +/-20% of 0.2 is only +/-0.04 and changes nothing visible.

def best_elasticity_gear():
    """Fastest 80-120 gear with the current params."""
    valid = [g for g in range(1, N_GEARS+1) if gear_is_valid(g)]
    return min(valid, key=lambda g: elasticity_run(g)["t_run"])

def run_studies(elasticity_gear):
    """Studies 1-3 with the current params."""
    res = {"power": power_curves(),
           "accel": accel_run(),
           "elasticity": elasticity_run(elasticity_gear)}
    res["elasticity"]["gear"] = elasticity_gear
    return res

def coeff_sensitivity(name, values, elasticity_gear=None):
    """Rerun the studies for each value of params.<name> ("CD" or "CL").
    Elasticity uses one fixed gear for every value (the baseline best gear by default)
    so the runs are comparable. Returns {value: {"power", "accel", "elasticity"}}."""
    gear = elasticity_gear or best_elasticity_gear()
    base = getattr(p, name)
    out = {}
    try:
        for val in values:
            setattr(p, name, val)           # model.py reads params at call time
            out[val] = run_studies(gear)
    finally:
        setattr(p, name, base)              # always restore the baseline
    return out


# --- 5. YAW (CROSSWIND) SENSITIVITY ---
# A crosswind w gives airspeed sqrt(v^2 + w^2) and yaw angle atan(w/v); C_d and C_L follow
# the CD_PSI / CL_PSI tables. Yaw is largest at low speed and falls as the car speeds up.

WIND_CASES = np.array([0.0, 5.0, 10.0, 15.0])   # [m/s] crosswind speeds

def yaw_cases(winds=WIND_CASES, elasticity_gear=None):
    """Rerun the studies for each crosswind speed.
    Returns {wind: {"power", "accel", "elasticity"}}."""
    gear = elasticity_gear or best_elasticity_gear()
    base = p.WIND
    out = {}
    try:
        for w in winds:
            p.WIND = w                      # model.py reads params at call time
            out[w] = run_studies(gear)
    finally:
        p.WIND = base                       # always restore the baseline
    return out


# --- RESULTS SUMMARY ---

def summary():
    """Run every study, print the metrics table and return the results for plotting."""
    pc = power_curves()
    acc = accel_run()
    ela = {g: elasticity_run(g) for g in range(1, N_GEARS+1) if gear_is_valid(g)}
    best = min(ela, key=lambda g: ela[g]["t_run"])

    print("=" * 52)
    print(f"Conditions: {p.GRADE*100:.0f}% grade, Cd = {p.CD}, A = {p.AREA_FRONT} m^2, m = {p.MASS} kg")
    print("=" * 52)

    print("\nPower balance")
    print("  Gear | v_max [km/h] | limited by")
    for g, (vg, why) in pc["vmax_gear"].items():
        print(f"    {g}  |    {vg*3.6:6.1f}    | {why}")
    print(f"  v_max unrestricted: {pc['v_max']*3.6:.1f} km/h (gear {pc['vmax_best_gear']})")
    print(f"  v_max limited:      {pc['v_max_lim']*3.6:.1f} km/h (limiter {p.V_LIMIT*3.6:.0f} km/h)")
    print(f"  Aero dominates above {pc['v_aero']*3.6:.1f} km/h, "
          f"drag share at v_max {np.interp(pc['v_max'], pc['v'], pc['drag_share'])*100:.1f} %")

    print(f"\nAcceleration (optimal upshift, shift time {p.SHIFT_TIME} s)")
    print(f"  0-100 km/h: {time_to_speed(acc, 100):.2f} s")
    print(f"  0-200 km/h: {time_to_speed(acc, 200):.2f} s")

    print("\nElasticity 80-120 km/h")
    for g in range(1, N_GEARS+1):
        if g in ela:
            print(f"  Gear {g}: {ela[g]['t_run']:.2f} s")
        else:
            print(f"  Gear {g}: outside torque map")
    print(f"  Best gear: {best} ({ela[best]['t_run']:.2f} s)")

    sens = {"CD": coeff_sensitivity("CD", CD_VALUES),
            "CL": coeff_sensitivity("CL", CL_VALUES)}
    for name, label, fixed in [("CD", "Drag", f"C_L = {p.CL}"), ("CL", "Lift", f"C_d = {p.CD}")]:
        gear = next(iter(sens[name].values()))["elasticity"]["gear"]
        print(f"\n{label} coefficient sensitivity ({fixed}, A = {p.AREA_FRONT} m^2 fixed)")
        print(f"   {name}   | v_max [km/h] | 0-100 [s] | 0-200 [s] | 80-120 gear {gear} [s]")
        for val, r in sens[name].items():
            print(f"  {val:+.3f} |    {r['power']['v_max']*3.6:6.1f}    |   {time_to_speed(r['accel'], 100):5.2f}   |"
                  f"   {time_to_speed(r['accel'], 200):5.2f}   |   {r['elasticity']['t_run']:5.2f}")

    yaw = yaw_cases()
    gear = next(iter(yaw.values()))["elasticity"]["gear"]
    print(f"\nYaw sensitivity (crosswind, C_d and C_L from yaw table up to {p.PSI[-1]} deg)")
    print(f"  Wind [m/s] | yaw at v_max [deg] | v_max [km/h] | 0-100 [s] | 0-200 [s] | 80-120 gear {gear} [s]")
    for w, r in yaw.items():
        psi_vmax = np.degrees(np.arctan2(w, r["power"]["v_max"]))
        print(f"     {w:4.0f}    |       {psi_vmax:5.1f}        |    {r['power']['v_max']*3.6:6.1f}    |"
              f"   {time_to_speed(r['accel'], 100):5.2f}   |   {time_to_speed(r['accel'], 200):5.2f}   |"
              f"   {r['elasticity']['t_run']:5.2f}")

    return {"power": pc, "accel": acc, "elasticity": ela, "sensitivity": sens, "yaw": yaw}


if __name__ == "__main__":
    summary()
