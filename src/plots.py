from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt
import params as p
from model import drag_force, rolling_force, grade_force, resistance_force

# --- SHARED STYLE (import this module from other sections to reuse it) ---
FIG_DIR  = Path(__file__).resolve().parent.parent / "figures"
FIGSIZE  = (8, 5)
GREY     = "0.5"
HIGHLIGHT = "tab:red"

plt.rcParams.update({
    "figure.figsize":    FIGSIZE,
    "figure.dpi":        110,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "legend.frameon":    False,
    "lines.linewidth":   1.5,
})

def save(fig, name, retries=5):
    """Save to figures/. Retries because Windows can briefly lock a PNG that is open
    in a preview or being scanned/synced, which shows up as OSError 22."""
    FIG_DIR.mkdir(exist_ok=True)
    for attempt in range(retries):
        try:
            fig.savefig(FIG_DIR / f"{name}.png")
            return
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(0.5)


# --- FIGURES ---

def fig_resistance_forces(v_max_kmh=300):
    v_kmh = np.linspace(0, v_max_kmh, 301)
    v = v_kmh/3.6
    fig, ax = plt.subplots()
    ax.plot(v_kmh, drag_force(v), label="Drag")
    ax.plot(v_kmh, rolling_force(v), label="Rolling resistance")
    ax.plot(v_kmh, grade_force(v), label=f"Gradient ({p.GRADE*100:.0f}%)")
    ax.plot(v_kmh, resistance_force(v), "k", lw=2, label="Total")
    ax.set_xlabel("Speed [km/h]")
    ax.set_ylabel("Force [N]")
    ax.set_title("Resistance forces vs speed")
    ax.legend(loc="upper left")
    return fig

def fig_power_curves(pc):
    v_kmh = pc["v"]*3.6
    show = v_kmh <= 1.15*pc["v_max"]*3.6
    fig, ax = plt.subplots(figsize=(9, 6))
    for g, P in pc["P_gear"].items():
        ax.plot(v_kmh[show], P[show]/1e3, lw=1, label=f"Gear {g}")
    ax.plot(v_kmh[show], pc["P_tot"][show]/1e3, "k", lw=2, label="P_total")
    ax.axvline(pc["v_max"]*3.6, color=HIGHLIGHT, linestyle="--",
               label=f"v_max unrestricted {pc['v_max']*3.6:.0f} km/h")
    ax.axvline(p.V_LIMIT*3.6, color=GREY, linestyle="--",
               label=f"Limiter {p.V_LIMIT*3.6:.0f} km/h")
    ax.set_xlabel("Speed [km/h]")
    ax.set_ylabel("Power [kW]")
    ax.set_title("Power available at the wheel vs power required")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=8)
    return fig

def fig_accel(log):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6.5))
    ax1.plot(log["t"], log["v"]*3.6)
    ax1.axhline(100, color=GREY, linestyle="--")
    ax1.axhline(200, color=GREY, linestyle="--")
    ax1.set_ylabel("Speed [km/h]")
    ax1.set_title("Full-throttle acceleration from standstill")

    ax2.plot(log["t"], log["rpm"])
    ax2.set_ylabel("Engine speed [rpm]")
    ax2b = ax2.twinx()
    ax2b.step(log["t"], log["gear"], color="tab:orange", where="post")
    ax2b.set_ylabel("Gear", color="tab:orange")
    ax2b.grid(False)
    ax2.set_xlabel("Time [s]")
    fig.tight_layout()
    return fig

def fig_accel_baseline(log):
    """Speed against time for the baseline C_d, with the 0-100 and 0-200 times marked."""
    from simulate import time_to_speed
    fig, ax = plt.subplots()
    ax.plot(log["t"], log["v"]*3.6, label=f"C_d {p.CD:.3f} (baseline)")
    for v_kmh in (100, 200):
        t = time_to_speed(log, v_kmh)
        ax.plot([0, t], [v_kmh, v_kmh], color=GREY, linestyle="--", lw=0.8)
        ax.plot([t, t], [0, v_kmh], color=GREY, linestyle="--", lw=0.8)
        ax.plot(t, v_kmh, "o", color=HIGHLIGHT)
        ax.annotate(f"0-{v_kmh} km/h: {t:.2f} s", (t, v_kmh), xytext=(8, -14),
                    textcoords="offset points")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Speed [km/h]")
    ax.set_title("Full-throttle acceleration, baseline drag coefficient")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right")
    return fig

def fig_elasticity(ela):
    fig, ax = plt.subplots()
    for g, log in ela.items():
        ax.plot(log["t"], log["v"]*3.6, label=f"Gear {g}: {log['t_run']:.2f} s")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Speed [km/h]")
    ax.set_title("In-gear acceleration 80-120 km/h")
    ax.legend()
    return fig


def fig_coeff_sensitivity(sens, name):
    """One line per coefficient value: max speed, acceleration, elasticity, and the
    resistance power the coefficient acts on (drag for C_d, rolling for C_L)."""
    from simulate import time_to_speed
    label = {"CD":   lambda val: f"C_d {val:.3f}",
             "CL":   lambda val: f"C_L {val:.3f}",
             "WIND": lambda val: f"Crosswind {val:.0f} m/s"}[name]
    vals = sorted(sens)
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(vals)))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax_v, ax_a, ax_e, ax_d = axes.flat

    # Maximum speed: power required per value against the best-gear power available
    pc0 = sens[vals[0]]["power"]
    v_kmh = pc0["v"]*3.6
    show = v_kmh <= 1.1*max(r["power"]["v_max"] for r in sens.values())*3.6
    P_avail = np.fmax.reduce(list(pc0["P_gear"].values()))   # tyre grip barely matters here
    ax_v.plot(v_kmh[show], P_avail[show]/1e3, "k", lw=2, label="Available (best gear)")
    for val, c in zip(vals, colors):
        pc = sens[val]["power"]
        ax_v.plot(v_kmh[show], pc["P_tot"][show]/1e3, color=c,
                  label=f"{label(val)}: {pc['v_max']*3.6:.1f} km/h")
        ax_v.plot(pc["v_max"]*3.6, np.interp(pc["v_max"], pc["v"], pc["P_tot"])/1e3, "o", color=c)
    ax_v.set_xlabel("Speed [km/h]")
    ax_v.set_ylabel("Power [kW]")
    ax_v.set_title("Maximum speed")
    ax_v.set_ylim(bottom=0)

    # Acceleration 0-100 and 0-200
    for val, c in zip(vals, colors):
        log = sens[val]["accel"]
        ax_a.plot(log["t"], log["v"]*3.6, color=c,
                  label=f"{label(val)}: {time_to_speed(log, 100):.2f} s / {time_to_speed(log, 200):.2f} s")
    ax_a.set_xlabel("Time [s]")
    ax_a.set_ylabel("Speed [km/h]")
    ax_a.set_title("Acceleration times (0-100 / 0-200 km/h)")

    # Elasticity 80-120 in a fixed gear
    for val, c in zip(vals, colors):
        log = sens[val]["elasticity"]
        ax_e.plot(log["t"], log["v"]*3.6, color=c, label=f"{label(val)}: {log['t_run']:.2f} s")
    ax_e.set_xlabel("Time [s]")
    ax_e.set_ylabel("Speed [km/h]")
    ax_e.set_title(f"Elasticity 80-120 km/h (gear {log['gear']})")

    # The resistance power this coefficient changes
    key, title = {"CD": ("P_drag", "Power absorbed by aerodynamic drag"),
                  "WIND": ("P_drag", "Power absorbed by aerodynamic drag"),
                  "CL": ("P_roll", "Power absorbed by rolling resistance")}[name]
    for val, c in zip(vals, colors):
        pc = sens[val]["power"]
        ax_d.plot(v_kmh[show], pc[key][show]/1e3, color=c, label=f"{label(val)}")
    ax_d.set_xlabel("Speed [km/h]")
    ax_d.set_ylabel("Power [kW]")
    ax_d.set_title(title)
    ax_d.set_ylim(bottom=0)

    for ax in axes.flat:
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle({"CD": "Effect of drag coefficient on performance",
                  "CL": "Effect of lift coefficient on performance",
                  "WIND": "Effect of crosswind (yaw) on performance, C_d and C_L from yaw table"}[name])
    fig.tight_layout()
    return fig


def fig_yaw_coefficients():
    """C_d and C_L against yaw angle, from the params.py table."""
    psi = np.linspace(0, p.PSI[-1], 100)
    fig, ax_c = plt.subplots()
    ax_c.plot(psi, np.interp(psi, p.PSI, p.CD_PSI), color="tab:blue", label="C_d")
    ax_c.plot(p.PSI, p.CD_PSI, "o", color="tab:blue")
    ax_c.plot(psi, np.interp(psi, p.PSI, p.CL_PSI), color="tab:orange", label="C_L")
    ax_c.plot(p.PSI, p.CL_PSI, "o", color="tab:orange")
    ax_c.set_xlabel("Yaw angle [deg]")
    ax_c.set_ylabel("Coefficient [-]")
    ax_c.set_title("Aerodynamic coefficients against yaw")
    ax_c.legend()
    return fig

if __name__ == "__main__":
    from simulate import summary
    res = summary()

    figures = {
        "resistance_forces": fig_resistance_forces(),
        "power_curves":      fig_power_curves(res["power"]),
        "accel_0_200":       fig_accel(res["accel"]),
        "accel_baseline":    fig_accel_baseline(res["accel"]),
        "elasticity_80_120": fig_elasticity(res["elasticity"]),
        "cd_sensitivity":    fig_coeff_sensitivity(res["sensitivity"]["CD"], "CD"),
        "cl_sensitivity":    fig_coeff_sensitivity(res["sensitivity"]["CL"], "CL"),
        "yaw_coefficients":  fig_yaw_coefficients(),
        "yaw_sensitivity":   fig_coeff_sensitivity(res["yaw"], "WIND"),
    }
    for name, fig in figures.items():
        save(fig, name)
    print(f"\nSaved {len(figures)} figures to {FIG_DIR}")
    plt.show()
