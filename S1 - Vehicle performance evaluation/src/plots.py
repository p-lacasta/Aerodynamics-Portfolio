from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt
import params as p
from model import drag_force, rolling_force, resistance_force

# --- STYLE ---
FIG_DIR  = Path(__file__).resolve().parent.parent / "figures"
FIGSIZE  = (8, 5)
GREY     = "0.5"

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

# Viridis colours: purple -> teal -> green
PURPLE = plt.cm.viridis(0.0)
TEAL   = plt.cm.viridis(0.425)
GREEN  = plt.cm.viridis(0.85)

def series_colours(n):
    """n viridis colours from purple to green."""
    return plt.cm.viridis(np.linspace(0, 0.85, n))

def legend_below(ax, ncol=2, y=-0.12, **kw):
    """Legend centred below the axes."""
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, **kw)

def ms_axis_top(ax):
    """Secondary speed axis on top in m/s (bottom axis is km/h)."""
    top = ax.secondary_xaxis("top", functions=(lambda x: x/3.6, lambda x: x*3.6))
    top.set_xlabel("Speed [m/s]")
    return top

def save(fig, name, retries=5):
    """Save to figures/, retrying if Windows briefly locks the file."""
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

def fig_engine_curves():
    """Full-load torque and power against engine speed, with datasheet power as a check."""
    rpm = np.linspace(p.RPM[0], p.REV_LIMIT, 501)
    T = np.interp(rpm, p.RPM, p.TORQUE)
    P = T*rpm*2*np.pi/60/1e3

    fig, ax_T = plt.subplots()
    ax_P = ax_T.twinx()
    ax_P.grid(False)
    ax_P.spines["right"].set_visible(True)

    ax_T.plot(rpm, T, color=PURPLE, lw=2, label="Torque, $T$")
    ax_T.plot(p.RPM, p.TORQUE, "o", ms=3, color=PURPLE)
    ax_P.plot(rpm, P, color=GREEN, lw=2, label=r"Power, $P = T\omega$")
    ax_P.plot(p.RPM, p.POWER, "s", ms=3.5, mfc="none", color=GREEN, label="Power (datasheet)")

    # Peaks: torque plateau and maximum power
    T_max = p.TORQUE.max()
    plateau = p.RPM[p.TORQUE == T_max]
    ax_T.annotate(f"{T_max:.0f} Nm, {plateau[0]:.0f}-{plateau[-1]:.0f} rpm",
                  (plateau[0], T_max), xytext=(0, 8), textcoords="offset points",
                  ha="right")
    i = np.argmax(P)
    ax_P.plot(rpm[i], P[i], "o", color=GREEN, ms=6)
    ax_P.annotate(f"{P[i]:.0f} kW ({P[i]*1.35962:.0f} PS), {rpm[i]:.0f} rpm", (rpm[i], P[i]),
                  xytext=(-10, 10), textcoords="offset points", ha="right")

    ax_T.set_xlabel("Engine speed [rpm]")
    ax_T.set_ylabel("Torque, $T$ [Nm]")
    ax_P.set_ylabel("Power, $P$ [kW]")
    ax_T.set_ylim(0, 1.15*T_max)
    ax_P.set_ylim(0, 1.15*P.max())
    ax_T.set_xlim(p.RPM[0] - 250, p.REV_LIMIT + 250)
    ax_T.set_title("Full-load engine curves, CUPRA Formentor VZ5 (2.5 TSI)")

    h1, l1 = ax_T.get_legend_handles_labels()
    h2, l2 = ax_P.get_legend_handles_labels()
    ax_T.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    return fig

def fig_resistance_forces(v_max_kmh=300):
    v_kmh = np.linspace(0, v_max_kmh, 301)
    v = v_kmh/3.6
    fig, ax = plt.subplots()
    ax.plot(v_kmh, drag_force(v), color=PURPLE, label="Aerodynamic drag, $D$")
    ax.plot(v_kmh, rolling_force(v), color=GREEN, label="Rolling resistance, $R_{r}$")
    ax.plot(v_kmh, resistance_force(v), "k", lw=2, label="Total resistance, $D + R_{r}$")

    # Drag = rolling resistance crossover (first sign change of the difference)
    diff = drag_force(v) - rolling_force(v)
    i = np.argmax(diff >= 0)
    v_x = np.interp(0, diff[i-1:i+1], v_kmh[i-1:i+1])
    F_x = drag_force(v_x/3.6)
    ax.plot(v_x, F_x, "o", color=TEAL)
    ax.annotate(f"$D = R_r$: {v_x:.1f} km/h, {F_x:.0f} N", (v_x, F_x), xytext=(2*v_x, 3*F_x),
                textcoords="data", va="center",
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))

    ax.set_xlabel("Speed [km/h]")
    ax.set_ylabel("Force [N]")
    ax.set_title("Resistance forces against speed")
    ax.set_xlim(0, v_max_kmh)
    ax.set_ylim(bottom=0)
    ms_axis_top(ax)
    legend_below(ax, ncol=3)
    return fig

def fig_power_curves(pc):
    v_kmh = pc["v"]*3.6
    show = v_kmh <= 1.15*pc["v_max"]*3.6
    fig, ax = plt.subplots(figsize=(9, 6))
    gears = list(pc["P_gear"])
    for (g, P), c in zip(pc["P_gear"].items(), series_colours(len(gears))):
        ax.plot(v_kmh[show], P[show]/1e3, color=c, lw=1.2, label=f"Gear {g}")
    ax.plot(v_kmh[show], pc["P_tot"][show]/1e3, "k", lw=2, label="Power required")
    P_vmax = np.interp(pc["v_max"], pc["v"], pc["P_tot"])/1e3
    ax.axvline(pc["v_max"]*3.6, color="k", linestyle="--", lw=1,
               label=f"$v_{{max}}$ unrestricted: {pc['v_max']*3.6:.0f} km/h")
    ax.plot(pc["v_max"]*3.6, P_vmax, "o", color="k")
    ax.axvline(p.V_LIMIT*3.6, color=GREY, linestyle=":", lw=1.2,
               label=f"Speed limiter: {p.V_LIMIT*3.6:.0f} km/h")
    ax.set_xlabel("Speed [km/h]")
    ax.set_ylabel("Power at the wheels [kW]")
    ax.set_title("Power available at the wheels against power required")
    ax.set_xlim(0, v_kmh[show].max())
    ax.set_ylim(bottom=0)
    ms_axis_top(ax)
    legend_below(ax, ncol=4, y=-0.1)
    return fig

def fig_accel(log):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6.5))
    ax1.plot(log["t"], log["v"]*3.6, color=PURPLE)
    for v_ref in (100, 200):
        ax1.axhline(v_ref, color=GREY, linestyle="--", lw=0.8)
    ax1.set_ylabel("Speed [km/h]")
    ax1.set_title("Vehicle speed")
    ax1.set_ylim(bottom=0)

    ax2.plot(log["t"], log["rpm"], color=PURPLE, label="Engine speed")
    ax2.axhline(p.REV_LIMIT, color=GREY, linestyle="--", lw=0.8)
    ax2.set_ylabel("Engine speed [rpm]")
    ax2.set_title("Engine speed and gear")
    ax2b = ax2.twinx()
    ax2b.spines["right"].set_visible(True)
    ax2b.step(log["t"], log["gear"], color=GREEN, where="post", label="Gear")
    ax2b.set_ylabel("Gear [-]")
    ax2b.set_yticks(range(1, int(log["gear"].max()) + 1))
    ax2b.set_ylim(0.5, log["gear"].max() + 1)   # keep the top gear clear of the rev-limit line
    ax2b.grid(False)
    ax2.set_xlabel("Time [s]")
    ax2.set_xlim(left=0)

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    fig.suptitle("Full-throttle acceleration from standstill")
    fig.tight_layout()
    return fig

def fig_accel_baseline(log):
    """Speed against time for the baseline C_d, with the 0-100 and 0-200 times marked."""
    from simulate import time_to_speed
    fig, ax = plt.subplots()
    ax.plot(log["t"], log["v"]*3.6, color=PURPLE, label=f"$C_d$ = {p.CD:.2f} (baseline)")
    for v_kmh in (100, 200):
        t = time_to_speed(log, v_kmh)
        ax.plot([0, t], [v_kmh, v_kmh], color=GREY, linestyle="--", lw=0.8)
        ax.plot([t, t], [0, v_kmh], color=GREY, linestyle="--", lw=0.8)
        ax.plot(t, v_kmh, "o", color=GREEN)
        ax.annotate(f"0-{v_kmh} km/h: {t:.2f} s", (t, v_kmh), xytext=(8, -14),
                    textcoords="offset points")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Speed [km/h]")
    ax.set_title("Full-throttle acceleration, baseline drag coefficient")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    legend_below(ax, ncol=1)
    return fig

def fig_elasticity(ela):
    fig, ax = plt.subplots()
    for (g, log), c in zip(ela.items(), series_colours(len(ela))):
        ax.plot(log["t"], log["v"]*3.6, color=c, label=f"Gear {g}: {log['t_run']:.2f} s")
    ax.axhline(80, color=GREY, linestyle="--", lw=0.8)
    ax.axhline(120, color=GREY, linestyle="--", lw=0.8)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Speed [km/h]")
    ax.set_title("In-gear acceleration, 80-120 km/h")
    ax.set_xlim(left=0)
    legend_below(ax, ncol=min(len(ela), 4))
    return fig


def fig_coeff_sensitivity(sens, name):
    """Four stacked panels, one line per value: maximum speed, acceleration, elasticity
    and the resistance power the parameter acts on."""
    from simulate import time_to_speed
    label = {"CD":   lambda val: f"$C_d$ = {val:.2f}",
             "CL":   lambda val: f"$C_L$ = {val:.2f}",
             "WIND": lambda val: f"$w$ = {val:.0f} m/s"}[name]
    vals = sorted(sens)
    colors = series_colours(len(vals))
    fig, (ax_v, ax_a, ax_e, ax_d) = plt.subplots(4, 1, figsize=(10, 11))

    pc0 = sens[vals[0]]["power"]
    v_kmh = pc0["v"]*3.6
    show = v_kmh <= 1.1*max(r["power"]["v_max"] for r in sens.values())*3.6

    # (a) Maximum speed: power required per value against the best-gear power available
    P_avail = np.fmax.reduce(list(pc0["P_gear"].values()))   # tyre grip barely matters here
    ax_v.plot(v_kmh[show], P_avail[show]/1e3, "k", lw=2, label="Available (best gear)")
    for val, c in zip(vals, colors):
        pc = sens[val]["power"]
        ax_v.plot(v_kmh[show], pc["P_tot"][show]/1e3, color=c,
                  label=f"{label(val)}: $v_{{max}}$ = {pc['v_max']*3.6:.1f} km/h")
        ax_v.plot(pc["v_max"]*3.6, np.interp(pc["v_max"], pc["v"], pc["P_tot"])/1e3, "o", color=c)
    ax_v.set_xlabel("Speed [km/h]")
    ax_v.set_ylabel("Power [kW]")
    ax_v.set_title("(a) Maximum speed: power at the wheels against power required")
    ax_v.set_xlim(0, v_kmh[show].max())
    ax_v.set_ylim(bottom=0)

    # (b) Acceleration 0-100 and 0-200
    for val, c in zip(vals, colors):
        log = sens[val]["accel"]
        ax_a.plot(log["t"], log["v"]*3.6, color=c,
                  label=f"{label(val)}: {time_to_speed(log, 100):.2f} s / {time_to_speed(log, 200):.2f} s")
    for v_ref in (100, 200):
        ax_a.axhline(v_ref, color=GREY, linestyle="--", lw=0.8)
    ax_a.set_xlabel("Time [s]")
    ax_a.set_ylabel("Speed [km/h]")
    ax_a.set_title("(b) Acceleration, 0-100 / 0-200 km/h")
    ax_a.set_xlim(left=0)
    ax_a.set_ylim(bottom=0)

    # (c) Elasticity 80-120 in a fixed gear
    for val, c in zip(vals, colors):
        log = sens[val]["elasticity"]
        ax_e.plot(log["t"], log["v"]*3.6, color=c, label=f"{label(val)}: {log['t_run']:.2f} s")
    ax_e.set_xlabel("Time [s]")
    ax_e.set_ylabel("Speed [km/h]")
    ax_e.set_title(f"(c) Elasticity, 80-120 km/h in gear {log['gear']}")
    ax_e.set_xlim(left=0)

    # (d) The resistance power this coefficient changes
    key, what = {"CD":   ("P_drag", "aerodynamic drag"),
                 "WIND": ("P_drag", "aerodynamic drag"),
                 "CL":   ("P_roll", "rolling resistance")}[name]
    for val, c in zip(vals, colors):
        pc = sens[val]["power"]
        ax_d.plot(v_kmh[show], pc[key][show]/1e3, color=c, label=label(val))
    ax_d.set_xlabel("Speed [km/h]")
    ax_d.set_ylabel("Power [kW]")
    ax_d.set_title(f"(d) Power absorbed by {what}")
    ax_d.set_xlim(0, v_kmh[show].max())
    ax_d.set_ylim(bottom=0)

    # Legends to the right of each panel
    for ax in (ax_v, ax_a, ax_e, ax_d):
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    fig.suptitle({"CD":   "Effect of drag coefficient $C_d$ on performance",
                  "CL":   "Effect of lift coefficient $C_L$ on performance",
                  "WIND": r"Effect of crosswind on performance ($C_d$ and $C_L$ from yaw data)"}[name])
    fig.tight_layout()
    return fig


def fig_yaw_coefficients():
    """C_d and C_L against yaw angle, from the params.py table."""
    psi = np.linspace(0, p.PSI[-1], 100)
    fig, ax_c = plt.subplots()
    ax_c.plot(psi, np.interp(psi, p.PSI, p.CD_PSI), color=PURPLE, label="$C_d$")
    ax_c.plot(p.PSI, p.CD_PSI, "o", color=PURPLE)
    ax_c.plot(psi, np.interp(psi, p.PSI, p.CL_PSI), color=GREEN, label="$C_L$")
    ax_c.plot(p.PSI, p.CL_PSI, "o", color=GREEN)
    ax_c.set_xlabel(r"Yaw angle, $\psi$ [deg]")
    ax_c.set_ylabel("Coefficient [-]")
    ax_c.set_title(r"Aerodynamic coefficients against yaw angle")
    ax_c.set_xlim(0, p.PSI[-1])
    legend_below(ax_c, ncol=2)
    return fig

if __name__ == "__main__":
    from simulate import summary
    res = summary()

    figures = {
        "engine_curves":     fig_engine_curves(),
        "resistance_forces": fig_resistance_forces(),
        "power_curves":      fig_power_curves(res["power"]),
        "accel_0_200":       fig_accel(res["accel"]),
        "accel_baseline":    fig_accel_baseline(res["accel"]),
        "elasticity_80_120": fig_elasticity(res["elasticity"]),
        "yaw_coefficients":  fig_yaw_coefficients(),
        "cd_sensitivity":    fig_coeff_sensitivity(res["sensitivity"]["CD"], "CD"),
        "cl_sensitivity":    fig_coeff_sensitivity(res["sensitivity"]["CL"], "CL"),
        "yaw_sensitivity":   fig_coeff_sensitivity(res["yaw"], "WIND"),
    }
    for name, fig in figures.items():
        save(fig, name)
    print(f"\nSaved {len(figures)} figures to {FIG_DIR}")
    plt.show()
