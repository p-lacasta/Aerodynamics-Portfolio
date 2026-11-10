import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
import time

# --- STYLE ---
GREY = "0.5"
COMP_COLOUR = "#dfe9f3"   # compression section background
WS_COLOUR   = "#f3ebdc"   # working section background

plt.rcParams.update({
    "figure.figsize":    (8, 5),
    "figure.dpi":        110,
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "legend.frameon":    False,
    "lines.linewidth":   1.5,
})


BASE_DIR = Path(__file__).resolve().parent
PLOT_DIR = BASE_DIR / "plots"

def save(fig, name, retries=5):
    """Save to plots/, retrying if Windows briefly locks the file."""
    PLOT_DIR.mkdir(exist_ok=True)
    for attempt in range(retries):
        try:
            fig.savefig(PLOT_DIR / f"{name}.png", bbox_inches="tight", dpi=300)
            return
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(0.5)


# --- INPUTS ---
A_ws    = 0.31*0.31     # [m^2] working section area
nu      = 0.000015      # [m^2/s] kinematic viscosity of air
Re_tr   = 500000        # [-] flat-plate transition Re_x
ws_x    = 1.125         # [m] working section start

# Transition as Re_theta via Blasius: theta = 0.664 x / sqrt(Re_x)
Re_theta_tr = 0.664*np.sqrt(Re_tr)     # ~470

PROFILE_XLSX = BASE_DIR / "Armfield Wind Tunnel Inlet Contraction and Working Section Profile.xlsx"

# --- GEOMETRY ---
# header=None keeps the first data row (inlet point)
profile = pd.read_excel(PROFILE_XLSX, skiprows=2, usecols="D:F", nrows=76, header=None).to_numpy()/1000
x_pts, y_pts, z_pts = profile[:, 0], profile[:, 1], profile[:, 2]

# Distance along the wall (arc length) from the inlet
l_pts = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x_pts), np.diff(y_pts)))])

# Resample onto a 1 mm grid (PCHIP: smooth, no overshoot)
l = np.arange(0, l_pts[-1], 0.001)
l = np.append(l, l_pts[-1])
wt_x = PchipInterpolator(l_pts, x_pts)(l)
wt_y = PchipInterpolator(l_pts, y_pts)(l)
wt_z = PchipInterpolator(l_pts, z_pts)(l)
# Working section start as a wall length
ws_l = np.interp(ws_x, wt_x, l)

speeds = np.array([5, 10, 20, 30, 40])   # [m/s] working section speeds
colors = plt.cm.viridis(np.linspace(0, 0.85, len(speeds)))


# --- THWAITES CLOSURE RELATIONS ---
# Curve fits to Thwaites' table (White, Viscous Fluid Flow), valid for -0.09 <= lambda <= 0.25
LAM_MIN, LAM_MAX = -0.09, 0.25

def thwaites_H(lam):
    """Shape factor H = delta*/theta as a function of lambda (H = 2.59 at lambda = 0)."""
    z = 0.25 - np.clip(lam, LAM_MIN, LAM_MAX)
    return 2.0 + 4.14*z - 83.5*z**2 + 854*z**3 - 3337*z**4 + 4576*z**5

def thwaites_shear(lam):
    """Shear correlation l(lambda) = tau_w*theta/(mu*U); zero at separation (lambda = -0.09)."""
    return (np.clip(lam, LAM_MIN, LAM_MAX) + 0.09)**0.62


def thwaites_theta(U_x, l):
    """Laminar theta from Thwaites: theta^2 = 0.45 nu / U^6 * integral(U^5 dl), theta = 0 at the inlet."""
    integral = cumulative_trapezoid(U_x**5, l, initial=0)
    return np.sqrt(0.45*nu*integral/U_x**6)


# Turbulent 1/7th power law: delta = 0.16 x / Re_x^(1/7), delta* = delta/8, theta = 7 delta/72 (H = 9/7)
C_DELTA = 0.16
C_DSTAR = C_DELTA/8         # 0.0200
C_THETA = C_DELTA*7/72      # 0.0156

def turbulent_boundary_layer(Re_x, x):
    delta = x*C_DELTA/Re_x**(1/7)
    delta_star = x*C_DSTAR/Re_x**(1/7)
    theta = x*C_THETA/Re_x**(1/7)
    return delta, delta_star, theta


def compute_layer(U_ws):
    # Area(x)
    A_x = wt_y*wt_z*4

    # Speed(x) from continuity (uniform core flow)
    U_x = U_ws*(A_ws/A_x)
    dUdl = np.gradient(U_x, l)

    # --- LAMINAR: Thwaites along the whole tunnel ---
    theta_lam = thwaites_theta(U_x, l)
    lam = theta_lam**2/nu*dUdl                 # pressure-gradient parameter
    H_lam = thwaites_H(lam)
    delta_star_lam = H_lam*theta_lam
    # Thwaites gives no delta: Blasius ratio delta/theta = 7.4
    delta_lam = (4.91/0.664)*theta_lam
    Re_theta = U_x*theta_lam/nu

    # --- TRANSITION ---
    # Criterion applied only where the gradient is zero or adverse
    no_favourable = lam <= 1e-6
    trips = (Re_theta >= Re_theta_tr) & no_favourable
    if trips.any():
        i_tr = np.argmax(trips)
        x_tr = l[i_tr]
        is_turbulent = l >= x_tr
    else:
        i_tr = None
        x_tr = np.nan
        is_turbulent = np.zeros_like(l, dtype=bool)

    delta, delta_star, theta = delta_lam.copy(), delta_star_lam.copy(), theta_lam.copy()

    # --- TURBULENT: 1/7th power law from a virtual origin ---
    if i_tr is not None:
        U_tr = U_x[i_tr]
        theta_tr = theta_lam[i_tr]
        # Virtual origin that matches theta at transition
        origin_turb = ((theta_tr*(U_tr/nu)**(1/7))/C_THETA)**(7/6)
        x_turb = origin_turb + (l[is_turbulent] - x_tr)
        Re_turb = U_x[is_turbulent]*x_turb/nu
        d_t, ds_t, th_t = turbulent_boundary_layer(Re_turb, x_turb)
        delta[is_turbulent] = d_t
        delta_star[is_turbulent] = ds_t
        theta[is_turbulent] = th_t

    return dict(U_x=U_x, lam=lam, Re_theta=Re_theta, delta=delta, delta_star=delta_star,
                theta=theta, is_turbulent=is_turbulent, x_tr=x_tr)


# Convert to mm for readability
l_mm = l*1000
z_mm = wt_z*1000  # lateral width, used for the top-view shading


def shade_tunnel(ax, x, z):
    """Background shaped like the tunnel width (top view), scaled to the panel height."""
    # Working section first, compression section drawn over it
    top = ax.get_ylim()[1]
    xf = np.linspace(x.min(), x.max(), 2000)
    width = np.interp(xf, x, z)/z.max()*top
    compression = xf < ws_l*1000
    ax.fill_between(xf, 0, width, color=WS_COLOUR, linewidth=0, zorder=0)
    ax.fill_between(xf, 0, width, where=compression, color=COMP_COLOUR, linewidth=0, zorder=0)

    # Section legend inside the plot
    handles = [Patch(color=COMP_COLOUR, label="Compression section"),
               Patch(color=WS_COLOUR, label="Working section")]
    ax.add_artist(ax.legend(handles=handles, loc="lower right", frameon=True,
                            facecolor="white", edgecolor="none", framealpha=0.9))


# --- FIGURES ---
# Local velocity
fig0, ax0 = plt.subplots()
# Boundary layer thickness
fig1, ax1 = plt.subplots()
# Displacement thickness
fig2, ax2 = plt.subplots()
# Momentum thickness
fig3, ax3 = plt.subplots()
# Shape factor H = delta_star/theta (about 2.59 laminar, 1.29 turbulent)
fig4, ax4 = plt.subplots()
# Thwaites pressure-gradient parameter lambda (laminar region only)
fig5, ax5 = plt.subplots()

A_settle = np.max(wt_y[wt_x > 0.1]*wt_z[wt_x > 0.1]*4)   # settling section, after the inlet lip
print(f"Contraction ratio (settling section / working section) = {A_settle/A_ws:.2f}"
      f"  |  Re_theta,tr = {Re_theta_tr:.0f}")
print(f"Working section starts at l = {ws_l*1000:.0f} mm")
print(f"{'U_ws [m/s]':>10} {'l_tr [mm]':>10} {'into WS [mm]':>13} {'theta at WS [mm]':>17} "
      f"{'d* at WS [mm]':>14} {'d* at WS end [mm]':>18} {'area loss end [%]':>18} {'max Re_th (fav.)':>16}")

# One curve per freestream speed on each plot
for U, c in zip(speeds, colors):
    r = compute_layer(U)
    # theta = 0 at the inlet point, so skip that division
    H = np.divide(r["delta_star"], r["theta"], out=np.full_like(l, np.nan), where=r["theta"] > 0)
    x_tr = r["x_tr"]

    lines = [(ax0, r["U_x"]), (ax1, r["delta"]*1000), (ax2, r["delta_star"]*1000),
             (ax3, r["theta"]*1000), (ax4, H)]
    for ax, y in lines:
        ax.plot(l_mm, y, color=c, label=f"{U} m/s")
        if np.isfinite(x_tr):
            ax.axvline(x_tr*1000, color=c, linestyle="--", linewidth=0.8)

    th_ws = np.interp(ws_l, l, r["theta"])*1000
    ds_ws = np.interp(ws_l, l, r["delta_star"])*1000
    into_ws = (x_tr - ws_l)*1000
    # Max Re_theta in the favourable-gradient region
    fav = r["lam"] > 1e-6
    max_Re_theta_fav = r["Re_theta"][fav].max()
    # Displacement effect: the square side shrinks by 2 delta*
    side = np.sqrt(A_ws)
    area_loss = 1 - ((side - 2*r["delta_star"][-1])/side)**2
    print(f"{U:>10} {x_tr*1000:>10.0f} {into_ws:>13.0f} {th_ws:>17.3f} {ds_ws:>14.3f} "
          f"{r['delta_star'][-1]*1000:>18.3f} {area_loss*100:>18.2f} {max_Re_theta_fav:>16.0f}")

# lambda is the same for every speed (theta^2 ~ 1/U, dU/dx ~ U)
ax5.plot(l_mm, r["lam"], color=GREY, label="All speeds")

for ax in (ax0, ax1, ax2, ax3, ax4, ax5):
    shade_tunnel(ax, l_mm, z_mm)

ax0.set_xlabel("Length along wind tunnel, l [mm]")
ax0.set_ylabel(r"Local velocity, $U(x)$ [m/s]")
ax0.set_title("Local velocity along the wind tunnel")
ax0.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=len(speeds)+1)
fig0.tight_layout()

ax1.set_xlabel("Length along wind tunnel, l [mm]")
ax1.set_ylabel(r"Boundary layer thickness, $\delta$ [mm]")
ax1.set_title("Boundary layer thickness along the wind tunnel")
ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=len(speeds)+1)
fig1.tight_layout()

ax2.set_xlabel("Length along wind tunnel, l [mm]")
ax2.set_ylabel(r"Displacement thickness, $\delta^*$ [mm]")
ax2.set_title("Displacement thickness along the wind tunnel")
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=len(speeds)+1)
fig2.tight_layout()

ax3.set_xlabel("Length along wind tunnel, l [mm]")
ax3.set_ylabel(r"Momentum thickness, $\theta$ [mm]")
ax3.set_title("Momentum thickness along the wind tunnel")
ax3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=len(speeds)+1)
fig3.tight_layout()

ax4.set_xlabel("Length along wind tunnel, l [mm]")
ax4.set_ylabel(r"Shape factor, $H = \delta^*/\theta$")
ax4.set_title("Shape factor along the wind tunnel")
ax4.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=len(speeds)+1)
fig4.tight_layout()

ax5.axhline(0, color=GREY, linewidth=0.8)
ax5.set_xlabel("Length along wind tunnel, l [mm]")
ax5.set_ylabel(r"Thwaites parameter, $\lambda = (\theta^2/\nu)\,dU/dx$")
ax5.set_title("Pressure-gradient parameter (laminar region)")
ax5.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=len(speeds)+1)
fig5.tight_layout()

save(fig0, "local_velocity")
save(fig1, "boundary_layer_thickness")
save(fig2, "displacement_thickness")
save(fig3, "momentum_thickness")
save(fig4, "shape_factor")
save(fig5, "thwaites_lambda")
print(f"Saved 6 figures to {PLOT_DIR}")
