import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator

# --- INPUTS ---
nu     = 1.5e-5                 # kinematic viscosity of air [m^2/s]
A_ws   = 0.31*0.31              # working section area [m^2]
ws_x   = 1.125                  # working section starts at x = 1125 mm [m]
speeds = [10, 20, 30, 40]       # working-section speeds [m/s]

# Transition: flat-plate Re_x = 5e5, written as Re_theta using Blasius (theta = 0.664 x / sqrt(Re_x)).
# Re_theta is used because Thwaites gives theta directly and it doesn't depend on where x = 0 is.
Re_theta_tr = 0.664*np.sqrt(5e5)          # ~470

# Turbulent 1/7th power law: delta = 0.16 x / Re_x^(1/7), theta = 7/72 delta, delta* = delta/8 (H = 9/7)
C_THETA = 0.16*7/72

COMP_COLOUR = "#dfe9f3"   # contraction background
WS_COLOUR   = "#f3ebdc"   # working section background

BASE_DIR = Path(__file__).resolve().parent
XLSX = BASE_DIR / "Armfield Wind Tunnel Inlet Contraction and Working Section Profile.xlsx"

# --- GEOMETRY ---
# header=None: otherwise pandas uses the first data row (the inlet point) as column names
x_pts, y_pts, z_pts = pd.read_excel(XLSX, skiprows=2, usecols="D:F", nrows=76, header=None).to_numpy().T/1000
A_pts = 4*y_pts*z_pts                     # full cross-section area (y, z are half-widths)

# Distance along the wall (arc length), starting at the inlet
l_pts = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x_pts), np.diff(y_pts)))])
ws_l  = np.interp(ws_x, x_pts, l_pts)     # start of working section, as a wall length

# Resample onto a ~1 mm grid. PCHIP = smooth spline that can't overshoot, so dU/dl has no jumps.
l = np.linspace(0, l_pts[-1], 2000)
A = PchipInterpolator(l_pts, A_pts)(l)


def shade_tunnel(ax):
    """Background shaped like the tunnel width (top view), scaled to the panel height."""
    top = ax.get_ylim()[1]
    xf = np.linspace(0, l_pts[-1], 2000)
    width = PchipInterpolator(l_pts, z_pts)(xf)/z_pts.max()*top
    # Working section colour first, then the contraction over it, so the boundary sits exactly at ws_l
    ax.fill_between(xf*1000, 0, width, color=WS_COLOUR, linewidth=0, zorder=0)
    ax.fill_between(xf*1000, 0, width, where=xf < ws_l, color=COMP_COLOUR, linewidth=0, zorder=0)


def thwaites_H(lam):
    """Shape factor H(lambda), curve fit to Thwaites' table (White, Viscous Fluid Flow)."""
    z = 0.25 - np.clip(lam, -0.09, 0.25)
    return 2.0 + 4.14*z - 83.5*z**2 + 854*z**3 - 3337*z**4 + 4576*z**5


def compute_layer(U_ws):
    U = U_ws*A_ws/A                       # continuity, uniform core flow

    # Laminar (Thwaites): theta^2 = 0.45 nu / U^6 * integral(U^5 dl), theta = 0 at the inlet
    theta = np.sqrt(0.45*nu*cumulative_trapezoid(U**5, l, initial=0)/U**6)
    lam   = theta**2/nu*np.gradient(U, l) # pressure-gradient parameter (> 0 = favourable)
    H     = thwaites_H(lam)
    delta = (4.91/0.664)*theta            # Thwaites gives no delta: Blasius ratio delta/theta = 7.4

    # Transition: first point where Re_theta reaches the criterion, turbulent from there on
    tripped = U*theta/nu >= Re_theta_tr
    if not tripped.any():
        return U, delta, H*theta, theta, lam, np.nan
    i_tr = np.argmax(tripped)
    turb = l >= l[i_tr]

    # Turbulent: 1/7th law from a virtual origin x0 that matches theta at transition
    x0 = (theta[i_tr]*(U[i_tr]/nu)**(1/7)/C_THETA)**(7/6)
    x_t = x0 + l[turb] - l[i_tr]
    theta[turb] = C_THETA*x_t/(U[turb]*x_t/nu)**(1/7)
    delta[turb] = theta[turb]*72/7
    H[turb] = 9/7

    return U, delta, H*theta, theta, lam, l[i_tr]


# --- RUN + TABLE ---
print(f"Contraction ratio = {A_pts[x_pts > 0.1].max()/A_ws:.2f}  |  Re_theta,tr = {Re_theta_tr:.0f}")
print(f"{'U_ws':>5} {'transition, into WS [mm]':>25} {'theta at WS start [mm]':>23} "
      f"{'delta at WS end [mm]':>21} {'d* at WS end [mm]':>18} {'area loss [%]':>14}")

fig, axs = plt.subplots(5, 1, sharex=True, figsize=(7, 11))
labels = ["$U$ [m/s]", r"$\lambda = (\theta^2/\nu)\,dU/dl$", r"$\delta$ [mm]", r"$\delta^*$ [mm]", r"$\theta$ [mm]"]
colors = plt.cm.viridis(np.linspace(0, 0.85, len(speeds)))

for U_ws, c in zip(speeds, colors):
    U, delta, dstar, theta, lam, l_tr = compute_layer(U_ws)

    # lambda only applies while laminar, so blank it from transition onwards.
    # In the laminar region it is the same for every speed (theta^2 ~ 1/U, dU/dl ~ U), so the curves overlap.
    lam_laminar = np.where(l >= l_tr, np.nan, lam)

    for ax, y in zip(axs, [U, lam_laminar, delta*1000, dstar*1000, theta*1000]):
        ax.plot(l*1000, y, color=c, label=f"{U_ws} m/s")
        ax.axvline(l_tr*1000, color=c, linestyle="--", linewidth=0.8)

    # Displacement effect: each wall pushes the flow in by delta*, so the square side shrinks by 2 delta*
    side = np.sqrt(A_ws)
    area_loss = 1 - ((side - 2*dstar[-1])/side)**2
    print(f"{U_ws:>5} {(l_tr - ws_l)*1000:>25.0f} {np.interp(ws_l, l, theta)*1000:>23.3f} "
          f"{delta[-1]*1000:>21.2f} {dstar[-1]*1000:>18.3f} {area_loss*100:>14.2f}")

for ax, lab in zip(axs, labels):
    shade_tunnel(ax)
    ax.set_ylabel(lab)
    ax.grid(alpha=0.3)
axs[1].axhline(0, color="0.5", linewidth=0.8)   # lambda > 0 = favourable gradient
axs[0].set_title("Boundary layer along the OBU wind tunnel (dashed = transition)")
axs[-1].set_xlabel("Distance along wall from inlet, $l$ [mm]")
fig.tight_layout()

# One legend for the whole figure, below the bottom panel: speed lines + section colours
handles = axs[0].get_legend_handles_labels()[0] + [Patch(color=COMP_COLOUR, label="Contraction"),
                                                   Patch(color=WS_COLOUR, label="Working section")]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0), ncol=3, frameon=False)

(BASE_DIR / "plots").mkdir(exist_ok=True)
fig.savefig(BASE_DIR / "plots" / "boundary_layer_summary.png", dpi=300, bbox_inches="tight")
