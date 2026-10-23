import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch
import time

# --- SHARED STYLE (matches the coefficient sensitivity figures) ---
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
    """Save to plots/. Retries because Windows can briefly lock a PNG that is open
    in a preview or being synced, which shows up as OSError 22."""
    PLOT_DIR.mkdir(exist_ok=True)
    for attempt in range(retries):
        try:
            fig.savefig(PLOT_DIR / f"{name}.png", bbox_inches="tight", dpi=300)
            return
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(0.5)


A_ws    = 0.31*0.31
nu      = 0.000015
Re_tr   = 500000
ws_x    = 1.125

PROFILE_XLSX = BASE_DIR / "Armfield Wind Tunnel Inlet Contraction and Working Section Profile.xlsx"

wt_x = np.array(pd.read_excel(PROFILE_XLSX, skiprows=2, usecols="D", nrows=76)).ravel()/1000
wt_y = np.array(pd.read_excel(PROFILE_XLSX, skiprows=2, usecols="E", nrows=76)).ravel()/1000
wt_z = np.array(pd.read_excel(PROFILE_XLSX, skiprows=2, usecols="F", nrows=76)).ravel()/1000

# Length along the tunnel: cumulative arc length of the x-y profile, starting at the inlet
dl = np.hypot(np.diff(wt_x), np.diff(wt_y))
l = np.concatenate([[0], np.cumsum(dl)])
# Start of the working section expressed as a length, not an x coordinate
ws_l = np.interp(ws_x, wt_x, l)

speeds = np.array([10, 20, 30, 40])   # [m/s]
colors = plt.cm.viridis(np.linspace(0, 0.85, len(speeds)))

def laminar_boundary_layer(Re_x,wt_x):
    delta = wt_x*4.91/np.sqrt(Re_x)
    delta_star = wt_x*1.72/np.sqrt(Re_x)
    theta = wt_x*0.664/np.sqrt(Re_x)
    return delta, delta_star, theta

def turbulent_boundary_layer(Re_x,wt_x):
    delta = wt_x*0.16/Re_x**(1/7)
    delta_star = wt_x*0.020/Re_x**(1/7)
    theta = wt_x*0.016/Re_x**(1/7)
    return delta, delta_star, theta

def compute_layer(U_ws):
    # Area(x)
    A_x = wt_y*wt_z*4

    # Speed(x)
    U_x = U_ws*(A_ws/A_x)

    # Reynolds(l)
    Re_x = (U_x*l)/nu

    # Boundary layer
    is_turbulent = Re_x >= Re_tr
    i_tr = np.argmax(is_turbulent)
    # Turbulent transition length (NaN if the flow never turns turbulent)
    x_tr = l[i_tr] if is_turbulent.any() else np.nan


    U_tr = U_x[i_tr]
    theta_tr = l[i_tr]*0.664/np.sqrt(Re_x[i_tr])
    origin_turb = ((theta_tr*(U_tr/nu)**(1/7))/0.016)**(7/6)
    x_turb = origin_turb + (l-x_tr)
    Re_turb = (U_x*x_turb)/nu

    delta, delta_star, theta = np.where(is_turbulent, turbulent_boundary_layer(Re_turb, x_turb), laminar_boundary_layer(Re_x, l))

    return (U_x, Re_x, delta, delta_star, theta, is_turbulent, x_tr)


# Convert to mm for readability
l_mm = l*1000
z_mm = wt_z*1000  # lateral width, used for the top-view shading


def shade_tunnel(ax, x, z):
    # Background bands behind the lines, following the tunnel width along x.
    # Scaled to the plot height so it doesn't squash the thickness curves.
    # Working section is filled first, then the compression section is drawn on
    # a fine grid over it, so the boundary sits exactly at ws_x with no gap.
    top = ax.get_ylim()[1]
    xf = np.linspace(x.min(), x.max(), 2000)
    width = np.interp(xf, x, z)/z.max()*top
    compression = xf < ws_l*1000
    ax.fill_between(xf, 0, width, color=WS_COLOUR, linewidth=0, zorder=0)
    ax.fill_between(xf, 0, width, where=compression, color=COMP_COLOUR, linewidth=0, zorder=0)

    # Second legend inside the plot, with a white background; kept when the speed legend is drawn below
    handles = [Patch(color=COMP_COLOUR, label="Compression section"),
               Patch(color=WS_COLOUR, label="Working section")]
    ax.add_artist(ax.legend(handles=handles, loc="lower right", frameon=True,
                            facecolor="white", edgecolor="none", framealpha=0.9))


# Local velocity
fig0, ax0 = plt.subplots()
# Boundary layer thickness
fig1, ax1 = plt.subplots()
# Displacement thickness
fig2, ax2 = plt.subplots()
# Momentum thickness
fig3, ax3 = plt.subplots()
# Shape factor H = delta_star/theta (about 2.59 laminar, 1.25 turbulent)
fig4, ax4 = plt.subplots()

# One curve per freestream speed on each plot
for U, c in zip(speeds, colors):
    U_x, _, delta, delta_star, theta, _, x_tr = compute_layer(U)
    H = delta_star/theta

    ax0.plot(l_mm, U_x.ravel(), color=c, label=f"{U} m/s")
    ax0.axvline(x_tr*1000, color=c, linestyle="--", linewidth=0.8)

    ax1.plot(l_mm, delta.ravel()*1000, color=c, label=f"{U} m/s")
    ax1.axvline(x_tr*1000, color=c, linestyle="--", linewidth=0.8)

    ax2.plot(l_mm, delta_star.ravel()*1000, color=c, label=f"{U} m/s")
    ax2.axvline(x_tr*1000, color=c, linestyle="--", linewidth=0.8)

    ax3.plot(l_mm, theta.ravel()*1000, color=c, label=f"{U} m/s")
    ax3.axvline(x_tr*1000, color=c, linestyle="--", linewidth=0.8)

    ax4.plot(l_mm, H.ravel(), color=c, label=f"{U} m/s")
    ax4.axvline(x_tr*1000, color=c, linestyle="--", linewidth=0.8)

for ax in (ax0, ax1, ax2, ax3, ax4):
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

save(fig0, "local_velocity")
save(fig1, "boundary_layer_thickness")
save(fig2, "displacement_thickness")
save(fig3, "momentum_thickness")
save(fig4, "shape_factor")
print(f"Saved 5 figures to {PLOT_DIR}")