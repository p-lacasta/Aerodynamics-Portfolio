from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# --- SHARED STYLE (same as S1 and S2) ---
GREY = "0.5"

plt.rcParams.update({
    "figure.figsize":    (8, 5),
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


# --- IMPORT DATA ---

G = 9.81
RHO = 1.225

SCALE = 4  # model-to-full-size linear scale factor (1:4)

data_dir = Path(__file__).parent / "datasets" / "DrivAer experimental"

ESTATEBACK  = pd.read_csv(data_dir / "ESTATEBACK_Force.csv")
FASTBACK    = pd.read_csv(data_dir / "FASTBACK_Force.csv")
NOTCHBACK   = pd.read_csv(data_dir / "NOTCHBACK_Force.csv")

VEHICLES = {
    "FASTBACK" : FASTBACK,
    "NOTCHBACK": NOTCHBACK,
    "ESTATEBACK": ESTATEBACK,
}

MASS = {
    "FASTBACK" : 1713,
    "NOTCHBACK": 1681,
    "ESTATEBACK": 1754,
}

MASS_DISTRIBUTION = {
    "FASTBACK" : 49.3,
    "NOTCHBACK": 51.8,
    "ESTATEBACK": 49.3,
}

# Full-size values: lengths scale with SCALE, areas with SCALE squared
AREA = {name: data["Frontal Area (m^2)"].mean() * SCALE**2 for name, data in VEHICLES.items()}
WHEELBASE = {name: data["Wheelbase Length (m)"].mean() * SCALE for name, data in VEHICLES.items()}

SPEEDS = [20, 30, 40, 50]  # m/s

# One colour per vehicle, used in every figure (same viridis series colours as S1 and S2)
COLOURS = dict(zip(VEHICLES, plt.cm.viridis(np.linspace(0, 0.85, len(VEHICLES)))))

# Coefficient names for axis labels, written as in S2 (mathtext)
COEFF_LABELS = {"Cd": r"$C_d$", "Cl": r"$C_L$", "Cl_f": r"$C_{L,f}$", "Cl_r": r"$C_{L,r}$"}

FIGURE_DIR = Path(__file__).parent / "figures"


def aerobalance(vehiclce):
    Cd = np.mean(vehiclce["Drag Coefficient"])
    Cl = np.mean(vehiclce["Lift Coefficient"])

    Cm = np.mean(vehiclce["Pitch Coefficient"])
    Cl_f = Cl/2 + Cm
    Cl_r = Cl/2 - Cm

    ab = (Cl_f / (Cl_f + Cl_r)) * 100

    return Cd, Cl, Cl_f, Cl_r, ab

def dynamic_loads(vehicle, v):
    _, _, Cl_f, Cl_r, _ = aerobalance(VEHICLES[vehicle])
    W = MASS[vehicle]*G
    dist_f = MASS_DISTRIBUTION[vehicle]/100
    N_f = W*dist_f - (1/2)*RHO*AREA[vehicle]*Cl_f*v**2
    N_r = W*(1-dist_f) - (1/2)*RHO*AREA[vehicle]*Cl_r*v**2

    return N_f, N_r


# --- PLOT SETUP ---

def save(fig, name, retries=5):
    """Save to figures/. Retries because Windows can briefly lock a PNG that is open
    in a preview or being scanned/synced, which shows up as OSError 22."""
    FIGURE_DIR.mkdir(exist_ok=True)  # create the figures folder if it does not exist yet
    for attempt in range(retries):
        try:
            fig.savefig(FIGURE_DIR / f"{name}.png")
            return
        except OSError:
            if attempt == retries - 1:  # last try failed, so give up and raise the error
                raise
            time.sleep(0.5)  # wait half a second, then try again


# --- FIGURES ---

def plot_coefficients_bar(results):
    # Build a table with one row per vehicle and the four coefficients as columns
    table = pd.DataFrame.from_dict(results, orient="index")[["Cd", "Cl", "Cl_f", "Cl_r"]]

    fig, ax = plt.subplots()
    x = np.arange(table.shape[1])  # one x position per coefficient
    width = 0.25  # width of each bar

    for i, name in enumerate(table.index):
        # Offset each vehicle's bars so the three bars in a group sit side by side
        bars = ax.bar(x + (i - 1)*width, table.loc[name], width, color=COLOURS[name], label=name)
        # Write the value on top of each bar, with 3 decimal places, turned sideways so neighbours don't overlap
        ax.bar_label(bars, fmt="%.3f", padding=2, rotation=90)

    ax.margins(y=0.2)  # leave room above and below the bars for the sideways labels
    ax.set_xticks(x, [COEFF_LABELS[c] for c in table.columns])  # label each group with its coefficient name
    ax.set_ylabel("Coefficient [-]")
    ax.set_title("Aerodynamic coefficients by body shape")
    ax.axhline(0, color=GREY, linewidth=0.8)  # zero line, so negative bars are clear
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(VEHICLES))
    fig.tight_layout()
    save(fig, "A_coefficients_bar")

    return table


def plot_axle_load_change():
    # Continuous speed range from 0 to 60 m/s, for smooth curves
    v = np.linspace(0, 60, 121)

    fig, ax = plt.subplots()
    table = {"Speed (m/s)": v}  # keeps the plotted numbers so they can go to Excel

    for name in VEHICLES:
        N_f0, N_r0 = dynamic_loads(name, 0)  # static axle loads, at zero speed
        N_f, N_r = dynamic_loads(name, v)    # axle loads at every speed in v (works on arrays)

        # Change from static, as a percentage of the static load
        dN_f = (N_f - N_f0) / N_f0 * 100
        dN_r = (N_r - N_r0) / N_r0 * 100

        ax.plot(v, dN_f, color=COLOURS[name], linestyle="-", label=f"{name} front")   # front: solid
        ax.plot(v, dN_r, color=COLOURS[name], linestyle="--", label=f"{name} rear")   # rear: dashed

        table[f"{name} dN_f (%)"] = dN_f
        table[f"{name} dN_r (%)"] = dN_r

    ax.axhline(0, color=GREY, linewidth=0.8)
    ax.set_xlabel("Speed [m/s]")
    ax.set_ylabel("Change in axle load [% of static]")
    # Secondary axis on top in km/h: 1 m/s = 3.6 km/h, and back again
    top = ax.secondary_xaxis("top", functions=(lambda x: x*3.6, lambda x: x/3.6))
    top.set_xlabel("Speed [km/h]")
    ax.set_title("Change in axle load with speed")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(VEHICLES))
    fig.tight_layout()
    save(fig, "B_axle_load_change_vs_speed")

    return pd.DataFrame(table)


def plot_front_load_share():
    # Same continuous speed range as the axle load plot
    v = np.linspace(0, 60, 121)

    fig, ax = plt.subplots()
    table = {"Speed (m/s)": v}  # keeps the plotted numbers so they can go to Excel

    for name in VEHICLES:
        N_f, N_r = dynamic_loads(name, v)
        share = N_f / (N_f + N_r) * 100  # front share of the total normal load, in %
        ax.plot(v, share, color=COLOURS[name], label=name)

        # Static front share: MASS_DISTRIBUTION is already a percentage
        ax.axhline(MASS_DISTRIBUTION[name], color=COLOURS[name], linestyle=":", label=f"{name} static")

        table[f"{name} front share (%)"] = share

    ax.set_xlabel("Speed [m/s]")
    ax.set_ylabel("Front share of normal load [%]")
    ax.set_title("Front share of normal load with speed")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(VEHICLES))
    fig.tight_layout()
    save(fig, "C_front_load_share_vs_speed")

    return pd.DataFrame(table)


def plot_cl_r_correction(results):
    rows = {}
    for name in results:
        Cl_f = results[name]["Cl_f"]
        Cl_r = results[name]["Cl_r"]
        wf = MASS_DISTRIBUTION[name] / 100  # static front weight fraction (0 to 1)

        Cl_r_target = Cl_f * (1 - wf) / wf  # Cl_r that would split the aero lift like the weight
        dCl_r = Cl_r_target - Cl_r          # how much Cl_r must change to get there

        rows[name] = {"Cl_r target": Cl_r_target, "dCl_r": dCl_r}

    table = pd.DataFrame.from_dict(rows, orient="index")

    fig, ax = plt.subplots()
    bars = ax.bar(table.index, table["dCl_r"], color=[COLOURS[name] for name in table.index])
    ax.bar_label(bars, fmt="%.3f", padding=2)  # value on top of each bar
    ax.axhline(0, color=GREY, linewidth=0.8)
    ax.set_ylabel(r"Required change in $C_{L,r}$ [-]")
    ax.set_title(r"Rear lift correction to match the static weight distribution")
    fig.tight_layout()
    save(fig, "E_cl_r_correction")

    return table


# --- MAIN ---

results = {}

for name, data in VEHICLES.items():
    Cd, Cl, Cl_f, Cl_r, ab = aerobalance(data)
    results[name] = {"Cd": Cd, "Cl": Cl, "Cl_f": Cl_f, "Cl_r": Cl_r, "ab": ab}

df = pd.DataFrame.from_dict(results, orient="index")
df.index.name = "Vehicle"

loads = {}

for name in VEHICLES:
    for v in SPEEDS:
        N_f, N_r = dynamic_loads(name, v)
        loads[(name, v)] = {"N_f": N_f, "N_r": N_r}

loads_df = pd.DataFrame.from_dict(loads, orient="index")
loads_df.index.names = ["Vehicle", "Speed (m/s)"]

fig_a = plot_coefficients_bar(results)
fig_b = plot_axle_load_change()
fig_c = plot_front_load_share()
fig_e = plot_cl_r_correction(results)

with pd.ExcelWriter(Path(__file__).parent/"Body_shape_results.xlsx") as writer:
    df.to_excel(writer, sheet_name="Aero_balance")
    loads_df.to_excel(writer, sheet_name="Dynamic_loads")
    fig_a.to_excel(writer, sheet_name="Fig_A_coefficients")
    fig_b.to_excel(writer, sheet_name="Fig_B_axle_change", index=False)
    fig_c.to_excel(writer, sheet_name="Fig_C_front_share", index=False)
    fig_e.to_excel(writer, sheet_name="Fig_E_cl_r_correction")

print(results)
print(loads_df)
