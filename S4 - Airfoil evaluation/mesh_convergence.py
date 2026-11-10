from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# --- STYLE ---
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

FIGURE_DIR = Path(__file__).parent / "figures"


# --- MESH STUDY DATA (NACA0015, alpha = 8 deg, k-omega SST, low y+) ---
# Base size shrinks by 0.75 each mesh; inflation layers are fixed, so base size is used as h

MESHES = {
    "A-M1": {"base": 30.0,     "N":  48502, "Cl": -0.8543, "Cd": 0.04239, "time": 138.5},
    "A-M2": {"base": 22.5,     "N":  65494, "Cl": -0.8972, "Cd": 0.03576, "time": 155.6},
    "A-M3": {"base": 16.875,   "N":  89669, "Cl": -0.9174, "Cd": 0.03306, "time": 202.7},
    "A-M4": {"base": 12.65625, "N": 121991, "Cl": -0.9312, "Cd": 0.03148, "time": 223.5},
    "A-M5": {"base": 9.492188, "N": 168654, "Cl": -0.9386, "Cd": 0.03060, "time": 280.6},
    "A-M6": {"base": 7.119141, "N": 230354, "Cl": -0.9408, "Cd": 0.03019, "time": 409.7},
    "A-M7": {"base": 5.339355, "N": 311393, "Cl": -0.9390, "Cd": 0.02993, "time": 523.4},
    "A-M8": {"base": 4.004517, "N": 435514, "Cl": -0.9397, "Cd": 0.02980, "time": 726.1},
    "A-M9": {"base": 3.003387, "N": 603828, "Cl": -0.939922, "Cd": 0.029804, "time": 1404.0},
}

QUANTITIES = ["Cl", "Cd"]
P_FORMAL = 2          # formal order of the scheme (second order)
SAFETY_FACTOR = 1.25  # Roache's factor for three or more meshes
TARGET = 1.0          # acceptance criterion: uncertainty below 1 % for every quantity
STEP = 2              # use every second mesh, so r = (1/0.75)^2 = 1.78 (above the 1.3 minimum)


# --- GCI FUNCTIONS (Celik et al., 2008) ---

def observed_order(phi1, phi2, phi3, r21, r32, tol=1e-8, max_iter=100):
    """Observed order p by fixed-point iteration. 1 = fine, 2 = medium, 3 = coarse."""
    e21 = phi2 - phi1
    e32 = phi3 - phi2
    s = np.sign(e32 / e21)

    p = np.abs(np.log(np.abs(e32 / e21))) / np.log(r21)   # first guess, q = 0
    for i in range(max_iter):
        q = np.log((r21**p - s) / (r32**p - s))
        p_new = np.abs(np.log(np.abs(e32 / e21)) + q) / np.log(r21)
        if abs(p_new - p) < tol:
            return p_new
        p = p_new

    print("Warning: observed order did not converge")
    return p


def extrapolated(phi1, phi2, r21, p):
    """Richardson-extrapolated (zero cell size) value."""
    return (r21**p * phi1 - phi2) / (r21**p - 1)


def gci(phi_fine, phi_coarse, r, p):
    """Grid Convergence Index of the finer mesh of a pair, in %."""
    e_a = np.abs((phi_fine - phi_coarse) / phi_fine)
    return SAFETY_FACTOR * e_a / (r**p - 1) * 100


def analyse(triplet, key):
    """Run the full GCI procedure for one quantity on one triplet."""
    coarse, medium, fine = (MESHES[name] for name in triplet)
    phi1, phi2, phi3 = fine[key], medium[key], coarse[key]

    r21 = medium["base"] / fine["base"]
    r32 = coarse["base"] / medium["base"]
    R = (phi2 - phi1) / (phi3 - phi2)

    result = {"r21": r21, "r32": r32, "R": R}

    if 0 < R < 1:
        # Monotonic convergence: the GCI procedure applies
        p = observed_order(phi1, phi2, phi3, r21, r32)
        p_used = min(p, P_FORMAL)   # conservative: never use p above the formal order
        result["type"] = "monotonic"
        result["p"] = p
        result["p_used"] = p_used
        result["phi_ext"] = extrapolated(phi1, phi2, r21, p_used)
        result["GCI_obs"] = gci(phi1, phi2, r21, p)
        result["GCI"] = gci(phi1, phi2, r21, p_used)
    else:
        # Oscillatory (R < 0) or divergent (R > 1): half range of the three values
        values = [phi1, phi2, phi3]
        half_range = (max(values) - min(values)) / 2
        result["type"] = "oscillatory" if R < 0 else "divergent"
        result["phi_ext"] = np.mean(values)
        result["GCI"] = np.abs(half_range / phi1) * 100

    return result


def print_triplet(triplet, label):
    print(f"\n{label}: {' / '.join(triplet)} (coarse / medium / fine)")
    results = {}
    for key in QUANTITIES:
        res = analyse(triplet, key)
        results[key] = res
        print(f"  {key}: r21 = {res['r21']:.3f}, r32 = {res['r32']:.3f}, R = {res['R']:.3f} ({res['type']})")
        if res["type"] == "monotonic":
            print(f"      p observed = {res['p']:.2f}, p used = {res['p_used']:.2f}")
            print(f"      phi_ext = {res['phi_ext']:.5f}")
            print(f"      GCI fine = {res['GCI']:.2f} % (with observed p: {res['GCI_obs']:.2f} %)")
        else:
            print(f"      mean = {res['phi_ext']:.5f}, uncertainty (half range) = {res['GCI']:.2f} %")
    return results


# --- SUCCESSIVE CHANGE TABLE ---

print("Successive change between meshes (relative to the previous, coarser mesh)")
print(f"{'Mesh':6} {'N':>8} {'Cl':>8} {'Cd':>9} {'dCl [%]':>8} {'dCd [%]':>8} {'time [s]':>9}")
names = list(MESHES)
for i, name in enumerate(names):
    m = MESHES[name]
    if i == 0:
        dcl = dcd = ""
    else:
        prev = MESHES[names[i - 1]]
        dcl = f"{abs((m['Cl'] - prev['Cl']) / prev['Cl']) * 100:.2f}"
        dcd = f"{abs((m['Cd'] - prev['Cd']) / prev['Cd']) * 100:.2f}"
    print(f"{name:6} {m['N']:8d} {m['Cl']:8.4f} {m['Cd']:9.5f} {dcl:>8} {dcd:>8} {m['time']:9.1f}")


# --- MESH SELECTION ---
# Coarsest mesh with uncertainty below TARGET for every quantity

def uncertainty(triplet):
    """GCI for each quantity on a triplet, with the scatter check applied.
    If a quantity changes direction across the triplet's fine mesh and its
    neighbours (it wobbles, e.g. iterative noise), the GCI is too optimistic,
    so half the range of those meshes is used if it is larger."""
    res = {key: analyse(triplet, key) for key in QUANTITIES}
    i = names.index(triplet[2])
    nearby = names[max(i - 2, 0): i + 2]          # fine mesh, two coarser and one finer
    for key in QUANTITIES:
        values = np.array([MESHES[n][key] for n in nearby])
        steps = np.diff(values)
        res[key]["note"] = "GCI" if res[key]["type"] == "monotonic" else "half range of triplet"
        if np.any(np.sign(steps) != np.sign(steps[0])):
            scatter = (values.max() - values.min()) / 2 / abs(MESHES[triplet[2]][key]) * 100
            if scatter > res[key]["GCI"]:        # only replace the GCI if the wobble is larger
                res[key]["GCI"] = scatter
                res[key]["note"] = "half range of nearby meshes (wobbles)"
    return res


print(f"\nUncertainty of the fine mesh of each triplet (target < {TARGET:.1f} %)")
print(f"{'Triplet':22} {'Cl [%]':>8} {'Cd [%]':>8}  Pass?")
triplets = [tuple(names[c:c + 2 * STEP + 1:STEP]) for c in range(len(names) - 2 * STEP)]
selected_triplet = None
for t in triplets:
    res = uncertainty(t)
    passed = all(res[k]["GCI"] < TARGET for k in QUANTITIES)
    print(f"{' / '.join(t):22} {res['Cl']['GCI']:8.2f} {res['Cd']['GCI']:8.2f}  {'yes' if passed else 'no'}")
    if passed and selected_triplet is None:
        selected_triplet = t                     # first (coarsest) triplet that passes

MAIN_TRIPLET = selected_triplet
later = triplets.index(MAIN_TRIPLET) + 1
CHECK_TRIPLET = triplets[later] if later < len(triplets) else None
print(f"\nSelected final mesh: {MAIN_TRIPLET[2]} (GCI triplet {' / '.join(MAIN_TRIPLET)})")


# --- GCI RESULTS FOR THE SELECTED MESH ---

print_triplet(MAIN_TRIPLET, "Main triplet")
if CHECK_TRIPLET:
    print_triplet(CHECK_TRIPLET, "Check triplet")
main = uncertainty(MAIN_TRIPLET)

print(f"\nReported uncertainty for {MAIN_TRIPLET[2]}")
for key in QUANTITIES:
    print(f"  {key}: {main[key]['GCI']:.2f} % ({main[key]['note']})")

finest = names[-1]
sel = MESHES[MAIN_TRIPLET[2]]
print(f"\nChange from {MAIN_TRIPLET[2]} to the finest mesh {finest}:")
for key in QUANTITIES:
    change = abs((MESHES[finest][key] - sel[key]) / sel[key]) * 100
    print(f"  {key}: {change:.2f} %")


# --- FIGURE ---

h = np.array([m["base"] for m in MESHES.values()])
selected = MAIN_TRIPLET[2]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, key, label in zip(axes, QUANTITIES, [r"$C_l$", r"$C_d$"]):
    values = np.array([m[key] for m in MESHES.values()])
    ax.plot(h, values, "o-", color="tab:blue", label="CFD meshes")

    # Selected mesh, with its uncertainty band as an error bar
    sel = MESHES[selected]
    err = abs(sel[key]) * main[key]["GCI"] / 100
    ax.errorbar(sel["base"], sel[key], yerr=err, fmt="s", color="tab:red",
                capsize=4, label=f"{selected} (selected) ± uncertainty")

    # Extrapolated value at zero cell size, if the convergence was monotonic
    if main[key]["type"] == "monotonic":
        ax.plot(0, main[key]["phi_ext"], "D", color="black", label=r"$\phi_{ext}$ (h = 0)")

    ax.set_xlim(left=-1)
    ax.set_xlabel("Base cell size h [mm]")
    ax.set_ylabel(f"{label} [-]")
    ax.set_title(f"{label} against cell size")
    ax.legend()

fig.suptitle(r"NACA0015 mesh convergence, $\alpha$ = 8°")
fig.tight_layout()
FIGURE_DIR.mkdir(exist_ok=True)
fig.savefig(FIGURE_DIR / "mesh_convergence.png")
print(f"\nFigure saved to {FIGURE_DIR / 'mesh_convergence.png'}")
