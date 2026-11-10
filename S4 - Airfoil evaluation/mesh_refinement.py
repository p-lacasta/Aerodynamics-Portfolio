import numpy as np

# --- MESH DATA ---
A_DOMAIN = 1.5 * 0.3  # [m^2] 2D fluid domain area

# Mesh results: 1 = fine, 2 = medium, 3 = coarse
coarse = {   # A-M3
    "N":  89669,
    "Cl": -0.9174,
    "Cd": 0.03306,
}

medium = {   # A-M5
    "N":  168654,
    "Cl": -0.9386,
    "Cd": 0.03060,
}

fine = {     # A-M7
    "N":  311393,
    "Cl": -0.9390,
    "Cd": 0.02993,
}


# --- GCI FUNCTIONS (Celik et al., 2008) ---

def cell_size(A_domain, N):
    """Representative 2D cell size h = sqrt(A / N)."""
    return np.sqrt(A_domain / N)


def ref_ratio(h_coarser, h_finer):
    """Refinement ratio r = h_coarser / h_finer (always > 1)."""
    return h_coarser / h_finer


def epsilon(phi_coarser, phi_finer):
    """Difference between solutions on two meshes."""
    return phi_coarser - phi_finer


def observed_order(phi1, phi2, phi3, r21, r32, tol=1e-6, max_iter=100):
    """Observed order p and iteration count, by fixed-point iteration. 1 = fine, 2 = medium, 3 = coarse."""
    e21 = epsilon(phi2, phi1)
    e32 = epsilon(phi3, phi2)
    s = np.sign(e32 / e21)

    p = np.abs(np.log(np.abs(e32 / e21))) / np.log(r21)   # first guess, q = 0

    for i in range(max_iter):
        q = np.log((r21**p - s) / (r32**p - s))
        p_new = np.abs(np.log(np.abs(e32 / e21)) + q) / np.log(r21)
        if abs(p_new - p) < tol:
            return p_new, i + 1
        p = p_new

    print("Warning: observed order did not converge")
    return p, max_iter


# --- RESULTS ---
# Cell sizes and refinement ratios
h1 = cell_size(A_DOMAIN, fine["N"])
h2 = cell_size(A_DOMAIN, medium["N"])
h3 = cell_size(A_DOMAIN, coarse["N"])

r21 = ref_ratio(h2, h1)
r32 = ref_ratio(h3, h2)

print(f"h1 = {h1:.5e} m, h2 = {h2:.5e} m, h3 = {h3:.5e} m")
print(f"r21 = {r21:.4f}, r32 = {r32:.4f}")
print()

# Observed order for each quantity
for key in ["Cl", "Cd"]:
    phi1, phi2, phi3 = fine[key], medium[key], coarse[key]
    R = epsilon(phi2, phi1) / epsilon(phi3, phi2)
    p, n_iter = observed_order(phi1, phi2, phi3, r21, r32)
    print(f"{key}: R = {R:.4f}, p = {p:.4f} ({n_iter} iterations)")