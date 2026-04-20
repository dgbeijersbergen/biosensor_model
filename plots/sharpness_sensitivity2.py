"""
Sharpness sensitivity analysis using simulate() with sharpness= kwarg.

Loops over the same parameter combinations as the validation script
(b_m sweep + Q sweep), runs each for alpha in alpha_vals, and compares
F, t_eq and V_req against the reference alpha.

Output: single 3-panel figure  sharpness_sensitivity.svg / .png
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

from biosensor.parameters.parameters_QCM import params
from biosensor.model.simulate_ODE import simulate
from biosensor.model.calculate_Sherwood import compute_k_m

# ══════════════════════════════════════════════════════════════
# SHARED FIXED PARAMETERS  (same as validation script)
# ══════════════════════════════════════════════════════════════
D     = 1e-11
H_c   = 1e-4
W_c   = 1e-4
W_s   = 1e-4
k_on  = 1e3
k_off = 1e-3
K_D   = k_off / k_on
V_in  = 1e-3

eq_factor = -np.log(0.05)   # factor to convert t_R → t_eq in reaction-limited case

# ══════════════════════════════════════════════════════════════
# PARAMETER COMBINATIONS
# ══════════════════════════════════════════════════════════════

# --- b_m sweep ---
bm_combinations = [
    # --- inside blending (central region) ---
    {"lambda": 1,   "c_ratio": 1, "Pe_H": 1,    "k_on": k_on, "b_m": 1e-8},   # Pe_s = 6
    {"lambda": 1,   "c_ratio": 1, "Pe_H": 10,   "k_on": k_on, "b_m": 1e-8},   # Pe_s = 60

    # --- near lower boundary ---
    {"lambda": 1,   "c_ratio": 1, "Pe_H": 0.002, "k_on": k_on, "b_m": 1e-8},  # Pe_s ≈ 0.012

    # --- near upper boundary ---
    {"lambda": 1,   "c_ratio": 1, "Pe_H": 16,   "k_on": k_on, "b_m": 1e-8},   # Pe_s ≈ 96

    # --- small λ (tests λ scaling) ---
    {"lambda": 0.1, "c_ratio": 1, "Pe_H": 100,  "k_on": k_on, "b_m": 1e-8},   # Pe_s = 6

    # --- large λ (tests λ scaling) ---
    {"lambda": 10,  "c_ratio": 1, "Pe_H": 0.1,  "k_on": k_on, "b_m": 1e-8},   # Pe_s = 6

]

b_m_vals = np.logspace(-13, -7, 1)

# --- Q sweep ---
q_systems = [
    # {"lambda": 10, "b_m_fixed": 1e-7, "k_on": 18},
    # {"lambda": 10, "b_m_fixed": 1e-8, "k_on": 18},
    # {"lambda": 10, "b_m_fixed": 1e-6, "k_on": 18},
]
Pe_H_vals_sweep = np.array([0.0002, 0.0057, 0.1790, 5.6727]) * 1e5

# ══════════════════════════════════════════════════════════════
# BUILD FLAT CASE LIST
# ══════════════════════════════════════════════════════════════

def make_params(lam, c_ratio, Pe_H_val, b_m, k_on_val, flow_off=False):
    """Return a fully configured params object for one case."""
    p = deepcopy(params)
    L_s = lam * H_c
    p.D       = D
    p.H_c     = H_c
    p.W_c     = W_c
    p.W_s     = W_s
    p.L_s     = L_s
    p.L_c     = L_s          # square channel cross-section
    p.k_on    = k_on_val
    p.k_off   = k_off
    p.b_m     = b_m
    p.c_in    = c_ratio * K_D
    p.c_0     = 0.0
    p.Q_in    = Pe_H_val * D * W_c
    p.V_in    = V_in
    p.flow_off = flow_off
    p.char_length = 'H'
    return p


cases = []   # list of dicts: {params, max_time, t_R, Pe_s, in_blending}

# b_m sweep
for combo in bm_combinations:
    lam      = combo["lambda"]
    c_in_val = combo["c_ratio"] * K_D
    Q_in_val = combo["Pe_H"] * D * W_c
    t_R      = 1.0 / (combo["k_on"] * c_in_val + k_off)
    Pe_s     = 6 * lam**2 * combo["Pe_H"]

    for b_m in b_m_vals:
        p = make_params(lam, combo["c_ratio"], combo["Pe_H"], b_m, combo["k_on"])
        k_m_est, _ = compute_k_m(Q_in_val, p)
        Da_est = combo["k_on"] * b_m * lam * H_c / (k_m_est * H_c) if k_m_est > 0 else 1
        max_time = max(10 * (1 + Da_est) * t_R, 100)

        cases.append({
            "params":      p,
            "max_time":    max_time,
            "t_R":         t_R,
            "Pe_s":        Pe_s,
            "Pe_H": combo["Pe_H"],
            "lambda": lam,
            "in_blending": 0.01 < Pe_s < 100,
            "sweep":       "b_m",
        })

# Q sweep
for cfg in q_systems:
    lam      = cfg["lambda"]
    c_in_val = K_D
    t_R      = 1.0 / (cfg["k_on"] * c_in_val + k_off)

    for Pe_H_val in Pe_H_vals_sweep:
        Q_in_val = Pe_H_val * D * W_c
        Pe_s     = 6 * lam**2 * Pe_H_val
        p = make_params(lam, 1.0, Pe_H_val, cfg["b_m_fixed"], cfg["k_on"])
        k_m_est, _ = compute_k_m(Q_in_val, p)
        Da_est = cfg["k_on"] * cfg["b_m_fixed"] * lam * H_c / (k_m_est * H_c) if k_m_est > 0 else 1
        max_time = max(10 * (1 + Da_est) * t_R, 100)

        cases.append({
            "params":      p,
            "max_time":    max_time,
            "t_R":         t_R,
            "Pe_s":        Pe_s,
            "in_blending": 0.01 < Pe_s < 100,
            "sweep":       "Q",
        })

print("\nBlending region cases:")
for i, c in enumerate(cases):
    if c["in_blending"]:
        print(f"i={i:3d} | λ={c['lambda']:.2g}, Pe_H={c['Pe_H']:.3e}, Pe_s={c['Pe_s']:.3e}")

n_blend = sum(1 for c in cases if c["in_blending"])
print(f"Total cases       : {len(cases)}")
print(f"In blending region: {n_blend}")
print(f"Outside blending  : {len(cases) - n_blend}")

# ══════════════════════════════════════════════════════════════
# HELPER: run one case and extract (F, t_eq_norm, V_req_norm)
# ══════════════════════════════════════════════════════════════

def run_case(case, sharpness):
    p = deepcopy(case["params"])
    try:
        r = simulate(p, print_results=False, plot_results=False,
                     max_time=case["max_time"], sharpness=sharpness)
    except Exception as e:
        return None

    t_R    = case["t_R"]
    t_eq   = r["time_eq"]
    F      = r["F"]
    Q_in   = r["Q_in"]
    k_on_v = r["k_on"]
    b_m_v  = r["b_m"]
    L_s_v  = r["L_s"]
    W_c_v  = r["W_c"]

    if not np.isfinite(t_eq):
        return None

    t_eq_norm = t_eq / (eq_factor * t_R)
    V_req     = t_eq * Q_in
    V_min     = k_on_v * b_m_v * L_s_v * W_c_v * t_R * eq_factor
    V_norm    = V_req / V_min if V_min > 0 else np.inf

    return {"F": F, "t_eq_norm": t_eq_norm, "V_norm": V_norm}

# ══════════════════════════════════════════════════════════════
# SWEEP
# ══════════════════════════════════════════════════════════════

alpha_vals = [2, 3, 4, 5, 6, 7, 8, 10]
ref_alpha  = 4

print(f"\nRunning reference (α = {ref_alpha}) ...")
ref_results = []
for case in tqdm(cases, desc=f"α={ref_alpha}"):
    ref_results.append(run_case(case, ref_alpha))

# Collect per-alpha deviations
stats = {}   # alpha → {dF, dt, dV, dF_blend, dt_blend}
raw = {}

for alpha in alpha_vals:
    dF, dt, dV        = [], [], []
    dF_blend, dt_blend = [], []

    for i, case in enumerate(tqdm(cases, desc=f"α={alpha}", leave=False)):
        ref = ref_results[i]
        if ref is None:
            continue

        if alpha == ref_alpha:
            # reference vs itself → 0 by definition
            dF.append(0.0); dt.append(0.0); dV.append(0.0)
            if case["in_blending"]:
                dF_blend.append(0.0); dt_blend.append(0.0)
            continue

        res = run_case(case, alpha)
        if res is None:
            continue

        if ref["F"] > 0:
            d = 100 * (res["F"] - ref["F"]) / ref["F"]
            dF.append(d)
            if case["in_blending"]:
                dF_blend.append(d)

        if ref["t_eq_norm"] > 0 and np.isfinite(ref["t_eq_norm"]):
            d = 100 * (res["t_eq_norm"] - ref["t_eq_norm"]) / ref["t_eq_norm"]
            dt.append(d)
            if case["in_blending"]:
                dt_blend.append(d)

        if ref["V_norm"] > 0 and np.isfinite(ref["V_norm"]) and np.isfinite(res["V_norm"]):
            d = 100 * (res["V_norm"] - ref["V_norm"]) / ref["V_norm"]
            dV.append(d)

    def _s(lst): return (np.mean(lst), np.max(np.abs(lst))) if lst else (0.0, 0.0)

    stats[alpha] = {
        "dF_mean":  _s(dF)[0],  "dF_max":  _s(dF)[1],
        "dt_mean":  _s(dt)[0],  "dt_max":  _s(dt)[1],
        "dV_mean":  _s(dV)[0],  "dV_max":  _s(dV)[1],
        "dF_blend_mean": _s(dF_blend)[0], "dF_blend_max": _s(dF_blend)[1],
        "dt_blend_mean": _s(dt_blend)[0], "dt_blend_max": _s(dt_blend)[1],
        "n": len(dF), "n_blend": len(dF_blend),
    }

    raw[alpha] = {
        "dF": dF_blend.copy(),
        "dt": dt_blend.copy(),
        "dV": dV.copy(),
    }

# ══════════════════════════════════════════════════════════════
# PRINT TABLE
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*85}")
print(f"Deviation from α = {ref_alpha}  (reference)   |  n = total valid cases")
print(f"{'='*85}")
print(f"{'α':>3} {'n':>5} │ {'ΔF mean':>9} {'max':>8} │ {'Δt_eq mean':>11} {'max':>8} │ {'ΔV mean':>9} {'max':>8}")
print(f"{'─'*85}")
for a in alpha_vals:
    s = stats[a]
    print(f"{a:3d} {s['n']:5d} │ "
          f"{s['dF_mean']:+8.3f}% {s['dF_max']:7.3f}% │ "
          f"{s['dt_mean']:+10.4f}% {s['dt_max']:7.3f}% │ "
          f"{s['dV_mean']:+8.4f}% {s['dV_max']:7.3f}%")

print(f"\nBlending region only  (Pe_s ∈ [0.01, 100]):")
print(f"{'α':>3} {'n':>5} │ {'ΔF mean':>9} {'max':>8} │ {'Δt_eq mean':>11} {'max':>8}")
print(f"{'─'*60}")
for a in alpha_vals:
    s = stats[a]
    print(f"{a:3d} {s['n_blend']:5d} │ "
          f"{s['dF_blend_mean']:+8.3f}% {s['dF_blend_max']:7.3f}% │ "
          f"{s['dt_blend_mean']:+10.4f}% {s['dt_blend_max']:7.3f}%")

# ══════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════

plt.rcParams.update({
    "font.size": 11, "axes.linewidth": 1.2,
    "xtick.direction": "in", "ytick.direction": "in",
    "font.family": "sans-serif",
})

col_all   = "#1f77b4"
col_blend = "#000000"
ref_col   = "#2ca02c"

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)
fig.suptitle(
    f"Sensitivity of model predictions to blending sharpness α  "
    f"(reference α = {ref_alpha})\n"
    f"{len(cases)} parameter combinations  |  {n_blend} in blending region  "
    f"(Pe_s ∈ [0.01, 100])",
    fontsize=11, y=1.03,
)

panel_cfg = [
    ("dF_mean",  "dF_max",  "dF_blend_mean",  "dF_blend_max",
     r"$\Delta F$ relative to α=" + str(ref_alpha) + r" [%]",
     "(a) Sherwood number"),
    ("dt_mean",  "dt_max",  "dt_blend_mean",  "dt_blend_max",
     r"$\Delta t_{eq}$ relative to α=" + str(ref_alpha) + r" [%]",
     "(b) Equilibration time"),
    ("dV_mean",  "dV_max",  "dV_mean", "dV_mean",
     r"$\Delta V_{req}$ relative to α=" + str(ref_alpha) + r" [%]",
     "(c) Required volume"),
]

key_map = {
    "dF_mean": "dF",
    "dt_mean": "dt",
    "dV_mean": "dV",
}

for ax, (mean_key, max_key, blend_mean_key, blend_max_key, ylabel, title) in zip(axes, panel_cfg):

    raw_key = key_map[mean_key]

    # --- scatter: individual blending points ---
    for a in alpha_vals:
        yvals = raw[a][raw_key]
        xvals = np.full(len(yvals), a)

        ax.scatter(
            xvals,
            yvals,
            color=col_blend,
            alpha=0.35,
            s=18,
            edgecolors="none",
        )

    # --- mean line (blending) ---
    if blend_mean_key is not None:
        blend_means = [stats[a][blend_mean_key] for a in alpha_vals]

        ax.plot(
            alpha_vals,
            blend_means,
            "*-",
            color=col_blend,
            lw=2,
            ms=7,
            label="Mean"
        )

    # --- reference + zero lines ---
    ax.axvline(ref_alpha, color=ref_col, ls="--", lw=1.5, alpha=0.7,
               label=f"Reference (α={ref_alpha})")
    ax.axhline(0, color="k", ls="-", lw=0.5, alpha=0.3)

    # --- labels ---
    ax.set_xlabel("Sharpness α", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(alpha_vals)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, framealpha=0.9)

plt.tight_layout()
plt.savefig("sharpness_sensitivity.svg", format="svg", bbox_inches="tight")
plt.savefig("sharpness_sensitivity.png", dpi=200,      bbox_inches="tight")
print("\nSaved  sharpness_sensitivity.svg / .png")

plt.show()

plt.close("all")