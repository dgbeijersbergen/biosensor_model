"""
Sherwood sharpness sensitivity across diverse geometries and flow regimes.

Uses the same parameter combinations as the model validation (b_m sweep + Q sweep)
and runs each for alpha = 1..10. Compares t_eq and V_req against alpha=6 reference.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from dataclasses import dataclass, replace
from tqdm import tqdm
from biosensor.model.calculate_Sherwood import *

# ══════════════════════════════════════════════════════
# Sherwood with variable sharpness
# ══════════════════════════════════════════════════════

def F_combine_alpha(Pe_H, lambda_ratio, sharpness, Pe_H_cutoff=1e-2, Pe_s_low=1e-2, Pe_s_high=1e2):
    Pe_s = 6 * (lambda_ratio ** 2) * Pe_H
    if Pe_H <= Pe_H_cutoff:
        return F_retained(Pe_H)
    F_small = F_Ackerberg(Pe_s)
    F_large = F_Newman(Pe_s)
    if Pe_s <= Pe_s_low:
        return F_small
    if Pe_s >= Pe_s_high:
        return F_large
    kappa = (np.log10(Pe_s) - np.log10(Pe_s_low)) / (np.log10(Pe_s_high) - np.log10(Pe_s_low))
    omega = smoothstep(kappa, sharpness)
    return blend_functions(F_small, F_large, omega)


def compute_k_m_alpha(Q_in, params, sharpness):
    D, W_c, L_s, H_c = params.D, params.W_c, params.L_s, params.H_c
    Pe_H = Q_in / (D * W_c) if Q_in > 0 else 0
    Lambda = L_s / H_c
    F = F_combine_alpha(Pe_H, Lambda, sharpness) if Pe_H > 1e-2 else (Pe_H if Pe_H > 0 else 0)
    if Pe_H > 0 and F > Pe_H:
        F = Pe_H

    b_eq = params.b_m * (params.k_on * params.c_in) / (params.k_on * params.c_in + params.k_off)
    fraction = F / Pe_H if Pe_H > 0 else 1
    V = params.W_s * params.H_c * params.L_s
    N_sensor = b_eq * L_s * params.W_s
    N_depletion = fraction * V * params.c_in
    #if not (N_depletion * 1e1 < N_sensor):
    #    F = Pe_H
    #    print("here")
    k_m = F * (D / H_c)
    return k_m, F


# ══════════════════════════════════════════════════════
# ODE system (same as biosensor_model)
# ══════════════════════════════════════════════════════

def ode_binding(t_hat, y, params, sharpness):
    b_hat, c_hat, c_s_hat, N_out1, N_out2 = y
    W_c, L_c, H_c = params.W_c, params.L_c, params.H_c
    k_on, k_off = params.k_on, params.k_off
    b_m, L_s, W_s = params.b_m, params.L_s, params.W_s
    c_in, V_in, Q_in = params.c_in, params.V_in, params.Q_in

    S = L_s * W_s
    V = W_c * L_c * H_c
    tau = V / Q_in
    gamma = (S * b_m) / (V * c_in)
    t_pulse_hat = V_in / V

    injecting = (t_hat < t_pulse_hat)
    Q_eff = Q_in if injecting else 0.0
    k_m, F = compute_k_m_alpha(Q_eff, params, sharpness)

    db_hat_dt = tau * (k_on * c_in * c_s_hat * (1 - b_hat) - k_off * b_hat)
    b_eq_hat = (k_on * c_in) / (k_on * c_in + k_off)
    if (b_hat >= 1.0 or b_hat > b_eq_hat) and db_hat_dt > 0:
        db_hat_dt = 0.0

    J_R = tau * (k_on * c_s_hat * b_m * (1 - b_hat) - (k_off * b_m * b_hat) / c_in)
    J_D = tau * (k_m / L_s)

    if injecting or not params.flow_off:
        dcs_hat_dt = J_D - (1 / H_c) * J_R - c_s_hat
        dc_hat_dt = (1.0 if injecting else 0.0) - J_D - c_hat
        dN1 = c_hat
        dN2 = c_s_hat
    else:
        dcs_hat_dt = -gamma * db_hat_dt
        dc_hat_dt = 0.0
        dN1 = 0.0
        dN2 = 0.0

    return [db_hat_dt, dc_hat_dt, dcs_hat_dt, dN1, dN2]


def run_sim(params, sharpness, max_time):
    W_c, L_c, H_c = params.W_c, params.L_c, params.H_c
    k_on, k_off, b_m = params.k_on, params.k_off, params.b_m
    L_s, W_s = params.L_s, params.W_s
    c_in, V_in, Q_in = params.c_in, params.V_in, params.Q_in

    V = W_c * L_c * H_c
    S = L_s * W_s
    tau = V / Q_in
    t_pulse_hat = V_in / V

    k_m, F = compute_k_m_alpha(Q_in, params, sharpness)
    J_in = c_in * Q_in
    J_D = W_c * H_c * c_in * k_m
    fill_frac = J_D / J_in if (k_m > 0 and c_in > 0) else 1.0
    params.fill_frac = fill_frac

    t_span_hat = (0, max_time / tau)
    y0 = [0, 0, 0, 0, 0]  # c_0 = 0
    t_eval = np.linspace(t_span_hat[0], t_span_hat[1], 20000)

    sol = solve_ivp(ode_binding, t_span_hat, y0, method='LSODA',
                    t_eval=t_eval, args=(params, sharpness), rtol=1e-5, atol=1e-8)

    t = sol.t * tau
    b_hat = sol.y[0]

    c_eff = c_in * fill_frac
    b_eq1 = (k_on * c_in * b_m) / (k_on * c_in + k_off)
    b_eq_sim = (k_on * c_eff * b_m) / (k_on * c_eff + k_off)
    frac_b_eq = b_eq1 / b_eq_sim if b_eq_sim > 0 else 1.0
    b = b_hat * b_m
    mol_capt = b * S
    mol_eq = b_eq1 * S

    # 95% equilibration time
    idx_eq = np.where(mol_capt >= 0.95 * mol_eq)[0]
    time_eq = t[idx_eq[0]] if len(idx_eq) > 0 else np.inf

    return {"F": F, "k_m": k_m, "time_eq": time_eq, "b_final": b[-1]}


# ══════════════════════════════════════════════════════
# Parameter sweep definitions (from validation script)
# ══════════════════════════════════════════════════════

@dataclass
class Params:
    W_c: float; L_c: float; H_c: float; D: float
    k_on: float; k_off: float; b_m: float; L_s: float; W_s: float
    c_0: float; c_in: float; V_in: float; Q_in: float
    flow_off: bool; char_length: str
    fill_frac: float = 1.0

# Fixed parameters (same as validation)
D = 1e-11
H_c = 1e-4
W_c = 1e-4
W_s = 1e-4
k_on_base = 1e3
k_off = 1e-3
K_D = k_off / k_on_base
V_in = 1e-3

# b_m sweep combinations (from validation)
bm_combinations = [
    {"lambda": 0.1, "c_ratio": 1,    "Pe_H": 0.1, "k_on": k_on_base},
    {"lambda": 1,   "c_ratio": 1,    "Pe_H": 1,   "k_on": k_on_base},
    {"lambda": 10,  "c_ratio": 1,    "Pe_H": 1,   "k_on": k_on_base},
    {"lambda": 10,  "c_ratio": 1,    "Pe_H": 10,  "k_on": k_on_base},
    {"lambda": 1,   "c_ratio": 0.01, "Pe_H": 1,   "k_on": k_on_base},
    {"lambda": 1,   "c_ratio": 100,  "Pe_H": 1,   "k_on": k_on_base},
    {"lambda": 100, "c_ratio": 1,    "Pe_H": 10,  "k_on": k_on_base},
    {"lambda": 100, "c_ratio": 1,    "Pe_H": 100, "k_on": k_on_base},
]

# Additional cases that exercise the blending region (low lambda, high Pe_H)
# lambda=0.1: Pe_s = 0.06*Pe_H => blending for Pe_H in [0.17, 1667]
# These have F << Pe_H so the complete delivery cap doesn't mask the effect
blending_combinations = [
    {"lambda": 0.1, "c_ratio": 1,  "Pe_H": 10,   "k_on": k_on_base},
    {"lambda": 0.1, "c_ratio": 1,  "Pe_H": 50,   "k_on": k_on_base},
    {"lambda": 0.1, "c_ratio": 1,  "Pe_H": 200,  "k_on": k_on_base},
    {"lambda": 0.1, "c_ratio": 1,  "Pe_H": 500,  "k_on": k_on_base},
    {"lambda": 0.1, "c_ratio": 1,  "Pe_H": 1000, "k_on": k_on_base},
]
b_m_vals = np.logspace(-13, -7, 13)

# Q sweep combinations
q_systems = [
    {"lambda": 10, "b_m_fixed": 1e-7,  "k_on": 18},
    {"lambda": 10, "b_m_fixed": 1e-8,  "k_on": 18},
    {"lambda": 10, "b_m_fixed": 1e-6,  "k_on": 18},
]
Pe_H_vals_sweep = np.array([0.0002, 0.0057, 0.1790, 5.6727]) * 1e5

# ══════════════════════════════════════════════════════
# Build flat list of all (combo, params) to simulate
# ══════════════════════════════════════════════════════

eq_factor = -np.log(0.05)
cases = []

# b_m sweep
for combo in bm_combinations:
    lam = combo["lambda"]
    L_s = lam * H_c
    c_in = combo["c_ratio"] * K_D
    Q_in = combo["Pe_H"] * D * W_c
    k_on_use = combo["k_on"]
    t_R = 1.0 / (k_on_use * c_in + k_off)

    for b_m in b_m_vals:
        p = Params(
            W_c=W_c, L_c=L_s, H_c=H_c, D=D,
            k_on=k_on_use, k_off=k_off, b_m=b_m,
            L_s=L_s, W_s=W_s,
            c_0=0, c_in=c_in, V_in=V_in,
            Q_in=Q_in, flow_off=False, char_length='H'
        )
        # Compute Da for max_time estimation
        k_m_est, _ = compute_k_m_alpha(Q_in, p, 4)
        Da_est = k_on_use * b_m * L_s / (k_m_est * H_c) if k_m_est > 0 else 1
        max_time = max(10 * (1 + Da_est) * t_R, 100)

        Pe_s = 6 * lam**2 * combo["Pe_H"]
        cases.append({
            "params": p, "max_time": max_time, "t_R": t_R,
            "lambda": lam, "Pe_H": combo["Pe_H"], "Pe_s": Pe_s,
            "sweep": "b_m", "b_m": b_m,
            "in_blending": 0.01 < Pe_s < 100
        })

# Blending-region cases (high b_m only, to ensure epsilon valid)
for combo in blending_combinations:
    lam = combo["lambda"]
    L_s = lam * H_c
    c_in = combo["c_ratio"] * K_D
    Q_in = combo["Pe_H"] * D * W_c
    k_on_use = combo["k_on"]
    t_R = 1.0 / (k_on_use * c_in + k_off)

    for b_m in [1e-9, 1e-8, 1e-7]:  # high b_m for valid epsilon
        p = Params(
            W_c=W_c, L_c=L_s, H_c=H_c, D=D,
            k_on=k_on_use, k_off=k_off, b_m=b_m,
            L_s=L_s, W_s=W_s,
            c_0=0, c_in=c_in, V_in=V_in,
            Q_in=Q_in, flow_off=False, char_length='H'
        )
        k_m_est, _ = compute_k_m_alpha(Q_in, p, 6)
        Da_est = k_on_use * b_m * L_s / (k_m_est * H_c) if k_m_est > 0 else 1
        max_time = max(10 * (1 + Da_est) * t_R, 100)

        Pe_s = 6 * lam**2 * combo["Pe_H"]
        cases.append({
            "params": p, "max_time": max_time, "t_R": t_R,
            "lambda": lam, "Pe_H": combo["Pe_H"], "Pe_s": Pe_s,
            "sweep": "blending", "b_m": b_m,
            "in_blending": 0.01 < Pe_s < 100
        })

# Q sweep combinations
for cfg in q_systems:
    lam = cfg["lambda"]
    L_s = lam * H_c
    c_in = K_D
    k_on_use = cfg["k_on"]
    b_m = cfg["b_m_fixed"]
    t_R = 1.0 / (k_on_use * c_in + k_off)

    for Pe_H_val in Pe_H_vals_sweep:
        Q_in = Pe_H_val * D * W_c
        p = Params(
            W_c=W_c, L_c=L_s, H_c=H_c, D=D,
            k_on=k_on_use, k_off=k_off, b_m=b_m,
            L_s=L_s, W_s=W_s,
            c_0=0, c_in=c_in, V_in=V_in,
            Q_in=Q_in, flow_off=False, char_length='H'
        )
        k_m_est, _ = compute_k_m_alpha(Q_in, p, 4)
        Da_est = k_on_use * b_m * L_s / (k_m_est * H_c) if k_m_est > 0 else 1
        max_time = max(10 * (1 + Da_est) * t_R, 100)

        Pe_s = 6 * lam**2 * Pe_H_val
        cases.append({
            "params": p, "max_time": max_time, "t_R": t_R,
            "lambda": lam, "Pe_H": Pe_H_val, "Pe_s": Pe_s,
            "sweep": "Q", "b_m": b_m,
            "in_blending": 0.01 < Pe_s < 100
        })

n_cases = len(cases)
n_blend = sum(1 for c in cases if c["in_blending"])
print(f"Total cases: {n_cases}")
print(f"In blending region (Pe_s ∈ [0.01, 100]): {n_blend}")
print(f"Outside blending (α has no effect): {n_cases - n_blend}")

# ══════════════════════════════════════════════════════
# Run sweep over alpha values
# ══════════════════════════════════════════════════════

alpha_vals = [1, 2, 3, 4, 5, 6, 7, 8, 10]
ref_alpha = 4

# First run reference (alpha=4)
print(f"\nRunning reference (α={ref_alpha})...")
ref_results = []
for case in tqdm(cases, desc=f"α={ref_alpha}"):
    p = replace(case["params"])
    try:
        r = run_sim(p, ref_alpha, case["max_time"])
        t_eq_norm = r["time_eq"] / (eq_factor * case["t_R"])
        V_req = r["time_eq"] * case["params"].Q_in
        V_min = case["params"].k_on * case["b_m"] * case["params"].L_s * W_c * case["t_R"] * eq_factor
        V_norm = V_req / V_min if V_min > 0 else np.inf
        ref_results.append({
            "F": r["F"], "t_eq": r["time_eq"], "t_eq_norm": t_eq_norm,
            "V_norm": V_norm, "b_final": r["b_final"],
            "reached": r["time_eq"] < np.inf,
            "in_blending": case["in_blending"],
            "Pe_s": case["Pe_s"], "lambda": case["lambda"]
        })
    except:
        ref_results.append({
            "F": np.nan, "t_eq": np.inf, "t_eq_norm": np.inf,
            "V_norm": np.inf, "b_final": np.nan,
            "reached": False, "in_blending": case["in_blending"],
            "Pe_s": case["Pe_s"], "lambda": case["lambda"]
        })

# Now run for each alpha and compare
print("\nRunning alpha sweep...")
alpha_stats = {}

for alpha in alpha_vals:
    if alpha == ref_alpha:
        # Reference against itself
        alpha_stats[alpha] = {
            "dF_mean": 0, "dF_max": 0, "dF_median": 0,
            "dt_mean": 0, "dt_max": 0, "dt_median": 0,
            "dV_mean": 0, "dV_max": 0, "dV_median": 0,
            "db_mean": 0, "db_max": 0, "db_median": 0,
            "n_valid": sum(1 for r in ref_results if r["reached"]),
            # Blending subset
            "dF_mean_blend": 0, "dF_max_blend": 0,
            "dt_mean_blend": 0, "dt_max_blend": 0,
            "n_blend": n_blend,
        }
        continue

    dF_all, dt_all, dV_all, db_all = [], [], [], []
    dF_blend, dt_blend = [], []

    for i, case in enumerate(tqdm(cases, desc=f"α={alpha}", leave=False)):
        ref = ref_results[i]
        if not ref["reached"] or np.isnan(ref["F"]):
            continue

        p = replace(case["params"])
        try:
            r = run_sim(p, alpha, case["max_time"])
        except:
            continue

        if r["time_eq"] >= np.inf or ref["t_eq"] >= np.inf:
            continue

        # F deviation
        if ref["F"] > 0:
            dF = 100 * (r["F"] - ref["F"]) / ref["F"]
            dF_all.append(dF)
            if case["in_blending"]:
                dF_blend.append(dF)

        # t_eq deviation
        t_eq_norm = r["time_eq"] / (eq_factor * case["t_R"])
        if ref["t_eq_norm"] > 0 and np.isfinite(ref["t_eq_norm"]):
            dt = 100 * (t_eq_norm - ref["t_eq_norm"]) / ref["t_eq_norm"]
            dt_all.append(dt)
            if case["in_blending"]:
                dt_blend.append(dt)

        # V_req deviation
        V_req = r["time_eq"] * case["params"].Q_in
        V_min = case["params"].k_on * case["b_m"] * case["params"].L_s * W_c * case["t_R"] * eq_factor
        V_norm = V_req / V_min if V_min > 0 else np.inf
        if ref["V_norm"] > 0 and np.isfinite(ref["V_norm"]) and np.isfinite(V_norm):
            dV = 100 * (V_norm - ref["V_norm"]) / ref["V_norm"]
            dV_all.append(dV)

        # b_final deviation
        if ref["b_final"] > 0 and not np.isnan(ref["b_final"]):
            db = 100 * (r["b_final"] - ref["b_final"]) / ref["b_final"]
            db_all.append(db)

    alpha_stats[alpha] = {
        "dF_mean": np.mean(dF_all) if dF_all else 0,
        "dF_max": np.max(dF_all) if dF_all else 0,
        "dF_median": np.median(dF_all) if dF_all else 0,
        "dt_mean": np.mean(dt_all) if dt_all else 0,
        "dt_max": np.max(dt_all) if dt_all else 0,
        "dt_median": np.median(dt_all) if dt_all else 0,
        "dV_mean": np.mean(dV_all) if dV_all else 0,
        "dV_max": np.max(dV_all) if dV_all else 0,
        "dV_median": np.median(dV_all) if dV_all else 0,
        "db_mean": np.mean(db_all) if db_all else 0,
        "db_max": np.max(db_all) if db_all else 0,
        "db_median": np.median(db_all) if db_all else 0,
        "n_valid": len(dF_all),
        "dF_mean_blend": np.mean(dF_blend) if dF_blend else 0,
        "dF_max_blend": np.max(dF_blend) if dF_blend else 0,
        "dt_mean_blend": np.mean(dt_blend) if dt_blend else 0,
        "dt_max_blend": np.max(dt_blend) if dt_blend else 0,
        "n_blend": len(dF_blend),
    }

# ══════════════════════════════════════════════════════
# Print summary
# ══════════════════════════════════════════════════════

print(f"\n{'='*90}")
print(f"Deviation from α = {ref_alpha} (reference)")
print(f"{'='*90}")
print(f"{'α':>3} {'n':>4} │ {'|ΔF| mean':>9} {'max':>7} │ {'|Δt_eq| mean':>12} {'max':>7} │ {'|ΔV| mean':>10} {'max':>7} │ {'|Δb| mean':>10} {'max':>7}")
print(f"{'─'*90}")
for alpha in alpha_vals:
    s = alpha_stats[alpha]
    print(f"{alpha:3d} {s['n_valid']:4d} │ "
          f"{s['dF_mean']:8.3f}% {s['dF_max']:6.2f}% │ "
          f"{s['dt_mean']:11.4f}% {s['dt_max']:6.3f}% │ "
          f"{s['dV_mean']:9.4f}% {s['dV_max']:6.3f}% │ "
          f"{s['db_mean']:9.4f}% {s['db_max']:6.3f}%")

print(f"\nBlending region only (Pe_s ∈ [0.01, 100]):")
print(f"{'α':>3} {'n':>4} │ {'|ΔF| mean':>9} {'max':>7} │ {'|Δt_eq| mean':>12} {'max':>7}")
print(f"{'─'*60}")
for alpha in alpha_vals:
    s = alpha_stats[alpha]
    print(f"{alpha:3d} {s['n_blend']:4d} │ "
          f"{s['dF_mean_blend']:8.3f}% {s['dF_max_blend']:6.2f}% │ "
          f"{s['dt_mean_blend']:11.4f}% {s['dt_max_blend']:6.3f}%")

# ══════════════════════════════════════════════════════
# Plot
# ══════════════════════════════════════════════════════

plt.rcParams.update({
    "font.size": 10, "axes.linewidth": 1.2,
    "xtick.direction": "in", "ytick.direction": "in",
    "font.family": "sans-serif",
})

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# ── Panel A: ΔF ──
ax = axes[0]
ax.plot(alpha_vals, [alpha_stats[a]["dF_mean"] for a in alpha_vals],
        'o-', color='#1f77b4', lw=2, ms=7, label='Mean |ΔF|')
#ax.plot(alpha_vals, [alpha_stats[a]["dF_max"] for a in alpha_vals],
#        's--', color='#d62728', lw=1.5, ms=6, label='Max |ΔF|')
#ax.plot(alpha_vals, [alpha_stats[a]["dF_max_blend"] for a in alpha_vals],
#        '^:', color='#ff7f0e', lw=1.5, ms=6, label='Max |ΔF| (blending only)')
ax.axvline(4, color='#2ca02c', ls='--', lw=1.5, alpha=0.6, label='selected (α=4)')
ax.axhline(0, color='k', ls='-', lw=0.5, alpha=0.3)
ax.set_xlabel('Sharpness α', fontsize=11)
ax.set_ylabel('|ΔF| relative to α=4 [%]', fontsize=11)
ax.set_title('(a) Sherwood number deviation', fontsize=11)
ax.legend(fontsize=7.5)
ax.grid(True, alpha=0.3)
ax.set_xticks(alpha_vals)

# ── Panel B: Δt_eq ──
ax = axes[1]
ax.plot(alpha_vals, [alpha_stats[a]["dt_mean"] for a in alpha_vals],
        'o-', color='#1f77b4', lw=2, ms=7, label='Mean |Δτ$_{eq}$|')
#ax.plot(alpha_vals, [alpha_stats[a]["dt_max"] for a in alpha_vals],
#        's--', color='#d62728', lw=1.5, ms=6, label='Max |Δτ$_{eq}$|')
#ax.plot(alpha_vals, [alpha_stats[a]["dt_max_blend"] for a in alpha_vals],
#        '^:', color='#ff7f0e', lw=1.5, ms=6, label='Max |Δτ$_{eq}$| (blending only)')
ax.axvline(4, color='#2ca02c', ls='--', lw=1.5, alpha=0.6, label='selected (α=4)')
ax.axhline(0, color='k', ls='-', lw=0.5, alpha=0.3)
ax.set_xlabel('Sharpness α', fontsize=11)
ax.set_ylabel('|Δτ$_{eq}$| relative to α=6 [%]', fontsize=11)
ax.set_title('(b) Equilibration time deviation', fontsize=11)
ax.legend(fontsize=7.5)
ax.grid(True, alpha=0.3)
ax.set_xticks(alpha_vals)

# ── Panel C: ΔV_req ──
ax = axes[2]
ax.plot(alpha_vals, [alpha_stats[a]["dV_mean"] for a in alpha_vals],
        'o-', color='#1f77b4', lw=2, ms=7, label='Mean |ΔV$_{req}$|')
#ax.plot(alpha_vals, [alpha_stats[a]["dV_max"] for a in alpha_vals],
#        's--', color='#d62728', lw=1.5, ms=6, label='Max |ΔV$_{req}$|')
ax.axvline(4, color='#2ca02c', ls='--', lw=1.5, alpha=0.6, label='selected (α=4)')
ax.axhline(0, color='k', ls='-', lw=0.5, alpha=0.3)
ax.set_xlabel('Sharpness α', fontsize=11)
ax.set_ylabel('|ΔV$_{req}$| relative to α=6 [%]', fontsize=11)
ax.set_title('(c) Required volume deviation', fontsize=11)
ax.legend(fontsize=7.5)
ax.grid(True, alpha=0.3)
ax.set_xticks(alpha_vals)

plt.suptitle(f'Sensitivity of model predictions to blending sharpness α\n'
             f'({n_cases} parameter combinations across diverse geometries and flow regimes)',
             fontsize=12, y=1.03)
plt.tight_layout()
plt.savefig('alpha_validation_sweep.png', dpi=200, bbox_inches='tight')
plt.savefig('alpha_validation_sweep.svg', format='svg', bbox_inches='tight')
print("\nSaved alpha_validation_sweep.png/svg")
plt.close('all')