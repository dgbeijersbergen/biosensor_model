# Squires-style validation: b_m sweep + Q sweep
# Full code with three plots

from biosensor.parameters.parameters_QCM import params
from biosensor.model.simulate_ODE import simulate
from biosensor.model.calculate_Sherwood import compute_k_m
from biosensor.model.biosensor_model import compute_delta
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

print_results = False
plot_results = False

# =====================================================
# FIXED PARAMETERS
# =====================================================
D = 1e-11
H_c = 1e-4
W_c = 1e-4
W_s = 1e-4
k_on = 1e3
k_off = 1e-3
K_D = k_off / k_on
V_in = 1e-0

# =====================================================
# PART 1: b_m SWEEP (9 combinations)
# =====================================================
# bm_combinations = [
#     {"label": "λ=0.1, c=K_D, Pe_H=1",   "lambda": 0.1,  "c_ratio": 1,    "Pe_H": 1,  "marker": "o",  "color": "r"},
#     {"label": "λ=1, c=K_D, Pe_H=1",     "lambda": 1,    "c_ratio": 1,    "Pe_H": 1,  "marker": "s",  "color": "r"},
#     {"label": "λ=10, c=K_D, Pe_H=1",    "lambda": 10,   "c_ratio": 1,    "Pe_H": 1,  "marker": "^",  "color": "r"},
#     {"label": "λ=0.1, c=K_D, Pe_H=10",  "lambda": 0.1,  "c_ratio": 1,    "Pe_H": 10, "marker": "o",  "color": "b"},
#     {"label": "λ=1, c=K_D, Pe_H=10",    "lambda": 1,    "c_ratio": 1,    "Pe_H": 10, "marker": "s",  "color": "b"},
#     {"label": "λ=10, c=K_D, Pe_H=10",   "lambda": 10,   "c_ratio": 1,    "Pe_H": 10, "marker": "^",  "color": "b"},
#     {"label": "λ=1, c=0.01K_D, Pe_H=1", "lambda": 1,    "c_ratio": 0.01, "Pe_H": 1,  "marker": "d",  "color": "g"},
#     {"label": "λ=1, c=100K_D, Pe_H=1",  "lambda": 1,    "c_ratio": 100,  "Pe_H": 1,  "marker": "*",  "color": "g"},
#     {"label": "λ=1, c=0.01K_D, Pe_H=10","lambda": 1,    "c_ratio": 0.01, "Pe_H": 10, "marker": "d",  "color": "m"},
# ]
bm_combinations = [
    # --- Pe_H well inside CD for each lambda ---
    # λ=0.1: Pe_H_c = 0.178, use Pe_H = 0.1
    {"label": "λ=0.1, c=K_D, Pe_H=0.1", "lambda": 0.1, "c_ratio": 1, "Pe_H": 0.1, "marker": "o", "color": "k", "fillstyle": 'none'},

    # λ=1: Pe_H_c = 1.78, use Pe_H = 1
    {"label": "λ=1, c=K_D, Pe_H=1", "lambda": 1, "c_ratio": 1, "Pe_H": 1, "marker": "*", "color": "k", "fillstyle": 'none'},

    # λ=10: Pe_H_c = 17.8, use Pe_H = 1 and 10
    #{"label": "λ=10, c=K_D, Pe_H=10", "lambda": 10, "c_ratio": 1, "Pe_H": 10, "marker": "^", "color": "r"},  # problem

    # --- Concentration variants (all at safe Pe_H) ---
    {"label": "λ=1, c=0.01K_D, Pe_H=1", "lambda": 1, "c_ratio": 0.01, "Pe_H": 1, "marker": "*", "color": "r", "fillstyle": 'full'},
    {"label": "λ=1, c=0.1K_D, Pe_H=1", "lambda": 1, "c_ratio": 0.1, "Pe_H": 1, "marker": "*", "color": "g", "fillstyle": 'full'},

    # --- Extra: higher lambda for better coverage ---
    #{"label": "λ=100, c=K_D, Pe_H=10", "lambda": 100, "c_ratio": 1, "Pe_H": 10, "marker": "p", "color": "m"}, # problem
    {"label": "λ=10, c=K_D, Pe_H=10", "lambda": 10, "c_ratio": 1, "Pe_H": 10, "marker": "^", "color": "k", "fillstyle": 'none'},
    {"label": "λ=100, c=K_D, Pe_H=100", "lambda": 100, "c_ratio": 1, "Pe_H": 100, "marker": "D", "color": "k", "fillstyle": 'none'},
]

marker_size_map = {
    'o': 8,
    's': 8,
    '^': 6,
    'd': 3,   # smaller (diamonds look big)
    'p': 7,
    '*': 10   # stars need to be bigger to look comparable
}

# Verify all are in CD
print("b_m sweep CD verification:")
print(f"{'Label':<35} {'Pe_H':>6} {'Pe_H_c':>8} {'margin':>8}")
print("-" * 65)
for combo in bm_combinations:
    pe_c = 1.78 * combo["lambda"]
    margin = pe_c / combo["Pe_H"]
    status = "OK" if combo["Pe_H"] < pe_c else "FAIL"
    print(f"{combo['label']:<35} {combo['Pe_H']:6.1f} {pe_c:8.2f} {margin:7.1f}x  {status}")

b_m_vals = np.logspace(-13, -7, 13)

# =====================================================
# PART 2: Q SWEEP (3 systems)
# =====================================================
q_systems = [
    {"lambda": 10, "b_m_fixed": 1e-7,  "color": '#E08060', "marker": "x"},   # Da_c ~ 10
    {"lambda": 10, "b_m_fixed": 1e-8,  "color": '#9050A0', "marker": "+"},   # Da_c ~ 1
    {"lambda": 10, "b_m_fixed": 1e-6, "color": '#4090A0', "marker": "1"},   # Da_c ~ 100
]

#Pe_H_vals_sweep = np.logspace(0, 4, 5)
Pe_H_vals_sweep = np.array([0.00003, 0.000179,  0.0062, 0.1790, 6.62]) * 1e5

q_sweep_combinations = []
for cfg in q_systems:
    for Pe_H_val in Pe_H_vals_sweep:
        q_sweep_combinations.append({
            "label": f"Q-sweep λ={cfg['lambda']}",
            "lambda": cfg["lambda"],
            "c_ratio": 1,
            "Pe_H": Pe_H_val,
            "k_on": 18,
            "marker": cfg["marker"],
            "color": cfg["color"],
            "b_m_fixed": cfg["b_m_fixed"],
        })

# =====================================================
# RUN b_m SWEEP
# =====================================================
results = []

for combo in tqdm(bm_combinations, desc="b_m sweep"):
    lam = combo["lambda"]
    L_s = lam * H_c
    L_c = L_s
    c_in = combo["c_ratio"] * K_D
    Q_in = combo["Pe_H"] * D * W_c
    t_R = 1.0 / (k_on * c_in + k_off)

    for b_m in b_m_vals:
        params.D = D
        params.H_c = H_c
        params.W_c = W_c
        params.W_s = W_s
        params.k_on = k_on
        params.k_off = k_off
        params.L_s = L_s
        params.L_c = L_c
        params.c_in = c_in
        params.c_0 = 0.0
        params.Q_in = Q_in
        params.V_in = V_in
        params.b_m = b_m

        k_m, F = compute_k_m(Q_in, params)
        Da = k_on * b_m * L_s / (k_m * H_c)
        max_time = max(100 * (1 + Da) * t_R, 100)
        print(max_time)

        try:
            result = simulate(params, print_results, plot_results, max_time)
            results.append({
                "label": combo["label"],
                "marker": combo["marker"],
                "color": combo["color"],
                "sweep_type": "b_m",
                "lambda": lam,
                "c_ratio": combo["c_ratio"],
                "Pe_H": combo["Pe_H"],
                "b_m": b_m,
                "Da_computed": Da,
                "t_R": t_R,
                "F": F,
                **result
            })
        except Exception as e:
            print(f"Failed: {combo['label']}, b_m={b_m:.1e}: {e}")

# =====================================================
# RUN Q SWEEP
# =====================================================
for combo in tqdm(q_sweep_combinations, desc="Q sweep"):
    k_on = 18
    lam = combo["lambda"]
    L_s = lam * H_c
    L_c = L_s
    c_in = combo["c_ratio"] * K_D
    Q_in = combo["Pe_H"] * D * W_c
    t_R = 1.0 / (k_on * c_in + k_off)
    b_m = combo["b_m_fixed"]

    params.D = D
    params.H_c = H_c
    params.W_c = W_c
    params.W_s = W_s
    params.k_on = k_on
    params.k_off = k_off
    params.L_s = L_s
    params.L_c = L_c
    params.c_in = c_in
    params.c_0 = 0.0
    params.Q_in = Q_in
    params.V_in = V_in
    params.b_m = b_m

    k_m, F = compute_k_m(Q_in, params)
    Da = k_on * b_m * L_s / (k_m * H_c)
    Pe_H = combo["Pe_H"]

    print(f"Pe_H={Pe_H:.1f}, Da={Da:.2e}, F={F:.2f}, F/Pe_H={F/Pe_H:.3f}")

    max_time = max(10 * (1 + Da) * t_R, 100)
    print(max_time)

    try:
        result = simulate(params, print_results, plot_results, max_time)
        results.append({
            "label": combo["label"],
            "marker": combo["marker"],
            "color": combo["color"],
            "sweep_type": "Q",
            "lambda": lam,
            "c_ratio": combo["c_ratio"],
            "Pe_H": combo["Pe_H"],
            "b_m": b_m,
            "Da_computed": Da,
            "t_R": t_R,
            "F": F,
            **result
        })
    except Exception as e:
        print(f"Failed: {combo['label']}, Pe_H={combo['Pe_H']:.2f}: {e}")

df = pd.DataFrame(results)


# =====================================================
# COMPUTE NORMALIZED QUANTITIES
# =====================================================
eq_factor = -np.log(0.05)

df["Da"] = df["Da_computed"]
df["t_eq_norm"] = df["time_eq"] / (eq_factor * df["t_R"])
df["V_req"] = df["time_eq"] * df["Q_in"]
df["V_min"] = df.apply(
    lambda row: row["k_on"] * row["b_m"] * row["L_s"] * row["W_c"] * row["t_R"] * eq_factor,
    axis=1
)
df["V_req_over_Vmin"] = df["V_req"] / df["V_min"]

# =====================================================
# FILTER: Q-sweep only outside CD
# =====================================================
Pe_H_c_series = 1.78 * df["lambda"]
df["in_CD_geo"] = df["Pe_H"] <= Pe_H_c_series

# Masks for plotting
bm_mask = df["sweep_type"] == "b_m"
q_mask  = (df["sweep_type"] == "Q") & (~df["in_CD_geo"])   # <-- only outside CD

Da_theory = np.logspace(-3, 3, 500)

# =====================================================
# COMSOL VALIDATION DATA
# =====================================================
# Add after df computation, before plotting

# Row: [Da, Pe_H, t_eq_comsol (s), t_R (s), Q (m3/s), k_on, b_m, L_s, W_c]
comsol_data = [
    # Row 1: Da=0.01, Pe_H=18 (Q_c, CD)
    {"Da": 0.01, "Pe_H": 18, "t_eq": 3350, "t_R": 1000,
     "Q": 1.8e-14, "k_on": 0.018, "b_m": 1e-7, "L_s": 1e-3, "W_c": 1e-4,
     "label": "COMSOL (CD)", "color": "k", "marker": "P"},  # filled plus

    # Row 2: Da=1, Pe_H=18 (Q_c, CD)
    {"Da": 1, "Pe_H": 18, "t_eq": 6700.1, "t_R": 999,
     "Q": 1.8e-14, "k_on": 1.8, "b_m": 1e-7, "L_s": 1e-3, "W_c": 1e-4,
     "label": "COMSOL (CD)", "color": "k", "marker": "P"},

    # Row 3: Da=10, Pe_H=18 (Q_c, CD)
    {"Da": 10, "Pe_H": 18, "t_eq": 36662, "t_R": 990,
     "Q": 1.8e-14, "k_on": 18, "b_m": 1e-7, "L_s": 1e-3, "W_c": 1e-4,
     "label": "COMSOL (CD)", "color": "k", "marker": "P"},

    ## Row 4: Da=0.01, Pe_H=100 (outside CD)
    #{"Da": 0.01, "Pe_H": 100, "t_eq": 3100, "t_R": 1000,
    # "Q": 1e-13, "k_on": 0.032, "b_m": 1e-7, "L_s": 1e-3, "W_c": 1e-4,
    # "label": "COMSOL (outside CD)", "color": "k", "marker": "X"},

    # Row 5: Da=1, Pe_H=100 (outside CD)
    {"Da": 1, "Pe_H": 18300, "t_eq": 5340.7, "t_R": 982,
     "Q": 1.83e-11, "k_on": 18, "b_m": 1e-7, "L_s": 1e-3, "W_c": 1e-4,
     "label": "COMSOL (outside CD)", "color": "k", "marker": "X"},

    # Row 6: Da=10, Pe_H=100 (Da_c = 100)
    {"Da": 10, "Pe_H": 17900.00, "t_eq": 20115, "t_R": 848.17,
     "Q": 1.79e-11, "k_on": 179, "b_m": 1e-7, "L_s": 1e-3, "W_c": 1e-4,
     "label": "COMSOL (outside CD)", "color": "k", "marker": "X"},

    # Row 7: Da=0.1, Pe_H=100 (Da_c = 1)
    {"Da": 0.1, "Pe_H": 18.3, "t_eq": 3678.9, "t_R": 999.8,
     "Q": 1.83e-14, "k_on": 0.18, "b_m": 1e-7, "L_s": 1e-3, "W_c": 1e-4,
     "label": "COMSOL (CD)", "color": "k", "marker": "X"},

    # Row 8: Da=0.1, Pe_H=100 (Da_c = 1)
    {"Da": 0.1, "Pe_H": 17900.00, "t_eq": 3271.3, "t_R": 998.2,
     "Q": 1.79e-11, "k_on": 1.79, "b_m": 1e-7, "L_s": 1e-3, "W_c": 1e-4,
     "label": "COMSOL (outside CD)", "color": "k", "marker": "X"},

    # Row 11: Da=100, Pe_H=18 (Da_c = 100)
    {"Da": 99.1, "Pe_H": 18.33, "t_eq": 257604.9, "t_R": 847.45,
     "Q": 1.83e-14, "k_on": 1.8e2, "b_m": 1e-7, "L_s": 1e-3, "W_c": 1e-4,
     "label": "COMSOL (CD)", "color": "k", "marker": "X"},
]

# Compute normalized quantities for COMSOL
for pt in comsol_data:
    pt["t_eq_norm"] = pt["t_eq"] / (eq_factor * pt["t_R"])
    pt["V_req"] = pt["t_eq"] * pt["Q"]
    pt["V_min"] = pt["k_on"] * pt["b_m"] * pt["L_s"] * pt["W_c"] * pt["t_R"] * eq_factor
    pt["V_req_over_Vmin"] = pt["V_req"] / pt["V_min"]

comsol_df = pd.DataFrame(comsol_data)



# =====================================================
# SUPPLEMENTARY PLOT 1: t_eq collapse (detailed)
# =====================================================
fig, ax = plt.subplots(figsize=(8, 6))

ax.loglog(Da_theory, 1 + Da_theory, 'k--', lw=2, label='$(1 + Da)$', zorder=5)
ax.loglog(Da_theory, np.ones_like(Da_theory), 'k:', lw=1, alpha=0.4)
ax.loglog(Da_theory, Da_theory, 'k:', lw=1, alpha=0.4)

ms_supp = marker_size_map.get(combo["marker"], 8)

for combo in bm_combinations:
    mask = (df["label"] == combo["label"]) & (df["sweep_type"] == "b_m")
    subset = df[mask]
    if len(subset) == 0:
        continue
    ax.loglog(subset["Da"], subset["t_eq_norm"],
              marker=combo["marker"], color=combo["color"],
              markersize=ms_supp, linestyle='none', label=combo["label"],
              markeredgecolor='k', markeredgewidth=1.5, fillstyle=combo["fillstyle"])

q_labels_plotted = set()
for cfg in q_systems:
    lbl = f"Q-sweep λ={cfg['lambda']}"
    mask = (df["label"] == lbl) & q_mask
    subset = df[mask]
    if len(subset) == 0:
        continue
    label = lbl if lbl not in q_labels_plotted else None
    q_labels_plotted.add(lbl)
    ax.loglog(subset["Da"], subset["t_eq_norm"],
              marker='D', color=cfg["color"],
              markersize=7, linestyle='none', label=label,
              markeredgecolor='k', markeredgewidth=1.5)

ax.set_xlabel('Da', fontsize=13)
ax.set_ylabel('$\\t_{eq} \\;/\\; 3\\t_R$', fontsize=13)
#ax.set_title('Supplementary: Equilibration time collapse', fontsize=13)
ax.legend(fontsize=7, ncol=2, loc='upper left', framealpha=0.9)
ax.grid(True, which='both', alpha=0.3)
ax.set_xlim(1e-3, 1e3)
ax.set_ylim(0.5, 1e4)
ax.text(300, 400, 'Slope = 1', fontsize=9, color='0.5', rotation=42)
plt.tight_layout()
plt.savefig('t_eq_supplement.svg', dpi=300)
plt.close()
#plt.show()

# =====================================================
# SUPPLEMENTARY PLOT 2: V_req collapse (detailed)
# =====================================================
fig, ax = plt.subplots(figsize=(8, 6))

ax.loglog(Da_theory, 1 + 1./Da_theory, 'k--', lw=2, label='$1 + 1/Da$ (CD)', zorder=5)

# Guide lines for Da_c
Da_c_guides = [
    {"Da_c": 100, "color": '#E08060', "ls": "--"},
    {"Da_c": 10,  "color": '#9050A0', "ls": "-."},
    {"Da_c": 1,   "color": '#4090A0', "ls": ":"},
]
for guide in Da_c_guides:
    Da_c = guide["Da_c"]
    Da_branch = Da_theory[Da_theory < Da_c]
    V_branch = (1 + Da_branch) * Da_c**2 / Da_branch**3
    ax.loglog(Da_branch, V_branch, guide["ls"], color=guide["color"],
              lw=1.5, alpha=0.5, label=f'Outside CD ($Da_c = {Da_c}$)')
    #ax.loglog(Da_c, 1 + 1/Da_c, 'o', color=guide["color"], markersize=10,
    #          markeredgecolor='k', markeredgewidth=1.5, zorder=6)

for combo in bm_combinations:
    mask = (df["label"] == combo["label"]) & (df["sweep_type"] == "b_m")
    subset = df[mask]
    if len(subset) == 0:
        continue
    ax.loglog(subset["Da"], subset["V_req_over_Vmin"],
              marker=combo["marker"], color=combo["color"],
              markersize=ms_supp, linestyle='none', label=combo["label"],
              markeredgecolor='k', markeredgewidth=1.5, fillstyle=combo["fillstyle"])

q_labels_plotted = set()
for cfg in q_systems:
    lbl = f"Q-sweep λ={cfg['lambda']}"
    mask = (df["label"] == lbl) & q_mask
    subset = df[mask]
    if len(subset) == 0:
        continue
    label = lbl if lbl not in q_labels_plotted else None
    q_labels_plotted.add(lbl)
    ax.loglog(subset["Da"], subset["V_req_over_Vmin"],
              marker=cfg["marker"], color=cfg["color"],
              markersize=9, linestyle='none', label=label,
              markeredgecolor='k', markeredgewidth=1.5)

ax.set_xlabel('Da', fontsize=13)
ax.set_ylabel('$V_{req} \\;/\\;3 V_{min}$', fontsize=13)
#ax.set_title('Supplementary: Required volume scaling', fontsize=13)
ax.legend(fontsize=7, ncol=2, loc='upper right', framealpha=0.9)
ax.grid(True, which='both', alpha=0.3)
ax.set_xlim(1e-3, 1e3)
ax.set_ylim(0.5, 1e6)
plt.tight_layout()
plt.savefig('V_req_supplement.svg', dpi=300)
plt.close()
#plt.show()

# =====================================================
# COMBINED FIGURE: Main panels (a, b) + Q/Q_c panels (c, d, e)
# =====================================================

import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 13,
    'axes.linewidth': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
})

# ─────────────────────────────────────────────────────
# SHARED COLOR PALETTE
# ─────────────────────────────────────────────────────
col_ana_a    = '#2166AC'   # blue  – analytical line, t_eq (panel a / d)
col_ana_b    = '#236a43ff' # dark green – analytical line, V_req (panel b / e)
col_sim_a    = '#5EA1D6'   # light blue  – simulated markers in t_eq panels
col_sim_b    = '#3cb371ff' # saturated green – simulated markers in V_req panels
col_comsol   = '#FFCC00'   # gold – COMSOL everywhere
comsol_edge  = '#333333'   # dark gray edge for COMSOL stars (pops without harshness)
comsol_ms    = 22          # slightly larger for visibility
comsol_ms_out = 12         # comsol outside CD
comsol_mk    = '*'         # COMSOL mar ker shape
col_cd       = '#2166AC'   # blue  – CD branch lines in Q/Qc panels
col_out      = '#666666'   # red   – outside-CD branch lines in Q/Qc panels
col_hyp      = '0.65'      # gray  – hypothetical extension
col_Dac10    = '#800000'   # maroon – highlighted Da_c = 10 guide line
col_Dac_rest = 'k'         # black – Da_c = 100 and Da_c = 1 guides

def _shade_regimes(ax):
    ax.axvspan(3e-2, 1,   alpha=0.06, color='0.5', zorder=0)
    ax.axvspan(1,    3e3, alpha=0.03, color='0.5', zorder=0)
    ax.axvline(x=1, color='k', ls='-', lw=1, alpha=0.4)

# ─────────────────────────────────────────────────────
# LAYOUT: left 2-panel column + right 3-panel column
# ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 13.5))

gs_left = gridspec.GridSpec(
    2, 1, figure=fig,
    left=0.09, right=0.650,
    top=0.95,   bottom=0.07,
    hspace=0.18,   # tighter: shared x-axis, tick labels suppressed on (a)
)
gs_right = gridspec.GridSpec(
    3, 1, figure=fig,
    left=0.750, right=0.980,
    top=0.95,   bottom=0.07,
    hspace=0.52,   # slightly more breathing room between right panels
)

ax1 = fig.add_subplot(gs_left[0])   # (a) τ_CRD / τ_R  vs  Da
ax2 = fig.add_subplot(gs_left[1])   # (b) V_req / V_min  vs  Da
ax3 = fig.add_subplot(gs_right[0])  # (c) Da  vs  Q/Q_c
ax4 = fig.add_subplot(gs_right[1])  # (d) τ_CRD / τ_R  vs  Q/Q_c
ax5 = fig.add_subplot(gs_right[2])  # (e) V_req / V_min  vs  Q/Q_c


# ═══════════════════════════════════════════════════
# PANEL (a)  –  Equilibration time vs Da
# ═══════════════════════════════════════════════════
ax1.loglog(Da_theory, 1 + Da_theory, color=col_ana_a, lw=4, zorder=3)
#ax1.loglog(Da_theory, np.ones_like(Da_theory), 'k-', lw=1, alpha=0.15)
#ax1.loglog(Da_theory, Da_theory, 'k-', lw=1, alpha=0.15)

# Simulated – b_m sweep
ax1.loglog(df[bm_mask]["Da"], df[bm_mask]["t_eq_norm"],
           'o', color=col_sim_a, markersize=8,
           markeredgecolor='k', markeredgewidth=1.5, alpha=1, zorder=5)
# Simulated – Q sweep
ax1.loglog(df[q_mask]["Da"], df[q_mask]["t_eq_norm"],
           's', color=col_sim_a, markersize=8,
           markeredgecolor='k', markeredgewidth=1.5, alpha=1, zorder=5)
# COMSOL
comsol_mk_cd  = '*'   # inside CD
comsol_mk_out = 'D'   # outside CD

for _, pt in comsol_df.iterrows():
    mk = comsol_mk_cd if pt["label"] == "COMSOL (CD)" else comsol_mk_out
    ms = comsol_ms if mk == '*' else comsol_ms_out
    ax1.loglog(pt["Da"], pt["t_eq_norm"],
               marker=mk, color=col_comsol, markersize=ms,
               markeredgecolor=comsol_edge, markeredgewidth=2.0, zorder=7, alpha=1)

# Regime shading
ax1.axvspan(1e-3, 1,   alpha=0.06, color='0.5')   # gray, reaction-limited
ax1.axvspan(1,    1e3, alpha=0.03, color='0.5')    # lighter gray, transport-limited
ax1.axvline(x=1,  color='k', ls='-', lw=1, alpha=0.4)

ax1.text(30, 5, r'$t_{eq} / (3 t_R) = 1 + Da$',
         fontsize=14, color=col_ana_a, ha='left', va='bottom',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.9))

# Suppress x tick labels on (a) — shared axis with (b) below
#ax1.tick_params(labelbottom=False)
ax1.set_xlabel('Damköhler number, $Da$', fontsize=14)

ax1.set_ylabel(r'$t_{eq} / 3 t_R$', fontsize=14)
ax1.set_xlim(1e-3, 1e3)
ax1.set_ylim(0.5, 1e4)
ax1.grid(True, which='major', alpha=0.12)

# Panel label: outside top-left, consistent position for all panels
ax1.text(-0.12, 1.02, '(a)', transform=ax1.transAxes,
         fontsize=15, fontweight='bold', va='bottom')


analytical        = Line2D([0],[0], color='k', lw=2)
complete_delivery = Line2D([0],[0], color='k', lw=2.1, ls='-.')
inside_cd         = Line2D([], [], marker='o', color='w', markerfacecolor='k',
                            markersize=8, markeredgecolor='k', markeredgewidth=0.6)
outside_cd        = Line2D([], [], marker='s', color='w', markerfacecolor='k',
                            markersize=8, markeredgecolor='k', markeredgewidth=0.6)
comsol_cd_entry  = Line2D([], [], marker='*', color='w',
                           markerfacecolor=col_comsol, markersize=18,
                           markeredgecolor=comsol_edge, markeredgewidth=2.0)
comsol_out_entry = Line2D([], [], marker='D', color='w',
                           markerfacecolor=col_comsol, markersize=10,
                           markeredgecolor=comsol_edge, markeredgewidth=2.0)

ax1.legend(
    handles=[analytical, complete_delivery, inside_cd, outside_cd, comsol_cd_entry, comsol_out_entry],
    labels=['Analytical', 'Analytical (outside CD)', 'Simulated (inside CD)', 'Simulated (outside CD)', 'COMSOL (inside CD)', 'COMSOL (outside CD)'],
    fontsize=11, loc='upper left', framealpha=0.95, edgecolor='0.85'
)

ax1.text(0.15, 2e3, 'Reaction\nlimited',   fontsize=10, color='0.45', ha='center', style='italic')
ax1.text(50,   2e3, 'Transport\nlimited',  fontsize=10, color='0.45', ha='center', style='italic')



# ═══════════════════════════════════════════════════
# PANEL (b)  –  Required volume vs Da
# ═══════════════════════════════════════════════════
ax2.loglog(Da_theory, 1 + 1. / Da_theory, color=col_ana_b, lw=4, zorder=3)

# Outside-CD guide lines
_guides = [
    {"Da_c": 100, "color": '#9e9e9e', "ls": "-.", "lw": 2.5, "alpha": 1},
    {"Da_c":  10, "color": '#9e9e9e', "ls": "-.", "lw": 2.5, "alpha": 1},   # slightly bolder
    {"Da_c":   1, "color": '#9e9e9e', "ls": "-.", "lw": 2.5, "alpha": 1},
]

for g in _guides:
    Da_c_val = g["Da_c"]
    Da_br    = Da_theory[Da_theory < Da_c_val]
    V_br     = (1 + Da_br) * Da_c_val**2 / Da_br**3
    ax2.loglog(Da_br, V_br, g["ls"], color=g["color"],
               lw=g["lw"], alpha=g["alpha"], zorder=2)


# Simulated – b_m sweep
ax2.loglog(df[bm_mask]["Da"], df[bm_mask]["V_req_over_Vmin"],
           'o', color=col_sim_b, markersize=8,
           markeredgecolor='k', markeredgewidth=1.5, alpha=0.85, zorder=5)
# Simulated – Q sweep
ax2.loglog(df[q_mask]["Da"], df[q_mask]["V_req_over_Vmin"],
           's', color=col_sim_b, markersize=8,
           markeredgecolor='k', markeredgewidth=1.5, alpha=0.85, zorder=5)
# COMSOL

for _, pt in comsol_df.iterrows():
    mk = comsol_mk_cd if pt["label"] == "COMSOL (CD)" else comsol_mk_out
    ms = comsol_ms if mk == '*' else comsol_ms_out
    mk_zorder = 7 if mk == '*' else 4
    ax2.loglog(pt["Da"], pt["V_req_over_Vmin"],
               marker=mk, color=col_comsol, markersize=ms,
               markeredgecolor=comsol_edge, markeredgewidth=2.0, zorder=mk_zorder, alpha=1)

# Regime shading — matches panel (a) exactly
ax2.axvspan(1e-3, 1,   alpha=0.06, color='0.5')   # gray, reaction-limited
ax2.axvspan(1,    1e3, alpha=0.03, color='0.5')    # lighter gray, transport-limited
ax2.axvline(x=1,  color='k', ls='-', lw=1, alpha=0.4)

ax2.text(3e-3, 3, '$V_{req}/3 V_{min} = 1 + 1/Da$',
         fontsize=14, color=col_ana_b, ha='left', va='bottom',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=1))

ax2.text(6, 1.5e3, '$V_{req}/3 V_{min} = (1+1/Da)(\\dfrac{Da_c}{Da})^2$',
         fontsize=14, color=col_ana_b, ha='left', va='bottom',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=1))

ax2.set_xlabel('Damköhler number, $Da$', fontsize=14)
ax2.set_ylabel('$V_{req}\\; / \\;3V_{min}$', fontsize=14)
ax2.set_xlim(1e-3, 1e3)
ax2.set_ylim(0.5, 1e4)
ax2.grid(True, which='major', alpha=0.12)

# Panel label: outside top-left — same position as (a)
ax2.text(-0.12, 1.02, '(b)', transform=ax2.transAxes,
         fontsize=15, fontweight='bold', va='bottom')

#ax2.text(0.5, 3000, 'c, d, e', rotation=-62, rotation_mode='anchor', fontsize=12, fontweight='bold', va='bottom')
ax2.text(0.32, 1e2, '$Da_c=1$',  color='#9e9e9e', rotation=-57, rotation_mode='anchor', fontsize=12, fontweight='bold', va='bottom')
ax2.text(2.2, 1e2, '$Da_c=10$ (c–e)',  color='#9e9e9e', rotation=-57, rotation_mode='anchor',fontsize=12, fontweight='bold', va='bottom')
ax2.text(18.5, 1e2, '$Da_c=100$', color='#9e9e9e', rotation=-57, rotation_mode='anchor', fontsize=12, fontweight='bold', va='bottom')


# ─────────────────────────────────────────────────────
# COMPUTE Q/Q_c DATA
# ─────────────────────────────────────────────────────
k_on_sys = 18
b_m_sys  = 1e-7
lam_sys  = 10
c_in_sys = 1.0 * K_D
t_R_sys  = 1.0 / (k_on_sys * c_in_sys + k_off)

Q_c_ana = 1.78 * lam_sys * D * W_c
Pe_H_c  = Q_c_ana / (D * W_c)
Da_c_ana = k_on_sys * b_m_sys * H_c / (1.78 * D)

print(f"Da_c (analytical) = {Da_c_ana:.2f}")
print(f"Q_c  = {Q_c_ana:.2e} m3/s  (Pe_H,c = {Pe_H_c:.1f})")
print(f"t_R  = {t_R_sys:.1f} s")

q_mask_sys = (df["sweep_type"] == "Q") & (np.isclose(df["b_m"], b_m_sys))
df_q = df[q_mask_sys].copy()
df_q["Q"] = df_q["Pe_H"] * D * W_c
df_q["q"] = df_q["Q"] / Q_c_ana

# Analytical curves
q_cd  = np.logspace(-1.5, 0,   300)
q_out = np.logspace(0,    3.5, 500)

Da_cd  = Da_c_ana / q_cd
Da_out = Da_c_ana / q_out ** (1 / 3)
Da_hyp = Da_c_ana / q_out

teq_cd  = 1 + Da_cd
teq_out = 1 + Da_out

V_cd  = 1 + q_cd / Da_c_ana
V_out = q_out ** (2 / 3) + q_out / Da_c_ana
V_hyp = 1 + q_out / Da_c_ana

# COMSOL → Q/Q_c
comsol_q_pts = []
for _, pt in comsol_df.iterrows():
    comsol_q_pts.append({**pt, "q": pt["Q"] / Q_c_ana})
comsol_q_df = pd.DataFrame(comsol_q_pts)
comsol_k18  = comsol_q_df[np.isclose(comsol_q_df["k_on"].astype(float), 18, rtol=0.1)]

df_q_in  = df_q[df_q["q"] <= 1]
df_q_out = df_q[df_q["q"] >  1]

# ═══════════════════════════════════════════════════
# PANEL (c)  –  Da vs Q/Q_c
# ═══════════════════════════════════════════════════
ax3.loglog(q_cd,  Da_cd,  '-',  color='k', lw=2.0)
ax3.loglog(q_out, Da_out, '-',  color='k', lw=2.0)
ax3.loglog(q_out, Da_hyp, '--', color='k', lw=2.0, alpha=1)

# Simulated (Q sweep)
#ax3.loglog(df_q["q"], df_q["Da"], 'o', color='white', markersize=9,
#           markeredgecolor='k', markeredgewidth=1.5, zorder=6, alpha=1)

ax3.loglog(df_q_in["q"],  df_q_in["Da"],  'o', color='white', markersize=7,
           markeredgecolor='k', markeredgewidth=1.5, zorder=6, alpha=1)
ax3.loglog(df_q_out["q"], df_q_out["Da"], 's', color='white', markersize=7,
           markeredgecolor='k', markeredgewidth=1.5, zorder=6, alpha=1)

# COMSOL

for _, pt in comsol_k18.iterrows():
    mk = comsol_mk_cd if pt["label"] == "COMSOL (CD)" else comsol_mk_out
    ms = comsol_ms if mk == '*' else comsol_ms_out
    ax3.loglog(pt["q"], pt["Da"],
               marker=mk, color=col_comsol, markersize=ms,
               markeredgecolor=comsol_edge, markeredgewidth=2.0, zorder=4, alpha=1)

ax3.set_xlabel('$Q \\;/\\; Q_c$', fontsize=14)
ax3.set_ylabel('$Da$', fontsize=14)
ax3.set_xlim(3e-2, 3e3)
ax3.set_ylim(1e-2, 9e2)
ax3.set_box_aspect(1)
ax3.grid(True, which='major', alpha=0.15)

ax3.text(0.05, 250, '$ \\sim Q^{-1}$', fontsize=12)

ax3.text(80, 4.2, '$ \\sim Q^{-1/3}$', fontsize=12)

# Panel label: outside top-left + title below it
ax3.text(-0.18, 1.02, '(c)', transform=ax3.transAxes,
         fontsize=15, fontweight='bold', va='bottom')
ax3.set_title('Damköhler number', fontsize=13, pad=6)

_shade_regimes(ax3)

ax3.legend(handles=[
    Line2D([0],[0], color='k', ls='-',  lw=1.5, label='Analytical'),
    Line2D([0],[0], color='k', ls='--', lw=1.5, label='Complete delivery'),
], fontsize=9, loc='lower left', framealpha=0.9, edgecolor='0.85')


# ═══════════════════════════════════════════════════
# PANEL (d)  –  τ_CRD / τ_R  vs  Q/Q_c
# ═══════════════════════════════════════════════════
ax4.loglog(q_cd,  teq_cd,       '-',  color='k', lw=2.0)
ax4.loglog(q_out, teq_out,      '-',  color='k', lw=2.0)
ax4.loglog(q_out, 1 + Da_hyp,   '--', color='k', lw=2.0, alpha=1)

#ax4.loglog(df_q["q"], df_q["t_eq_norm"], 'o', color=col_sim_a, markersize=9,
#           markeredgecolor='k', markeredgewidth=1.5, zorder=6, alpha=1)

ax4.loglog(df_q_in["q"],  df_q_in["t_eq_norm"],  'o', color=col_sim_a, markersize=7,
           markeredgecolor='k', markeredgewidth=1.5, zorder=6, alpha=1)
ax4.loglog(df_q_out["q"], df_q_out["t_eq_norm"], 's', color=col_sim_a, markersize=7,
           markeredgecolor='k', markeredgewidth=1.5, zorder=6, alpha=1)


for _, pt in comsol_k18.iterrows():
    mk = comsol_mk_cd if pt["label"] == "COMSOL (CD)" else comsol_mk_out
    ms = comsol_ms if mk == '*' else comsol_ms_out
    ax4.loglog(pt["q"], pt["t_eq_norm"],
               marker=mk, color=col_comsol, markersize=ms,
               markeredgecolor=comsol_edge, markeredgewidth=2.0, zorder=4, alpha=1)


#ax4.annotate('$\\sim Q^{-1}$',
#             xy=(0.8, Da_c_ana / 0.15), fontsize=11, color='k', ha='left',
#             xytext=(0.055, Da_c_ana / 0.15 * 2.5),
#             arrowprops=dict(arrowstyle='->', color='k', lw=1.0))

# ax4.text(0.1, 200, '$ \\sim Q^{-1}$', fontsize=12)

# ax4.text(170, 4.6, '$ \\sim Q^{-1/3}$', fontsize=12)

#ax4.annotate('$\\sim Q^{-1/3}$',
#             xy=(20, Da_c_ana / 20 ** (1/3)), fontsize=11, color='k', ha='left',
#             xytext=(80, Da_c_ana / 20 ** (1/3) * 4),
#             arrowprops=dict(arrowstyle='->', color='k', lw=1.0))

ax4.set_xlabel('$Q \\;/\\; Q_c$', fontsize=14)
ax4.set_ylabel(r'$t_{eq} / 3 t_R$', fontsize=14)
ax4.set_xlim(3e-2, 3e3)
ax4.set_ylim(3e-1, 3e3)
ax4.set_box_aspect(1)
ax4.grid(True, which='major', alpha=0.15)

ax4.text(-0.18, 1.02, '(d)', transform=ax4.transAxes,
         fontsize=15, fontweight='bold', va='bottom')
ax4.set_title('Equilibration time', fontsize=13, pad=6)

_shade_regimes(ax4)

# ═══════════════════════════════════════════════════
# PANEL (e)  –  V_req / V_min  vs  Q/Q_c
# ═══════════════════════════════════════════════════
ax5.loglog(q_cd,  V_cd,  '-',  color='k', lw=2.0)
ax5.loglog(q_out, V_out, '-',  color='k', lw=2.0)
ax5.loglog(q_out, V_hyp, '--', color='k', lw=2.0, alpha=1)

#ax5.loglog(df_q["q"], df_q["V_req_over_Vmin"], 'o', color=col_sim_b, markersize=9,
#           markeredgecolor='k', markeredgewidth=1.5, zorder=6, alpha=1)

ax5.loglog(df_q_in["q"],  df_q_in["V_req_over_Vmin"],  'o', color=col_sim_b, markersize=7,
           markeredgecolor='k', markeredgewidth=1.5, zorder=6, alpha=1)
ax5.loglog(df_q_out["q"], df_q_out["V_req_over_Vmin"], 's', color=col_sim_b, markersize=7,
           markeredgecolor='k', markeredgewidth=1.5, zorder=6, alpha=1)


for _, pt in comsol_k18.iterrows():
    mk = comsol_mk_cd if pt["label"] == "COMSOL (CD)" else comsol_mk_out
    ms = comsol_ms if mk == '*' else comsol_ms_out
    ax5.loglog(pt["q"], pt["V_req_over_Vmin"],
               marker=mk, color=col_comsol, markersize=ms,
               markeredgecolor=comsol_edge, markeredgewidth=2.0, zorder=4, alpha=1)


ax5.set_xlabel('$Q \\;/\\; Q_c$', fontsize=14)
ax5.set_ylabel('$V_{req} \\;/\\; 3V_{min}$', fontsize=14)
ax5.set_xlim(3e-2, 3e3)
ax5.set_ylim(3e-1, 3e3)
ax5.set_box_aspect(1)
ax5.grid(True, which='major', alpha=0.15)

ax5.text(-0.18, 1.02, '(e)', transform=ax5.transAxes,
         fontsize=15, fontweight='bold', va='bottom')
ax5.set_title('Required volume', fontsize=13, pad=6)

_shade_regimes(ax5)


# ─────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────
plt.savefig('tradeoff_combined.svg', dpi=300, bbox_inches='tight')
plt.savefig('tradeoff_combined.png', dpi=300, bbox_inches='tight')
plt.show()


# ─────────────────────────────────────────────────────
# Print summary table
# ─────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Q-sweep summary for Da_c ≈ {Da_c_ana:.1f} system")
print(f"{'='*70}")
print(f"{'Q/Qc':>8} {'Pe_H':>8} {'Da':>8} {'Da(ana)':>8} {'t_eq/tR':>8} {'V/Vmin':>10} {'CD?':>5}")
print(f"{'-'*70}")
for _, row in df_q.sort_values("q").iterrows():
    q = row["q"]
    Da_ana = Da_c_ana / q if q <= 1 else Da_c_ana / q ** (1 / 3)
    V_ana  = 1 + q / Da_c_ana if q <= 1 else q ** (2 / 3) + q / Da_c_ana
    print(f"{q:8.2f} {row['Pe_H']:8.1f} {row['Da']:8.2f} {Da_ana:8.2f} "
          f"{row['t_eq_norm']:8.2f} {row['V_req_over_Vmin']:10.2f} "
          f"{'Yes' if q <= 1 else 'No':>5}")