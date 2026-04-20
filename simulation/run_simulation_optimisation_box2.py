from biosensor.parameters.parameters_Madaboosi2015 import params
from biosensor.model.simulate_ODE import simulate
from biosensor.plots.plot_results_optimisation import *
import numpy as np
import pandas as pd
from tqdm import tqdm
from biosensor.model.calculate_Sherwood import *

# print results in consol
print_results = False
plot_data = True
export_data = False
plot_results = False

# SI units
params.c_in = params.c_in * 1e3
params.k_on = params.k_on * 1e-3
params.c_0 = params.c_0 * 1e3

# Store original values to reset between scenarios
original_H_c = params.H_c
original_b_m = params.b_m
original_L_c = params.L_c

# Fixed concentration (0.38 nM, near LoD)
c_in = 0.38e-9 * 1e3  # mol/m3

# Flow rate range
Q_in_uL_min = np.logspace(-1, 1, 9)
Q_conversion_factor = (1/60) * 1e-9
Q_in_vals = Q_in_uL_min * Q_conversion_factor

# Define scenarios: label + parameter overrides
scenarios = [
    {"label": "Original",
     "H_c": 20e-6, "b_m": 1.5e-7, "L_c": original_L_c},
    {"label": "H_c / 2",
     "H_c": 10e-6, "b_m": 1.5e-7, "L_c": original_L_c},
    {"label": "b_m / 10",
     "H_c": 20e-6, "b_m": 1.5e-8, "L_c": original_L_c},
]

results = []
total = len(scenarios) * len(Q_in_vals)

for scenario in tqdm(scenarios, desc="Scenarios"):
    # Apply parameter overrides
    params.H_c = scenario["H_c"]
    params.b_m = scenario["b_m"]
    params.L_c = scenario["L_c"]
    params.c_in = c_in

    for Q_in in Q_in_vals:
        params.Q_in = Q_in
        V_in = params.V_in

        # System characteristics
        Pe_H = Q_in / (params.D * params.W_c)
        Lambda = params.L_s / params.H_c

        # Sherwood
        output = compute_k_m(Q_in, params)
        F = output[1]
        full_collection = F > 0.95 * Pe_H

        result = simulate(params, print_results, plot_results)

        results.append({
            "label": scenario["label"],
            "Q_in": Q_in,
            "V_in": V_in,
            "full_collection": full_collection,
            **result
        })

# Reset params
params.H_c = original_H_c
params.b_m = original_b_m
params.L_c = original_L_c

df = pd.DataFrame(results)



# ── Plotting ──
if plot_data:
    fig, ax1 = plt.subplots(figsize=(7, 6))
    ax2 = ax1.twinx()

    colors = ["#2a2a2a", "#1a7a4a", "#c45a20"]
    labels_short = df["label"].unique()

    for i, lbl in enumerate(labels_short):
        sub = df[df["label"] == lbl].copy()
        sub = sub[np.isfinite(sub["time_eq"]) & np.isfinite(sub["V_eq"])]

        x = sub["Q_in"].values * 1e9 * 60
        y_t = sub["time_eq"].values
        y_v = (sub["Q_in"] * sub["time_eq"]).values * 1e9

        idx = np.argsort(x)
        x, y_t, y_v = x[idx], y_t[idx], y_v[idx]

        color = colors[i]

        # Volume (solid, squares)
        ax1.plot(x, y_v, marker="s", linewidth=2, color=color,
                 linestyle="-", label=lbl, markersize=6)

        # t_eq (dotted, circles — filled/empty for delivery regime)
        ax2.plot(x, y_t, marker=None, linewidth=2, color=color, linestyle=":")

        fc_sorted = sub["full_collection"].values[idx]
        for xi, yi, filled in zip(x, y_t, fc_sorted):
            ax2.plot(xi, yi, marker='o', markersize=7,
                     markerfacecolor=color if filled else 'none',
                     markeredgecolor=color)

    # Operating flow rate
    #ax1.axvline(0.5, color='gray', linestyle='--', linewidth=1)
    #ax1.text(0.53, 2, 'Q = 0.5 µL/min', fontsize=9, color='gray', rotation=90)

    # 5 µL available volume line
    #   ax1.axhline(5, color='indianred', linestyle='--', linewidth=1, alpha=0.6)
    #ax1.text(0.12, 5.5, '5 µL available', fontsize=9, color='indianred')

    # Labels
    ax1.set_xlabel("Flow rate [µL/min]")
    ax1.set_ylabel("Volume [µL]")
    ax2.set_ylabel("Equilibration time [s]")

    # Log scales and limits
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax2.set_yscale("log")
    ax1.set_xlim(1e-1, 1e1)
    ax1.set_ylim(1e0, 5e2)
    ax2.set_ylim(1e2, 5e4)

    # Grid
    ax1.grid(True, which="major", ls="-", alpha=0.8)
    ax1.grid(True, which="minor", ls="-", alpha=0.2)

    # Legend
    ax1.legend(title="Scenario", loc='lower right', fontsize=9)

    fig.tight_layout()
    plt.show()