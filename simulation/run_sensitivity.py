"""
Sensitivity analysis for biosensor model.
Varies one parameter at a time over a specified range while keeping all
others fixed at their QCM baseline values. Output metric: time_eq.
"""

import copy
import numpy as np
import pandas as pd
from tqdm import tqdm

from biosensor.parameters.parameters_QCM import params as _base_params
from biosensor.model.simulate_ODE import simulate

# ---------------------------------------------------------------------------
# 1.  Convert baseline params to SI (same convention as run_simulation_batch)
# ---------------------------------------------------------------------------
baseline = copy.deepcopy(_base_params)
baseline.c_in = baseline.c_in * 1e3       # mol/L  →  mol/m³
baseline.k_on = baseline.k_on * 1e-3      # L mol⁻¹ s⁻¹  →  m³ mol⁻¹ s⁻¹
baseline.c_0  = baseline.c_0  * 1e3
baseline.Q_in = baseline.Q_in * (1/60) * 1e-9   # µL/min  →  m³/s

# ---------------------------------------------------------------------------
# 2.  Define parameters to sweep and their ranges
#     Each entry:  (attr_name, display_label, unit_label, values_array)
#     Values are already in SI units.
# ---------------------------------------------------------------------------
N_POINTS = 15   # number of points per parameter sweep

def _logspace_around(center, decades=3, n=N_POINTS):
    """Return n log-spaced values spanning ±decades around center."""
    return np.logspace(
        np.log10(center) - decades,
        np.log10(center) + decades,
        n
    )

sensitivity_params = [
    (
        "k_on",
        r"$k_{on}$",
        r"m$^3$ mol$^{-1}$ s$^{-1}$",
        _logspace_around(baseline.k_on),
    ),
    (
        "k_off",
        r"$k_{off}$",
        r"s$^{-1}$",
        _logspace_around(baseline.k_off),
    ),
    (
        "D",
        r"$D$",
        r"m$^2$ s$^{-1}$",
        _logspace_around(baseline.D),
    ),
    (
        "c_in",
        r"$c_{in}$",
        r"mol m$^{-3}$",
        _logspace_around(baseline.c_in),
    ),
    (
        "Q_in",
        r"$Q_{in}$",
        r"m$^3$ s$^{-1}$",
        _logspace_around(baseline.Q_in, decades=3),
    ),
    (
        "H_c",
        r"$H_c$",
        r"m",
        _logspace_around(baseline.H_c),
    ),
    (
        "b_m",
        r"$b_m$",
        r"mol m$^{-2}$",
        _logspace_around(baseline.b_m),
    ),
    (
        "V_in",
        r"$V_{in}$",
        r"m$^3$",
        _logspace_around(baseline.V_in),
    ),
    (
        "L_s",
        r"$L_s$",
        r"m",
        _logspace_around(baseline.L_s),
    ),
]

# ---------------------------------------------------------------------------
# 3.  Run OAT (one-at-a-time) sensitivity analysis
# ---------------------------------------------------------------------------
print_results = False
plot_results  = False

all_records = []

for attr, label, unit, values in sensitivity_params:
    print(f"\nSweeping {label}  ({attr})  over {len(values)} points …")
    for val in tqdm(values, desc=attr):
        p = copy.deepcopy(baseline)
        setattr(p, attr, val)
        try:
            result = simulate(p, print_results, plot_results)
            time_eq = result["time_eq"]
            k_m     = result["k_m"]
            F       = result["F"]
            Pe_H    = result["Pe_H"]
            Da_1      = result["Da_1"]
            Da_2      = result["Da_2"]
            Da_3 =      result["Da_3"]
            Da_squires      = result["Da_squires"]
            # full collection: F was capped at Pe_H inside simulate(), so
            # if stored F == Pe_H the original F exceeded Pe_H
            full_collection = np.isclose(F, Pe_H, rtol=1e-6)
        except Exception as e:
            print(f"  Simulation failed for {attr}={val:.3e}: {e}")
            time_eq = np.nan
            k_m     = np.nan
            F       = np.nan
            Pe_H    = np.nan
            Da_1      = np.nan
            Da_2    = np.nan
            Da_3 = np.nan
            Da_squires      = np.nan
            full_collection = False

        all_records.append({
            "param_attr":     attr,
            "param_label":    label,
            "param_unit":     unit,
            "param_value":    val,
            "baseline_value": getattr(baseline, attr),
            "relative_value": val / getattr(baseline, attr),
            "time_eq":        time_eq,
            "k_m":            k_m,
            "F":              F,
            "Pe_H":           Pe_H,
            "Da_1":           Da_1,
            "Da_2":           Da_2,
            "Da_3":             Da_3,
            "Da_squires":     Da_squires,
            "full_collection": full_collection,
        })

df_sens = pd.DataFrame(all_records)
df_sens.to_pickle("sensitivity_results.pkl")
df_sens.to_csv("sensitivity_results.csv", index=False)
print("\nResults saved to sensitivity_results.pkl / .csv")

# ---------------------------------------------------------------------------
# 4.  Plot
# ---------------------------------------------------------------------------
from biosensor.plots.plot_sensitivity import plot_sensitivity_all
plot_sensitivity_all(df_sens, baseline)