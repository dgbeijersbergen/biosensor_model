# from biosensor.parameters.parameters import params
from biosensor.parameters.parameters_QCM import params
from biosensor.model.simulate_ODE import simulate
from biosensor.plots.plot_results_batch import *
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
from biosensor.utils.save_results import save_simulation_results
from biosensor.plots.plot_results_batch import *

# print results in consol
print_results = False
plot_data = True
export_data = True

# show graphs of individual measurements
plot_results = False

k_on_vals = np.logspace(4,6,3) * 1e-3
#k_on_vals = np.array([params.k_on]) * 1e-3

#k_off_vals = np.logspace(-6,-1,1)
#k_off_vals = [params.k_off*0.1, params.k_off, params.k_off*10]
k_off_vals = [params.k_off]

# SI units
params.c_in = params.c_in * 1e3  # input concentration in SI units [mol/m3]
params.k_on = params.k_on * 1e-3  # on rate in SI units [m^3 mol^-1 s^-1]
params.c_0 = params.c_0 * 1e3

# parameter ranges
D_vals = [1e-9, 1e-10, 1e-11]
c_in_vals = np.logspace(-8,-5,6) * 1e3  #input concentration [mol/L = M]
L_s_vals = [params.L_s]
V_in_vals = [params.V_in]      # input volume [m^3]
H_c_vals = [params.H_c]
Q_in_uL_min = np.logspace(0,5,5)                # flow rate in uL/min
b_m_vals = [params.b_m]

Q_conversion_factor = (1/60) * 10 ** (-9)
Q_in_vals = np.array(Q_in_uL_min) * Q_conversion_factor   # convert to m3/s

results = []
total = len(L_s_vals) * len(b_m_vals) * len(k_on_vals) * len(k_off_vals) * len(c_in_vals) * len(D_vals) * len(H_c_vals) * len(Q_in_vals) * len(V_in_vals) # number of combinations

# loop over parameters and calculate results
for k_on, k_off, c_in, Q_in, D, H_c, V_in, b_m, L_s in tqdm(itertools.product(k_on_vals, k_off_vals, c_in_vals, Q_in_vals, D_vals, H_c_vals, V_in_vals, b_m_vals, L_s_vals),total=total,desc="Running simulations"):
    params.k_on = k_on
    params.k_off = k_off
    params.c_in = c_in
    params.Q_in = Q_in
    params.D = D
    params.H_c = H_c
    params.V_in = V_in
    params.b_m = b_m
    params.L_s = L_s

    result = simulate(params, print_results, plot_results, wait_error_response=False)
    results.append({
        "k_on": k_on,
        "k_off": k_off,
        "D": D,
        "Q_in": Q_in,
        "H_c": H_c,
        "V_in": V_in,
        "c_in": c_in,
        **result
    })

df = pd.DataFrame(results)

if export_data == True:
    csv_file = save_simulation_results(df, params, run_type="batch", file_format="csv")
    df.to_pickle("simulation_results.pkl")

    plot_dir = os.path.join(os.path.dirname(csv_file), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    # Save summary plots

if plot_data == True:
    True