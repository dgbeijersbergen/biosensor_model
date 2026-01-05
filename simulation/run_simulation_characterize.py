from biosensor.parameters.parameters_QCM import params
#from biosensor.parameters.parameters_Madaboosi2015 import params
from biosensor.model.simulate_ODE import simulate
from biosensor.plots.plot_results_characterize import *
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
from biosensor.utils.save_results import save_simulation_results
from biosensor.plots.plot_results_other import *
from biosensor.model.calculate_Sherwood import *

# print results in consol
print_results = False
plot_data = True
export_data = False

# show graphs of individual measurements
plot_results = False

# SI units
params.c_in = params.c_in * 1e3  # input concentration in SI units [mol/m3]
params.k_on = params.k_on * 1e-3  # on rate in SI units [m^3 mol^-1 s^-1]
params.c_0 = params.c_0 * 1e3

# grid size simulation
k = 9

# parameter ranges
c_in_vals = np.array([1e2*1e-9]) * 1e3
V_in_vals = np.logspace(-9, -4,k)   # Input volume [m3, 1e-9  = 1uL]
Q_in_uL_min = np.logspace(-1,3,k)                # flow rate in uL/min

Q_conversion_factor = (1/60) * 10 ** (-9)
Q_in_vals = np.array(Q_in_uL_min) * Q_conversion_factor   # convert to m3/s

results = []
total =  len(Q_in_vals) * len(V_in_vals) * len(c_in_vals) # number of combinations

# loop over parameters and calculate results
for Q_in, V_in, c_in in tqdm(itertools.product(Q_in_vals, V_in_vals, c_in_vals),total=total,desc="Running simulations"):
    params.Q_in = Q_in
    params.V_in = V_in
    params.c_in = c_in

    # system charactersitics
    Pe_H = params.Q_in / (params.D * params.W_c)
    Lambda = params.L_s / params.H_c  # ratio of sensor length to channel height

    # obtain k_m from calculate_Sherwood.py, with F minimum of 1 (pure diffusion)
    output = compute_k_m(Q_in,params)
    F = output[1]

    if F > 0.95*Pe_H:
        full_collection = True
    else:
        full_collection = False

    result = simulate(params, print_results, plot_results)
    results.append({
        "Q_in": Q_in,
        "V_in": V_in,
        "full_collection": full_collection,
        **result
    })




df = pd.DataFrame(results)


if export_data == True:
    csv_file = save_simulation_results(df, params, run_type="batch", file_format="csv")
    df.to_pickle("simulation_results.pkl")

    plot_dir = os.path.join(os.path.dirname(csv_file), "plots")
    os.makedirs(plot_dir, exist_ok=True)


    # Save summary plots
    #plot_peclet_batch(df, save_path=os.path.join(plot_dir, "peclet_batch.png"))
    #plot_time_eq_interp(df, save_path=os.path.join(plot_dir, "time_eq_interp.png"))
    #plot_capt_perc_interp(df, save_path=os.path.join(plot_dir, "capture_percentage.png"))
    #plot_site_occupancy_interp(df, params, save_path=os.path.join(plot_dir, "site_occupancy.png"))

if plot_data == True:
    # plot
    plot_site_occupancy_interp(df, params, grid_size=9, save_path=None) # ,
    plot_time_eq_interp(df)
    #plot_damkohler_varying_c(df, 'damkohler_QCM.svg')
    plot_damkohler_batch(df, "Q_in")
    plot_flow_volume(df)
    # plot_capt_perc_interp(df)   # capture percentage - good for full capture systems
    # #plot_capture_vs_peH_lambda(df)