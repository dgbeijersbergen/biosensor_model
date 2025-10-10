from biosensor_project.biosensor.parameters.parameters_QCM import params
from biosensor_project.biosensor.model.simulate_ODE import simulate
from biosensor_project.biosensor.plots.plot_results import *
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
from biosensor_project.biosensor.utils.save_results import save_simulation_results


# print results in consol
print_results = False
export_data = True

# show graphs
plot_results = False

# parameter ranges
D_vals = [1.11e-11]      # Diffusion coefficient
#D_vals = [5e-10]     # Diffusion coefficient
c_in_vals = [50e-9]  #input concentration [mol/L = M]

#V_in_vals = [500e-9, 100e-9, 50e-9, 30e-9, 5e-9, 1e-9,0.5e-9]   # Input volume [m3, 1e-9  = 1uL]
V_in_vals = np.logspace(-10,-6,10)  # input volume [m3]
#V_in_vals = [5000e-9]   # Input volume [m3, 1e-9  = 1uL]
H_c_vals = [params.H_c]


#Q_in_uL_min = np.array([50])
# Q_in_uL_min = np.array([0.1e1,1e1,1e2,1e3,1e4,1e5])                   # flow rate in uL/min
Q_in_uL_min = np.logspace(-3,5,10)

#Q_in_vals = [1e-12,1e-11,1e-10,5e-10,1e-9]

# SI units
params.c_in = params.c_in * 1e3  # input concentration in SI units [mol/m3]
params.k_on = params.k_on * 1e-3  # on rate in SI units [mol^-1 m^-3 s^-1]
params.c_0 = params.c_0 * 1e3
Q_conversion_factor = (1/60) * 10 ** (-9)
Q_in_vals = Q_in_uL_min * Q_conversion_factor   # convert to m3/s

results = []
total = len(c_in_vals) * len(D_vals) * len(H_c_vals) * len(Q_in_vals) * len(V_in_vals) # number of combinations

# loop over parameters and calculate results
for c_in, Q_in, D, H_c, V_in in tqdm(itertools.product(c_in_vals,Q_in_vals, D_vals, H_c_vals, V_in_vals),total=total,desc="Running simulations"):
    params.c_in = c_in
    params.Q_in = Q_in
    params.D = D
    params.H_c = H_c
    params.V_in = V_in

    result = simulate(params, print_results, plot_results)
    results.append({
        "D": D,
        "Q_in": Q_in,
        "H_c": H_c,
        "V_in": V_in,
        **result
    })

df = pd.DataFrame(results)

# plot
# plot_peclet_batch(df)
# #plot_flow_volume(df)
# plot_time_eq_interp(df)
# plot_capt_perc_interp(df)   # capture percentage - good for full capture systems
# plot_site_occupancy_interp(df)  # occupancy rate - good for equilibrium systems (?)
# #plot_capture_vs_peH_lambda(df)

# save data
save_simulation_results(df, params, run_type="batch", file_format="csv")

if export_data:
    csv_file = save_simulation_results(df, params, run_type="batch", file_format="csv")

    plot_dir = os.path.join(os.path.dirname(csv_file), "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Save summary plots
    plot_peclet_batch(df, save_path=os.path.join(plot_dir, "peclet_batch.png"))
    #plot_time_eq_interp(df, save_path=os.path.join(plot_dir, "time_eq_interp.png"))
    #plot_capt_perc_interp(df, save_path=os.path.join(plot_dir, "capture_percentage.png"))
    plot_site_occupancy_interp(df, save_path=os.path.join(plot_dir, "site_occupancy.png"))