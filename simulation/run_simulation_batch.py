# from biosensor.parameters.parameters import params
from biosensor.parameters.parameters_QCM import params
from biosensor.model.simulate_ODE import simulate
from biosensor.plots.plot_results_batch import *
from biosensor.model.simulate_sensivitity import simulate_with_sensitivity
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
from biosensor.utils.save_results import save_simulation_results
from biosensor.plots.plot_results_other import *

# print results in consol
print_results = False
plot_data = True
export_data = False

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
#D_vals = [params.D, 10*params.D, 0.1*params.D]      # Diffusion coefficient [m^2/s]
D_vals = [1e-9, 1e-10, 1e-11]
#D_vals = [1e-10] # Diffusion coefficient
c_in_vals = np.logspace(-8,-5,6) * 1e3  #input concentration [mol/L = M]
#c_in_vals = np.array([1e-7]) * 1e3
# 100 nM - 10 uM - 1 mM
#c_in_vals = [params.c_in]  #input concentration [mol/L = M]
print(c_in_vals)


#V_in_vals = [100e-9]   # Input volume [m3, 1e-9  = 1uL]
L_s_vals = [params.L_s]
V_in_vals = [params.V_in]      # input volume [m^3]
H_c_vals = [params.H_c]
#H_c_vals = [params.H_c]
#Q_in_uL_min = np.array([50])
Q_in_uL_min = np.logspace(0,5,5)                # flow rate in uL/min
#Q_in_uL_min = [100]
#b_m_vals = [params.b_m, params.b_m/100]
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

    result = simulate(params, print_results, plot_results)
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
    #plot_peclet_batch(df, save_path=os.path.join(plot_dir, "peclet_batch.png"))
    #plot_time_eq_interp(df, save_path=os.path.join(plot_dir, "time_eq_interp.png"))
    #plot_capt_perc_interp(df, save_path=os.path.join(plot_dir, "capture_percentage.png"))
    #plot_site_occupancy_interp(df, params, save_path=os.path.join(plot_dir, "site_occupancy.png"))

if plot_data == True:
    # showcase summary plots

    # optimization for single volume
    #plot_optimization(df,params,Q_in_vals)

    # plot damkohler [df, [Pe_H, Q_in]]
    #plot_damkohler_batch(df,"Q_in")

    #plot_time_eq_interp(df)
    #plot_capt_perc_interp(df)
    #plot_site_occupancy_interp(df, params)

    # plot error
    #plot_error(df)

    # plot overview
    #plot_km_Q(df)
    plot_t_eq_two_axis_scatter_new(df)
    t_eq_new(df,'time_eq.svg')
    plot_V_min_vs_Q_in_new(df)
    plot_t_eq_two_axis_scatter(df)
    #plot_V_min_two_axis_scatter(df)
    plot_V_min_vs_Q_in(df)
    #plot_timescale_collapse_with_labels(df)
    #plot_relative_rate_map(df)
    #plot_t_eq_overview_scatter3D(df)
    #plot_V_min_overview_scatter3D(df)
    #plot_t_eq_overview_scatter3D_animated(df, "test.png", "test.gif")
    #plot_t_eq_overview_scatter(df)
    plot_Da_overview_scatter(df)
    #plot_volume_required_scatter(df)

    # plot
    # plot_peclet_batch(df)
    # #plot_flow_volume(df)
    # plot_time_eq_interp(df)
    # plot_capt_perc_interp(df)   # capture percentage - good for full capture systems
    # plot_site_occupancy_interp(df)  # occupancy rate - good for equilibrium systems (?)
    # #plot_capture_vs_peH_lambda(df)