#from parameters import params
from biosensor.parameters.parameters_QCM import params
from biosensor.model.simulate_ODE import simulate
from biosensor.plots.plot_results import *
from biosensor.utils.save_results import save_simulation_results
import os

# simulation
print_results = True
plot_results = True
export_data = True
simulate_flow = False


# define simulation parameters
max_time = 15*60 # [s]
c_in = 10e-6     # uM (mol/L)
Q_in_uL = 50    # flow rate [uL/min]
V_in = 50e-9     # uL

# change model parameters
Q_conversion_factor = (1/60) * 10 ** (-9)
Q_in = Q_in_uL * Q_conversion_factor
params.Q_in = Q_in
params.c_in = c_in
params.V_in = V_in
params.char_length = 'H'

# SI units
params.c_in = params.c_in * 1e3  # input concentration in SI units [mol/m3]
params.k_on = params.k_on * 1e-3  # on rate in SI units [mol^-1 m^-3 s^-1]
params.c_0 = params.c_0 * 1e3

# simulate
results = simulate(params, print_results, plot_results,max_time)

# export data
if export_data == True:
    csv_file = save_simulation_results(results, params, run_type="single", file_format="csv")

    # Save plots in same folder
    plot_dir = os.path.join(os.path.dirname(csv_file), "plots")
    os.makedirs(plot_dir, exist_ok=True)

    plot_time_series(
        results["t"], results["b"], results["c"], results["c_s"],
        results["N_injected"], results["N_out"], results["S"], results["V"],
        save_path=os.path.join(plot_dir, "time_series.png")
    )