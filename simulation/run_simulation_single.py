from biosensor.parameters.parameters import params
#from biosensor.parameters.parameters_Madaboosi2015 import params
#from biosensor.parameters.parameters_QCM import params
from biosensor.model.simulate_ODE import simulate
from biosensor.plots.plot_results_single import *
from biosensor.utils.save_results import save_simulation_results
import pandas as pd
import os
from datetime import datetime


# simulation
print_results = True
plot_results = True
export_data = False

startTime = datetime.now()

# SI units
Q_conversion_factor = (1/60) * 10 ** (-9)
params.Q_in = params.Q_in * Q_conversion_factor
params.c_in = params.c_in * 1e3  # input concentration in SI units [mol/m3]
params.k_on = params.k_on * 1e-3  # on rate in SI units [mol^-1 m^-3 s^-1]
params.c_0 = params.c_0 * 1e3

# Define maximum stimulation time   (if None: 3x injection time)
max_time = None

# simulate
results = simulate(params, print_results, plot_results,max_time)

df = pd.DataFrame(results)

t_pulse = df["t_pulse_hat"].values[0] * df["tau"].values[0]

elapsed = datetime.now() - startTime
print(f"Simulation time: {elapsed.total_seconds():.2f} seconds")


# export data
if export_data == True:
    csv_file = save_simulation_results(df, params, run_type="single", file_format="csv")

    # Save plots in same folder
    plot_dir = os.path.join(os.path.dirname(csv_file), "plots")
    os.makedirs(plot_dir, exist_ok=True)

    plot_time_series(df,t_pulse,
        save_path=os.path.join(plot_dir, "time_series.svg"))

    plot_dimensionless(df,t_pulse,
        save_path=os.path.join(plot_dir, "dimensionless.png"))

    plot_system_performance(df, t_pulse,
        save_path=os.path.join(plot_dir, "system_performance.png"))

    plot_mass_balance_error(df,t_pulse,
        save_path=os.path.join(plot_dir, "error.png"))

if plot_results == True:

    # plot time series of mol values
    plot_time_series(df,t_pulse)

    # plot_cs_time(df,t_pulse)

    # plot time series of dimensionless values
    plot_dimensionless(df, t_pulse)

    # plot system performance
    plot_system_performance(df, t_pulse)

    # plot error over time
    plot_mass_balance_error(df,t_pulse)

    # Damkohler number over time (useful?)
    plot_Damkohler_time(df)
