from parameters_QCM import params
from biosensor_project.biosensor.model.simulate_ODE import simulate
from plot_results import *
from tqdm import tqdm

# script to optimize flow rate for capture efficiency and/or eq. time

# print results in consol
print_results = False

# show graphs
plot_results = False

# simulation time [s]
max_time = 15*60

Q_low = 1e-2
Q_high = 100
Q_points = 30
Q_in_uL_min = np.linspace(Q_low, Q_high, Q_points)

Q_conversion_factor = (1/60) * 10 ** (-9)

Q_in_vals = Q_in_uL_min * Q_conversion_factor

results = []
total = len(Q_in_vals)

# SI units
params.c_in = params.c_in * 1e3  # input concentration in SI units [mol/m3]
params.k_on = params.k_on * 1e-3  # on rate in SI units [mol^-1 m^-3 s^-1]
params.c_0 = params.c_0 * 1e3

for Q_in in tqdm(Q_in_vals,total=total,desc="Running simulations"):
    params.Q_in = Q_in

    result = simulate(params, print_results, plot_results,max_time)
    results.append({
        "Q_in": Q_in,
        **result
    })

df = pd.DataFrame(results)

exp_data = {
    'Bound (sim H/2)': [0.9327, 0.0037, 0.911],
    'Bound (sim H)': [0.866, 0.00079, 0.81],
    'Bound (exp.)': [0.8935, 0.0047, 0.8497]
}

plot_optimization2(df, params, Q_in_vals, exp_data)

