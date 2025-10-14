from biosensor.model.biosensor_model import ode_binding_hat
from scipy.integrate import solve_ivp
from biosensor.plots.plot_results import *
from biosensor.model.calculate_Sherwood import F_combine, compute_k_m

def simulate(params, print_results = False, plot_results = False, max_time = None):

    # unpack params
    W_c, L_c, H_c = params.W_c, params.L_c, params.H_c
    D = params.D
    k_on, k_off, b_m, L_s, W_s = params.k_on, params.k_off, params.b_m, params.L_s, params.W_s
    c_0, c_in, V_in, Q_in, flow_off = params.c_0, params.c_in, params.V_in, params.Q_in, params.flow_off


    # initial conditions
    # y0_hat = [0, c_0 / c_in, 0, 0] # b0, c0, N0, c_s_hat
    y0_hat = [0, params.c_0 / params.c_in, 0] # b0, c0, N0


    # time range (in terms of residence times)
    V = W_c * L_c * H_c  # channel volume [m^3]
    tau = V / Q_in  # residence time

    t_pulse_hat = V_in / V  # nondimensional pulse duration = V_in / V

    # time length of simulation (if undefined: 3x pulse time)
    if max_time == None:
        t_span_hat = (0, 3 * t_pulse_hat)
    else:
        max_time_hat = max_time / tau   # non dimensionalized time
        t_span_hat = (0, max_time_hat)  # simulate specified time

    dt_hat = 1e-4  # nondimensional time step
    nt = int(np.ceil((t_span_hat[1] - t_span_hat[0]) / dt_hat)) + 1
    t_eval_hat = np.linspace(t_span_hat[0], t_span_hat[1], nt)

    # solve biosensor ODE (best result: LSODA)
    sol = solve_ivp(ode_binding_hat, t_span_hat, y0_hat, method='LSODA',t_eval=t_eval_hat, args=(params,))

    # --- get results --- #
    t_hat = sol.t       # type: ignore
    b_hat = sol.y[0]    # type: ignore
    c_hat = sol.y[1]    # type: ignore
    N_out_hat = sol.y[2]    # type: ignore

    # calculate c_s analytically (repeat from ODE)
    c_s_vals = []
    for t_i, b_i, c_i in zip(t_hat,b_hat,c_hat):
        if t_i >= t_pulse_hat:
            Q_in_temp = 0
        else:
            Q_in_temp = params.Q_in

        k_m = compute_k_m(Q_in_temp, params)

        numerator = (k_m * params.c_in * c_i) + (params.k_off * params.b_m * b_i)
        denom = k_m + params.k_on * (params.b_m - (params.b_m * b_i))
        c_s_val = max(numerator / (params.c_in * denom), 0.0)

        c_s_vals.append(c_s_val)

    c_s_hat = np.array(c_s_vals)

    #c_s_hat = sol.y[3]  # type: ignore

    # System parameters
    V = W_c * L_c * H_c         # channel volume [m^3]
    S = L_s * W_s               # sensor area [m^2]

    tau = V / Q_in              # residence time of molecules [s]

    # values with dimensions
    t = t_hat * tau                 # s
    b = b_hat * b_m              # mol/m^2
    c = c_hat * c_in             # mol/m^3
    c_s = c_s_hat * c_in        # mol/m^3
    N_out = N_out_hat * (c_in * V)   # mol

    # cumulative injected (dimensional)
    N_injected = np.minimum(c_in * Q_in * t, c_in * V_in)  # mol

    ## -- analysis -- ##

    # conservation check (should be near zero)
    residual = N_injected - N_out - b * S - c * V


    # system characteristics
    Pe_H = Q_in / (D * W_c)                 # Peclet number (channel)
    Lambda = L_s / H_c                      # ratio of sensor length to channel height
    Pe_s = 6 * (Lambda ** 2) * Pe_H         # Peclet number (shear)
    F = F_combine(Pe_H, Lambda)             # Sherwood number
    if F > Pe_H + 1:
        F = Pe_H + 1

    k_m = F * (D / (H_c/2))               # mass transport rate

    # Damkohler number
    Da = (k_on * b_m) / k_m                 # definition like in Squires
    Da_2 = k_on * c_in * tau                # alternative definition
    Da_t = (k_on * (b_m - b)) / k_m         # time dependent definition

    # results
    N_loss = N_out[-1]             # lost molecules [mol]
    loss_perc = 100 * (N_loss / N_injected[-1]) # percentage lost
    N_capt = b * S                          # captured molecules (array) [mol]
    capt_perc = 100 * (N_capt[-1] / N_injected[-1])      # percentage captured
    N_bulk = c[-1] * V
    N_bulk_conc = c[-1]
    bulk_perc = 100 * (N_bulk / N_injected[-1])      # percentage captured

    b_eq = (k_on * c_in * b_m) / (k_on * c_in + k_off)

    b_hat_eq = b / b_eq
    N_eq = b_eq * S
    eq_target = 0.95    # percentage of equilibrium to reach as target

    capt_target = 0.90

    # find the first index where captured molecules satisfies equilbrium target
    indices_eq = np.where(N_capt >= eq_target * N_eq)[0]

    # find the first index where captured molecules satisfies equilbrium target
    indices_capt = np.where(N_capt >= capt_target * N_injected[-1])[0]

    if len(indices_eq) > 0 and len(indices_capt) > 0:
        reached_eq = True
        time_eq = t[indices_eq[0]]
        time_capt = t[indices_capt[0]]

    elif len(indices_eq) > 0 and len(indices_capt) == 0:
        reached_eq = True
        time_eq = t[indices_eq[0]]
        time_capt = np.inf

    elif len(indices_eq) == 0 and len(indices_capt) > 0:
        reached_eq = False
        time_eq = np.inf
        time_capt = t[indices_capt[0]]

    else:
        reached_eq = False
        time_capt = np.inf
        time_eq = np.inf

    if print_results == True:
        print("Simulation of system")
        print(f"Flow rate [uL/min]:, {Q_in*60*1e9:.2f}")
        print(f"Volume [uL]:, {V*1e9}")
        print("Max abs conservation residual [mol]:", np.max(np.abs(residual)))
        print("-----------")
        print("System charactersitics:")
        print(f"Channel Peclet (Pe_H): {Pe_H:.2f}")
        print(f"Sensor Peclet (Pe_s): {Pe_s:.2f}")
        print(f"Ratio of sensor length to channel height (lambda) {Lambda:.2f}")
        print(f"Sherwood number: {F:.2f}")
        print(f"Mass transport rate (k_m): {k_m:.3e}")
        print(f"Damkohler number (Da): {Da:.2e}")

        print("-----------")

        print("System performance (at end):")
        print(f"Lost molecules [mol] : {N_loss:.3e}")
        print(f"Lost molecules [%] : {loss_perc:.2f}")
        print(f"Captured molecules [mol] : {N_capt[-1]:.3e}")
        print(f"Captured molecules [%] : {capt_perc:.2f}")
        print(f"Bulk molecules remaining [mol] : {N_bulk:.3e}")
        print(f"Bulk molecules  [%] : {bulk_perc:.2f}")
        print(f"Bulk concentration [mol/L] : {N_bulk_conc:.3e}")

        print("--- equilbirium ---")
        print(f"Bound equilibrium [mol] [%] : {N_eq:.3e}")
        print(f"Bound equilibrium [M] [%] : {N_eq/S:.3e}")
        print("Reached equilbirium: ", reached_eq)
        print(f"Time to equilibrium [s] : {time_eq:.2f}")

    ## --- plotting --- ##
    if plot_results == True:
        # plot damkohler number
        # plot_Damkohler_time(t, Da_t)

        # plot time series of mol values
        plot_time_series(t, b, c, c_s, N_injected, N_out, S, V)

        # plot time series of dimensionless values
        plt.subplot(1, 1, 1)
        plt.figure(figsize=(8, 5))
        plot_dimensionless(t, S, N_injected, N_out, b,b_hat, b_hat_eq, c_hat, t_pulse_hat * tau)

        plt.tight_layout()
        plt.show()

    return {
        "t": t,
        "b": b,
        "c": c,
        "S": S,
        "b_m": b_m,
        "b_eq": b_eq,
        "c_s": c_s,
        "Pe_H": Pe_H,
        "Pe_s": Pe_s,
        "Lambda": Lambda,
        "F": F,
        "k_m": k_m,
        "Da": Da,
        "Da_2": Da_2,
        "Da_t": Da_t,
        "N_injected": N_injected,
        "N_out": N_out,
        "N_loss": N_loss,
        "N_eq": N_eq,
        "reached_eq": reached_eq,
        "time_eq": time_eq,
        "time_capt": time_capt,
        "N_capt": N_capt,   # array of captured moles
        "loss_perc": loss_perc,
        "capt_perc": capt_perc, # one value
        "bulk_perc": bulk_perc,
        "N_bulk": N_bulk,
        "N_bulk_conc": N_bulk_conc
    }


