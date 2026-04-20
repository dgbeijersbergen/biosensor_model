from biosensor.model.biosensor_model import *
from scipy.integrate import solve_ivp
from biosensor.plots.plot_results_single import *
from biosensor.plots.plot_results_batch import *
from biosensor.model.calculate_Sherwood import F_combine, compute_k_m
import cProfile

def simulate(params, print_results = False, plot_results = False, max_time = None, sharpness=4, wait_error_response=True):
    # unpack params
    W_c, L_c, H_c = params.W_c, params.L_c, params.H_c
    D = params.D
    k_on, k_off, b_m, L_s, W_s = params.k_on, params.k_off, params.b_m, params.L_s, params.W_s
    c_0, c_in, V_in, Q_in, flow_off = params.c_0, params.c_in, params.V_in, params.Q_in, params.flow_off

    # initial conditions
    y0_hat = [0, params.c_0 / params.c_in, params.c_0 / params.c_in, 0, 0] # b0, c0, c_s_hat,  N0, N0

    # fraction of filling possible
    output = compute_k_m(params.Q_in, params, sharpness)
    k_m = output[0]

    delta = compute_delta(params.Q_in, params, sharpness)

    if k_m > 0 and c_in > 0:
        params.fill_frac = delta / params.H_c

    # time range (in terms of residence times)
    V = W_c * L_c * H_c  # channel volume [m^3]
    tau = V / Q_in  # residence time

    t_pulse_hat = V_in / V  # nondimensional pulse duration = V_in / V
    t_R = 1 / (k_on * c_in + k_off)


    # time length of simulation (if undefined: 3x pulse time)
    if max_time == None:
        t_span_hat = (0, 3 * t_pulse_hat)
    else:
        max_time_hat = max_time / tau   # non dimensionalized time
        t_span_hat = (0, max_time_hat)  # simulate specified time

    dt_hat = min(t_R/1000, tau/100) # define timestep of system

    # determine number of simulation points based on time step dt_hat, with a maximum of n_max
    n_max = int(5e4)
    n_plot = min(int(np.ceil((t_span_hat[1] - t_span_hat[0]) / dt_hat)) + 1, n_max)

    if max_time != None:
        time_step = max_time / n_plot

        if time_step > t_R:
            answer = input("Very large timestep. Do you want to continue anyway? (y/n): ").strip().lower()

            if answer != "y":
                print("Aborting.")
                return  # or break, or exit depending on context


    t_eval_hat = np.linspace(t_span_hat[0], t_span_hat[1], n_plot)

    # Obtain delta and k_m
    params.delta = compute_delta(params.Q_in, params, sharpness)
    params._k_m_cached, params._F_cached = compute_k_m(params.Q_in, params, sharpness)

    # validity test
    if tau > t_R:

        if wait_error_response == True:

            print("")
            print(f"tau [s] : {tau:.2f}")
            print(f"t_R [s]: {t_R:.2f}")
            print("")

            answer = input("t_R > tau (unsteady). Do you want to continue anyway? (y/n): ").strip().lower()

            if answer != "y":
                print("Aborting.")
                return  # or break, or exit depending on context
        else:
            print("t_R > tau (unsteady)")


    # solve biosensor ODE (best result: LSODA)
    # cProfile.run('solve_ivp(ode_binding_hat, t_span_hat, y0_hat, method="LSODA", t_eval=t_eval_hat, args=(params,))') # for profiling
    sol = solve_ivp(ode_binding_hat,
                    t_span_hat,
                    y0_hat,
                    method='LSODA',
                    t_eval=t_eval_hat,
                    args=(params,),
                    rtol=1e-5,
                    atol=1e-5)

    # --- get results --- #
    t_hat = sol.t           # type: ignore
    b_hat = sol.y[0]        # type: ignore
    c_hat = sol.y[1]        # type: ignore
    c_s_hat = sol.y[2]      # type: ignore
    mol_out_hat1 = sol.y[3] # type: ignore
    mol_out_hat2 = sol.y[4] # type: ignore

    db_hat_dt = []

    for i, t in enumerate(t_hat):
        y = sol.y[:, i]
        dydt = ode_binding_hat(t, y, params)
        db_hat_dt.append(dydt[0])  # first entry = db_hat_dt

    db_hat_dt = np.array(db_hat_dt)

    # System parameters
    V = W_c * L_c * H_c         # channel volume [m^3]
    S = L_s * W_s               # sensor area [m^2]

    tau = V / Q_in              # residence time of molecules [s]

    # check if targets reached
    c_eff = c_in * params.fill_frac
    b_eq = (k_on * c_in * b_m) / (k_on * c_in + k_off)     # real equilibirium

    # values with dimensions
    t = t_hat * tau                 
    b = b_hat * b_m
    db_dt = db_hat_dt * b_m / tau

    # b = b_hat * b_m   # mol/m^2               (unscaled)
    c = c_hat * c_in                # mol/m^3
    c_s = c_s_hat * c_in            # mol/m^3
    V_s = delta * W_s * L_s
    V_b = V - V_s

    mol_out = (mol_out_hat1 * c_in * V_b + mol_out_hat2 * c_in * V_s)   # mol

    b_last = b[-1]

    # cumulative injected (dimensional)
    mol_injected = np.minimum(c_in * Q_in * t, c_in * V_in)  # mol

    ## -- analysis -- ##
    # conservation check (should be near zero)
    residual = mol_injected - mol_out - (b * S) - (c * V_b) - (c_s * V_s)

    # system characteristics
    Pe_H = Q_in / (D * W_c)                 # Peclet number (channel)
    Lambda = L_s / H_c                      # ratio of sensor length to channel height
    Pe_s = 6 * (Lambda ** 2) * Pe_H         # Peclet number (shear)
    output = compute_k_m(Q_in,params, sharpness)             # Sherwood number
    k_m = output[0]
    F = output[1]

    if F > Pe_H:
        F = Pe_H

    # Damkohler number
    Da = ( (k_on * b_m) / k_m ) * (L_s / H_c)   # correct definition
    epsilon = b_eq * params.W_s / (delta * W_c * params.c_in)

    # analytical
    t_R = 1 / (k_on * c_in + k_off)
    t_eq_analytical = 3 * t_R * (1 + Da)
    V_min = 3 * W_s * L_s * k_on * b_m * t_R

    Q_c = 1.79 * Lambda * W_c * D

    Da_c = (k_on * b_m * H_c ) / (1.79 * D)

    if Q_in > Q_c:
        # print("Outside complete delivery")
        CD_regime = "Outside complete delivery"
        V_req_analytical = V_min * (1 + 1/Da) * (Da_c/Da)**2
    else:
        # print("Inside complete delivery")
        CD_regime = "Inside complete delivery"
        V_req_analytical = V_min * (1 + 1 / Da)

    # results
    mol_capt = b * S                          # captured molecules (array) [mol]

    mol_c = c * V_b
    mol_cs = c_s * V_s
    mol_bulk = mol_c + mol_cs
    mol_theory = c_in * V

    # end values
    mol_bulk_end = c[-1] * V
    mol_bulk_conc_end = c[-1]
    bulk_perc = 100 * (mol_bulk_end / mol_injected[-1])      # percentage captured

    # performance
    loss_perc = 100 * (mol_out[-1] / mol_injected[-1])     # percentage lost at end
    capt_perc = 100 * (mol_capt[-1] / mol_injected[-1])  # percentage captured

    mol_eq = b_eq * S
    eq_target = 0.95
    capt_target = 0.95

    # find indices where targets are satisfied
    indices_eq = np.where(mol_capt >= eq_target * mol_eq)[0]
    indices_capt = np.where(mol_capt >= capt_target * mol_injected[-1])[0]

    # Capture and equilibrium times
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

    # Compute relative mass balance error (abs(x-y) / y)
    mol_bound = b * S  # bound molecules [mol]
    mol_total = mol_bound + mol_c + mol_cs + mol_out  # molecules in system [mol]

    mass_error = np.zeros_like(t)
    nonzero_mask = mol_injected > 0     # avoid division by zero
    mass_error[nonzero_mask] = (abs(mol_total[nonzero_mask] - mol_injected[nonzero_mask])) / mol_injected[nonzero_mask]
    mass_error[~nonzero_mask] = 0       # set error to 0 when no injection
    error_last = mass_error[-1]
    error_max = max(mass_error)

    # Obtain required volume for eq.
    if time_eq < t_pulse_hat * tau:
        V_eq = time_eq * Q_in
    else:
        V_eq = math.nan

    if print_results == True:
        print("Simulation of system")
        print(f"Flow rate [uL/min]:, {Q_in*1e9*60:.2f}")
        print(f"Sensor volume [uL]:, {V*1e9}")
        print("Max abs conservation residual [mol]:", np.max(np.abs(residual)))
        print("")

        print("System charactersitics:")
        print(f"Channel Peclet (Pe_H): {Pe_H:.2f}")
        print(f"Sensor Peclet (Pe_s): {Pe_s:.2f}")
        print(f"Lambda: {Lambda:.2f}")
        print(f"Sherwood number: {F:.2f}")
        print(f"Mass transport rate (k_m): {k_m:.3e}")
        print(f"Damkohler number (Da): {Da:.2e}")
        print(f"Delta [um]: {delta*1e6:.2e}")
        print(f"Delivery regime: " + CD_regime)
        print(f"Epsilon: {epsilon:.2e}")
        print("")

        print("Analtyical results:")
        print(f"Binding timescale (t_R): {t_R:.2e}")
        print(f"Residency time (tau): {tau:.2e}")
        print(f"Equilibration time (analytical): {t_eq_analytical:.2e}")
        print(f"V min [uL]: {V_min * 1e9:.2e}")
        print(f"V req [uL]: {V_req_analytical * 1e9:.2e}")

        print("")
        print("System performance (at end):")
        print(f"Lost molecules [mol] : {mol_out[-1]:.3e}")
        print(f"Lost molecules [%] : {loss_perc:.2f}")
        print(f"Captured molecules [mol] : {mol_capt[-1]:.3e}")
        print(f"Captured molecules [%] : {capt_perc:.2f}")
        print(f"Bulk molecules remaining [mol] : {mol_bulk_end:.3e}")
        print(f"Bulk molecules [%] : {bulk_perc:.2f}")
        print(f"Bulk concentration [mol/L] : {mol_bulk_conc_end:.3e}")

        print("")
        print("Equilibrium:")
        print(f"Bound equilibrium [mol] : {mol_eq:.3e}")
        print(f"Bound equilibrium [mol/m2] : {mol_eq/S:.3e}")
        print("Reached equilbirium: ", reached_eq)
        print(f"Time to equilibrium [s] : {time_eq:.2f}")
        print(f"Volume required [uL] : {V_eq*1e9:.2f}")

        print("")

    return {
        "t": t,
        "b": b,
        "db_dt": db_dt,
        "S": S,
        "L_s": L_s,
        "b_last": b_last,
        "b_hat": b_hat,
        "W_c": W_c,
        "c": c,
        "k_on": k_on,
        "k_off": k_off,
        "c_hat": c_hat,
        "c_s_hat": c_s_hat,
        "b_m": b_m,
        "c_in": c_in,
        "c_eff": c_eff,
        "b_eq": b_eq,
        "c_s": c_s,
        "Pe_H": Pe_H,
        "Pe_s": Pe_s,
        "Q_in": Q_in,
        "Lambda": Lambda,
        "F": F,
        "k_m": k_m,
        "Da": Da,
        "mol_injected": mol_injected,
        "mol_out": mol_out,
        "mol_eq": mol_eq,
        "reached_eq": reached_eq,
        "time_eq": time_eq,
        "time_capt": time_capt,
        "mol_capt": mol_capt,   # array of captured moles
        "loss_perc": loss_perc,
        "capt_perc": capt_perc, # one value
        "bulk_perc": bulk_perc,
        "mol_bulk": mol_bulk,
        "mol_bulk_end": mol_bulk_end,
        "mol_bulk_conc": mol_bulk_conc_end,
        "t_pulse_hat": t_pulse_hat,
        "tau": tau,
        "V": V,
        "V_eq": V_eq,
        "mass_error": mass_error,
        "error_last": error_last,
        "error_max": error_max
    }


