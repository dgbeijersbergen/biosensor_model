from biosensor.model.calculate_Sherwood import *

def ode_binding_hat(t_hat, y, params):
    b_hat, c_hat, c_s_hat, N_out_hat1, N_out_hat2 = y

    # --- parameters ---
    W_c, L_c, H_c = params.W_c, params.L_c, params.H_c
    k_on, k_off = params.k_on, params.k_off
    b_m, L_s, W_s = params.b_m, params.L_s, params.W_s
    c_in, V_in, Q_in = params.c_in, params.V_in, params.Q_in
    flow_off = params.flow_off

    # --- geometry ---
    S = L_s * W_s
    V = W_c * L_c * H_c
    params.V = V

    # --- dimensionless groups ---
    tau = V / Q_in
    gamma = (S * b_m) / (V * c_in)
    t_pulse_hat = V_in / V

    # --- injection switch ---
    injecting = (t_hat < t_pulse_hat)
    if injecting == True:
        H = 1.0
        Q_eff = Q_in
    else:
        H = 0.0
        Q_eff = 0.0

    # --- mass transfer ---
    k_m, F = compute_k_m(Q_eff, params)

    # --- binding kinetics ---
    db_hat_dt = tau * (
        k_on * c_in * c_s_hat * (1 - b_hat) - k_off * b_hat
    )

    # enforce saturation
    b_eq_hat = (k_on * c_in) / (k_on * c_in + k_off)
    if (b_hat >= 1.0 or b_hat > b_eq_hat) and db_hat_dt > 0:
        db_hat_dt = 0.0

    # --- reaction flux (dimensionless) ---
    J_R = tau * (
        k_on * c_s_hat * b_m * (1 - b_hat)
        - (k_off * b_m * b_hat) / c_in
    )

    # --- transport terms ---
    J_D = tau * (k_m / L_s)
    J_out = c_hat
    J_s_out = c_s_hat

    # ===============================
    # REGIMES
    # ===============================

    if injecting or not flow_off:
        # flow on (injection or continuous)

        dNouthat1_dt = c_hat
        dNouthat2_dt = c_s_hat

        dcs_hat_dt = J_D - (1 / H_c) * J_R - J_s_out
        dc_hat_dt = H - J_D - J_out

    else:
        # flow off, post injection

        dcs_hat_dt = -gamma * db_hat_dt
        dc_hat_dt = 0.0
        dNouthat1_dt = 0.0
        dNouthat2_dt = 0.0

    return [
        db_hat_dt,
        dc_hat_dt,
        dcs_hat_dt,
        dNouthat1_dt,
        dNouthat2_dt,
    ]