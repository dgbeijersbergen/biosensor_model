from biosensor.model.calculate_Sherwood import *


#def compute_delta(Q_in, params):
#    """Compute depletion layer height delta from Squires' scaling.
#
#    Uses the same Pe_s regimes as the Sherwood number calculation:
#        Pe_H << 1:              delta = H_c  (complete delivery)
#        Pe_H >> 1, Pe_s >> 1:   delta = L_s * Pe_s^(-1/3)  (thin depletion zone)
#        Pe_H >> 1, Pe_s << 1:   delta = L_s * Pe_s^(-1/2)  (thick relative to sensor)
#
#    Delta is capped at H_c (cannot exceed channel height).
#
#    Parameters
#    ----------
#    Q_in : float
#        Volumetric flow rate [m^3/s]
#    params : ModelParams
#        Biosensor parameters
#
#    Returns
#    -------
#    delta : float
#        Depletion layer height [m]
#    """
#    if Q_in <= 0:
#        return params.H_c  # no flow -> full channel
#
#    D = params.D
#    W_c = params.W_c
#    H_c = params.H_c
#    L_s = params.L_s
#
#    Pe_H = Q_in / (D * W_c)
#    Lambda = L_s / H_c
#    Pe_s = 6 * (Lambda ** 2) * Pe_H
#
#    # thresholds (same as in calculate_Sherwood.py)
#    Pe_H_cutoff = 1e-2
#    Pe_s_low = 1e-2
#    Pe_s_high = 1e2
#
#    # complete delivery regime
#    if Pe_H <= Pe_H_cutoff:
#        return H_c
#
#    # thin depletion zone (Pe_s >> 1)
#    if Pe_s >= Pe_s_high:
#        delta = L_s * Pe_s ** (-1.0 / 3.0)
#        return min(delta, H_c)
#
#    # thick depletion zone relative to sensor (Pe_s << 1)
#    if Pe_s <= Pe_s_low:
#        delta = L_s * Pe_s ** (-0.5)
#        return min(delta, H_c)
#
#    # intermediate: blend between the two scalings (log-space)
#    delta_low = L_s * Pe_s ** (-0.5)
#    delta_high = L_s * Pe_s ** (-1.0 / 3.0)
#
#    kappa = (np.log10(Pe_s) - np.log10(Pe_s_low)) / (np.log10(Pe_s_high) - np.log10(Pe_s_low))
#    omega = 0.5 * (1 + np.tanh(4 * (kappa - 0.5)))
#
#    # log-space interpolation
#    log_delta = (1 - omega) * np.log10(delta_low) + omega * np.log10(delta_high)
#    delta = 10 ** log_delta
#
#    return min(delta, H_c)


def compute_delta(Q_in, params, sharpness):
    if Q_in <= 0:
        return params.H_c

    k_m, F = compute_k_m(Q_in, params, sharpness)

    Pe_H = Q_in / (params.D * params.W_c)

    if F >= Pe_H:
        return params.H_c  # complete delivery

    delta = params.L_s / F
    #delta = params.H_c / F

    return min(delta, params.H_c)


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
    # k_m, F = compute_k_m(Q_eff, params)

    if Q_eff > 0:
        k_m = params._k_m_cached
        F = params._F_cached
    else:
        k_m = 0.0
        F = 0.0

    # --- depletion layer height ---
    # delta = compute_delta(Q_eff, params)
    delta = params.delta if (injecting or not flow_off) else params.H_c

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
        alpha = J_D * (H_c / delta)

        if delta == H_c:
           beta = 0
           dc_hat_dt = 0
           dNouthat1_dt = 0
        else:
            beta = H_c / (H_c - delta)
            dc_hat_dt = H_c / (H_c - delta) * (1 - J_D) * (H - c_hat)
            dNouthat1_dt = H_c / (H_c - delta) * (1 - J_D) * c_hat

        # dNouthat1_dt = c_hat
        # dNouthat1_dt = (1 - J_D) * c_hat
        dNouthat2_dt = alpha * c_s_hat


        # old
        #dcs_hat_dt = J_D - (1 / H_c) * J_R - J_s_out
        # dcs_hat_dt = J_D * (H_c / delta) - (1.0 / delta) * J_R - J_s_out
        # dc_hat_dt = H - J_D - J_out

        # new

        # dcs_hat_dt = alpha * (1 - c_s_hat) - (1.0 / delta) * J_R
        dcs_hat_dt = (tau * k_m * H_c) / (delta * L_s) * (1 - c_s_hat) - (1.0 / delta) * J_R

        # dc_hat_dt = H - J_D - J_out
        # dc_hat_dt = H - J_D * (1 - c_s_hat) - J_out
        # dc_hat_dt = (1 - J_D) * (H - c_hat)
        #dc_hat_dt = beta * (1 - J_D) * (H - c_hat)


        # --- outflow tracking ---
        # mol_out_bulk = N_out1 * c_in * V_b


        # mol_out_interface = N_out2 * c_in * V_s
        dNouthat2_dt = (tau * k_m * H_c) / (delta * L_s) * c_s_hat

    else:
        # flow off, post injection

        # dcs_hat_dt = -gamma * db_hat_dt
        dcs_hat_dt = -(gamma * H_c / delta) * db_hat_dt

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