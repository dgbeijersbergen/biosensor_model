import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from biosensor.model.calculate_Sherwood import F_combine, compute_k_m

# Ordinary differential equation (ODE) for two-compartment model, dimensionless
def ode_binding_hat(t_hat,y,params):
    b_hat, c_hat, c_s_hat, N_out_hat1, N_out_hat2 = y

    # unpack params
    W_c, L_c, H_c = params.W_c, params.L_c, params.H_c
    D = params.D
    k_on, k_off, b_m, L_s, W_s = params.k_on, params.k_off, params.b_m, params.L_s, params.W_s
    c_0, c_in, V_in, Q_in, flow_off = params.c_0, params.c_in, params.V_in, params.Q_in, params.flow_off

    # system size
    S = L_s * W_s           # sensor area [m^2]
    V = W_c * L_c * H_c     # channel volume [m^3]

    # compute dimensionless values
    tau = V / Q_in                      # residence time [s]
    gamma = (S * b_m) / (V * c_in)      # ratio of surface capacity to bulk content [ ]
    t_pulse_hat = V_in / V              # input time dimensionless

    # pulse function to stop flow
    if t_hat < t_pulse_hat:
        H = 1.0
        Q_eff = Q_in
    else:
        H = 0.0
        Q_eff = 0

    # obtain k_m value
    k_m = compute_k_m(Q_eff,params)

    # non-dimensional interface concentration [ ] (eq.1)
    # c_s_numerator = (k_m * c_in) + (k_off * b_m * b_hat)
    # c_s_denom = k_m + k_on * (b_m - (b_m*b_hat))
    # c_s_hat = (1 / c_in) * (c_s_numerator / c_s_denom)

    # interface concentration
    dNouthat2_dt = (Q_in / V) * c_s_hat
    #dcs_hat_dt = tau * ( (k_m / L_s) - ((1 / H_c) * (k_on * c_s_hat * b_m * (1 - b_hat) - (k_off * b_m * b_hat) / c_in) ) - (Q_in / V) * c_s_hat)   # new
    dcs_hat_dt = tau * ( (k_m * (1 - c_in * c_s_hat) / L_s) - ((1 / H_c) * (k_on * c_s_hat * b_m * (1 - b_hat) - (k_off * b_m * b_hat) / c_in) ) - (Q_in / V) * c_s_hat)   # new

    # limit c_s_hat to 1 (physical limit) (don't enforce)
    #c_s_hat = min(c_s_hat, 1.0)

    # langmuir kinetics (eq. 2)
    db_hat_dt = tau * ((k_on * c_in * c_s_hat * (1 - b_hat)) - (k_off * b_hat))
    #dbhat_dt = tau * ((k_on * c_in * c_s_hat * (1 - b_hat)))

    # compute Langmuir equilibrium fraction for current c_s_hat
    b_eq_hat = (k_on * c_in) / (k_on * c_in + k_off)

    # limit b_hat to 1 (physical limit)
    if (b_hat >= 1.0 or b_hat > b_eq_hat) and db_hat_dt > 0:
        dbhat_dt = 0.0

    # conservation of mass (eq. 3)
    gamma = (S * b_m) / (V * c_in) # ratio of bulk bound molecules to bulk molecules



    # during injection or flow
    if t_hat < t_pulse_hat or flow_off == False:
        dNouthat1_dt = c_hat
        #dchat_dt = H - gamma * dbhat_dt - dNouthat_dt  # old
        dc_hat_dt = H - tau * (k_m / L_s) - dNouthat1_dt  # new

    # after injection and stopped flow
    elif t_hat >= t_pulse_hat and flow_off == True:
        dchat_dt = - gamma * dbhat_dt
        dNouthat1_dt = 0

    return [db_hat_dt, dc_hat_dt, dcs_hat_dt, dNouthat1_dt, dNouthat2_dt]


