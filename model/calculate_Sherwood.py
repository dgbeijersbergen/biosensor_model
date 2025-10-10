## function to calculate k_m for given input of Pe_s and lambda
import numpy as np

# formulas for limits
# for Pe_H << 1
def F_retained(Pe_H):
    return Pe_H

# for Pe_H >> 1 and Pe_s << 1
def F_Ackerberg(Pe_s):
    return np.pi / (np.log(4 / np.sqrt(Pe_s)) + 1.06)

# for Pe_H >> 1 and Pe_s >> 1
def F_Newman(Pe_s):
    return (0.81 * Pe_s ** (1/3)) + (0.71 * Pe_s**(-1/6)) - (0.2 * Pe_s**(-1/3))

# for mixed cases

# blending formula with hyperbolic tangent (transposed to t[0,1] with output [0,1])
def smoothstep(kappa, sharpness = 4):
    kappa = np.clip(kappa, 0, 1)    # clip values for t < 0 and t > 1
    return 0.5 * (1 + np.tanh(sharpness * (kappa - 0.5)))

# blend
def blend_functions(F_low,F_high, omega):
    return (F_low ** (1 - omega)) * (F_high ** omega)

def F_combine(Pe_H,lambda_ratio,sharpness=4,Pe_H_cutoff=1e-2,Pe_s_low=1e-2,Pe_s_high=1e2):
    Pe_s = 6 * (lambda_ratio ** 2) * Pe_H

    # for Pe_H << 1
    if Pe_H <= Pe_H_cutoff:
        return F_retained(Pe_H)

    # calculate limits for Pe_H > 1
    F_small = F_Ackerberg(Pe_s)
    F_large = F_Newman(Pe_s)

    # for Pe_s << 1
    if Pe_s <= Pe_s_low:
        return F_small

    # for Pe_s >> 1
    if Pe_s >= Pe_s_high:
        return F_large

    # for mixed cases, blend smoothly
    kappa = (np.log10(Pe_s) - np.log10(Pe_s_low)) / (np.log10(Pe_s_high) - np.log10(Pe_s_low))
    omega = smoothstep(kappa,sharpness)

    return blend_functions(F_small,F_large,omega)

def compute_k_m(Q_in,params):
    D = params.D
    W_c = params.W_c
    L_s = params.L_s
    H_c = params.H_c

    # system charactersitics
    Pe_H = Q_in / (D * W_c)
    Lambda = L_s / H_c      # ratio of sensor length to channel height

    # obtain k_m from calculate_Sherwood.py, with F minimum of 1 (pure diffusion)
    F = max(F_combine(Pe_H,Lambda),1)

    # F cannot exceed Pe_H (limit) - don't enforce for the model, breaks down at low Pe_H
    #if F > Pe_H:
    #    F = Pe_H

    # calculate mass transport rate
    if params.char_length == 'H':
        k_m = F * (D / H_c)         # characteristic length H
    elif params.char_length == 'H_2':
        k_m = F * (D / (H_c / 2))
    else:
        raise ValueError("Unkown characteristic length (char_length)")

    return k_m