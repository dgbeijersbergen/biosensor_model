## Calculate diffusion coefficients for different hydrodynamic radii

import math
import numpy as np

# define variables
k_B = 1.380649e-23  # Boltzmann constant [m^2 kg s^-2 K-1]
T_celcius = 25      # Temperature [C]
nu = 0.89e-3        # Dynamic viscosity [Pa s]

# define R_H for MCH, EpCAM
R_H_molecules = np.array([3.75e-10, 2.879e-9])

# ssDNA
p = 7.2e-9          # Persistence length [m]
N = np.array([52,58])              # Number of base pairs [ ]
b = 340e-12         # Length of one base pair [m]

L_D = N*b           # Length of DNA [m]

ratio = p / L_D     # Ratio persistence length to DNA length
term = (1 - (3 * ratio) + (6 * ratio**2) - (6 * ratio**3) * (1 - np.exp(-L_D/p)))
R_H_DNA = (2/3) * np.sqrt((p * L_D / 3) * term)

R_H = np.append(R_H_molecules,R_H_DNA)


T = T_celcius + 273.15  # Temperature [K]

D = (k_B * T) / (6 * math.pi * nu * R_H)

print("Diffusion coefficients:")
print("[MCH, EpCAM, ssDNA 52, ssDNA 58]")
print(D)