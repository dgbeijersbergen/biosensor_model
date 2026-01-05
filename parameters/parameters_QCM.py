from dataclasses import dataclass

@dataclass
class ModelParams:    # physical parameters of model
    W_c: float    # channel width [m]
    L_c: float   # channel length [m]
    H_c: float    # channel height [m]


    # molecular parameters
    D: float      # diffusion coefficient [m^2/s]

    # sensor parameters
    k_on: float    # association constant [mol m^-3 s^-1]
    k_off: float  # disassociation constant [1/s]
    b_m: float   # max binding density [mol/m^2]
    L_s: float  # sensor length [m]
    W_s: float  # sensor width [m]

    # input parameters
    c_0: float       # initial bulk concentration [mol/m^3]
    c_in: float   # input concentration [mol/m^3]
    V_in: float      # volume input [m^3]
    Q_in: float     # volume flow rate [m^3/s]
    flow_off: bool
    char_length: str

# define model parameters
params = ModelParams(
    # physical parameters of model
    W_c = 0.9571e-2,    # channel width [m]
    L_c = 0.9571e-2,   # channel length [m]
    H_c = 5.4579e-4,    # channel height [m]

    # molecular parameters
    D = 0.9378e-9,      # diffusion coefficient EpCAM [m^2/s]
    #D = 6.54e-10,      # diffusion coefficient MCH [m^2/s]
    #D = 0.74e-10,      # diffusion coefficient ssDNA 52nt [m^2/s]
    #D = 8.15e-11,      # diffusion coefficient ssDNA 58nt [m^2/s]

    # sensor parameters
    k_on = 4.5e4,        # association constant [mol m^-3 s^-1]
    k_off = 7.03e-4,   # disassociation constant [1/s]
    b_m = 5.46e-8,          # max binding density [mol/m^2]
    L_s = 0.9571e-2,    # sensor length [m]     # circular: 0.9571e-2,    # sensor length [m]
    W_s = 0.9571e-2,    # sensor width [m]

    # input parameters
    c_0 = 0,        # initial bulk concentration [mol/m^3]
    #c_in = 100e-6,      # input concentration [mol/m^3] (1 M = 1e3 mol/m^3)
    c_in = 1e0*1e-9,      # input concentration [mol/L] (1 mol/m^3 = 1 mM)
    V_in = 50*1e-9,      # volume input [m^3]   (1uL = 1e-9 m^3)
    Q_in = 40,     # volume flow rate [uL/min]
    flow_off = True,  # determines if flow turns off after injection
    char_length = 'H' # characteristic length to calculate k_m [H, H_2]

)

