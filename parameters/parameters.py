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
    W_c= 1e-3,    # channel width [m]
    L_c= 1e-3,   # channel length [m]
    H_c= .2e-3,    # channel height [m]

    # molecular parameters
    D= 5e-10,      # diffusion coefficient [m^2/s]

    # sensor parameters
    k_on= 1e3,    # association constant [mol m^-3 s^-1]
    k_off= 1e-4,  # disassociation constant [1/s]
    b_m= 2e-5,   # max binding density [mol/m^2]
    L_s= 1e-3,    # sensor length [m]
    W_s= 0.1e-3,    # sensor width [m]

    # input parameters
    c_0= 0,       # initial bulk concentration [mol/m^3]
    c_in= 1e-4,   # input concentration [mol/m^3]
    V_in= 50e-9,      # volume input [m^3]   (1uL = 1e-9 m^3)
    Q_in= 2e-11,     # volume flow rate [m^3/s]
    flow_off = True,  # determines if flow turns off after injection
    char_length = 'H_2' # characteristic length to calculate k_m [H, H_2]

)

