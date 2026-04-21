# Biosensor Two-Compartment Model

A lightweight Python model for simulating affinity-based biosensor kinetics, integrating simplified mass transport and Langmuir binding. This model contains mass conservation and takes into account the sample volume.

Companion code to:
> **D. Beijersbergen & J. Charmet (2026)** — *Sample volume as a key design parameter in affinity-based biosensors* (in revision, preprint: https://arxiv.org/abs/2512.21997)

---

## Overview

Accurately modeling biosensor performance normally requires computationally intensive finite-element simulations (e.g. COMSOL). This model offers a tractable alternative: a two-compartment ODE system that splits the biosensor volume into an interacting interface layer and a non-interacting bulk, governed by 13 physically meaningful parameters.

**Key capabilities:**
- Accurately simulates affinity-based biosensor binding kinetics (RMSE < 5% vs. COMSOL)
- Predicts equilibration time and required sample volume (mean relative error ~7% vs. COMSOL)
- Runs >100× faster than finite-element simulations (runtime per simulation < 1.0 seconds)
- Provides rapid insights into biosensor optimisation through parametric sweeps
- Computes intermediate parameters (Damköhler number, Péclet numbers, Critical flow rate)

---


## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-org>/biosensor-model.git
cd biosensor-model
pip install numpy scipy pandas sympy
```

Python 3.10+ recommended.

---

## Quick Start

1. **Set your parameters** in `biosensor/parameters/parameters.py` (or create your own parameters file):

```python
params = ModelParams(
    W_c = 0.9571e-2,   # channel width [m]
    H_c = 5.4579e-4,   # channel height [m]
    D   = 0.9378e-9,   # diffusion coefficient [m²/s]
    k_on  = 4.5e4,     # association rate [M⁻¹s⁻¹]
    k_off = 7.03e-4,   # dissociation rate [s⁻¹]
    c_in  = 1e-9,      # input concentration [mol/L]
    V_in  = 50e-9,     # injection volume [m³]
    Q_in  = 40,        # flow rate [µL/min]
    flow_off = True,   # stopped flow after injection
    ...
)
```

2. **Adjust simulations to import parameters**
```python
from biosensor.parameters.parameters import params
```

3. **Run a simulation:**

```bash
python run_simulation_single.py
```

```bash
python run_simulation_characterize.py
```

```bash
python run_simulation_optimisation.py
```

This prints system characteristics, performance metrics (captured/lost molecule fractions, equilibration time, required volume), and optionally exports results and plots.

---

## Repository Structure

```
biosensor/
├── model/
│   ├── biosensor_model.py       # ODE system (dimensionless)
│   ├── calculate_Sherwood.py    # Sherwood interpolation & k_m
│   └── simulate_ODE.py          # Solver setup, post-processing, mass balance
│
├── parameters/
│   ├── parameters.py            # Default parameters
│   ├── parameters_box1.py       
│   └── parameters_box2.py
│   └── parameters_figure5.py
│
├── plots/
│   ├── plot_results_single.py
│   ├── plot_results.py
│
└── simulation/
    ├── run_simulation_single.py       # Single-condition simulation
    ├── run_simulation_batch.py        # Batch over parameter sets
    ├── run_simulation_characterize.py # Sweep V_in × Q_in → capture fraction
    └── run_simulation_optimisation.py # Sweep Q_in → V_req, t_eq

---

## Parameters

All parameters are defined in a `ModelParams` dataclass. Units are SI throughout (the run script handles conversion from practical units).

| Parameter | Symbol | Unit | Description |
|-----------|--------|------|-------------|
| `W_c`, `L_c`, `H_c` | — | m | Channel width, length, height |
| `D` | D | m²/s | Diffusion coefficient |
| `k_on` | k_on | M⁻¹s⁻¹ | Association rate constant |
| `k_off` | k_off | s⁻¹ | Dissociation rate constant |
| `b_m` | b_m | mol/m² | Maximum surface binding density |
| `L_s`, `W_s` | — | m | Sensor length and width |
| `c_in` | c_in | mol/L | Input analyte concentration |
| `V_in` | V_in | m³ | Injection volume |
| `Q_in` | Q_in | µL/min | Volumetric flow rate |
| `flow_off` | — | bool | Stopped flow after injection |
| `char_length` | — | `'H'` | Characteristic length for k_m (`'H'` = channel height) |

---

## Outputs

`simulate()` returns a dictionary with all time-resolved and scalar results, including:

- `t`, `b`, `c`, `c_s` — time, bound density, bulk and interface concentrations
- `Pe_H`, `Pe_s`, `Lambda`, `F`, `k_m`, `Da` — dimensionless system characteristics
- `mol_injected`, `mol_out`, `mol_capt` — molecule accounting
- `time_eq`, `V_eq` — equilibration time and required volume
- `mass_error` — relative mass balance error over time (useful for validation)

---

## Model Assumptions and Limitations

- **Sensor geometry:** The sensor is assumed to span the full channel width (W_s = W_c). For narrower sensors, a volume penalty scales as W_s / W_c (depending on lateral diffusion).
- **No inter-compartment diffusion:** Transport between the bulk and interface compartments is not modelled — valid for continuous flow, but may underestimate diffusive transport in stopped-flow conditions.
- **Quasi-steady transport:** The model assumes a transport steady state forms before significant binding occurs. Violations (channel residence time τ > binding timescale t_R) can cause over- or underestimation.
- **Depletion layer switch:** The condition for a valid depletion layer (ε = N_sensor / N_layer) is applied as a binary switch. In practice, this transition is gradual.

---

## Adapting the Model

The model is designed to be straightforward to extend:

- **New parameter sets:** Copy `parameters.py`, fill in your system's values, and point `run_simulation_single.py` (or other simulation files) to the new parameter file.
- **Custom analytes:** Set `D` (diffusion coefficient), `k_on`, `k_off`, and `b_m` for your target molecule. An approximation of the diffusion coefficient is possible through Einstein approximation.
- **Batch sweeps:** Use `simulate_ODE.simulate()` directly in a loop over parameter ranges and collect results into a DataFrame. The `plot_results_batch.py` module provides batch plotting utilities.
- **Stopped vs. continuous flow:** Toggle `flow_off = True/False`. In stopped-flow mode, the interface ODE reduces to binding/unbinding only.

---

## Citation

If you use this model, please cite:

```
Beijersbergen, D. & Charmet, J. (2026). Sample volume as a key design parameter 
in affinity-based biosensors. [Biosensors and Bioelectronics: X]
```

---

## License

-
