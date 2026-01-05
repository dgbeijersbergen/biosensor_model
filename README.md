# Biosensor Two-Compartment Model
Two-compartment model for affinity-based biosensors that includes simplified transport and reaction processes, with a volume limitation.
Manuscript available on: https://arxiv.org/abs/2512.21997

## Author: 
Daan Beijersbergen (dgbeijersbergen@gmail.com)

## Updates:
-  

## Known issues:
-

Instructions to run: (single simulation)
- Change biosensor parameters in parameters > parameters.py.
- Adjust settings in simulation > run_simulation_single.py (i.e., max time, plot results (y/n), save results (y/n))
- Make sure correct parameters are imported (e.g., from biosensor.parameters.parameters import params)
- Run the file run_simulation_single.py

Instructions to run: (batch simulation)
- Change biosensor parameters in parameters > parameters.py.
- Adjust settings in simulation > run_simulation_characterize.py or run_simulation_optimisation.py or run_simulation_batch.py
- Make sure correct parameters are imported (e.g., from biosensor.parameters.parameters import params)
- Run the file run_simulation_single.py

## Background


## Repository Structure

biosensor/
- model/
- - biosensor_model.py              # defines the ODE system
- - calculate_Sherwood.py           # obtains Sherwood number and mass transfer coefficient
- - simulate_ODE.py                 # runs simulations and setup for solver
- parameters/
- - parameters.py                   # contains generic parmater set 
- - parameters_madaboosi2015        # parameters for box 2
- - parameters_QCM.py               # parameters for box 1
- plots/
- - plot_results_batch.py           # contains functions for plotting
- - plot_results_characterize.py    # contains functions for plotting
- - plot_results_optimisation.py    # contains functions for plotting
- - plot_results_single.py          # contains functions for plotting
- - plot_transport_flux.py          # plots sherwood number for a range of flow rates 
- simulation/
- - results/
- - - batch/                        # batch simulations end up here
- - - single/                       # single simulations end up here
- run_simulation_batch.py           # run a batch simulation
- run_simulation_characterize.py    # run a range of simulations in flow rates and volumes
- run_simulation_optimisation.py    # run a range of simulations in flow rates
- run_simulation_single.py          # run a single simulation
