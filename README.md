# Biosensor Two-Compartment Model
Two-compartment model for affinity-based biosensors that includes simplified transport and reaction processes, with a volume limitation.
Manuscript available on: https://arxiv.org/abs/2512.21997

## Author: 
Daan Beijersbergen (dgbeijersbergen@gmail.com)

## Updates:
-  

## Known issues:
-

## Instruction to run:
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
> model/ 
> biosensor_model.py              # defines the ODE system
> calculate_Sherwood.py           # obtains Sherwood number and mass transfer coefficient 
> simulate_ODE.py                 # runs simulations and setup for solver 

parameters/
> parameters.py           
> parameters_madaboosi2015 
> parameters_QCM.py         
- plots/
- - plot_results_batch.py         
- - plot_results_characterize.py  
- - plot_results_optimisation.py  
- - plot_results_single.py        
- - plot_transport_flux.py        
- simulation/
- - results/
- - - batch/                        
- - - single/                       
- run_simulation_batch.py           
- run_simulation_characterize.py    
- run_simulation_optimisation.py    
- run_simulation_single.py          
