# Biosensor Two-Compartment Model
Two-compartment model for affinity-based biosensors that includes simplified transport and reaction processes, with a volume limitation. It can be used to provide insights into the performance of a biosensor and for rapid optimisation of biosensor parameters.
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
- Adjust settings in desired file (simulation > run_simulation_characterize.py or run_simulation_optimisation.py or run_simulation_batch.py)
- Make sure correct parameters are imported (e.g., from biosensor.parameters.parameters import params)
- Run the desired file

## Background
