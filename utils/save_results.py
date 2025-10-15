import os
import numpy as np
import pandas as pd
from datetime import datetime

def save_simulation_results(results, params, run_type="single", file_format="csv"):
    # Create folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", run_type, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Build filename from key parameters
    if hasattr(params, "char_length"):
        char_len = params.char_length
    else:
        char_len = "N/A"

    filename_base = f"sim_{char_len}_{1e9*getattr(params,'V_in',0):.0f}uL_{1e3*getattr(params,'c_in',0):.0f}uM_{getattr(params,'Q_in',0):.0e}m3_s"

    if file_format == "csv":
        if isinstance(results, dict):
            data = np.column_stack((results["t"], results["mol_capt"]))
            filepath = os.path.join(output_dir, filename_base + ".csv")
            np.savetxt(filepath, data, delimiter=",")
        elif isinstance(results, pd.DataFrame):
            filepath = os.path.join(output_dir, filename_base + ".csv")
            results.to_csv(filepath, index=False)
    elif file_format == "npz":
        filepath = os.path.join(output_dir, filename_base + ".npz")
        if isinstance(results, dict):
            np.savez(filepath, **results)
        elif isinstance(results, pd.DataFrame):
            np.savez(filepath, **results.to_dict('list'))
    print(f"Results saved to: {filepath}")
    return filepath
