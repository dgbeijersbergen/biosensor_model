import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'font.family': 'Verdana',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12
})

def plot_time_series(df, t_pulse=None, save_path=None):
    # unpack result variables
    t = df["t"]
    b = df["b"]
    c = df["c"]
    mol_injected = df["mol_injected"]
    mol_out = df["mol_out"]
    S = df["S"]
    V = df["V"]

    # Compute moles in each compartment
    mol_bound = b * S                           # bound molecules [mol]
    mol_bulk = c * V                            # free bulk molecules [mol]
    mol_total = mol_bound + mol_bulk + mol_out  # molecules in system [mol]

    # Compute relative mass balance error
    mass_error = np.zeros_like(t)
    nonzero_mask = mol_injected > 0     # avoid division by zero
    mass_error[nonzero_mask] = (mol_injected[nonzero_mask] - mol_total[nonzero_mask]) / mol_injected[nonzero_mask]
    mass_error[~nonzero_mask] = 0       # set error to 0 when no injection

    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=100)

    # Left axis: molecules
    ax1.plot(t, mol_injected, label='Injected', color="xkcd:black", linewidth=2.5)
    ax1.plot(t, mol_bound, label='Bound', color="xkcd:grass green",linewidth=2.5)
    ax1.plot(t, mol_bulk, label='Bulk', color="xkcd:cerulean",linewidth=2.5)
    ax1.plot(t, mol_out, label='Lost', color="xkcd:red",linewidth=2.5)
    #ax1.plot(t, c_s, label='interface c')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Molecules [mol]')
    ax1.set_title("Biosensor kinetics")
    ax1.grid(True, linestyle='--', linewidth=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Annotations
    if t_pulse:
        plt.axvline(t_pulse, color='k', ls='--', label='Injection end',linewidth=2)


    # Combine legends
    ax1.legend(frameon=True, loc='best')

    plt.tight_layout()

    # Save figure if specified
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_dimensionless(df, t_pulse=None, save_path=None):
    # Unpack result variables
    t = df["t"] / df["tau"]
    S = df["S"]
    mol_injected = df["mol_injected"]
    mol_out = df["mol_out"]
    b = df["b"]
    b_hat = df["b_hat"]
    b_eq = df["b_eq"].values[0]
    b_m = df["b_m"].values[0]
    c_hat = df["c_hat"]
    t_pulse_hat = t_pulse / df["tau"].values[0]

    # Compute moles in each compartment
    mol_bound = b * S

    # Plot
    plt.plot(t, b_hat, label=r'Bound $\hat{b}$', color="xkcd:grass green",linewidth=2.5)
    plt.axhline(b_eq / b_m, ls='--', label=r'Equilibrium $\hat{b_{eq}}$', color="xkcd:grass green",linewidth=2)


    plt.plot(t, c_hat, label=r'Bulk $\hat{c}$', color="xkcd:cerulean",linewidth=2.5)

    # Annotations
    if t_pulse:
        plt.axvline(t_pulse_hat, color='k', ls='--', label='Injection end',linewidth=2)


    plt.xlabel(r'Dimensionless time $\hat{t}$ [ ]')
    plt.ylabel(r'Value [ ]')
    plt.legend(frameon=True, loc='best')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.ylim(0, 1)

    plt.title('Dimensionless biosensor kinetics')

    plt.tight_layout()

    # Save figure if specified
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_system_performance(df,t_pulse=None, save_path=None):
    # Unpack result variables
    t = df["t"]
    c = df["c"]
    V = df["V"]
    S = df["S"]
    mol_injected = df["mol_injected"]
    mol_out = df["mol_out"]
    b = df["b"]

    # Compute moles in each compartment
    mol_bound = b * S
    mol_bulk = c * V  # free bulk molecules [mol]

    with np.errstate(invalid='ignore', divide='ignore'):    # ignore div by zero errors
        capt_ratio = 100 * mol_bound/mol_injected
        loss_ratio = 100 * mol_out/mol_injected
        bulk_ratio = 100 * mol_bulk/mol_injected

    # Plot
    plt.plot(t, capt_ratio, label='Capture', color="xkcd:grass green",linewidth=2.5)

    plt.plot(t, bulk_ratio, label='Bulk', color="xkcd:cerulean",linewidth=2.5)
    plt.plot(t, loss_ratio, label='Loss', color="xkcd:red", linewidth=2.5)
    if t_pulse:
        plt.axvline(t_pulse, color='k', ls='--', label='Injection end',linewidth=2)

    plt.ylabel('Part of injected molecules [%]')
    plt.xlabel('Time [s]')
    plt.legend(frameon=True, loc='best')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.title('System performance')
    plt.tight_layout()

    # Save figure if specified
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_mass_balance_error(df,t_pulse=None, save_path=None):
    fig = plt.subplots(figsize=(8, 5), dpi=100)
    # unpack result variables
    t = df["t"]
    b = df["b"]
    c = df["c"]
    mol_injected = df["mol_injected"]
    mol_out = df["mol_out"]
    S = df["S"]
    V = df["V"]
    mass_error = df["mass_error"]

    # Compute moles in each compartment
    mol_bound = b * S                           # bound molecules [mol]
    mol_bulk = c * V                            # free bulk molecules [mol]

    plt.plot(t, 100 * abs(mass_error), '-', label='Error', color="xkcd:black", linewidth=2)
    plt.ylabel('Mass balance error [%]')
    plt.xlabel('Time [s]')
    plt.title('Mass Balance Error Over Time')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    if t_pulse:
        plt.axvline(t_pulse, color='k', ls='--', label='Injection end')

    plt.legend(frameon=True, loc='best')
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Mass balance error plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_Damkohler_time(df, save_path=None):
    t = df["t"].values
    Da_t = df["Da_t"].values

    plt.plot(t, Da_t)
    plt.ylabel('Damkohler number')
    plt.xlabel('Time')
    plt.grid(True)

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_cs_time(df, t_pulse, save_path=None):
    t = df["t"].values
    c_s = df["c_s"].values - df["c"]

    plt.plot(t, c_s)
    plt.ylabel('Delta c [M]')
    plt.xlabel('Time')
    plt.grid(True)

    plt.tight_layout()

    if t_pulse:
        plt.axvline(t_pulse, color='k', ls='--', label='Injection end')

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()