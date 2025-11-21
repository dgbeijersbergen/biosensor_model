from bdb import Breakpoint

import matplotlib
matplotlib.use("QtAgg")   # Best backend for PyCharm
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
import matplotlib.tri as tri
import math
import seaborn as sns
import pandas as pd
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import Rbf
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import os

## Batch simulation plots
# Plot capture rate for different flow rates
def plot_optimization(df,params,Q_in_vals, save_path=None):
    Q_in = df["Q_in"]
    Pe_H = df["Q_in"] / (params.D * params.W_c)
    eq_perc = 100 * df["b_last"] / df["b_eq"]
    capt_perc = df["capt_perc"]
    #time_eq = df["time_eq"]
    time_capt = df["time_capt"]
    fig, ax1 = plt.subplots(figsize=(7,6))
    Q_conversion_factor = (1 / 60) * 10 ** (-9)
    Q_in_uL = Q_in_vals / Q_conversion_factor


    # Left y-axis for capture percentage
    ax1.plot(Q_in_uL, eq_perc, 'k-', label='Capture percentage')
    ax1.set_xlabel('Flow rate [uL/min]')
    ax1.set_ylabel('Capture percentage [%]', color='k')
    ax1.tick_params(axis='y', labelcolor='k')
    ax1.grid(True)
    plt.xlim(0, max(Q_in_uL))  # set x-axis limits
    plt.ylim(0, 100)  # set y-axis limits

    # Right y-axis for capture time
    ax2 = ax1.twinx()
    ax2.plot(Q_in_uL,time_capt/60, 'r--', label='Capture time')
    ax2.set_ylabel('90% capture time [min]', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim(0,60)

    # Right y-axis for Pe_H
    #ax2 = ax1.twinx()
    #ax2.plot(Q_in_uL, Pe_H, 'r--', label='Peclet number')
    #ax2.set_ylabel('Peclet number', color='r')
    #ax2.tick_params(axis='y', labelcolor='r')
    ## ax2.set_ylim(0,10)

    # Optional: add a combined legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

# same as above but added experimental data?
def plot_optimization2(df, params, Q_in_vals, exp_data=None, save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np

    Q_in = df["Q_in"]
    Pe_H = df["Q_in"] / (params.D * params.W_c)
    capt_perc = df["capt_perc"]
    time_capt = df["time_capt"]
    fig, ax1 = plt.subplots(figsize=(7,6))

    Q_conversion_factor = (1 / 60) * 1e-9
    Q_in_uL = Q_in_vals / Q_conversion_factor

    # Left y-axis for capture percentage
    ax1.plot(Q_in_uL, capt_perc, 'k-', label='Capture percentage')
    ax1.set_xlabel('Flow rate [uL/min]')
    ax1.set_ylabel('Capture percentage [%]', color='k')
    ax1.tick_params(axis='y', labelcolor='k')
    ax1.grid(True)
    plt.xlim(0, max(Q_in_uL))
    plt.ylim(0, 100)

    # Right y-axis for capture time
    ax2 = ax1.twinx()
    ax2.plot(Q_in_uL, time_capt / 60, 'r--', label='Capture time')
    ax2.set_ylabel('90% capture time [min]', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim(0, 60)

    # Add experimental data if provided
    if exp_data is not None:
        Q_exp = 50  # uL/min
        # Capture percentage
        ax1.plot(Q_exp, exp_data['Bound (exp.)'][0]*100, 'bo', label='Exp. Bound 1')
        ax1.plot(Q_exp, exp_data['Bound (exp.)'][1]*100, 'go', label='Exp. Bound 2')
        ax1.plot(Q_exp, exp_data['Bound (exp.)'][2]*100, 'mo', label='Exp. Bound 3')

        # Optional: also add simulated values for comparison
        ax1.plot(Q_exp, exp_data['Bound (sim H/2)'][0]*100, 'b^', label='Sim H/2 Bound 1')
        ax1.plot(Q_exp, exp_data['Bound (sim H/2)'][1]*100, 'g^', label='Sim H/2 Bound 2')
        ax1.plot(Q_exp, exp_data['Bound (sim H/2)'][2]*100, 'm^', label='Sim H/2 Bound 3')

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

## --- experimental graphs ----
def plot_error(df, save_path = None):
    fig, ax = plt.subplots(figsize=(7, 6))

    sc = plt.scatter(
        df["Pe_H"],
        df["V_in"],
        c=100*df["error_max"],
        cmap="jet",
        s=50,
        edgecolor='k',
        vmin=0,
    )

    # Colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Error")

    # Axis scaling and grid
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", ls="-", linewidth=1, alpha=0.8)
    ax.grid(True, which="minor", ls="-", linewidth=1, alpha=0.2)

    # Labels and title
    ax.set_xlabel("Peclet number")
    ax.set_ylabel("Sample volume [uL]")
    ax.set_title("Mass error")

    plt.tight_layout()

    plt.show()



def plot_damkohler_batch(df, x_axis = None, save_path = None):
    fig, ax = plt.subplots(figsize=(7, 6))

    # heatmap (interpolated colors)
    #tri = ax.tricontourf(df["Pe_H"], df["Da"], df["time_eq"],
    #                     levels=10, cmap="plasma")

    # Split data
    finite = df[np.isfinite(df["time_eq"])]
    infinite = df[np.isinf(df["time_eq"])]

    # set colormap range
    if len(finite) > 0:
        vmin = np.log10(finite["time_eq"].values).min()  # or manually, e.g., 2
        vmax = np.log10(finite["time_eq"].values).max()  # corrected
    else:
        vmin = 0
        vmax = 4  # fallback value

    if x_axis == "Pe_H":
        x_values_inf = infinite["Pe_H"]
        x_values = finite["Pe_H"]
        x_label = "Peclet number [ ]"

    elif x_axis == "Q_in":
        x_values_inf = infinite["Q_in"].values * 1e9 * 60
        x_values = finite["Q_in"].values * 1e9 * 60
        x_label = "Flow rate [uL/min]"

    # infinite
    plt.scatter(
        x_values_inf,
        infinite["Da"],
        color='white',
        s=50,
        edgecolor='k',
        label='inf'
    )

    # Plot finite values with colormap
    sc = plt.scatter(
        x_values,
        finite["Da"],
        c=np.log10(finite["time_eq"].values),
        cmap="jet",
        s=50,
        edgecolor='k',
        vmin=0,
        vmax=vmax
    )

    plt.colorbar(label="log(t_eq)")

    # log scale for both axes
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.grid(True, which="major", ls="-", linewidth=1, alpha=0.8)
    plt.grid(True, which="minor", ls="-", linewidth=1, alpha=0.2)
    #plt.xlim(1e-2, 1e4)  # set x-axis limits
    plt.ylim(1e-3, 1e3)  # set y-axis limits

    # overlay equilibrium markers
    #for _, row in df.iterrows():
    #    if row["reached_eq"]:
    #        ax.scatter(row["Pe_H"], row["F"], color="black", marker="o", s=30, label="Reached eq")
    #    else:
    #        ax.scatter(row["Pe_H"], row["F"], color="red", marker="+", s=30, label="Not reached eq")

    # avoid duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="best")

    ax.set_xlabel(x_label)

    ax.set_ylabel("Damkohler number [ ]")
    ax.set_title("EpCAM capture characteristics")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()




def plot_time_eq_interp(df, grid_size=100, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 6))

    # Separate finite and infinite points
    finite = df[np.isfinite(df["time_eq"])]
    infinite = df[np.isinf(df["time_eq"])]

    # Convert units
    x_f = finite["Q_in"].values * 1e9 * 60  # uL/min
    # x_f = finite["Pe_H"].values
    y_f = finite["V_in"].values * 1e9       # uL
    z_f = np.log10(finite["time_eq"].values)  # log10(s)

    x_inf = infinite["Q_in"].values * 1e9 * 60
    y_inf = infinite["V_in"].values * 1e9

    # Create fine grid for interpolation
    xi = np.logspace(np.log10(x_f.min()), np.log10(x_f.max()), grid_size)
    yi = np.logspace(np.log10(y_f.min()), np.log10(y_f.max()), grid_size)
    XI, YI = np.meshgrid(xi, yi)

    # Interpolate using linear method
    ZI = griddata((x_f, y_f), z_f, (XI, YI), method='linear')

    # Plot interpolated contour
    cntr = ax.contourf(XI, YI, ZI, levels=30, cmap="turbo")

    # Overlay original finite points
    ax.scatter(x_f, y_f, c="k", s=10, alpha=0.6)

    # Overlay original infinite points in white
    if len(x_inf) > 0:
        ax.scatter(x_inf, y_inf, color="white", s=50, edgecolor="k", label="inf")

    # Colorbar
    cbar = plt.colorbar(cntr, ax=ax)
    cbar.set_label("log10(t_eq [s])")

    # Axis scaling and grid
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", ls="-", linewidth=1, alpha=0.8)
    ax.grid(True, which="minor", ls="-", linewidth=1, alpha=0.2)

    # Labels and title
    ax.set_xlabel()
    ax.set_xlabel(x_label)
    ax.set_ylabel("Sample volume [uL]")
    ax.set_title("Equilibrium time (interpolated, griddata)")

    # Legend
    if len(x_inf) > 0:
        ax.legend(loc="best")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_capt_perc_interp(df, grid_size=100, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 6))

    # Use all points, ignore where Q_in or V_in or capt_perc is NaN
    valid = df[["Q_in", "V_in", "capt_perc"]].replace([np.inf, -np.inf], np.nan).dropna()

    # Convert units
    x = valid["Q_in"].values * 1e9 * 60  # Flow rate [uL/min]
    y = valid["V_in"].values * 1e9  # Volume [uL]
    z = valid["capt_perc"].values  # Capture percentage

    # Create log-spaced interpolation grid
    xi = np.logspace(np.log10(x.min()), np.log10(x.max()), grid_size)
    yi = np.logspace(np.log10(y.min()), np.log10(y.max()), grid_size)
    XI, YI = np.meshgrid(xi, yi)

    # Interpolate using griddata (linear)
    ZI = griddata((x, y), z, (XI, YI), method="linear")

    # Mask out invalid interpolation regions (NaN)
    ZI_masked = np.ma.masked_invalid(ZI)

    # Plot interpolated contour
    cntr = ax.contourf(XI, YI, ZI_masked, levels=30, cmap="turbo")

    # Overlay data points
    ax.scatter(x, y, c="k", s=10, alpha=0.5)

    # Colorbar
    cbar = plt.colorbar(cntr, ax=ax)
    cbar.set_label("Capture rate [%]")

    # Axis scaling and grid
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", ls="-", linewidth=1, alpha=0.8)
    ax.grid(True, which="minor", ls="-", linewidth=1, alpha=0.2)

    # Labels and title
    ax.set_xlabel("Flow rate [uL/min]")
    ax.set_ylabel("Sample volume [uL]")
    ax.set_title("Capture percentage (interpolated, griddata)")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_site_occupancy_interp(df, params, grid_size=100, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 6))

    # Keep only valid numeric rows
    valid = df[["Q_in", "V_in", "b_last"]].replace([np.inf, -np.inf], np.nan).dropna()

    # Convert units
    x = valid["Q_in"].values * 1e9 * 60  # Flow rate [uL/min]
    y = valid["V_in"].values * 1e9  # Volume [uL]
    # z = 100 * valid["b_last"].values / params.b_m  # Occupancy rate [%]
    z = 100 * valid["b_last"].values / df["b_eq"]  # Ratio to equilibrium [%]

    # Create log-spaced interpolation grid
    xi = np.logspace(np.log10(x.min()), np.log10(x.max()), grid_size)
    yi = np.logspace(np.log10(y.min()), np.log10(y.max()), grid_size)
    XI, YI = np.meshgrid(xi, yi)

    # Interpolate using griddata (linear)
    ZI = griddata((x, y), z, (XI, YI), method="linear")

    # Mask out invalid interpolation regions (NaN)
    ZI_masked = np.ma.masked_invalid(ZI)

    # Plot interpolated contour
    cntr = ax.contourf(XI, YI, ZI_masked, levels=30, cmap="turbo")

    # Overlay data points
    ax.scatter(x, y, c="k", s=10, alpha=0.5)

    # Colorbar
    cbar = plt.colorbar(cntr, ax=ax)
    cbar.set_label("Occupancy rate [%]")

    # Axis scaling and grid
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", ls="-", linewidth=1, alpha=0.8)
    ax.grid(True, which="minor", ls="-", linewidth=1, alpha=0.2)

    # Labels and title
    ax.set_xlabel("Flow rate [uL/min]")
    ax.set_ylabel("Sample volume [uL]")
    ax.set_title("Capture percentage (interpolated, griddata)")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_flow_volume(df, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 6))

    # Split data
    finite = df[np.isfinite(df["time_eq"])]
    infinite = df[np.isinf(df["time_eq"])]

    # log scale of equilibrium time
    vmin = np.log10(finite["time_eq"].values).min()
    vmax = np.log10(finite["time_eq"].values).max()

    # Plot infinite values separately
    ax.scatter(
        infinite["Q_in"]*1e9*60,   # flow rate [uL/min]
        infinite["V_in"]*1e9,      # volume [uL]
        color="white",
        s=50,
        edgecolor="k",
        label="inf"
    )

    # Plot finite values
    sc = ax.scatter(
        finite["Q_in"]*1e9*60,     # flow rate [uL/min]
        finite["V_in"]*1e9,        # volume [uL]
        c=np.log10(finite["time_eq"].values),
        cmap="turbo",
        s=50,
        edgecolor="k",
        vmin=vmin,
        vmax=vmax
    )

    # Colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("log10(t_eq [s])")

    # Axis scaling
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", ls="-", linewidth=1, alpha=0.8)
    ax.grid(True, which="minor", ls="-", linewidth=1, alpha=0.2)

    # Labels and title
    ax.set_xlabel("Flow rate [nL/min]")
    ax.set_ylabel("Sample volume [nL]")
    ax.set_title("Equilibrium time vs flow rate and volume")

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="best")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()



def plot_capture_vs_peH_lambda(df, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 6))

    tri = ax.tricontourf(df["Pe_H"], df["F"], df["capt_perc"],
                         levels=10, cmap="plasma", norm=mcolors.LogNorm())
    fig.colorbar(tri, ax=ax, label="capt_perc")

    ax.scatter(df["Pe_H"], df["F"], color="black", marker="o", s=30)

    # log scale for both axes
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.grid(True, which="major", ls="-", linewidth=1, alpha=0.8)
    plt.grid(True, which="minor", ls="-", linewidth=1, alpha=0.2)
    plt.xlim(1e-2, 1e4)  # set x-axis limits
    plt.ylim(1e-3, 1e4)  # set y-axis limits

    # avoid duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="best")

    ax.set_xlabel("Pe_H (log)")
    ax.set_ylabel("Lambda (log)")
    ax.set_title("Capture % across Pe_H and F")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_t_eq_overview(df, grid_size=100, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 6))

    # Keep only valid numeric rows
    valid = df[["b_m","k_m", "Q_in", "k_on", "c_in", "time_eq", "tau", "k_off"]].replace([np.inf, -np.inf], np.nan).dropna()

    # Convert units / extract arrays
    x = valid["k_m"]
    y = (valid["k_on"] * valid["c_in"] + valid["k_off"]).values
    #y = (valid["k_on"] * valid["c_in"] * valid["b_m"]).values
    #y = valid["k_on"].values
    z = valid["time_eq"].values

    # Create log-spaced interpolation grid
    xi = np.logspace(np.log10(x.min()), np.log10(x.max()), grid_size)
    yi = np.logspace(np.log10(y.min()), np.log10(y.max()), grid_size)
    XI, YI = np.meshgrid(xi, yi)

    # Interpolate using griddata (linear)
    ZI = griddata((x, y), z, (XI, YI), method="linear")

    # Mask out invalid interpolation regions (NaN)
    ZI_masked = np.ma.masked_invalid(ZI)

    # --- Log-scaled color normalization ---
    norm = LogNorm(vmin=np.nanmin(ZI_masked), vmax=np.nanmax(ZI_masked))

    # Plot contour with log color scaling
    cntr = ax.contourf(XI, YI, ZI_masked, norm=norm, levels=30, cmap="turbo")

    # Overlay data points
    ax.scatter(x, y, c="k", s=10, alpha=0.5)

    # Colorbar
    cbar = plt.colorbar(cntr, ax=ax)
    cbar.set_label("Equilibrium time [s]")

    # Axis scaling and grid
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", ls="-", linewidth=1, alpha=0.8)
    ax.grid(True, which="minor", ls="-", linewidth=1, alpha=0.2)

    # Labels and title
    ax.set_xlabel("k_m")
    ax.set_ylabel("k_on * c_in + k_off")
    #ax.set_ylabel("k_on")
    ax.set_title("Equilibrium time")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_t_eq_overview_scatter(df, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 6))

    # Keep only valid numeric rows
    valid = df[["V_in","b_m","H_c","D","Q_in","Da","F","k_m", "k_on", "c_in", "time_eq", "tau", "k_off"]].replace([np.inf, -np.inf], np.nan).dropna()

    # Extract arrays
    tau_mt = 1 / valid["k_m"]
    tau_bind = 1 / (valid["k_on"] * valid["c_in"] + valid["k_off"])

    #x = valid["k_m"] * valid["c_in"] * valid["Q_in"]
    #x = valid["k_m"] * valid["Q_in"] * valid["c_in"] * valid["H_c"] / (2*valid["D"])
    # y = (valid["k_on"] * valid["c_in"] + valid["k_off"]).values
    #y = valid["k_off"] / valid["k_on"]
    # z = valid["time_eq"]
    z = valid["time_eq"]

    #x = valid["k_m"] / valid["tau"]
    #y = (valid["k_on"] * valid["c_in"]).values
    #

    # flux based
    t_pulse = valid["V_in"] / valid["Q_in"]
    #x = valid["k_m"] * valid["c_in"]    # J_D = m/s * mol/m3 = mol/m^2*s
    #y = (valid["k_on"] * valid["c_in"] * valid["b_m"]).values   # J_R = 1/(Ms) * M * mol/m2 = mol/m2*s

    # transport based
    x = valid["k_m"]
    #x = valid["F"] * valid["D"]
    #y = (valid["k_on"] * valid["c_in"] + valid["k_off"]).values
    y = (valid["k_on"] * valid["c_in"])
    #y = (valid["k_on"])

    #y = valid["k_on"]
    #x = valid["k_m"]

    # compute pulse time (s)


    # axes (mol / m^2)
    #x = valid["k_m"]# delivered per area during pulse
    #y = valid["k_on"] * valid["c_in"] + valid["k_off"]  # potential bound per area during pulse

    # Ensure z > 0 for log scaling
    z = np.clip(z, np.nanmin(z[z > 0]), None)

    # --- Log-scaled color normalization ---
    norm = LogNorm(vmin=z.min(), vmax=z.max())

    # Scatter plot
    sc = ax.scatter(x, y, c=z, cmap="turbo", norm=norm, s=40, edgecolor="none")

    # Colorbar (log scale)
    cbar = plt.colorbar(sc, ax=ax)
    #cbar.set_label("Equilibrium time [s] (log scale)")
    cbar.set_label("Eq. time [s]")

    # Axis scaling and grid
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", ls="-", linewidth=1, alpha=0.8)
    ax.grid(True, which="minor", ls="-", linewidth=1, alpha=0.2)

    # Labels and title
    ax.set_xlabel("F [ ]")
    ax.set_ylabel("k_on * c_in [1/s]")
    ax.set_title("Equilibrium time (scatter view)")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_Da_overview_scatter(df, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 6))

    # Keep only valid numeric rows
    valid = df[["Pe_H","V_in","b_m","H_c","D","Q_in","Da","F","k_m", "k_on", "c_in", "time_eq", "tau", "k_off"]].replace([np.inf, -np.inf], np.nan).dropna()

    # Extract arrays
    tau_mt = 1 / valid["k_m"]
    tau_bind = 1 / (valid["k_on"] * valid["c_in"] + valid["k_off"])

    #x = valid["k_m"] * valid["c_in"] * valid["Q_in"]
    #x = valid["k_m"] * valid["Q_in"] * valid["c_in"] * valid["H_c"] / (2*valid["D"])
    # y = (valid["k_on"] * valid["c_in"] + valid["k_off"]).values
    #y = valid["k_off"] / valid["k_on"]
    # z = valid["time_eq"]
    z = valid["Pe_H"]

    #x = valid["k_m"] / valid["tau"]
    #y = (valid["k_on"] * valid["c_in"]).values
    #

    # flux based
    t_pulse = valid["V_in"] / valid["Q_in"]
    #x = valid["k_m"] * valid["c_in"]    # J_D = m/s * mol/m3 = mol/m^2*s
    #y = (valid["k_on"] * valid["c_in"] * valid["b_m"]).values   # J_R = 1/(Ms) * M * mol/m2 = mol/m2*s

    # transport based
    #x = valid["k_m"]
    x = valid["F"]
    #y = (valid["k_on"] * valid["c_in"] + valid["k_off"]).values
    y = (valid["k_on"] * valid["c_in"])
    #y = (valid["k_on"])

    #y = valid["k_on"]
    #x = valid["k_m"]

    # compute pulse time (s)


    # axes (mol / m^2)
    #x = valid["k_m"]# delivered per area during pulse
    #y = valid["k_on"] * valid["c_in"] + valid["k_off"]  # potential bound per area during pulse

    # Ensure z > 0 for log scaling
    z = np.clip(z, np.nanmin(z[z > 0]), None)

    # --- Log-scaled color normalization ---
    norm = LogNorm(vmin=z.min(), vmax=z.max())

    # Scatter plot
    sc = ax.scatter(x, y, c=z, cmap="turbo", norm=norm, s=40, edgecolor="none")

    # Colorbar (log scale)
    cbar = plt.colorbar(sc, ax=ax)
    #cbar.set_label("Equilibrium time [s] (log scale)")
    cbar.set_label("Damkohler number [ ]")

    # Axis scaling and grid
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", ls="-", linewidth=1, alpha=0.8)
    ax.grid(True, which="minor", ls="-", linewidth=1, alpha=0.2)

    # Labels and title
    ax.set_xlabel("F [ ]")
    ax.set_ylabel("k_on * c_in [1/s]")
    ax.set_title("Damkohler number")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_volume_required_scatter(df, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 6))

    # Keep only valid numeric rows
    valid = df[["V_in","b_m","H_c","D","Q_in","Da","F","k_m", "k_on", "c_in", "time_eq", "tau", "k_off"]].replace([np.inf, -np.inf], np.nan).dropna()

    # Extract arrays
    tau_mt = 1 / valid["k_m"]
    tau_bind = 1 / (valid["k_on"] * valid["c_in"] + valid["k_off"])

    #x = valid["k_m"] * valid["c_in"] * valid["Q_in"]
    #x = valid["k_m"] * valid["Q_in"] * valid["c_in"] * valid["H_c"] / (2*valid["D"])
    # y = (valid["k_on"] * valid["c_in"] + valid["k_off"]).values
    #y = valid["k_off"] / valid["k_on"]
    # z = valid["time_eq"]
    z = valid["time_eq"] * valid["Q_in"] * 1e9 # volume in uL

    #x = valid["k_m"] / valid["tau"]
    #y = (valid["k_on"] * valid["c_in"]).values
    #

    # flux based
    t_pulse = valid["V_in"] / valid["Q_in"]
    #x = valid["k_m"] * valid["c_in"]    # J_D = m/s * mol/m3 = mol/m^2*s
    #y = (valid["k_on"] * valid["c_in"] * valid["b_m"]).values   # J_R = 1/(Ms) * M * mol/m2 = mol/m2*s

    # transport based
    x = valid["k_m"]
    #x = valid["F"]
    #y = (valid["k_on"] * valid["c_in"] + valid["k_off"]).values
    y = (valid["k_on"] * valid["c_in"])
    #y = (valid["k_on"])

    #y = valid["k_on"]
    #x = valid["k_m"]

    # compute pulse time (s)


    # axes (mol / m^2)
    #x = valid["k_m"]# delivered per area during pulse
    #y = valid["k_on"] * valid["c_in"] + valid["k_off"]  # potential bound per area during pulse

    # Ensure z > 0 for log scaling
    z = np.clip(z, np.nanmin(z[z > 0]), None)

    # --- Log-scaled color normalization ---
    norm = LogNorm(vmin=z.min(), vmax=z.max())

    # Scatter plot
    sc = ax.scatter(x, y, c=z, cmap="turbo", norm=norm, s=40, edgecolor="none")

    # Colorbar (log scale)
    cbar = plt.colorbar(sc, ax=ax)
    #cbar.set_label("Equilibrium time [s] (log scale)")
    cbar.set_label("Volume [uL]")

    # Axis scaling and grid
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", ls="-", linewidth=1, alpha=0.8)
    ax.grid(True, which="minor", ls="-", linewidth=1, alpha=0.2)

    # Labels and title
    ax.set_xlabel("F [ ]")
    ax.set_ylabel("k_on * c_in [1/s]")
    ax.set_title("Minimum volume required")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_timescale_collapse_with_labels(df, save_path=None):

    valid = df[["Q_in","c_in","k_m","k_on","k_off","b_m","H_c","time_eq"]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    # Characteristic times
    tau_supply    = 1 / ((valid["Q_in"] * valid["c_in"]) / valid["b_m"])
    tau_transport = 1 / (valid["k_m"] / valid["H_c"])
    tau_bind      = 1 / (valid["k_on"] * valid["c_in"] + valid["k_off"])

    taus = np.vstack([tau_supply, tau_transport, tau_bind]).T  # shape N×3

    # Names for reference
    names = np.array(["supply", "transport", "binding"])

    # Sort each row but also track which mechanism
    sort_idx = np.argsort(taus, axis=1)
    sorted_taus = np.take_along_axis(taus, sort_idx, axis=1)

    # Extract slowest and 2nd slowest
    tau_2nd = sorted_taus[:, 1]
    tau_slow = sorted_taus[:, 2]

    slowest_mech = names[sort_idx[:, 2]]       # slowest mechanism name
    second_mech  = names[sort_idx[:, 1]]       # second slowest mechanism name

    # Assign markers for slowest process
    marker_map = {
        "supply": "o",
        "transport": "s",
        "binding": "^"
    }
    markers = [marker_map[m] for m in slowest_mech]

    # Assign edgecolors for second slowest
    color_map = {
        "supply": "black",
        "transport": "gray",
        "binding": "white"
    }
    edges = [color_map[m] for m in second_mech]

    t_val = np.clip(valid["time_eq"], np.nanmin(valid["time_eq"][valid["time_eq"] > 0]), None)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 6))

    # Dummy mappable for colorbar
    mappable = ax.scatter(
        np.log10(tau_2nd), np.log10(tau_slow),
        c=np.log10(t_val),
        cmap="viridis",
        s=0  # not visible
    )
    cb = plt.colorbar(mappable, label="log10(time_eq)")

    # Now draw each point with its marker & edgecolor
    for x, y, m, ec, t in zip(
            np.log10(tau_2nd),
            np.log10(tau_slow),
            markers,
            edges,
            t_val
    ):
        ax.scatter(x, y, c=np.log10(t), cmap="viridis",
                   s=60, marker=m, edgecolors=ec, linewidths=1)

    ax.set_xlabel("log10(Second Slowest Timescale) [s]")
    ax.set_ylabel("log10(Slowest Timescale) [s]")
    ax.set_title("Timescale Collapsed Regime Map\n(Markers denote limiting mechanisms)")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_relative_rate_map(df, save_path=None):

    valid = df[["H_c","b_m","Q_in","c_in","k_m","k_on","k_off","time_eq"]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    # Absolute rates
    r_binding = valid["k_on"] * valid["c_in"] + valid["k_off"]   # [1/s]
    r_supply = valid["Q_in"] * valid["c_in"] / valid["b_m"]                     # [mol/s scaled to rate-like]
    r_transport = valid["k_m"] / valid["H_c"]                                   # [m/s]

    # Relative (dimensionless) rates
    R1 = r_supply / r_binding     # supply / binding
    R2 = r_transport / r_binding  # transport / binding

    t_val = valid["time_eq"]
    t_val = np.clip(t_val, np.nanmin(t_val[t_val > 0]), None)

    fig, ax = plt.subplots(figsize=(7,6))

    sc = ax.scatter(
        np.log10(R1),
        np.log10(R2),
        c=np.log10(t_val),
        s=30, alpha=0.8
    )

    plt.colorbar(sc, label="log10(time_eq)")

    ax.set_xlabel("log10(R1 = r_supply / r_binding)")
    ax.set_ylabel("log10(R2 = r_transport / r_binding)")
    plt.title("Relative Rate Regime Map")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_t_eq_overview_scatter3D(df, save_path=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Keep only valid numeric rows
    valid = df[["S","V_in","b_m","H_c","D","Q_in","Da","F","k_m", "k_on", "c_in", "time_eq", "tau", "k_off"]].replace([np.inf, -np.inf], np.nan).dropna()

    t_val = valid["time_eq"]
    # Ensure t_val > 0 for log scaling
    t_val = np.clip(t_val, np.nanmin(t_val[t_val > 0]), None)

    # --- Log-transform axes manually (3D cannot use ax.set_xscale) ---
    x_log = np.log10(valid["k_m"])
    k_obs = valid["k_on"] * valid["c_in"] + valid["k_off"]
    y_log = np.log10(1 / k_obs)
    #y_log = np.log10(valid["k_on"] * valid["c_in"])
    z_log = np.log10(valid["Q_in"] * valid["c_in"])

    # test below --
    moles_mt = valid["S"] * valid["c_in"] * valid["k_m"]  # mol / s
    moles_bind = valid["b_m"] * valid["S"] * (valid["k_on"] * valid["c_in"] + valid["k_off"])
    #moles_bind =    # [1/s]

    moles_supply = valid["c_in"] * valid["Q_in"]    # mol/s

    x_log = np.log10(moles_mt)
    y_log = np.log10(moles_bind)
    z_log = np.log10(moles_supply)
    # -- end test


    # --- Color normalization (log) ---
    norm = LogNorm(vmin=t_val.min(), vmax=t_val.max())

    # --- Scatter ---
    sc = ax.scatter(x_log, y_log, z_log, c=t_val, cmap="turbo", norm=norm,
                    s=40, edgecolor="none")

    # --- Colorbar ---
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Eq. time [s]")

    # --- Axis labels ---
    #ax.set_xlabel("log10(k_m)  [m/s]")
    #ax.set_ylabel("log10(k_on * c_in)  [1/s]")
    #ax.set_zlabel("log10(Q_in * c_in)  [mol/s]")

    ax.set_xlabel("log10(moles_mt)  [mol/s]")
    ax.set_ylabel("log10(moles_bind)  [mol/s]")
    ax.set_zlabel("log10(moles_supply)  [mol/s]")

    # Colorbar (log scale)
    cbar = plt.colorbar(sc, ax=ax)
    #cbar.set_label("Equilibrium time [s] (log scale)")
    cbar.set_label("Eq. time [s]")

    # Axis scaling and grid
    #ax.set_xscale("log")
    #ax.set_yscale("log")
    #ax.set_zscale("log")
    #ax.grid(True, which="major", ls="-", linewidth=1, alpha=0.8)
    #ax.grid(True, which="minor", ls="-", linewidth=1, alpha=0.2)

    # Labels and title
    ax.set_title("Equilibrium time (scatter view)")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_V_min_overview_scatter3D(df, save_path=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Keep only valid numeric rows
    valid = df[["V_in","b_m","H_c","D","Q_in","Da","F","k_m", "k_on", "c_in", "time_eq", "tau", "k_off"]].replace([np.inf, -np.inf], np.nan).dropna()

    c_val = valid["time_eq"] * valid["Q_in"]
    # Ensure t_val > 0 for log scaling
    c_val = np.clip(c_val, np.nanmin(c_val[c_val > 0]), None)

    # --- Log-transform axes manually (3D cannot use ax.set_xscale) ---
    x_log = np.log10(valid["k_m"])
    k_obs = valid["k_on"] * valid["c_in"] + valid["k_off"]
    y_log = np.log10(1 / k_obs)
    #y_log = np.log10(valid["k_on"] * valid["c_in"])
    z_log = np.log10(valid["Q_in"] * valid["c_in"])


    # --- Color normalization (log) ---
    norm = LogNorm(vmin=c_val.min(), vmax=c_val.max())

    # --- Scatter ---
    sc = ax.scatter(x_log, y_log, z_log, c=c_val, cmap="turbo", norm=norm,
                    s=40, edgecolor="none")

    # --- Colorbar ---
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Eq. time [s]")

    # --- Axis labels ---
    ax.set_xlabel("log10(k_m)  [m/s]")
    ax.set_ylabel("log10(k_on * c_in)  [1/s]")
    ax.set_zlabel("log10(Q_in * c_in)  [mol/s]")

    # Colorbar (log scale)
    cbar = plt.colorbar(sc, ax=ax)
    #cbar.set_label("Equilibrium time [s] (log scale)")
    cbar.set_label("Eq. time [s]")

    # Axis scaling and grid
    #ax.set_xscale("log")
    #ax.set_yscale("log")
    #ax.set_zscale("log")
    #ax.grid(True, which="major", ls="-", linewidth=1, alpha=0.8)
    #ax.grid(True, which="minor", ls="-", linewidth=1, alpha=0.2)

    # Labels and title
    ax.set_title("Equilibrium time (scatter view)")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm
import os

def plot_t_eq_overview_scatter3D_animated(df, save_path=None, gif_path=None):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # --- Clean valid data ---
    valid = df[["V_in","b_m","H_c","D","Q_in","Da","F","k_m", "k_on", "c_in", "time_eq", "tau", "k_off"]] \
            .replace([np.inf, -np.inf], np.nan).dropna()

    t_val = valid["time_eq"]
    t_val = np.clip(t_val, np.nanmin(t_val[t_val > 0]), None)

    # --- Log-transform axes manually (3D cannot use ax.set_xscale) ---
    x_log = np.log10(valid["k_m"])
    k_obs = valid["k_on"] * valid["c_in"] + valid["k_off"]
    y_log = np.log10(1 / k_obs)
    #y_log = np.log10(valid["k_on"] * valid["c_in"])
    z_log = np.log10(valid["Q_in"] * valid["c_in"])

    # --- Color normalization (log) ---
    norm = LogNorm(vmin=t_val.min(), vmax=t_val.max())

    # --- Scatter ---
    sc = ax.scatter(x_log, y_log, z_log, c=t_val, cmap="turbo", norm=norm,
                    s=40, edgecolor="none")

    # --- Colorbar ---
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Eq. time [s]")

    # --- Axis labels ---
    ax.set_xlabel("log10(k_m)  [m/s]")
    ax.set_ylabel("log10(k_on * c_in)  [1/s]")
    ax.set_zlabel("log10(Q_in * c_in)  [mol/s]")

    ax.set_title("Equilibrium time (3D log-scaled scatter)")

    # --- Nice ticks (example) ---
    xticks = [-8, -7, -6, -5, -4]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"1e{t}" for t in xticks])
    # Do same for y/z if desired

    plt.tight_layout()

    # Save static figure if needed
    if save_path is not None:
        #os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Static plot saved to {save_path}")

    # --- Animation: rotate around azimuth ---
    def update(frame):
        ax.view_init(elev=25, azim=frame)
        return fig,

    anim = FuncAnimation(fig, update, frames=360, interval=30)

    # Save GIF if requested
    if gif_path is not None:
        #os.makedirs(os.path.dirname(gif_path), exist_ok=True)
        anim.save(gif_path, writer=PillowWriter(fps=20))
        print(f"GIF animation saved to {gif_path}")

    plt.show()

def plot_t_eq_two_axis_scatter(df, save_path=None):


    # Keep only valid numeric rows
    valid = df[["b_eq","S","V","L_s","V_in","b_m","H_c","D","Q_in","Da","F","k_m", "k_on", "c_in", "time_eq", "tau", "k_off"]].replace([np.inf, -np.inf], np.nan).dropna()

    # Extract arrays
    #tau_mt = 1 / (valid["k_m"] / ((valid["H_c"]) / 2))
    tau_mt = 1 / (valid["k_m"] / valid["L_s"])
    tau_mt = 1 / (valid["k_m"] / 1)
    tau_bind =  (valid["k_on"] * valid["c_in"] + valid["k_off"])
    tau_supply = 1 / (valid ["V"] / valid["Q_in"])

    # with moles
    moles_mt = valid["S"] * valid["c_in"] * valid["k_m"]  # mol / s
    moles_bind =  (valid["k_on"] * valid["c_in"] + valid["k_off"])
    #moles_bind =    # [1/s]

    moles_supply = valid["c_in"] * valid["Q_in"]    # mol/s

    #moles_mt = valid["F"] * moles_supply

    tau_transport = tau_mt / tau_supply
    tau_limiting = np.minimum(tau_mt, tau_supply)
    tau_eff = 1 / (1 / tau_mt + 1 / tau_supply)
    moles_eff = 1 / (1 / moles_mt + 1 / moles_supply)   # mol/s
    #moles_eff = np.minimum(moles_mt, moles_supply)
    moles_eff = np.minimum(moles_mt, moles_supply)

    z = valid["time_eq"]

    # transport based
    #x = tau_eff
    #y = tau_bind
    x = moles_eff
    y = moles_bind

    # Ensure z > 0 for log scaling
    z = np.clip(z, np.nanmin(z[z > 0]), None)

    # --- Log-scaled color normalization ---
    norm = LogNorm(vmin=z.min(), vmax=z.max())
    norm_tau = LogNorm(vmin=tau_eff.min(), vmax=tau_eff.max())
    norm_mol = LogNorm(vmin=moles_eff.min(), vmax=moles_eff.max())

    # 2D scatter plot
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(
        moles_mt, moles_supply, #tau_mt, tau_supply,
        c=moles_eff, #c=tau_eff,
        cmap="viridis",
        norm = norm_mol,
        s=50,
        alpha=0.8
    )

    plt.xscale("log")
    plt.yscale("log")
    #plt.xlabel("tau_mt [s] (transport timescale)")
    #plt.ylabel("tau_supply [s] (supply timescale)")
    plt.xlabel("moles_km [mol/s] (transport timescale)")
    plt.ylabel("moles_supply [mol/s] (supply timescale)")
    plt.title("2D timescale plot: color = tau_eff")
    plt.colorbar(sc, label="moles_eff [s]")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 6))
    # Scatter plot
    sc = ax.scatter(x, y, c=z, cmap="turbo", norm=norm, s=40, edgecolor="none")

    # Colorbar (log scale)
    cbar = plt.colorbar(sc, ax=ax)
    #cbar.set_label("Equilibrium time [s] (log scale)")
    cbar.set_label("Eq. time [s]")

    # Axis scaling and grid
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", ls="-", linewidth=1, alpha=0.8)
    ax.grid(True, which="minor", ls="-", linewidth=1, alpha=0.2)

    # Labels and title
    #ax.set_xlabel("tau_eff [1/s] ")
    #ax.set_ylabel("tau_b [1/s]")
    ax.set_xlabel("transport rate [mol/s] ")
    ax.set_ylabel("binding rate [mol/s]")
    ax.set_title("Equilibrium time (scatter view)")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_V_min_two_axis_scatter(df, save_path=None):


    # Keep only valid numeric rows
    valid = df[["V","L_s","V_in","b_m","H_c","D","Q_in","Da","F","k_m", "k_on", "c_in", "time_eq", "tau", "k_off"]].replace([np.inf, -np.inf], np.nan).dropna()

    # Extract arrays
    #tau_mt = 1 / (valid["k_m"] / ((valid["H_c"]) / 2))
    tau_mt = 1 / (valid["k_m"] / valid["L_s"])
    tau_mt = 1 / (valid["k_m"] / 1)
    tau_bind =  (valid["k_on"] * valid["c_in"] + valid["k_off"])
    tau_supply = 1 / (valid ["V"] / valid["Q_in"])

    tau_transport = tau_mt / tau_supply
    tau_limiting = np.minimum(tau_mt, tau_supply)
    tau_eff = 1 / (1 / tau_mt + 1 / tau_supply)

    z = 1e9 * valid["time_eq"] * valid["Q_in"]

    # transport based
    x = tau_eff
    y = tau_bind

    # Ensure z > 0 for log scaling
    z = np.clip(z, np.nanmin(z[z > 0]), None)

    # --- Log-scaled color normalization ---
    norm = LogNorm(vmin=z.min(), vmax=z.max())
    norm_tau = LogNorm(vmin=tau_eff.min(), vmax=tau_eff.max())



    fig, ax = plt.subplots(figsize=(7, 6))
    # Scatter plot
    sc = ax.scatter(x, y, c=z, cmap="turbo", norm=norm, s=40, edgecolor="none")

    # Colorbar (log scale)
    cbar = plt.colorbar(sc, ax=ax)
    #cbar.set_label("Equilibrium time [s] (log scale)")
    cbar.set_label("Required volume [uL]")

    # Axis scaling and grid
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", ls="-", linewidth=1, alpha=0.8)
    ax.grid(True, which="minor", ls="-", linewidth=1, alpha=0.2)

    # Labels and title
    ax.set_xlabel("tau_eff [1/s] ")
    ax.set_ylabel("tau_b [1/s]")
    ax.set_title("Required volume [uL]")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_V_min_vs_Q_in(df, save_path=None):


    # Keep only valid numeric rows
    valid = df[["V","L_s","V_in","b_m","H_c","D","Q_in","Da","F","k_m", "k_on", "c_in", "time_eq", "tau", "k_off"]].replace([np.inf, -np.inf], np.nan).dropna()

    # Extract arrays
    #tau_mt = 1 / (valid["k_m"] / ((valid["H_c"]) / 2))
    tau_mt = 1 / (valid["k_m"] / valid["L_s"])
    tau_mt = 1 / (valid["k_m"] / 1)
    tau_bind =  (valid["k_on"] * valid["c_in"] + valid["k_off"])
    tau_supply = 1 / (valid ["V"] / valid["Q_in"])

    tau_transport = tau_mt / tau_supply
    tau_limiting = np.minimum(tau_mt, tau_supply)
    tau_eff = 1 / (1 / tau_mt + 1 / tau_supply)

    y = 1e9 * valid["time_eq"] * valid["Q_in"]

    # transport based
    x = valid["Q_in"]*60*1e9
    z = valid["time_eq"]

    # Ensure z > 0 for log scaling
    z = np.clip(z, np.nanmin(z[z > 0]), None)

    # --- Log-scaled color normalization ---
    norm = LogNorm(vmin=z.min(), vmax=z.max())
    norm_tau = LogNorm(vmin=tau_eff.min(), vmax=tau_eff.max())



    fig, ax = plt.subplots(figsize=(7, 6))
    # Scatter plot
    sc = ax.scatter(x, y, c=z, cmap="turbo", norm=norm, s=40, edgecolor="none")

    # Colorbar (log scale)
    cbar = plt.colorbar(sc, ax=ax)
    #cbar.set_label("Equilibrium time [s] (log scale)")
    cbar.set_label("Equilibrium time [s]")

    # Axis scaling and grid
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", ls="-", linewidth=1, alpha=0.8)
    ax.grid(True, which="minor", ls="-", linewidth=1, alpha=0.2)

    # Labels and title
    ax.set_xlabel("Q_in [uL/min]")
    ax.set_ylabel("Required volume [uL]")
    ax.set_title("Volume requirement")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()