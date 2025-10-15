import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
def plot_Damkohler_time(t,Da_t, save_path=None):
    plt.plot(t, Da_t, label='b_hat (fraction of b_m)')
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

def plot_optimization(df,params,Q_in_vals, save_path=None):
    Q_in = df["Q_in"]
    Pe_H = df["Q_in"] / (params.D * params.W_c)
    capt_perc = df["capt_perc"]
    #time_eq = df["time_eq"]
    time_capt = df["time_capt"]
    fig, ax1 = plt.subplots(figsize=(7,6))
    Q_conversion_factor = (1 / 60) * 10 ** (-9)
    Q_in_uL = Q_in_vals / Q_conversion_factor


    # Left y-axis for capture percentage
    ax1.plot(Q_in_uL, capt_perc, 'k-', label='Capture percentage')
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

def plot_peclet_batch(df, save_path=None):
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

    plt.scatter(
        #infinite["Q_in"]*1e9*60,

        infinite["Pe_H"],
        infinite["Da"],
        color='white',
        s=50,
        edgecolor='k',
        label='inf'
    )

    # Plot finite values with colormap
    sc = plt.scatter(
        finite["Pe_H"],
        #finite["Q_in"]*1e9*60,
        finite["Da"],
        c=np.log10(finite["time_eq"].values),
        cmap="jet",
        s=50,
        edgecolor='k',
        vmin=0,
        vmax=vmax
    )

    ## Plot inf values in a fixed color, e.g., red
    #plt.scatter(
    #    infinite["Q_in"]*1e9*60,
    #    #infinite["Pe_H"],
    #    infinite["Da_2"],
    #    color='white',
    #    s=50,
    #    edgecolor='r',
    #    label='inf'
    #)
    #
    ## Plot finite values with colormap
    #sc = plt.scatter(
    #    #finite["Pe_H"],
    #    finite["Q_in"]*1e9*60,
    #    finite["Da_2"],
    #    c=np.log10(finite["time_eq"].values),
    #    cmap="jet",
    #    s=50,
    #    edgecolor='r',
    #    vmin=0,
    #    vmax=vmax
    #)

    plt.colorbar(label="log(t_eq)")

    # log scale for both axes
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.grid(True, which="major", ls="-", linewidth=1, alpha=0.8)
    plt.grid(True, which="minor", ls="-", linewidth=1, alpha=0.2)
    #plt.xlim(1e-2, 1e4)  # set x-axis limits
    plt.ylim(1e-2, 1e7)  # set y-axis limits

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

    #ax.set_xlabel("Flow rate [uL/min]")
    ax.set_xlabel("Peclet number [ ]")
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
    ax.set_xlabel("Flow rate [uL/min]")
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

def plot_site_occupancy_interp(df, grid_size=100, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 6))

    # Compute site occupancy [%]

    df["occupancy"] = 100 * df["b"].apply(lambda arr: arr[-1]) / df["b_eq"]

    # Keep only valid numeric rows
    valid = df[["Q_in", "V_in", "occupancy"]].replace([np.inf, -np.inf], np.nan).dropna()

    # Convert units
    x = valid["Q_in"].values * 1e9 * 60  # Flow rate [uL/min]
    y = valid["V_in"].values * 1e9  # Volume [uL]
    z = valid["occupancy"].values  # Occupancy rate [%]

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

