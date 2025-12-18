import matplotlib
matplotlib.use("QtAgg")   # Best backend for PyCharm
import matplotlib.pyplot as plt

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

    V_req = finite["Q_in"].values * finite["time_eq"]

    # set colormap range
    if len(finite) > 0:
        #vmin = np.log10(finite["time_eq"].values).min()  # or manually, e.g., 2
        #vmax = np.log10(finite["time_eq"].values).max()  # corrected
        vmin = np.log10(V_req).min()  # or manually, e.g., 2
        vmax = np.log10(V_req).max()  # corrected
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
        infinite["Da_2"],
        #color='white',
        c='k',
        s=50,
        edgecolor='k',
        label='inf'
    )

    # Plot finite values with colormap
    sc = plt.scatter(
        x_values,
        finite["Da_2"],
        #c=np.log10(finite["time_eq"].values),
        # c=np.log10(V_req)# ,
        c = 'k',
        # cmap="jet",
        s=50,
        edgecolor='k',
        vmin=vmin,
        vmax=vmax
    )

    # plt.colorbar(label="log(t_eq)")



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

from scipy.interpolate import Rbf
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.interpolate import SmoothBivariateSpline
from scipy.ndimage import gaussian_filter
import matplotlib.tri as tri

def plot_site_occupancy_interp(df, params, grid_size=25, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 6))

    # Keep only valid numeric rows
    valid = df[["Q_in", "V_in", "b_last"]].replace([np.inf, -np.inf], np.nan).dropna()

    # Convert units
    x = valid["Q_in"].values * 1e9 * 60  # Flow rate [uL/min]
    y = valid["V_in"].values * 1e9  # Volume [uL]
    # z = 100 * valid["b_last"].values / params.b_m  # Occupancy rate [%]
    z = 100 * valid["b_last"].values / df["b_eq"]  # Ratio   to equilibrium [%]

    # Create log-spaced interpolation grid
    xi = np.logspace(np.log10(x.min()), np.log10(x.max()), grid_size)
    yi = np.logspace(np.log10(y.min()), np.log10(y.max()), grid_size)
    XI, YI = np.meshgrid(xi, yi)

    # Interpolate using griddata (linear)

    #interp = LinearNDInterpolator(list(zip(x, y)), z)
    #ZI = interp(XI, YI)

    rbf = Rbf(x, y, z, function='linear',smooth=.1)
    #ZI = rbf(XI, YI)
    # x, y, z are your scattered data points

    # Evaluate on your mesh grid

    #ZI = griddata((x, y), z, (XI, YI), method="cubic")



    # Optional: log-transform to handle wide dynamic ranges
    lx = np.log10(x)
    ly = np.log10(y)
    lz = z

    ZI = griddata((x, y), z, (XI, YI), method='linear')
    ZI = gaussian_filter(ZI, sigma=1.0)

    # Mask out invalid interpolation regions (NaN)
    ZI_masked = np.ma.masked_invalid(ZI)

    ZI_masked[ZI_masked>95] = 96

    # --- Define your band ---
    low = 94
    high = 96


    # Plot interpolated contour
    cntr = ax.contourf(XI, YI, ZI_masked, levels=30, cmap="YlGn", alpha=0.8)
    # Colorbar
    cbar = plt.colorbar(cntr, ax=ax)
    cbar.set_label("Part of equilibrium bound [%]")

    #cs = ax.contourf(XI, YI, ZI_masked, levels=[low, high], alpha=0.3, colors=["red"])
    #cs = ax.contour(XI, YI, ZI_masked, levels=[95], colors="red", linewidths=2)


    triang = tri.Triangulation(x, y)
    contours = ax.tricontour(triang, z, levels=[95], colors='black', linewidths=2)
    #ax.clabel(contours, fmt="95%%", fontsize=10)
    manual_positions = [(0.1,20)]  # Example positions
    #ax.clabel(contours, fmt='%1.0f', inline=False, fontsize=12)
    #ax.labels(contour,"test", fontsize=12)

    # plot simulation points
    #ax.scatter(x, y, c="k", s=10, alpha=0.5)

    # Axis scaling and grid
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", ls="-", linewidth=1, alpha=0.8)
    ax.grid(True, which="minor", ls="-", linewidth=1, alpha=0.2)

    # Labels and title
    ax.set_xlabel("Flow rate [uL/min]")
    ax.set_ylabel("Sample volume [uL]")
    #ax.set_title("Capture percentage (interpolated, griddata)")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_occupancy_multi_contours(dfs, params_list=None, labels=None, c2_index=1, grid_size=100, save_path=None):
    """
    Plot full occupancy surface for one concentration (c2) and 95% contours for all datasets.

    dfs: list of DataFrames [df1, df2, df3]
    params_list: optional list of parameter objects for normalization
    labels: optional labels for the contour lines
    c2_index: index of the concentration to show full interpolation (default 1 -> middle concentration)
    """
    import matplotlib.tri as tri
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter
    import numpy as np
    from scipy.interpolate import Rbf
    import os

    fig, ax = plt.subplots(figsize=(7, 6))

    colors = ['red', 'blue', 'green']  # colors for contours of c1, c2, c3

    for i, df in enumerate(dfs):
        valid = df[["b_eq","Q_in", "V_in", "b_last"]].replace([np.inf, -np.inf], np.nan).dropna()
        x = valid["Q_in"].values * 1e9 * 60  # uL/min
        y = valid["V_in"].values * 1e9       # uL
        z = 100 * valid["b_last"].values / df["b_eq"]  # occupancy %
        #z = valid["time_eq"]

        valid_time_eq = df[["time_eq"]].replace([np.inf, -np.inf], np.nan).dropna()
        # Interpolation grid
        xi = np.logspace(np.log10(x.min()), np.log10(x.max()), grid_size)
        yi = np.logspace(np.log10(y.min()), np.log10(y.max()), grid_size)
        XI, YI = np.meshgrid(xi, yi)

        # Full interpolation only for c2
        if i == c2_index:
            # Separate finite and infinite points
            finite = df[np.isfinite(df["time_eq"])]
            infinite = df[np.isinf(df["time_eq"])]

            # Convert units
            x_f = finite["Q_in"].values * 1e9 * 60  # uL/min
            # x_f = finite["Pe_H"].values
            y_f = finite["V_in"].values * 1e9  # uL
            z_f = np.log10(finite["time_eq"].values)  # log10(s)

            x_inf = infinite["Q_in"].values * 1e9 * 60
            y_inf = infinite["V_in"].values * 1e9

            # Create fine grid for interpolation
            xi = np.logspace(np.log10(x_f.min()), np.log10(x_f.max()), grid_size)
            yi = np.logspace(np.log10(y_f.min()), np.log10(y_f.max()), grid_size)
            XI, YI = np.meshgrid(xi, yi)

            # Interpolate using linear method
            ZI = griddata((x_f, y_f), z_f, (XI, YI), method='cubic')

            # Plot interpolated contour
            cntr = ax.contourf(XI, YI, ZI, levels=30, cmap="turbo")

            # Colorbar
            cbar = plt.colorbar(cntr, ax=ax)
            cbar.set_label("log10(eq. time) [s])")   # Filled contour for c2

            # cntr = ax.contourf(XI, YI, ZI, levels, cmap="plasma", norm=LogNorm(vmin=vmin,vmax=vmax))
            # cbar = plt.colorbar(cntr, ax=ax)
            # cbar.set_label("Occupancy rate [%]")

        # 95% contour for all datasets
        triang = tri.Triangulation(x, y)
        contours = ax.tricontour(triang, z, levels=[95], colors='black', linewidths=2)

        # Scatter the simulation points
        #ax.scatter(x, y, c="k", s=10, alpha=0.3)

    # Axis settings
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Flow rate [uL/min]")
    ax.set_ylabel("Sample volume [uL]")
    ax.set_title("Percentage of equilibrium reached")
    ax.grid(True, which="major", ls="-", linewidth=1, alpha=0.8)
    ax.grid(True, which="minor", ls="-", linewidth=1, alpha=0.2)

    if labels:
        ax.legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
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

def plot_varying_Q(df, params):
    fig, ax1 = plt.subplots(figsize=(7, 6))

    # Keep only valid numeric rows
    valid = df[["time_eq","Q_in", "V_in", "b_last"]].replace([np.inf, -np.inf], np.nan).dropna()

    # Convert units
    x = valid["Q_in"].values * 1e9 * 60  # Flow rate [uL/min]

    y = valid["time_eq"].values     # Eq. time
    yy = valid["Q_in"].values * valid["time_eq"].values * 1e9  # Volume [uL]

    # ---------------- Left axis: Volume (black) ----------------
    ax1.plot(x, yy, marker='s', color='black', label="Volume")
    ax1.set_xlabel("Flow rate [µL/min]")
    ax1.set_ylabel("Volume [µL]", color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # ---------------- Right axis: Equilibration time (blue) ----------------
    ax2 = ax1.twinx()
    ax2.plot(x, y, marker='o', color='blue', label="Equilibration time")
    ax2.set_ylabel("Equilibration time [s]", color='blue')
    ax2.tick_params(axis='y', labelcolor='black')

    ax1.set_xlim(1e-1, 1e2)  # x-axis limits (example)
    ax1.set_ylim(5e0, 1e2)

    ax1.set_xscale('log')  # log scale on x-axis
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax1.grid(True, which="major", ls="-", linewidth=1, alpha=0.8)
    ax1.grid(True, which="minor", ls="-", linewidth=1, alpha=0.2)

    # Make right axis tick positions match left axis
    ax2.set_yticks(ax1.get_yticks())
    ax2.set_ylim(ax1.get_ylim())

    # DISPLACE RIGHT AXIS BY ONE DECADE
    left_min, left_max = ax1.get_ylim()

    # Add ONE DECADE → multiply limits by 10
    ax2.set_ylim(left_min * 10, left_max * 10)



    # Shade region left of x = 20
    ax1.axvspan(ax1.get_xlim()[0], 20, color='gray', alpha=0.2)
    ax1.axvline(20, color='gray', linestyle='-')

    # Text
    ax1.text(
        0.25, 0.85,  # (x, y) as a fraction of the axis (0–1)
        "Full collection",
        transform=ax1.transAxes,
        fontsize=14,
        color="black",
    )

    # Optional: cleaner layout
    fig.tight_layout()
    plt.show()


def plot_varying_Q_collection(df, params):
    fig, ax1 = plt.subplots(figsize=(7, 6))

    # Keep only valid numeric rows
    valid = df[["full_collection","time_eq", "Q_in", "V_in", "b_last"]].replace([np.inf, -np.inf], np.nan).dropna()

    # Make sure full_collection matches the filtered df
    full_collection = valid["full_collection"]

    full_collection = full_collection[valid.index]

    # Convert units
    x = valid["Q_in"].values * 1e9 * 60  # Flow rate [uL/min]
    y = valid["time_eq"].values  # Eq. time
    yy = valid["Q_in"].values * valid["time_eq"].values * 1e9  # Volume [uL]

    # ---------------- Left axis: Volume (black) ----------------
    ax1.plot(x, yy, marker='s', color='black', label="Volume")
    ax1.set_xlabel("Flow rate [µL/min]")
    ax1.set_ylabel("Volume [µL]", color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # ---------------- Right axis: Equilibration time (blue) ----------------
    ax2 = ax1.twinx()
    ax2.plot(x, y, color='blue', linewidth=2)

    # Plot points individually depending on full_collection
    for xi, yi, filled in zip(x, y, full_collection):
        if filled:
            ax2.plot(xi, yi, marker='o', color='blue', markersize=8, markerfacecolor='blue', markeredgecolor='blue')
        else:
            ax2.plot(xi, yi, marker='o', color='blue', markersize=8, markerfacecolor='none', markeredgecolor='blue')

    ax2.set_ylabel("Equilibration time [s]", color='blue')
    ax2.tick_params(axis='y', labelcolor='black')

    ax1.set_xlim(1e-1, 1e3)
    ax1.set_ylim(5e1, 1e4)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax1.grid(True, which="major", ls="-", linewidth=1, alpha=0.8)
    ax1.grid(True, which="minor", ls="-", linewidth=1, alpha=0.2)

    # Make right axis tick positions match left axis
    ax2.set_yticks(ax1.get_yticks())
    ax2.set_ylim(ax1.get_ylim())

    # DISPLACE RIGHT AXIS BY ONE DECADE
    left_min, left_max = ax1.get_ylim()
    ax2.set_ylim(left_min * 10, left_max * 10)

    # Shade region left of x = 20
    ax1.axvspan(ax1.get_xlim()[0], 0.3, color='gray', alpha=0.2)
    ax1.axvline(0.3, color='gray', linestyle='-')

    # Text
    ax1.text(
        0.10, 0.85,
        "Full collection",
        transform=ax1.transAxes,
        fontsize=14,
        color="black",
    )

    fig.tight_layout()
    plt.show()

def plot_varying_Q_varying_c(df, params):
    fig, ax1 = plt.subplots(figsize=(7, 6))

    # Filter valid rows
    valid = df[[
        "full_collection", "time_eq", "Q_in", "V_in",
        "b_last", "c_in"
    ]].replace([np.inf, -np.inf], np.nan).dropna()

    unique_c = sorted(valid["c_in"].unique())
    N = len(unique_c)

    # Create grayscale colors (avoid pure black/white extremes)
    grayscale_colors = [
        str(0.1 + 0.5 * (N - 1 - i) / max(N - 1, 1)) for i in range(N)
    ]
    # These are strings like '0.2', '0.4', which matplotlib interprets as grays

    ax2 = ax1.twinx()

    # Loop over concentration groups
    for i, cval in enumerate(unique_c):

        sub = valid[valid["c_in"] == cval]

        # Unit conversions
        x = sub["Q_in"].values * 1e9 * 60        # flow [uL/min]
        y = sub["time_eq"].values               # t_eq [s]
        yy = (sub["Q_in"] * sub["time_eq"]).values * 1e9  # Volume [uL]

        # Sort each c-group for smooth lines
        idx = np.argsort(x)
        x_sorted = x[idx]
        y_sorted = y[idx]
        yy_sorted = yy[idx]

        color = grayscale_colors[i]

        # ------- LEFT axis: Volume (solid grayscale) -------
        ax1.plot(
            x_sorted, yy_sorted,
            marker="s", linewidth=2,
            color=color,
            linestyle="-",                       # SOLID
        )

        # ------- RIGHT axis: t_eq (dotted grayscale) -------
        ax2.plot(
            x_sorted, y_sorted,
            marker=None, linewidth=2,
            color=color,
            linestyle=":",                       # DOTTED
        )

        # Markers (full vs empty)
        for xi, yi, filled in zip(x, y, sub["full_collection"]):
            if filled:
                ax2.plot(
                    xi, yi, marker='o', markersize=8,
                    markerfacecolor=color,
                    markeredgecolor=color
                )
            else:
                ax2.plot(
                    xi, yi, marker='o', markersize=8,
                    markerfacecolor='none',
                    markeredgecolor=color
                )

    # --- Labels ---
    ax1.set_xlabel("Flow rate [µL/min]")
    ax1.set_ylabel("Volume [µL]", color='black')
    ax2.set_ylabel("Equilibration time [s]", color='black')

    # --- Log scales ---
    ax1.set_xscale("log");  ax1.set_yscale("log")
    ax2.set_yscale("log")

    # --- Limits ---
    ax1.set_xlim(1e-1, 5e3)
    ax1.set_ylim(1e-1, 5e4)
    ax2.set_ylim(1e1, 5e6)
    #ax2.set_yticks(ax1.get_yticks())

    # --- Grid ---
    ax1.grid(True, which="major", ls="-", alpha=0.8)
    ax1.grid(True, which="minor", ls="-", alpha=0.2)

    # --- Shading ---
    ax1.axvspan(ax1.get_xlim()[0], 20, color='gray', alpha=0.2)
    ax1.axvline(20, color='gray')

    # --- Text ---
    ax1.text(
        0.15, 0.90, "Full collection",
        transform=ax1.transAxes, fontsize=14
    )

    fig.tight_layout()
    plt.show()

def plot_varying_Q_varying_c_Da(df):
    fig, ax = plt.subplots(figsize=(7, 6))

    # Filter valid rows
    valid = df[[
        "full_collection", "time_eq", "Q_in", "Da_2", "c_in"
    ]].replace([np.inf, -np.inf], np.nan).dropna()

    unique_c = sorted(valid["c_in"].unique())
    N = len(unique_c)

    # Grayscale colors
    grayscale_colors = [
        str(0.1 + 0.5 * (N - 1 - i) / max(N - 1, 1)) for i in range(N)
    ]

    # Loop over concentration groups
    for i, cval in enumerate(unique_c):

        sub = valid[valid["c_in"] == cval]

        # Unit conversions
        x = sub["Q_in"].values * 1e9 * 60   # µL/min
        Da = sub["Da_2"].values               # Da values

        # Sort for clean lines
        idx = np.argsort(x)
        x_sorted = x[idx]
        Da_sorted = Da[idx]     # ✔ correct

        color = grayscale_colors[i]

        # ---- Da line ----
        ax.plot(
            x_sorted, Da_sorted,
            marker="s",
            linewidth=2,
            color=color,
            linestyle="-",
            label=f"c = {cval}"
        )

        # ---- full / empty markers ----
        for xi, yi, filled in zip(x, Da, sub["full_collection"]):
            if filled:
                ax.plot(
                    xi, yi, marker='o', markersize=8,
                    markerfacecolor=color, markeredgecolor=color
                )
            else:
                ax.plot(
                    xi, yi, marker='o', markersize=8,
                    markerfacecolor='none', markeredgecolor=color
                )

    # Labels and scales
    ax.set_xlabel("Flow rate [µL/min]")
    ax.set_ylabel("Damköhler number (Da)")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlim(1e-1, 1e3)

    ax.grid(True, which="major", ls="-", alpha=0.8)
    ax.grid(True, which="minor", ls="-", alpha=0.2)

    ax.legend(title="Concentration")

    plt.tight_layout()
    plt.show()

def plot_damkohler_varying_c(df):
    fig, ax = plt.subplots(figsize=(7, 6))

    c_labels = ["1 nM", "100 nM", r"10 $\mu$M"]
    # keep only required columns
    valid = df[[
        "Da_2", "Q_in", "c_in"
    ]].replace([np.inf, -np.inf], np.nan).dropna()

    unique_c = sorted(valid["c_in"].unique())
    N = len(unique_c)

    # grayscale colors
    grayscale_colors = [
        str(0.1 + 0.6 * i / max(N - 1, 1)) for i in range(N)
    ]

    grayscale_colors = [
        str(0.1 + 0.5 * (N - 1 - i) / max(N - 1, 1)) for i in range(N)
    ]

    for i, cval in enumerate(unique_c):
        sub = valid[valid["c_in"] == cval]

        x = sub["Q_in"].values * 1e9 * 60   # µL/min
        y = sub["Da_2"].values

        # sort for connected lines
        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]

        ax.plot(
            x, y,
            color=grayscale_colors[i],
            linewidth=2,
            marker="o",
            markersize=6,
            label=c_labels[i]
        )

    # axes, scales, styling
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1e3)

    ax.grid(True, which="major", ls="-", alpha=0.8)
    ax.grid(True, which="minor", ls="-", alpha=0.2)

    ax.set_xlabel("Flow rate [µL/min]")
    ax.set_ylabel("Damköhler number [ ]")
    # ax.set_title("Damköhler number vs flow for varying concentration")

    ax.legend(title="Concentration", loc="best")
    plt.tight_layout()
    plt.show()
