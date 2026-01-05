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
    valid = df[["V","Pe_H","V_in","b_m","H_c","D","Q_in","Da_2","F","k_m", "k_on", "c_in", "time_eq", "tau", "k_off"]].replace([np.inf, -np.inf], np.nan).dropna()

    # Extract arrays
    tau_mt = 1 / valid["k_m"]
    tau_bind = 1 / (valid["k_on"] * valid["c_in"] + valid["k_off"])

    #x = valid["k_m"] * valid["c_in"] * valid["Q_in"]
    #x = valid["k_m"] * valid["Q_in"] * valid["c_in"] * valid["H_c"] / (2*valid["D"])
    # y = (valid["k_on"] * valid["c_in"] + valid["k_off"]).values
    #y = valid["k_off"] / valid["k_on"]
    # z = valid["time_eq"]
    z = valid["Da_2"]

    #x = valid["k_m"] / valid["tau"]
    #y = (valid["k_on"] * valid["c_in"]).values
    #

    # flux based
    t_pulse = valid["V_in"] / valid["Q_in"]
    #x = valid["k_m"] * valid["c_in"]    # J_D = m/s * mol/m3 = mol/m^2*s
    #y = (valid["k_on"] * valid["c_in"] * valid["b_m"]).values   # J_R = 1/(Ms) * M * mol/m2 = mol/m2*s

    # transport based
    #x = valid["k_m"]
    x = valid["Q_in"] / valid["V"]
    #y = (valid["k_on"] * valid["c_in"] + valid["k_off"]).values
    y = valid["k_on"] * valid["c_in"] + valid["k_off"]
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
    ax.set_xlabel("Transport rate [1/s]")
    ax.set_ylabel("Binding rate [1/s]")
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
    valid = df[["b_eq","W_c","S","V_in","b_m","H_c","D","Q_in","Da","F","k_m", "k_on", "c_in", "time_eq", "tau", "k_off"]].replace([np.inf, -np.inf], np.nan).dropna()

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
    moles_mt = valid["c_in"] * valid["k_m"]
    #moles_mt = valid["c_in"] * valid["k_m"] * valid["S"]
    moles_bind = (valid["k_on"] * valid["c_in"] + valid["k_off"])
    #moles_bind =    # [1/s]

    #moles_supply = valid["c_in"] * valid["Q_in"]    # mol/s
    #moles_bind =    # [1/s]

    moles_supply = (valid["c_in"] * valid["Q_in"]) / ((valid["H_c"]) * valid["W_c"])    # mol/s

    # moles_mt = valid["c_in"] * valid["k_m"] * valid["S"]
    # moles_bind =  (valid["k_on"] * valid["c_in"] + valid["k_off"])
    # moles_supply = (valid["c_in"] * valid["Q_in"])    # mol/s


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

    cols = ["Pe_H","W_c","b_eq","S","V","L_s","V_in","b_m","H_c","D","Q_in","Da","F","k_m", "k_on", "c_in", "time_eq", "tau", "k_off"]

    invalid_mask = (
            df[cols].isna() | df[cols].isin([np.inf, -np.inf])
    ).any(axis=1)

    invalid_df = (
        df.loc[invalid_mask, cols]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(how="all")
    )

    valid = df[["Pe_H","W_c","b_eq","S","V","L_s","V_in","b_m","H_c","D","Q_in","Da","F","k_m", "k_on", "c_in", "time_eq", "tau", "k_off"]].replace([np.inf, -np.inf], np.nan).dropna()
    invalid = invalid_df[["Pe_H","W_c","b_eq","S","V","L_s","V_in","b_m","H_c","D","Q_in","Da","F","k_m", "k_on", "c_in", "time_eq", "tau", "k_off"]].replace([np.inf, -np.inf], np.nan).dropna()

    # with moles
    moles_mt = valid["k_m"] * valid["c_in"] * ((valid["H_c"]) * valid["W_c"]) # mol/m3 * m/s = mol/(m2/s) *
    moles_bind =  (valid["k_on"] * valid["c_in"] + valid["k_off"])
    moles_supply = (valid["c_in"] * valid["Q_in"])     # mol/s
    moles_eff = np.maximum(moles_mt,moles_supply)

    test = ((valid["k_m"]**2) * (valid["c_in"]**2) * valid["V"])  / ((valid["Q_in"]))


    tau_transport = valid["Q_in"] / (valid["V"])
    c_eq = valid["k_m"] * valid["c_in"] * valid["V"] / valid["Q_in"]
    k_D = valid["k_off"] / valid["k_on"]
    c_crit = k_D
    tau_bind = valid["k_on"] * valid["c_in"] + valid["k_off"]
    #tau_bind = valid["k_on"] * valid["c_in"] + valid["k_off"]
    tau_bind_dilute = valid["k_off"]


    mask = valid["c_in"] <= k_D

    #Da = (k_on * b_m) / k_m                 # definition like in Squires

    t_crit = - valid["V"] / valid["Q_in"] * np.log(1 - (c_crit / c_eq))
    eff1 = valid["F"] / valid["Pe_H"]

    moles_eff = moles_supply * eff1

    z = valid["time_eq"]
    x = (valid["Q_in"]) * 60 * 1e9
    x_invalid = (invalid["Q_in"]) * 60 * 1e9
    #x = (valid["k_m"]) / valid["L_s"]
    #x = valid["c_in"] * valid["Q_in"] * eff1
    y = (valid["Q_in"]) * valid["time_eq"] * 1e9
    y_invalid = (invalid["Q_in"]) * invalid["time_eq"] * 1e9

    moles_mt = valid["c_in"] * valid["k_m"]
    moles_supply = valid["c_in"] * valid["Q_in"]
    moles_bind = (valid["k_on"] * valid["c_in"] + valid["k_off"])




    y[mask] = tau_bind_dilute[mask]

    # Ensure z > 0 for log scaling
    z = np.clip(z, np.nanmin(z[z > 0]), None)

    # --- Log-scaled color normalization ---
    norm = LogNorm(vmin=z.min(), vmax=z.max())

    tau_supply = valid["Q_in"] / valid["V"]

    supply_total = moles_eff

    # Ensure z > 0 for log scaling
    supply_total = np.clip(supply_total, np.nanmin(supply_total[supply_total > 0]), None)

    # --- Log-scaled color normalization ---
    norm_transport = LogNorm(vmin=supply_total.min(), vmax=supply_total.max())

    # # 2D scatter plot
    # plt.figure(figsize=(7, 6))
    # sc = plt.scatter(
    #     tau_transport, tau_supply, #tau_mt, tau_supply,
    #     c=moles_eff, #c=tau_eff,
    #     cmap="inferno",
    #     norm = norm_transport,
    #     s=50,
    #     alpha=0.8
    # )
    #
    # plt.xscale("log")
    # plt.yscale("log")
    # #plt.xlabel("tau_mt [s] (transport timescale)")
    # #plt.ylabel("tau_supply [s] (supply timescale)")
    # plt.xlabel("k_m / L_s [1/s] (transport timescale)")
    # plt.ylabel("Q / V [1/s] (supply timescale)")
    # plt.title("transport time scales")
    # plt.colorbar(sc, label="time_eq [s]")
    # plt.grid(True, which="both", ls="--", alpha=0.5)

    # --- Add diagonal y=x line ---
    lims = [
        min(plt.xlim()[0], plt.ylim()[0]),
        max(plt.xlim()[1], plt.ylim()[1])
    ]
    plt.plot(lims, lims, '--', linewidth=1, alpha=0.7)


    plt.show()

    fig, ax = plt.subplots(figsize=(7, 6))
    # Scatter plot
    # Normal valid points (non-dilute)
    normal_mask = ~mask
    sc = ax.scatter(x[normal_mask], y[normal_mask], c=z[normal_mask], cmap="turbo",
                    norm=LogNorm(vmin=z.min(), vmax=z.max()), s=40, edgecolor="none", label="valid")

    # Dilute points highlighted (different marker/edge)
    ax.scatter(x_invalid, y_invalid, facecolors='none', edgecolors='black', s=50, label="dilute")


    # Plot invalid points as black
    # Compute same quantities as for valid points
    c_eq_invalid = invalid_df["k_m"] * invalid_df["c_in"] * invalid_df["V"] / invalid_df["Q_in"]
    k_D_invalid = invalid_df["k_off"] / invalid_df["k_on"]
    c_crit_invalid = k_D_invalid
    tau_bind_invalid = invalid_df["k_on"] * invalid_df["c_in"] + invalid_df["k_off"]
    tau_bind_dilute_invalid = invalid_df["k_off"]

    mask_invalid = invalid_df["c_in"] <= k_D_invalid

    tau_transport_invalid = (invalid_df["Q_in"] / invalid_df["V"])
    #tau_transport_invalid = invalid_df["k_m"]  / invalid_df["L_s"]
    x_invalid = tau_transport_invalid
    z_invalid = invalid_df["time_eq"]
    y_invalid = tau_bind_invalid.copy()
    y_invalid[mask_invalid] = tau_bind_dilute_invalid[mask_invalid]
    #ax.scatter(x_invalid, y_invalid, color="black", s=10, alpha=0.3, label="invalid")

    # Colorbar (log scale)
    cbar = plt.colorbar(sc, ax=ax)
    #cbar.set_label("Equilibrium time [s] (log scale)")
    cbar.set_label("Eq. time [s]")

    # Axis scaling and grid
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", ls="-", linewidth=1, alpha=0.8)
    ax.grid(True, which="minor", ls="-", linewidth=1, alpha=0.2)

    #ax.scatter(0.0067, 0.0052, color = "black", s = 20)

    # Labels and title
    #ax.set_xlabel("tau_eff [1/s] ")
    #ax.set_ylabel("tau_b [1/s]")
    ax.set_xlabel("tau_transport (Q/V) [1/s]")
    ax.set_ylabel("tau_binding (k_on*c_in + k_off) [1/s]")
    ax.set_title("Equilibrium time")

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

    css = valid["k_m"] / valid["L_s"]
    test = valid["Q_in"] / valid["V"]
    tau_transport = css + test
    #z = test / css
    tau_bind = valid["k_on"] * valid["c_in"] + valid["k_off"]
    tau_bind_dilute = valid["k_off"]
    x = valid["Q_in"]
    z = valid["time_eq"]

    # transport based
    #x = valid["Q_in"]*60*1e9
    #z = valid["time_eq"]

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
    cbar.set_label("Equilibrium time [s]")
    #cbar.set_label("Minimum volume requied [uL]")

    # Axis scaling and grid
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", ls="-", linewidth=1, alpha=0.8)
    ax.grid(True, which="minor", ls="-", linewidth=1, alpha=0.2)

    # Labels and title
    ax.set_xlabel("Q_in [uL/min]")
    ax.set_ylabel("Minimum volume required [uL]")
    ax.set_title("Volume requirement")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

from matplotlib.colors import LogNorm

def plot_km_Q(df, save_path=None):


    # Keep only valid numeric rows
    valid = df[["W_c","Pe_H","b_eq","S","V","L_s","V_in","b_m","H_c","D","Q_in","Da","F","k_m", "k_on", "c_in", "time_eq", "tau", "k_off"]].replace([np.inf, -np.inf], np.nan).dropna()

    moles_mt = valid["c_in"] * valid["k_m"]
    moles_supply = valid["c_in"] * valid["Q_in"]
    moles_bind = (valid["k_on"] * valid["c_in"] + valid["k_off"])

    eff1 = valid["F"] / valid["Pe_H"]
    eff2 = moles_mt / moles_supply

    eff3 = np.minimum(eff2,1)

    z = valid["time_eq"]
    x = valid["Q_in"]
    #x = valid["c_in"] * valid["k_m"] * valid["S"]
    y = valid["k_m"]


    #x = eff1
    #y = eff2

    #x = moles_mt
    #y = moles_bind



    # Ensure z > 0 for log scaling
    z = np.clip(z, np.nanmin(z[z > 0]), None)

    norm = LogNorm(vmin=z.min(), vmax=z.max())

    # 2D scatter plot    fig, ax = plt.subplots(figsize=(7, 6))
    #     # Scatter plot
    #     sc = ax.scatter(x, y, c=z, cmap="turbo", norm=norm, s=40, edgecolor="none")
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        x, y, #tau_mt, tau_supply,
        c=z,
        cmap="viridis",
        norm=norm,
        s=50,
        alpha=0.8
    )


    # Colorbar (log scale)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Eq time [s]")

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Q_in [m3/s]")
    plt.ylabel("k_m [m/s]")
    plt.title("")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_V_min_vs_Q_in_new(df, save_path=None):

    # Keep only valid numeric rows
    valid = df[[
        "V","L_s","V_in","b_m","H_c","D","Q_in","Da","F",
        "k_m","k_on","c_in","time_eq","tau","k_off"
    ]].replace([np.inf, -np.inf], np.nan).dropna()

    # Extract arrays
    x = valid["Q_in"] * 60 * 1e9
    y = valid["time_eq"] * valid["Q_in"] * 1e9
    z = valid["time_eq"]

    # Ensure z > 0 (required for logs)
    z = np.clip(z, np.nanmin(z[z > 0]), None)

    # ---------------- RBF INTERPOLATION (log–log space) ----------------

    logx = np.log10(x)
    logy = np.log10(y)
    logz = np.log10(z)

    Xg, Yg = np.meshgrid(
        np.logspace(np.log10(x.min()), np.log10(x.max()), 400),
        np.logspace(np.log10(y.min()), np.log10(y.max()), 400)
    )

    Xg_log = np.log10(Xg)
    Yg_log = np.log10(Yg)

    rbf = Rbf(logx, logy, logz,
              function='linear',
              smooth=0.05)

    Zg_log = rbf(Xg_log, Yg_log)
    Zg = 10**Zg_log

    # -------------------------------------------------------------------

    # Determine contour levels: decades of z
    zmin_dec = np.floor(np.log10(z.min()))
    zmax_dec = np.ceil(np.log10(z.max()))
    levels = 10**np.arange(zmin_dec, zmax_dec + 1)

    fig, ax = plt.subplots(figsize=(7, 6))

    # ---------------- SHADED CONTOURF ----------------
    contour = ax.contourf(
        Xg, Yg, Zg,
        levels=levels,
        cmap="plasma",
        norm=LogNorm(),
        alpha=0.6
    )
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label("Equilibrium time [s] (interpolated)")
    # -------------------------------------------------

    # ---------------- CONTOUR LINES ----------------
    contour_lines = ax.contour(
        Xg, Yg, Zg,
        levels=levels,
        colors="black",
        linewidths=0.8
    )

    ax.clabel(
        contour_lines,
        fmt=lambda v: f"{v:.0e}",
        fontsize=8,
        inline=False
    )
    # -------------------------------------------------

    # ---------------- SCATTER OF RAW DATA ----------------
    # sc = ax.scatter(
    #     x, y, c=z,
    #     cmap="turbo",
    #     norm=LogNorm(vmin=z.min(), vmax=z.max()),
    #     s=40,
    #     edgecolor="black", linewidth=0.4
    # )
    # # ------------------------------------------------------

    # ax.set_aspect('equal', adjustable='box')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", ls="-", linewidth=1, alpha=0.8)
    ax.grid(True, which="minor", ls="-", linewidth=1, alpha=0.2)

    ax.set_xlim(1e-3, 1e1)
    ax.set_ylim(1e-3, 1e1)

    ax.set_xlabel("Flow rate [uL/min]")
    ax.set_ylabel("Required volume [uL]")
    ax.set_title("Volume required")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

    return Xg, Yg, Zg


def t_eq_new(df, save_path=None):

    # Keep only valid numeric rows
    valid = df[[
        "V","L_s","V_in","b_m","H_c","D","Q_in","Da","Da_2","F",
        "k_m","k_on","c_in","time_eq","tau","k_off"
    ]].replace([np.inf, -np.inf], np.nan).dropna()

    # Extract arrays
    x = valid["Q_in"] / valid["V"]
    y = valid["k_on"] * valid["c_in"] + valid["k_off"]
    z = valid["time_eq"].values

    # Ensure z > 0 (required for logs)
    z = np.clip(z, np.nanmin(z[z > 0]), None)

    # ---------------- RBF INTERPOLATION (log–log space) ----------------

    logx = np.log10(x)
    logy = np.log10(y)
    logz = np.log10(z)

    Xg, Yg = np.meshgrid(
        np.logspace(np.log10(x.min()), np.log10(x.max()), 25),
        np.logspace(np.log10(y.min()), np.log10(y.max()), 25)
    )

    Xg_log = np.log10(Xg)
    Yg_log = np.log10(Yg)

    rbf = Rbf(logx, logy, logz,
              function='linear',
              smooth=0.10)

    Zg_log = rbf(Xg_log, Yg_log)
    Zg = 10**Zg_log

    # -------------------------------------------------------------------

    # Determine contour levels: decades of z
    zmin_dec = np.floor(np.log10(z.min()))
    zmax_dec = np.ceil(np.log10(z.max()))
    levels = 10**np.arange(zmin_dec, zmax_dec + 1)
    z_color_max = 1e4
    levels_plot = levels[levels <= z_color_max]

    fig, ax = plt.subplots(figsize=(7, 6))

    # ---------------- SHADED CONTOURF ----------------
    contour = ax.contourf(
        Xg, Yg, Zg,
        levels=levels_plot,
        cmap="Blues",
        norm=LogNorm(vmin=levels[0], vmax=z_color_max),
        alpha=0.9
    )
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label("Equilibrium time [s] (interpolated)")
    # -------------------------------------------------

    # ---------------- CONTOUR LINES ----------------
    contour_lines = ax.contour(
        Xg, Yg, Zg,
        levels=levels_plot,
        colors="black",
        linewidths=0.8
    )

    ax.clabel(
        contour_lines,
        fmt=lambda v: f"{v:.0e}",
        fontsize=8,
        inline=False
    )
    # -------------------------------------------------

    # ---------------- SCATTER OF RAW DATA ----------------
    # sc = ax.scatter(
    #     x, y, c=z,
    #     cmap="turbo",
    #     norm=LogNorm(vmin=z.min(), vmax=z.max()),
    #     s=40,
    #     edgecolor="black", linewidth=0.4
    # )
    # # ------------------------------------------------------

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", ls="-", linewidth=1, alpha=0.8)
    ax.grid(True, which="minor", ls="-", linewidth=1, alpha=0.2)


    ax.set_xlim(1e-3, 1e1)
    ax.set_ylim(1e-3, 1e1)

    ax.set_xlabel("Transport rate [1/s]")
    ax.set_ylabel("Binding rate [1/s]")
    #ax.set_title("Equilibrium time")

    plt.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, format="svg", bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return Xg, Yg, Zg

def plot_t_eq_two_axis_scatter_new(df, save_path=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import os

    # Columns required for the final plot
    cols = ["L_s","Pe_H","F","k_m","k_on","k_off","c_in","c_eff","Q_in","V","time_eq"]

    # Remove NaN/inf rows
    valid = (
        df[cols]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    # ---- Filter to only points where F = Pe_H ----
    #equality_mask = valid["F"] == valid["Pe_H"]
    #valid = valid[equality_mask]

    # If nothing left, exit cleanly
    if len(valid) == 0:
        print("No points satisfy F = Pe_H")
        return

    # Compute quantities actually used in the final figure
    x = valid["Q_in"] / valid["V"]                       # tau_transport
    y = valid["k_on"] * valid["c_in"] + valid["k_off"]   # tau_bind
    z = valid["time_eq"]

    # Ensure z > 0 for log color scale
    z = np.clip(z, np.nanmin(z[z > 0]), None)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(7, 6))

    sc = ax.scatter(
        x, y,
        c=z,
        cmap="turbo",
        norm=LogNorm(vmin=z.min(), vmax=z.max()),
        s=40,
        edgecolor="none"
    )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Eq. time [s]")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.grid(True, which="major", ls="-", linewidth=1, alpha=0.8)
    ax.grid(True, which="minor", ls="-", linewidth=1, alpha=0.2)

    ax.set_xlabel("tau_transport (Q/V) [1/s]")
    ax.set_ylabel("tau_binding (k_on*c_in + k_off) [1/s]")
    ax.set_title("Equilibrium Time (Only points where F = Pe_H)")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()
