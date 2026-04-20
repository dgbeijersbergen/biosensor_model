import matplotlib
from matplotlib.lines import lineStyles

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import os
from scipy.interpolate import Rbf
from scipy.ndimage import gaussian_filter
import matplotlib.tri as tri

def plot_site_occupancy_interp(df, params, grid_size, save_path=None, save_contour_csv=None):
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
    contours = ax.tricontour(triang, z, levels=[95], colors="black", linewidths=2)

    # Extract vertices
    contour_vertices = []

    for seg_group in contours.allsegs:  # allsegs is a list of lists: one per level
        for seg in seg_group:  # each seg is an (N,2) array of vertices
            contour_vertices.append(seg)

    # Combine into single array
    contour_vertices = np.vstack(contour_vertices)

    x_c = contour_vertices[:, 0]
    y_c = contour_vertices[:, 1]


    idx = np.argsort(x_c)
    x_c, y_c = x_c[idx], y_c[idx]

    # Target x-values where contour is sampled
    k = 20
    Q_in_uL_min = np.logspace(-1, 3, k)

    all_contour_samples = []


    # Interpolate at requested Q_in values
    mask = (Q_in_uL_min >= x_c.min()) & (Q_in_uL_min <= x_c.max())

    V_in_uL = np.full_like(Q_in_uL_min, np.nan, dtype=float)
    V_in_uL[mask] = np.interp(Q_in_uL_min[mask], x_c, y_c)

    # Store results
    df_sample = pd.DataFrame({
        "Q_in_uL_min": Q_in_uL_min,
        "V_in_uL": V_in_uL
    })

    all_contour_samples.append(df_sample)

    # Save contour CSV if requested
    if save_contour_csv is not None:
        save_dir = os.path.dirname(save_contour_csv)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        df_sample.to_csv(save_contour_csv, index=False)

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
    plt.xlim(1e-1, 4e2)  # set x-axis limits

    # Labels and title

    ax.set_xlabel("Flow rate [µL/min]")
    ax.set_ylabel("Sample volume [µL]")

    #ax.set_title("Capture percentage (interpolated, griddata)")

    plt.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, format="svg", bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_Q_optimisation(df, params, save_path=None):
    fig, ax1 = plt.subplots(figsize=(7, 6))

    # Filter valid rows
    valid = df[[
        "full_collection", "time_eq", "Q_in", "V_in", "V_eq",
        "b_last", "c_in"
    ]].replace([np.inf, -np.inf], np.nan).dropna()

    unique_c = sorted(valid["c_in"].unique(), reverse=True)
    N = len(unique_c)

    # Create grayscale colors (avoid pure black/white extremes)
    grayscale_colors = [
        str(0.1 + 0.5 * i / max(N - 1, 1)) for i in range(N)
    ]

    ax2 = ax1.twinx()

    labels = ["38 nM", "0.38 nM"]

    # Loop over concentration groups
    for i, cval in enumerate(unique_c):

        sub = valid[valid["c_in"] == cval]

        # Unit conversions
        x = sub["Q_in"].values * 1e9 * 60        # flow [uL/min]
        y = sub["time_eq"].values               # t_eq [s]
        #yy = sub["V_eq"].values*1e9
        yy = (sub["Q_in"] * sub["time_eq"]).values * 1e9  # Volume [uL]

        # Sort each c-group for smooth lines
        idx = np.argsort(x)
        x_sorted = x[idx]
        y_sorted = y[idx]
        yy_sorted = yy[idx]

        color = grayscale_colors[i]

        x_comsol_1nM = np.array([0.1,0.28, 0.77, 2.2,6, 16.7 , 46.7, 129, 1003])
        y_comsol_volume_1nM = np.array([67.7, 73.3, 91.0, 180, 449, 1200, 3300, 9082, 70207])
        y_comsol_time_1nM = np.array([40650, 15800, 7050, 5015, 4501, 4320, 4240, 4220, 4200])

        x_comsol_100nM = np.array([2.2, 6.0, 16.7, 46.7, 129, 1003])
        y_comsol_volume_100nM = np.array([97.7, 134.7, 252.9, 588, 1481, 10297])
        y_comsol_time_100nM = np.array([2723, 1350, 910, 756, 688, 616])

        x_comsol_10uM = np.array([2.2, 6.0, 16.7, 46.7, 129, 1003])
        y_comsol_time_10uM = np.array([991, 412, 160, 66, 34.5, 16.5])
        y_comsol_volume_10uM = np.array([35.5642, 41.1068, 44.4646, 51.3704, 74.2505,275.8143])

        # ------- LEFT axis: Volume (solid grayscale) -------
        ax1.plot(
            x_sorted, yy_sorted,
            marker="s", linewidth=2,
            color=color,
            linestyle="-",
            label=f"{cval*1e-3:.1e} M"
        )

        #ax1.plot(
        #    x_comsol_1nM, y_comsol_volume_1nM,
        #    marker="+", linewidth=2,  markersize=10, markeredgewidth=2.5,
        #    color="cornflowerblue",
        #    linestyle="None"
        #)

        #ax1.plot(
        #    x_comsol_100nM, y_comsol_volume_100nM,
        #    marker="+", linewidth=2, markersize=11, markeredgewidth=2.5,
        #    color="cornflowerblue",
        #    linestyle="None"
        #)

        #ax1.plot(
        #    x_comsol_10uM, y_comsol_volume_10uM,
        #    marker="+", linewidth=2, markersize=10, markeredgewidth=2.5,
        #    color="cornflowerblue",
        #    linestyle="None"
        #)


        # ------- RIGHT axis: t_eq (dotted grayscale) -------

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


        ax2.plot(
            x_sorted, y_sorted,
            marker=None, linewidth=2,
            color=color,
            linestyle=":",
        )

        #ax2.plot(
        #    x_comsol_1nM, y_comsol_time_1nM,
        #    marker="x", linewidth=2, markersize = 7,markeredgewidth=2.5,
        #    color="mediumseagreen",
        #    linestyle="None",
        #)

        #ax2.plot(
        #    x_comsol_100nM, y_comsol_time_100nM,
        #    marker="x", linewidth=2, markersize = 8, markeredgewidth=2.5,
        #    color="mediumseagreen",
        #    linestyle="None",
        #)

        #ax2.plot(
        #    x_comsol_10uM, y_comsol_time_10uM,
        #    marker="x", linewidth=2, markersize = 7, markeredgewidth=2.5,
        #    color="mediumseagreen",
        #    linestyle="None",
        #)



    # --- Labels ---
    ax1.set_xlabel("Flow rate [µL/min]")
    ax1.set_ylabel("Volume [µL]", color='black')
    ax2.set_ylabel("Equilibration time [s]", color='black')

    # --- Log scales ---
    ax1.set_xscale("log");  ax1.set_yscale("log")
    ax2.set_yscale("log")

    # --- Limits ---
    # ax1.set_xlim(1e-1, 1e3)
    # ax1.set_ylim(1e-1, 5e4)
    # ax2.set_ylim(1e1, 5e6)
    # #ax2.set_yticks(ax1.get_yticks())

    ax1.set_xlim(1e-1, 1e1)
    ax1.set_ylim(1e0, 5e2)
    ax2.set_ylim(1e2, 5e4)

    # --- Grid ---
    ax1.grid(True, which="major", ls="-", alpha=0.8)
    ax1.grid(True, which="minor", ls="-", alpha=0.2)

    # --- Shading ---
    ax1.axvspan(ax1.get_xlim()[0], 0.27, color='gray', alpha=0.2)
    ax1.axvline(0.27, color='gray')

    ax1.axvspan(ax1.get_xlim()[0], 0.27, color='gray', alpha=0.2)
    ax1.axvline(0.27, color='gray')
    ax1.legend(title="Concentration", loc='best')

    ax1.axvline(0.5, color='black', linestyle='dashed')

    # --- Text ---
    ax1.text(
        0.03, 0.90, r"Complete " "\n" "delivery",
        transform=ax1.transAxes, fontsize=12
    )

    fig.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, format="svg", bbox_inches="tight")
        plt.close()
    else:
        plt.show()