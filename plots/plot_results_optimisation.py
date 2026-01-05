import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import os

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
    ax1.set_ylim(5e1, 1e3)

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

def plot_varying_Q_varying_c(df, params, save_path=None):
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
            linestyle="-"
        )

        #ax1.plot(
        #    x_comsol_1nM, y_comsol_volume_1nM,
        #    marker="+", linewidth=2,  markersize=10, markeredgewidth=2.5,
        #    color="royalblue",
        #    linestyle="None"
        #)

        ax1.plot(
            x_comsol_100nM, y_comsol_volume_100nM,
            marker="+", linewidth=2, markersize=11, markeredgewidth=2.5,
            color="cornflowerblue",
            linestyle="None"
        )

        #ax1.plot(
        #    x_comsol_10uM, y_comsol_volume_10uM,
        #    marker="+", linewidth=2, markersize=10, markeredgewidth=2.5,
        #    color="royalblue",
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
        #    color="red",
        #    linestyle="None",
        #)

        ax2.plot(
            x_comsol_100nM, y_comsol_time_100nM,
            marker="x", linewidth=2, markersize = 8, markeredgewidth=2.5,
            color="mediumseagreen",
            linestyle="None",
        )

        #ax2.plot(
        #    x_comsol_10uM, y_comsol_time_10uM,
        #    marker="x", linewidth=2, markersize = 7, markeredgewidth=2.5,
        #    color="red",
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

    ax1.set_xlim(1e-1, 4e2)
    ax1.set_ylim(1e0, 1e4)
    ax2.set_ylim(1e2, 1e6)

    # --- Grid ---
    ax1.grid(True, which="major", ls="-", alpha=0.8)
    ax1.grid(True, which="minor", ls="-", alpha=0.2)

    # --- Shading ---
    # ax1.axvspan(ax1.get_xlim()[0], 20, color='gray', alpha=0.2)
    # ax1.axvline(20, color='gray')
    #
    # ax1.axvspan(ax1.get_xlim()[0], 20, color='gray', alpha=0.2)
    # ax1.axvline(20, color='gray')
    # #ax1.legend(title="Concentration", loc='best')

    # --- Text ---
    #ax1.text(
    #    0.15, 0.90, "Full collection",
    #    transform=ax1.transAxes, fontsize=14
    #)

    fig.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, format="svg", bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_varying_Q_varying_c_Da(df, save_path = None):
    fig, ax = plt.subplots(figsize=(7, 6))

    # Filter valid rows
    valid = df[[
        "full_collection", "time_eq", "Q_in", "Da_2", "c_in"
    ]].replace([np.inf, -np.inf], np.nan).dropna()

    unique_c = sorted(valid["c_in"].unique(), reverse=True)
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


    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, format="svg", bbox_inches="tight")
        plt.close()
    else:
        plt.show()


    plt.show()

def plot_damkohler_varying_c(df, save_path = None):
    fig, ax = plt.subplots(figsize=(7, 6))

    c_labels = [r"10 $\mu$M","100 nM", "1 nM"]
    # keep only required columns
    valid = df[[
        "Da_2", "Q_in", "c_in"
    ]].replace([np.inf, -np.inf], np.nan).dropna()

    unique_c = sorted(valid["c_in"].unique(), reverse=True)
    N = len(unique_c)

    # grayscale colors
    grayscale_colors = [
        str(0.1 + 0.6 * i / max(N - 1, 1)) for i in range(N)
    ]

    grayscale_colors = [
        str(0.1 + 0.5 * i / max(N - 1, 1)) for i in range(N)
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
    ax.set_ylim(1e-2, 1e3)
    ax.set_xlim(1e-1, 1e3)
    ax.legend(title="Concentration")

    ax.grid(True, which="major", ls="-", alpha=0.8)
    ax.grid(True, which="minor", ls="-", alpha=0.2)

    ax.set_xlabel("Flow rate [µL/min]")
    ax.set_ylabel("Damköhler number [ ]")
    # ax.set_title("Damköhler number vs flow for varying concentration")

    ax.legend(title="Concentration", loc="best")
    plt.tight_layout()
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, format="svg", bbox_inches="tight")
        plt.close()
    else:
        plt.show()


    plt.show()


def plot_varying_Q_varying_c_from_csv(df, csv_files, save_path=None):
    """
    Left y-axis: sample volume from CSV contour files (solid lines)
    Right y-axis: equilibration time from original simulation df (dotted lines with markers)

    df: original simulation dataframe containing columns Q_in, time_eq, full_collection, c_in
    csv_files: list of CSV file paths, one per concentration (highest first)
    """
    fig, ax1 = plt.subplots(figsize=(7, 6))
    ax2 = ax1.twinx()

    N = len(csv_files)

    # Grayscale colors
    grayscale_colors = [str(0.1 + 0.5 * i / max(N - 1, 1)) for i in range(N)]

    # --- LEFT axis: CSV contour lines ---
    for i, csv_file in enumerate(csv_files):
        grayscale_temp = grayscale_colors[::-1]  # now darkest first
        color = grayscale_temp[i]

        df_csv = pd.read_csv(csv_file)
        Q_in = df_csv['Q_in_uL_min'].values
        V_in = df_csv['V_in_uL'].values

        #if i == 2:
        #    ax1.plot(Q_in, V_in,
        #             marker='s', linewidth=2,
        #             color=color, linestyle='-',
        #             label=f'c={i}')  # optional label


    # --- RIGHT axis: original simulation time_eq points ---
    # Filter valid rows
    valid = df[['Q_in', 'time_eq', 'full_collection', 'c_in']].replace([np.inf, -np.inf], np.nan).dropna()
    unique_c = sorted(valid['c_in'].unique(), reverse=True)

    for i, cval in enumerate(unique_c):
        color = grayscale_colors[i % N]  # match CSV color

        sub = valid[valid['c_in'] == cval]
        x = sub['Q_in'].values * 1e9 * 60
        y = sub['time_eq'].values

        idx = np.argsort(x)
        x_sorted = x[idx]
        y_sorted = y[idx]
        yy_sorted = x_sorted/60 * y_sorted

        if False: #i != 0:
            ax1.plot(x_sorted, yy_sorted,
                     marker='s', linewidth=2,
                     color=color, linestyle='-',
                     label=f'c={i}')  # optional label

        ax2.plot(x_sorted, y_sorted,
                 marker=None, linewidth=2,
                 color=color, linestyle=':')  # dotted line

        # Markers for full vs empty collection
        for xi, yi, filled in zip(x, y, sub['full_collection']):
            if filled:
                ax2.plot(xi, yi, marker='o', markersize=8,
                         markerfacecolor=color,
                         markeredgecolor=color)
            else:
                ax2.plot(xi, yi, marker='o', markersize=8,
                         markerfacecolor='none',
                         markeredgecolor=color)

    # --- Labels ---
    ax1.set_xlabel("Flow rate [µL/min]")
    ax1.set_ylabel("Sample volume [µL]", color='black')
    ax2.set_ylabel("Equilibration time [s]", color='black')

    # --- Log scales ---
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax2.set_yscale("log")

    # --- Axis limits ---
    ax1.set_xlim(1e-1, 1e3)
    ax1.set_ylim(1e-1, 5e4)
    ax2.set_ylim(1e1, 5e6)

    # --- Grid ---
    ax1.grid(True, which="major", ls="-", alpha=0.8)
    ax1.grid(True, which="minor", ls="-", alpha=0.2)

    # --- Shading and reference line ---
    ax1.axvspan(ax1.get_xlim()[0], 20, color='gray', alpha=0.2)
    ax1.axvline(20, color='gray')
    ax1.text(0.15, 0.90, "Full collection", transform=ax1.transAxes, fontsize=14)

    fig.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, format="svg", bbox_inches="tight")
        plt.close()
    else:
        plt.show()