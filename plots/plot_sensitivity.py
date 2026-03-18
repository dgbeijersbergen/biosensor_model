"""
plot_sensitivity.py
Plotting functions for the OAT sensitivity analysis.

Produces:
  1. Individual sweep plots  – time_eq vs parameter value (one subplot per param)
  2. Tornado plot            – sensitivity rank by effect on time_eq
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _finite(series):
    """Return series with inf replaced by NaN so they are excluded from plots."""
    return series.replace([np.inf, -np.inf], np.nan)


def _baseline_time_eq(df, attr):
    """
    Estimate the baseline time_eq for a given parameter by interpolating at
    relative_value == 1  (i.e. the baseline point).
    Falls back to the closest available point.
    """
    sub = df[df["param_attr"] == attr].copy()
    sub["time_eq"] = _finite(sub["time_eq"])
    sub = sub.dropna(subset=["time_eq"])
    if sub.empty:
        return np.nan
    # find point closest to relative_value = 1
    idx = (sub["relative_value"] - 1.0).abs().idxmin()
    return sub.loc[idx, "time_eq"]


# ---------------------------------------------------------------------------
# 1. Individual sweep plots
# ---------------------------------------------------------------------------

def plot_sensitivity_sweeps(df: pd.DataFrame, baseline, save_path: str = "sensitivity_sweeps.png"):
    """
    One subplot per parameter showing time_eq vs parameter value.
    The baseline value is marked with a vertical dashed line.
    Y-axis limits are fixed globally across all subplots so that missing
    equilibrium points (inf) do not distort the scale.
    """
    params = df["param_attr"].unique()
    n = len(params)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    # ── global y limits from all finite time_eq values ──
    all_t = _finite(df["time_eq"])
    finite_t = all_t.dropna()
    if len(finite_t) > 0:
        y_lo = 10 ** np.floor(np.log10(finite_t.min()))
        y_hi = 10 ** np.ceil( np.log10(finite_t.max()))
    else:
        y_lo, y_hi = 1e0, 1e6

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for ax, attr in zip(axes, params):
        sub = df[df["param_attr"] == attr].copy()
        sub = sub.sort_values("param_value")
        sub["time_eq"] = _finite(sub["time_eq"])

        label    = sub["param_label"].iloc[0]
        unit     = sub["param_unit"].iloc[0]
        base_val = sub["baseline_value"].iloc[0]

        # ── main line ──
        ax.plot(
            sub["param_value"],
            sub["time_eq"],
            color="steelblue",
            linewidth=2,
            marker="o",
            markersize=4,
            zorder=3,
        )

        # ── baseline marker ──
        ax.axvline(base_val, color="tomato", linestyle="--", linewidth=1.5,
                   label=f"baseline = {base_val:.2e}")

        # ── annotation for points that did not reach equilibrium ──
        n_inf = sub["time_eq"].isna().sum()
        if n_inf > 0:
            ax.text(
                0.97, 0.05,
                f"{n_inf} pt(s) did not reach eq.",
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=7, color="grey",
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel(f"{label}  [{unit}]", fontsize=9)
        ax.set_ylabel(r"$t_{eq}$ [s]", fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("OAT Sensitivity Analysis – Equilibration time", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# 2. Tornado plot
# ---------------------------------------------------------------------------

def plot_tornado(df: pd.DataFrame, baseline, save_path: str = "sensitivity_tornado.png"):
    """
    Tornado plot ranking each parameter by its effect on time_eq.
    Effect is measured as  log10(t_eq_max / t_eq_min)  over the sweep range,
    using only finite values.
    """
    params  = df["param_attr"].unique()
    labels  = []
    effects = []  # log10 ratio
    t_mins  = []
    t_maxs  = []

    for attr in params:
        sub = df[df["param_attr"] == attr].copy()
        sub["time_eq"] = _finite(sub["time_eq"])
        sub = sub.dropna(subset=["time_eq"])

        if sub.empty or sub["time_eq"].min() <= 0:
            continue

        t_min = sub["time_eq"].min()
        t_max = sub["time_eq"].max()

        t_lo = np.percentile(sub["time_eq"], 10)
        t_hi = np.percentile(sub["time_eq"], 90)
        effect = np.log10(t_hi / t_lo)


        # effect = np.log10(t_max / t_min)

        labels.append(sub["param_label"].iloc[0])
        effects.append(effect)
        t_mins.append(t_min)
        t_maxs.append(t_max)

    # sort by effect magnitude
    order = np.argsort(effects)
    labels  = [labels[i]  for i in order]
    effects = [effects[i] for i in order]
    t_mins  = [t_mins[i]  for i in order]
    t_maxs  = [t_maxs[i]  for i in order]

    # ── figure ──
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(labels) + 1.5))

    y_pos = np.arange(len(labels))
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(labels)))

    bars = ax.barh(y_pos, effects, color=colors, edgecolor="white", height=0.6)

    # annotate bars with min/max time_eq
    for bar, t_lo, t_hi in zip(bars, t_mins, t_maxs):
        w = bar.get_width()
        ax.text(
            w + 0.03, bar.get_y() + bar.get_height() / 2,
            f"{t_lo:.1e} – {t_hi:.1e} s",
            va="center", ha="left", fontsize=7, color="dimgrey"
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel(r"Sensitivity  $\log_{10}(t_{eq,max} \,/\, t_{eq,min})$", fontsize=10)
    ax.set_title("Tornado plot – parameter sensitivity on $t_{eq}$", fontsize=11)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# 3. Normalised sensitivity plot  (time_eq vs relative parameter value x/x0)
# ---------------------------------------------------------------------------

def plot_sensitivity_normalised(df: pd.DataFrame, baseline, save_path: str = "sensitivity_normalised.png"):
    """
    All parameters on a single axes, x-axis = x / x_baseline (log scale),
    y-axis = t_eq / t_eq_baseline.  Useful for direct comparison.
    """
    params = df["param_attr"].unique()

    fig, ax = plt.subplots(figsize=(8, 5))

    cmap = plt.cm.tab10
    for i, attr in enumerate(params):
        sub = df[df["param_attr"] == attr].copy()
        sub = sub.sort_values("relative_value")
        sub["time_eq"] = _finite(sub["time_eq"])

        label = sub["param_label"].iloc[0]

        # baseline time_eq for normalisation
        t_base = _baseline_time_eq(df, attr)
        if np.isnan(t_base) or t_base <= 0:
            continue

        rel_t = sub["time_eq"] / t_base

        ax.plot(
            sub["relative_value"],
            rel_t,
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=label,
            color=cmap(i % 10),
        )

    ax.axvline(1, color="black", linestyle="--", linewidth=1, label="baseline")
    ax.axhline(1, color="black", linestyle=":",  linewidth=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Relative parameter value  $x \,/\, x_0$", fontsize=11)
    ax.set_ylabel(r"Relative $t_{eq}$  $( t_{eq} \,/\, t_{eq,0} )$", fontsize=11)
    ax.set_title("Normalised sensitivity – all parameters", fontsize=12)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# 4. k_m sweep plot with full-collection regime marker
# ---------------------------------------------------------------------------

def plot_sensitivity_km(df: pd.DataFrame, baseline, save_path: str = "sensitivity_km.png"):
    """
    One subplot per parameter showing k_m vs parameter value.

    Points where the model is in the full-collection regime (F was capped at
    Pe_H inside simulate()) are highlighted with a distinct marker so the
    user knows k_m there equals the convective ceiling, not the diffusive value.

    Y-axis limits are fixed globally across all subplots.
    """
    params = df["param_attr"].unique()
    n = len(params)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    # ── global y limits from all finite k_m values ──
    finite_km = df["k_m"].replace([np.inf, -np.inf], np.nan).dropna()
    finite_km = finite_km[finite_km > 0]
    if len(finite_km) > 0:
        y_lo = 10 ** np.floor(np.log10(finite_km.min()))
        y_hi = 10 ** np.ceil( np.log10(finite_km.max()))
    else:
        y_lo, y_hi = 1e-6, 1e-2

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    # legend handles built once, reused
    from matplotlib.lines import Line2D
    legend_partial  = Line2D([0], [0], color="steelblue", marker="o",
                              markersize=5, linewidth=1.8, label="partial collection")
    legend_full     = Line2D([0], [0], color="steelblue", marker="^",
                              markersize=7, linewidth=0,
                              markerfacecolor="none", markeredgecolor="darkorange",
                              markeredgewidth=1.8, label="full collection  (F = Pe_H)")
    legend_baseline = Line2D([0], [0], color="tomato", linestyle="--",
                              linewidth=1.5, label="baseline")

    for ax, attr in zip(axes, params):
        sub = df[df["param_attr"] == attr].copy()
        sub = sub.sort_values("param_value")
        sub["k_m"] = sub["k_m"].replace([np.inf, -np.inf], np.nan)

        label    = sub["param_label"].iloc[0]
        unit     = sub["param_unit"].iloc[0]
        base_val = sub["baseline_value"].iloc[0]

        partial = sub[~sub["full_collection"]]
        full    = sub[ sub["full_collection"]]

        # ── partial-collection points (connected line) ──
        ax.plot(
            partial["param_value"],
            partial["k_m"],
            color="steelblue",
            linewidth=1.8,
            marker="o",
            markersize=4,
            zorder=3,
        )

        # ── full-collection points (open triangle, no connecting line) ──
        if not full.empty:
            ax.scatter(
                full["param_value"],
                full["k_m"],
                marker="^",
                s=55,
                facecolors="none",
                edgecolors="darkorange",
                linewidths=1.8,
                zorder=4,
            )

        # ── baseline vertical line ──
        ax.axvline(base_val, color="tomato", linestyle="--", linewidth=1.5)

        # ── full-collection count annotation ──
        n_full = len(full)
        if n_full > 0:
            ax.text(
                0.97, 0.05,
                f"{n_full} pt(s): full collection",
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=7, color="darkorange",
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel(f"{label}  [{unit}]", fontsize=9)
        ax.set_ylabel(r"$k_m$  [m s$^{-1}$]", fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)

    # shared legend on first visible axis
    axes[0].legend(handles=[legend_partial, legend_full, legend_baseline],
                   fontsize=7, loc="upper left")

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(r"OAT Sensitivity Analysis – Mass transport rate $k_m$", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# 5. Damköhler number sweep plot  (Da_1, Da_2, Da_squires per parameter)
# ---------------------------------------------------------------------------

def plot_sensitivity_Da(df: pd.DataFrame, baseline, save_path: str = "sensitivity_Da.png"):
    """
    One subplot per parameter showing all three Damköhler number variants
    vs parameter value on the same axes.

    Variants plotted:
        Da_squires  –  (k_on * b_m) / k_m          (transport-based)
        Da_1        –  (k_on * c_in) / k_m * H_c   (or equivalent stored scalar)
        Da_2        –  (k_on * c_in) * tau          (kinetic / residence-time based)

    A horizontal reference line at Da = 1 marks the transport/reaction boundary.
    The baseline parameter value is shown as a vertical dashed line.
    Y-axis limits are fixed globally across all subplots.
    """
    DA_COLS = {
        "Da_squires": ("Da (Squires)",  "steelblue",  "o"),
        #"Da_1":       ("Da₁",           "darkorange", "s"),
        #"Da_2":       ("Da₂",           "seagreen",   "^"),
        #"Da_3":         ("Da3", "gray", "+"),
    }

    # keep only the Da columns that actually exist in the dataframe
    da_cols_present = {k: v for k, v in DA_COLS.items() if k in df.columns}
    if not da_cols_present:
        raise ValueError(
            "None of the expected Da columns (Da_squires, Da_1, Da_2) were found in the "
            "dataframe.  Please check that the sensitivity sweep exports these fields."
        )

    params = df["param_attr"].unique()
    n      = len(params)
    ncols  = 3
    nrows  = int(np.ceil(n / ncols))

    # ── global y limits from all finite Da values ──
    all_da = pd.concat(
        [df[col].replace([np.inf, -np.inf], np.nan) for col in da_cols_present],
        ignore_index=True,
    ).dropna()
    all_da = all_da[all_da > 0]
    if len(all_da) > 0:
        y_lo = 10 ** np.floor(np.log10(all_da.min()))
        y_hi = 10 ** np.ceil( np.log10(all_da.max()))
    else:
        y_lo, y_hi = 1e-4, 1e4

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for ax, attr in zip(axes, params):
        sub = df[df["param_attr"] == attr].copy()
        sub = sub.sort_values("param_value")

        label    = sub["param_label"].iloc[0]
        unit     = sub["param_unit"].iloc[0]
        base_val = sub["baseline_value"].iloc[0]

        for col, (da_label, color, marker) in da_cols_present.items():
            y = sub[col].replace([np.inf, -np.inf], np.nan)
            ax.plot(
                sub["param_value"],
                y,
                color=color,
                linewidth=1.8,
                marker=marker,
                markersize=4,
                label=da_label,
                zorder=3,
            )

        # ── Da = 1 reference line (transport/reaction boundary) ──
        ax.axhline(
            1,
            color="black",
            linestyle=":",
            linewidth=1.2,
            label="Da = 1",
            zorder=2,
        )

        # ── baseline parameter value ──
        ax.axvline(
            base_val,
            color="tomato",
            linestyle="--",
            linewidth=1.5,
            label=f"baseline = {base_val:.2e}",
            zorder=2,
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel(f"{label}  [{unit}]", fontsize=9)
        ax.set_ylabel(r"$Da$  [–]", fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)

        # legend only on the first subplot to avoid clutter
        if ax is axes[0]:
            ax.legend(fontsize=7, loc="best")

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(r"OAT Sensitivity Analysis – Damköhler numbers", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()

# ---------------------------------------------------------------------------
# Master call
# ---------------------------------------------------------------------------

def plot_sensitivity_sweeps_with_Da(df: pd.DataFrame, baseline, save_path: str = "sensitivity_sweeps_Da.png"):
    """
    One subplot per parameter showing:
        - t_eq (left axis, solid black)
        - Da_squires (right axis, dotted black)

    Uses twin y-axis.
    """

    params = df["param_attr"].unique()
    n = len(params)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for ax, attr in zip(axes, params):
        sub = df[df["param_attr"] == attr].copy()
        sub = sub.sort_values("param_value")

        # clean data
        sub["time_eq"] = _finite(sub["time_eq"])
        sub["Da_squires"] = sub["Da_squires"].replace([np.inf, -np.inf], np.nan)

        label    = sub["param_label"].iloc[0]
        unit     = sub["param_unit"].iloc[0]
        base_val = sub["baseline_value"].iloc[0]

        # ── LEFT AXIS: t_eq ──
        ax.plot(
            sub["param_value"],
            sub["time_eq"],
            color="black",
            linewidth=2,
            linestyle="-",
            marker="o",
            markersize=4,
            label=r"$t_{eq}$",
            zorder=3,
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e1, 1e7)
        ax.set_xlabel(f"{label}  [{unit}]", fontsize=9)
        ax.set_ylabel(r"$t_{eq}$ [s]", fontsize=9)

        # ── RIGHT AXIS: Da ──
        ax2 = ax.twinx()

        ax2.plot(
            sub["param_value"],
            sub["Da_squires"],
            color="black",
            linewidth=1.8,
            linestyle=":",
            marker=None,
            label=r"$Da$",
            zorder=2,
        )

        ax2.set_yscale("log")
        ax2.set_ylim(1e-3, 1e3)
        ax2.set_ylabel(r"$Da$ [–]", fontsize=9)

        # ── reference lines ──
        ax.axvline(base_val, color="tomato", linestyle="--", linewidth=1.5)

        ax2.axhline(
            1,
            color="grey",
            linestyle=":",
            linewidth=1,
        )

        # ── title & grid ──
        ax.set_title(label, fontsize=10)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)

        # ── combined legend (clean) ──
        lines = ax.get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, fontsize=7, loc="best")

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Sensitivity: $t_{eq}$ and Damköhler number", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()



def plot_sensitivity_all(df: pd.DataFrame, baseline):
    plot_sensitivity_sweeps_with_Da(df, baseline, save_path="sensitivity_sweep_with_Da.png")
    #plot_sensitivity_Da(df, baseline, save_path="sensitivity_Da.png")
    #plot_sensitivity_km(df, baseline,         save_path="sensitivity_km.png")
    #plot_sensitivity_sweeps(df, baseline,     save_path="sensitivity_sweeps.png")
    plot_tornado(df, baseline,                save_path="sensitivity_tornado.png")
    plot_sensitivity_normalised(df, baseline, save_path="sensitivity_normalised.png")
