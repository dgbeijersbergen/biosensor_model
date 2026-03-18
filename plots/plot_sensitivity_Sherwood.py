"""
Sensitivity analysis for the Sherwood interpolation — v2
Builds on the existing flux.svg plot style and overlays:
  1. Sensitivity points (D, H_c, Q, W_c sweeps) on the F vs Pe_H background
  2. Sensitivity of each parameter vs k_m (bottom panel)
"""

import sys, os, copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

sys.path.insert(0, "/mnt/project")
from biosensor.model.calculate_Sherwood import *

OUT_DIR = "/mnt/user-data/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ── rcParams (match existing style) ──────────────────────────────────────────
plt.rcParams.update({
    "font.size": 10,
    "axes.linewidth": 1.5,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 6,   "ytick.major.size": 6,
    "xtick.minor.size": 3,   "ytick.minor.size": 3,
})

# ── Base parameters (SI) ─────────────────────────────────────────────────────
Q_CONV = (1 / 60) * 1e-9       # µL/min → m³/s

# We use a simple namespace so compute_k_m can use it directly
from types import SimpleNamespace
BASE = SimpleNamespace(
    D       = 0.9378e-9,
    H_c     = 5.4579e-4,
    W_c     = 0.9571e-2,
    L_s     = 0.9571e-2,
    W_s     = 0.9571e-2,
    Q_in    = 40 * Q_CONV,
    # needed for compute_k_m validity check
    k_on    = 4.5e4  * 1e-3,    # SI: m³ mol⁻¹ s⁻¹
    k_off   = 7.03e-4,
    b_m     = 5.46e-8,
    c_in    = 1e-9   * 1e3,     # SI: mol m⁻³
    char_length = "H",
)

def make_params(**overrides):
    p = copy.copy(BASE)
    for k, v in overrides.items():
        setattr(p, k, v)
    return p

def get_F_and_km(params):
    """Return (Pe_H, Pe_s, F, k_m) for given params namespace."""

    lam   = params.L_s / params.H_c
    Pe_H  = params.Q_in / (params.D * params.W_c)
    Pe_s  = 6 * lam**2 * Pe_H
    k_m, F = compute_k_m(params.Q_in, params)
    return Pe_H, Pe_s, F, k_m

# ── Sweep definitions ─────────────────────────────────────────────────────────
sweeps = {
    "D":   dict(
        label   = "Diffusion coeff. $D$",
        unit    = "m²/s",
        values  = np.array([0.74e-10, 6.54e-10, 0.9378e-9, 5e-9, 1e-8]),
        tick_labels = ["0.74e-10\nssDNA", "6.54e-10\nMCH", "0.94e-9\nEpCAM\n(def.)",
                        "5e-9", "1e-8"],
        color   = "#E63946",
        markers = ["v", "s", "o", "^", "D"],
    ),
    "H_c": dict(
        label   = "Channel height $H_c$",
        unit    = "m",
        values  = np.array([1e-4, 2e-4, 5.4579e-4, 1e-3, 2e-3]),
        tick_labels = ["100 µm", "200 µm", "546 µm\n(def.)", "1 mm", "2 mm"],
        color   = "#457B9D",
        markers = ["v", "s", "o", "^", "D"],
    ),
    "Q_in": dict(
        label   = "Flow rate $Q$",
        unit    = "m³/s",
        values  = np.array([1, 10, 40, 100, 500]) * Q_CONV,
        tick_labels = ["1 µL/min", "10 µL/min", "40 µL/min\n(def.)",
                        "100 µL/min", "500 µL/min"],
        color   = "#2A9D8F",
        markers = ["v", "s", "o", "^", "D"],
    ),
    "W_c": dict(
        label   = "Channel width $W_c$",
        unit    = "m",
        values  = np.array([1e-3, 3e-3, 9.571e-3, 2e-2, 5e-2]),
        tick_labels = ["1 mm", "3 mm", "9.6 mm\n(def.)", "20 mm", "50 mm"],
        color   = "#E9C46A",
        markers = ["v", "s", "o", "^", "D"],
    ),
}

# Pre-compute all sweep points
for key, sw in sweeps.items():
    Pe_H_pts, Pe_s_pts, F_pts, km_pts = [], [], [], []
    for val in sw["values"]:
        p = make_params(**{key: val})
        ph, ps, F, km = get_F_and_km(p)
        Pe_H_pts.append(ph)
        Pe_s_pts.append(ps)
        F_pts.append(F)
        km_pts.append(km)
    sw["Pe_H"] = np.array(Pe_H_pts)
    sw["F"]    = np.array(F_pts)
    sw["k_m"]  = np.array(km_pts)

# Default operating point
Ph0, Ps0, F0, km0 = get_F_and_km(BASE)

# ══════════════════════════════════════════════════════════════════════════════
# Background: % full-collection scatter (same as existing code)
# ══════════════════════════════════════════════════════════════════════════════
Pe_H_vals          = np.logspace(-2, 4, 500)
lambda_ratio_vals  = np.logspace(-2, 2, 100)

lambda_low  = 1e-2
lambda_high = 1e2
Pe_s_low_  = 6 * (lambda_low**2)  * Pe_H_vals
Pe_s_high_ = 6 * (lambda_high**2) * Pe_H_vals

F_small_vals    = [F_Ackerberg(ps) for ps in Pe_s_low_]
F_large_vals    = [F_Newman(ps)    for ps in Pe_s_high_]
F_retained_vals = [F_retained(ph)  for ph  in Pe_H_vals]

perc    = np.zeros((len(lambda_ratio_vals), len(Pe_H_vals)))
F_found = np.zeros_like(perc)

for i, lv in enumerate(lambda_ratio_vals):
    for j, ph in enumerate(Pe_H_vals):
        F_found[i, j] = F_combine(ph, lv)
        perc[i, j]    = 100 * (F_found[i, j] / ph)

mask         = perc > 100
perc_masked  = np.where(mask, np.nan, perc)
PeH_grid, _  = np.meshgrid(Pe_H_vals, lambda_ratio_vals)

norm = LogNorm(vmin=1, vmax=100)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE — two rows
#   Row 1: F vs Pe_H  (background + sensitivity overlay)
#   Row 2: four subplots — each parameter vs k_m
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 11))
gs  = gridspec.GridSpec(2, 4, figure=fig,
                        hspace=0.42, wspace=0.40,
                        height_ratios=[1.5, 1])

ax_top = fig.add_subplot(gs[0, :])   # full-width top panel
ax_km  = [fig.add_subplot(gs[1, k]) for k in range(4)]

# ── TOP PANEL: F vs Pe_H ─────────────────────────────────────────────────────
# Background scatter (% full collection)
sc = ax_top.scatter(
    PeH_grid[~mask].flatten(),
    F_found[~mask].flatten(),
    c=perc[~mask].flatten(),
    s=8, cmap="YlGn", norm=norm, zorder=1, rasterized=True
)
cbar = fig.colorbar(sc, ax=ax_top, pad=0.01)
cbar.set_label("% of complete delivery ($F / Pe_H \\times 100$)", fontsize=9)

# Asymptote lines
ax_top.loglog(Pe_H_vals, F_retained_vals, "g-",  lw=2.0, zorder=3,
              label="Full collection ($F = Pe_H$)")
ax_top.loglog(Pe_H_vals, F_small_vals,    "r-",  lw=2.0, zorder=3,
              label="Ackerberg limit ($Pe_s \\ll 1$)")
ax_top.loglog(Pe_H_vals, F_large_vals,    "b-",  lw=2.0, zorder=3,
              label="Newman limit ($Pe_s \\gg 1$)")

# Blended curves for λ = 0.01, 0.1, 1, 10, 100
lambda_labels = [r"$\lambda=0.01$", r"$\lambda=0.1$", r"$\lambda=1$",
                 r"$\lambda=10$",   r"$\lambda=100$"]
for lv, ll in zip(np.logspace(-2, 2, 5), lambda_labels):
    Fv   = np.array([F_combine(ph, lv, sharpness=4) for ph in Pe_H_vals])
    good = Fv <= Pe_H_vals
    ax_top.loglog(Pe_H_vals[good], Fv[good], "k--", lw=1.2, alpha=0.7, zorder=2)

# λ annotations (right side, matching existing style)
ann_x = 1.05e4
for txt, yy in zip(lambda_labels, [1.9, 7, 3e1, 1.3e2, 7e2]):
    ax_top.text(ann_x, yy, txt, fontsize=8, va="center", clip_on=False)

# ── Overlay sensitivity points ──────────────────────────────────────────────
for key, sw in sweeps.items():
    n = len(sw["values"])
    for idx in range(n):
        is_default = (sw["markers"][idx] == "o")
        ax_top.scatter(
            sw["Pe_H"][idx], sw["F"][idx],
            color=sw["color"],
            marker=sw["markers"][idx],
            s=90 if is_default else 60,
            edgecolors="k", linewidths=0.7,
            zorder=6,
            label=f"_nolegend_"
        )

# Default operating point (larger star)
ax_top.scatter(Ph0, F0, color="white", marker="*", s=220,
               edgecolors="k", linewidths=0.9, zorder=7,
               label=f"Default ($Pe_H$={Ph0:.1f}, $F$={F0:.1f})")

# Legend — parameter colours
custom_handles = [
    Line2D([0],[0], color="g",  lw=2,  label="Full collection"),
    Line2D([0],[0], color="r",  lw=2,  label="Ackerberg limit"),
    Line2D([0],[0], color="b",  lw=2,  label="Newman limit"),
    Line2D([0],[0], color="k",  lw=1.2, ls="--", label="Interpolated ($a=4$)"),
    Line2D([0],[0], color="none", marker="*", markersize=10,
           markeredgecolor="k", markerfacecolor="white", label="Default"),
    *[Line2D([0],[0], color=sw["color"], marker="o", markersize=7,
             markeredgecolor="k", lw=0, label=sw["label"])
      for sw in sweeps.values()],
]
ax_top.legend(handles=custom_handles, fontsize=8, ncol=2, loc="upper left")

ax_top.set_xscale("log"); ax_top.set_yscale("log")
ax_top.set_xlim(1e-2, 1e4); ax_top.set_ylim(1e-3, 1e4)
ax_top.set_xlabel("$Pe_H$", fontsize=12)
ax_top.set_ylabel("Dimensionless flux $F$", fontsize=12)
ax_top.set_title("Sherwood interpolation: $F$ vs $Pe_H$ with sensitivity analysis overlay",
                 fontsize=12, fontweight="bold")
ax_top.grid(True, which="major", ls="-",  lw=1.0, alpha=0.8)
ax_top.grid(True, which="minor", ls="-",  lw=0.5, alpha=0.2)

# ── BOTTOM PANELS: parameter vs k_m ─────────────────────────────────────────
for ax, (key, sw) in zip(ax_km, sweeps.items()):
    vals = sw["values"]
    kms  = sw["k_m"]
    n    = len(vals)

    # Line connecting the points
    ax.plot(range(n), kms * 1e6, color=sw["color"], lw=1.5, zorder=2)

    # Individual markers
    for idx in range(n):
        is_default = (sw["markers"][idx] == "o")
        ax.scatter(idx, kms[idx] * 1e6,
                   color=sw["color"],
                   marker=sw["markers"][idx],
                   s=90 if is_default else 55,
                   edgecolors="k", linewidths=0.7, zorder=3)

    # Default horizontal reference
    ax.axhline(km0 * 1e6, color="gray", ls=":", lw=1.2, alpha=0.8, zorder=1)

    # Annotate default point
    default_idx = [m == "o" for m in sw["markers"]].index(True)
    ax.annotate("default", xy=(default_idx, kms[default_idx] * 1e6),
                xytext=(0, 10), textcoords="offset points",
                fontsize=7, ha="center", color="gray")

    ax.set_xticks(range(n))
    ax.set_xticklabels(sw["tick_labels"], fontsize=7)
    ax.set_xlabel(sw["label"], fontsize=9)
    ax.set_ylabel("$k_m$ (µm/s)", fontsize=9)
    ax.set_title(sw["label"], fontsize=9, fontweight="bold", color=sw["color"])
    ax.grid(True, which="major", ls="-", lw=0.8, alpha=0.5)
    ax.tick_params(axis="x", which="major", length=0)  # hide x tick lines (text labels enough)

fig.suptitle("Sherwood number sensitivity analysis", fontsize=14, fontweight="bold", y=1.01)

plt.savefig(os.path.join(OUT_DIR, "sherwood_sensitivity_v2.svg"),
            format="svg", dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(OUT_DIR, "sherwood_sensitivity_v2.png"),
            dpi=160, bbox_inches="tight")
print("Saved.")
plt.show()