from biosensor.model.calculate_Sherwood import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize


Pe_H = 10
lambda_ratio = 1

x = F_combine(Pe_H,lambda_ratio)

Pe_H_vals = np.logspace(-2,4,500)

lambda_low = 1e-2
lambda_high = 1e2
Pe_s_low = 6 * (lambda_low**2) * Pe_H_vals
Pe_s_high = 6 * (lambda_high**2) * Pe_H_vals

F_small_vals = [F_Ackerberg(ps) for ps in Pe_s_low]
F_large_vals = [F_Newman(ps) for ps in Pe_s_high]
F_retained_vals = [F_retained(ph) for ph in Pe_H_vals]

# Squires figure 3b
plt.figure(figsize=(6,4))

# plot asymptotes
plt.loglog(Pe_H_vals, F_retained_vals, 'g-',lw=2, label='Retained limit ($F=Pe_H$)')
plt.loglog(Pe_H_vals, F_small_vals, 'r-',lw=2, label='Small-$Pe_s$ asymptote')
plt.loglog(Pe_H_vals, F_large_vals, 'b-',lw=2, label='Large-$Pe_s$ asymptote')

# plot blended values
for lambda_val in np.logspace(-2,2,5):
#lambda_ratio = 10   # choose sensor-to-channel ratio

    F_vals = np.array([F_combine(ph, lambda_ratio=lambda_val,sharpness=4) for ph in Pe_H_vals])

    # mask: only keep points where F_vals >= Pe_H_vals
    mask = F_vals <= Pe_H_vals


    # also show asymptotic limits for reference
    plt.loglog(Pe_H_vals[mask], F_vals[mask], 'k--',alpha=0.9, label='Blended $F(Pe_H)$' + str(lambda_val))

# Plot settings
plt.rcParams.update({
    "font.size": 10,           # base font size
    "axes.linewidth": 1.5,     # axis thickness
    "xtick.direction": "in",   # ticks inward
    "ytick.direction": "in",
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
})



plt.xlabel('Pe$_H$',fontsize='large')
plt.ylabel('Dimensionless flux $F$')
plt.title('Dimensionless flux ')
plt.legend(["Full collection","Ackerberg limit","Newman limit","Combined"])
plt.grid(True, which="major", ls="-",linewidth=1,alpha=0.8)
plt.grid(True, which="minor", ls="-",linewidth=1,alpha=0.2)
plt.xlim(1e-2, 1e4)  # set x-axis limits
plt.ylim(1e-3, 1e4)  # set y-axis limits
plt.show()
plt.close('all')

## -- new -- ##

Pe_H_vals = np.logspace(-2,4,500)
lambda_ratio_vals = np.logspace(-2,2,100)

perc = np.zeros((len(lambda_ratio_vals), len(Pe_H_vals)))
F_found = np.zeros((len(lambda_ratio_vals), len(Pe_H_vals)))

for i, lambda_val in enumerate(lambda_ratio_vals):
    for j, Pe_H in enumerate(Pe_H_vals):
        F_found[i,j] = F_combine(Pe_H, lambda_val)
        perc[i,j] = 100 * (F_found[i,j] / Pe_H)




# Squires figure 3b - with capture percentage
plt.figure(figsize=(7,4))

# plot asymptotes
plt.loglog(Pe_H_vals, F_retained_vals, 'g-',lw=2, label='Retained limit ($F=Pe_H$)')
plt.loglog(Pe_H_vals, F_small_vals, 'r-',lw=2, label='Small-$Pe_s$ asymptote')
plt.loglog(Pe_H_vals, F_large_vals, 'b-',lw=2, label='Large-$Pe_s$ asymptote')

norm = Normalize(vmin=0, vmax=100)
norm = LogNorm(vmin=1, vmax=100)





# plot blended values
for lambda_val in np.logspace(-2,2,5):
#lambda_ratio = 10   # choose sensor-to-channel ratio

    F_vals = np.array([F_combine(ph, lambda_ratio=lambda_val,sharpness=4) for ph in Pe_H_vals])

    # mask: only keep points where F_vals >= Pe_H_vals
    mask = F_vals <= Pe_H_vals


    # also show asymptotic limits for reference
    plt.loglog(Pe_H_vals[mask], F_vals[mask], 'k--',alpha=0.9, label='Blended $F(Pe_H)$' + str(lambda_val))

# Plot settings
plt.rcParams.update({
    "font.size": 10,           # base font size
    "axes.linewidth": 1.5,     # axis thickness
    "xtick.direction": "in",   # ticks inward
    "ytick.direction": "in",
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
})

mask = perc > 100        # boolean mask for unphysical values
perc_masked = np.copy(perc)
perc_masked[mask] = np.nan

# grid
PeH_grid, LR_grid = np.meshgrid(Pe_H_vals, lambda_ratio_vals)

sc = plt.scatter(
    PeH_grid[~mask].flatten(),
    F_found[~mask].flatten(),  # convert % back to F for plotting
    c=perc[~mask].flatten(),
    s=10,
    cmap="YlGn",
    norm=norm
)

cbar = plt.colorbar(sc)
cbar.ax.set_position([0.83, 0.1, 0.03, 0.8])  # [left, bottom, width, height] in figure coords
cbar.set_label("Percentage of complete delivery (%)")



plt.xlabel('Pe$_H$',fontsize='large')
plt.ylabel('Dimensionless flux $F$')
# plt.title('Dimensionless flux ')
plt.legend(["Complete delivery","Ackerberg limit","Newman limit","Interpolated"])
plt.grid(True, which="major", ls="-",linewidth=1,alpha=0.8)
plt.grid(True, which="minor", ls="-",linewidth=1,alpha=0.2)
plt.xlim(1e-2, 1e4)  # set x-axis limits
plt.ylim(1e-3, 1e4)  # set y-axis limits

plt.text(1.15e4, 1.9, r"$\lambda = 0.01$", fontsize=8)
plt.text(1.15e4, 7, r"$\lambda = 0.1$", fontsize=8)
plt.text(1.15e4, 3e1, r"$\lambda = 1$", fontsize=8)
plt.text(1.15e4, 1.3e2, r"$\lambda = 10$", fontsize=8)
plt.text(1.15e4, 7e2, r"$\lambda = 100$", fontsize=8)

plt.savefig('flux.svg', format='svg', dpi=300)
plt.show()


plt.close('all')
