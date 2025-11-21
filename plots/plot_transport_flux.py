from biosensor.model.calculate_Sherwood import *
import matplotlib.pyplot as plt


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