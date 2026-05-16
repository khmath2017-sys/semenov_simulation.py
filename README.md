import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

T0 = 1.0
Tc = 1.0 / T0
dt = 1e-5
t_max = 0.99 * Tc
t = np.arange(0, t_max, dt)
T_exact = 1.0 / (Tc - t)

ell = 0.1
sigma = ell / 2.0

def stepanov_norm_causal(u, t_arr, idx, ell, sigma, dt):
    t_curr = t_arr[idx]
    start_idx = max(0, idx - int(ell / dt))
    t_win = t_arr[start_idx:idx+1]
    weights = np.exp(-(t_curr - t_win)**2 / sigma**2)
    integrand = weights * (u[start_idx:idx+1])**2
    integral = np.trapz(integrand, dx=dt)
    return np.sqrt(integral)

S = np.array([stepanov_norm_causal(T_exact, t, i, ell, sigma, dt) for i in range(len(t))])
S_smooth = savgol_filter(S, 11, 3, mode='interp')
dS = np.gradient(S_smooth, dt)
H = ell * dS / (S_smooth**2 + 1e-12)

plt.figure(figsize=(8,5))
plt.plot(t, H, 'r-', linewidth=2.5, label=r'$H(t)$')
plt.axhline(y=5, color='gold', linestyle='--', linewidth=2, label='Threshold')
plt.axvline(x=0.9*Tc, color='gray', linestyle=':', linewidth=2, label='90% of blow-up time')
plt.fill_between(t, 0, H, where=(H>=5), color='red', alpha=0.2, label='Alert region')
plt.xlabel('Time t', fontsize=12)
plt.ylabel(r'$H(t)$', fontsize=12)
plt.title('Algebraic blow-up model: Early warning signal', fontsize=14)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('algebraic_plot_new.png', dpi=200)
plt.show()
print("New image saved: algebraic_plot_new.png")
