import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter

# Parameters
q, E, R, lamb, T0 = 100.0, 10.0, 8.314, 0.1, 300.0
T_initial = 300.0
t_max = 0.5
dt = 1e-4
ell = 0.05
sigma = ell / 2.0
noise_level = 0.10  # 10% noise

def semenov_rhs(t, T, q, E, R, lamb, T0):
    return q * np.exp(-E / (R * T)) - lamb * (T - T0)

def simulate_semenov(T_initial, q, E, R, lamb, T0, t_max, dt, noise_level=0.0):
    t_eval = np.arange(0, t_max, dt)
    sol = solve_ivp(semenov_rhs, (0, t_max), [T_initial], args=(q, E, R, lamb, T0),
                    t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-10)
    T_clean = sol.y[0]
    if noise_level > 0:
        noise = noise_level * np.random.normal(0, np.std(T_clean), size=len(T_clean))
        T = T_clean + noise
    else:
        T = T_clean
    return t_eval, T

def causal_stepanov_norm(u, t, dt, ell, sigma):
    idx = int(round(t / dt))
    start = max(0, idx - int(ell / dt))
    t_vals = np.arange(start, idx + 1) * dt
    weights = np.exp(-(t - t_vals) ** 2 / sigma ** 2)
    integrand = weights * (u[start:idx + 1]) ** 2
    integral = np.trapz(integrand, dx=dt)
    return np.sqrt(integral)

print("Running deterministic simulation to estimate C*...")
t_clean, T_clean = simulate_semenov(T_initial, q, E, R, lamb, T0, t_max, dt, noise_level=0.0)
S_clean = np.array([causal_stepanov_norm(T_clean, ti, dt, ell, sigma) for ti in t_clean])
S_clean_smooth = savgol_filter(S_clean, window_length=11, polyorder=3, mode='interp')
dS_clean = np.gradient(S_clean_smooth, dt)
H_clean = ell * dS_clean / (S_clean_smooth ** 2 + 1e-12)
n_last = int(0.2 * len(H_clean))
C_star = np.mean(H_clean[-n_last:])
print(f"Estimated C* = {C_star:.4f}")

print(f"\nRunning noisy simulation with {noise_level*100}% noise...")
t_noisy, T_noisy = simulate_semenov(T_initial, q, E, R, lamb, T0, t_max, dt, noise_level=noise_level)
S_noisy = np.array([causal_stepanov_norm(T_noisy, ti, dt, ell, sigma) for ti in t_noisy])
S_noisy_smooth = savgol_filter(S_noisy, window_length=11, polyorder=3, mode='interp')
dS_noisy = np.gradient(S_noisy_smooth, dt)
H_noisy = ell * dS_noisy / (S_noisy_smooth ** 2 + 1e-12)

yellow_thresh = C_star / 2
red_thresh = 0.9 * C_star
yellow_idx = np.where(H_noisy >= yellow_thresh)[0]
red_idx = np.where(H_noisy >= red_thresh)[0]
first_yellow = t_noisy[yellow_idx[0]] if len(yellow_idx) > 0 else None
first_red = t_noisy[red_idx[0]] if len(red_idx) > 0 else None
print(f"First yellow alert at t = {first_yellow:.5f} s")
print(f"First red alert at t = {first_red:.5f} s")

plt.figure(figsize=(10, 6))
plt.plot(t_noisy, H_noisy, 'r-', linewidth=2, label='H(t)')
plt.axhline(y=yellow_thresh, color='gold', linestyle='--', label='Yellow threshold')
plt.axhline(y=red_thresh, color='red', linestyle='--', label='Red threshold')
if first_yellow:
    plt.axvline(x=first_yellow, color='gold', linestyle=':', linewidth=2)
if first_red:
    plt.axvline(x=first_red, color='red', linestyle=':', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('H(t)')
plt.title('Early warning signal H(t) under 10% noise')
plt.legend()
plt.grid(True)
plt.savefig('algebraic_plot.png', dpi=150)
plt.show()
print("Image saved as algebraic_plot.png")
