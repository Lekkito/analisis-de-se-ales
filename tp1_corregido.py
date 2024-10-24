import numpy as np
import matplotlib.pyplot as plt

def escalon(t):
    return np.where((t >= -0.5) & (t <= 0.5), 1, 0)

def rampa(t):
    return np.where(t >= 0, t, 0)
# defino g(t)
def g(t):
    return np.where(t < -2, 0,
           np.where(t < -1, rampa(t+2) -1,
           np.where(t < 0, escalon(t+1/2),
           np.where(t < 1, 2*escalon(t-1/2),
           np.where(t <= 2, 2 - rampa(t), 0)))))


t_val = np.linspace(-15, 3, 400)

# valores de g(t), g(-t-5) y 2g(-1/2(t + 10))
g_val = [g(t) for t in t_val]
g_c = [g(-t - 5) for t in t_val]
g_f = [2 * g(-1/2 * (t + 10)) for t in t_val]


fig, axs = plt.subplots(3, 1, figsize=(8, 12))

# g(t)
axs[0].plot(t_val, g_val, label='$g(t)$', color='black', linewidth=2)
axs[0].axhline(0, color='black', linewidth=0.5)
axs[0].axvline(0, color='black', linewidth=0.5)
axs[0].set_xlabel('$t$')
axs[0].grid(True)
axs[0].legend()

# g(-t-5)
axs[1].plot(t_val, g_c, label='$g(-t-5)$', color='red', linewidth=2)
axs[1].axhline(0, color='black', linewidth=0.5)
axs[1].axvline(0, color='black', linewidth=0.5)
axs[1].set_xlabel('$t$')
axs[1].grid(True)
axs[1].legend()

# 2g(-1/2(t + 10))
axs[2].plot(t_val, g_f, label='$2g(-\\frac{1}{2}(t + 10))$', color='blue', linewidth=2)
axs[2].axhline(0, color='black', linewidth=0.5)
axs[2].axvline(0, color='black', linewidth=0.5)
axs[2].set_xlabel('$t$')
axs[2].grid(True)
axs[2].legend()

plt.tight_layout()
plt.show()