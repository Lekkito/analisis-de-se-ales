import numpy as np
import matplotlib.pyplot as plt

def escalon(t):
    return np.where((t >= -0.5) & (t <= 0.5), 1, 0)

def rampa(t):
    return np.where(t >= 0, t, 0)

def g(t):
    return np.where(t < -2, 0,
           np.where(t < -1, rampa(t+2) -1,
           np.where(t < 0, escalon(t+1/2),
           np.where(t < 1, 2*escalon(t-1/2),
           np.where(t <= 2, 2 - rampa(t), 0)))))
    

# rango de t
t_values_cont = np.linspace(-3, 3, 400)
g_values_cont = [g(t) for t in t_values_cont]

# Ploteo g(t)
plt.figure(figsize=(8, 6))
plt.plot(t_values_cont, g_values_cont, label='$g(t)$', color='black', linewidth=2)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.xlim(-3, 3)
plt.ylim(-1, 3)
plt.xlabel('$t$')
plt.ylabel('$g(t)$')
plt.grid(True)
plt.show()