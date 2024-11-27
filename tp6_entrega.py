import matplotlib.pyplot as plt
import numpy as np


# --- ACTIVIDAD 1 ---
t = np.arange(-2*0.01, 2*0.01, 0.0001)
x = 3 * np.cos(100 * np.pi * t)

# Frecuencias de muestreo
fm1 = 300
fm2 = 400
fm3 = 540

Tm1 = 1 / fm1
Tm2 = 1 / fm2
Tm3 = 1 / fm3

N1 = round(0.02 / Tm1)
N2 = round(0.02 / Tm2)
N3 = round(0.02 / Tm3)

n1 = np.arange(-N1, N1 + 1)
n2 = np.arange(-N2, N2 + 1)
n3 = np.arange(-N3, N3 + 1)

ts1 = n1 * Tm1
ts2 = n2 * Tm2
ts3 = n3 * Tm3

xs1 = 3 * np.cos(100 * np.pi * ts1)
xs2 = 3 * np.cos(100 * np.pi * ts2)
xs3 = 3 * np.cos(100 * np.pi * ts3)

# figura 
fig, axs = plt.subplots(4, 1, figsize=(10, 15))

# x(t)
axs[0].plot(t, x, label=r'$x(t) = 3 \cos(100 \pi t)$', color='blue')
axs[0].set_title(r'$x(t)$')
axs[0].set_xlabel(r'$t$ (s)')
axs[0].set_ylabel(r'$x(t)$')
axs[0].grid(True)
axs[0].legend()

# x[n] con T = 1/300 s
axs[1].plot(t, x, label=r'$x(t)$', color='blue', alpha=0.25)
axs[1].stem(ts1, xs1, label=r'$x[n]$ con $T = 1/300$ s', basefmt=" ", linefmt='r-', markerfmt='ro')
axs[1].set_title(r'$T = 1/300$ s')
axs[1].set_xlabel(r'$t$ (s)')
axs[1].set_ylabel(r'$x[n]$')
axs[1].grid(True)
axs[1].legend()

# x[n] con T = 1/400 s
axs[2].plot(t, x, label='$x(t)$', color='blue', alpha=0.25)
axs[2].stem(ts2, xs2, label=r'$x[n]$ con $T = 1/400$ s', basefmt=" ", linefmt='r-', markerfmt='ro')
axs[2].set_title(r'$T = 1/400$ s')
axs[2].set_xlabel(r'$t$ (s)')
axs[2].set_ylabel(r'$x[n]$')
axs[2].grid(True)
axs[2].legend()

# x[n] con T = 1/540 s
axs[3].plot(t, x, label=r'$x(t)$', color='blue', alpha=0.25)
axs[3].stem(ts3, xs3, label=r'$x[n]$ con $T = 1/540$ s', basefmt=" ", linefmt='r-', markerfmt='ro')
axs[3].set_title(r'$T = 1/540$ s')
axs[3].set_xlabel(r'$t$ (s)')
axs[3].set_ylabel(r'$x[n]$')
axs[3].grid(True)
axs[3].legend()

plt.tight_layout()
plt.show()

# --- ACTIVIDAD 8 ---

def triangular(f, W):
    return np.maximum(1 - np.abs(f) / W, 0)

W = 1000

# Frecuencias de muestreo
fm1 = 1500
fm2 = 2500

# rango de frecuencias
f = np.linspace(-3*W, 3*W, 1000)

X_f = triangular(f, W)

# señales muestreadas
X_fm1 = triangular(f - fm1, W) + triangular(f + fm1, W)
X_fm2 = triangular(f - fm2, W) + triangular(f + fm2, W)

fig, axs = plt.subplots(3, 1, figsize=(10, 15))
# X_f
axs[0].plot(f, X_f, label=r'$X(f)$', color='blue')
axs[0].set_title(r'$X(f)$')
axs[0].set_xlabel('Frecuencia (Hz)')
axs[0].set_ylabel('Amplitud')
axs[0].grid(True)
axs[0].legend()

# fm = 1500 Hz
axs[1].plot(f, X_fm1, label=r'$f_m = 1500$ Hz', color='red')
axs[1].set_title(r'$f_m = 1500$ Hz')
axs[1].set_xlabel('Frecuencia (Hz)')
axs[1].set_ylabel('Amplitud')
axs[1].grid(True)
axs[1].legend()

# fm = 2500 Hz
axs[2].plot(f, X_fm2, label=r'$f_m = 2500$ Hz', color='red')
axs[2].set_title(r'$f_m = 2500$ Hz')
axs[2].set_xlabel('Frecuencia (Hz)')
axs[2].set_ylabel('Amplitud')
axs[2].grid(True)
axs[2].legend()

plt.tight_layout()
plt.show()

# --- ACTIVIDAD 9 ---

