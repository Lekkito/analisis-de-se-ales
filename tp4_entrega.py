import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

# --- ACTIVIDAD 1 ---

def rect(t):
    return sp.Heaviside(t + 1/2) - sp.Heaviside(t - 1/2)

# variables simbólicas
t, w, f = sp.symbols('t w f')

# Defino las funciones de cada inciso
x1 = sp.Heaviside(t) - sp.Heaviside(t - 1)
x2 = 5 * rect((t+2)/4)
x3 = 2 * sp.exp(-3 * t) * sp.Heaviside(3 * t)

# Calcular las transformadas de Fourier de las funciones
X1 = sp.fourier_transform(x1, t, w)
X2 = sp.fourier_transform(x2, t, w)
X3 = sp.fourier_transform(x3, t, w)
# Substituir
X1_f = X1.subs(w, 2 * sp.pi * f)
X2_f = X2.subs(w, 2 * sp.pi * f)
X3_f = X3.subs(w, 2 * sp.pi * f)

# modulo de la transformada de Fourier
XM1 = sp.Abs(X1_f)
XM2 = sp.Abs(X2_f)
XM3 = sp.Abs(X3_f)
# fase de la transformada de Fourier
XP1 = sp.arg(X1_f)
XP2 = sp.arg(X2_f)
XP3 = sp.arg(X3_f)


fig, axs = plt.subplots(3, 3, figsize=(15, 15))

t_vals = np.linspace(-5, 5, 400)
f_vals = np.linspace(-5, 5, 400)

# x1(t)
x1_vals = [sp.re(x1.subs(t, val)) for val in t_vals]
XM1_vals = [sp.Abs(XM1.subs(f, val)) for val in f_vals]
XP1_vals = [sp.N(XP1.subs(f, val)) for val in f_vals]

axs[0, 0].plot(t_vals, x1_vals, 'b')
axs[0, 0].set_title('x1(t)')
axs[0, 0].grid()

axs[1, 0].plot(f_vals, XM1_vals, 'r')
axs[1, 0].set_title('|X1(f)|')
axs[1, 0].grid()

axs[2, 0].plot(f_vals, XP1_vals, 'g')
axs[2, 0].set_title('fase de X1(f)')
axs[2, 0].grid()

# x2(t)
x2_vals = [sp.re(x2.subs(t, val)) for val in t_vals]
XM2_vals = [sp.Abs(XM2.subs(f, val)) for val in f_vals]
XP2_vals = [sp.re(XP2.subs(f, val)) for val in f_vals]

axs[0, 1].plot(t_vals, x2_vals, 'b')
axs[0, 1].set_title('x2(t)')
axs[0, 1].grid()

axs[1, 1].plot(f_vals, XM2_vals, 'r')
axs[1, 1].set_title('|X2(f)|')
axs[1, 1].grid()

axs[2, 1].plot(f_vals, XP2_vals, 'g')
axs[2, 1].set_title('fase de X2(f)')
axs[2, 1].grid()

# x3(t)
x3_vals = [sp.re(x3.subs(t, val)) for val in t_vals]
XM3_vals = [sp.Abs(XM3.subs(f, val)) for val in f_vals]
XP3_vals = [sp.re(XP3.subs(f, val)) for val in f_vals]

axs[0, 2].plot(t_vals, x3_vals, 'b')
axs[0, 2].set_title('x3(t)')
axs[0, 2].grid()

axs[1, 2].plot(f_vals, XM3_vals, 'r')
axs[1, 2].set_title('|X3(f)|')
axs[1, 2].grid()

axs[2, 2].plot(f_vals, XP3_vals, 'g')
axs[2, 2].set_title('fase de X3(f)')
axs[2, 2].grid()
plt.suptitle('Actividad 1')
plt.tight_layout()
plt.show()
# ¡ TARDA ALREDEDOR DE 2 MINUTOS EN CARGAR !

# --- ACTIVIDAD 3 ---

t, f = sp.symbols('t f')
f0 = 10
F1 = 2
def rect(t):
    return sp.Heaviside(t + 1/2) - sp.Heaviside(t - 1/2)
Z = rect(t/(2*F1))

# transformada de Fourier calculada a mano
X = Z.subs(t, f-f0)/2 + Z.subs(t, f+f0)/2

# Graficar el modulo de X
f_vals = np.linspace(-15, 15, 400)
XM = [sp.Abs(X.subs(f, val)) for val in f_vals]
plt.plot(f_vals, XM, 'r')
plt.title('Actividad 3: |X(f)|')
plt.grid()
plt.show()

# --- ACTIVIDAD 4 ---

f, t = sp.symbols('f t')
f0 = 3
#x = sp.cos(2 * sp.pi * f0 * t)
X = 5 * (sp.sinc(10 *sp.pi* (f - f0)) + sp.sinc(10 * sp.pi*(f + f0)))
# Grafico el modulo y fase de X
f_vals = np.linspace(-5, 5, 400)
XM = [sp.Abs(X.subs(f, val)) for val in f_vals]
XP = [sp.arg(X.subs(f, val)) for val in f_vals]

fig, axs = plt.subplots(2, 1, figsize=(10, 8))
axs[0].plot(f_vals, XM, 'r')
axs[0].set_title('|X(f)|')
axs[0].grid()

axs[1].plot(f_vals, XP, 'g')
axs[1].set_title('angle X(f)')
axs[1].grid()
plt.suptitle('Actividad 4')
plt.tight_layout()
plt.show()

# --- ACTIVIDAD 8 ---

t, f = sp.symbols('t f')

# Defino las funciones X(f) (solo los incisos a y b)
X_a = -15 * sp.Heaviside(f + 2) * sp.Heaviside(2 - f)
X_b = sp.sinc(-10 * sp.pi* f) / 30

# Calculo las antitransformadas de Fourier
x_a = sp.inverse_fourier_transform(X_a, f, t)
x_b = sp.inverse_fourier_transform(X_b, f, t)
# Cargo manualmente las antitrafo de c y d.
x_c = 2 * sp.pi * sp.exp(-20 * sp.pi * t) * sp.Heaviside(t)
x_d = -3 * sp.sign(t)

t_vals = np.linspace(-10, 10, 400)

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
#Graficar x_a(t)
x_a_vals = [x_a.subs(t, val).evalf() for val in t_vals]
axs[0, 0].plot(t_vals, x_a_vals, 'b')
axs[0, 0].set_title(r'$x_a(t) = -60*sinc(4t)$')
axs[0, 0].grid()

#Graficar x_b(t)
x_b_vals = [x_b.subs(t, val).evalf() for val in t_vals]
axs[0, 1].plot(t_vals, x_b_vals, 'r')
axs[0, 1].set_title(r'$x_b(t) = \frac{1}{300} * rect(\frac{-t}{10})$')
axs[0, 1].grid()

#Graficar x_c(t)
# Definir los valores de t para graficar c
t_vals_c = np.linspace(-0.1, 0.5, 400)
x_c_vals = [x_c.subs(t, val).evalf() for val in t_vals_c]
x_c_vals = [float(val) for val in x_c_vals]
axs[1, 0].plot(t_vals_c, x_c_vals, 'g')
axs[1, 0].set_title(r'$x_c(t) = 2 \pi * e^{-20 \pi t} * u(t)$')
axs[1, 0].grid()

#Graficar x_d(t)
x_d_vals = [x_d.subs(t, val).evalf() for val in t_vals]
x_d_vals = [float(val) for val in x_d_vals]
axs[1, 1].plot(t_vals, x_d_vals, 'm')
axs[1, 1].set_title(r'$x_d(t) = -3*sgn(t)$')
axs[1, 1].grid()
plt.suptitle('Actividad 8')
plt.tight_layout()
plt.show()
