import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Definición de variables simbólicas
t = sp.symbols('t')
k = sp.symbols('k')
p = 1  # Ancho del escalón
T = 5  # Periodo
f = 1 / T

# Cálculo simbólico del coeficiente de Fourier
expk = sp.integrate(sp.exp(-1 * sp.I * k * 2 * sp.pi * f * t), (t, -p / 2, p / 2)) / T

am = 25
armo = np.arange(-am, am + 1, 1)

# a) Cálculo de los coeficientes de Fourier
coef = [complex(expk.subs(k, i)) for i in armo]
modcoef = np.abs(coef)
fascoef = np.angle(coef)

#  b) Desplazo la señal en el dominio del tiempo
expk_2 = expk * sp.exp(-1 * sp.I * k * 2 * sp.pi * f * p / 2)
coef_2 = [complex(expk_2.subs(k, i)) for i in armo]
modcoef_2 = np.abs(coef_2)
fascoef_2 = np.angle(coef_2)

# c) Comprimo la señal en el dominio del tiempo
a = 2
expk_comp = expk * sp.exp(sp.I * k * 2 * sp.pi * f * a)
coef_com = [complex(expk_comp.subs(k, i)) for i in armo]
modcoef_com = np.abs(coef_com)
fascoef_com = np.angle(coef_com)

plt.figure(figsize=(18, 12))
plt.suptitle(r'Actividad 1 - Coeficientes de Fourier para $f(t)$, $f(t-(p/2))$ y $f(2t)$')

# inciso a)
plt.subplot(3, 2, 1)
plt.stem(armo, modcoef, basefmt=" ", linefmt='b-', markerfmt='bo')
plt.title(r'Modulo de los coeficientes de Fourier $f(t)$')
plt.xlabel('Armónicos')
plt.ylabel('Módulo')
plt.grid()

plt.subplot(3, 2, 2)
plt.stem(armo, fascoef, basefmt=" ", linefmt='r-', markerfmt='ro')
plt.title(r'Fase de los coeficientes de Fourier $f(t)$')
plt.xlabel('Armónicos')
plt.ylabel('Fase')
plt.grid()

# inciso b)
plt.subplot(3, 2, 3)
plt.stem(armo, modcoef_2, basefmt=" ", linefmt='b-', markerfmt='bo')
plt.title(r'Modulo de los coeficientes de Fourier $f(t-(p/2))$')
plt.xlabel('Armónicos')
plt.ylabel('Módulo')
plt.grid()

plt.subplot(3, 2, 4)
plt.stem(armo, fascoef_2, basefmt=" ", linefmt='r-', markerfmt='ro')
plt.title(r'Fase de los coeficientes de Fourier $f(t-(p/2))$')
plt.xlabel('Armónicos')
plt.ylabel('Fase')
plt.grid()

# inciso c)
plt.subplot(3, 2, 5)
plt.stem(armo, modcoef_com, basefmt=" ", linefmt='b-', markerfmt='bo')
plt.title(r'Modulo de los coeficientes de Fourier $f(2t)$')
plt.xlabel('Armónicos')
plt.ylabel('Módulo')
plt.grid()

plt.subplot(3, 2, 6)
plt.stem(armo, fascoef_com, basefmt=" ", linefmt='r-', markerfmt='ro')
plt.title(r'Fase de los coeficientes de Fourier $f(2t)$')
plt.xlabel('Armónicos')
plt.ylabel('Fase')
plt.grid()

plt.tight_layout()
plt.show()

# --- Actividad 2 ---
t = sp.symbols('t')
k = sp.symbols('k')
d = sp.symbols('d') # Ancho del escalon, luego reemplazaré
T_2 = 20
f_2 = 1/T_2
w = 2*sp.pi*f_2
# Calculo simbolicament ela expresion del coef. de la serie de Fourier compleja con f(t)=1
expk2 = sp.integrate(sp.exp(-1*sp.I*k*w*t), (t, -d/2, d/2))/T_2
am = 25
armo = np.arange(-am, am+1, 1)
expk2_2 = expk2.subs(d, 2) # d=2
expk2_4 = expk2.subs(d, 4) # d=4
expk2_5 = expk2.subs(d, 5) # d=5

coef2_2 = [complex(expk2_2.subs(k, i).evalf()) for i in armo]
coef2_4 = [complex(expk2_4.subs(k, i).evalf()) for i in armo]
coef2_5 = [complex(expk2_5.subs(k, i).evalf()) for i in armo]

# Modulo de los coef de fourier
modcoef2_2 = np.abs(coef2_2)
modcoef2_4 = np.abs(coef2_4)
modcoef2_5 = np.abs(coef2_5)

fascoef2_2 = np.angle(coef2_2)
fascoef2_4 = np.angle(coef2_4)
fascoef2_5 = np.angle(coef2_5)

plt.figure(figsize=(18, 12))
plt.suptitle('Actividad 2 - Coeficientes de Fourier con distintos valores de d')

# d=2
plt.subplot(3, 2, 1)
plt.stem(armo, modcoef2_2, label='Modulo', linefmt='b-', markerfmt='bo', basefmt='b-')
plt.title('Modulo de los Coeficientes de Fourier (d=2)')
plt.xlabel('Armonicos')
plt.ylabel('Modulo')
plt.legend()

plt.subplot(3, 2, 2)
plt.stem(armo, fascoef2_2, label='Fase', linefmt='r-', markerfmt='ro', basefmt='r-')
plt.title('Fase de los Coeficientes de Fourier (d=2)')
plt.xlabel('Armonicos')
plt.ylabel('Fase')
plt.legend()

# d=4
plt.subplot(3, 2, 3)
plt.stem(armo, modcoef2_4, label='Modulo', linefmt='b-', markerfmt='bo', basefmt='b-')
plt.title('Modulo de los Coeficientes de Fourier (d=4)')
plt.xlabel('Armonicos')
plt.ylabel('Modulo')
plt.legend()

plt.subplot(3, 2, 4)
plt.stem(armo, fascoef2_4, label='Fase', linefmt='r-', markerfmt='ro', basefmt='r-')
plt.title('Fase de los Coeficientes de Fourier (d=4)')
plt.xlabel('Armonicos')
plt.ylabel('Fase')
plt.legend()

# d=5
plt.subplot(3, 2, 5)
plt.stem(armo, modcoef2_5, label='Modulo', linefmt='b-', markerfmt='bo', basefmt='b-')
plt.title('Modulo de los Coeficientes de Fourier (d=5)')
plt.xlabel('Armonicos')
plt.ylabel('Modulo')
plt.legend()

plt.subplot(3, 2, 6)
plt.stem(armo, fascoef2_5, label='Fase', linefmt='r-', markerfmt='ro', basefmt='r-')
plt.title('Fase de los Coeficientes de Fourier (d=5)')
plt.xlabel('Armonicos')
plt.ylabel('Fase')
plt.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# --- Actividad 3 ---

t = sp.symbols('t')
k = sp.symbols('k')
d = 2 # Ancho del escalon
T = sp.symbols('T')
f3 = 1/T
w3 = 2*sp.pi*f3
# Calculo simbolicamente la expresion del coef. de la serie de Fourier compleja con f(t)=1
expk3 = sp.integrate(sp.exp(-1*sp.I*k*w3*t), (t, -d/2, d/2))/T

am = 25
armo = np.arange(-am, am+1, 1)
# T=8
expk_8 = expk3.subs(T, 8)
coef_8 = [complex(expk_8.subs(k, i).evalf()) for i in armo]
modcoef_8 = np.abs(coef_8)
fascoef_8 = np.angle(coef_8)
# T=12
expk_12 = expk3.subs(T, 12)
coef_12 = [complex(expk_12.subs(k, i).evalf()) for i in armo]
modcoef_12 = np.abs(coef_12)
fascoef_12 = np.angle(coef_12)
# T=18
expk_18 = expk3.subs(T, 18)
coef_18 = [complex(expk_18.subs(k, i).evalf()) for i in armo]
modcoef_18 = np.abs(coef_18)
fascoef_18 = np.angle(coef_18)

plt.figure(figsize=(18, 12))
plt.suptitle('Actividad 3 - Coeficientes de Fourier con distintos valores de T')

# T=8
plt.subplot(3, 2, 1)
plt.stem(armo, modcoef_8, label='Modulo', linefmt='b-', markerfmt='bo', basefmt='b-')
plt.title('Modulo de los Coeficientes de Fourier (T=8)')
plt.xlabel('Armonicos')
plt.ylabel('Modulo')
plt.legend()

plt.subplot(3, 2, 2)
plt.stem(armo, fascoef_8, label='Fase', linefmt='r-', markerfmt='ro', basefmt='r-')
plt.title('Fase de los Coeficientes de Fourier (T=8)')
plt.xlabel('Armonicos')
plt.ylabel('Fase')
plt.legend()

# T=12
plt.subplot(3, 2, 3)
plt.stem(armo, modcoef_12, label='Modulo', linefmt='b-', markerfmt='bo', basefmt='b-')
plt.title('Modulo de los Coeficientes de Fourier (T=12)')
plt.xlabel('Armonicos')
plt.ylabel('Modulo')
plt.legend()

plt.subplot(3, 2, 4)
plt.stem(armo, fascoef_12, label='Fase', linefmt='r-', markerfmt='ro', basefmt='r-')
plt.title('Fase de los Coeficientes de Fourier (T=12)')
plt.xlabel('Armonicos')
plt.ylabel('Fase')
plt.legend()

# T=18
plt.subplot(3, 2, 5)
plt.stem(armo, modcoef_18, label='Modulo', linefmt='b-', markerfmt='bo', basefmt='b-')
plt.title('Modulo de los Coeficientes de Fourier (T=18)')
plt.xlabel('Armonicos')
plt.ylabel('Modulo')
plt.legend()

plt.subplot(3, 2, 6)
plt.stem(armo, fascoef_18, label='Fase', linefmt='r-', markerfmt='ro', basefmt='r-')
plt.title('Fase de los Coeficientes de Fourier (T=18)')
plt.xlabel('Armonicos')
plt.ylabel('Fase')
plt.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()