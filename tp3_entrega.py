import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# f(t) = 1 si |x-kT| < |p/2|, 0 en otro caso
p = 1 # Ancho del escalon
t = sp.symbols('t')
k = sp.symbols('k')
T = 5 # Periodo
f = 1/T
# Calcula simbolicamente la expresion del coeficiente de fourier
expk = sp.integrate(sp.exp(-1*sp.I*k*2*sp.pi*f*t),(t,-p/2,p/2))/T

# Rango de los armonicos para los coeficientes
am = 25
armo = np.arange(-am, am+1, 1)
# Eval de la expresion de los coef de Fourier
coef = [complex(expk.subs(k, i)) for i in armo]
# Modulos
modcoef = np.abs(coef)
# Fases
fascoef = np.angle(coef)

# Grafico de los coeficientes
plt.figure(figsize=(10,5))
plt.subplot(2,1,1)
plt.stem(armo, modcoef)
plt.title('Modulo de los coeficientes de Fourier')
plt.grid()
plt.subplot(2,1,2)
plt.stem(armo, fascoef)
plt.title('Fase de los coeficientes de Fourier')
plt.grid()

plt.tight_layout()
plt.show()

# Inciso b)
# f(t-(p/2))

# Desplazo la señal en el dom del t, usando las propiedades de la serie de Fourier
expk_2 = expk * sp.exp(-1*sp.I*k*2*sp.pi*f*p/2)

coef_2 = [complex(expk_2.subs(k, i)) for i in armo]
modcoef_2 = np.abs(coef_2)
fascoef_2 = np.angle(coef_2)

plt.figure(figsize=(10,5))
plt.subplot(2,1,1)
plt.stem(armo, modcoef_2)
plt.title('Modulo de los coeficientes de Fourier')
plt.grid()
plt.subplot(2,1,2)
plt.stem(armo, fascoef_2)
plt.title('Fase de los coeficientes de Fourier')
plt.grid()
plt.tight_layout()
plt.show()

# Inciso c)
# f(2t)
# Comprimo la señal en el dom del t, usando las props de la serie de Fourier
a = 2
expk_comp = expk * sp.exp(sp.I*k*2*sp.pi*f*a)

coef_com = [complex(expk_comp.subs(k, i)) for i in armo]
modcoef_com = np.abs(coef_com)
fascoef_com = np.angle(coef_com)

plt.figure(figsize=(10,5))
plt.subplot(2,1,1)
plt.stem(armo, modcoef_com)
plt.title('Modulo de los coeficientes de Fourier')
plt.grid()
plt.subplot(2,1,2)
plt.stem(armo, fascoef_com)
plt.title('Fase de los coeficientes de Fourier')
plt.grid()
plt.tight_layout()
plt.show()


# --- Actividad 2 ---
d = sp.symbols('d') # Ancho del escalon, luego reemplazaré
T_2 = 20
f_2 = 1/T_2
w = 2*sp.pi*f
# Calculo simbolicament ela expresion del coef. de la serie de Fourier compleja con f(t)=1
expk2 = sp.integrate(sp.exp(-1*sp.I*k*w*t), (t, -d/2, d/2))/T_2

expk2_2 = expk.subs(d, 2) # d=2
expk2_4 = expk.subs(d, 4) # d=4
expk2_5 = expk.subs(d, 5) # d=5

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
plt.suptitle('Coeficientes de Fourier con distintos valores de d')
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

plt.tight_layout()
plt.show()

# TODO! Corregir este ultimo punto
# TODO! Falta Actividad 3!