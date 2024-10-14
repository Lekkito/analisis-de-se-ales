import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
import sympy as sp

# --- ACTIVIDAD 3 ---

# impulso unitario
delta = np.zeros(20)
delta[5] = 1

# Inciso a)
A1 = [1]
B1 = [0, 5, -1]
Y1 = lfilter(B1, A1, delta)

# Inciso b)
A2 = [0.5, 0, 0.5]
B2 = [1, 0, 3]
Y2 = lfilter(B2, A2, delta)


# Inciso c)
A3 = [1, -0.4]
B3 = [1, 0, -1]
Y3 = lfilter(B3, A3, delta)

# Grafico las respuestas
n1 = np.arange(len(Y1))
n2 = np.arange(len(Y2))
n3 = np.arange(len(Y3))

plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.stem(n1, Y1, basefmt=" ")
plt.grid()
plt.title("Actividad 3a) $y[n] = 5x[n-1]-x[n-2]$")
plt.xlabel('n')
plt.ylabel('y[n]')

plt.subplot(3, 1, 2)
plt.stem(n2, Y2, basefmt=" ")
plt.grid()
plt.title("b) $2y[n] +6y[n-2] = x[n] + x[n-2]$")
plt.xlabel('n')
plt.ylabel('y[n]')

plt.subplot(3, 1, 3)
plt.stem(n3, Y3, basefmt=" ")
plt.grid()
plt.title("c) $y[n]-0.4y[n-1]=x[n]-x[n-2]$")
plt.xlabel('n')
plt.ylabel('y[n]')

plt.tight_layout()
plt.show()


# --- ACTIVIDAD 4b ---
n4 = np.arange(-20, 21)

delta4 = np.zeros(41)
delta4[20] = 1 

# Subsistema h1
A4 = [1, 0.5]  # Coeficientes de la salida
B4 = [1]       # Coeficientes de la entrada

# Subsistema h2
C4 = [1]               # Coeficiente de la salida
D4 = [1, -1, 1]        # Coeficientes de la entrada

# respuestas al impulso de subsistemas
h2 = lfilter(D4, C4, delta4)  #  h2
h1 = lfilter(B4, A4, delta4)  #  h1 

# convolución de h2 con h1
ht = np.convolve(h1, h2)
nc = np.arange(1,82)


plt.figure(figsize=(8, 6))

# h1
plt.subplot(3, 1, 1)
plt.stem(n4, h1, 'b', markerfmt='bo')
plt.title('Actividad 4b) Respuesta sistema h1')
plt.grid(True)

# sistema h2
plt.subplot(3, 1, 2)
plt.stem(n4, h2, 'r', markerfmt='ro')
plt.title('Respuesta sistema h2')
plt.grid(True)

# convolución
plt.subplot(3, 1, 3)
plt.stem(nc, ht, 'g', markerfmt='go')
plt.title('Respuesta de la convolución (h1 * h2)')
plt.grid(True)

plt.tight_layout()
plt.show()


# --- ACTIVIDAD 5 ---

# Inciso a
n = np.arange(-8,13)
def rect4(n):
    return np.where(np.abs(n)<= 4, 1, 0)

def u(n):
    return np.where(n>=0, 1,0)
# Inciso b
def d(n):
    return np.where(n==0,1,0)

a1 = 2*rect4(n)
a2 = (7/8)**n * u(n)
a = np.convolve(a1,a2)
b1 = (0.2)**n * u(n)
b2 = d(n-1)+3*d(n-2)+2*d(n-6)
b = np.convolve(b1,b2)
nc = np.arange(-16,25)

plt.figure(figsize=(12, 8))

# Primer conjunto de señales
plt.subplot(2, 1, 1)
plt.stem(nc, a, 'ko', label='Convolución', basefmt=" ", markerfmt='ko')
plt.setp(plt.gca().lines[::2], markersize=10)
plt.stem(n, a1, 'y.', label='$2rect_{4}[n]$', basefmt=" ")
plt.stem(n, a2, 'r.', label=r'$(\frac{7}{8})^{n} u[n]$', basefmt=" ")
plt.xlabel('n')
plt.title(r' Actividad 5: $2rect_{4}[n] \ast (\frac{7}{8})^{n} u[n]$')
plt.legend(loc='best')
plt.grid()

# Segundo conjunto de señales
plt.subplot(2, 1, 2)
plt.stem(nc, b, linefmt='k-', markerfmt='ko', label='Convolución', basefmt=" ")
plt.setp(plt.gca().lines[::2], markersize=10)
plt.stem(n, b2, linefmt='r-' ,markerfmt='ro', label='b2', basefmt=" ")
plt.stem(n, b1, linefmt='g-', markerfmt='go', label='$0.2^{n}u[n]$', basefmt=" ")
plt.grid()
plt.xlabel('n')
plt.ylabel('y[n]')
plt.title(r'$0.2^{n}u[n]  \ast (d[n-1]+3d[n-2]+2d[n-6])$')
plt.legend(loc='best')

plt.tight_layout()
plt.show()

# --- ACTIVIDAD 8 ---

# Definir las variables y la función
t = sp.symbols('t')
y = sp.Function('y')
# Defino las ecuaciones diferenciales
eq1 = sp.Eq(y(t).diff(t, 2) + 2*y(t).diff(t) + 4*y(t), sp.DiracDelta(t))
eq2 = sp.Eq(y(t).diff(t, 2) + 6*y(t).diff(t) + 5*y(t), sp.DiracDelta(t))
eq3 = sp.Eq(y(t).diff(t, 2) - 2*y(t).diff(t) + 3*y(t), sp.DiracDelta(t))
# Condiciones iniciales
conds1 = {y(0): 0, y(t).diff(t).subs(t, 0): 0}
conds2 = {y(0): 0, y(t).diff(t).subs(t, 0): 0}
conds3 = {y(0): 0, y(t).diff(t).subs(t, 0): 0}
# Resolver las ecuaciones
sol1 = sp.dsolve(eq1, y(t), ics=conds1)
sol2 = sp.dsolve(eq2, y(t), ics=conds2)
sol3 = sp.dsolve(eq3, y(t), ics=conds3)
# Simplificar las soluciones
sol_simp1 = sp.simplify(sol1.rhs)
sol_simp2 = sp.simplify(sol2.rhs)
sol_simp3 = sp.simplify(sol3.rhs)
# Convertir la solución simplificada a una función numérica
y_impulso1 = sp.lambdify(t, sol_simp1, 'numpy')
y_impulso2 = sp.lambdify(t, sol_simp2, 'numpy')
y_impulso3 = sp.lambdify(t, sol_simp3, 'numpy')
# rango de tiempo para la evaluacion
c = np.arange(0, 5, 0.01)
# Evaluar la función
impulso_resp_eval1 = y_impulso1(c)
impulso_resp_eval2 = y_impulso2(c)
impulso_resp_eval3 = y_impulso3(c)
# Graficar la respuesta al impulso
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(c, impulso_resp_eval1, label="8a)", color='r')
plt.title(r"Actividad 8: Respuesta al impulso $y''(t) + 2y'(t) + 4y(t) = x(t)$")
plt.xlabel('t')
plt.ylabel('h(t)')
plt.grid()
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(c, impulso_resp_eval2, label="8b)", color='g')
plt.title(r"Respuesta al impulso $y''(t) + 6y'(t) + 5y(t) = x(t)$")
plt.xlabel('t')
plt.ylabel('h(t)')
plt.grid()
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(c, impulso_resp_eval3, label="8c)", color='b')
plt.title(r"Respuesta al impulso $y''(t)-2y'(t)+3y(t)=x(t)$")
plt.xlabel('t')
plt.ylabel('h(t)')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

# --- Actividad 10 ---

ts = 0.02
n = np.arange(-200, 200)
# inciso a
# Construyo la funcion rectangulo con la Heaviside
rec_a1 = np.heaviside((n * ts) + 0.5, 1) - np.heaviside((n * ts) - 0.5, 1)
# Construyo rect(t/2)
rec_a2 = np.heaviside((n * ts) + 1, 1) - np.heaviside((n * ts) - 1, 1)
# b)
# Construyo la funcion rect(t-1) con la Heaviside
rec_b1 = np.heaviside((n * ts) -0.5, 1) - np.heaviside((n * ts) -1.5, 1)
# Construyo la rect(t/2) con la Heaviside
rec_b2 = np.heaviside((n * ts) + 1, 1) - np.heaviside((n * ts) - 1, 1)
# c) y d)
# Definir la función rampa
def rampa(t):
    return np.where(t >= 0, t, 0)
# función rectángulo (como funcion)
def rect(t):
    return np.where(np.abs(t) <= 0.5, 1, 0)

nc = np.arange(n[0] + n[0], n[-1] + n[-1] + 1)
# convoluciono las señales
y_a = ts * np.convolve(rec_a1, rec_a2)
y_b = ts * np.convolve(rec_b1, rec_b2)
# inciso c
t = n * ts
rampa_rect = rampa(t) * rect(t - 0.5)
rect_t = rect(t)
y_c = ts * np.convolve(rampa_rect, rect_t)

# inciso d
exp_u = np.exp(-t) * np.heaviside(t, 1)
y_d = ts * np.convolve(rampa_rect, exp_u)

# Gráfico de todas las señales y sus convoluciones
plt.figure(figsize=(12, 16))
plt.suptitle('ACTIVIDAD 10', fontsize=16)

# Gráficos del inciso a
plt.subplot(4, 3, 1)
plt.plot(ts * n, rec_a1)
plt.title(r'a) $rect(t)$')

plt.subplot(4, 3, 2)
plt.plot(ts * n, rec_a2)
plt.title(r'$rect(t/2)$')

plt.subplot(4, 3, 3)
plt.plot(ts * nc, y_a)
plt.title(r'Convolución $rect(t) \ast rect(t/2)$')

# Gráficos del inciso b
plt.subplot(4, 3, 4)
plt.plot(ts * n, rec_b1)
plt.title(r'b )$rect(t-1)$')

plt.subplot(4, 3, 5)
plt.plot(ts * n, rec_b2)
plt.title(r'$rect(t/2)$')

plt.subplot(4, 3, 6)
plt.plot(ts * nc, y_b)
plt.title(r'Convolución $rect(t-1) \ast rect(t/2)$')

# Gráficos del inciso c
plt.subplot(4, 3, 7)
plt.plot(t, rampa_rect)
plt.title(r'c) $rampa(t) \cdot rect(t - 1/2)$')

plt.subplot(4, 3, 8)
plt.plot(t, rect_t)
plt.title(r'$rect(t)$')

plt.subplot(4, 3, 9)
plt.plot(ts * nc, y_c)
plt.title(r'Convolución $(rampa(t) \cdot rect(t - 1/2)) \ast rect(t)$')

# Gráficos del inciso d
plt.subplot(4, 3, 10)
plt.plot(t, rampa_rect)
plt.title(r'd) $rampa(t) \cdot rect(t - 1/2)$')

plt.subplot(4, 3, 11)
plt.plot(t, exp_u)
plt.title(r'$exp(-t) \cdot u(t)$')

plt.subplot(4, 3, 12)
plt.plot(ts * nc, y_d)
plt.title(r'Convolución $(rampa(t) \cdot rect(t - 1/2)) \ast exp(-t) \cdot u(t)$')

plt.tight_layout()
plt.show()
