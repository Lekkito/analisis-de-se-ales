#Importaciones
import numpy as np
import matplotlib.pyplot as plt

# Actividad 1, ejercicios c y f

# Defino la función g
def g(x):
    return np.maximum(2*(1 - np.abs((x-1)/2)), 0)

t = np.linspace(-18, 4, 300)

plt.plot(t,g(t), label='$g(t)$', color='b')
plt.plot(t, g(-t-5), label='c) $g(-t-5)$', color='r') # Aplico la transformacion c
plt.plot(t,2*g(-1/2*(t+10)), label='f) $2*g(-1/2*(t+10))$', color='g') # Transformacion f
plt.legend(loc=0)
plt.grid()
plt.xlabel('t')
plt.xlim(-18,4)
plt.xticks(np.arange(-18, 5, 2))
plt.title('Actividad 1')
plt.show()

# Actividad 2 - Transformacion b, d y e

# defino mi funcion h(n) 
def h(n):
    if n in (-4, 4):
        return 0.2
    elif n in (-3,3):
        return 0.4
    elif n in (-2, 2):
        return 0.6
    elif n in (-1, 1):
        return 0.8
    elif n == 0:
        return 1
    else:
        return 0
    
# Valores discretos
n_val = np.arange(-18, 9, 1)

# Grafico
fig, axs = plt.subplots(2, 2)

axs[0,0].stem(n_val, [h(n) for n in n_val],linefmt='k-', markerfmt='k.', basefmt=' ')
axs[0,0].set_title('$h[n]$')
axs[0,0].grid(True)

axs[0,1].stem(n_val, [h(2*n) for n in n_val], linefmt='b-', markerfmt='b.', basefmt=' ' )
axs[0,1].set_title('b) $h[2n]$')
axs[0,1].grid(True)

axs[1,0].stem(n_val, [h(n/2 + 4) for n in n_val], linefmt='r-',markerfmt='r.', basefmt=' ')
axs[1,0].set_title('d) $h[n/2 + 4]$')
axs[1,0].grid(True)

axs[1,1].stem(n_val, [h(-4*n+4) for n in n_val],linefmt='g-', markerfmt='g.', basefmt=' ')
axs[1,1].set_title('e) $h[-4n+4]$')
axs[1,1].grid(True)

plt.tight_layout()
plt.show()