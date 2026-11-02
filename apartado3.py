import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Apartado 3: cruce entre crecimiento del PIB y consumo.
# Se ejecuta en Windows, macOS o Linux.

r = np.sqrt(3.0)
C1 = (1.2 + 0.2 * np.exp(-4)) / (np.exp(4) - np.exp(-4))
C2 = -0.2 - C1
C3 = (2/15 + (7/15) * np.exp(-4*r)) / (np.exp(4*r) - np.exp(-4*r))
C4 = -7/15 - C3

def s(t):
    return C1*np.exp(t) + C2*np.exp(-t) + 0.1*t + 2.4

def d(t):
    return C3*np.exp(r*t) + C4*np.exp(-r*t) + 0.1*t - 2/15

def d_prima(t):
    return r*C3*np.exp(r*t) - r*C4*np.exp(-r*t) + 0.1

def x(t):
    return (s(t) + d(t))/2

def y(t):
    return (s(t) - d(t))/2

# Biseccion: criterio de parada por cota de error del punto medio
izq, der = 0.0, 4.0
tolerancia = 1e-6
iteraciones = 0
assert d(izq)*d(der) < 0
while (der-izq)/2 > tolerancia:
    medio = (izq+der)/2
    if d(izq)*d(medio) <= 0:
        der = medio
    else:
        izq = medio
    iteraciones += 1

t_estrella = (izq+der)/2
print(f'Iteraciones: {iteraciones}')
print(f'Intervalo final: [{izq:.12f}, {der:.12f}]')
print(f'Instante de cruce: {t_estrella:.9f} años')
print(f'Cota de error: {(der-izq)/2:.3e}')
print(f'd(t*): {d(t_estrella):.3e}')
print(f'x(t*): {x(t_estrella):.9f} %')
print(f'y(t*): {y(t_estrella):.9f} %')
print(f'd prima(0): {d_prima(0):.6f}')
print(f'd prima(4): {d_prima(4):.6f}')

t = np.linspace(0,4,500)
fig, ax = plt.subplots(figsize=(8,4.7))
ax.plot(t,x(t),label='PIB: x(t)',linewidth=2)
ax.plot(t,y(t),label='Consumo: y(t)',linewidth=2)
ax.axvline(t_estrella,color='gray',ls='--',label=f'Cruce: t = {t_estrella:.6f}')
ax.scatter([t_estrella],[x(t_estrella)],color='black',zorder=5)
ax.set(xlabel='Tiempo (años)',ylabel='Tasa de crecimiento (%)',title='Apartado 3: instante de cruce entre PIB y consumo',xlim=(0,4))
ax.grid(alpha=.3)
ax.legend()
fig.tight_layout()
# Guardar en la misma carpeta que este archivo Python
ruta = Path(__file__).resolve().parent / 'cruce_apartado3.png'
fig.savefig(ruta,dpi=220,bbox_inches='tight')
print(f'Gráfica guardada en: {ruta}')
plt.show()