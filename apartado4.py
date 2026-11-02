
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==================================================
# APARTADO 4: DEBILIDAD TRANSITORIA DEL CONSUMO
# Técnicas de Optimización y Control
# ==================================================

# 1. CONDICIONES DE CONTORNO

x0, y0 = 0.8, 1.4
x4, y4 = 2.2, 1.8

s0 = x0 + y0
s4 = x4 + y4
d0 = x0 - y0
d4 = x4 - y4

r = np.sqrt(3)

# 2. CALCULO DE LAS CONSTANTES

M1 = np.array([
    [1, 1],
    [np.exp(4), np.exp(-4)]
])

b1 = np.array([s0 - 2.4, s4 - 2.8])

C1, C2 = np.linalg.solve(M1, b1)

M2 = np.array([
    [1, 1],
    [np.exp(4*r), np.exp(-4*r)]
])

b2 = np.array([
    d0 + 2/15,
    d4 - 0.4 + 2/15
])

C3, C4 = np.linalg.solve(M2, b2)

print("CONSTANTES DE INTEGRACION")
print(f"C1 = {C1:.10f}")
print(f"C2 = {C2:.10f}")
print(f"C3 = {C3:.10f}")
print(f"C4 = {C4:.10f}")

# 3. FUNCIONES AUXILIARES

def s(t):
    return C1*np.exp(t) + C2*np.exp(-t) + 0.1*t + 2.4

def d(t):
    return C3*np.exp(r*t) + C4*np.exp(-r*t) + 0.1*t - 2/15

# 4. TRAYECTORIAS OPTIMAS

def x(t):
    return (s(t) + d(t))/2

def y(t):
    return (s(t) - d(t))/2

# 5. PRIMERA DERIVADA DEL CONSUMO

def dy(t):
    return 0.5 * (
        C1*np.exp(t)
        - C2*np.exp(-t)
        - r*C3*np.exp(r*t)
        + r*C4*np.exp(-r*t)
    )

# 6. SEGUNDA DERIVADA DEL CONSUMO

def ddy(t):
    return 0.5 * (
        C1*np.exp(t)
        + C2*np.exp(-t)
        - 3*C3*np.exp(r*t)
        - 3*C4*np.exp(-r*t)
    )

# 7. PRIMERA DERIVADA DEL PIB

def dx(t):
    return 0.5 * (
        C1*np.exp(t)
        - C2*np.exp(-t)
        + r*C3*np.exp(r*t)
        - r*C4*np.exp(-r*t)
    ) + 0.1

# 8. METODO DE NEWTON-RAPHSON

t = 1.0
tolerancia = 1e-10
max_iter = 100

print("\nMETODO DE NEWTON-RAPHSON")
print("Iteracion       t                y'(t)")

for i in range(max_iter):

    print(f"{i:3d}       {t:.12f}     {dy(t):.12e}")

    if abs(dy(t)) < tolerancia:
        break

    if abs(ddy(t)) < 1e-14:
        raise ValueError("Segunda derivada demasiado pequena")

    t_nuevo = t - dy(t)/ddy(t)

    if not (0 <= t_nuevo <= 4):
        raise ValueError("Newton ha salido del intervalo [0,4]")

    t = t_nuevo

else:
    raise RuntimeError("Newton no ha convergido")

tm = t
ym = y(tm)

# 9. RESULTADOS DEL MINIMO

print("\nRESULTADOS DEL CONSUMO")
print(f"Instante del minimo: {tm:.10f} anos")
print(f"Crecimiento minimo: {ym:.10f} %")
print(f"Primera derivada: {dy(tm):.12e}")
print(f"Segunda derivada: {ddy(tm):.10f}")
print(f"Iteraciones realizadas: {i}")

# 10. COMPARACION CON LOS EXTREMOS

print("\nCOMPARACION DE VALORES")
print(f"y(0)  = {y(0):.10f} %")
print(f"y(tm) = {y(tm):.10f} %")
print(f"y(4)  = {y(4):.10f} %")

# 11. COMPROBACION NUMERICA DE DERIVADAS

t_malla = np.linspace(0, 4, 10000)

segunda_derivada = ddy(t_malla)
derivada_pib = dx(t_malla)

print("\nCOMPROBACION NUMERICA")
print(f"Minimo muestreado de y''(t): "
      f"{np.min(segunda_derivada):.10f}")

print(f"Minimo muestreado de x'(t): "
      f"{np.min(derivada_pib):.10f}")

if np.all(segunda_derivada > 0):
    print("y''(t) es positiva en todos los puntos muestreados")

if np.all(derivada_pib > 0):
    print("x'(t) es positiva en todos los puntos muestreados")

# Nota: el muestreo numerico no sustituye
# una demostracion analitica de positividad.

# 12. REPRESENTACION GRAFICA

plt.figure(figsize=(9, 5))

plt.plot(
    t_malla, x(t_malla),
    label="Crecimiento del PIB",
    linewidth=2
)

plt.plot(
    t_malla, y(t_malla),
    label="Crecimiento del consumo",
    linewidth=2
)

# Marcar el minimo del consumo
plt.scatter(
    tm, ym,
    color="red",
    s=65,
    zorder=5,
    label="Minimo del consumo"
)

plt.annotate(
    f"Minimo: ({tm:.3f}, {ym:.3f}%)",
    xy=(tm, ym),
    xytext=(tm + 0.35, ym - 0.18),
    arrowprops=dict(arrowstyle="->"),
    fontsize=10
)

plt.xlabel("Tiempo (anos)")
plt.ylabel("Tasa de crecimiento (%)")
plt.title("Debilidad transitoria del consumo")

plt.xlim(0, 4)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# 13. GUARDAR LA GRAFICA

# Se guarda junto al archivo .py
carpeta = Path(__file__).resolve().parent

plt.savefig(
    carpeta / "minimo_consumo_apartado4.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    carpeta / "minimo_consumo_apartado4.pdf",
    bbox_inches="tight"
)

print("\nGraficas guardadas correctamente.")

# 14. MOSTRAR LA GRAFICA

plt.show()
