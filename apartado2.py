
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# APARTADO 2
# Trayectorias optimas del PIB y el consumo
# ==========================================

# 1. Condiciones de contorno
x0, y0 = 0.8, 1.4
x4, y4 = 2.2, 1.8

# 2. Condiciones para s = x + y y d = x - y
s0 = x0 + y0
s4 = x4 + y4
d0 = x0 - y0
d4 = x4 - y4

# 3. Calculo de las constantes C1 y C2
M1 = np.array([
    [1, 1],
    [np.exp(4), np.exp(-4)]
])

b1 = np.array([
    s0 - 2.4,
    s4 - 2.8
])

C1, C2 = np.linalg.solve(M1, b1)

# 4. Calculo de las constantes C3 y C4
r = np.sqrt(3)

M2 = np.array([
    [1, 1],
    [np.exp(4*r), np.exp(-4*r)]
])

b2 = np.array([
    d0 + 2/15,
    d4 - 0.4 + 2/15
])

C3, C4 = np.linalg.solve(M2, b2)

# 5. Definicion de las trayectorias
def s(t):
    return C1*np.exp(t) + C2*np.exp(-t) + 0.1*t + 2.4

def d(t):
    return C3*np.exp(r*t) + C4*np.exp(-r*t) + 0.1*t - 2/15

def x(t):
    return (s(t) + d(t))/2

def y(t):
    return (s(t) - d(t))/2

# 6. Valores de las constantes
print("CONSTANTES DE INTEGRACION")
print(f"C1 = {C1:.10f}")
print(f"C2 = {C2:.10f}")
print(f"C3 = {C3:.10f}")
print(f"C4 = {C4:.10f}")

# 7. Comprobacion de las condiciones
print("\nCOMPROBACION DE CONTORNO")
print(f"x(0) = {x(0):.6f}")
print(f"y(0) = {y(0):.6f}")
print(f"x(4) = {x(4):.6f}")
print(f"y(4) = {y(4):.6f}")

# 8. Representacion grafica
t = np.linspace(0, 4, 500)

plt.figure(figsize=(9, 5))

plt.plot(t, x(t), label="Crecimiento del PIB", linewidth=2)
plt.plot(t, y(t), label="Crecimiento del consumo", linewidth=2)

plt.scatter([0, 4], [x0, x4], color="C0", zorder=3)
plt.scatter([0, 4], [y0, y4], color="C1", zorder=3)

plt.xlabel("Tiempo (años)")
plt.ylabel("Tasa de crecimiento (%)")
plt.title("Trayectorias optimas del PIB y del consumo")

plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 4)
plt.tight_layout()

# 9. Guardar la grafica en la misma carpeta
# donde se ejecuta el programa
plt.savefig("trayectorias_apartado2.png", dpi=300,
            bbox_inches="tight")

plt.savefig("trayectorias_apartado2.pdf",
            bbox_inches="tight")

# 10. Mostrar la grafica
plt.show()
