
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================================
# CAPITULO 3 - APARTADO 6
# CONTROL OPTIMO Y POLITICA DE ESTABILIZACION
# ==========================================================

CARPETA = Path(__file__).resolve().parent

T = 4.0
xd = 1.5
yd = 1.5
X0 = -1.0
Y0 = -0.5

t = np.linspace(0, T, 1001)

# ----------------------------------------------------------
# 1. SOLUCION ANALITICA PARA CUALQUIER k
# ----------------------------------------------------------

def solucion(k, tiempos):
    tiempos = np.asarray(tiempos, dtype=float)

    # Autovalores de la matriz de la dinamica
    a_mas = -0.5
    a_menos = -1.5

    # Condiciones iniciales en la base de autovectores
    q0_mas = (X0 + Y0) / np.sqrt(2)
    q0_menos = (X0 - Y0) / np.sqrt(2)

    def modo(a, q0):
        omega = np.sqrt(a**2 + k**2)

        # Formula normalizada para evitar numeros grandes
        # D = cosh(omega*T) - (a/omega)*sinh(omega*T)
        D = (
            np.cosh(omega*T)
            - (a/omega)*np.sinh(omega*T)
        )

        tau = T - tiempos

        q = q0 * (
            np.cosh(omega*tau)
            - (a/omega)*np.sinh(omega*tau)
        ) / D

        p = q0 * np.sinh(omega*tau) / (omega*D)

        return q, p

    q_mas, p_mas = modo(a_mas, q0_mas)
    q_menos, p_menos = modo(a_menos, q0_menos)

    # Regreso a las variables originales
    X = (q_mas + q_menos) / np.sqrt(2)
    Y = (q_mas - q_menos) / np.sqrt(2)

    lam1 = (p_mas + p_menos) / np.sqrt(2)
    lam2 = (p_mas - p_menos) / np.sqrt(2)

    # Condiciones de optimalidad de Pontryagin
    u = -k * lam1
    v = -k * lam2

    return X, Y, u, v, lam1, lam2


# ----------------------------------------------------------
# 2. SOLUCIONES CON Y SIN INTERVENCION
# ----------------------------------------------------------

X1, Y1, u1, v1, l1, l2 = solucion(1.0, t)
Xsin, Ysin, usin, vsin, _, _ = solucion(0.0, t)

# Tasas reales de crecimiento
pib = xd + X1
consumo = yd + Y1

# ----------------------------------------------------------
# 3. VERIFICACION DE LAS CONDICIONES
# ----------------------------------------------------------

print("=" * 65)
print("APARTADO 6: VERIFICACION")
print("=" * 65)

print(f"X(0) = {X1[0]:.10f}  (debe ser -1)")
print(f"Y(0) = {Y1[0]:.10f}  (debe ser -0.5)")

print(f"lambda1(4) = {l1[-1]:.10e}")
print(f"lambda2(4) = {l2[-1]:.10e}")

print(f"u(4) = {u1[-1]:.10e}")
print(f"v(4) = {v1[-1]:.10e}")

print("\nVALORES EN INSTANTES REPRESENTATIVOS")
print(
    f"{'t':>5} {'X':>12} {'Y':>12} "
    f"{'u':>12} {'v':>12}"
)

for ti in [0, 1, 2, 3, 4]:
    X, Y, u, v, _, _ = solucion(1.0, [ti])
    print(
        f"{ti:5.1f} {X[0]:12.6f} {Y[0]:12.6f} "
        f"{u[0]:12.6f} {v[0]:12.6f}"
    )

# Verificacion de las ecuaciones diferenciales
# Se usa la derivada numerica en los puntos interiores
dX = np.gradient(X1, t, edge_order=2)
dY = np.gradient(Y1, t, edge_order=2)

rhsX = -X1 + 0.5*Y1 + u1
rhsY = 0.5*X1 - Y1 + v1

print("\nRESIDUOS DE LAS ECUACIONES DE ESTADO")
print(
    "Error maximo ecuacion X:",
    np.max(np.abs(dX[1:-1] - rhsX[1:-1]))
)
print(
    "Error maximo ecuacion Y:",
    np.max(np.abs(dY[1:-1] - rhsY[1:-1]))
)

# ----------------------------------------------------------
# 4. COSTE OPTIMO PARA k=1 Y k=0
# ----------------------------------------------------------

def coste(X, Y, u, v):
    desviaciones = 0.5*(X**2 + Y**2)
    intervencion = 0.5*(u**2 + v**2)

    Jdesv = np.trapezoid(desviaciones, t)
    Jint = np.trapezoid(intervencion, t)

    return Jdesv, Jint, Jdesv + Jint


print("\nCOSTES OPTIMOS")

for k, X, Y, u, v in [
    (0, Xsin, Ysin, usin, vsin),
    (1, X1, Y1, u1, v1)
]:
    Jd, Ji, J = coste(X, Y, u, v)
    print(
        f"k={k}: Jdesv={Jd:.8f}, "
        f"Jinterv={Ji:.8f}, Jtotal={J:.8f}"
    )

# ----------------------------------------------------------
# 5. GRAFICA 1: DESVIACIONES OPTIMAS
# ----------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(t, X1, label="Desviacion PIB X(t)", linewidth=2)
plt.plot(t, Y1, label="Desviacion consumo Y(t)", linewidth=2)

plt.axhline(0, color="black", linestyle="--", linewidth=1)

plt.xlabel("Tiempo (anos)")
plt.ylabel("Desviacion (puntos porcentuales)")
plt.title("Desviaciones optimas con k=1")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    CARPETA / "apartado6_desviaciones.png",
    dpi=300, bbox_inches="tight"
)

# ----------------------------------------------------------
# 6. GRAFICA 2: POLITICAS OPTIMAS
# ----------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(t, u1, label="Politica sobre PIB u(t)", linewidth=2)
plt.plot(t, v1, label="Politica sobre consumo v(t)", linewidth=2)

plt.axhline(0, color="black", linestyle="--", linewidth=1)

plt.xlabel("Tiempo (anos)")
plt.ylabel("Intensidad de intervencion")
plt.title("Politicas optimas con k=1")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    CARPETA / "apartado6_controles.png",
    dpi=300, bbox_inches="tight"
)

# ----------------------------------------------------------
# 7. GRAFICA 3: COMPARACION k=0 Y k=1
# ----------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(t, Xsin, "--", label="PIB sin intervencion")
plt.plot(t, X1, label="PIB con intervencion")

plt.plot(t, Ysin, "--", label="Consumo sin intervencion")
plt.plot(t, Y1, label="Consumo con intervencion")

plt.axhline(0, color="black", linewidth=1)

plt.xlabel("Tiempo (anos)")
plt.ylabel("Desviacion (puntos porcentuales)")
plt.title("Comparacion de intervencion k=0 y k=1")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    CARPETA / "apartado6_comparacion.png",
    dpi=300, bbox_inches="tight"
)

# ----------------------------------------------------------
# 8. GRAFICA 4: TASAS REALES DE CRECIMIENTO
# ----------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(t, pib, label="Crecimiento PIB", linewidth=2)
plt.plot(t, consumo, label="Crecimiento consumo", linewidth=2)

plt.axhline(
    1.5, color="black", linestyle="--",
    label="Referencia: 1.5 %"
)

plt.xlabel("Tiempo (anos)")
plt.ylabel("Tasa de crecimiento (%)")
plt.title("Recuperacion economica con k=1")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    CARPETA / "apartado6_crecimiento.png",
    dpi=300, bbox_inches="tight"
)

print("\nSe han generado cuatro graficas en:")
print(CARPETA)

plt.show()
