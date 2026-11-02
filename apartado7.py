
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CAPITULO 3 - APARTADO 7
# ANALISIS DE SENSIBILIDAD DEL CONTROL OPTIMO
# ============================================================

CARPETA = Path(__file__).resolve().parent
T = 4.0

X0 = -1.0
Y0 = -0.5

VALORES_K = [0.0, 0.25, 0.5, 0.75, 1.0]
t = np.linspace(0, T, 10001)


# ------------------------------------------------------------
# 1. SOLUCION ANALITICA PARA CUALQUIER k
# ------------------------------------------------------------

def solucion(k, tiempos):
    tiempos = np.asarray(tiempos, dtype=float)

    # Autovalores de la matriz de la dinamica
    a_mas = -0.5
    a_menos = -1.5

    # Condiciones iniciales en la base de autovectores
    q0_mas = (X0 + Y0) / np.sqrt(2)
    q0_menos = (X0 - Y0) / np.sqrt(2)

    def resolver_modo(a, q0):
        omega = np.sqrt(a**2 + k**2)

        D = (
            np.cosh(omega*T)
            - (a/omega)*np.sinh(omega*T)
        )

        tau = T - tiempos

        q = q0 * (
            np.cosh(omega*tau)
            - (a/omega)*np.sinh(omega*tau)
        ) / D

        p = (
            q0 * np.sinh(omega*tau)
            / (omega*D)
        )

        return q, p

    q_mas, p_mas = resolver_modo(a_mas, q0_mas)
    q_menos, p_menos = resolver_modo(a_menos, q0_menos)

    # Regreso a las variables originales
    X = (q_mas + q_menos) / np.sqrt(2)
    Y = (q_mas - q_menos) / np.sqrt(2)

    lambda1 = (p_mas + p_menos) / np.sqrt(2)
    lambda2 = (p_mas - p_menos) / np.sqrt(2)

    # Controles optimos de Pontryagin
    u = -k * lambda1
    v = -k * lambda2

    return X, Y, u, v, lambda1, lambda2


# ------------------------------------------------------------
# 2. INTEGRACION NUMERICA CON SIMPSON
# ------------------------------------------------------------

def integrar_simpson(valores, tiempos):
    """
    Regla compuesta de Simpson 1/3.
    Requiere un numero par de subintervalos.
    """
    h = tiempos[1] - tiempos[0]
    n = len(tiempos) - 1

    if n % 2 != 0:
        raise ValueError("Simpson requiere n par.")

    resultado = (
        valores[0]
        + valores[-1]
        + 4*np.sum(valores[1:-1:2])
        + 2*np.sum(valores[2:-1:2])
    )

    return h*resultado/3


# ------------------------------------------------------------
# 3. TIEMPO DE REDUCCION A LA MITAD
# ------------------------------------------------------------

def calcular_t_mitad(tiempos, X, Y):
    D = np.sqrt(X**2 + Y**2)
    objetivo = D[0] / 2

    indices = np.where(D <= objetivo)[0]

    if len(indices) == 0:
        return np.nan

    i = indices[0]

    if i == 0:
        return tiempos[0]

    # Interpolacion lineal en el primer cruce
    t1, t2 = tiempos[i-1], tiempos[i]
    d1, d2 = D[i-1], D[i]

    return t1 + (
        (objetivo - d1)*(t2 - t1)/(d2 - d1)
    )


# ------------------------------------------------------------
# 4. CALCULOS PARA LOS CINCO VALORES DE k
# ------------------------------------------------------------

resultados = []

for k in VALORES_K:

    X, Y, u, v, lambda1, lambda2 = solucion(k, t)

    # Coste de las desviaciones
    integrando_desviaciones = 0.5*(X**2 + Y**2)

    # Coste de la intervencion
    integrando_intervencion = 0.5*(u**2 + v**2)

    J_desv = integrar_simpson(
        integrando_desviaciones, t
    )

    J_int = integrar_simpson(
        integrando_intervencion, t
    )

    J_total = J_desv + J_int

    t_mitad = calcular_t_mitad(t, X, Y)

    resultados.append({
        "k": k,
        "X": X,
        "Y": Y,
        "u": u,
        "v": v,
        "J_desv": J_desv,
        "J_int": J_int,
        "J_total": J_total,
        "u0": u[0],
        "v0": v[0],
        "X4": X[-1],
        "Y4": Y[-1],
        "t_mitad": t_mitad
    })


# ------------------------------------------------------------
# 5. TABLA DE COSTES
# ------------------------------------------------------------

print("=" * 72)
print("APARTADO 7 - ANALISIS DE SENSIBILIDAD")
print("=" * 72)

print("\nTABLA 1. COSTES OPTIMOS")
print(
    f"{'k':>6} {'J_desv':>14} "
    f"{'J_int':>14} {'J_total':>14}"
)

for r in resultados:
    print(
        f"{r['k']:6.2f} "
        f"{r['J_desv']:14.6f} "
        f"{r['J_int']:14.6f} "
        f"{r['J_total']:14.6f}"
    )


# ------------------------------------------------------------
# 6. TABLA DE CONTROLES Y ESTADOS FINALES
# ------------------------------------------------------------

print("\nTABLA 2. CONTROLES INICIALES Y ESTADOS FINALES")

print(
    f"{'k':>6} {'u(0)':>12} {'v(0)':>12} "
    f"{'X(4)':>12} {'Y(4)':>12}"
)

for r in resultados:
    print(
        f"{r['k']:6.2f} "
        f"{r['u0']:12.6f} "
        f"{r['v0']:12.6f} "
        f"{r['X4']:12.6f} "
        f"{r['Y4']:12.6f}"
    )


# ------------------------------------------------------------
# 7. TABLA DE TIEMPOS DE ESTABILIZACION
# ------------------------------------------------------------

print("\nTABLA 3. TIEMPO DE REDUCCION A LA MITAD")

for r in resultados:
    print(
        f"k = {r['k']:.2f}  "
        f"t_1/2 = {r['t_mitad']:.6f} anos"
    )


# ------------------------------------------------------------
# 8. COMPARACION ENTRE k=0 Y k=1
# ------------------------------------------------------------

J0 = resultados[0]["J_total"]
J1 = resultados[-1]["J_total"]

reduccion = 100*(J0-J1)/J0

print("\nCOMPARACION FINAL")
print(f"Coste sin intervencion: {J0:.6f}")
print(f"Coste con k=1:          {J1:.6f}")
print(f"Reduccion del coste:    {reduccion:.2f} %")

# ------------------------------------------------------------
# 9. GRAFICA 1: COSTES EN FUNCION DE k
# ------------------------------------------------------------

ks = np.array([r["k"] for r in resultados])
costes_desv = np.array([r["J_desv"] for r in resultados])
costes_int = np.array([r["J_int"] for r in resultados])
costes_total = np.array([r["J_total"] for r in resultados])

plt.figure(figsize=(9, 5))

plt.plot(
    ks, costes_total, "o-",
    linewidth=2.2, label="Coste total"
)

plt.plot(
    ks, costes_desv, "s-",
    linewidth=2, label="Coste de desviaciones"
)

plt.plot(
    ks, costes_int, "^-",
    linewidth=2, label="Coste de intervencion"
)

plt.xlabel("Grado de intervencion k")
plt.ylabel("Coste acumulado")
plt.title("Analisis de sensibilidad: costes optimos")
plt.xticks(VALORES_K)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    CARPETA / "apartado7_costes.png",
    dpi=300, bbox_inches="tight"
)


# ------------------------------------------------------------
# 10. GRAFICA 2: TRAYECTORIAS X(t) PARA CADA k
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

for r in resultados:
    plt.plot(
        t, r["X"],
        linewidth=2,
        label=f"k = {r['k']:.2f}"
    )

plt.axhline(
    0, color="black", linestyle="--", linewidth=1
)

plt.xlabel("Tiempo (anos)")
plt.ylabel("Desviacion del PIB (puntos porcentuales)")
plt.title("Trayectorias optimas del PIB segun k")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    CARPETA / "apartado7_trayectorias.png",
    dpi=300, bbox_inches="tight"
)


# ------------------------------------------------------------
# 11. GRAFICA 3: POLITICAS INICIALES
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    ks,
    [r["u0"] for r in resultados],
    "o-", label="Politica inicial u(0)"
)

plt.plot(
    ks,
    [r["v0"] for r in resultados],
    "s-", label="Politica inicial v(0)"
)

plt.xlabel("Grado de intervencion k")
plt.ylabel("Intensidad inicial")
plt.title("Intensidad inicial de las politicas")
plt.xticks(VALORES_K)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    CARPETA / "apartado7_politicas.png",
    dpi=300, bbox_inches="tight"
)


# ------------------------------------------------------------
# 12. GRAFICA 4: TIEMPO DE ESTABILIZACION
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    ks,
    [r["t_mitad"] for r in resultados],
    "o-", linewidth=2
)

plt.xlabel("Grado de intervencion k")
plt.ylabel("Tiempo de reduccion a la mitad (anos)")
plt.title("Rapidez de estabilizacion")
plt.xticks(VALORES_K)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    CARPETA / "apartado7_tiempo_mitad.png",
    dpi=300, bbox_inches="tight"
)

print("\nGRAFICAS GUARDADAS EN:")
print(CARPETA)

print("\nArchivos:")
print("apartado7_costes.png")
print("apartado7_trayectorias.png")
print("apartado7_politicas.png")
print("apartado7_tiempo_mitad.png")

plt.show()
