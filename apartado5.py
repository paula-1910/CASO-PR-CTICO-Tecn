
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CAPITULO 2 - APARTADO 5
# CONTRASTE DEL MODELO CON LOS DATOS
# ============================================================

# Carpeta donde se guardaran las graficas
CARPETA = Path(__file__).resolve().parent

# ------------------------------------------------------------
# 1. DATOS DEL ENUNCIADO
# ------------------------------------------------------------

t = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0,
    2.5, 3.0, 3.5, 4.0
])

pib_obs = np.array([
    0.80, 0.95, 1.15, 1.31, 1.47,
    1.65, 1.84, 2.03, 2.20
])

consumo_obs = np.array([
    1.40, 1.46, 1.50, 1.55, 1.61,
    1.67, 1.70, 1.75, 1.80
])

inflacion = np.array([
    3.6, 3.2, 2.9, 2.6, 2.4,
    2.2, 2.1, 2.0, 2.0
])

desempleo = np.array([
    6.8, 6.7, 6.6, 6.5, 6.4,
    6.3, 6.2, 6.1, 6.0
])

interes = np.array([
    4.00, 4.00, 3.75, 3.50, 3.25,
    3.00, 2.75, 2.50, 2.50
])

# ------------------------------------------------------------
# 2. CONSTANTES DEL CAPITULO 1
# ------------------------------------------------------------

r = np.sqrt(3)

# Sistema correspondiente a s(t) = x(t) + y(t)
M_s = np.array([
    [1.0, 1.0],
    [np.exp(4), np.exp(-4)]
])

b_s = np.array([-0.2, 1.2])

C1, C2 = np.linalg.solve(M_s, b_s)

# Sistema correspondiente a d(t) = x(t) - y(t)
M_d = np.array([
    [1.0, 1.0],
    [np.exp(4*r), np.exp(-4*r)]
])

b_d = np.array([-7/15, 2/15])

C3, C4 = np.linalg.solve(M_d, b_d)

print("=" * 60)
print("CONSTANTES DE INTEGRACION")
print("=" * 60)

print(f"C1 = {C1:.10f}")
print(f"C2 = {C2:.10f}")
print(f"C3 = {C3:.10f}")
print(f"C4 = {C4:.10f}")

# ------------------------------------------------------------
# 3. TRAYECTORIAS OPTIMAS DEL CAPITULO 1
# ------------------------------------------------------------

def x_modelo(t):
    """Tasa de crecimiento del PIB."""
    return (
        0.5 * (
            C1*np.exp(t)
            + C2*np.exp(-t)
            + C3*np.exp(r*t)
            + C4*np.exp(-r*t)
        )
        + 0.10*t
        + 17/15
    )


def y_modelo(t):
    """Tasa de crecimiento del consumo."""
    return (
        0.5 * (
            C1*np.exp(t)
            + C2*np.exp(-t)
            - C3*np.exp(r*t)
            - C4*np.exp(-r*t)
        )
        + 19/15
    )


def diferencia_modelo(t):
    return x_modelo(t) - y_modelo(t)


# Valores del modelo en los instantes observados
pib_mod = x_modelo(t)
consumo_mod = y_modelo(t)

# ------------------------------------------------------------
# 4. APARTADO 5(a): RESIDUOS
# ------------------------------------------------------------

# Residuo = dato observado - prediccion del modelo

res_pib = pib_obs - pib_mod
res_consumo = consumo_obs - consumo_mod

print("\n" + "=" * 60)
print("APARTADO 5(a): TABLA DE RESIDUOS")
print("=" * 60)

print(
    f"{'t':>5} {'PIB obs':>10} {'PIB mod':>10} "
    f"{'e PIB':>10} {'Cons obs':>10} "
    f"{'Cons mod':>10} {'e Cons':>10}"
)

for i in range(len(t)):
    print(
        f"{t[i]:5.1f} "
        f"{pib_obs[i]:10.5f} "
        f"{pib_mod[i]:10.5f} "
        f"{res_pib[i]:10.5f} "
        f"{consumo_obs[i]:10.5f} "
        f"{consumo_mod[i]:10.5f} "
        f"{res_consumo[i]:10.5f}"
    )

# ------------------------------------------------------------
# 5. MEDIDAS DE ERROR
# ------------------------------------------------------------

def calcular_errores(residuos):
    error_medio = np.mean(residuos)
    mae = np.mean(np.abs(residuos))
    rmse = np.sqrt(np.mean(residuos**2))
    error_maximo = np.max(np.abs(residuos))

    return error_medio, mae, rmse, error_maximo


errores_pib = calcular_errores(res_pib)
errores_consumo = calcular_errores(res_consumo)

print("\n" + "=" * 60)
print("MEDIDAS DE ERROR")
print("=" * 60)

print(f"{'Medida':<20} {'PIB':>12} {'Consumo':>12}")

nombres = [
    "Error medio",
    "MAE",
    "RMSE",
    "Error maximo"
]

for i, nombre in enumerate(nombres):
    print(
        f"{nombre:<20} "
        f"{errores_pib[i]:12.6f} "
        f"{errores_consumo[i]:12.6f}"
    )

# Instantes de mayor desviacion
i_pib = np.argmax(np.abs(res_pib))
i_consumo = np.argmax(np.abs(res_consumo))

print("\nMAYORES DESVIACIONES")

print(
    f"PIB: t = {t[i_pib]:.1f}, "
    f"error = {res_pib[i_pib]:.6f}"
)

print(
    f"Consumo: t = {t[i_consumo]:.1f}, "
    f"error = {res_consumo[i_consumo]:.6f}"
)

if errores_pib[2] < errores_consumo[2]:
    print("\nEl PIB presenta un RMSE menor.")
    print("El modelo representa mejor el PIB.")
else:
    print("\nEl consumo presenta un RMSE menor.")
    print("El modelo representa mejor el consumo.")

# ------------------------------------------------------------
# 6. APARTADO 5(b): CRUCE OBSERVADO
# ------------------------------------------------------------

# Diferencia PIB - consumo en los datos
d_obs = pib_obs - consumo_obs

# Buscar intervalos con cambio de signo
intervalos_cruce = []

for i in range(len(t) - 1):
    if d_obs[i] * d_obs[i+1] < 0:
        intervalos_cruce.append(i)

if len(intervalos_cruce) != 1:
    raise ValueError(
        "Se esperaba un unico intervalo con cambio de signo."
    )

i = intervalos_cruce[0]

# Interpolacion lineal de la diferencia
t_cruce_obs = (
    t[i]
    - d_obs[i] * (t[i+1] - t[i])
    / (d_obs[i+1] - d_obs[i])
)

# Cruce del modelo mediante biseccion
def biseccion(f, a, b, tolerancia=1e-12):
    fa = f(a)
    fb = f(b)

    if fa * fb >= 0:
        raise ValueError("No hay cambio de signo.")

    iteraciones = 0

    while (b - a)/2 > tolerancia:
        m = (a + b)/2
        fm = f(m)

        if fm == 0:
            return m, iteraciones + 1

        if fa * fm < 0:
            b = m
            fb = fm
        else:
            a = m
            fa = fm

        iteraciones += 1

    return (a + b)/2, iteraciones


t_cruce_modelo, n_iter = biseccion(
    diferencia_modelo, 0, 4
)

print("\n" + "=" * 60)
print("APARTADO 5(b): CRUCE PIB - CONSUMO")
print("=" * 60)

print(f"Cruce observado interpolado: {t_cruce_obs:.9f}")
print(f"Cruce del modelo:           {t_cruce_modelo:.9f}")
print(
    f"Diferencia temporal:        "
    f"{t_cruce_obs - t_cruce_modelo:.9f} anos"
)
print(f"Iteraciones biseccion:      {n_iter}")

# ------------------------------------------------------------
# 7. MINIMO DEL CONSUMO DEL MODELO
# ------------------------------------------------------------

def derivada_consumo(t):
    return 0.5 * (
        C1*np.exp(t)
        - C2*np.exp(-t)
        - r*C3*np.exp(r*t)
        + r*C4*np.exp(-r*t)
    )


def segunda_consumo(t):
    return 0.5 * (
        C1*np.exp(t)
        + C2*np.exp(-t)
        - 3*C3*np.exp(r*t)
        - 3*C4*np.exp(-r*t)
    )


# Newton-Raphson para y'(t) = 0
tm = 1.0

for n in range(100):
    nuevo = (
        tm
        - derivada_consumo(tm)/segunda_consumo(tm)
    )

    if abs(nuevo - tm) < 1e-12:
        tm = nuevo
        break

    tm = nuevo

ym = y_modelo(tm)

# Minimo en los datos
i_min_obs = np.argmin(consumo_obs)

print("\nMINIMO DEL CONSUMO")

print(f"Minimo modelo: t = {tm:.9f}")
print(f"Valor modelo:    {ym:.9f} %")

print(
    f"Minimo observado: t = {t[i_min_obs]:.1f}"
)
print(
    f"Valor observado:    {consumo_obs[i_min_obs]:.2f} %"
)

# ------------------------------------------------------------
# 8. APARTADO 5(c): CORRELACIONES
# ------------------------------------------------------------

variables = {
    "Inflacion": inflacion,
    "Desempleo": desempleo,
    "Tipo de interes": interes
}

print("\n" + "=" * 60)
print("APARTADO 5(c): CORRELACIONES DE PEARSON")
print("=" * 60)

print(
    f"{'Variable':<20} "
    f"{'Residuos PIB':>15} "
    f"{'Residuos consumo':>18}"
)

for nombre, datos in variables.items():

    corr_pib = np.corrcoef(
        res_pib, datos
    )[0, 1]

    corr_consumo = np.corrcoef(
        res_consumo, datos
    )[0, 1]

    print(
        f"{nombre:<20} "
        f"{corr_pib:15.6f} "
        f"{corr_consumo:18.6f}"
    )

# ------------------------------------------------------------
# 9. GRAFICA 1: DATOS FRENTE AL MODELO
# ------------------------------------------------------------

t_fino = np.linspace(0, 4, 1000)

plt.figure(figsize=(9, 5))

plt.plot(
    t_fino, x_modelo(t_fino),
    label="PIB - modelo",
    linewidth=2
)

plt.scatter(
    t, pib_obs,
    label="PIB - datos",
    marker="o",
    s=45
)

plt.plot(
    t_fino, y_modelo(t_fino),
    label="Consumo - modelo",
    linewidth=2
)

plt.scatter(
    t, consumo_obs,
    label="Consumo - datos",
    marker="s",
    s=45
)

plt.xlabel("Tiempo (anos)")
plt.ylabel("Tasa de crecimiento (%)")
plt.title("Contraste del modelo con los datos")

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    CARPETA / "apartado5_ajuste.png",
    dpi=300,
    bbox_inches="tight"
)

# ------------------------------------------------------------
# 10. GRAFICA 2: RESIDUOS
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.axhline(
    0,
    color="black",
    linestyle="--",
    linewidth=1
)

plt.plot(
    t, res_pib,
    "o-",
    label="Residuos PIB"
)

plt.plot(
    t, res_consumo,
    "s-",
    label="Residuos consumo"
)

plt.xlabel("Tiempo (anos)")
plt.ylabel("Residuo (puntos porcentuales)")
plt.title("Errores del modelo")

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    CARPETA / "apartado5_residuos.png",
    dpi=300,
    bbox_inches="tight"
)

# ------------------------------------------------------------
# 11. GRAFICA 3: COMPARACION DE CRUCES
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    t_fino,
    diferencia_modelo(t_fino),
    label="PIB - consumo (modelo)",
    linewidth=2
)

plt.plot(
    t,
    d_obs,
    "o--",
    label="PIB - consumo (datos)"
)

plt.axhline(
    0,
    color="black",
    linewidth=1
)

plt.axvline(
    t_cruce_modelo,
    linestyle=":",
    label=f"Cruce modelo: {t_cruce_modelo:.3f}"
)

plt.axvline(
    t_cruce_obs,
    linestyle="--",
    label=f"Cruce datos: {t_cruce_obs:.3f}"
)

plt.xlabel("Tiempo (anos)")
plt.ylabel("Diferencia PIB - consumo")
plt.title("Comparacion de instantes de cruce")

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    CARPETA / "apartado5_cruces.png",
    dpi=300,
    bbox_inches="tight"
)

# ------------------------------------------------------------
# 12. GRAFICA 4: MINIMO DEL CONSUMO
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    t_fino,
    y_modelo(t_fino),
    label="Consumo - modelo",
    linewidth=2
)

plt.plot(
    t,
    consumo_obs,
    "o--",
    label="Consumo - datos"
)

plt.scatter(
    tm, ym,
    color="red",
    s=80,
    zorder=5,
    label=f"Minimo modelo: t={tm:.3f}"
)

plt.scatter(
    t[i_min_obs],
    consumo_obs[i_min_obs],
    color="black",
    marker="s",
    s=70,
    zorder=5,
    label="Minimo observado"
)

plt.xlabel("Tiempo (anos)")
plt.ylabel("Crecimiento del consumo (%)")
plt.title("Minimo teorico frente a datos")

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    CARPETA / "apartado5_minimo.png",
    dpi=300,
    bbox_inches="tight"
)

# ------------------------------------------------------------
# 13. FINAL
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("GRAFICAS GENERADAS")
print("=" * 60)

print("apartado5_ajuste.png")
print("apartado5_residuos.png")
print("apartado5_cruces.png")
print("apartado5_minimo.png")

print(f"\nGuardadas en: {CARPETA}")

plt.show()
