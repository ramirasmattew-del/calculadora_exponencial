
import math

import numpy as np
import pandas as pd
import streamlit as st


# ==========================================================
# CALCULADORA DE DISTRIBUCION EXPONENCIAL
# Probabilidad - Funcion de densidad
# Aplicacion web con Streamlit
# ==========================================================

st.set_page_config(
    page_title="Calculadora Exponencial",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Calculadora de Distribucion Exponencial")
st.caption(
    "Herramienta de probabilidad continua: densidad, "
    "distribucion acumulada, probabilidades y estadisticos."
)

st.markdown("""
La distribucion exponencial modela el tiempo de espera entre
eventos que ocurren a una tasa constante e independiente.
La variable aleatoria debe ser no negativa y **lambda debe ser mayor que cero**.
""")

# ----------------------------------------------------------
# 1. PARAMETROS
# ----------------------------------------------------------

st.sidebar.header("⚙️ Configuracion")

metodo = st.sidebar.selectbox(
    "Como deseas calcular lambda?",
    [
        "Ingresar lambda directamente",
        "Usar la media o tiempo promedio",
        "Usar la varianza",
        "Usar la desviacion estandar"
    ]
)

if metodo == "Ingresar lambda directamente":
    lam = st.sidebar.number_input(
        "Lambda (tasa)",
        min_value=0.000001,
        value=0.2,
        step=0.1,
        format="%.6f"
    )

elif metodo == "Usar la media o tiempo promedio":
    media_ingresada = st.sidebar.number_input(
        "Media o tiempo promedio",
        min_value=0.000001,
        value=5.0,
        step=1.0
    )
    lam = 1 / media_ingresada

elif metodo == "Usar la varianza":
    var_ingresada = st.sidebar.number_input(
        "Varianza",
        min_value=0.000001,
        value=25.0,
        step=1.0
    )
    lam = 1 / math.sqrt(var_ingresada)

else:
    sd_ingresada = st.sidebar.number_input(
        "Desviacion estandar",
        min_value=0.000001,
        value=5.0,
        step=1.0
    )
    lam = 1 / sd_ingresada

media = 1 / lam
varianza = 1 / (lam ** 2)
desviacion = 1 / lam
mediana = math.log(2) / lam

st.sidebar.divider()
st.sidebar.write("**Unidad de medida**")
unidad = st.sidebar.text_input(
    "Ej. minutos, horas, dias",
    value="minutos"
)
st.sidebar.caption(
    f"Lambda se expresa en 1/{unidad or 'unidad de tiempo'}."
)

# ----------------------------------------------------------
# 2. RESUMEN ESTADISTICO
# ----------------------------------------------------------

st.subheader("Parametros de la distribucion")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Lambda (λ)", f"{lam:.6f}")
c2.metric("Media", f"{media:.4f} {unidad}")
c3.metric("Varianza", f"{varianza:.4f}")
c4.metric("Desviacion estandar", f"{desviacion:.4f} {unidad}")
c5.metric("Mediana", f"{mediana:.4f} {unidad}")

with st.expander("Ver formulas estadisticas"):
    st.latex(r"f(x)=\lambda e^{-\lambda x}, \quad x\geq 0")
    st.latex(r"F(x)=P(X\leq x)=1-e^{-\lambda x}")
    st.latex(r"E(X)=\frac{1}{\lambda}")
    st.latex(r"\operatorname{Var}(X)=\frac{1}{\lambda^2}")
    st.latex(r"\sigma=\frac{1}{\lambda}")
    st.latex(r"\operatorname{Mediana}=\frac{\ln(2)}{\lambda}")

st.divider()

# ----------------------------------------------------------
# 3. FUNCIONES AUXILIARES
# ----------------------------------------------------------

def densidad(x):
    return lam * math.exp(-lam * x)


def acumulada(x):
    return -math.expm1(-lam * x)


def cola_derecha(x):
    return math.exp(-lam * x)


def probabilidad_intervalo(a, b):
    # Formula numericamente estable:
    # P(a < X < b) = exp(-lambda*a) - exp(-lambda*b)
    return math.exp(-lam * a) * (
        -math.expm1(-lam * (b - a))
    )


def mostrar_probabilidad(nombre, prob, formula):
    st.metric(nombre, f"{prob:.6f} ({prob * 100:.4f}%)")
    st.latex(formula)


def crear_grafica(x_max):
    x_max = max(float(x_max), 5 / lam, 0.001)
    x_vals = np.linspace(0, x_max, 400)

    datos = pd.DataFrame({
        "x": x_vals,
        "Densidad f(x)": lam * np.exp(-lam * x_vals),
        "Acumulada F(x)": 1 - np.exp(-lam * x_vals)
    })

    return datos.set_index("x")


# ----------------------------------------------------------
# 4. PESTANAS
# ----------------------------------------------------------

tab_prob, tab_dens, tab_dist, tab_percentil, tab_teoria = st.tabs([
    "🎯 Probabilidades",
    "📉 Funcion de densidad",
    "📈 Distribucion acumulada",
    "🔎 Percentiles",
    "📚 Teoria y formulas"
])


# ==========================================================
# TAB 1: PROBABILIDADES
# ==========================================================

with tab_prob:
    st.header("Calculo de probabilidades")

    tipo = st.selectbox(
        "Selecciona la probabilidad que necesitas",
        [
            "P(X ≤ x): menor o igual que x",
            "P(X > x): mayor que x",
            "P(a < X < b): entre dos valores",
            "P(X ≥ x): mayor o igual que x",
            "P(X = x): igual a un valor exacto"
        ]
    )

    if tipo == "P(a < X < b): entre dos valores":
        ca, cb = st.columns(2)

        with ca:
            a = st.number_input(
                f"Limite inferior a ({unidad})",
                min_value=0.0,
                value=2.0,
                step=0.5
            )

        with cb:
            b = st.number_input(
                f"Limite superior b ({unidad})",
                min_value=0.0,
                value=5.0,
                step=0.5
            )

        if b <= a:
            st.error("El limite superior b debe ser mayor que a.")
        else:
            p = probabilidad_intervalo(a, b)
            st.success("Resultado del calculo")
            mostrar_probabilidad(
                f"P({a:g} < X < {b:g})",
                p,
                rf"P(a<X<b)=e^{{-\lambda a}}-e^{{-\lambda b}}"
            )
            st.info(
                f"La probabilidad de que el tiempo este entre "
                f"{a:g} y {b:g} {unidad} es {p*100:.4f}%."
            )

    elif tipo == "P(X = x): igual a un valor exacto":
        x = st.number_input(
            f"Valor x ({unidad})",
            min_value=0.0,
            value=3.0,
            step=0.5
        )
        st.metric("P(X = x)", "0 (0%)")
        st.info(
            "En una variable aleatoria continua, la probabilidad "
            "de un unico valor exacto es cero. La densidad f(x) "
            "no es lo mismo que la probabilidad P(X = x)."
        )

    else:
        x = st.number_input(
            f"Valor x ({unidad})",
            min_value=0.0,
            value=3.0,
            step=0.5
        )

        if tipo == "P(X ≤ x): menor o igual que x":
            p = acumulada(x)
            formula = r"P(X\leq x)=1-e^{-\lambda x}"
            nombre = f"P(X ≤ {x:g})"

        elif tipo == "P(X > x): mayor que x":
            p = cola_derecha(x)
            formula = r"P(X>x)=e^{-\lambda x}"
            nombre = f"P(X > {x:g})"

        else:
            p = cola_derecha(x)
            formula = r"P(X\geq x)=e^{-\lambda x}"
            nombre = f"P(X ≥ {x:g})"

        st.success("Resultado del calculo")
        mostrar_probabilidad(nombre, p, formula)
        st.info(
            f"La probabilidad equivale a {p*100:.4f}%."
        )

    st.caption(
        "En la distribucion exponencial continua, "
        "P(X ≥ x) = P(X > x) y P(X ≤ x) = P(X < x)."
    )


# ==========================================================
# TAB 2: FUNCION DE DENSIDAD
# ==========================================================

with tab_dens:
    st.header("Funcion de densidad de probabilidad")

    st.latex(r"f(x)=\begin{cases}"
             r"\lambda e^{-\lambda x}, & x\geq 0\\"
             r"0, & x<0"
             r"\end{cases}")

    x_dens = st.number_input(
        f"Ingresa el valor de x ({unidad})",
        min_value=0.0,
        value=3.0,
        step=0.5,
        key="x_dens"
    )

    fx = densidad(x_dens)

    st.metric(f"Densidad f({x_dens:g})", f"{fx:.8f}")

    st.latex(
        rf"f({x_dens:g})={lam:.6f}"
        rf"\,e^{{-{lam:.6f}\times {x_dens:g}}}"
        rf"={fx:.8f}"
    )

    st.warning(
        "Importante: f(x) es una densidad, no una probabilidad. "
        "Puede ser mayor que 1. Las probabilidades se obtienen "
        "mediante el area bajo la curva en un intervalo."
    )

    grafica = crear_grafica(max(5 / lam, x_dens * 1.5))
    st.subheader("Grafica de la funcion de densidad")
    st.line_chart(grafica[["Densidad f(x)"]])

    st.caption(
        "El area total bajo la curva de densidad es igual a 1."
    )


# ==========================================================
# TAB 3: DISTRIBUCION ACUMULADA
# ==========================================================

with tab_dist:
    st.header("Funcion de distribucion acumulada")

    st.latex(r"F(x)=P(X\leq x)=1-e^{-\lambda x}")

    x_acum = st.number_input(
        f"Ingresa x ({unidad})",
        min_value=0.0,
        value=3.0,
        step=0.5,
        key="x_acum"
    )

    F = acumulada(x_acum)
    S = cola_derecha(x_acum)

    c1, c2 = st.columns(2)

    with c1:
        st.metric("P(X ≤ x)", f"{F:.6f} ({F*100:.4f}%)")

    with c2:
        st.metric("P(X > x)", f"{S:.6f} ({S*100:.4f}%)")

    st.latex(
        rf"F({x_acum:g})=1-e^{{-{lam:.6f}\times"
        rf"{x_acum:g}}}={F:.8f}"
    )

    grafica = crear_grafica(max(5 / lam, x_acum * 1.5))
    st.subheader("Grafica de la distribucion acumulada")
    st.line_chart(grafica[["Acumulada F(x)"]])

    st.caption(
        "F(x) siempre se encuentra entre 0 y 1 y nunca decrece."
    )


# ==========================================================
# TAB 4: PERCENTILES
# ==========================================================

with tab_percentil:
    st.header("Calcular x a partir de una probabilidad")

    st.write(
        "Encuentra el tiempo x para el que la probabilidad "
        "acumulada sea igual a un valor p."
    )

    st.latex(r"F(x)=p\quad\Rightarrow\quad"
             r"x=\frac{-\ln(1-p)}{\lambda}")

    p = st.number_input(
        "Probabilidad acumulada p (entre 0 y 1)",
        min_value=0.0,
        max_value=0.999999,
        value=0.90,
        step=0.01,
        format="%.6f"
    )

    x_percentil = -math.log1p(-p) / lam

    st.metric(
        "Valor x correspondiente",
        f"{x_percentil:.6f} {unidad}"
    )

    st.write(f"Percentil: {p*100:.4f}%")

    st.latex(
        rf"x=\frac{{-\ln(1-{p:.6f})}}{{{lam:.6f}}}"
        rf"={x_percentil:.6f}"
    )

    st.info(
        f"Interpretacion: aproximadamente el {p*100:.2f}% "
        f"de los tiempos son menores o iguales a "
        f"{x_percentil:.4f} {unidad}."
    )


# ==========================================================
# TAB 5: TEORIA
# ==========================================================

with tab_teoria:
    st.header("Guia teorica: distribucion exponencial")

    st.subheader("1. Funcion de densidad")
    st.latex(r"f(x)=\lambda e^{-\lambda x},\quad x\geq0")
    st.write(
        "Describe como se distribuye la densidad de una variable "
        "aleatoria continua. Lambda es la tasa y x es el tiempo "
        "o valor de la variable."
    )

    st.subheader("2. Funcion acumulada")
    st.latex(r"F(x)=1-e^{-\lambda x}")
    st.write(
        "Representa la probabilidad acumulada desde cero hasta x."
    )

    st.subheader("3. Probabilidad de cola")
    st.latex(r"P(X>x)=e^{-\lambda x}")
    st.write(
        "Calcula la probabilidad de que el tiempo de espera "
        "supere un valor determinado."
    )

    st.subheader("4. Probabilidad entre dos valores")
    st.latex(
        r"P(a<X<b)=F(b)-F(a)"
        r"=e^{-\lambda a}-e^{-\lambda b}"
    )

    st.subheader("5. Media, varianza y desviacion")
    st.latex(r"E(X)=\frac{1}{\lambda}")
    st.latex(r"\operatorname{Var}(X)=\frac{1}{\lambda^2}")
    st.latex(r"\sigma=\frac{1}{\lambda}")

    st.subheader("6. Propiedad sin memoria")
    st.latex(
        r"P(X>s+t\mid X>s)=P(X>t)"
    )
    st.write(
        "El tiempo adicional de espera tiene la misma distribucion "
        "que al comienzo, bajo el modelo exponencial."
    )

    st.subheader("7. Condiciones de uso")
    st.markdown("""
    - La variable X no puede tomar valores negativos.
    - Lambda debe ser estrictamente positiva.
    - La tasa de ocurrencia se supone constante.
    - El modelo suele emplearse para tiempos entre eventos.
    - La integral de la densidad sobre todo su dominio es 1.
    """)

    st.subheader("8. Ejemplo resuelto")
    st.write(
        "Si el tiempo promedio de espera es de 5 minutos, "
        "lambda = 1/5 = 0.2 por minuto."
    )
    st.latex(
        r"P(X\leq3)=1-e^{-0.2(3)}"
        r"\approx0.4512=45.12\%"
    )

st.divider()

st.caption(
    "Calculadora educativa de distribucion exponencial. "
    "Verifica que las unidades de lambda y x sean compatibles."
)
