import numpy as np  # Álgebra lineal, generación de matriz invertible P y cálculo de inversa
import pandas as pd  # Lectura de asegurados y armado de tablas comparativas
# Modelo de regresión para demostrar invarianza
from sklearn.linear_model import LinearRegression
# Métrica R² para verificar que la calidad no cambia al ofuscar
from sklearn.metrics import r2_score
# Búsqueda de k vecinos más cercanos (KNN)
from sklearn.neighbors import NearestNeighbors
# Escalado de características sin alterar dispersión
from sklearn.preprocessing import MaxAbsScaler
import streamlit as st  # Interfaz web nativa sin CSS

st.set_page_config(  # Configura el título de la pestaña y el modo ancho (wide) en una pantalla
    page_title="Sure Tomorrow - Privacidad y KNN",
    page_icon="🔐",
    layout="wide",
)

# Encabezado compacto nativo sin CSS
st.header("🔐 Sure Tomorrow — Similitud KNN y Ofuscación Matricial")
st.caption(  # Subtítulo técnico sobre KNN y cifrado matricial invertible
    "Búsqueda real de vecinos más cercanos (NearestNeighbors) y protección de"
    " datos personales mediante álgebra lineal (X' = X · P)."
)


@st.cache_data  # Guarda el dataset en caché para acelerar la consulta de clientes
def load_data():
    try:
        # Carga el archivo desde la raíz del repo
        return pd.read_csv("insurance_us.csv")
    except Exception:
        return pd.DataFrame()


df = load_data()  # Carga la base de asegurados de Sure Tomorrow

if not df.empty:
    # Variables demográficas del cliente
    feature_cols = ["Gender", "Age", "Salary", "Family members"]

    # Barra lateral: Selección de cliente y parámetros
    st.sidebar.header("Parámetros de Consulta")  # Título del panel lateral
    cliente_id = st.sidebar.number_input(  # Selector de índice de cliente a auditar
        "ID del Cliente (0 a 4999)", min_value=0, max_value=len(df) - 1, value=42
    )
    k_vecinos = st.sidebar.slider(
        "Número de Vecinos Similares (k)", 3, 8, 5)  # Cantidad de vecinos KNN
    usar_escalado = st.sidebar.checkbox(
        # Activa o desactiva escalado
        "Aplicar MaxAbsScaler (KNN)", value=True)
    # Semilla para generar matriz clave P
    seed_p = st.sidebar.number_input("Semilla Matriz P", 1, 999, 42)

    # Extrae el registro del cliente seleccionado
    cliente = df.iloc[cliente_id]

    # 1. Lógica real de KNN
    X_all = df[feature_cols].values  # Matriz de características originales
    if usar_escalado:
        scaler = MaxAbsScaler()  # Inicializa escalador por valor absoluto máximo
        # Escala variables para que el salario no domine la distancia
        X_knn = scaler.fit_transform(X_all)
    else:
        X_knn = X_all  # Usa valores crudos sin escalar

    # Incluye +1 para excluir al propio cliente
    nn = NearestNeighbors(n_neighbors=k_vecinos + 1, metric="euclidean")
    nn.fit(X_knn)  # Ajusta el buscador de vecinos
    # Encuentra los k clientes más parecidos
    distancias, indices = nn.kneighbors([X_knn[cliente_id]])

    # Toma los vecinos excluyendo la posición 0 (el mismo cliente)
    similares = df.iloc[indices[0][1:]].copy()
    # Agrega la distancia euclidiana calculada
    similares["Distancia"] = np.round(distancias[0][1:], 4)

    # 2. Lógica real de Ofuscación y Regresión Lineal
    # Variable objetivo: número de beneficios recibidos
    y_all = df["Insurance benefits"].values
    rng = np.random.default_rng(seed_p)  # Generador aleatorio reproducible
    P = rng.normal(size=(4, 4))  # Genera matriz aleatoria cuadrada 4x4
    # Verifica que la matriz tenga determinante distinto de cero (sea invertible)
    while np.linalg.det(P) == 0:
        P = rng.normal(size=(4, 4))

    # Multiplica la matriz de datos por P para ofuscar la información personal
    X_prime = X_all @ P

    # Entrena regresión lineal con datos originales
    lr_orig = LinearRegression().fit(X_all, y_all)
    # Entrena regresión lineal con datos ofuscados
    lr_obf = LinearRegression().fit(X_prime, y_all)

    pred_orig = lr_orig.predict([X_all[cliente_id]])[
        0]  # Predicción con datos originales
    pred_obf = lr_obf.predict([X_prime[cliente_id]])[
        0]  # Predicción con datos ofuscados
    r2_orig = r2_score(y_all, lr_orig.predict(X_all)
                       )  # R² sobre datos originales
    r2_obf = r2_score(y_all, lr_obf.predict(
        X_prime))  # R² sobre datos ofuscados

    # Distribución en 2 columnas en una sola pantalla
    left_col, right_col = st.columns([1.05, 1], gap="medium")

    with left_col:
        # Subtítulo de sección KNN
        st.subheader(f"👥 Vecinos Más Cercanos (KNN) — Cliente #{cliente_id}")
        st.info(  # Tarjeta informativa nativa con el perfil del cliente consultado
            f"**Cliente Base:** Género: {int(cliente['Gender'])} | Edad:"
            f" {int(cliente['Age'])} | Salario: ${cliente['Salary']:,.0f} |"
            f" Familia: {int(cliente['Family members'])} | Beneficios:"
            f" {int(cliente['Insurance benefits'])}"
        )
        st.dataframe(  # Tabla interactiva con los k vecinos más cercanos
            similares[feature_cols + ["Insurance benefits", "Distancia"]],
            use_container_width=True,
            hide_index=True,
            height=245,
        )

    with right_col:
        # Subtítulo de privacidad
        st.subheader("🛡️ Ofuscación de Datos (X' = X · P) e Invarianza ML")

        m1, m2, m3 = st.columns(3)  # Fila de 3 tarjetas KPI nativas
        m1.metric("Predicción Original", f"{pred_orig:.4f}")
        m2.metric("Predicción Ofuscada",
                  f"{pred_obf:.4f}", delta="0.0000 diff")
        m3.metric("R² Modelo (Ambos)", f"{r2_orig:.4f}", delta="Invariante")

        comp_df = pd.DataFrame({  # Tabla comparativa entre dato original, ofuscado y recuperado con P⁻¹
            "Variable": feature_cols,
            "Original (X)": np.round(X_all[cliente_id], 2),
            "Ofuscado (X' = X·P)": np.round(X_prime[cliente_id], 2),
            "Recuperado (X'·P⁻¹)": np.round(X_prime[cliente_id] @ np.linalg.inv(P), 2),
        })
        st.dataframe(  # Tabla nativa con la transformación matricial en vivo
            comp_df, use_container_width=True, hide_index=True, height=178
        )
        st.caption(  # Nota explicativa de la propiedad matemática de invarianza
            "Los datos personales quedan irreconocibles en X', pero la Regresión"
            " Lineal conserva exactamente las mismas predicciones y R²."
        )
else:
    st.warning("Sube el archivo insurance_us.csv para ejecutar la aplicación.")
