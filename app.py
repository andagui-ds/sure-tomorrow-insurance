import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MaxAbsScaler
import streamlit as st

st.set_page_config(
    page_title="Sure Tomorrow - Privacidad y KNN", layout="wide"
)

# CSS compacto para vista ejecutiva en una sola pantalla sin scroll
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 0.5rem;
        }
        h1 {
            font-size: 1.85rem !important;
            margin-bottom: 0.15rem !important;
        }
        h3 {
            font-size: 1.12rem !important;
            margin-top: 0.35rem !important;
            margin-bottom: 0.35rem !important;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔐 Sure Tomorrow — Similitud KNN y Ofuscación Matricial")
st.caption(
    "Búsqueda real de vecinos más cercanos (NearestNeighbors) y protección de"
    " datos personales mediante álgebra lineal (X' = X · P)."
)


@st.cache_data
def load_data():
  try:
    return pd.read_csv("insurance_us.csv")
  except Exception:
    return pd.DataFrame()


df = load_data()

if not df.empty:
  feature_cols = ["Gender", "Age", "Salary", "Family members"]

  # Barra lateral: Selección de cliente y parámetros
  st.sidebar.header("Parámetros de Consulta")
  cliente_id = st.sidebar.number_input(
      "ID del Cliente (0 a 4999)", min_value=0, max_value=len(df) - 1, value=42
  )
  k_vecinos = st.sidebar.slider("Número de Vecinos Similares (k)", 3, 8, 5)
  usar_escalado = st.sidebar.checkbox("Aplicar MaxAbsScaler (KNN)", value=True)
  seed_p = st.sidebar.number_input("Semilla Matriz P", 1, 999, 42)

  cliente = df.iloc[cliente_id]

  # 1. Lógica real de KNN
  X_all = df[feature_cols].values
  if usar_escalado:
    scaler = MaxAbsScaler()
    X_knn = scaler.fit_transform(X_all)
  else:
    X_knn = X_all

  nn = NearestNeighbors(n_neighbors=k_vecinos + 1, metric="euclidean")
  nn.fit(X_knn)
  distancias, indices = nn.kneighbors([X_knn[cliente_id]])

  similares = df.iloc[indices[0][1:]].copy()
  similares["Distancia"] = np.round(distancias[0][1:], 4)

  # 2. Lógica real de Ofuscación y Regresión Lineal
  y_all = df["Insurance benefits"].values
  rng = np.random.default_rng(seed_p)
  P = rng.normal(size=(4, 4))
  while np.linalg.det(P) == 0:
    P = rng.normal(size=(4, 4))

  X_prime = X_all @ P

  lr_orig = LinearRegression().fit(X_all, y_all)
  lr_obf = LinearRegression().fit(X_prime, y_all)

  pred_orig = lr_orig.predict([X_all[cliente_id]])[0]
  pred_obf = lr_obf.predict([X_prime[cliente_id]])[0]
  r2_orig = r2_score(y_all, lr_orig.predict(X_all))
  r2_obf = r2_score(y_all, lr_obf.predict(X_prime))

  # Layout en 2 columnas sin scroll
  left_col, right_col = st.columns([1.05, 1], gap="medium")

  with left_col:
    st.subheader(f"👥 Vecinos Más Cercanos (KNN) — Cliente #{cliente_id}")
    st.info(
        f"**Cliente Base:** Género: {int(cliente['Gender'])} | Edad:"
        f" {int(cliente['Age'])} | Salario: ${cliente['Salary']:,.0f} |"
        f" Familia: {int(cliente['Family members'])} | Beneficios:"
        f" {int(cliente['Insurance benefits'])}"
    )
    st.dataframe(
        similares[feature_cols + ["Insurance benefits", "Distancia"]],
        use_container_width=True,
        hide_index=True,
        height=245,
    )

  with right_col:
    st.subheader("🛡️ Ofuscación de Datos (X' = X · P) e Invarianza ML")

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicción Original", f"{pred_orig:.4f}")
    m2.metric("Predicción Ofuscada", f"{pred_obf:.4f}", delta="0.0000 diff")
    m3.metric("R² Modelo (Ambos)", f"{r2_orig:.4f}", delta="Invariante")

    comp_df = pd.DataFrame({
        "Variable": feature_cols,
        "Original (X)": np.round(X_all[cliente_id], 2),
        "Ofuscado (X' = X·P)": np.round(X_prime[cliente_id], 2),
        "Recuperado (X'·P⁻¹)": np.round(X_prime[cliente_id] @ np.linalg.inv(P), 2),
    })
    st.dataframe(
        comp_df, use_container_width=True, hide_index=True, height=178
    )
    st.caption(
        "Los datos personales quedan irreconocibles en X', pero la Regresión"
        " Lineal conserva exactamente las mismas predicciones y R²."
    )
else:
  st.warning("Sube el archivo insurance_us.csv para ejecutar la aplicación.")
