import streamlit as st
import pandas as pd
import numpy as np

st.title("🔐 Sure Tomorrow - Privacidad y KNN")

@st.cache_data
def load_data():
    try:
        return pd.read_csv('insurance_us.csv')
    except:
        return pd.DataFrame()

df = load_data()

tab1, tab2 = st.tabs(["Búsqueda de Similares (KNN)", "Ofuscación de Datos"])

with tab1:
    st.header("Encuentra clientes similares")
    if not df.empty:
        # Tomamos un cliente aleatorio como ejemplo
        cliente_id = st.number_input("ID del cliente (0 al 4999)", 0, len(df)-1, 42)
        cliente = df.iloc[cliente_id]
        
        st.write(f"**Cliente seleccionado:** Edad {cliente['Age']}, Ingreso {cliente['Salary']}, Familia {cliente['Family members']}")
        
        if st.button("Buscar 3 similares"):
            # Simulamos KNN buscando por diferencia de edad y salario
            df['dist'] = abs(df['Age'] - cliente['Age']) + abs(df['Salary'] - cliente['Salary'])/1000
            similares = df.sort_values('dist').iloc[1:4]
            st.dataframe(similares[['Gender', 'Age', 'Salary', 'Family members']])
    else:
        st.warning("Sube el archivo insurance_us.csv para usar esta pestaña.")

with tab2:
    st.header("Matriz de Ofuscación (P)")
    st.write("Multiplicamos los datos originales por una matriz P invertible.")
    
    if st.button("Ofuscar mis datos"):
        X = np.array([[1, 35, 45000, 2]]) # Datos de ejemplo
        P = np.random.rand(4, 4)
        
        X_prime = X @ P
        
        st.write("**Datos Originales (X):**")
        st.code(str(X))
        st.write("**Datos Ofuscados (X'):**")
        st.code(str(np.round(X_prime, 2)))
        st.success("¡La información personal está protegida! El modelo de ML seguirá funcionando igual.")
