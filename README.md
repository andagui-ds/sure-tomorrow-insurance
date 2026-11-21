[🇺🇸 Read in English](#-sure-tomorrow--insurance-machine-learning-and-data-privacy) | [🇲🇽 Leer en Español](#-sure-tomorrow--seguros-machine-learning-y-privacidad-de-datos)

---

# 🔐 Sure Tomorrow — Insurance, Machine Learning, and Data Privacy

> Machine Learning system for customer similarity recommendations, insurance benefit prediction, and **Data Obfuscation** to protect Personally Identifiable Information (PII) without degrading predictive model performance.

🚀 **[Launch Interactive Streamlit App](https://sure-tomorrow-insurance.streamlit.app/)** | 🌐 **[View Quarto Web Report](https://andagui-ds.github.io/sure-tomorrow-insurance/)**

---

## 🎯 Business Problem

The insurance company **Sure Tomorrow** aims to deploy Machine Learning solutions across four core tasks:
1. **Similar Customer Search:** Identify customers with comparable profiles to optimize marketing campaigns.
2. **Benefit Classification:** Predict the probability that a new customer will receive an insurance benefit.
3. **Benefit Regression:** Predict the exact number of insurance benefits a customer is likely to receive.
4. **Data Privacy:** Obfuscate personal customer data (Data Protection Compliance) so that leaked records cannot be recovered, while ensuring ML models maintain identical predictive accuracy.

## 📊 Data

* **Source:** `insurance_us.csv`
* **Records:** 4,847 (after removing duplicates)
* **Features:**
  * `gender`: Customer gender (encoded as 0 or 1)
  * `age`: Customer age
  * `income`: Annual income
  * `family_members`: Number of family members
  * `insurance_benefits`: Number of insurance benefits received (Target)

## 🧠 Approaches and Solutions

### 1. Similar Customers (KNN)
Implemented the K-Nearest Neighbors (KNN) algorithm to retrieve the $k$ closest customers. Evaluated both Euclidean and Manhattan distance metrics, proving that **feature scaling is strictly critical**. Without scaling, `income` completely dominated distance calculations, ignoring age and family size. With `MaxAbsScaler`, the model accurately identifies true neighbors.

### 2. Classification (Benefit Prediction)
Built a KNN classifier to predict whether a customer will receive at least one benefit. Given the strong class imbalance (~88% Class 0), performance was benchmarked against a Dummy (random) classifier.
* **Result:** Scaled KNN achieved an **F1-Score of 0.9264**, vastly outperforming the Dummy baseline (maximum F1-Score of 0.21).

### 3. Linear Regression (Benefit Count)
Implemented a Linear Regression model from scratch using the closed-form matrix analytical solution:

$$ w = (X^T X)^{-1} X^T y $$

* **Result:** Achieved an **RMSE of 0.36** and an **$R^2$ of 0.66**.
* **Key Finding:** Empirically demonstrated that feature scaling does not alter Linear Regression predictions, as model weights adjust proportionally.

### 4. Data Obfuscation and Privacy (Project Differentiator)
To protect sensitive PII, the feature matrix $X$ was multiplied by an invertible random matrix $P$: $X' = X \times P$.

**Analytical Proof of Regression Invariance under Obfuscation:**
When training the model on obfuscated data, the new weight vector is:

$$ w_P = [(XP)^T XP]^{-1} (XP)^T y $$

Applying matrix transpose and inverse properties ($(AB)^{-1} = B^{-1}A^{-1}$ and $(AB)^T = B^TA^T$):

$$ w_P = [P^T X^T X P]^{-1} P^T X^T y = P^{-1} (X^T X)^{-1} (P^T)^{-1} P^T X^T y $$

$$ w_P = P^{-1} [(X^T X)^{-1} X^T y] = P^{-1} w $$

During inference, the obfuscated model computes:

$$ \hat{y}_P = X' w_P = (XP)(P^{-1}w) = X(PP^{-1})w = Xw = \hat{y} $$

**Conclusion:** Predicted values remain **mathematically identical**. Both RMSE and $R^2$ are completely unaffected (maximum floating-point precision variance of $9.07 \times 10^{-8}$). Data is unreadable to an attacker without matrix $P$, yet fully usable by the ML pipeline.

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Math, Streamlit
* **Concepts:** KNN, Classification (F1-Score), Linear Regression (Matrix Algebra), Data Obfuscation, Feature Scaling (`MaxAbsScaler`).

## 🔗 Full Analysis & Production App

* 📓 [Original Jupyter Notebook with full step-by-step code](notebooks/analisis-sure-tomorrow.ipynb)
* 🚀 [Interactive KNN & Matrix Obfuscation App (Streamlit)](https://sure-tomorrow-insurance.streamlit.app/)

## ▶️ How to Run Locally

    pip install -r requirements.txt
    streamlit run app.py

---

# 🔐 Sure Tomorrow — Seguros, Machine Learning y Privacidad de Datos

> Sistema de Machine Learning para la recomendación de clientes, predicción de prestaciones y **ofuscación de datos (Data Obfuscation)** para proteger la privacidad (PII) sin afectar el rendimiento de los modelos predictivos.

🚀 **[Abrir App Interactiva en Streamlit](https://sure-tomorrow-insurance.streamlit.app/)** | 🌐 **[Ver Reporte Web en Quarto](https://andagui-ds.github.io/sure-tomorrow-insurance/)**

---

## 🎯 Problema de Negocio

La compañía de seguros **Sure Tomorrow** busca implementar soluciones de Machine Learning para cuatro desafíos clave:
1. **Búsqueda de clientes similares:** Identificar clientes con perfiles parecidos para optimizar las campañas de marketing.
2. **Clasificación de prestaciones:** Predecir la probabilidad de que un nuevo cliente reciba una prestación del seguro.
3. **Regresión de prestaciones:** Predecir el número exacto de prestaciones que un cliente podría recibir.
4. **Privacidad de Datos:** Ofuscar la información personal de los clientes (Ley de Protección de Datos) garantizando que, si los datos se filtran, no puedan ser recuperados, pero asegurando que los modelos de ML sigan funcionando con la misma precisión.

## 📊 Datos

* **Fuente:** `insurance_us.csv`
* **Registros:** 4,847 (tras eliminar duplicados)
* **Variables:**
  * `gender`: Sexo del cliente (0 o 1)
  * `age`: Edad
  * `income`: Ingresos anuales
  * `family_members`: Número de familiares
  * `insurance_benefits`: Número de prestaciones recibidas (Target)

## 🧠 Enfoques y Soluciones

### 1. Clientes Similares (KNN)
Implementación del algoritmo K-Nearest Neighbors (KNN) para encontrar los $k$ clientes más cercanos. Se evaluaron dos métricas de distancia (Euclidiana y Manhattan) y se demostró que **el escalado de los datos es absolutamente crítico**. Sin escalar, la variable `income` (ingresos) dominaba por completo la distancia, ignorando edad o familia. Con `MaxAbsScaler`, el modelo identifica vecinos reales.

### 2. Clasificación (Predicción de Prestaciones)
Se construyó un clasificador KNN para predecir si un cliente recibirá al menos una prestación. Al existir un fuerte desbalanceo de clases (~88% clase 0), se comparó contra un modelo Dummy (aleatorio).
* **Resultado:** El KNN escalado logró un **F1-Score de 0.9264**, destrozando al modelo Dummy (cuyo F1 máximo fue 0.21).

### 3. Regresión Lineal (Conteo de Prestaciones)
Se implementó un modelo de Regresión Lineal desde cero utilizando la solución analítica matricial:

$$ w = (X^T X)^{-1} X^T y $$

* **Resultado:** Se obtuvo un **RMSE de 0.36** y un **$R^2$ de 0.66**.
* **Hallazgo clave:** Se demostró empíricamente que el escalado de los datos no afecta los resultados de la regresión lineal, ya que los pesos del modelo se ajustan automáticamente.

### 4. Ofuscación de Datos y Privacidad (El diferenciador del proyecto)
Para proteger los datos, se multiplicó la matriz de características $X$ por una matriz invertible $P$ de números aleatorios: $X' = X \times P$.

**Prueba Analítica de la Regresión con Ofuscación:**
Si entrenamos el modelo con datos ofuscados, los nuevos pesos serían:

$$ w_P = [(XP)^T XP]^{-1} (XP)^T y $$

Aplicando propiedades matriciales ($(AB)^{-1} = B^{-1}A^{-1}$ y $(AB)^T = B^TA^T$):

$$ w_P = [P^T X^T X P]^{-1} P^T X^T y = P^{-1} (X^T X)^{-1} (P^T)^{-1} P^T X^T y $$

$$ w_P = P^{-1} [(X^T X)^{-1} X^T y] = P^{-1} w $$

Al predecir, el modelo ofuscado calcula:

$$ \hat{y}_P = X' w_P = (XP)(P^{-1}w) = X(PP^{-1})w = Xw = \hat{y} $$

**Conclusión:** Los valores predichos son **exactamente los mismos**. El RMSE y el $R^2$ no se ven afectados en absoluto (diferencia máxima por precisión de punto flotante de $9.07 \times 10^{-8}$). Los datos son inútiles para un hacker (sin la matriz $P$), pero perfectos para el modelo.

## 🛠️ Stack Técnico

* **Lenguajes:** Python
* **Librerías:** Pandas, NumPy, Scikit-learn, Math, Streamlit
* **Conceptos:** KNN, Clasificación (F1-Score), Regresión Lineal (Álgebra Matricial), Data Obfuscation, Escalado de datos (`MaxAbsScaler`).

## 🔗 Ver el Análisis Completo y App en Vivo

* 📓 [Notebook original con todo el código paso a paso](notebooks/analisis-sure-tomorrow.ipynb)
* 🚀 [Simulador Interactivo KNN y Ofuscación Matricial (Streamlit)](https://sure-tomorrow-insurance.streamlit.app/)

## ▶️ Cómo ejecutar localmente

    pip install -r requirements.txt
    streamlit run app.py
