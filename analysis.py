import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.title("Análisis de Mantenimiento Predictivo")

# ========================================
# 📌 Leer dataset con fix de BOM y fallback
# ========================================
df = pd.read_csv(
    "data/maintenance_data.csv",
    sep=",", 
    encoding="utf-8-sig",  # elimina BOM oculto
    engine="python"
)

# Si por alguna razón pandas lo cargó como 1 sola columna → forzar split
if df.shape[1] == 1:
    df = df.iloc[:,0].str.split(",", expand=True)
    df.columns = ["machine_id","temperature","vibration","pressure","hours_operated","failure_next_week"]

# Normalizar nombres de columnas
df.columns = df.columns.str.strip().str.lower()

st.subheader("Columnas detectadas en el CSV")
st.write(df.columns.tolist())

# ========================================
# 📌 Conversión de columnas numéricas
# ========================================
expected_num_cols = ['temperature', 'vibration', 'pressure', 'hours_operated', 'failure_next_week']
num_cols = [col for col in expected_num_cols if col in df.columns]

if len(num_cols) < len(expected_num_cols):
    missing = set(expected_num_cols) - set(num_cols)
    st.warning(f"Faltan columnas esperadas en el CSV: {missing}")

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ========================================
# 📌 Vista previa y estadísticos
# ========================================
st.subheader("Vista previa del DataFrame")
st.write(df.head())

st.subheader("Estadísticos descriptivos")
st.write(df.describe())

# ========================================
# 📌 Distribución de fallas
# ========================================
if 'failure_next_week' in num_cols:
    st.subheader("Distribución de Fallas Semanales")
    plt.figure(figsize=(6,4))
    sns.countplot(x='failure_next_week', data=df, palette="Set2")
    plt.title("Failure Distribution")
    st.pyplot(plt)

# ========================================
# 📌 Scatter plot
# ========================================
if 'hours_operated' in num_cols and 'temperature' in num_cols:
    st.subheader("Scatter: Hours Operated vs Temperature")
    plt.figure(figsize=(6,4))
    sns.scatterplot(
        x='hours_operated',
        y='temperature',
        hue='failure_next_week',
        data=df,
        palette="Set2"
    )
    plt.title("Hours Operated vs Temperature")
    st.pyplot(plt)

# ========================================
# 📌 Heatmap de correlación
# ========================================
st.subheader("Mapa de Correlación")
numeric_df = df.select_dtypes(include='number')

if numeric_df.shape[1] > 1:
    plt.figure(figsize=(8,6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    st.pyplot(plt)
else:
    st.warning("No hay suficientes columnas numéricas para calcular correlación.")
