import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.title("Análisis de Mantenimiento Predictivo")

# Leer dataset
df = pd.read_csv("data/maintenance_data.csv")  # Cambia por la ruta de tu CSV

# Limpiar nombres de columnas
df.columns = df.columns.str.strip()

# Convertir columnas numéricas
num_cols = ['temperature', 'vibration', 'pressure', 'hours_operated', 'failure_next_week']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Mostrar información básica
st.subheader("Vista previa del DataFrame")
st.write(df.head())

st.subheader("Estadísticos descriptivos")
st.write(df.describe())

# Distribución de fallas
st.subheader("Distribución de Fallas Semanales")
plt.figure(figsize=(6,4))
sns.countplot(x='failure_next_week', data=df, palette="Set2")
plt.title("Failure Distribution")
plt.savefig("output_failure_distribution.png")
st.pyplot(plt)

# Scatter plot: horas operadas vs temperatura
st.subheader("Scatter: Hours Operated vs Temperature")
plt.figure(figsize=(6,4))
sns.scatterplot(x='hours_operated', y='temperature', hue='failure_next_week', data=df, palette="Set2")
plt.title("Hours Operated vs Temperature")
plt.savefig("output_scatter.png")
st.pyplot(plt)

# Mapa de correlación de columnas numéricas
st.subheader("Mapa de Correlación")
numeric_df = df.select_dtypes(include='number')
plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("output_correlation.png")
st.pyplot(plt)

