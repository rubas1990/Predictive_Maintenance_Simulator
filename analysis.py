import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Leer dataset
df = pd.read_csv("data/maintenance_data.csv")

# Limpiar nombres de columnas (elimina espacios al inicio/final)
df.columns = df.columns.str.strip()

# Revisar columnas
print("Columnas disponibles en el DataFrame:")
print(df.columns)

# Vista previa y estadísticos
print(df.head())
print(df.describe())

# Verificar si existe la columna 'failure_next_week'
target_col = 'failure_next_week'
if target_col in df.columns:
    print(df[target_col].value_counts())

    # Distribución de fallas
    sns.countplot(x=target_col, data=df, palette="Set2")
    plt.title("Failure Distribution")
    plt.savefig("output_failure_distribution.png")
    plt.show()

    # Scatter de variables clave
    sns.scatterplot(x="hours_operated", y="temperature", hue=target_col, data=df)
    plt.title("Hours Operated vs Temperature")
    plt.savefig("output_scatter.png")
    plt.show()
else:
    print(f"¡Error! La columna '{target_col}' no existe en el DataFrame.")

# Correlaciones (se puede hacer aunque falte la columna target)
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("output_correlation.png")
plt.show()
