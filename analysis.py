import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Leer dataset
df = pd.read_csv("data/maintenance_data.csv")

# Limpiar nombres de columnas (elimina espacios al inicio/final)
df.columns = df.columns.str.strip()

# Revisar columnas disponibles
print("Columnas disponibles en el DataFrame:")
print(df.columns)

# Vista previa y estadísticos
print(df.head())
print(df.describe())

# Nombre de la columna target
target_col = 'failure_next_week'

# Verificar si existe la columna target
if target_col in df.columns:
    # Conteo de valores
    print(df[target_col].value_counts())

    # Distribución de fallas
    plt.figure(figsize=(6,4))
    sns.countplot(x=target_col, data=df, palette="Set2")
    plt.title("Failure Distribution")
    plt.savefig("output_failure_distribution.png")
    plt.show()

    # Scatter de variables clave
    if 'hours_operated' in df.columns and 'temperature' in df.columns:
        plt.figure(figsize=(6,4))
        sns.scatterplot(x="hours_operated", y="temperature", hue=target_col, data=df)
        plt.title("Hours Operated vs Temperature")
        plt.savefig("output_scatter.png")
        plt.show()
    else:
        print("No se encontraron las columnas 'hours_operated' y/o 'temperature' para el scatter plot.")
else:
    print(f"¡Error! La columna '{target_col}' no existe en el DataFrame.")

# Correlaciones solo de columnas numéricas
numeric_df = df.select_dtypes(include='number')

if not numeric_df.empty:
    plt.figure(figsize=(8,6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig("output_correlation.png")
    plt.show()
else:
    print("No hay columnas numéricas para calcular correlación.")

