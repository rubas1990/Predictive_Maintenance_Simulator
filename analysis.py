import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Leer dataset
df = pd.read_csv("data/maintenance_data.csv")

print(df.head())
print(df.describe())
print(df['failure_next_week'].value_counts())

# Distribución de fallas
sns.countplot(x="failure_next_week", data=df, palette="Set2")
plt.title("Failure Distribution")
plt.savefig("output_failure_distribution.png")
plt.show()

# Correlaciones
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("output_correlation.png")
plt.show()

# Scatter de variables clave
sns.scatterplot(x="hours_operated", y="temperature", hue="failure_next_week", data=df)
plt.title("Hours Operated vs Temperature")
plt.savefig("output_scatter.png")
plt.show()
