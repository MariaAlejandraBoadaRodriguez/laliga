import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===========================
# 1. Cargar dataset completo (fixtures + stats + limpio)
# ===========================
df = pd.read_csv("laliga_2022_clean.csv")

print("========== SHAPE ==========")
print(df.shape)

print("\n========== COLUMNAS ==========")
print(df.columns)

print("\n========== VALORES NULOS ==========")
print(df.isna().sum())

print("\n========== ESTADÍSTICAS DESCRIPTIVAS ==========")
print(df.describe().T)


# ===========================
# 2. Crear variables objetivo (targets)
# ===========================
print("\n========== CREANDO TARGETS ==========")

if {"home_goals", "away_goals"}.issubset(df.columns):
    # Diferencia de goles (local - visitante)
    df["goal_diff"] = df["home_goals"] - df["away_goals"]

    # Etiqueta categórica del resultado
    def label_result(row):
        if row["goal_diff"] > 0:
            return "home_win"
        elif row["goal_diff"] < 0:
            return "away_win"
        else:
            return "draw"

    df["result_label"] = df.apply(label_result, axis=1)

    print("\nEjemplo de targets creados:")
    print(df[["home_team", "away_team", "home_goals", "away_goals", "goal_diff", "result_label"]].head())

    print("\nDistribución de resultados:")
    print(df["result_label"].value_counts())
    print("\nDistribución de resultados (%)")
    print(df["result_label"].value_counts(normalize=True) * 100)
else:
    print("⚠ No se encontraron columnas 'home_goals' y 'away_goals'; no se crean variables objetivo.")


# ===========================
# 3. Heatmap de correlación
# ===========================
print("\n========== CORRELACIÓN ==========")

# Filtrar solo columnas numéricas
df_num = df.select_dtypes(include=['int64', 'float64'])

corr = df_num.corr()

plt.figure(figsize=(20, 12))
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Heatmap de Correlación – LaLiga 2022 (stats + goles)")
plt.tight_layout()
plt.show()

# ===========================
# 4. Top correlaciones absolutas
# ===========================
print("\n========== TOP CORRELACIONES ==========")

# Matriz en formato de pares
corr_pairs = corr.unstack().sort_values(ascending=False)

# Filtrar para quitar autocorrelaciones
corr_pairs_filtered = corr_pairs[corr_pairs < 0.9999]

print(corr_pairs_filtered.head(20))


# ===========================
# 5. Histogramas para analizar distribuciones
# ===========================
print("\n========== HISTOGRAMAS ==========")

for col in df_num.columns:
    plt.figure(figsize=(6, 4))
    plt.hist(df[col], bins=20, color='steelblue', edgecolor='black')
    plt.title(f"Distribución de {col}")
    plt.xlabel(col)
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.show()


# ===========================
# 6. Insights automáticos
# ===========================
print("\n========== INSIGHTS ==========")

# Promedios de tiros al arco
if "home_Shots on Goal" in df.columns and "away_Shots on Goal" in df.columns:
    avg_home_shots = df["home_Shots on Goal"].mean()
    avg_away_shots = df["away_Shots on Goal"].mean()

    print(f"- Promedio de tiros al arco (local): {avg_home_shots:.2f}")
    print(f"- Promedio de tiros al arco (visitante): {avg_away_shots:.2f}")

# Posesión promedio local
if "home_Ball Possession" in df.columns:
    print(f"- Posesión promedio del equipo local: {df['home_Ball Possession'].mean():.2f}%")

# Goles promedio
if {"home_goals", "away_goals"}.issubset(df.columns):
    print(f"- Goles promedio del equipo local: {df['home_goals'].mean():.2f}")
    print(f"- Goles promedio del equipo visitante: {df['away_goals'].mean():.2f}")
    print(f"- Diferencia de goles media (local - visitante): {df['goal_diff'].mean():.2f}")

print("\nEDA COMPLETADO ✔")
