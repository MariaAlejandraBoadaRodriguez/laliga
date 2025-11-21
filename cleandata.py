import pandas as pd

# 1. Cargar el dataset combinado
df = pd.read_csv("laliga_2022_full.csv")

print("Antes de limpiar:", df.shape)

# 2. Quitar columnas duplicadas conservando la primera aparición
df = df.loc[:, ~df.columns.duplicated()]
print("Después de quitar columnas duplicadas:", df.shape)

# 3. Eliminar filas completamente vacías (por seguridad)
df = df.dropna(how="all")
print("Después de eliminar filas totalmente vacías:", df.shape)

# 4. Eliminar filas duplicadas por fixture_id (si la columna existe)
if "fixture_id" in df.columns:
    duplicados = df.duplicated(subset="fixture_id").sum()
    print(f"Filas duplicadas por fixture_id antes de limpiar: {duplicados}")
    df = df.drop_duplicates(subset="fixture_id")
    print("Después de eliminar duplicados por fixture_id:", df.shape)

# 5. Convertir columnas numéricas (todas menos las claramente categóricas)
#    Aquí excluimos columnas de texto como equipos y liga
cols_excluir = ["fixture_id", "date", "league", "round", "home_team", "away_team"]

cols_numericas = [c for c in df.columns if c not in cols_excluir]

df[cols_numericas] = df[cols_numericas].apply(
    pd.to_numeric, errors="coerce"
)

# 6. Rellenar valores numéricos faltantes con 0
df[cols_numericas] = df[cols_numericas].fillna(0)

# 7. Convertir la columna de fecha a tipo datetime (si existe)
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

print("\nColumnas finales:")
print(df.columns)

# 8. Guardar CSV limpio
df.to_csv("laliga_2022_clean.csv", index=False)
print("\n✅ Archivo limpio guardado como 'laliga_2022_clean.csv'")
print("Shape final:", df.shape)
