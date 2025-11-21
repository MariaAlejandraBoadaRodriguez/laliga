import pandas as pd

# 1. Cargar los dos archivos
fixtures_df = pd.read_csv("soccer_api_2022.csv")
stats_df = pd.read_csv("stats_2022_sample.csv")

print("Fixtures shape:", fixtures_df.shape)
print("Stats shape:", stats_df.shape)

print("\nColumnas de fixtures:")
print(fixtures_df.columns)

print("\nColumnas de stats:")
print(stats_df.columns)

# 2. Unir por fixture_id (clave común)
merged_df = fixtures_df.merge(stats_df, on="fixture_id", how="inner")

print("\nShape después del merge:", merged_df.shape)
print(merged_df.head())

# 3. Guardar dataset combinado
merged_df.to_csv("laliga_2022_full.csv", index=False)
print("\n✅ Archivo 'laliga_2022_full.csv' generado correctamente.")
