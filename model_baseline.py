import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Cargar dataset limpio (fixtures + stats)
df = pd.read_csv("doc/laliga_2022_clean.csv")

print("Shape original:", df.shape)

# 2. Crear targets: goal_diff y result_label
if {"home_goals", "away_goals"}.issubset(df.columns):
    df["goal_diff"] = df["home_goals"] - df["away_goals"]

    def label_result(row):
        if row["goal_diff"] > 0:
            return "home_win"
        elif row["goal_diff"] < 0:
            return "away_win"
        else:
            return "draw"

    df["result_label"] = df.apply(label_result, axis=1)
else:
    raise ValueError("El dataset no tiene columnas home_goals y away_goals")

print("\nDistribución de la variable objetivo:")
print(df["result_label"].value_counts())

# 3. Seleccionar features numéricos (excluyendo IDs y goles para no hacer trampa)
cols_to_drop = [
    "fixture_id",
    "date",
    "league",
    "round",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result_label",  # target
]

X = df.drop(columns=cols_to_drop, errors="ignore")
y = df["result_label"]

print("\nFeatures usados para el modelo:")
print(X.columns)
print("\nShape X:", X.shape)

# 4. Codificar la variable objetivo
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print("\nMapping de clases:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

print("\nTrain size:", X_train.shape[0], " | Test size:", X_test.shape[0])

# 6. Modelo Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42
)

rf.fit(X_train, y_train)

# 7. Evaluación
y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy Random Forest: {acc:.3f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("\nMatriz de confusión:")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix – Random Forest (baseline)")
plt.tight_layout()
plt.savefig("img/confusion_matrix_baseline.png")
plt.close()

# --- Gráfico de distribución de clases (target) ---
plt.figure(figsize=(5, 4))
df["result_label"].value_counts().plot(kind="bar")
plt.title("Distribución de resultados (LaLiga sample)")
plt.xlabel("Resultado")
plt.ylabel("Número de partidos")
plt.tight_layout()
plt.savefig("img/class_distribution.png")
plt.close()
