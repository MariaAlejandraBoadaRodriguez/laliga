import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pennylane as qml

# Crear carpeta de imágenes si no existe
os.makedirs("img", exist_ok=True)

# 1. Cargar y preparar datos
df = pd.read_csv("doc/laliga_2022_clean.csv")

# Crear goal_diff y result_label (igual que en el baseline)
df["goal_diff"] = df["home_goals"] - df["away_goals"]

def label_result(row):
    if row["goal_diff"] > 0:
        return "home_win"
    elif row["goal_diff"] < 0:
        return "away_win"
    else:
        return "draw"

df["result_label"] = df.apply(label_result, axis=1)

print("Shape original:", df.shape)
print("\nDistribución de la variable objetivo:")
print(df["result_label"].value_counts())

# 2. Elegimos 4 features para el circuito cuántico
feat_cols = [
    "home_Shots on Goal",
    "away_Shots on Goal",
    "home_Ball Possession",
    "away_Ball Possession",
]

X_raw = df[feat_cols].copy()
y = df["result_label"]

# Normalizar a [0, π] para usar como ángulos de rotación
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X_raw)

# Codificar y
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

print("\nTrain size:", X_train.shape[0], " | Test size:", X_test.shape[0])

# 3. Definir dispositivo cuántico
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

# 4. Definir circuito QNN sencillo
@qml.qnode(dev)
def qnn_circuit(x, weights):
    # Codificación de features en rotaciones RX
    for i in range(n_qubits):
        qml.RX(x[i], wires=i)

    # Capa variacional simple con RY + CNOT en cadena
    for i in range(n_qubits):
        qml.RY(weights[i], wires=i)

    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])

    # Devolvemos expectativas de Z en cada qubit
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


# 5. Generar embedding cuántico (features transformados)
def quantum_embedding(X, weights):
    embedded = []
    for x in X:
        out = qnn_circuit(x, weights)
        embedded.append(out)
    return np.array(embedded)


# Inicializamos pesos del circuito (uno por qubit, muy sencillo)
init_weights = np.random.uniform(0, 2 * np.pi, size=(n_qubits,))

# Obtenemos representación cuántica para train y test
X_train_q = quantum_embedding(X_train, init_weights)
X_test_q = quantum_embedding(X_test, init_weights)

print("Shape embedding cuántico train:", X_train_q.shape)

# 6. Random Forest sobre representación cuántica
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train_q, y_train)
y_pred = rf.predict(X_test_q)

acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy modelo híbrido (QNN embedding + RF): {acc:.3f}")
print("Clases:", label_encoder.classes_)

print("\nClassification report (híbrido QNN + RF):")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 7. Matriz de confusión (y guardar imagen)
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
plt.title("Confusion Matrix – Hybrid QNN + Random Forest")
plt.tight_layout()
plt.savefig("img/confusion_matrix_hybrid_qnn_rf.png")
plt.close()

# 8. Distribución de clases (target) y guardar imagen
plt.figure(figsize=(5, 4))
df["result_label"].value_counts().plot(kind="bar")
plt.title("Class Distribution – LaLiga Sample (Hybrid Model)")
plt.xlabel("Result")
plt.ylabel("Number of matches")
plt.tight_layout()
plt.savefig("img/class_distribution_hybrid.png")
plt.close()

print("\n✔ Imágenes guardadas en la carpeta 'img':")
print("   - img/confusion_matrix_hybrid_qnn_rf.png")
print("   - img/class_distribution_hybrid.png")
