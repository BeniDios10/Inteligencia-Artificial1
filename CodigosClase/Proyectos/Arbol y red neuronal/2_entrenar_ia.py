import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib # Para guardar el árbol

# 1. Cargar datos
df = pd.read_csv("datos_juego.csv")
X = df[["distancia", "velocidad"]].values
y = df["salto"].values

# Balancear datos (opcional pero recomendado: suele haber muchos más "no saltos" que "saltos")
# Para este ejemplo simple, usaremos los datos tal cual.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- A. ÁRBOL DE DECISIÓN ---
print("Entrenando Árbol...")
arbol = DecisionTreeClassifier(max_depth=4) # Profundidad limitada para evitar memorización
arbol.fit(X_train, y_train)
joblib.dump(arbol, "modelo_arbol.pkl") # Guardar modelo
print(f"Precisión Árbol: {arbol.score(X_test, y_test):.2f}")

# --- B. RED NEURONAL MULTICAPA (Tu código adaptado) ---
print("Entrenando Red Neuronal...")
model = Sequential([
    Dense(8, input_dim=2, activation='relu'), # Capa oculta
    Dense(4, activation='relu'),              # Capa oculta extra
    Dense(1, activation='sigmoid')            # Salida (Probabilidad de salto)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)
model.save("modelo_red.h5") # Guardar modelo Keras
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Precisión Red Neuronal: {acc:.2f}")

print("\n¡Modelos guardados exitosamente!")