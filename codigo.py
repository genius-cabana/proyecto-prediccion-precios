import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models

# Cargar el dataset
california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = california.target

print("Primeras filas del dataset:")
print(X.head())

print("Variable objetivo (precio):")
print(y[:5])

# Seleccionamos una característica
X_simple = X[['MedInc']].values

# Dividimos los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_simple, y, test_size=0.2, random_state=42)

# Normalización
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelo de regresión lineal
model = models.Sequential([
    layers.Dense(1, input_shape=(1,))
])

model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train_scaled, y_train, epochs=50, verbose=0)

# Predicciones
y_pred = model.predict(X_test_scaled)

# Gráfico de predicciones vs valores reales
plt.scatter(X_test, y_test, label='Valores reales')
plt.scatter(X_test, y_pred, label='Predicciones', color='red', alpha=0.6)
plt.title('Regresión Lineal Simple')
plt.xlabel('Ingreso Medio (MedInc)')
plt.ylabel('Precio de la vivienda')
plt.legend()
plt.show()

# Usamos todas las características
X_multi = X.values

# Dividimos los datos
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y, test_size=0.2, random_state=42)

# Normalización
scaler_multi = StandardScaler()
X_train_scaled_m = scaler_multi.fit_transform(X_train_m)
X_test_scaled_m = scaler_multi.transform(X_test_m)

# Construimos la red neuronal
model_nn = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_scaled_m.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

model_nn.compile(optimizer='adam', loss='mse')
history_nn = model_nn.fit(X_train_scaled_m, y_train_m, epochs=100, batch_size=32, verbose=0)

# Evaluación
loss = model_nn.evaluate(X_test_scaled_m, y_test_m)
print(f"\nError cuadrático medio en prueba (Red Neuronal): {loss:.4f}")

# Predicciones
y_pred_nn = model_nn.predict(X_test_scaled_m)

# Gráfico comparativo
plt.figure(figsize=(10, 5))
plt.plot(y_test_m[:50], label='Valores reales', marker='o')
plt.plot(y_pred_nn[:50], label='Predicciones', marker='x', linestyle='--')
plt.title('Comparación: Valores Reales vs Predicciones (Red Neuronal)')
plt.legend()
plt.grid(True)
plt.show()

if loss < 0.5:
    print("El modelo tiene buen desempeño.")
else:
    print("El modelo necesita ajustes.")

plt.savefig('resultados_regresion.png')