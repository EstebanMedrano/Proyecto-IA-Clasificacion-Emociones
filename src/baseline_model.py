import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

print("Iniciando la carga y simulacion de datos extraidos...")

# 1. SIMULACIÓN DE DATOS CON PATRONES
X, y = make_classification(n_samples=1000, n_features=140, n_informative=30, 
                           n_redundant=0, n_classes=4, random_state=42)

# 2. PREPROCESAMIENTO
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. DEFINICIÓN DEL MODELO (Red Feed-Forward)
print("Entrenando el modelo Baseline (Perceptrón Multicapa)...")
# Aumentamos las iteraciones max_iter a 300 para que tenga tiempo de aprender
model = MLPClassifier(hidden_layer_sizes=(64, 32), 
                      activation='relu', 
                      solver='adam', 
                      max_iter=300, 
                      random_state=42)

# 4. ENTRENAMIENTO
model.fit(X_train, y_train)

# 5. EVALUACIÓN
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nPrecisión (Accuracy) del Baseline: {accuracy * 100:.2f}%")
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, zero_division=0))

# 6. GENERAR GRÁFICA
print("\nGenerando grafica de la curva de aprendizaje...")
plt.figure(figsize=(8, 5))
plt.plot(model.loss_curve_, color='blue', label='Error de Entrenamiento')
plt.title('Curva de Aprendizaje del Modelo Baseline')
plt.xlabel('Iteraciones (Epocas)')
plt.ylabel('Pérdida (Loss)')
plt.legend()
plt.grid(True)
plt.savefig('curva_aprendizaje.png')
print("Grafica guardada como 'curva_aprendizaje.png'")