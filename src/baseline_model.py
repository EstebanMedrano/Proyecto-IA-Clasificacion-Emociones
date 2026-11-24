import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. SIMULACIÓN DE DATOS (Simulando la extracción de MediaPipe)
print("Iniciando la carga y simulacion de datos extraidos...")
np.random.seed(42)
num_samples = 1000
num_features = 140 # Coordenadas X,Y de puntos faciales y corporales
X = np.random.rand(num_samples, num_features) 

# Simulamos 4 emociones (0: Feliz, 1: Triste, 2: Ansioso, 3: Neutral)
y = np.random.randint(0, 4, num_samples)

# 2. PREPROCESAMIENTO
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. DEFINICIÓN DEL MODELO (Red Feed-Forward)
print("Entrenando el modelo Baseline (Perceptrón Multicapa)...")
model = MLPClassifier(hidden_layer_sizes=(64, 32), 
                      activation='relu', 
                      solver='adam', 
                      max_iter=100, 
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
plt.plot(model.loss_curve_, color='blue', label='Error de Entrenamiento')
plt.title('Curva de Aprendizaje del Modelo Baseline')
plt.xlabel('Iteraciones (Epocas)')
plt.ylabel('Pérdida (Loss)')
plt.legend()
plt.grid(True)
plt.savefig('curva_aprendizaje.png') # Guarda la imagen en la carpeta
print("Grafica guardada como 'curva_aprendizaje.png'")