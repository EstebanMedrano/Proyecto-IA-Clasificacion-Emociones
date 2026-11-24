import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. SIMULACIÓN DE DATOS
np.random.seed(42)
num_samples = 1000
num_features = 140 
X = np.random.rand(num_samples, num_features) 

# Simulamos 4 emociones: 0: Feliz, 1: Triste, 2: Ansioso, 3: Neutral
y = np.random.randint(0, 4, num_samples)

# 2. PREPROCESAMIENTO
print("Preprocesando datos...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. DEFINICIÓN DEL MODELO BASELINE
# Usamos una red de 2 capas ocultas
print("Entrenando el modelo Baseline (Red Feed-Forward)...")
model = MLPClassifier(hidden_layer_sizes=(64, 32), 
                      activation='relu', # Función de activación vista en clase
                      solver='adam',     # Algoritmo de optimización (Backpropagation)
                      max_iter=100, 
                      random_state=42)

# 4. ENTRENAMIENTO
model.fit(X_train, y_train)

# 5. EVALUACIÓN
print("Evaluando el modelo...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Precisión (Accuracy) del Baseline: {accuracy * 100:.2f}%")
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, zero_division=0))