# Proyecto-IA-Clasificacion-Emociones

**Materia:** Inteligencia Artificial (SIS-252)
**Autor:** Victor Esteban Medrano Zeballos
**Docente:** MSC. ING. OSCAR CONTRERAS CARRASCO

## Descripción del Proyecto
Este repositorio contiene el avance del proyecto enfocado en la clasificación de estados emocionales en adolescentes mediante visión artificial multimodal (análisis facial y de postura esquelética).

## Estructura del Repositorio
*   `src/`: Contiene el código fuente del proyecto.
    *   `baseline_model.py`: Implementación del algoritmo base (Red Neuronal Feed-Forward).
*   `requirements.txt`: Librerías necesarias para ejecutar el proyecto.
*   `.gitignore`: Archivos excluidos del control de versiones.

## Avance Actual (Baseline)
Se ha implementado un modelo base utilizando `scikit-learn` (Perceptrón Multicapa) para clasificar emociones basándose en características espaciales (coordenadas simuladas). El modelo utiliza `ReLU` y el optimizador `Adam`, demostrando la viabilidad del enfoque antes de pasar a la arquitectura CNN+RNN.

Para ejecutar el algoritmo:
1. Instalar dependencias: `pip install -r requirements.txt`
2. Ejecutar script: `python src/baseline_model.py`

*(El script generará una gráfica de la curva de aprendizaje demostrando la reducción del error).*