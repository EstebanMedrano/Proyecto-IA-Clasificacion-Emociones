# Script para la extraccion de caracteristicas usando MediaPipe

import cv2
import mediapipe as mp

# Inicializamos MediaPipe para rostro y cuerpo
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

def extraer_coordenadas(video_path):
    """
    Recibe un video y devuelve una lista con las coordenadas (x, y) 
    de la malla facial y la postura esqueletica.
    """
    print(f"Procesando video: {video_path}")
    
    # Aqui iria la logica de OpenCV para leer el video
    # y MediaPipe para extraer los puntos clave.
    # Retornamos datos simulados por ahora.
    
    print("Extraccion completada (simulada).")
    return [0.1, 0.2, 0.3] # Placeholder

if __name__ == "__main__":
    datos = extraer_coordenadas("data/video_ejemplo.mp4")
    print(f"Datos extraidos: {datos}")