import cv2
import mediapipe as mp
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib 

# ⚠️ PASO CRÍTICO: Modifica esta ruta para que apunte a tu carpeta de entrenamiento
DATASET_PATH = r'C:\Users\284\Desktop\Escuela\9no Semestre\Inteligencia Artificial\CodigosClase\Emociones\emociones\train'

# Inicializar MediaPipe Face Mesh para el procesamiento de imágenes estáticas
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) 

# IDs de puntos clave: Usamos 4 puntos para generar 2 características
RIGHT_EYE_TOP = 386 
RIGHT_BROW_TOP = 296
LEFT_BROW_INNER = 105   # Punto interno de ceja izquierda
RIGHT_BROW_INNER = 334  # Punto interno de ceja derecha

def get_landmark_coords(landmark, width, height):
    """Convierte coordenadas normalizadas a píxeles."""
    return int(landmark.x * width), int(landmark.y * height)

# 1. Recolección de datos
# Features ahora almacenará una lista de listas: [[vertical_1, horizontal_1], [vertical_2, horizontal_2], ...]
features = [] 
labels = []   

print("Iniciando extracción de características del dataset (2 features)...")
for emotion_folder in os.listdir(DATASET_PATH):
    emotion_path = os.path.join(DATASET_PATH, emotion_folder)
    
    if os.path.isdir(emotion_path):
        print(f"  Procesando: {emotion_folder}")
        for filename in os.listdir(emotion_path):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(emotion_path, filename)
                
                image = cv2.imread(img_path)
                if image is None: continue
                
                height, width, _ = image.shape
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_image)

                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    
                    # Landmarks para la distancia vertical
                    p_eye_top = face_landmarks.landmark[RIGHT_EYE_TOP]
                    p_brow_top = face_landmarks.landmark[RIGHT_BROW_TOP]
                    # Landmarks para la distancia horizontal
                    p_brow_inner_left = face_landmarks.landmark[LEFT_BROW_INNER]
                    p_brow_inner_right = face_landmarks.landmark[RIGHT_BROW_INNER]
                    
                    # -- Característica 1: Distancia Vertical Ceja-Ojo
                    _, y_eye_top = get_landmark_coords(p_eye_top, width, height)
                    _, y_brow_top = get_landmark_coords(p_brow_top, width, height)
                    distance_vertical = y_eye_top - y_brow_top 

                    # -- Característica 2: Distancia Horizontal de las Cejas (fruncimiento)
                    x_brow_inner_left, _ = get_landmark_coords(p_brow_inner_left, width, height)
                    x_brow_inner_right, _ = get_landmark_coords(p_brow_inner_right, width, height)
                    # La distancia horizontal absoluta disminuye con el fruncimiento
                    distance_horizontal = abs(x_brow_inner_right - x_brow_inner_left)
                    
                    # Guardamos ambas características
                    features.append([distance_vertical, distance_horizontal]) 
                    labels.append(emotion_folder)
        
if not features:
    print("\n⚠️ ERROR: No se detectaron caras o imágenes válidas. Revisa la ruta y el dataset.")
else:
    # 2. Preprocesamiento y Entrenamiento
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    X = np.array(features)

    print(f"\nEtiquetas detectadas y codificadas: {list(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Entrenar el Árbol de Decisión
    dt_classifier = DecisionTreeClassifier(max_depth=7, random_state=42) # Aumentamos max_depth para más complejidad
    dt_classifier.fit(X_train, y_train)

    # Evaluación
    accuracy = dt_classifier.score(X_test, y_test)
    print(f"\nPrecisión del Árbol de Decisión (con 2 features): {accuracy:.2f}")

    # 4. Guardar el Modelo (sobrescribe los anteriores)
    joblib.dump(dt_classifier, 'decision_tree_emotion_model.joblib')
    joblib.dump(le, 'label_encoder.joblib')
    print("Modelo y codificador guardados como 'decision_tree_emotion_model.joblib' y 'label_encoder.joblib'.")