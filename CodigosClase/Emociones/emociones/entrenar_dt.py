import cv2
import mediapipe as mp
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib 

DATASET_PATH = r'C:\Users\284\Desktop\Escuela\9no Semestre\Inteligencia Artificial\CodigosClase\Emociones\emociones\train'

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) 

RIGHT_EYE_TOP = 386 
RIGHT_BROW_TOP = 296
LEFT_BROW_INNER = 105
RIGHT_BROW_INNER = 334

def get_landmark_coords(landmark, width, height):
    return int(landmark.x * width), int(landmark.y * height)

# -------------------------------------------------------------
# PASO 1A: CALCULAR EL BASELINE DE ENTRENAMIENTO (PROMEDIO NEUTRO)
# -------------------------------------------------------------
baseline_v_list = []
baseline_h_list = []
neutral_path = os.path.join(DATASET_PATH, 'neutral')

print("Calculando el baseline (promedio de la carpeta 'neutral')...")
if os.path.exists(neutral_path):
    for filename in os.listdir(neutral_path):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(neutral_path, filename)
            image = cv2.imread(img_path)
            if image is None: continue
            
            height, width, _ = image.shape
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # Vertical
                _, y_eye_top = get_landmark_coords(face_landmarks.landmark[RIGHT_EYE_TOP], width, height)
                _, y_brow_top = get_landmark_coords(face_landmarks.landmark[RIGHT_BROW_TOP], width, height)
                baseline_v_list.append(y_eye_top - y_brow_top)
                
                # Horizontal
                x_brow_inner_left, _ = get_landmark_coords(face_landmarks.landmark[LEFT_BROW_INNER], width, height)
                x_brow_inner_right, _ = get_landmark_coords(face_landmarks.landmark[RIGHT_BROW_INNER], width, height)
                baseline_h_list.append(abs(x_brow_inner_right - x_brow_inner_left))

BASELINE_VERTICAL = np.mean(baseline_v_list) if baseline_v_list else 1.0
BASELINE_HORIZONTAL = np.mean(baseline_h_list) if baseline_h_list else 1.0

print(f"Baseline Vertical (Neutro): {BASELINE_VERTICAL:.2f} px")
print(f"Baseline Horizontal (Neutro): {BASELINE_HORIZONTAL:.2f} px")
# -------------------------------------------------------------

# -------------------------------------------------------------
# PASO 1B: EXTRACCIÓN DE CARACTERÍSTICAS NORMALIZADAS
# -------------------------------------------------------------
features = [] 
labels = []   

print("\nExtrayendo características NORMALIZADAS...")
for emotion_folder in os.listdir(DATASET_PATH):
    emotion_path = os.path.join(DATASET_PATH, emotion_folder)
    
    if os.path.isdir(emotion_path):
        for filename in os.listdir(emotion_path):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(emotion_path, filename)
                
                image = cv2.imread(img_path)
                if image is None: continue
                
                height, width, _ = image.shape
                results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    
                    # CÁLCULO DE DISTANCIAS
                    _, y_eye_top = get_landmark_coords(face_landmarks.landmark[RIGHT_EYE_TOP], width, height)
                    _, y_brow_top = get_landmark_coords(face_landmarks.landmark[RIGHT_BROW_TOP], width, height)
                    current_distance_vertical = y_eye_top - y_brow_top 

                    x_brow_inner_left, _ = get_landmark_coords(face_landmarks.landmark[LEFT_BROW_INNER], width, height)
                    x_brow_inner_right, _ = get_landmark_coords(face_landmarks.landmark[RIGHT_BROW_INNER], width, height)
                    current_distance_horizontal = abs(x_brow_inner_right - x_brow_inner_left)
                    
                    # NORMALIZACIÓN (El nuevo input X para el modelo!)
                    # El cambio porcentual hace que el modelo sea más robusto al tamaño de la cara
                    change_vertical = ((current_distance_vertical - BASELINE_VERTICAL) / BASELINE_VERTICAL) * 100
                    change_horizontal = ((current_distance_horizontal - BASELINE_HORIZONTAL) / BASELINE_HORIZONTAL) * 100
                    
                    features.append([change_vertical, change_horizontal]) 
                    labels.append(emotion_folder)
        
if not features:
    print("\n⚠️ ERROR: No se detectaron features. Revisa la ruta y el dataset.")
else:
    # 2. Preprocesamiento y Entrenamiento (Resto del código igual)
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    X = np.array(features)

    print(f"\nEtiquetas detectadas y codificadas: {list(zip(le.classes_, le.transform(le.classes_)))}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    dt_classifier = DecisionTreeClassifier(max_depth=7, random_state=42)
    dt_classifier.fit(X_train, y_train)

    accuracy = dt_classifier.score(X_test, y_test)
    print(f"\nPrecisión del Árbol de Decisión (NORMALIZADA): {accuracy:.2f}")

    joblib.dump(dt_classifier, 'decision_tree_emotion_model.joblib')
    joblib.dump(le, 'label_encoder.joblib')
    print("Modelo y codificador guardados.")