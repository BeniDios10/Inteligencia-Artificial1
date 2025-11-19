import cv2
import mediapipe as mp
import numpy as np
import joblib 

# -------------------------------------------------------------------
# CONFIGURACIÓN Y CARGA DE MODELO
# -------------------------------------------------------------------

try:
    dt_classifier = joblib.load('decision_tree_emotion_model.joblib')
    le = joblib.load('label_encoder.joblib')
    EMOTION_LABELS = {i: label for i, label in enumerate(le.classes_)}
    print(f"Modelo de Árbol de Decisión cargado exitosamente. Clases: {EMOTION_LABELS}")
    MODEL_LOADED = True
except FileNotFoundError:
    print("⚠️ ERROR: No se encontraron los archivos del modelo. Ejecuta 'entrenar_dt.py' primero.")
    MODEL_LOADED = False
    exit() 

# ... (Configuración de MediaPipe, puntos clave, y funciones son iguales)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(234, 255, 233)) 

RIGHT_EYE_TOP = 386 
RIGHT_BROW_TOP = 296 
LEFT_BROW_INNER = 105
RIGHT_BROW_INNER = 334 

def get_landmark_coords(landmark, width, height):
    x = int(landmark.x * width)
    y = int(landmark.y * height)
    return x, y
# ...

cap = cv2.VideoCapture(0)

# CALIBRACIÓN: Ahora calibramos ambos baselines en tiempo real.
baseline_v = 0
baseline_h = 0
calibration_frames = 30
calibrated = False

print("Calibrando la distancia base (Vertical y Horizontal)... Mantén una expresión neutra.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    emotion_status = "Esperando..."
    color = (255, 255, 255)
    
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # 1. CÁLCULO DE DISTANCIAS
        p_eye_top = face_landmarks.landmark[RIGHT_EYE_TOP]
        p_brow_top = face_landmarks.landmark[RIGHT_BROW_TOP]
        p_brow_inner_left = face_landmarks.landmark[LEFT_BROW_INNER]
        p_brow_inner_right = face_landmarks.landmark[RIGHT_BROW_INNER]
        
        _, y_eye_top = get_landmark_coords(p_eye_top, width, height)
        _, y_brow_top = get_landmark_coords(p_brow_top, width, height)
        current_distance_vertical = y_eye_top - y_brow_top
        
        x_brow_inner_left, _ = get_landmark_coords(p_brow_inner_left, width, height)
        x_brow_inner_right, _ = get_landmark_coords(p_brow_inner_right, width, height)
        current_distance_horizontal = abs(x_brow_inner_right - x_brow_inner_left)

        # 2. CALIBRACIÓN
        if not calibrated and calibration_frames > 0:
            baseline_v += current_distance_vertical
            baseline_h += current_distance_horizontal
            calibration_frames -= 1
            cv2.putText(frame, f"Calibrando... {calibration_frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            if calibration_frames == 0:
                baseline_v /= 30 
                baseline_h /= 30
                calibrated = True
                print(f"Calibración completa. Baseline V: {baseline_v:.2f} px, Baseline H: {baseline_h:.2f} px")

        # 3. ANÁLISIS CON ÁRBOL DE DECISIÓN (USANDO NORMALIZACIÓN)
        elif calibrated and MODEL_LOADED:
            
            # NORMALIZACIÓN EN TIEMPO REAL
            # Esta es la CARACTERÍSTICA que el modelo espera: el cambio porcentual
            change_vertical = ((current_distance_vertical - baseline_v) / baseline_v) * 100
            change_horizontal = ((current_distance_horizontal - baseline_h) / baseline_h) * 100
            
            feature_input = np.array([[change_vertical, change_horizontal]])
            
            predicted_class_index = dt_classifier.predict(feature_input)[0]
            emotion_status = EMOTION_LABELS.get(predicted_class_index, "Clase Desconocida")

            # Mapear colores (ejemplo)
            if 'surprise' in emotion_status or 'fear' in emotion_status:
                color = (0, 165, 255)
            elif 'angry' in emotion_status or 'sad' in emotion_status:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            # Mostrar estado en pantalla
            cv2.putText(frame, emotion_status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            # Mostrar ambas características normalizadas
            cv2.putText(frame, f"V%:{change_vertical:.1f} H%:{change_horizontal:.1f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, drawing_spec, drawing_spec)
    
    else:
        cv2.putText(frame, "Cara No Detectada", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


    cv2.imshow('Detector de Emociones con Arbol de Decision (Normalizado)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()