import cv2
import mediapipe as mp
import numpy as np
import joblib 

# -------------------------------------------------------------------
# CONFIGURACIÓN Y CARGA DE MODELO
# -------------------------------------------------------------------

try:
    # Cargar el modelo de Árbol de Decisión y el codificador de etiquetas
    dt_classifier = joblib.load('decision_tree_emotion_model.joblib')
    le = joblib.load('label_encoder.joblib')
    # Crear un diccionario para mapear las clases numéricas a sus nombres
    EMOTION_LABELS = {i: label for i, label in enumerate(le.classes_)}
    print(f"Modelo de Árbol de Decisión cargado exitosamente. Clases: {EMOTION_LABELS}")
    MODEL_LOADED = True
except FileNotFoundError:
    print("⚠️ ERROR: No se encontraron los archivos del modelo. Ejecuta 'entrenar_dt.py' primero.")
    MODEL_LOADED = False
    exit() 


# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

# Inicializar dibujador de MediaPipe
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(234, 255, 233)) 

# IDs de puntos clave: DEBEN COINCIDIR con el script de entrenamiento
RIGHT_EYE_TOP = 386 
RIGHT_BROW_TOP = 296 
LEFT_BROW_INNER = 105
RIGHT_BROW_INNER = 334 

def get_landmark_coords(landmark, width, height):
    """Se escalan las coordenadas normalizadas (0 a 1) a píxeles."""
    x = int(landmark.x * width)
    y = int(landmark.y * height)
    return x, y

# Captura de video
cap = cv2.VideoCapture(0)

# CALIBRACIÓN DE LÍNEA BASE (Solo usamos la vertical para el promedio neutro)
baseline_distance_vertical = 0
calibration_frames = 30
calibrated = False

print("Calibrando la distancia base de la ceja/ojo... Mantén una expresión neutra.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1) # Espejo
    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    emotion_status = "Esperando..."
    color = (255, 255, 255)
    
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # 1. Obtener coordenadas de los puntos
        p_eye_top = face_landmarks.landmark[RIGHT_EYE_TOP]
        p_brow_top = face_landmarks.landmark[RIGHT_BROW_TOP]
        p_brow_inner_left = face_landmarks.landmark[LEFT_BROW_INNER]
        p_brow_inner_right = face_landmarks.landmark[RIGHT_BROW_INNER]
        
        # CÁLCULO DE LA CARACTERÍSTICA VERTICAL
        _, y_eye_top = get_landmark_coords(p_eye_top, width, height)
        _, y_brow_top = get_landmark_coords(p_brow_top, width, height)
        current_distance_vertical = y_eye_top - y_brow_top
        
        # CÁLCULO DE LA CARACTERÍSTICA HORIZONTAL
        x_brow_inner_left, _ = get_landmark_coords(p_brow_inner_left, width, height)
        x_brow_inner_right, _ = get_landmark_coords(p_brow_inner_right, width, height)
        current_distance_horizontal = abs(x_brow_inner_right - x_brow_inner_left)

        # 2. CALIBRACIÓN
        if not calibrated and calibration_frames > 0:
            baseline_distance_vertical += current_distance_vertical
            calibration_frames -= 1
            cv2.putText(frame, f"Calibrando... {calibration_frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            if calibration_frames == 0:
                baseline_distance_vertical /= 30 
                calibrated = True
                print(f"Calibración completa. Distancia vertical base: {baseline_distance_vertical:.2f} px")

        # 3. ANÁLISIS CON ÁRBOL DE DECISIÓN
        elif calibrated and MODEL_LOADED:
            
            # El input del modelo debe ser: [distancia_vertical, distancia_horizontal]
            feature_input = np.array([[current_distance_vertical, current_distance_horizontal]])
            
            # Predicción de la clase (índice numérico)
            predicted_class_index = dt_classifier.predict(feature_input)[0]
            
            # Mapear el índice de clase a la etiqueta de emoción
            emotion_status = EMOTION_LABELS.get(predicted_class_index, "Clase Desconocida")

            # Mapear colores (ejemplo, ajusta según tus clases)
            if 'surprise' in emotion_status or 'fear' in emotion_status:
                color = (0, 165, 255) # Naranja/Azul (emociones de ceja alta)
            elif 'angry' in emotion_status or 'sad' in emotion_status:
                color = (0, 0, 255) # Rojo (emociones de ceja baja o fruncida)
            else:
                color = (0, 255, 0) # Verde (Neutro/Happy)

            # Mostrar estado en pantalla
            cv2.putText(frame, emotion_status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            # Mostrar ambas características para depuración
            cv2.putText(frame, f"Vert: {current_distance_vertical:.1f} Horiz: {current_distance_horizontal:.1f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Dibujar los landmarks faciales
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, drawing_spec, drawing_spec)
    
    else:
        cv2.putText(frame, "Cara No Detectada", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


    cv2.imshow('Detector de Emociones con Arbol de Decision', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()