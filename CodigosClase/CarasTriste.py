import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
# Usamos un conjunto de conexiones más densas para las cejas, FACEMESH_CONTOURS es más preciso
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, # Solo procesaremos una cara para simplificar
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

# Inicializar dibujador de MediaPipe
mp_drawing = mp.solutions.drawing_utils
# Puntos y líneas verdes (ajustado para ser menos intrusivo)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(234, 255, 233)) 

# IDs de puntos clave de MediaPipe Face Mesh
# Puedes buscar la imagen de las 468 coordenadas, pero aquí tienes algunos importantes:
# Ojo derecho (borde superior e inferior, en el centro): 386, 374
# Ceja derecha (parte superior central): 296
RIGHT_EYE_TOP = 386 
RIGHT_EYE_BOTTOM = 374
RIGHT_BROW_TOP = 296 

# Función para obtener las coordenadas (x, y) de un landmark
def get_landmark_coords(landmark, width, height):
    # Se escalan las coordenadas normalizadas (0 a 1) a píxeles
    x = int(landmark.x * width)
    y = int(landmark.y * height)
    return x, y

# Captura de video
cap = cv2.VideoCapture(0)

# -----------------
# CALIBRACIÓN DE LÍNEA BASE
# -----------------
# Esta es la métrica más simple: la distancia vertical entre la ceja y el ojo.
# La calibraremos con el primer frame.
baseline_distance = 0
calibration_frames = 30
calibrated = False

print("Calibrando la distancia base de la ceja/ojo... Mantén una expresión neutra.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) # Espejo
    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Inicialmente, la emoción es 'Neutro'
    emotion_status = "Neutro"
    
    if results.multi_face_landmarks:
        # Procesamos solo la primera cara (índice 0)
        face_landmarks = results.multi_face_landmarks[0]

        # 1. Obtener coordenadas de los puntos
        p_eye_top = face_landmarks.landmark[RIGHT_EYE_TOP]
        p_brow_top = face_landmarks.landmark[RIGHT_BROW_TOP]
        
        # Coordenadas en píxeles (y es la importante aquí)
        _, y_eye_top = get_landmark_coords(p_eye_top, width, height)
        _, y_brow_top = get_landmark_coords(p_brow_top, width, height)
        
        # 2. CALIBRACIÓN
        if not calibrated and calibration_frames > 0:
            current_distance = y_eye_top - y_brow_top # Distancia vertical (px)
            baseline_distance += current_distance
            calibration_frames -= 1
            cv2.putText(frame, f"Calibrando... {calibration_frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            if calibration_frames == 0:
                baseline_distance /= 30 # Promedio de los 30 frames
                calibrated = True
                print(f"Calibración completa. Distancia base: {baseline_distance:.2f} px")

        # 3. ANÁLISIS DESPUÉS DE LA CALIBRACIÓN
        if calibrated:
            current_distance = y_eye_top - y_brow_top
            
            # Cálculo de la diferencia porcentual (cuánto ha cambiado respecto a la línea base)
            # Un valor positivo significa que la ceja se ha alejado del ojo (distancia ha crecido)
            # Un valor negativo significa que la ceja se ha acercado al ojo (distancia ha disminuido)
            distance_change_percent = ((current_distance - baseline_distance) / baseline_distance) * 100
            
            # Lógica BÁSICA (Ajusta estos umbrales con pruebas)
            if distance_change_percent > 15: # Ceja levantada más del 15%
                emotion_status = "¡Sorpresa/Miedo! (Cejas arriba)"
                color = (0, 165, 255) # Naranja
            elif distance_change_percent < -10: # Ceja fruncida más del 10%
                emotion_status = "¡Enojo/Tristeza! (Cejas abajo)"
                color = (0, 0, 255) # Rojo
            else:
                emotion_status = "Neutro/Relajado"
                color = (0, 255, 0) # Verde

            # Mostrar estado en pantalla
            cv2.putText(frame, emotion_status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            # Mostrar la variación porcentual para depuración
            cv2.putText(frame, f"Cambio: {distance_change_percent:.1f}%", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Dibujar los landmarks faciales
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, drawing_spec, drawing_spec)
    
    else:
         cv2.putText(frame, "Cara No Detectada", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


    cv2.imshow('Detector de Patrones Faciales Simplificado', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()