import numpy as np
import os
from keras.models import load_model
from PIL import Image

# ----------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------
MODELO_H5 = "animales_model.h5" 
IMG_SIZE = (64, 64) 

# Estas clases deben coincidir EXACTAMENTE con el orden alfabético de tus carpetas
ANIMALES = [
    "ants",     # 0
    "cats",     # 1
    "dogs",     # 2
    "ladybug",  # 3
    "turtles"   # 4
]

def clasificar_imagen(ruta_imagen):
    # 1. Cargar modelo
    try:
        if not os.path.exists(MODELO_H5):
            print(f"ERROR: No encuentro el archivo '{MODELO_H5}'. Ejecuta el entrenamiento primero.")
            return
        modelo = load_model(MODELO_H5)
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return

    # 2. Cargar imagen
    if not os.path.exists(ruta_imagen):
        print(f"ERROR: No encuentro la imagen: {ruta_imagen}")
        return

    print(f"Analizando: {ruta_imagen}")
    
    try:
        img = Image.open(ruta_imagen).resize(IMG_SIZE)
        img_array = np.array(img).astype('float32') / 255.0
    except Exception as e:
        print(f"Error procesando imagen: {e}")
        return

    # Validar dimensiones
    if img_array.ndim != 3 or img_array.shape[2] != 3:
        print("ERROR: La imagen debe ser RGB (Color).")
        return

    # Expandir dimensiones para Keras (batch size de 1)
    img_array = np.expand_dims(img_array, axis=0) 

    # 3. Predecir
    predicciones = modelo.predict(img_array)
    clase_indice = np.argmax(predicciones[0])
    probabilidad = predicciones[0][clase_indice] * 100
    
    resultado = ANIMALES[clase_indice]

    print(f"\n>>> ES UN: {resultado.upper()} ({probabilidad:.2f}%)")


# ----------------------------------------------------------------
# EJECUCIÓN
# ----------------------------------------------------------------
if __name__ == "__main__":
    # CAMBIA ESTO POR LA RUTA DE TU IMAGEN DE PRUEBA
    # Puedes poner una ruta absoluta ejemplo: "C:/Users/TuUsuario/Desktop/foto_gato.jpg"
    # mi_imagen = "prueba.jpg" 
    mi_imagen = "C:\\Users\\284\\Downloads\\mira.jpg"
    #mi_imagen = "C:\\Users\\Josue\\Downloads\\nuevas_imagenes\\xolo.jpeg"
    # C:\Users\Josue\Downloads\nuevas_imagenes\fea.jpeg

    clasificar_imagen(mi_imagen)