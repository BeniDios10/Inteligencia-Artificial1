import numpy as np
import os
from keras.models import load_model
from PIL import Image

# --- CONFIGURACIÓN ---
MODELO = "modelo_animales_v2.h5"
IMG_SIZE = (64, 64)

# ESTRICTO ORDEN ALFABÉTICO (Igual que las carpetas)
CLASES = ['ants', 'cats', 'dogs', 'ladybug', 'turtles'] 

def predecir(ruta_imagen):
    if not os.path.exists(ruta_imagen):
        print(f"Error: No encuentro la imagen '{ruta_imagen}'")
        return

    # Cargar modelo
    try:
        model = load_model(MODELO)
    except:
        print("Error: No encuentro el archivo .h5 (¿Ejecutaste entrenar_v2.py primero?)")
        return

    print(f"Analizando: {ruta_imagen} ...")

    # Procesar imagen igual que en el entrenamiento
    try:
        img = Image.open(ruta_imagen).convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = np.array(img).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0) # Batch de 1
    except Exception as e:
        print(f"Error al abrir la imagen: {e}")
        return

    # Predecir
    prediccion = model.predict(img_array)
    indice = np.argmax(prediccion)
    confianza = np.max(prediccion) * 100
    animal = CLASES[indice]

    print(f"\n================================")
    print(f" RESULTADO: {animal.upper()}")
    print(f" Confianza: {confianza:.2f}%")
    print(f"================================\n")

# --- EJECUTAR PRUEBA ---
if __name__ == "__main__":
    # CAMBIA ESTO
    mi_foto = "C:\\Users\\284\\Desktop\\nuevas_imagenes\\Tortuga_cuello.jpg"
    
    predecir(mi_foto)