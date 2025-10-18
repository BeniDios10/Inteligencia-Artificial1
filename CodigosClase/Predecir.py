import numpy as np
import os
from keras.models import load_model
from PIL import Image

# ----------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------
MODELO_H5 = "sports_mnist_2.h5" 
IMG_SIZE = (64, 64) 

# ¡LISTA DE CLASES ACTUALIZADA! (debe coincidir con el orden alfabético)
DEPORTES = [
    "americano",   # Índice 0
    "basket",      # Índice 1
    "beisball",    # Índice 2
    "boxeo",       # Índice 3
    "ciclismo",    # Índice 4
    "f1",          # Índice 5
    "futbol",      # Índice 6
    "golf",        # Índice 7
    "natacion",    # Índice 8
    "tenis"        # Índice 9
] 

# ----------------------------------------------------------------
# FUNCIÓN DE PREDICCIÓN (El resto de la función es el mismo)
# ----------------------------------------------------------------
def clasificar_imagen(ruta_imagen, modelo, nombres_clases):
    """Carga, preprocesa una imagen y usa el modelo para clasificarla."""
    
    if not os.path.exists(ruta_imagen):
        print(f"ERROR: No se encontró el archivo en la ruta: {ruta_imagen}")
        return

    print(f"Clasificando imagen: {ruta_imagen}")
    
    # 1. Cargar y redimensionar la imagen (PREPROCESAMIENTO)
    try:
        img = Image.open(ruta_imagen).resize(IMG_SIZE)
    except Exception as e:
        print(f"Error al cargar la imagen: {e}")
        return

    # 2. Convertir a array de numpy
    img_array = np.array(img).astype('float32')

    # Verificar que la imagen tenga 3 canales (RGB)
    if img_array.ndim != 3 or img_array.shape[2] != 3:
        print("ERROR: La imagen debe ser a color (RGB) y tener 3 dimensiones.")
        return

    # 3. Normalizar los valores de píxeles (de 0-255 a 0-1)
    img_array = img_array / 255.0
    
    # 4. Agregar una dimensión de "lote" (batch) para Keras
    img_array = np.expand_dims(img_array, axis=0) 
    
    # 5. Realizar la predicción
    print("Realizando predicción...")
    predicciones = modelo.predict(img_array)
    
    # 6. Obtener el resultado
    clase_indice = np.argmax(predicciones[0])
    probabilidad = predicciones[0][clase_indice] * 100
    
    clase_predicha = nombres_clases[clase_indice]
    
    print("\n--- RESULTADO DE CLASIFICACIÓN ---")
    print(f"Clase predicha: {clase_predicha}")
    print(f"Probabilidad: {probabilidad:.2f}%")
    print("-----------------------------------")


# ----------------------------------------------------------------
# EJECUCIÓN PRINCIPAL
# ----------------------------------------------------------------

if __name__ == "__main__":
    
    # 1. Cargar el modelo guardado
    try:
        modelo_cargado = load_model(MODELO_H5)
        print(f"Modelo '{MODELO_H5}' cargado exitosamente.")
    except Exception as e:
        print(f"ERROR: No se pudo cargar el modelo. Asegúrate de que el archivo '{MODELO_H5}' esté en la misma carpeta.")
        exit()

    # 2. DEFINIR LA RUTA DE LA IMAGEN QUE QUIERES PROBAR
    # EJEMPLO: Asegúrate de que esta imagen exista
    #ruta_nueva_imagen = os.path.join(os.getcwd(), 'sportimages', 'futbol', 'futbol_0001.jpg') 
    ruta_nueva_imagen = os.path.join(os.getcwd(), 'play.jpg')

    # 3. Ejecutar la clasificación
    clasificar_imagen(ruta_nueva_imagen, modelo_cargado, DEPORTES)