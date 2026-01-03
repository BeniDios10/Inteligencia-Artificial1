import os
import re
import numpy as np # Para manejo de arrays
import matplotlib.pyplot as plt # Para gráficas
from sklearn.model_selection import train_test_split # Para dividir el dataset
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # Capas de la red neuronal
from PIL import Image

#El modelo tiene una entrada de dimensiones (64, 64, 3). Tiene 3 dimensiones de entrada, entran realmente a la red matemática, son 12,288 valores.

# ----------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------
nombre_carpeta_dataset = 'CNN_animales'
IMG_SIZE = (64, 64) # Ancho, Alto
BATCH_SIZE = 32 # Tamaño del lote para el entrenamiento
EPOCHS = 20 # Número de épocas para el entrenamiento
MAX_IMGS_PER_CLASS = 1500  # Límite para balancear el dataset

dirname = os.path.join(os.getcwd(), nombre_carpeta_dataset) # Ruta completa a la carpeta del dataset
imgpath = dirname + os.sep # Ruta base para las imágenes
images = [] # Lista de imágenes
labels = [] # Lista de etiquetas
class_names = [] # Nombres de las clases

# ----------------------------------------------------------------
# 1. CARGA DE IMÁGENES (BALANCEADA Y BLINDADA)
# ----------------------------------------------------------------
print(f"--- INICIANDO PROCESO ---")
print(f"Buscando dataset en: {dirname}")

if not os.path.exists(dirname):
    print("ERROR: No encuentro la carpeta. Revisa el nombre.")
    exit()

# Ordenar carpetas alfabéticamente
carpetas = sorted([d for d in os.listdir(dirname) if os.path.isdir(os.path.join(dirname, d))])
class_names = carpetas
print(f"Clases detectadas: {class_names}")

total_global = 0

for indice, nombre_clase in enumerate(carpetas):
    ruta_clase = os.path.join(dirname, nombre_clase)
    print(f"Procesando clase '{nombre_clase}'...")
    
    archivos = os.listdir(ruta_clase)
    np.random.shuffle(archivos) # Mezclar para no tomar siempre las mismas
    
    leidas_clase = 0
    
    for archivo in archivos:
        if leidas_clase >= MAX_IMGS_PER_CLASS:
            break # Ya tenemos suficientes de este animal

        if re.search(r"\.(jpg|jpeg|png|bmp|tiff)$", archivo, re.IGNORECASE): # Formatos permitidos
            path_completo = os.path.join(ruta_clase, archivo)
            try:
                #.convert('RGB') arregla PNGs y transparencias
                img = Image.open(path_completo).convert('RGB')
                img = img.resize(IMG_SIZE)
                
                images.append(np.array(img))
                labels.append(indice)
                
                leidas_clase += 1
                total_global += 1
                
            except Exception as e:
                pass 

    print(f"   -> Guardadas: {leidas_clase} imágenes.")

print(f"\nTotal imágenes para entrenamiento: {total_global}")

# ----------------------------------------------------------------
# 2. PREPARACIÓN DE DATOS
# ----------------------------------------------------------------
X = np.array(images, dtype=np.float32) # Convertir a array de numpy
y = np.array(labels) # Etiquetas como array de numpy

# Normalizar (0 a 1)
X = X / 255.0 # Normalización de píxeles. Las redes neuronales convergen (aprenden) mucho más rápido con números pequeños y flotantes que con enteros grandes

# One-Hot Encoding
nClasses = len(class_names) # Número de clases
y_one_hot = to_categorical(y, num_classes=nClasses) # Convertir etiquetas a one-hot encoding

# Split 80% Entrenar - 20% Probar
train_X, test_X, train_Y, test_Y = train_test_split(X, y_one_hot, test_size=0.2, random_state=42) #Separa el 80% de los datos para estudiar (entrenar) y guarda el 20% para el examen final

# ----------------------------------------------------------------
# 3. ARQUITECTURA (1 CAPA CONVOLUCIONAL)
# ----------------------------------------------------------------
model = Sequential()

# ÚNICA CAPA DE CONVOLUCIÓN
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(IMG_SIZE[1], IMG_SIZE[0], 3))) #Aplica 32 filtros de 3x3.
model.add(MaxPooling2D(pool_size=(2, 2))) #Reduce la imagen a la mitad (toma el valor máximo de cada cuadro 2x2). Se queda con lo más importante.

model.add(Flatten()) #Aplana la matriz 2D a 1D para conectarla a la capa densa. Aquí ya entendío la imágen
model.add(Dense(64, activation='relu')) #Capa oculta con 64 neuronas
model.add(Dropout(0.5)) #Evita overfitting apagando el 50% de las neuronas aleatoriamente durante el entrenamiento.
model.add(Dense(nClasses, activation='softmax')) # Salida. Que dé la más alta.

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #Función de pérdida y optimizador

# ----------------------------------------------------------------
# 4. ENTRENAMIENTO
# ----------------------------------------------------------------
print("\n--- ENTRENANDO MODELO... ---")
history = model.fit(train_X, train_Y, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(test_X, test_Y)) #Entrena el modelo con los datos de entrenamiento y valida con los datos de prueba.

# Guardar
nombre_archivo = "modelo_animales_v2.h5"
model.save(nombre_archivo)
print(f"\n¡EXITO! Modelo guardado como: {nombre_archivo}")