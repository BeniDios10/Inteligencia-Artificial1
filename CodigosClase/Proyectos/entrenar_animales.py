import os
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LeakyReLU
import keras
from PIL import Image

# ----------------------------------------------------------------
# PASO 1: Cargar las imágenes
# ----------------------------------------------------------------

# Asumimos que la carpeta se llama "CNN_animales" basada en tu captura
dirname = os.path.join(os.getcwd(), 'CNN_animales')
imgpath = dirname + os.sep

images = []
directories = []
dircount = []
prevRoot = ''
cant = 0

print("Leyendo imagenes de: ", imgpath)

for root, dirnames, filenames in os.walk(imgpath):
    # IMPORTANTE: Ordenar directorios para asegurar que las etiquetas 0,1,2... 
    # siempre correspondan al mismo animal en orden alfabético.
    dirnames.sort()
    
    for filename in filenames:
        if re.search(r"\.(jpg|jpeg|png|bmp|tiff)$", filename):
            cant = cant + 1
            filepath = os.path.join(root, filename)
            
            # Forzar lectura correcta incluso si hay problemas de formato
            try:
                image = plt.imread(filepath)
                # Validar que la imagen tenga 3 canales (evitar grises que rompen el código)
                if len(image.shape) == 3:
                    images.append(image)
                else:
                    print(f"Saltando imagen en escala de grises: {filename}")
                    cant -= 1 # Ajustar contador si saltamos
            except:
                print(f"Error leyendo archivo: {filename}")
                cant -= 1

            if cant % 100 == 0:
                print(f"Procesando... {cant}", end="\r")

    if prevRoot != root:
        print(root, cant)
        prevRoot = root
        directories.append(root)
        dircount.append(cant)
        cant = 0

dircount.append(cant)
dircount = dircount[1:]
print('Directorios leidos: ', len(directories))
print("Imagenes en cada directorio", dircount)
print('Suma Total de imagenes en subdirs:', sum(dircount))

# ----------------------------------------------------------------
# PASO 2: Redimensionar y Etiquetas
# ----------------------------------------------------------------

# Mantenemos 64x64 como pediste (aunque para animales es pequeño)
IMG_SIZE = (64, 64) 

resized_images = []
for image in images:
    img = Image.fromarray(image)
    img = img.resize(IMG_SIZE)
    resized_images.append(np.array(img))

images = np.array(resized_images) # Convertir a numpy array directo

labels = []
indice = 0
for cantidad in dircount:
    for i in range(cantidad):
        labels.append(indice)
    indice = indice + 1

# Recuperar nombres de las clases
clases_nombres = []
indice = 0
for directorio in directories:
    name = directorio.split(os.sep)
    print(f"Clase {indice}: {name[-1]}")
    clases_nombres.append(name[-1])
    indice = indice + 1

print("\nIMPORTANTE: Copia esta lista para tu script de predicción:")
print(clases_nombres)

y = np.array(labels)
X = images.astype('float32') # Asegurar tipo float para normalizar

# Clases únicas
classes = np.unique(y)
nClasses = len(classes)

# ----------------------------------------------------------------
# PASO 3: Split y Normalización
# ----------------------------------------------------------------

train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar (0-1)
train_X = train_X / 255.
test_X = test_X / 255.

# One-hot encoding
train_Y_one_hot = to_categorical(train_Y, num_classes=nClasses)
test_Y_one_hot = to_categorical(test_Y, num_classes=nClasses)

# Split validación
train_X, valid_X, train_label, valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)

# ----------------------------------------------------------------
# PASO 4: Modelo (Corregido)
# ----------------------------------------------------------------

INIT_LR = 1e-3
epochs = 50 # Bajé un poco las épocas porque el dataset puede ser menor
batch_size = 64

sport_model = Sequential()
sport_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', padding='same', input_shape=(IMG_SIZE[1], IMG_SIZE[0], 3)))
sport_model.add(LeakyReLU(alpha=0.1))
sport_model.add(MaxPooling2D((2, 2), padding='same'))
sport_model.add(Dropout(0.5))

sport_model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
sport_model.add(LeakyReLU(alpha=0.1))
sport_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
sport_model.add(Dropout(0.5))

sport_model.add(Flatten())
sport_model.add(Dense(128, activation='linear'))
sport_model.add(LeakyReLU(alpha=0.1))
sport_model.add(Dropout(0.5))
sport_model.add(Dense(nClasses, activation='softmax'))

sport_model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adagrad(learning_rate=INIT_LR),
    metrics=['accuracy']
)

# --- AQUI ESTABA EL ERROR ---
# Solo entrenamos UNA VEZ y guardamos el resultado en 'history' para poder graficar si quisieras
print("Iniciando entrenamiento...")
history = sport_model.fit(
    train_X, 
    train_label, 
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(valid_X, valid_label)
)

# Guardar el modelo
model_name = "animales_model.h5"
sport_model.save(model_name)
print(f"\nModelo guardado como: {model_name}")

# Evaluar
test_eval = sport_model.evaluate(test_X, test_Y_one_hot, verbose=1)
print(f"Test Loss: {test_eval[0]}")
print(f"Test Accuracy: {test_eval[1]}")