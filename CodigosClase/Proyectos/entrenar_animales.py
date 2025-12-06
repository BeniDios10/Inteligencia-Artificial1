import os
import re
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LeakyReLU
import keras
from PIL import Image

# ----------------------------------------------------------------
# PASO 1: Cargar y Procesar Imágenes (Versión Blindada)
# ----------------------------------------------------------------

# Ajusta esto si tu carpeta se llama diferente, pero según tu foto es esta:
dirname = os.path.join(os.getcwd(), 'CNN_animales')
imgpath = dirname + os.sep

# Configuración
IMG_SIZE = (64, 64)
images = []
directories = []
dircount = []
prevRoot = ''
cant = 0

print("Leyendo imagenes de: ", imgpath)

for root, dirnames, filenames in os.walk(imgpath):
    # Ordenamos carpetas alfabéticamente para que las etiquetas sean consistentes
    dirnames.sort()
    
    for filename in filenames:
        if re.search(r"\.(jpg|jpeg|png|bmp|tiff)$", filename):
            filepath = os.path.join(root, filename)
            
            try:
                # CAMBIO CLAVE: Usamos PIL directo. 
                # .convert('RGB') arregla los PNGs transparentes y los Grayscale de golpe.
                img_pil = Image.open(filepath).convert('RGB')
                
                # Redimensionamos de una vez
                img_pil = img_pil.resize(IMG_SIZE)
                
                # Convertimos a arreglo y guardamos
                images.append(np.array(img_pil))
                
                cant += 1
                
            except Exception as e:
                print(f"Archivo corrupto o ilegible: {filename} - Error: {e}")

            if cant % 100 == 0:
                print(f"Procesando... {cant}", end="\r")

    if prevRoot != root:
        # Solo agregar a la lista si encontramos archivos en esa carpeta
        if cant > 0: 
            print(f"\nCarpeta finalizada: {root} -> {cant} imagenes")
            directories.append(root)
            dircount.append(cant)
            prevRoot = root
            cant = 0

# Manejo del último directorio
if cant > 0:
    dircount.append(cant)
# El primer elemento de dircount suele ser 0 si la raiz no tenia fotos, lo limpiamos si es necesario
if len(dircount) > len(directories):
    dircount = dircount[1:]

print('\n--------------------------------')
print('Directorios leidos:', len(directories))
print("Imagenes por directorio:", dircount)
print('Total imagenes:', sum(dircount))
print('--------------------------------')

# ----------------------------------------------------------------
# PASO 2: Crear Etiquetas
# ----------------------------------------------------------------

labels = []
indice = 0
for cantidad in dircount:
    for i in range(cantidad):
        labels.append(indice)
    indice += 1

# Recuperar nombres de las clases para tu referencia
clases_nombres = []
for directorio in directories:
    name = directorio.split(os.sep)
    clases_nombres.append(name[-1])

print("Clases detectadas:", clases_nombres)

y = np.array(labels)
X = np.array(images, dtype=np.float32) # Convertimos la lista de arrays en un super array

# Validar que X y y tengan el mismo largo
if len(X) != len(y):
    print(f"¡ALERTA! Desajuste de dimensiones. X: {len(X)}, y: {len(y)}")
    # Esto no debería pasar con la lógica nueva, pero por seguridad:
    min_len = min(len(X), len(y))
    X = X[:min_len]
    y = y[:min_len]

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
# PASO 4: Modelo 
# ----------------------------------------------------------------

INIT_LR = 1e-3
epochs = 50 
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

print("\nIniciando entrenamiento...")
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
print(f"\nModelo guardado exitosamente como: {model_name}")

# Evaluar
test_eval = sport_model.evaluate(test_X, test_Y_one_hot, verbose=1)
print(f"Test Loss: {test_eval[0]}")
print(f"Test Accuracy: {test_eval[1]}")