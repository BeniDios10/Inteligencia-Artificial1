import cv2 as cv
import numpy as np

img = cv.imread('figura.png', 1)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
ubb=(0, 100, 100)
uba=(10, 255, 255)
ubb2=(170, 100, 100)
uba2=(180, 255, 255)

# Crear la máscara para el color rojo
mask1 = cv.inRange(hsv, ubb, uba)
mask2 = cv.inRange(hsv, ubb2, uba2)
mask = mask1 + mask2

# --- Nuevo código para encontrar el centroide sin contornos ---
# Obtener las coordenadas de todos los píxeles blancos (valor 255) en la máscara
y_coords, x_coords = np.where(mask == 255)

if len(x_coords) > 0:
    # Calcular el promedio de las coordenadas X e Y para encontrar el centroide
    cX = int(np.mean(x_coords))
    cY = int(np.mean(y_coords))
    
    # Dibujar un círculo en el centro de la "mancha"
    cv.circle(img, (cX, cY), 5, (0, 255, 0), -1) 
    cv.putText(img, f"({cX}, {cY})", (cX + 10, cY), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    print(f"Coordenadas del centro de la mancha roja: ({cX}, {cY})")
else:
    print("No se encontraron píxeles de color rojo.")

# --- Fin del nuevo código ---
res = cv.bitwise_and(img, img, mask=mask)
cv.imshow('mask', mask)
cv.imshow('res', res)
cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()

#*Verde:
#ubb=(35, 100, 100)
#uba=(85, 255, 255)

#Azul:
#ubb=(95, 100, 100)
#uba=(130, 255, 255)

#Amarillo:
#ubb=(20, 100, 100)
#uba=(30, 255, 255)

#Rojo:
#ACTIVAR Mmask2
#ubb=(0, 100, 100)
#uba=(10, 255, 255)
#ubb2=(170, 100, 100)
#uba2=(180, 255, 255)