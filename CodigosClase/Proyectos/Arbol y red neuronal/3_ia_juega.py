import pygame
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# --- CONFIGURACIÓN: ELIGIR MODELO AQUÍ ---
MODO = "ARBOL" # Cambiar a "ARBOL" o "RED" para probar cada uno
# -------------------------------------------

# Cargar el modelo seleccionado
if MODO == "ARBOL":
    modelo = joblib.load("modelo_arbol.pkl")
    print("Modo: Árbol de Decisión cargado.")
else:
    modelo = load_model("modelo_red.h5")
    print("Modo: Red Neuronal cargada.")

pygame.init()
w, h = 800, 400
pantalla = pygame.display.set_mode((w, h))
reloj = pygame.time.Clock()

jugador = pygame.Rect(50, h-50, 40, 50)
enemigo = pygame.Rect(w, h-50, 40, 50)
velocidad_enemigo = 10
saltando = False
salto_vel = 0
gravedad = 1

corriendo = True
while corriendo:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            corriendo = False
            
    # --- CEREBRO DE LA IA ---
    distancia = enemigo.x - jugador.x
    
    # Preparamos el dato igual que en el entrenamiento: [distancia, velocidad]
    dato_entrada = np.array([[distancia, velocidad_enemigo]])
    
    accion = 0
    if distancia > 0: # Solo decidir si el enemigo está enfrente
        if MODO == "ARBOL":
            accion = modelo.predict(dato_entrada)[0]
        else:
            # La red devuelve probabilidad (0.0 a 1.0). Cortamos en 0.5
            prediccion = modelo.predict(dato_entrada, verbose=0)
            accion = 1 if prediccion[0][0] > 0.5 else 0

    # Ejecutar la acción
    if accion == 1 and not saltando:
        saltando = True
        salto_vel = -15
    # ------------------------

    # Física del juego
    if saltando:
        jugador.y += salto_vel
        salto_vel += gravedad
        if jugador.y >= h-50:
            jugador.y = h-50
            saltando = False

    enemigo.x -= velocidad_enemigo
    if enemigo.x < -40:
        enemigo.x = w
        velocidad_enemigo = np.random.randint(8, 15)

    # Dibujar
    pantalla.fill((255, 255, 255))
    pygame.draw.rect(pantalla, (0, 255, 0), jugador) # La IA es Verde
    pygame.draw.rect(pantalla, (255, 0, 0), enemigo)
    
    # Mostrar qué está pensando
    fuente = pygame.font.SysFont("Arial", 20)
    texto = fuente.render(f"Distancia: {distancia} | IA Dice: {'SALTAR' if accion else 'CORRER'}", True, (0,0,0))
    pantalla.blit(texto, (10, 10))

    pygame.display.flip()
    reloj.tick(30)

pygame.quit()