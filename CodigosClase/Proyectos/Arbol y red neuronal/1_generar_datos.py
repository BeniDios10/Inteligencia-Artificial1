import pygame
import pandas as pd
import random

# Configuración básica
pygame.init()
w, h = 800, 400
pantalla = pygame.display.set_mode((w, h))
reloj = pygame.time.Clock()
datos = [] # Aquí guardaremos la información

# Variables del juego
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
        
        # Detectar el salto (Input humano)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and not saltando:
                saltando = True
                salto_vel = -15
                
    # Lógica del salto
    if saltando:
        jugador.y += salto_vel
        salto_vel += gravedad
        if jugador.y >= h-50:
            jugador.y = h-50
            saltando = False

    # Mover enemigo
    enemigo.x -= velocidad_enemigo
    if enemigo.x < -40:
        enemigo.x = w
        velocidad_enemigo = random.randint(8, 15) # Variar velocidad para que aprenda mejor

    # --- RECOLECCIÓN DE DATOS ---
    # Guardamos: [Distancia al enemigo, Velocidad del enemigo, ¿Estoy saltando?]
    # 1 si está saltando, 0 si no
    accion = 1 if saltando else 0
    distancia = enemigo.x - jugador.x
    
    # Solo guardamos datos relevantes (cuando el enemigo viene de frente)
    if distancia > 0 and distancia < 600:
        datos.append([distancia, velocidad_enemigo, accion])

    # Dibujar
    pantalla.fill((255, 255, 255))
    pygame.draw.rect(pantalla, (0, 0, 255), jugador) # Jugador Azul
    pygame.draw.rect(pantalla, (255, 0, 0), enemigo) # Enemigo Rojo
    pygame.display.flip()
    reloj.tick(30)

pygame.quit()

# Guardar en CSV
df = pd.DataFrame(datos, columns=["distancia", "velocidad", "salto"])
df.to_csv("datos_juego.csv", index=False)
print("Datos guardados en 'datos_juego.csv'. ¡Listo para entrenar!")