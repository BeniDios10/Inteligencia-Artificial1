import pygame
import math
from queue import PriorityQueue

# --- CONFIGURACIÓN DE COLORES ---
ROJO = (255, 0, 0)         # Nodo Visitado (Cerrado - Ya analizado)
VERDE = (0, 255, 0)        # Nodo en Espera (Abierto - Por analizar)
AZUL = (0, 0, 255)         
AMARILLO = (255, 255, 0)   
BLANCO = (255, 255, 255)   # Fondo (Camino libre)
NEGRO = (0, 0, 0)          # Pared (Obstáculo)
MORADO = (128, 0, 128)     # Camino Final (La ruta óptima)
NARANJA = (255, 165, 0)    # Punto de Inicio
TURQUESA = (64, 224, 208)  # Punto Final (Destino)
GRIS = (128, 128, 128)     # Líneas de la cuadrícula

# Configuración de la ventana
ANCHO_VENTANA = 800
VENTANA = pygame.display.set_mode((ANCHO_VENTANA, ANCHO_VENTANA))
pygame.display.set_caption("Algoritmo A* - Visualización")

class Nodo:
    def __init__(self, fila, col, ancho, total_filas):
        self.fila = fila
        self.col = col
        self.x = fila * ancho
        self.y = col * ancho
        self.color = BLANCO
        self.vecinos = []
        self.ancho = ancho
        self.total_filas = total_filas

    def obtener_pos(self):
        return self.fila, self.col

    # --- ESTADOS DEL NODO (Preguntas) ---
    def es_cerrado(self):
        return self.color == ROJO

    def es_abierto(self):
        return self.color == VERDE

    def es_pared(self):
        return self.color == NEGRO

    def es_inicio(self):
        return self.color == NARANJA

    def es_fin(self):
        return self.color == TURQUESA

    def resetear(self):
        self.color = BLANCO

    # --- ACCIONES VISUALES (Pintar) ---
    def hacer_inicio(self):
        self.color = NARANJA

    def hacer_cerrado(self):
        # Pinta de ROJO lo que ya analizamos y descartamos
        self.color = ROJO

    def hacer_abierto(self):
        # Pinta de VERDE lo que está en cola para ser revisado
        self.color = VERDE

    def hacer_pared(self):
        self.color = NEGRO

    def hacer_fin(self):
        self.color = TURQUESA

    def hacer_camino(self):
        self.color = MORADO

    def dibujar(self, ventana):
        pygame.draw.rect(ventana, self.color, (self.x, self.y, self.ancho, self.ancho))

    def actualizar_vecinos(self, cuadricula):
        self.vecinos = []
        # Direcciones posibles (Fila, Columna)
        # Incluye las 8 direcciones (Arriba, Abajo, Izq, Der y las 4 Diagonales)
        direcciones = [
            (1, 0), (-1, 0), (0, 1), (0, -1),   # Cardinales
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonales
        ]

        for dx, dy in direcciones:
            nueva_fila = self.fila + dx
            nueva_col = self.col + dy

            # Verificar que el vecino esté dentro del tablero
            if 0 <= nueva_fila < self.total_filas and 0 <= nueva_col < self.total_filas:
                # Verificar que no sea una pared
                if not cuadricula[nueva_fila][nueva_col].es_pared():
                    self.vecinos.append(cuadricula[nueva_fila][nueva_col])

    def __lt__(self, otro):
        return False

# --- HEURÍSTICA (Distancia Manhattan) ---
def heuristica(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

def reconstruir_camino(vino_de, actual, dibujar_func):
    while actual in vino_de:
        actual = vino_de[actual]
        actual.hacer_camino()
        dibujar_func()

# --- ALGORITMO PRINCIPAL A* ---
def algoritmo(dibujar_func, cuadricula, inicio, fin):
    contador = 0
    open_set = PriorityQueue() # Cola de prioridad para elegir siempre el mejor nodo
    open_set.put((0, contador, inicio))
    vino_de = {} # Para rastrear el camino de vuelta
    
    # g_score: Costo real desde el inicio hasta el nodo actual
    g_score = {nodo: float("inf") for fila in cuadricula for nodo in fila}
    g_score[inicio] = 0
    
    # f_score: g_score + heurística (Costo total estimado)
    f_score = {nodo: float("inf") for fila in cuadricula for nodo in fila}
    f_score[inicio] = heuristica(inicio.obtener_pos(), fin.obtener_pos())

    open_set_hash = {inicio}

    while not open_set.empty():
        # Permitir cerrar la ventana mientras calcula
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2] # Sacar el nodo con menor f_score
        open_set_hash.remove(current)

        # Si llegamos al final
        if current == fin:
            reconstruir_camino(vino_de, fin, dibujar_func)
            fin.hacer_fin()
            inicio.hacer_inicio()
            return True

        for vecino in current.vecinos:
            # --- CÁLCULO DE COSTO DE MOVIMIENTO ---
            # Si cambia tanto la fila como la columna, es un movimiento DIAGONAL.
            # Le damos costo 1.4 (aprox raíz de 2) para que sea realista.
            # Si es recto, el costo es 1.0.
            if current.fila != vecino.fila and current.col != vecino.col:
                costo_movimiento = 1.4
            else:
                costo_movimiento = 1.0
            
            temp_g_score = g_score[current] + costo_movimiento

            # Si encontramos un camino mejor hacia este vecino
            if temp_g_score < g_score[vecino]:
                vino_de[vecino] = current
                g_score[vecino] = temp_g_score
                f_score[vecino] = temp_g_score + heuristica(vecino.obtener_pos(), fin.obtener_pos())
                
                if vecino not in open_set_hash:
                    contador += 1
                    open_set.put((f_score[vecino], contador, vecino))
                    open_set_hash.add(vecino)
                    vecino.hacer_abierto() # Pintar VERDE (En espera)

        dibujar_func() # Actualizar la pantalla

        # Si el nodo actual no es el inicio, márcalo como CERRADO (ROJO)
        # Esto significa "Ya lo revisé y no es el destino".
        if current != inicio:
            current.hacer_cerrado()

    return False

# --- FUNCIONES DE DIBUJO Y GRID ---
def crear_cuadricula(filas, ancho):
    cuadricula = []
    ancho_nodo = ancho // filas
    for i in range(filas):
        cuadricula.append([])
        for j in range(filas):
            nodo = Nodo(i, j, ancho_nodo, filas)
            cuadricula[i].append(nodo)
    return cuadricula

def dibujar_lineas(ventana, filas, ancho):
    ancho_nodo = ancho // filas
    for i in range(filas):
        pygame.draw.line(ventana, GRIS, (0, i * ancho_nodo), (ancho, i * ancho_nodo))
        for j in range(filas):
            pygame.draw.line(ventana, GRIS, (j * ancho_nodo, 0), (j * ancho_nodo, ancho))

def dibujar_todo(ventana, cuadricula, filas, ancho):
    ventana.fill(BLANCO)
    for fila in cuadricula:
        for nodo in fila:
            nodo.dibujar(ventana)
    dibujar_lineas(ventana, filas, ancho)
    pygame.display.update()

def obtener_pos_clic(pos, filas, ancho):
    ancho_nodo = ancho // filas
    y, x = pos
    fila = y // ancho_nodo
    col = x // ancho_nodo
    return fila, col

# --- FUNCIÓN PRINCIPAL (MAIN) ---
def main(ventana, ancho):
    FILAS = 11 # MODIFICAR PARA HACER MÁS GRANDE O PEQUEÑO
    cuadricula = crear_cuadricula(FILAS, ancho)

    inicio = None
    fin = None
    corriendo = True

    while corriendo:
        dibujar_todo(ventana, cuadricula, FILAS, ancho)
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                corriendo = False

            # CLICK IZQUIERDO: Colocar Inicio, Fin y Paredes
            if pygame.mouse.get_pressed()[0]: 
                pos = pygame.mouse.get_pos()
                fila, col = obtener_pos_clic(pos, FILAS, ancho)
                nodo = cuadricula[fila][col]
                if not inicio and nodo != fin:
                    inicio = nodo
                    inicio.hacer_inicio()
                elif not fin and nodo != inicio:
                    fin = nodo
                    fin.hacer_fin()
                elif nodo != fin and nodo != inicio:
                    nodo.hacer_pared()

            # CLICK DERECHO: Borrar
            elif pygame.mouse.get_pressed()[2]: 
                pos = pygame.mouse.get_pos()
                fila, col = obtener_pos_clic(pos, FILAS, ancho)
                nodo = cuadricula[fila][col]
                nodo.resetear()
                if nodo == inicio:
                    inicio = None
                elif nodo == fin:
                    fin = None

            # TECLA ESPACIO: Iniciar Algoritmo
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_SPACE and inicio and fin:
                    for fila in cuadricula:
                        for nodo in fila:
                            nodo.actualizar_vecinos(cuadricula)
                    
                    algoritmo(lambda: dibujar_todo(ventana, cuadricula, FILAS, ancho), cuadricula, inicio, fin)

                # TECLA E: Limpiar Todo
                if evento.key == pygame.K_e:
                    inicio = None
                    fin = None
                    cuadricula = crear_cuadricula(FILAS, ancho)

    pygame.quit()

if __name__ == "__main__":
    main(VENTANA, ANCHO_VENTANA)