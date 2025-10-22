import pygame
import math
from queue import PriorityQueue

# Inicializar Pygame antes de usar sus funciones de tiempo o visualización
pygame.init()

# Configuraciones iniciales
ANCHO_VENTANA = 800
VENTANA = pygame.display.set_mode((ANCHO_VENTANA, ANCHO_VENTANA))
pygame.display.set_caption("Visualización de Algoritmo A. PROYECTO UNIDAD 1*")

# Colores (RGB)
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
GRIS = (128, 128, 128)
VERDE = (0, 255, 0)      # Abierto (Candidato a explorar)
ROJO = (255, 0, 0)       # Cerrado (Ya explorado)
AZUL = (0, 0, 255)       # Camino (Ruta final)
NARANJA = (255, 165, 0)  # Inicio
PURPURA = (128, 0, 128)  # Fin

# --- Clase Nodo ---
class Nodo:
    def __init__(self, fila, col, ancho, total_filas):
        self.fila = fila
        self.col = col
        # CORRECCIÓN: X es la columna (horizontal), Y es la fila (vertical)
        self.x = col * ancho
        self.y = fila * ancho
        self.color = BLANCO
        self.ancho = ancho
        self.total_filas = total_filas
        self.vecinos = []

    def get_pos(self):
        return self.fila, self.col

    def es_cerrado(self):
        return self.color == ROJO

    def es_abierto(self):
        return self.color == VERDE

    def es_pared(self):
        return self.color == NEGRO

    def es_inicio(self):
        return self.color == NARANJA

    def es_fin(self):
        return self.color == PURPURA

    def restablecer(self):
        self.color = BLANCO

    def hacer_inicio(self):
        self.color = NARANJA

    def hacer_pared(self):
        self.color = NEGRO

    def hacer_fin(self):
        self.color = PURPURA

    def hacer_cerrado(self):
        self.color = ROJO
    
    def hacer_abierto(self):
        self.color = VERDE

    def hacer_camino(self):
        self.color = AZUL

    def dibujar(self, ventana):
        pygame.draw.rect(ventana, self.color, (self.x, self.y, self.ancho, self.ancho))

    def actualizar_vecinos(self, grid):
        self.vecinos = []
        filas = self.total_filas
        
        # Vecinos (Arriba, Abajo, Izquierda, Derecha)
        if self.fila < filas - 1 and not grid[self.fila + 1][self.col].es_pared():
            self.vecinos.append(grid[self.fila + 1][self.col])
        if self.fila > 0 and not grid[self.fila - 1][self.col].es_pared():
            self.vecinos.append(grid[self.fila - 1][self.col])
        if self.col < filas - 1 and not grid[self.fila][self.col + 1].es_pared():
            self.vecinos.append(grid[self.fila][self.col + 1])
        if self.col > 0 and not grid[self.fila][self.col - 1].es_pared():
            self.vecinos.append(grid[self.fila][self.col - 1])

    def __lt__(self, other):
        return False

# --------------------------------------------------------------------------------------
# --- Funciones del Algoritmo A* ---

# Heurística (Distancia de Manhattan)
def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

# Reconstruir el camino (MODIFICADA para pintar del INICIO al FIN)
def reconstruir_camino(padres, actual, dibujar):
    camino = []
    
    # 1. Rastrear el camino del FIN al INICIO
    while actual in padres:
        actual = padres[actual]
        if not actual.es_inicio():
            camino.append(actual)

    # 2. la lista para ir del INICIO al FIN
    camino.reverse()

    # 3. Dibujar el camino en orden
    for nodo in camino:
        nodo.hacer_camino()
        dibujar()
        pygame.time.delay(30) # Retardo para visualizar el recorrido

# Algoritmo A*
def algoritmo_a_star(dibujar, grid, inicio, fin):
    contador = 0
    open_set = PriorityQueue()
    open_set.put((0, contador, inicio))
    
    padres = {} 

    g_score = {nodo: float("inf") for fila in grid for nodo in fila}
    g_score[inicio] = 0

    f_score = {nodo: float("inf") for fila in grid for nodo in fila}
    f_score[inicio] = h(inicio.get_pos(), fin.get_pos())

    open_set_hash = {inicio}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        actual = open_set.get()[2]
        open_set_hash.remove(actual)

        if actual == fin:
            # Llama a la función de reconstrucción (ahora pinta de Inicio a Fin)
            reconstruir_camino(padres, fin, dibujar)
            
            # Asegura que los puntos extremos se repinten al final (por si el camino los cubrió)
            inicio.hacer_inicio() 
            fin.hacer_fin()
            dibujar() # Último redibujo
            return True # Éxito

        for vecino in actual.vecinos:
            g_score_temporal = g_score[actual] + 1 

            if g_score_temporal < g_score[vecino]:
                padres[vecino] = actual
                g_score[vecino] = g_score_temporal
                f_score[vecino] = g_score_temporal + h(vecino.get_pos(), fin.get_pos())
                
                if vecino not in open_set_hash:
                    contador += 1
                    open_set.put((f_score[vecino], contador, vecino))
                    open_set_hash.add(vecino)
                    if not vecino.es_fin():
                         vecino.hacer_abierto()

        dibujar()
        pygame.time.delay(10) # Retardo de 10ms en cada paso de exploración

        if actual != inicio:
            actual.hacer_cerrado()
    
    return False # Fallo, el camino no existe

# --------------------------------------------------------------------------------------
# --- Funciones de Pygame ---

def crear_grid(filas, ancho):
    grid = []
    ancho_nodo = ancho // filas
    for i in range(filas):
        grid.append([])
        for j in range(filas):
            nodo = Nodo(i, j, ancho_nodo, filas)
            grid[i].append(nodo)
    return grid

def dibujar_grid(ventana, filas, ancho):
    ancho_nodo = ancho // filas
    for i in range(filas):
        pygame.draw.line(ventana, GRIS, (0, i * ancho_nodo), (ancho, i * ancho_nodo))
        for j in range(filas):
            pygame.draw.line(ventana, GRIS, (j * ancho_nodo, 0), (j * ancho_nodo, ancho))

def dibujar(ventana, grid, filas, ancho):
    ventana.fill(BLANCO)
    for fila in grid:
        for nodo in fila:
            nodo.dibujar(ventana)

    dibujar_grid(ventana, filas, ancho)
    pygame.display.update()

def obtener_click_pos(pos, filas, ancho):
    ancho_nodo = ancho // filas
    x, y = pos 
    fila = y // ancho_nodo 
    col = x // ancho_nodo
    return fila, col 

def main(ventana, ancho):
#PARA CAMBIAR TAMAÑO
    FILAS = 10
    grid = crear_grid(FILAS, ancho)

    inicio = None
    fin = None

    corriendo = True
    algoritmo_iniciado = False

    while corriendo:
        dibujar(ventana, grid, FILAS, ancho)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False

            if algoritmo_iniciado:
                continue

            # Click izquierdo
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                nodo = grid[fila][col]
                
                if not inicio and nodo != fin:
                    inicio = nodo
                    inicio.hacer_inicio()
                elif not fin and nodo != inicio:
                    fin = nodo
                    fin.hacer_fin()
                elif nodo != fin and nodo != inicio:
                    if not nodo.es_pared(): 
                        nodo.hacer_pared()

            # Click derecho (Restablecer)
            elif pygame.mouse.get_pressed()[2]: 
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                nodo = grid[fila][col]
                nodo.restablecer()
                if nodo == inicio:
                    inicio = None
                elif nodo == fin:
                    fin = None

            # Eventos de Teclado
            if event.type == pygame.KEYDOWN:
                # Iniciar el Algoritmo A*
                if event.key == pygame.K_SPACE and inicio and fin:
                    algoritmo_iniciado = True
                    for fila in grid:
                        for nodo in fila:
                            nodo.actualizar_vecinos(grid)

                    algoritmo_a_star(lambda: dibujar(ventana, grid, FILAS, ancho), grid, inicio, fin)
                    algoritmo_iniciado = False
                
                # Restablecer el Tablero (Tecla 'C' o 'R')
                if event.key == pygame.K_c or event.key == pygame.K_r:
                    inicio = None
                    fin = None
                    grid = crear_grid(FILAS, ancho)
                    algoritmo_iniciado = False

    pygame.quit()

if __name__ == '__main__':
    main(VENTANA, ANCHO_VENTANA)