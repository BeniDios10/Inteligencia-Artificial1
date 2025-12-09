import pygame
import math
from queue import PriorityQueue

# Inicializar Pygame
pygame.init()

# Configuraciones iniciales
ANCHO_VENTANA = 800
VENTANA = pygame.display.set_mode((ANCHO_VENTANA, ANCHO_VENTANA))
pygame.display.set_caption("Visualización A* - Modo Dark Neon")

# --- NUEVA PALETA DE COLORES (ESTÉTICA) ---
# Fondo oscuro para contraste
FONDO = (28, 28, 30)          # Gris muy oscuro (casi negro)
LINEAS = (50, 50, 50)         # Gris sutil para la cuadrícula

# Elementos interactivos (Colores Neón/Pastel)
PARED = (200, 200, 200)       # Blanco/Gris claro para obstáculos
INICIO = (255, 107, 107)      # Rojo/Coral suave (Start)
FIN = (78, 205, 196)          # Turquesa/Cyan (End)

# Estados del algoritmo
CANDIDATO = (255, 230, 109)   # Amarillo (Open Set - Nodos por visitar)
VISITADO = (85, 98, 112)      # Azul grisáceo oscuro (Closed Set - Ya explorados)
CAMINO = (199, 244, 100)      # Verde Lima Brillante (El camino final)

# --- Clase Nodo ---
class Nodo:
    def __init__(self, fila, col, ancho, total_filas):
        self.fila = fila
        self.col = col
        self.x = col * ancho
        self.y = fila * ancho
        self.color = FONDO # Inicia con el color de fondo
        self.ancho = ancho
        self.total_filas = total_filas
        self.vecinos = []

    def get_pos(self):
        return self.fila, self.col

    def es_cerrado(self):
        return self.color == VISITADO

    def es_abierto(self):
        return self.color == CANDIDATO

    def es_pared(self):
        return self.color == PARED

    def es_inicio(self):
        return self.color == INICIO

    def es_fin(self):
        return self.color == FIN

    def restablecer(self):
        self.color = FONDO

    def hacer_inicio(self):
        self.color = INICIO

    def hacer_pared(self):
        self.color = PARED

    def hacer_fin(self):
        self.color = FIN

    def hacer_cerrado(self):
        self.color = VISITADO
    
    def hacer_abierto(self):
        self.color = CANDIDATO

    def hacer_camino(self):
        self.color = CAMINO

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

def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

def reconstruir_camino(padres, actual, dibujar):
    camino = []
    while actual in padres:
        actual = padres[actual]
        if not actual.es_inicio():
            camino.append(actual)
    camino.reverse()
    for nodo in camino:
        nodo.hacer_camino()
        dibujar()
        pygame.time.delay(30)

def a_asterisco(dibujar, grid, inicio, fin):
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
            reconstruir_camino(padres, fin, dibujar)
            inicio.hacer_inicio() 
            fin.hacer_fin()
            dibujar()
            return True

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
        # Pequeña optimización visual: reduce delay si el grid es muy grande
        pygame.time.delay(10) 

        if actual != inicio:
            actual.hacer_cerrado()
    
    return False

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
        # Usamos el color LINEAS definido arriba
        pygame.draw.line(ventana, LINEAS, (0, i * ancho_nodo), (ancho, i * ancho_nodo))
        for j in range(filas):
            pygame.draw.line(ventana, LINEAS, (j * ancho_nodo, 0), (j * ancho_nodo, ancho))

def dibujar(ventana, grid, filas, ancho):
    ventana.fill(FONDO) # Llenamos con el color de fondo oscuro
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
    FILAS = 10 # Aumenté ligeramente las filas para que se vea mejor el algoritmo
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

            if pygame.mouse.get_pressed()[0]: # Izquierdo
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
                    nodo.hacer_pared()

            elif pygame.mouse.get_pressed()[2]: # Derecho
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                nodo = grid[fila][col]
                nodo.restablecer()
                if nodo == inicio:
                    inicio = None
                elif nodo == fin:
                    fin = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and inicio and fin:
                    algoritmo_iniciado = True
                    for fila in grid:
                        for nodo in fila:
                            nodo.actualizar_vecinos(grid)

                    a_asterisco(lambda: dibujar(ventana, grid, FILAS, ancho), grid, inicio, fin)
                    # Nota: El algoritmo termina y se queda pintado, el usuario debe resetear manualmente con 'R'
                
                if event.key == pygame.K_e or event.key == pygame.K_r:
                    inicio = None
                    fin = None
                    grid = crear_grid(FILAS, ancho)
                    algoritmo_iniciado = False

    pygame.quit()

if __name__ == '__main__':
    main(VENTANA, ANCHO_VENTANA)