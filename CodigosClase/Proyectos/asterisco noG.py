import pygame
import math
from queue import PriorityQueue

# Inicializar Pygame
pygame.init()

# Configuraciones iniciales
ANCHO_VENTANA = 800
VENTANA = pygame.display.set_mode((ANCHO_VENTANA, ANCHO_VENTANA))
pygame.display.set_caption("Visualización de Algoritmo A*")

# Colores (RGB)
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
GRIS = (128, 128, 128)
VERDE = (0, 255, 0)       # Open Set (Candidatos)
ROJO = (255, 0, 0)        # Closed Set (Visitados/Lista Cerrada)
NARANJA = (255, 165, 0)   # Inicio
PURPURA = (128, 0, 128)   # Fin
AZUL = (0, 0, 255)        # Camino Óptimo

class Nodo:
    def __init__(self, fila, col, ancho, total_filas):
        self.fila = fila
        self.col = col
        # CORRECCIÓN: x es columna (horizontal), y es fila (vertical)
        self.x = col * ancho
        self.y = fila * ancho
        self.color = BLANCO
        self.ancho = ancho
        self.total_filas = total_filas
        self.vecinos = []

    def get_pos(self):
        return self.fila, self.col

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
        # Revisar vecino de ABAJO
        if self.fila < self.total_filas - 1 and not grid[self.fila + 1][self.col].es_pared():
            self.vecinos.append(grid[self.fila + 1][self.col])
        # Revisar vecino de ARRIBA
        if self.fila > 0 and not grid[self.fila - 1][self.col].es_pared():
            self.vecinos.append(grid[self.fila - 1][self.col])
        # Revisar vecino de DERECHA
        if self.col < self.total_filas - 1 and not grid[self.fila][self.col + 1].es_pared():
            self.vecinos.append(grid[self.fila][self.col + 1])
        # Revisar vecino de IZQUIERDA
        if self.col > 0 and not grid[self.fila][self.col - 1].es_pared():
            self.vecinos.append(grid[self.fila][self.col - 1])

    # Método necesario para que PriorityQueue pueda comparar nodos si los f_score son iguales
    def __lt__(self, other):
        return False

# --- FUNCIONES DEL ALGORITMO ---

def h(p1, p2):
    """Heurística: Distancia Manhattan"""
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

def reconstruir_camino(padres, actual, dibujar):
    """Reconstruye el camino desde el fin hasta el inicio"""
    while actual in padres:
        actual = padres[actual]
        if not actual.es_inicio():
             actual.hacer_camino()
        dibujar()

def algoritmo_a_star(dibujar, grid, inicio, fin):
    """Ejecuta el algoritmo A*"""
    contador = 0
    open_set = PriorityQueue()
    open_set.put((0, contador, inicio))
    
    padres = {} # Para reconstruir el camino
    
    # Costo g: distancia desde el inicio hasta el nodo n
    g_score = {nodo: float("inf") for fila in grid for nodo in fila}
    g_score[inicio] = 0
    
    # Costo f: g_score + h (heurística)
    f_score = {nodo: float("inf") for fila in grid for nodo in fila}
    f_score[inicio] = h(inicio.get_pos(), fin.get_pos())

    open_set_hash = {inicio} # Para verificar existencia rápidamente
    
    lista_cerrada = [] # REQUISITO: Almacenar la lista cerrada

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # Sacar el nodo con el f_score más bajo
        actual = open_set.get()[2]
        open_set_hash.remove(actual)

        if actual == fin:
            reconstruir_camino(padres, fin, dibujar)
            fin.hacer_fin()
            inicio.hacer_inicio()
            
            # --- IMPRIMIR REQUISITO ---
            print("\n--- RUTA ENCONTRADA ---")
            print(f"Nodos en Lista Cerrada (Total: {len(lista_cerrada)}):")
            print([n.get_pos() for n in lista_cerrada])
            return True

        for vecino in actual.vecinos:
            # En grid simple, el peso de mover a un vecino es siempre 1
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
                        vecino.hacer_abierto() # Visualizar candidato (Verde)

        dibujar()

        if actual != inicio:
            actual.hacer_cerrado() # Visualizar lista cerrada (Rojo)
            lista_cerrada.append(actual)

    print("No se encontró solución.")
    return False

# --- FUNCIONES DE GRID ---

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
    x, y = pos # Pygame devuelve (x, y)
    fila = y // ancho_nodo
    col = x // ancho_nodo
    return fila, col

def main(ventana, ancho):
    FILAS = 10 # MODIFICAR PARA HACER MÁS GRANDE O PEQUEÑO
    grid = crear_grid(FILAS, ancho)

    inicio = None
    fin = None

    corriendo = True

    while corriendo:
        dibujar(ventana, grid, FILAS, ancho)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False

            # Click Izquierdo: Poner Inicio, Fin y Paredes
            if pygame.mouse.get_pressed()[0]: 
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                
                # Evitar errores si se hace click fuera del rango
                if fila < FILAS and col < FILAS:
                    nodo = grid[fila][col]
                    if not inicio and nodo != fin:
                        inicio = nodo
                        inicio.hacer_inicio()
                    elif not fin and nodo != inicio:
                        fin = nodo
                        fin.hacer_fin()
                    elif nodo != fin and nodo != inicio:
                        nodo.hacer_pared()

            # Click Derecho: Borrar
            elif pygame.mouse.get_pressed()[2]: 
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                if fila < FILAS and col < FILAS:
                    nodo = grid[fila][col]
                    nodo.restablecer()
                    if nodo == inicio:
                        inicio = None
                    elif nodo == fin:
                        fin = None
            
            # Tecla ESPACIO: Iniciar Algoritmo
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and inicio and fin:
                    # Actualizar vecinos antes de correr
                    for fila in grid:
                        for nodo in fila:
                            nodo.actualizar_vecinos(grid)
                    
                    # Llamar al algoritmo A*
                    algoritmo_a_star(lambda: dibujar(ventana, grid, FILAS, ancho), grid, inicio, fin)

                # Tecla E: Limpiar tablero
                if event.key == pygame.K_e:
                    inicio = None
                    fin = None
                    grid = crear_grid(FILAS, ancho)

    pygame.quit()

if __name__ == "__main__":
    main(VENTANA, ANCHO_VENTANA)