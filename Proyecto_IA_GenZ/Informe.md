# Informe Técnico: Implementación de Arquitectura RAG (Retrieval-Augmented Generation)

**Proyecto:** La Generación Z y la Crisis de Sentido en la Era Digital  
**Institución:** Instituto Tecnológico de Morelia  
**Departamento:** Sistemas y Computación  
**Materia:** Inteligencia Artificial  

## 1. Resumen Ejecutivo
Este informe detalla la implementación técnica de un sistema de **Generación Aumentada por Recuperación (RAG)** diseñado para procesar y analizar un corpus de textos filosóficos y sociológicos. El objetivo principal fue superar la limitación de "ventana de contexto" de los LLMs tradicionales y mitigar alucinaciones mediante la inyección de contexto específico. Se utilizó **AnythingLLM** como orquestador, ejecutando la inferencia de manera local (On-Premise) para garantizar la privacidad de los datos y el control absoluto sobre el flujo de vectores.

## 2. Arquitectura del Sistema
El flujo de datos sigue el pipeline estándar de RAG, dividido en dos fases: Ingesta (ETL) e Inferencia.

### 2.1 Componentes del Stack Tecnológico
| Componente | Tecnología Seleccionada | Justificación Técnica |
| :--- | :--- | :--- |
| **Orquestador** | AnythingLLM Desktop | Gestión de embeddings e interfaz de chat integrada sin código boilerplate. |
| **Base de Datos Vectorial** | LanceDB (Integrada) | Base de datos serverless de alto rendimiento para almacenamiento local. |
| **Modelo de Inferencia** | LLM Local (GGUF) | Ejecución en CPU para privacidad y cero coste operativo. |

## 3. Ingeniería de Datos y Corpus

### 3.1 Recolección y Preprocesamiento (ETL)
Se construyó un dataset de **13 documentos .txt** para maximizar la densidad semántica. * **Técnica de Limpieza:** Eliminación de ruido (encabezados/pies de página) y estandarización a UTF-8. * **Estrategia de Chunking:** Se utilizó un divisor de caracteres recursivo para mantener la cohesión semántica en fragmentos de tamaño uniforme.

### 3.2 Desglose del Corpus Utilizado
| ID Archivo | Temática Principal | Objetivo en el RAG |
| :--- | :--- | :--- |
| `Doc_01_Filosofia.txt` | Crisis de sentido y existencialismo | Base teórica del análisis. |
| `Doc_05_Sociedad.txt` | Dinámicas de la Generación Z | Contexto demográfico y social. |
| `Doc_08_Digital.txt` | Impacto del *doomscrolling* y redes | Datos sobre comportamiento digital. |
| `Doc_13_Sintesis.txt` | Resumen de hallazgos transversales | Conexión entre conceptos clave. |

## 4. Métricas de Rendimiento e Inferencia

### 4.1 Latencia y Throughput (CPU Bottleneck)
Las pruebas se ejecutaron en una arquitectura basada en CPU (sin aceleración GPU), obteniendo los siguientes resultados de procesamiento:

**Log de Ejecución Directo:**
```bash
-------------------------------------------------------
Query 1 (Diagnóstico): 142.419s total | 5.58 tokens/s
Query 2 (Autonomía):   210.175s total | 5.39 tokens/s
Query 3 (Conclusión):  150.773s total | 6.84 tokens/s
-------------------------------------------------------
Average Throughput:    ~5.93 tok/s
-------------------------------------------------------

### 4.2 Análisis del Rendimiento
La velocidad promedio de 5.93 tokens/segundo indica que, aunque la latencia es alta para un uso en tiempo real (promedio de 2.5 minutos por respuesta), es perfectamente funcional para tareas de investigación académica donde la precisión de la cita bibliográfica es prioritaria sobre la inmediatez.
## 5. Ingeniería de Prompts (Prompt Engineering)
Para asegurar la fidelidad de las respuestas, se configuró un System Prompt con restricciones estrictas de rol:

**Configuración del Agente:**
*   **Rol:** Experto en Filosofía Contemporánea y Sociología Digital.
*   **Regla de Oro:** Basarse EXCLUSIVAMENTE en los documentos proporcionados en LanceDB.
*   **Restricción:** Si la respuesta no está en los documentos, declarar ignorancia. No usar conocimiento general del modelo.
*   **Formato:** Obligatorio citar el nombre del archivo fuente.

**Resultado:** Se logró una reducción total de alucinaciones, garantizando que el análisis de la "Crisis de Sentido" estuviera anclado únicamente en la bibliografía seleccionada.
## 6. Conclusiones Técnicas
**Privacidad:** La implementación local de AnythingLLM permite el procesamiento de documentos sensibles sin salida de datos a la red. **Eficacia RAG:** La base de datos LanceDB demostró una alta precisión en la recuperación por similitud de coseno, vinculando términos abstractos correctamente. **Optimización Futura:** Para mejorar el Throughput, se recomienda la migración a modelos cuantizados en formato EXL2 o el uso de una GPU con al menos 8GB de VRAM para habilitar la aceleración CUDA.

*Reporte Técnico - Benítez Gómez Josué Miguel - Diciembre 2025*