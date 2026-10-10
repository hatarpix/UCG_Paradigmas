# Resultados verificados sobre el dataset suministrado

Datos y métricas computadas en Python, sin inferencia del modelo de lenguaje. Son resultados del experimento de clasificación ilustrativo, no de un planificador cloud real.

- Archivo: `cloud_task_scheduling_dataset_20k.csv` (SHA-256: `6b5dbfc3f18367394998bc277fb14ca51be2f96e0d1456e916f2a94b55f61cc4`).
- Registros: **20,000**; columnas: **9**; valores nulos: **0**; filas duplicadas: **0**.
- Target=0: **11,111**; Target=1: **8,889**.
- Regla comprobada: `Target = 1` si `Execution_Time (s) < 5`, caso contrario 0. **0 discrepancias** en 20,000 filas.
- Correlación Pearson más marcada: `Execution_Time (s)` frente a `Target (Optimal Scheduling)`: **r = -0.8615**. No implica causalidad.

## Evaluación de modelos (split stratificado 80/20, seed=42)

| Modo | Clasificador | Baseline Accuracy | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| `pre_operacional` | `logistica` | 0.5555 | 0.5555 | 0.0000 | 0.0000 | 0.0000 | 0.5050 |
| `pre_operacional` | `bosque` | 0.5555 | 0.5425 | 0.4550 | 0.1479 | 0.2233 | 0.5000 |
| `retrospectivo` | `logistica` | 0.5555 | 0.9982 | 0.9961 | 1.0000 | 0.9980 | 1.0000 |
| `retrospectivo` | `bosque` | 0.5555 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

En el modo **preoperacional** se excluyen el identificador y el tiempo real de ejecución. El AUC se aproxima a 0.5; los recursos registrados no explican bien la etiqueta. En el modo **retrospectivo** se incluye `Execution_Time` y las métricas se acercan a 1.0 por la definición de Target. Estas dos condiciones evalúan **problemas distintos** y no se deben comparar como si ambas predijeran el rendimiento de un scheduler antes de ejecutar una tarea.

Las métricas podrían variar ligeramente según las versiones de scikit-learn. Software de esta verificación: Python 3.13.5, pandas 2.2.3, scikit-learn 1.8.0.

No se ha verificado el UI Streamlit ni la inferencia con un servidor Ollama real en el entorno de creación. Para la demostración final, arrancar el programa en la máquina Linux y grabar el flujo con el modelo disponible.

## Verificación de contexto del agente (versión actualizada)

- Pregunta probada: `¿Cuál es el tiempo promedio de ejecución y qué relación tiene con la planificación óptima?`.
- Herramientas seleccionadas automáticamente: `relacion_tiempo`, `estadisticas`, `correlaciones`, `clases`.
- Media del tiempo de ejecución: **5.5038 s**.
- Clase `Target=0`: **11 111 tareas**, tiempos **5.00–10.00 s**, media **7.5041 s**.
- Clase `Target=1`: **8 889 tareas**, tiempos **1.00–4.99 s**, media **3.0034 s**.
- Comparación con el umbral de 5 s: **20 000 observaciones, 0 discordancias, 100 % de coincidencia**.
- Se probó el filtrado por una VM y por `Target=1`: la evidencia utiliza solo las filas filtradas.
- Existe una síntesis determinista separada, generada por Python, y la respuesta de Ollama aparece como interpretación distinta.
- Las pruebas automáticas utilizan un cliente Ollama simulado; la interpretación real del modelo local debe verificarse en la interfaz.
