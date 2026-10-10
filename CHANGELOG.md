# Actualización del agente y evidencia analítica

- Enrutamiento de preguntas que relacionan tiempo de ejecución y planificación óptima a la herramienta `relacion_tiempo`.
- Cálculo reproducible de número de tareas, porcentajes y rangos de ejecución por clase Target, más coincidencia y discordancias con el umbral de 5 segundos.
- Contexto Ollama restringido al dataset filtrado, con instrucciones explícitas para no confundir etiquetas con beneficios operativos demostrados.
- Resumen determinista de hechos calculados por Python, presentado separado de la interpretación generada por Ollama.
- Visualización ampliada en la pestaña de calidad de datos.
- Tratamiento de selecciones con una sola clase y casos con discrepancias de la regla.
- Documentación y ejemplos adaptados exclusivamente al dataset de 20 000 registros.
- 20 pruebas automáticas de regresión/EDA/agente y comprobación sobre el CSV completo.
