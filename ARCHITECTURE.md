# Arquitectura del proyecto

`app.py` es exclusivamente la capa de presentación Streamlit. Todos los cálculos analíticos se efectúan en `src/eda.py` o `src/ml.py`; los gráficos se generan en `src/visualizations.py`; Ollama se invoca desde `src/llm_client.py` y el agente de `src/agent.py` construye los datos que se enviarán como evidencia.

```text
CSV incluido o subida del usuario
       |
       v
src.data_loader -> data.frame verificado -> filtros Streamlit
       |                                       |
       +--> src.eda (perfil, correlaciones) ----+--> Plotly
       |
       +--> src.ml (Pipeline scikit-learn, test set, métricas)
       |
       +--> src.agent (herramientas autorizadas en Python)
                    -> contexto compacto JSON -> src.llm_client -> Ollama localhost
                                                                  -> respuesta
```

Multiparadigma:

- **Imperativo:** validaciones, filtros, flujo de interfaz y decisiones `if`.
- **Funcional / modular:** `perfil_calidad`, `resumen_numerico`, `resumen_por_grupo`, `analizar_relacion_target_tiempo`, `matriz_correlaciones`, etc., sin dependencias de Streamlit.
- **POO:** `AgenteDatos`, `ClienteOllama`, `Pipeline`, `LogisticRegression`, `RandomForestClassifier`.
- **Factory / Strategy:** `preparar_pipeline(modo, algoritmo)` selecciona objetos ML; distintas estrategias comparten `fit/predict`.
- **Facade:** `AgenteDatos` coordina selección de herramientas, construcción de contexto y consulta de Ollama.

No se concede a Ollama acceso a ejecución de Python o SQL. El enrutamiento a herramientas usa reglas explícitas y permitidas, sin interpretación de código dinámico. El contexto incluye `registros_filtrados` y resultados de cálculos para garantizar trazabilidad.

El enrutamiento automático identifica preguntas sobre el tiempo y la etiqueta de planificación y activa `relacion_tiempo`: la función Python devuelve distribución de clases, mínimos/máximos por clase, umbral contrastado, coincidencias y discordancias. Ollama no deduce esas cifras del texto de la pregunta. El análisis permanece condicionado a los filtros actuales.

Para consultas de tiempo y planificación, Streamlit también muestra `sintetizar_evidencia_tiempo()`: un resumen determinista calculado en Python, etiquetado independientemente de la interpretación generada por Ollama. No hay datos agregados por el LLM.
