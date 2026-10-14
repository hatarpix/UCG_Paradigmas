# Cloud Task Scheduling Analytics

Proyecto Final Integrador — Paradigmas de Programación para Inteligencia Artificial y Análisis de Datos, Universidad Casa Grande, C2026 P1. Estudiante: Valerii Voropai.

Aplicación académica **local** de Streamlit para explorar el *Cloud Task Scheduling Dataset*, verificar calidad de datos, comparar recursos y evaluar clasificadores, con un agente de interpretación que utiliza el modelo Ollama `llama3.2:3b`. Su arquitectura modular separa el análisis objetivo, la interfaz y la interpretación del modelo de lenguaje.

## Problema y dataset

El archivo incluido `data/cloud_task_scheduling_dataset_20k.csv` contiene **20 000 registros, 9 variables**. Referencia pública: [Cloud Task Scheduling Dataset (Kaggle)](https://www.kaggle.com/datasets/ziya07/cloud-task-scheduling-dataset). La fuente analizada es el CSV de **20 000 registros**, con esquema de 9 variables; su contenido y métricas se verifican al ejecutar la aplicación.

Las variables son `Task_ID`, `CPU_Usage (%)`, `RAM_Usage (MB)`, `Disk_IO (MB/s)`, `Network_IO (MB/s)`, `Priority`, `VM_ID`, `Execution_Time (s)` y `Target (Optimal Scheduling)`. El target tiene clases 0 y 1.

**Hallazgo crítico:** En el CSV de 20k filas, `Target = 1` corresponde exactamente a `Execution_Time (s) < 5` y `Target = 0` a tiempo >= 5. Se comprobó sobre los 20 000 registros (0 excepciones). Esto NO demuestra que el dataset capture un algoritmo de scheduling óptimo real; puede ser una etiqueta sintética derivada del tiempo observado. Usar `Execution_Time` en un modelo que supuestamente predice *antes de ejecutar* es fuga temporal de información. La aplicación ofrece un modo preoperacional que excluye ese atributo y otro modo retrospectivo ilustrativo. La evaluación compara contra un baseline de clase mayoritaria.

## Estructura

```text
ProyectoFinal_Cloud_Task_Scheduling/
├── app.py                     Streamlit: UI, filtros y vistas
├── data/
│   └── cloud_task_scheduling_dataset_20k.csv
├── src/
│   ├── data_loader.py         Lectura de CSV, validación, filtros
│   ├── eda.py                 Análisis, nulos, segmentación y correlaciones
│   ├── visualizations.py      Gráficos interactivos Plotly
│   ├── ml.py                  Clasificación sklearn y comparación baseline
│   ├── llm_client.py          Cliente HTTP Ollama y errores controlados
│   └── agent.py               Agente, selección de herramientas y contexto JSON
├── requirements.txt
├── run_linux.sh
└── README.md
```

## Instalación en Linux (Ubuntu/Debian)

Requiere Python 3.10+ (recomendado 3.11/3.12), Internet para instalar paquetes y descargar Ollama por primera vez. Usar `python3`, no el ejecutable `python` dependiente de la distribución.

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip curl
cd ProyectoFinal_Cloud_Task_Scheduling
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Ollama (instalar siguiendo instrucciones oficiales y revisar el script de instalación antes de ejecutarlo):

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b
ollama list
systemctl status ollama
```

Si el servicio no está activo, inicie Ollama según su instalación (`sudo systemctl start ollama` cuando se usa systemd o `ollama serve` en otra terminal). Compruebe el API:

```bash
curl -s http://127.0.0.1:11434/api/tags
```

Ejecutar en terminal con `.venv` activado:

```bash
python3 -m streamlit run app.py --server.address 127.0.0.1
```

Abrir `http://127.0.0.1:8501`. Alternativamente, `./run_linux.sh` comprueba y utiliza `.venv/bin/python3`.

Para equipos con poca RAM, puede descargarse `ollama pull llama3.2:1b` y ejecutar con `OLLAMA_MODEL=llama3.2:1b python3 -m streamlit run app.py`. Para Ollama en otra dirección (uso de laboratorio), configurar `OLLAMA_BASE_URL` antes del arranque.

**Funcionamiento sin Ollama:** las pestañas de calidad, exploración y ML continúan operativas; la pestaña del agente muestra un error explícito, no una respuesta ficticia.

## Ejemplo de uso / Demonstración

1. Abra la aplicación. Por defecto se carga el CSV de 20k registros incluido.
2. Revise las dimensiones, los nulos y la regla temporal. Aplique filtros por `VM_ID`, `Priority` y `Target`.
3. En Exploración, revise histogramas, segmentación y correlaciones; descargue los datos filtrados.
4. En Machine Learning, ejecute `pre_operacional` y compare su `accuracy` con el baseline; después pruebe `retrospectivo`, leyendo la advertencia sobre fuga temporal.
5. En Agente IA, pulse **Comprobar Ollama**. Pregunte por ejemplo:
   - `¿Cuál es el tiempo medio de ejecución?`
   - `¿Qué VM tiene más tareas y cuál es su CPU media?`
   - `¿Cuál es el tiempo promedio de ejecución y qué relación tiene con la planificación óptima?`
   - `¿Existe una relación entre Execution_Time y Target?`
   - `¿Cuáles son las métricas F1 y ROC AUC del último modelo?` (tras ejecutar ML)
6. Abra el desplegable de evidencia JSON y contraste los valores utilizados por Ollama con las tablas calculadas por Python.

**Regla del agente:** herramientas en lista blanca, cálculos en pandas/sklearn, envío de un contexto JSON compacto con el alcance y los resultados; ninguna ejecución de código propuesto por el LLM. La herramienta `relacion_tiempo` cuantifica los rangos y el número de tareas por clase de Target, y compara los datos contra el umbral de 5 segundos, siempre sobre los registros filtrados. Para consultas de esta relación, el panel muestra una síntesis objetiva generada por Python por separado de la interpretación del LLM. Preguntas sobre la relación del tiempo de ejecución con la planificación óptima seleccionan automáticamente esa herramienta. Ollama solo interpreta y puede equivocarse; el JSON permite la verificación humana.

## Pruebas reproducibles

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m pytest -q
python3 -m compileall -q app.py src tests
```

Las pruebas comprueban: esquema, filtrado sin mutación, cero NA del dataset real, correlaciones, regla exacta de la etiqueta, ausencia de `Execution_Time`/`Task_ID` en la predicción preoperacional, métricas de clasificación, enrutamiento de herramientas, construcción del contexto y errores Ollama. La prueba Ollama usa un cliente simulado; el modelo real requiere verificación interactiva local.

## Repositorio y evidencia académica

El código se publica en un repositorio GitHub accesible para evaluación (sin incluir `.venv`, modelo Ollama ni secretos). La evidencia formal es un único PDF usando el formato institucional; añadir capturas reales, resultados de análisis, enlace al repositorio (Anexo B) y vídeo de demostración local en YouTube de **máximo 5 minutos** (Anexo C). No publicar el informe con números inventados ni enlaces de relleno.

## Limitaciones y posibles mejoras

- Dataset aparentemente sintético, con target derivado del tiempo de ejecución. No demuestra causalidad ni decisiones de scheduling realmente óptimas.
- `Priority` y `VM_ID` son categorías; evitar tratarlas como magnitudes ordinales continuas.
- El modelo preoperacional tiene poca señal, un resultado científicamente relevante; las prestaciones retrospectivas no pueden compararse como capacidad de anticipación.
- Aleatorización estratificada en train/test: correcta para un dataset de corte transversal ilustrativo; para secuencias operativas reales debe evaluarse con división temporal y datos externos.
- Los filtros pueden sesgar las distribuciones; las métricas del agente indican el alcance de los registros.
- Los gráficos en Plotly muestran correlaciones observadas, no evidencia causal; Ollama puede producir interpretaciones incorrectas pese al contexto.
- Se recomienda añadir evaluación por 5 folds, datos de producción y un dataset con resultados reales de un scheduler antes de afirmar utilidad en operaciones.

## Referencias técnicas

- [Documentación Streamlit](https://docs.streamlit.io/)
- [Documentación pandas](https://pandas.pydata.org/docs/)
- [Documentación scikit-learn](https://scikit-learn.org/stable/)
- [API Ollama](https://docs.ollama.com/api)
- [Kaggle: Cloud Task Scheduling Dataset](https://www.kaggle.com/datasets/ziya07/cloud-task-scheduling-dataset)
