"""Interfaz Streamlit: análisis transparente de planificación de tareas cloud.

Ejecución: python3 -m streamlit run app.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.agent import AgenteDatos, HERRAMIENTAS, sintetizar_evidencia_tiempo
from src.data_loader import (EXECUTION, NUMERIC_FEATURES, TARGET, ErrorDataset,
                             cargar_csv, filtrar_dataset)
from src.eda import (analizar_relacion_target_tiempo, correlaciones_destacadas,
                     distribucion_objetivo, evidencia_regla_objetivo,
                     perfil_calidad, resumen_numerico, resumen_por_grupo)
from src.llm_client import ClienteOllama, ErrorOllama
from src.ml import ErrorModelo, MODELOS, MODOS, evaluar_modelo
from src.visualizations import (carga_por_vm, comparacion_prioridad,
                                dispersion_recursos, histograma, matriz_calor,
                                matriz_confusion)

st.set_page_config(page_title="Cloud Task Scheduling Analytics", page_icon="☁️",
                   layout="wide", initial_sidebar_state="expanded")

DATASET_INCLUIDO = Path(__file__).parent / "data" / "cloud_task_scheduling_dataset_20k.csv"


@st.cache_data(show_spinner=False)
def cargar_datos(bytes_csv: bytes) -> pd.DataFrame:
    """Cachear lectura inmutable; la validación vive en src/data_loader.py."""
    return cargar_csv(bytes_csv)


st.title("☁️ Cloud Task Scheduling Analytics")
st.caption("Proyecto Final Integrador · Streamlit + Python modular + Ollama local")
st.markdown("**Problema:** explorar recursos y tiempos de tareas en VMs; evaluar qué información sirve realmente para clasificar la etiqueta de planificación.")

with st.sidebar:
    st.header("Datos y filtros")
    origen = st.radio("Origen del dataset", ["Dataset incluido (20 000)", "Subir otro CSV"],
                      help="El CSV debe tener el mismo esquema de nueve columnas.")
    if origen == "Subir otro CSV":
        archivo = st.file_uploader("CSV de tareas cloud", type=["csv"],
                                   help="Máximo recomendado: 12 MiB.")
        if archivo is None:
            st.info("Adjunte un CSV válido para continuar.")
            st.stop()
        origen_bytes = archivo.getvalue()
    else:
        origen_bytes = DATASET_INCLUIDO.read_bytes()
    try:
        datos_originales = cargar_datos(origen_bytes)
    except (ErrorDataset, FileNotFoundError, OSError) as exc:
        st.error(f"Error al cargar el dataset: {exc}")
        st.stop()

    identidad = hashlib.sha256(origen_bytes).hexdigest()[:12]
    vms = sorted(datos_originales["VM_ID"].dropna().astype(int).unique().tolist())
    priorities = sorted(datos_originales["Priority"].dropna().astype(int).unique().tolist())
    vm_sel = st.multiselect("VM_ID", vms, default=vms)
    prioridad_sel = st.multiselect("Priority", priorities, default=priorities)
    clase_sel = st.multiselect("Target (0/1)", [0, 1], default=[0, 1])
    st.caption("Los filtros se aplican a los análisis y a las respuestas del agente.")

if not vm_sel or not prioridad_sel or not clase_sel:
    st.warning("Seleccione al menos un valor en cada filtro.")
    st.stop()

datos = filtrar_dataset(datos_originales, vm_sel, prioridad_sel, clase_sel)
if datos.empty:
    st.warning("Los filtros no devolvieron registros.")
    st.stop()

alcance = json.dumps({"csv": identidad, "vms": vm_sel, "prioridades": prioridad_sel,
                      "clases": clase_sel}, sort_keys=True)
info_calidad = perfil_calidad(datos)
info_objetivo = distribucion_objetivo(datos)
regla = evidencia_regla_objetivo(datos)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Registros filtrados", f"{len(datos):,}")
col2.metric("VMs seleccionadas", datos["VM_ID"].nunique())
col3.metric("Nulos (celdas)", info_calidad["nulos_totales"])
col4.metric("Regla temporal: coincidencia", f"{regla['coincidencia_pct']:.2f}%")

if regla["coincidencia_pct"] == 100:
    st.warning("**Hallazgo metodológico:** en esta selección, `Target=1` coincide exactamente con `Execution_Time < 5 s`. "
               "Usar el tiempo real de ejecución para predecir la etiqueta antes de comenzar una tarea sería una fuga de información temporal.")
else:
    st.info("La etiqueta no se explica completamente por la regla temporal en la selección actual.")

tab_datos, tab_eda, tab_ml, tab_ia, tab_arquitectura = st.tabs([
    "1 · Dataset y calidad", "2 · Exploración", "3 · Machine Learning",
    "4 · Agente IA (Ollama)", "5 · Arquitectura",
])

with tab_datos:
    st.subheader("Perfil del dataset seleccionado")
    st.write("Los cálculos son realizados por pandas, no por el modelo de lenguaje.")
    a, b = st.columns(2)
    with a:
        st.metric("Filas originales", len(datos_originales))
        st.metric("Duplicados exactos (filtrado)", info_calidad["duplicados"])
        st.dataframe(pd.DataFrame(info_calidad["variables"]), hide_index=True,
                     use_container_width=True)
    with b:
        st.markdown("**Distribución de Target**")
        st.dataframe(pd.DataFrame(info_objetivo), hide_index=True, use_container_width=True)
        st.markdown("**Comprobación de regla empírica**")
        st.json(regla)
    with st.expander("Relación Target / tiempo: recuentos y rangos por clase", expanded=False):
        detalle = analizar_relacion_target_tiempo(datos)
        st.dataframe(pd.DataFrame(detalle["distribucion_y_tiempos_por_clase"]),
                     hide_index=True, use_container_width=True)
        st.caption(detalle["interpretacion_metodologica"])
    st.markdown("**Vista de los registros filtrados**")
    st.dataframe(datos.head(200), use_container_width=True, height=260)
    st.download_button("Descargar muestra filtrada (CSV)",
                       data=datos.to_csv(index=False).encode("utf-8"),
                       file_name="tareas_filtradas.csv", mime="text/csv")
    st.caption("Procedencia: CSV suministrado; unidad analizada = tarea. La etiqueta no demuestra por sí sola que se ejecutó un scheduler real.")

with tab_eda:
    st.subheader("Exploración, segmentación y correlaciones")
    left, right = st.columns(2)
    with left:
        variable = st.selectbox("Variable numérica", [*NUMERIC_FEATURES, EXECUTION])
        st.plotly_chart(histograma(datos, variable), use_container_width=True)
    with right:
        st.markdown(f"**Estadísticos de {variable}**")
        st.dataframe(pd.DataFrame([resumen_numerico(datos, variable)]),
                     use_container_width=True, hide_index=True)
        st.plotly_chart(carga_por_vm(datos), use_container_width=True)
    u, v = st.columns(2)
    with u:
        st.plotly_chart(comparacion_prioridad(datos), use_container_width=True)
        st.markdown("**Por prioridad**")
        st.dataframe(pd.DataFrame(resumen_por_grupo(datos, "Priority")),
                     hide_index=True, use_container_width=True)
    with v:
        if len(datos) > 1:
            st.plotly_chart(matriz_calor(datos), use_container_width=True)
        st.markdown("**Correlaciones destacadas**")
        st.dataframe(pd.DataFrame(correlaciones_destacadas(datos)),
                     hide_index=True, use_container_width=True)
    st.plotly_chart(dispersion_recursos(datos), use_container_width=True)
    st.caption("Las correlaciones Pearson describen asociación, no causalidad. VM_ID y Priority son códigos categóricos, excluidos de la matriz numérica.")

with tab_ml:
    st.subheader("Clasificación de Target — evaluación reproducible")
    st.markdown("El experimento utiliza un `Pipeline` scikit-learn; imputación, escalado y codificación se ajustan **solo con train**, nunca con test.")
    modo = st.radio("Enfoque del experimento", list(MODOS),
                    format_func=lambda valor: MODOS[valor], horizontal=True)
    algoritmo = st.selectbox("Clasificador", list(MODELOS),
                             format_func=lambda valor: MODELOS[valor])
    if modo == "retrospectivo":
        st.error("**Solo retrospectivo:** incluye `Execution_Time`. La precisión alta puede ser trivial porque la etiqueta es una función de ese valor. No es un predictor preoperacional.")
    else:
        st.info("**Preoperacional:** se excluyen `Execution_Time` y `Task_ID`. Se codifican Priority y VM_ID como categorías. La señal predictiva puede ser escasa.")
    llave_modelo = f"{alcance}:{modo}:{algoritmo}"
    if st.button("Entrenar y evaluar", type="primary"):
        try:
            with st.spinner("Entrenando y evaluando sobre un test independiente..."):
                resultado = evaluar_modelo(datos, modo=modo, algoritmo=algoritmo)
            st.session_state["modelo_ultimo"] = {"llave": llave_modelo, "resultado": resultado}
        except (ErrorModelo, ValueError) as exc:
            st.error(f"No se pudo evaluar el modelo: {exc}")
    registrado = st.session_state.get("modelo_ultimo", {})
    if registrado.get("llave") == llave_modelo:
        resultado = registrado["resultado"]
        mets = resultado["metricas"]
        a, b, c, d, e = st.columns(5)
        a.metric("Accuracy", f"{mets['accuracy']:.3f}")
        b.metric("Precision", f"{mets['precision']:.3f}")
        c.metric("Recall", f"{mets['recall']:.3f}")
        d.metric("F1", f"{mets['f1']:.3f}")
        e.metric("ROC AUC", f"{mets['roc_auc']:.3f}")
        st.write(f"**Baseline (siempre clase mayoritaria):** accuracy = {resultado['baseline_accuracy']:.3f}")
        st.write(f"Train/Test: {resultado['filas_train']}/{resultado['filas_test']} · random_state = {resultado['seed']}")
        st.warning(resultado["advertencia"])
        izquierda, derecha = st.columns(2)
        with izquierda:
            st.plotly_chart(matriz_confusion(resultado["matriz_confusion"]), use_container_width=True)
        with derecha:
            st.markdown("**Muestra de predicciones del test**")
            st.dataframe(pd.DataFrame(resultado["muestra_predicciones"]),
                         hide_index=True, use_container_width=True)
            st.caption("ROC AUC se calcula con `predict_proba(X_test)[:, 1]`; F1, precision y recall con etiquetas 0/1.")
        st.download_button("Exportar métricas (JSON)",
                           data=json.dumps(resultado, ensure_ascii=False, indent=2).encode("utf-8"),
                           file_name="evaluacion_clasificacion.json", mime="application/json")
    else:
        st.caption("Pulse **Entrenar y evaluar** para generar métricas reales con los filtros y parámetros seleccionados.")

with tab_ia:
    st.subheader("Agente de interpretación — Ollama local")
    st.markdown("El agente **no calcula** estadísticas con el LLM. Ejecuta herramientas Python autorizadas, envía sus resultados como JSON y pide a Ollama una interpretación en español.")
    cliente = ClienteOllama.desde_entorno()
    col_a, col_b = st.columns([2, 3])
    with col_a:
        st.code(f"Modelo: {cliente.modelo}\nServicio: {cliente.base_url}", language="text")
        if st.button("Comprobar Ollama"):
            try:
                disponibles = cliente.modelos_disponibles()
                if cliente.modelo in disponibles:
                    st.success(f"Modelo disponible: {cliente.modelo}")
                else:
                    st.warning(f"Ollama funciona, pero falta {cliente.modelo}. Ejecute: ollama pull {cliente.modelo}")
            except ErrorOllama as exc:
                st.error(str(exc))
    with col_b:
        pregunta = st.text_area("Pregunta para el agente", max_chars=800, height=85,
                                placeholder="¿Cuál es el tiempo promedio de ejecución y qué relación tiene con la planificación óptima?")
        st.caption("Ejemplo: ¿Cuál es el tiempo promedio de ejecución y qué relación tiene con la planificación óptima?")
        seleccion = st.multiselect("Herramientas de análisis (vacío = selección automática)",
                                  list(HERRAMIENTAS), format_func=lambda nombre: HERRAMIENTAS[nombre])
    if st.button("Consultar agente", type="primary", key="btn_ia"):
        if not pregunta.strip():
            st.warning("Introduzca una pregunta.")
        else:
            agent = AgenteDatos(cliente)
            anterior = st.session_state.get("modelo_ultimo", {})
            resultado_modelo = anterior.get("resultado") if anterior.get("llave", "").startswith(f"{alcance}:") else None
            try:
                contexto = agent.construir_contexto(datos, pregunta,
                                                    herramientas=seleccion or None,
                                                    resultado_modelo=resultado_modelo)
                st.session_state["agente_contexto"] = contexto
                st.session_state["agente_alcance"] = alcance
                st.session_state.pop("agente_respuesta", None)
                with st.spinner("Consultando el modelo local (puede tardar)..."):
                    respuesta = agent.responder(pregunta, contexto)
                st.session_state["agente_respuesta"] = respuesta
            except (ErrorOllama, ValueError) as exc:
                st.error(f"El agente no pudo responder: {exc}")
                st.info("Los gráficos y cálculos de Python siguen disponibles sin Ollama.")
                st.session_state.pop("agente_respuesta", None)
                st.session_state.pop("agente_contexto", None)
    if st.session_state.get("agente_alcance") == alcance:
        if st.session_state.get("agente_respuesta"):
            hechos = sintetizar_evidencia_tiempo(st.session_state.get("agente_contexto", {}))
            if hechos:
                st.markdown("**Hechos verificados por Python (no por la IA)**")
                st.info(hechos)
            st.markdown("**Interpretación generada por Ollama**")
            st.write(st.session_state["agente_respuesta"])
        if "agente_contexto" in st.session_state:
            with st.expander("Ver evidencia JSON proporcionada al agente", expanded=False):
                st.json(st.session_state["agente_contexto"])
    st.caption("Advertencia: la salida del modelo es una interpretación automática que debe verificarse contra el JSON. Si Ollama no está disponible, no se simula una respuesta.")

with tab_arquitectura:
    st.subheader("Arquitectura modular y paradigmas")
    st.code("""Usuario → Streamlit (app.py)
    ├── src/data_loader.py     CSV, validación, filtros
    ├── src/eda.py             funciones estadísticas puras
    ├── src/visualizations.py  gráficos Plotly
    ├── src/ml.py              pipelines / estrategias sklearn
    └── src/agent.py           clase AgenteDatos + herramientas permitidas
              └── src/llm_client.py → Ollama local (HTTP :11434)
""", language="text")
    st.markdown("""**Multiparadigma:** procedimientos imperativos para controles y validaciones; funciones reutilizables en EDA; orientación a objetos en `AgenteDatos`, `ClienteOllama` y los estimadores de scikit-learn.

**Separación de responsabilidades:** los cálculos se producen en `src/eda.py` y `src/ml.py`; la IA solo recibe JSON con evidencia. Las herramientas se seleccionan en Python desde una lista permitida; el texto del modelo nunca se ejecuta como código.

**Reproducibilidad:** `requirements.txt`, `README.md`, tests con `pytest`, CSV de 20 000 registros, `random_state=42` y `ollama pull llama3.2:3b`.

**Limitación:** los datos muestran una etiqueta determinada por el tiempo real de ejecución. El clasificador retrospectivo no constituye un scheduler operativo ni una demostración causal.""")
