"""Gráficos Plotly reutilizables. No dependen de la interfaz Streamlit."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import EXECUTION, TARGET
from src.eda import matriz_correlaciones


def histograma(df: pd.DataFrame, variable: str):
    fig = px.histogram(df, x=variable, nbins=40, marginal="box", title=f"Distribución: {variable}")
    fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), bargap=0.08)
    return fig


def carga_por_vm(df: pd.DataFrame):
    tabla = (df.groupby("VM_ID", observed=True)
             .agg(tareas=("Task_ID", "count"), cpu_media=("CPU_Usage (%)", "mean"))
             .reset_index())
    tabla["VM_ID"] = tabla["VM_ID"].astype(str)
    fig = px.bar(tabla, x="VM_ID", y="tareas", color="cpu_media",
                 color_continuous_scale="Blues", title="Tareas por VM y CPU medio",
                 labels={"tareas": "Número de tareas", "cpu_media": "CPU medio (%)"})
    fig.update_layout(margin=dict(l=10, r=10, t=55, b=10))
    return fig


def comparacion_prioridad(df: pd.DataFrame):
    tabla = df.groupby(["Priority", TARGET], observed=True).size().rename("total").reset_index()
    tabla["Priority"] = tabla["Priority"].astype(str)
    tabla[TARGET] = tabla[TARGET].astype(str)
    fig = px.bar(tabla, x="Priority", y="total", color=TARGET, barmode="group",
                 title="Distribución de etiquetas por prioridad", labels={"total": "Tareas"})
    fig.update_layout(margin=dict(l=10, r=10, t=55, b=10))
    return fig


def dispersion_recursos(df: pd.DataFrame, muestra: int = 2500):
    # Una muestra fija para no enviar 20k puntos al navegador en cada interacción.
    subset = df.sample(n=min(muestra, len(df)), random_state=42).copy()
    subset[TARGET] = subset[TARGET].astype(str)
    return px.scatter(subset, x="CPU_Usage (%)", y="RAM_Usage (MB)", color=TARGET,
                      hover_data=["Priority", "VM_ID", EXECUTION], opacity=0.65,
                      title="Relación CPU–RAM (muestra reproducible)")


def matriz_calor(df: pd.DataFrame):
    corr = matriz_correlaciones(df)
    fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1,
                    color_continuous_scale="RdBu_r", aspect="auto",
                    title="Correlaciones Pearson de variables continuas")
    fig.update_layout(margin=dict(l=10, r=10, t=55, b=10))
    return fig


def matriz_confusion(matriz: list[list[int]]):
    fig = go.Figure(data=go.Heatmap(z=matriz, x=["Pred 0", "Pred 1"],
                                    y=["Real 0", "Real 1"],
                                    text=matriz, texttemplate="%{text}",
                                    colorscale="Blues", showscale=False))
    fig.update_layout(title="Matriz de confusión (test)",
                      xaxis_title="Predicción", yaxis_title="Clase real",
                      margin=dict(l=10, r=10, t=55, b=10))
    return fig
