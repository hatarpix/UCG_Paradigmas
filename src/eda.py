"""Funciones analíticas puras, independientes de Streamlit y Ollama."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import EXECUTION, NUMERIC_FEATURES, REQUIRED_COLUMNS, TARGET


def perfil_calidad(df: pd.DataFrame) -> dict:
    """Dimensiones, nulos y duplicados, por variable."""
    n = len(df)
    variables = []
    for col in df.columns:
        variables.append({
            "variable": str(col),
            "tipo": str(df[col].dtype),
            "nulos": int(df[col].isna().sum()),
            "porcentaje_nulos": round(100 * df[col].isna().mean(), 2) if n else 0.0,
            "valores_unicos": int(df[col].nunique(dropna=True)),
        })
    return {
        "registros": n,
        "columnas": len(df.columns),
        "nulos_totales": int(df.isna().sum().sum()),
        "duplicados": int(df.duplicated().sum()),
        "variables": variables,
    }


def resumen_numerico(df: pd.DataFrame, variable: str) -> dict:
    """Estadísticos descriptivos con percentiles y rango intercuartílico."""
    if variable not in df.columns:
        raise KeyError(f"No existe la variable '{variable}'.")
    if not pd.api.types.is_numeric_dtype(df[variable]):
        raise TypeError(f"La variable '{variable}' no es numérica.")
    s = df[variable].dropna()
    if s.empty:
        return {"variable": variable, "observaciones": 0, "mensaje": "Sin valores disponibles"}
    q1, q3 = s.quantile([0.25, 0.75])
    return {
        "variable": variable,
        "observaciones": len(s),
        "media": round(float(s.mean()), 4),
        "mediana": round(float(s.median()), 4),
        "desviacion_estandar": round(float(s.std()), 4) if len(s) > 1 else None,
        "minimo": float(s.min()),
        "maximo": float(s.max()),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": round(float(q3 - q1), 4),
    }


def distribucion_objetivo(df: pd.DataFrame) -> list[dict]:
    """Número y porcentaje de etiquetas 0/1."""
    n = len(df)
    return [
        {"clase": int(clase), "registros": int(cant), "porcentaje": round(100 * cant / n, 2)}
        for clase, cant in df[TARGET].value_counts().sort_index().items()
    ] if n else []


def resumen_por_grupo(df: pd.DataFrame, columna: str) -> list[dict]:
    """Comparación de grupos VM_ID/Priority, sin asumir relaciones causales."""
    if columna not in {"Priority", "VM_ID"}:
        raise ValueError("Agrupación permitida: 'Priority' o 'VM_ID'.")
    if df.empty:
        return []
    tabla = (df.groupby(columna, dropna=False, observed=True)
             .agg(registros=(TARGET, "size"),
                  porcentaje_optimo=(TARGET, "mean"),
                  ejecucion_media_s=(EXECUTION, "mean"),
                  cpu_media=("CPU_Usage (%)", "mean"),
                  ram_media_mb=("RAM_Usage (MB)", "mean"))
             .reset_index())
    tabla["porcentaje_optimo"] *= 100
    for c in ("porcentaje_optimo", "ejecucion_media_s", "cpu_media", "ram_media_mb"):
        tabla[c] = tabla[c].round(2)
    return tabla.to_dict(orient="records")


def matriz_correlaciones(df: pd.DataFrame) -> pd.DataFrame:
    """Correlaciones de variables continuas; excluye ID y códigos nominales."""
    columnas = [*NUMERIC_FEATURES, EXECUTION, TARGET]
    if len(df) < 2:
        return pd.DataFrame(index=columnas, columns=columnas, dtype=float)
    return df[columnas].corr(numeric_only=True)


def evidencia_regla_objetivo(df: pd.DataFrame) -> dict:
    """Verifica si la etiqueta equivale a Execution_Time < 5 s."""
    validos = df[[EXECUTION, TARGET]].dropna()
    if validos.empty:
        return {"observaciones": 0, "coincidencia_pct": None, "discordancias": None}
    etiqueta_regla = (validos[EXECUTION] < 5.0).astype(int)
    incorrectos = int((etiqueta_regla != validos[TARGET]).sum())
    return {
        "regla": "Target=1 si Execution_Time (s) < 5; Target=0 en caso contrario",
        "observaciones": len(validos),
        "discordancias": incorrectos,
        "coincidencia_pct": round(100 * (1 - incorrectos / len(validos)), 3),
    }



def analizar_relacion_target_tiempo(df: pd.DataFrame) -> dict:
    """Comprueba el umbral observado y describe los tiempos para cada clase.

    Calcula siempre sobre la selección recibida, sin aprender ni inferir etiquetas.
    En particular, una selección con una sola clase es válida para análisis EDA.
    """
    if EXECUTION not in df.columns or TARGET not in df.columns:
        raise KeyError(f"Se necesitan las columnas '{EXECUTION}' y '{TARGET}'.")

    validos = df[[EXECUTION, TARGET]].dropna()
    evidencia = evidencia_regla_objetivo(df)
    n = len(validos)
    por_clase = []
    for clase, grupo in validos.groupby(TARGET, sort=True):
        tiempos = grupo[EXECUTION]
        por_clase.append({
            "clase": int(clase),
            "registros": int(len(grupo)),
            "porcentaje": round(100 * len(grupo) / n, 2) if n else 0.0,
            "tiempo_minimo_s": round(float(tiempos.min()), 4),
            "tiempo_maximo_s": round(float(tiempos.max()), 4),
            "tiempo_medio_s": round(float(tiempos.mean()), 4),
        })

    return {
        "alcance": "Solo registros de la selección filtrada; no extrapolar a otros datos",
        "variable_tiempo": EXECUTION,
        "variable_etiqueta": TARGET,
        "observaciones_validas": int(n),
        "distribucion_y_tiempos_por_clase": por_clase,
        "regla_verificada": evidencia,
        "interpretacion_metodologica": (
            "La coincidencia observada describe el etiquetado del CSV; "
            "no prueba por sí sola que exista una planificación óptima real. "
            "Si Execution_Time solo se conoce al terminar, usarlo para pronosticar "
            "Target antes de ejecutar una tarea introduce fuga temporal de información."
        ),
    }


def correlaciones_destacadas(df: pd.DataFrame, limite: int = 6) -> list[dict]:
    """Pares de variables continuas con mayor correlación absoluta."""
    matriz = matriz_correlaciones(df)
    pares = []
    for i, a in enumerate(matriz.columns):
        for b in matriz.columns[i + 1:]:
            r = matriz.loc[a, b]
            if pd.notna(r):
                pares.append({"variable_1": a, "variable_2": b, "correlacion_pearson": round(float(r), 4)})
    return sorted(pares, key=lambda x: abs(x["correlacion_pearson"]), reverse=True)[:limite]
