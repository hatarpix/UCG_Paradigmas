"""Carga y validación de los datos de planificación de tareas cloud."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd

TARGET = "Target (Optimal Scheduling)"
EXECUTION = "Execution_Time (s)"
NUMERIC_FEATURES = ["CPU_Usage (%)", "RAM_Usage (MB)", "Disk_IO (MB/s)", "Network_IO (MB/s)"]
IDENTIFIERS = ["Task_ID", "VM_ID"]
CATEGORICAL_FEATURES = ["Priority", "VM_ID"]
REQUIRED_COLUMNS = ["Task_ID", *NUMERIC_FEATURES, "Priority", "VM_ID", EXECUTION, TARGET]
MAX_UPLOAD_BYTES = 12 * 1024 * 1024


class ErrorDataset(ValueError):
    """Los datos no cumplen el esquema esperado."""


def cargar_csv(origen: str | Path | bytes) -> pd.DataFrame:
    """Lee un CSV local o su contenido binario, y valida el esquema."""
    try:
        if isinstance(origen, bytes):
            if len(origen) > MAX_UPLOAD_BYTES:
                raise ErrorDataset("El archivo supera el límite de 12 MiB.")
            df = pd.read_csv(BytesIO(origen), low_memory=False)
        else:
            ruta = Path(origen)
            if not ruta.is_file():
                raise FileNotFoundError(f"No existe el CSV: {ruta}")
            df = pd.read_csv(ruta, low_memory=False)
    except (pd.errors.ParserError, UnicodeError, pd.errors.EmptyDataError) as exc:
        raise ErrorDataset(f"No se pudo leer el CSV: {exc}") from exc
    return validar_dataset(df)


def validar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Requiere columnas, valores numéricos y una etiqueta binaria 0/1."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ErrorDataset("El dataset debe contener registros.")
    faltantes = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if faltantes:
        raise ErrorDataset(f"Columnas obligatorias ausentes: {faltantes}")
    if df.columns.duplicated().any():
        raise ErrorDataset("Los nombres de columnas deben ser únicos.")
    datos = df.loc[:, REQUIRED_COLUMNS].copy()
    for col in REQUIRED_COLUMNS:
        originales = datos[col]
        datos[col] = pd.to_numeric(originales, errors="coerce")
        invalidos = originales.notna() & datos[col].isna()
        if invalidos.any():
            raise ErrorDataset(f"La columna '{col}' tiene {int(invalidos.sum())} valores no numéricos.")
    valid_target = datos[TARGET].dropna().isin([0, 1]).all()
    if not valid_target or datos[TARGET].dropna().nunique() < 2:
        raise ErrorDataset("La variable objetivo debe contener ambas clases binarias: 0 y 1.")
    if datos[TARGET].isna().any():
        raise ErrorDataset("Target no puede contener valores nulos.")
    if datos["Task_ID"].duplicated().any():
        raise ErrorDataset("Task_ID tiene identificadores duplicados.")
    # No se eliminan NA: se contabilizan; un Pipeline los imputa al entrenar.
    return datos


def filtrar_dataset(
    df: pd.DataFrame,
    vm_ids: list[int] | None = None,
    prioridades: list[int] | None = None,
    clases: list[int] | None = None,
) -> pd.DataFrame:
    """Filtra registros sin alterar los datos originales."""
    mask = pd.Series(True, index=df.index)
    if vm_ids is not None:
        mask &= df["VM_ID"].isin(vm_ids)
    if prioridades is not None:
        mask &= df["Priority"].isin(prioridades)
    if clases is not None:
        mask &= df[TARGET].isin(clases)
    return df.loc[mask].copy()
