"""Experimentos de clasificación reproducibles y prevención de fuga de información."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import EXECUTION, NUMERIC_FEATURES, TARGET

MODOS = {
    "pre_operacional": "Sin tiempo de ejecución (antes del trabajo)",
    "retrospectivo": "Con tiempo de ejecución (diagnóstico retrospectivo)",
}
MODELOS = {"logistica": "Regresión logística", "bosque": "Random Forest"}


class ErrorModelo(ValueError):
    """No hay suficientes datos adecuados para evaluar la clasificación."""


def preparar_pipeline(modo: str = "pre_operacional", algoritmo: str = "logistica", seed: int = 42) -> tuple[Pipeline, list[str]]:
    """Strategy/Factory simples: selecciona variables y clasificador sin cambiar el flujo."""
    if modo not in MODOS:
        raise ErrorModelo(f"Modo desconocido: {modo}")
    if algoritmo not in MODELOS:
        raise ErrorModelo(f"Algoritmo desconocido: {algoritmo}")
    numericas = [*NUMERIC_FEATURES]
    if modo == "retrospectivo":
        numericas.append(EXECUTION)
    categoricas = ["Priority", "VM_ID"]
    transformacion_numerica = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
    ])
    transformacion_categorica = Pipeline([
        ("imputar", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocesar = ColumnTransformer([
        ("numericas", transformacion_numerica, numericas),
        ("categoricas", transformacion_categorica, categoricas),
    ])
    if algoritmo == "logistica":
        estimador = LogisticRegression(max_iter=2000, random_state=seed)
    else:
        estimador = RandomForestClassifier(
            n_estimators=120, min_samples_leaf=2, max_depth=14,
            n_jobs=-1, random_state=seed,
        )
    return Pipeline([("preprocesar", preprocesar), ("modelo", estimador)]), numericas + categoricas


def evaluar_modelo(df: pd.DataFrame, modo: str = "pre_operacional",
                   algoritmo: str = "logistica", test_size: float = 0.2,
                   seed: int = 42) -> dict:
    """Separa primero train/test, ajusta Pipeline solo en train y evalúa test inédito."""
    if len(df) < 30:
        raise ErrorModelo("Se necesitan al menos 30 registros para evaluar el modelo.")
    if not 0 < test_size < 0.5:
        raise ErrorModelo("test_size debe estar entre 0 y 0.5.")
    if df[TARGET].isna().any() or not df[TARGET].isin([0, 1]).all():
        raise ErrorModelo("Target debe contener solamente etiquetas 0 y 1, sin nulos.")
    cuentas = df[TARGET].value_counts()
    if set(cuentas.index) != {0, 1} or cuentas.min() < 5:
        raise ErrorModelo("Se necesitan al menos 5 observaciones de cada clase (0 y 1).")
    pipeline, columnas = preparar_pipeline(modo, algoritmo, seed)
    X = df[columnas].copy()
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y,
    )
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]
    clase_mayoritaria = int(y_train.mode().iloc[0])
    pred_baseline = np.full(len(y_test), clase_mayoritaria, dtype=int)
    matriz = confusion_matrix(y_test, pred, labels=[0, 1]).tolist()
    metricas = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }
    return {
        "modo": modo,
        "modelo": algoritmo,
        "variables": columnas,
        "filas_train": len(X_train),
        "filas_test": len(X_test),
        "seed": seed,
        "metricas": metricas,
        "baseline_accuracy": float(accuracy_score(y_test, pred_baseline)),
        "matriz_confusion": matriz,
        "muestra_predicciones": [
            {"real": int(real), "predicho": int(pronostico), "p_clase_1": round(float(p), 4)}
            for real, pronostico, p in zip(y_test.iloc[:10], pred[:10], proba[:10])
        ],
        "advertencia": (
            "La etiqueta está determinada por Execution_Time < 5 s en el dataset original. "
            "Incluir esa variable es válido solo para diagnóstico retrospectivo, no como predictor anterior a la ejecución."
            if modo == "retrospectivo"
            else
            "La variable Execution_Time se excluyó por disponibilidad temporal. "
            "El dataset puede carecer de señal predictiva previa a la ejecución. "
            "Compare siempre con el baseline."
        ),
    }
