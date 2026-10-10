import pytest

from src.data_loader import EXECUTION
from src.ml import ErrorModelo, evaluar_modelo, preparar_pipeline


def test_variables_evitan_leakage_preoperacional():
    pipeline, columnas = preparar_pipeline("pre_operacional", "logistica")
    assert EXECUTION not in columnas
    assert "Task_ID" not in columnas
    assert set(["VM_ID", "Priority"]).issubset(columnas)
    assert pipeline.steps[-1][0] == "modelo"


def test_experiment_baseline_y_confusion(datos):
    res = evaluar_modelo(datos, "pre_operacional", "logistica")
    assert res["filas_train"] == 16000
    assert res["filas_test"] == 4000
    assert 0.53 <= res["baseline_accuracy"] <= 0.57
    assert 0 <= res["metricas"]["roc_auc"] <= 1
    assert sum(sum(row) for row in res["matriz_confusion"]) == 4000
    assert len(res["muestra_predicciones"]) == 10


def test_modo_retrospectivo_detecta_regla(datos):
    res = evaluar_modelo(datos, "retrospectivo", "logistica")
    assert EXECUTION in res["variables"]
    assert res["metricas"]["roc_auc"] > 0.995
    assert "retrospectivo" in res["advertencia"]


def test_modelo_no_admite_muestra_reducida(datos):
    with pytest.raises(ErrorModelo, match="30"):
        evaluar_modelo(datos.head(20))
    with pytest.raises(ErrorModelo, match="al menos 5"):
        evaluar_modelo(datos.query("`Target (Optimal Scheduling)` == 1").head(40))
    with pytest.raises(ErrorModelo, match="Modo desconocido"):
        preparar_pipeline("inexistente")
