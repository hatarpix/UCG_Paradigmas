import pandas as pd
import pytest

from src.data_loader import ErrorDataset, TARGET, cargar_csv, filtrar_dataset, validar_dataset
from src.eda import (correlaciones_destacadas, distribucion_objetivo,
                     evidencia_regla_objetivo, perfil_calidad,
                     resumen_numerico, resumen_por_grupo)


def test_dataset_real_tamano_y_calidad(datos):
    perfil = perfil_calidad(datos)
    assert perfil["registros"] == 20000
    assert perfil["columnas"] == 9
    assert perfil["nulos_totales"] == 0
    assert perfil["duplicados"] == 0
    assert sum(grupo["registros"] for grupo in distribucion_objetivo(datos)) == 20000


def test_regla_objetivo_exacta_en_csv_real(datos):
    regla = evidencia_regla_objetivo(datos)
    assert regla["observaciones"] == 20000
    assert regla["discordancias"] == 0
    assert regla["coincidencia_pct"] == 100.0


def test_correlaciones_sin_id_ni_categorias(datos):
    pares = correlaciones_destacadas(datos)
    assert pares
    assert "Task_ID" not in str(pares)
    assert "VM_ID" not in str(pares)
    assert "Priority" not in str(pares)
    assert any(abs(p["correlacion_pearson"]) > 0.8 for p in pares)


def test_resumen_y_segmentos(datos):
    res = resumen_numerico(datos, "Execution_Time (s)")
    assert 1 <= res["media"] <= 10
    assert res["observaciones"] == 20000
    assert len(resumen_por_grupo(datos, "VM_ID")) == 10
    assert len(resumen_por_grupo(datos, "Priority")) == 3


def test_filtro_no_muta_original(datos):
    copia = datos.copy(deep=True)
    filtrados = filtrar_dataset(datos, [100, 101], [1], [0])
    assert len(filtrados) < len(datos)
    assert set(filtrados["VM_ID"].unique()) <= {100, 101}
    assert set(filtrados[TARGET].unique()) == {0}
    pd.testing.assert_frame_equal(datos, copia)


def test_errores_validacion(datos):
    with pytest.raises(ErrorDataset, match="Columnas"):
        validar_dataset(datos.drop(columns=["Priority"]))
    malo = datos.iloc[:30].copy()
    malo.loc[0, TARGET] = 99
    with pytest.raises(ErrorDataset, match="binarias"):
        validar_dataset(malo)
    duplicados = datos.iloc[:30].copy()
    duplicados.loc[1, "Task_ID"] = duplicados.loc[0, "Task_ID"]
    with pytest.raises(ErrorDataset, match="Task_ID"):
        validar_dataset(duplicados)
    with pytest.raises(FileNotFoundError):
        cargar_csv("/ruta/inexistente/dataset.csv")


def test_cargar_bytes_y_permitir_nulos_en_predictores(datos):
    x = datos.iloc[:30].copy()
    x.loc[0, "CPU_Usage (%)"] = float("nan")
    valido = cargar_csv(x.to_csv(index=False).encode("utf-8"))
    assert valido["CPU_Usage (%)"].isna().sum() == 1


def test_analisis_tiempo_por_clase_sin_valores_extrapolados(datos):
    from src.eda import analizar_relacion_target_tiempo

    resultado = analizar_relacion_target_tiempo(datos)
    grupos = resultado["distribucion_y_tiempos_por_clase"]
    assert len(grupos) == 2
    assert grupos[0]["clase"] == 0 and grupos[0]["tiempo_minimo_s"] >= 5
    assert grupos[1]["clase"] == 1 and grupos[1]["tiempo_maximo_s"] < 5
    assert sum(g["registros"] for g in grupos) == len(datos)
    assert resultado["regla_verificada"]["coincidencia_pct"] == 100
    vacio = analizar_relacion_target_tiempo(datos.iloc[:0])
    assert vacio["observaciones_validas"] == 0
    assert vacio["distribucion_y_tiempos_por_clase"] == []
