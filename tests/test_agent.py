import pytest
from requests.exceptions import ConnectionError

from src.agent import AgenteDatos, seleccionar_herramientas
from src.llm_client import ClienteOllama, ErrorOllama


class ClienteFalso:
    def __init__(self):
        self.mensajes = None

    def responder(self, mensajes):
        self.mensajes = mensajes
        return "La media procede del contexto JSON."


def test_ruteo_herramientas():
    assert "maquinas" in seleccionar_herramientas("¿Qué VM tiene más tareas?")
    assert "prioridades" in seleccionar_herramientas("¿Cuál prioridad consume más CPU?")
    assert "modelo" in seleccionar_herramientas("¿Cuál es el F1 del modelo?")


def test_contexto_calculado_no_dataset_completo(datos):
    cliente = ClienteFalso()
    agente = AgenteDatos(cliente)
    contexto = agente.construir_contexto(datos, "¿Cuál es la media de CPU?", ["estadisticas", "clases"])
    assert contexto["registros_filtrados"] == 20000
    assert "estadisticas" in contexto
    assert "regla_objetivo" in contexto
    assert "muestra_original" not in contexto
    texto = agente.responder("¿Cuál es la media de CPU?", contexto)
    assert texto
    assert cliente.mensajes[0]["role"] == "system"
    assert '"registros_filtrados":20000' in cliente.mensajes[1]["content"]


def test_herramientas_no_autorizadas_y_pregunta_extensa(datos):
    agente = AgenteDatos(ClienteFalso())
    with pytest.raises(ValueError, match="no autorizadas"):
        agente.construir_contexto(datos, "eval", ["codigo_python"])
    with pytest.raises(ValueError, match="800"):
        agente.responder("x" * 801, {})


def test_error_ollama_sin_conexion(monkeypatch):
    def fallo(*args, **kwargs):
        raise ConnectionError("offline")
    import src.llm_client as m
    monkeypatch.setattr(m.requests, "get", fallo)
    cliente = ClienteOllama()
    with pytest.raises(ErrorOllama, match="No se puede contactar"):
        cliente.modelos_disponibles()
    monkeypatch.setattr(m.requests, "post", fallo)
    with pytest.raises(ErrorOllama, match="Ollama no responde"):
        cliente.responder([{"role": "user", "content": "Prueba"}])


def test_pregunta_natural_activa_relacion_tiempo(datos):
    from src.agent import sintetizar_evidencia_tiempo

    pregunta = "¿Cuál es el tiempo promedio de ejecución y qué relación tiene con la planificación óptima?"
    herramientas = seleccionar_herramientas(pregunta)
    assert "relacion_tiempo" in herramientas
    assert "estadisticas" in herramientas
    agente = AgenteDatos(ClienteFalso())
    contexto = agente.construir_contexto(datos, pregunta)
    evidencia = contexto["relacion_target_tiempo"]
    assert evidencia["observaciones_validas"] == 20000
    assert evidencia["regla_verificada"]["discordancias"] == 0
    assert evidencia["regla_verificada"]["coincidencia_pct"] == 100
    assert [g["registros"] for g in evidencia["distribucion_y_tiempos_por_clase"]] == [11111, 8889]
    assert [g["tiempo_minimo_s"] for g in evidencia["distribucion_y_tiempos_por_clase"]] == [5.0, 1.0]
    resumen = sintetizar_evidencia_tiempo(contexto)
    assert "5.5038" in resumen
    assert "20,000" in resumen
    assert "discordancias 0" in resumen
    assert "Target=0" in resumen and "Target=1" in resumen
    agente.responder(pregunta, contexto)
    assert '"relacion_target_tiempo"' in agente.cliente.mensajes[1]["content"]


def test_contexto_vm_y_clase_filtradas_no_se_extiende_a_dataset(datos):
    from src.agent import sintetizar_evidencia_tiempo

    vm = int(datos["VM_ID"].iloc[0])
    filtrados = datos[(datos["VM_ID"] == vm) & (datos["Target (Optimal Scheduling)"] == 1)]
    agente = AgenteDatos(ClienteFalso())
    contexto = agente.construir_contexto(
        filtrados, "¿Cuál es la relación entre tiempo y planificación óptima?",
    )
    resultado = contexto["relacion_target_tiempo"]
    assert 0 < resultado["observaciones_validas"] < 20000
    assert len(resultado["distribucion_y_tiempos_por_clase"]) == 1
    assert resultado["distribucion_y_tiempos_por_clase"][0]["clase"] == 1
    assert resultado["regla_verificada"]["coincidencia_pct"] == 100
    assert f"{len(filtrados):,}" in sintetizar_evidencia_tiempo(contexto)


def test_mismatch_regla_no_se_describe_como_perfecta(datos):
    from src.agent import sintetizar_evidencia_tiempo

    alterado = datos.head(100).copy(deep=True)
    target = "Target (Optimal Scheduling)"
    alterado.loc[alterado.index[0], target] = 1 - alterado.loc[alterado.index[0], target]
    contexto = AgenteDatos(ClienteFalso()).construir_contexto(
        alterado, "¿Qué relación tiene el tiempo de ejecución con Target?"
    )
    regla = contexto["relacion_target_tiempo"]["regla_verificada"]
    assert regla["discordancias"] == 1
    assert regla["coincidencia_pct"] == 99
    assert "discordancias 1" in sintetizar_evidencia_tiempo(contexto)


def test_contexto_manualmente_limitado_no_inventa_evidencia(datos):
    from src.agent import sintetizar_evidencia_tiempo

    contexto = AgenteDatos(ClienteFalso()).construir_contexto(
        datos, "¿Qué relación existe con Target?", herramientas=["correlaciones"]
    )
    assert "relacion_target_tiempo" not in contexto
    assert sintetizar_evidencia_tiempo(contexto) is None
