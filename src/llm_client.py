"""Cliente HTTP para Ollama local. No hay respuestas ficticias de respaldo."""

from __future__ import annotations

import os
from dataclasses import dataclass

import requests


class ErrorOllama(RuntimeError):
    """Error de conexión, modelo ausente o respuesta inválida de Ollama."""


@dataclass(frozen=True)
class ClienteOllama:
    """Adaptador que separa HTTP/Ollama del cálculo estadístico."""

    modelo: str = "llama3.2:3b"
    base_url: str = "http://127.0.0.1:11434"
    timeout: int = 120

    @classmethod
    def desde_entorno(cls, modelo: str | None = None) -> "ClienteOllama":
        return cls(modelo=modelo or os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
                   base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))

    def modelos_disponibles(self) -> list[str]:
        try:
            resp = requests.get(f"{self.base_url.rstrip('/')}/api/tags", timeout=5)
            resp.raise_for_status()
            return [str(item.get("name", "")) for item in resp.json().get("models", [])]
        except (requests.RequestException, ValueError) as exc:
            raise ErrorOllama(
                "No se puede contactar Ollama local. Compruebe 'systemctl status ollama' "
                "y 'ollama list'."
            ) from exc

    def responder(self, mensajes: list[dict[str, str]]) -> str:
        """Solicita una interpretación; lanza un error en lugar de inventar datos."""
        try:
            resp = requests.post(
                f"{self.base_url.rstrip('/')}/api/chat",
                json={"model": self.modelo, "messages": mensajes,
                      "stream": False, "options": {"temperature": 0.1, "num_predict": 450}},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            respuesta = str(resp.json().get("message", {}).get("content", "")).strip()
            if not respuesta:
                raise ErrorOllama("Ollama devolvió una respuesta vacía.")
            return respuesta
        except requests.Timeout as exc:
            raise ErrorOllama("La respuesta de Ollama excedió el tiempo de espera.") from exc
        except requests.ConnectionError as exc:
            raise ErrorOllama("Ollama no responde en la dirección configurada.") from exc
        except requests.HTTPError as exc:
            raise ErrorOllama(f"Ollama respondió con error HTTP: {exc}") from exc
        except ValueError as exc:
            raise ErrorOllama("Ollama devolvió JSON no válido.") from exc
