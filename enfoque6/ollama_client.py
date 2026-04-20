"""Cliente para la API de Ollama (chat completions)."""

import json
import logging
import time

import requests

import config

logger = logging.getLogger(__name__)


def translate(prompt: str) -> str:
    """
    Envía un prompt a Ollama y retorna la respuesta del modelo.

    Returns:
        Texto de respuesta del modelo (str).

    Raises:
        RuntimeError: si falla tras todos los reintentos.
    """
    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": config.TEMPERATURE,
            "num_predict": config.MAX_TOKENS,
        },
    }

    last_error = None
    for attempt in range(1, config.OLLAMA_MAX_RETRIES + 1):
        try:
            logger.debug("Ollama request (attempt %d/%d)", attempt, config.OLLAMA_MAX_RETRIES)
            resp = requests.post(
                config.OLLAMA_URL,
                json=payload,
                timeout=config.OLLAMA_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["message"]["content"]
            logger.debug("Ollama response: %s", content[:200])
            return content

        except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
            last_error = e
            wait = 2 ** attempt
            logger.warning(
                "Ollama error (attempt %d): %s — retrying in %ds", attempt, e, wait
            )
            time.sleep(wait)

    raise RuntimeError(
        f"Ollama falló tras {config.OLLAMA_MAX_RETRIES} intentos: {last_error}"
    )
