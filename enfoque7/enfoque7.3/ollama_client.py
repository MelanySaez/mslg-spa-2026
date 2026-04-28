"""Cliente para la API de Ollama (chat completions).

A diferencia del cliente de enfoque6 (que envía un único prompt user), este
cliente acepta `(system_prompt, user_prompt)` para coincidir con la firma de
`anthropic_client.translate` de enfoque7.x. El backend Ollama soporta el rol
`system` nativamente, así que el bloque estático (reglas LSM + ejemplos) se
manda en un mensaje system y la oración a traducir va como user.

`think=False` desactiva el modo de razonamiento explícito de deepseek-r1, que
tiende a alucinar en tareas de traducción corta determinista.
"""

import json
import logging
import time

import requests

import config

logger = logging.getLogger(__name__)


def translate(system_prompt: str, user_prompt: str) -> str:
    """Envía (system, user) a Ollama y retorna la respuesta del modelo."""
    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": config.TEMPERATURE,
            "num_predict": config.MAX_TOKENS,
        },
    }

    last_error = None
    for attempt in range(1, config.OLLAMA_MAX_RETRIES + 1):
        try:
            logger.debug("Ollama request (attempt %d/%d)",
                         attempt, config.OLLAMA_MAX_RETRIES)
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
                "Ollama error (attempt %d): %s — retrying in %ds",
                attempt, e, wait,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"Ollama falló tras {config.OLLAMA_MAX_RETRIES} intentos: {last_error}"
    )
