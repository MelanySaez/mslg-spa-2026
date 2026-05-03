"""Cliente para la API de NVIDIA NIM via HTTP + SSE (google/gemma-4-31b-it).

Llama directamente al endpoint OpenAI-compatible de NVIDIA Build con
`requests`, parseando el stream Server-Sent Events línea a línea.

Modelo por defecto: `google/gemma-4-31b-it` (31B instruction-tuned). El flag
`chat_template_kwargs.enable_thinking` se pasa a nivel top-level del payload
(no dentro de `extra_body`) según el ejemplo oficial de la página del modelo.
Se desactiva por defecto (`NVIDIA_ENABLE_THINKING=false`) para que la salida
sea solo la traducción y no incluya el bloque de razonamiento.
"""

import json
import logging
import time

import requests

import config

logger = logging.getLogger(__name__)


def _parse_sse_stream(response) -> str:
    """Acumula `delta.content` de un stream SSE OpenAI-compat."""
    chunks = []
    for raw in response.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        choices = obj.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if content:
            chunks.append(content)
    return "".join(chunks)


def translate(system_prompt: str, user_prompt: str) -> str:
    """Envía (system, user) al endpoint NVIDIA NIM y retorna texto agregado."""
    if not config.NVIDIA_API_KEY:
        raise RuntimeError(
            "NVIDIA_API_KEY no definida. Añádela al .env o export NVIDIA_API_KEY=..."
        )

    headers = {
        "Authorization": f"Bearer {config.NVIDIA_API_KEY}",
        "Accept": "text/event-stream" if config.NVIDIA_STREAM else "application/json",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.NVIDIA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": config.MAX_TOKENS,
        "temperature": config.TEMPERATURE,
        "top_p": config.TOP_P,
        "stream": config.NVIDIA_STREAM,
        "chat_template_kwargs": {"enable_thinking": config.NVIDIA_ENABLE_THINKING},
    }

    last_error = None
    for attempt in range(1, config.NVIDIA_MAX_RETRIES + 1):
        try:
            logger.debug("NVIDIA NIM request (attempt %d/%d)",
                         attempt, config.NVIDIA_MAX_RETRIES)
            response = requests.post(
                config.NVIDIA_URL,
                headers=headers,
                json=payload,
                stream=config.NVIDIA_STREAM,
                timeout=config.NVIDIA_TIMEOUT,
            )
            response.raise_for_status()

            if config.NVIDIA_STREAM:
                content = _parse_sse_stream(response)
            else:
                data = response.json()
                content = data["choices"][0]["message"]["content"]

            logger.debug("NVIDIA NIM response: %s", content[:200])
            return content

        except (requests.RequestException, KeyError, IndexError,
                json.JSONDecodeError) as e:
            last_error = e
            wait = 2 ** attempt
            logger.warning(
                "NVIDIA NIM error (attempt %d): %s — retrying in %ds",
                attempt, e, wait,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"NVIDIA NIM falló tras {config.NVIDIA_MAX_RETRIES} intentos: {last_error}"
    )
