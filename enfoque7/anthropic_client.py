"""Cliente para la API de Anthropic (Messages API).

Aprovecha prompt caching: todo el bloque estático (reglas LSM + glosario +
ejemplos + CoT) va en `system` con cache_control=ephemeral. La oración a
traducir va en el mensaje de usuario. Se pagan tokens completos la primera
vez de cada experimento; los siguientes requests leen del caché (≈10 %
del costo de input).
"""

import logging
import time

import config

logger = logging.getLogger(__name__)

try:
    import anthropic
except ImportError as exc:  # noqa: BLE001
    raise ImportError(
        "Falta la dependencia 'anthropic'. Instala con:\n"
        "  uv add anthropic python-dotenv\n"
        "o bien:\n"
        "  pip install anthropic python-dotenv"
    ) from exc


_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError(
                "ANTHROPIC_API_KEY no está definida. Crea un archivo .env "
                "en enfoque7/ o en la raíz del proyecto con:\n"
                "  ANTHROPIC_API_KEY=sk-ant-..."
            )
        _client = anthropic.Anthropic(
            api_key=config.ANTHROPIC_API_KEY,
            timeout=config.ANTHROPIC_TIMEOUT,
        )
    return _client


def translate(system_prompt: str, user_prompt: str) -> str:
    """Envía (system, user) a Claude y retorna la respuesta en texto.

    Args:
        system_prompt: Contenido estático (reglas, glosario, ejemplos).
                       Se cachea si ENABLE_PROMPT_CACHE y >= 1024 tokens.
        user_prompt:   Contenido variable (oración SPA a traducir).

    Raises:
        RuntimeError: si falla tras todos los reintentos.
    """
    client = _get_client()

    if config.ENABLE_PROMPT_CACHE:
        system = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    else:
        system = system_prompt

    last_error = None
    for attempt in range(1, config.ANTHROPIC_MAX_RETRIES + 1):
        try:
            logger.debug(
                "Anthropic request (attempt %d/%d)",
                attempt, config.ANTHROPIC_MAX_RETRIES,
            )
            resp = client.messages.create(
                model=config.ANTHROPIC_MODEL,
                max_tokens=config.MAX_TOKENS,
                temperature=config.TEMPERATURE,
                system=system,
                messages=[{"role": "user", "content": user_prompt}],
            )

            if resp.content and resp.content[0].type == "text":
                text = resp.content[0].text
            else:
                text = ""

            usage = getattr(resp, "usage", None)
            if usage is not None:
                logger.debug(
                    "usage: input=%s cache_read=%s cache_creation=%s output=%s",
                    getattr(usage, "input_tokens", None),
                    getattr(usage, "cache_read_input_tokens", None),
                    getattr(usage, "cache_creation_input_tokens", None),
                    getattr(usage, "output_tokens", None),
                )

            return text

        except anthropic.APIStatusError as e:
            last_error = e
            # 4xx no retriable salvo 429
            if e.status_code and 400 <= e.status_code < 500 and e.status_code != 429:
                raise RuntimeError(f"Anthropic API error {e.status_code}: {e}") from e
            wait = min(2 ** attempt, 30)
            logger.warning(
                "Anthropic status error (attempt %d): %s — retry en %ds",
                attempt, e, wait,
            )
            time.sleep(wait)

        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            last_error = e
            wait = min(2 ** attempt, 30)
            logger.warning(
                "Anthropic network error (attempt %d): %s — retry en %ds",
                attempt, e, wait,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"Anthropic falló tras {config.ANTHROPIC_MAX_RETRIES} intentos: {last_error}"
    )
