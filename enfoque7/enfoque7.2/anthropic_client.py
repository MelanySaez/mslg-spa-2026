"""Cliente Anthropic con soporte para override de temperature por llamada.

Necesario para Self-Consistency (N llamadas con temperature>0.1).
Copia del flujo de enfoque7/anthropic_client.py con un parámetro extra.
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
        "  uv add anthropic python-dotenv"
    ) from exc


_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError(
                "ANTHROPIC_API_KEY no está definida. Crea .env en enfoque7/ "
                "o en la raíz del proyecto con:\n  ANTHROPIC_API_KEY=sk-ant-..."
            )
        _client = anthropic.Anthropic(
            api_key=config.ANTHROPIC_API_KEY,
            timeout=config.ANTHROPIC_TIMEOUT,
        )
    return _client


def translate(system_prompt: str, user_prompt: str, temperature: float = None) -> str:
    """Envía (system, user) a Claude. `temperature` overridea config.TEMPERATURE."""
    if temperature is None:
        temperature = config.TEMPERATURE

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
            resp = client.messages.create(
                model=config.ANTHROPIC_MODEL,
                max_tokens=config.MAX_TOKENS,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user_prompt}],
            )

            if resp.content and resp.content[0].type == "text":
                return resp.content[0].text
            return ""

        except anthropic.APIStatusError as e:
            last_error = e
            if e.status_code and 400 <= e.status_code < 500 and e.status_code != 429:
                raise RuntimeError(f"Anthropic API error {e.status_code}: {e}") from e
            wait = min(2 ** attempt, 30)
            logger.warning("Anthropic status error (attempt %d): %s — retry %ds", attempt, e, wait)
            time.sleep(wait)

        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            last_error = e
            wait = min(2 ** attempt, 30)
            logger.warning("Anthropic network error (attempt %d): %s — retry %ds", attempt, e, wait)
            time.sleep(wait)

    raise RuntimeError(
        f"Anthropic falló tras {config.ANTHROPIC_MAX_RETRIES} intentos: {last_error}"
    )
