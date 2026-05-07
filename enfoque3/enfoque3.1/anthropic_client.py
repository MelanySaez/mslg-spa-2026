"""Cliente para la API de Anthropic (Messages API)."""

import logging
import time

import config

logger = logging.getLogger(__name__)

try:
    import anthropic
except ImportError as exc:
    raise ImportError("Falta anthropic") from exc

_client = None

def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY no está definida.")
        _client = anthropic.Anthropic(
            api_key=config.ANTHROPIC_API_KEY,
            timeout=config.ANTHROPIC_TIMEOUT,
        )
    return _client

def translate(system_prompt: str, user_prompt: str) -> str:
    client = _get_client()
    if config.ENABLE_PROMPT_CACHE:
        system = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]
    else:
        system = system_prompt

    last_error = None
    for attempt in range(1, config.ANTHROPIC_MAX_RETRIES + 1):
        try:
            resp = client.messages.create(
                model=config.ANTHROPIC_MODEL,
                max_tokens=config.MAX_TOKENS,
                temperature=config.TEMPERATURE,
                system=system,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return resp.content[0].text if resp.content and resp.content[0].type == "text" else ""
        except anthropic.APIStatusError as e:
            last_error = e
            if e.status_code and 400 <= e.status_code < 500 and e.status_code != 429:
                raise RuntimeError(f"API error {e.status_code}") from e
            time.sleep(min(2 ** attempt, 30))
        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            last_error = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"Anthropic falló: {last_error}")
