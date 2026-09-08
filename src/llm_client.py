"""
Minimal async client for OpenAI-compatible Chat Completions APIs.

Deliberately provider-agnostic: DeepSeek, Requesty (https://router.requesty.ai/v1,
routes to 600+ models — https://docs.requesty.ai), OpenAI itself, and most
other LLM gateways all speak the same shape — POST {base_url}/chat/completions
with a Bearer token, a `messages` array, and (usually) `response_format:
{"type": "json_object"}` for a guaranteed-JSON reply. `ai_prompt.py` picks
base_url/model/api_key per `provider` (see its module docstring); this class
doesn't care which provider it's actually talking to.

Hand-rolled on top of `aiohttp` (already a dependency, via okx_client.py)
rather than pulling in the `openai` SDK just for one optional strategy.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

import aiohttp

logger = logging.getLogger("okx_event_bot.llm_client")


class ChatAPIError(Exception):
    """Any network/HTTP/malformed-response failure talking to the LLM
    gateway. Callers (ai_prompt strategy) should catch this and degrade to
    "no signal" — an LLM being unavailable must never crash the trading
    engine.
    """


@dataclass
class ChatClientConfig:
    api_key: str
    base_url: str                      # e.g. https://router.requesty.ai/v1 or https://api.deepseek.com
    model: str                          # e.g. "anthropic/claude-haiku-4-5-20251001" (Requesty) or "deepseek-chat"
    timeout_sec: float = 20.0
    use_json_response_format: bool = True   # disable if your chosen model/gateway rejects this param


class ChatClient:
    def __init__(self, cfg: ChatClientConfig):
        self.cfg = cfg
        self._session: Optional[aiohttp.ClientSession] = None

    def _ensure_session(self) -> aiohttp.ClientSession:
        # Lazy creation: a strategy may hold this client for the bot's whole
        # lifetime but only actually need the session once the first signal
        # check happens inside the running event loop.
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.cfg.timeout_sec)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def chat_json(self, prompt: str, max_tokens: int = 300, temperature: float = 0.2) -> str:
        """Send one user message asking for a strict JSON reply. Returns the
        raw text of the model's answer (the *content* of its message, not
        the outer API envelope) — usually already valid JSON, but callers
        should still parse defensively since not every model/gateway
        combination honors response_format perfectly. Raises ChatAPIError
        on any failure.
        """
        if not self.cfg.api_key:
            raise ChatAPIError("no API key configured")

        session = self._ensure_session()
        url = f"{self.cfg.base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self.cfg.api_key}", "Content-Type": "application/json"}
        body = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self.cfg.use_json_response_format:
            body["response_format"] = {"type": "json_object"}

        try:
            async with session.post(url, headers=headers, json=body) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    raise ChatAPIError(f"HTTP {resp.status} from {self.cfg.base_url}: {text[:300]}")
        except aiohttp.ClientError as exc:
            raise ChatAPIError(f"network error calling {self.cfg.base_url}: {exc}") from exc
        except TimeoutError as exc:
            raise ChatAPIError(f"timeout calling {self.cfg.base_url}: {exc}") from exc

        try:
            payload = json.loads(text)
            choices = payload.get("choices") or []
            if not choices:
                raise ChatAPIError(f"no choices in response: {text[:300]}")
            return choices[0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise ChatAPIError(f"malformed response: {exc} — body: {text[:300]}") from exc
