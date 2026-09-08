"""
Minimal async OKX v5 REST client.

Covers exactly what this bot needs:
  * public, unauthenticated market data (ticker/orderbook for BTC-USDT spot,
    instruments/tickers for instType=EVENTS) — these are stable, well
    documented v5 endpoints.
  * authenticated account/order calls, used only if you opt into
    `execute_on_okx=true` (see README) to additionally route real orders
    through your OKX Demo Trading account for extra realism/audit trail.

Auth scheme (https://www.okx.com/docs-v5/en/#overview-rest-authentication):
  sign = base64( HMAC_SHA256(secret, timestamp + method + requestPath + body) )
  headers: OK-ACCESS-KEY, OK-ACCESS-SIGN, OK-ACCESS-TIMESTAMP,
           OK-ACCESS-PASSPHRASE, and x-simulated-trading: 1 for Demo Trading.

NOTE on Event Contracts (instType=EVENTS): OKX's public v5 docs are thin on
this newer product, so the endpoints/fields below were verified against
OKX's own open-source reference implementation,
https://github.com/okx/agent-trade-kit (packages/core/src/tools/event-trade.ts
and event-helpers.ts) rather than guessed. Two things worth knowing:
  * The "public" event-contract endpoints (series/events/markets) are
    called as AUTHENTICATED (signed) requests by OKX's own tooling despite
    the /public/ path — so reading Event Contract quotes needs a valid API
    key, unlike ordinary spot market data.
  * A market has exactly ONE instId per (series, expiry[, strike]) — there
    is no separate "UP instrument" vs "DOWN instrument". Direction is
    chosen at order time via the `outcome` field ("yes"/"no"); `px`
    (0.01-0.99) is the market-implied probability of the primary side
    (UP for price_up_down series, YES for price_above/hit).
"""
from __future__ import annotations

import asyncio
import base64
import hmac
import json
import logging
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Optional

import aiohttp

logger = logging.getLogger("okx_event_bot.okx_client")


class OKXAPIError(Exception):
    """Raised when OKX responds with a non-'0' business error code."""

    def __init__(self, code: str, msg: str, path: str):
        self.code = code
        self.msg = msg
        self.path = path
        super().__init__(f"OKX API error {code} on {path}: {msg}")


class OKXNetworkError(Exception):
    """Raised after all retries are exhausted on a network/timeout failure."""


@dataclass
class OKXClientConfig:
    base_url: str
    api_key: str = ""
    api_secret: str = ""
    api_passphrase: str = ""
    demo_trading: bool = True
    timeout_sec: float = 10.0
    max_retries: int = 5
    retry_backoff_base_sec: float = 1.5


class OKXClient:
    def __init__(self, cfg: OKXClientConfig):
        self.cfg = cfg
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "OKXClient":
        timeout = aiohttp.ClientTimeout(total=self.cfg.timeout_sec)
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, *exc) -> None:
        if self._session:
            await self._session.close()

    # -- signing -----------------------------------------------------------------
    def _sign(self, timestamp: str, method: str, request_path: str, body: str) -> str:
        message = f"{timestamp}{method}{request_path}{body}"
        mac = hmac.new(self.cfg.api_secret.encode(), message.encode(), sha256)
        return base64.b64encode(mac.digest()).decode()

    def _headers(self, method: str, request_path: str, body: str, auth: bool) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.cfg.demo_trading:
            headers["x-simulated-trading"] = "1"
        if auth:
            timestamp = (
                __import__("datetime")
                .datetime.utcnow()
                .isoformat(timespec="milliseconds")
                + "Z"
            )
            headers.update(
                {
                    "OK-ACCESS-KEY": self.cfg.api_key,
                    "OK-ACCESS-SIGN": self._sign(timestamp, method, request_path, body),
                    "OK-ACCESS-TIMESTAMP": timestamp,
                    "OK-ACCESS-PASSPHRASE": self.cfg.api_passphrase,
                }
            )
        return headers

    # -- core request w/ retry ----------------------------------------------------
    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
        body: Optional[dict] = None,
        auth: bool = False,
    ) -> Any:
        assert self._session is not None, "use 'async with OKXClient(...)'"

        query = ""
        if params:
            clean = {k: v for k, v in params.items() if v is not None}
            if clean:
                query = "?" + "&".join(f"{k}={v}" for k, v in clean.items())
        request_path = f"{path}{query}"
        body_str = json.dumps(body, separators=(",", ":")) if body else ""
        url = f"{self.cfg.base_url}{request_path}"

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                headers = self._headers(method, request_path, body_str, auth)
                async with self._session.request(
                    method, url, headers=headers, data=body_str or None
                ) as resp:
                    text = await resp.text()
                    if resp.status >= 500 or resp.status == 429:
                        raise OKXNetworkError(f"HTTP {resp.status} on {path}: {text[:200]}")
                    if resp.status >= 400:
                        # Client error - not worth retrying (bad params/auth).
                        try:
                            payload = json.loads(text)
                            raise OKXAPIError(payload.get("code", str(resp.status)), payload.get("msg", text), path)
                        except json.JSONDecodeError:
                            raise OKXAPIError(str(resp.status), text[:200], path)

                    payload = json.loads(text)
                    code = payload.get("code", "0")
                    if code not in ("0", 0):
                        raise OKXAPIError(str(code), payload.get("msg", ""), path)
                    return payload.get("data", [])

            except OKXAPIError:
                raise  # business errors are not retried
            except (aiohttp.ClientError, asyncio.TimeoutError, OKXNetworkError) as exc:
                last_exc = exc
                backoff = self.cfg.retry_backoff_base_sec * (2 ** (attempt - 1))
                logger.warning(
                    "OKX request failed (attempt %d/%d) %s %s: %s — retrying in %.1fs",
                    attempt, self.cfg.max_retries, method, path, exc, backoff,
                )
                if attempt < self.cfg.max_retries:
                    await asyncio.sleep(backoff)

        raise OKXNetworkError(f"Exhausted retries on {method} {path}: {last_exc}")

    # -- public market data --------------------------------------------------------
    async def get_ticker(self, inst_id: str) -> list[dict]:
        return await self._request("GET", "/api/v5/market/ticker", {"instId": inst_id})

    async def get_orderbook(self, inst_id: str, sz: int = 20) -> list[dict]:
        return await self._request("GET", "/api/v5/market/books", {"instId": inst_id, "sz": sz})

    async def get_candles(self, inst_id: str, bar: str = "1m", limit: int = 100) -> list[dict]:
        return await self._request(
            "GET", "/api/v5/market/candles", {"instId": inst_id, "bar": bar, "limit": limit}
        )

    async def get_index_ticker(self, inst_id: str) -> list[dict]:
        """Index price (e.g. BTC-USDT) — public/unauthenticated."""
        return await self._request("GET", "/api/v5/market/index-tickers", {"instId": inst_id})

    async def get_funding_rate(self, inst_id: str = "BTC-USDT-SWAP") -> list[dict]:
        """Current + predicted next funding rate for a perpetual swap —
        standard, stable, public/unauthenticated v5 endpoint (unlike the
        EVENTS ones above). Response fields: fundingRate, nextFundingRate,
        fundingTime."""
        return await self._request("GET", "/api/v5/public/funding-rate", {"instId": inst_id})

    # -- event contracts (instType=EVENTS) ------------------------------------------
    # IMPORTANT: unlike the rest of v5 public market data, OKX's own reference
    # implementation calls these three endpoints as AUTHENTICATED (signed)
    # requests even though the path starts with /public/ — verified against
    # OKX's own open-source agent-trade-kit (github.com/okx/agent-trade-kit,
    # packages/core/src/tools/event-trade.ts). Reading Event Contract quotes
    # therefore requires a valid API key even for the "read-only" calls below.
    async def get_event_series(self, series_id: Optional[str] = None) -> list[dict]:
        return await self._request(
            "GET", "/api/v5/public/event-contract/series",
            {"seriesId": series_id}, auth=True,
        )

    async def get_event_events(
        self, series_id: str, event_id: Optional[str] = None,
        state: Optional[str] = None, limit: Optional[int] = None,
    ) -> list[dict]:
        return await self._request(
            "GET", "/api/v5/public/event-contract/events",
            {"seriesId": series_id, "eventId": event_id, "state": state, "limit": limit},
            auth=True,
        )

    async def get_event_markets(
        self, series_id: str, event_id: Optional[str] = None, inst_id: Optional[str] = None,
        state: Optional[str] = None, limit: Optional[int] = None,
    ) -> list[dict]:
        """state: preopen | live | settling | expired. Use state="expired" to
        read the settlement `outcome` field ("0"=pending, "1"=UP/YES won,
        "2"=DOWN/NO won) — the authoritative way to resolve a trade."""
        return await self._request(
            "GET", "/api/v5/public/event-contract/markets",
            {"seriesId": series_id, "eventId": event_id, "instId": inst_id, "state": state, "limit": limit},
            auth=True,
        )

    async def place_event_order(
        self,
        inst_id: str,
        side: str,            # "buy" | "sell"
        outcome: str,          # "yes" | "no"  (UP/YES -> "yes", DOWN/NO -> "no")
        sz: str,                # limit/post_only: number of contracts; market: quote-currency amount
        px: Optional[str] = None,   # required for ordType="limit"; omit for market
        ord_type: str = "market",
    ) -> list[dict]:
        body = {
            "instId": inst_id,
            "tdMode": "isolated",     # always "isolated" for event contracts
            "side": side,
            "outcome": outcome,
            "ordType": ord_type,
            "sz": sz,
        }
        if px is not None:
            body["px"] = px
        if ord_type != "post_only":
            body["speedBump"] = "1"    # required by the exchange for non-post_only orders
        return await self._request("POST", "/api/v5/trade/order", body=body, auth=True)

    # -- authenticated account/order endpoints ----------------------------------------
    async def get_balance(self) -> list[dict]:
        return await self._request("GET", "/api/v5/account/balance", auth=True)

    async def get_positions(self, inst_type: str = "EVENTS") -> list[dict]:
        return await self._request("GET", "/api/v5/account/positions", {"instType": inst_type}, auth=True)

    async def get_order(self, inst_id: str, ord_id: str) -> list[dict]:
        return await self._request(
            "GET", "/api/v5/trade/order", {"instId": inst_id, "ordId": ord_id}, auth=True
        )
