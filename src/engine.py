"""
The strategy engine: ties market data, timing, strategies, wallets and
persistence together into one poll loop. Contains no OKX- or mock-specific
logic — it only talks to the `MarketDataProvider` interface, so swapping
mock <-> live OKX data is a one-line change in run.py.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

from .config import AppConfig
from .market_data import MarketDataProvider
from .metrics import ComboStats, Leaderboard, build_combo_stats, build_leaderboard
from .models import Direction, Trade
from .storage import Storage
from .strategies import STRATEGY_REGISTRY, BaseStrategy, StrategyContext
from .timing import EntryWindowManager
from .wallet import VirtualWallet

logger = logging.getLogger("okx_event_bot.engine")


@dataclass
class ActivityEvent:
    """One line of the live per-strategy activity feed (Dashboard's
    "Логи" tab) — every strategy evaluation at a due entry-window
    checkpoint produces exactly one of these, plus one more when a trade
    it opened later settles. In-memory only (a ring buffer on Engine, not
    persisted to SQLite) — this is a live-observability feed, not part of
    the durable trade history (that's data/bot.db, unaffected by this).
    `kind` drives the dashboard's color-coding:
      no_signal | opened | rejected | won | lost | unresolved
    """
    id: int
    ts: float
    strategy: str
    series_id: str
    window_min: Optional[int]
    kind: str
    message: str


@dataclass
class EngineSnapshot:
    """Immutable-ish view of engine state for dashboards. Built fresh each
    tick so a dashboard reading it can never observe a half-updated state."""
    wallets: dict[str, VirtualWallet]
    combo_stats: dict[tuple[str, int], ComboStats]
    leaderboard: Leaderboard
    underlying_price: float | None
    active_markets: dict = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)


class Engine:
    def __init__(self, cfg: AppConfig, provider: MarketDataProvider, storage: Storage):
        self.cfg = cfg
        self.provider = provider
        self.storage = storage
        self.timing = EntryWindowManager()

        self.wallets: dict[str, VirtualWallet] = {}
        self.strategy_instances: dict[str, BaseStrategy] = {}
        self._settlement_attempts: dict[str, int] = {}
        self._last_settlement_check: dict[str, float] = {}
        self._last_snapshot_write = 0.0
        self._running = False

        # Per-series (not per-strategy/per-trade) bookkeeping for
        # prior_window_momentum — see _update_previous_outcomes(). Tracked
        # independently of any strategy's open trades, since the whole
        # point is to know a window's outcome even when nobody traded it.
        self._last_inst_id: dict[str, str] = {}
        self._previous_outcome: dict[str, Direction] = {}
        self._pending_outcome_inst_id: dict[str, str] = {}
        self._pending_outcome_attempts: dict[str, int] = {}
        self._last_outcome_check: dict[str, float] = {}

        # Live activity feed for the dashboard's "Логи" tab — see
        # ActivityEvent's docstring. maxlen bounds memory for a
        # long-running process; the dashboard polls incrementally via
        # activity_since(), so trimming here just caps how far back a
        # freshly-opened tab (or one that was closed a while) can see.
        self._activity: Deque[ActivityEvent] = deque(maxlen=1000)
        self._activity_seq = 0

        self._build_wallets_and_strategies()

    def _build_wallets_and_strategies(self) -> None:
        enabled_names = self.cfg.enabled_strategy_names()
        if not enabled_names:
            raise RuntimeError("No strategies enabled in config.yaml — nothing to run.")
        # One query up front rather than one per strategy — cheap, and
        # keeps _build_wallets_and_strategies() the single place that
        # decides "restore vs fresh start" for every wallet at once.
        saved_wallets = self.storage.load_wallets()
        for s_cfg in self.cfg.strategies:
            if not s_cfg.enabled:
                continue
            # Each strategy gets its OWN independent deposit (config.yaml ->
            # strategies.<name>.deposit_usd, default $100) — not a slice of
            # one shared pool. That's what makes a fair head-to-head
            # comparison possible: every strategy is judged on the same
            # starting bankroll, not a fraction that shrinks as you enable more.
            self.wallets[s_cfg.name] = self._restore_or_create_wallet(s_cfg, saved_wallets.get(s_cfg.name))

            strat_cls = STRATEGY_REGISTRY.get(s_cfg.name)
            if strat_cls is None:
                raise RuntimeError(f"Unknown strategy '{s_cfg.name}' in config.yaml")
            merged_config = dict(s_cfg.extra)
            self.strategy_instances[s_cfg.name] = strat_cls(config=merged_config)
            logger.info(
                "Strategy '%s' ready: deposit=$%.2f windows=%s max_px=%.2f stake_frac=%.2f",
                s_cfg.name, s_cfg.deposit_usd, s_cfg.entry_windows_min, s_cfg.max_coefficient, s_cfg.stake_fraction,
            )

    @staticmethod
    def _restore_or_create_wallet(s_cfg, saved: Optional[dict]) -> VirtualWallet:
        """A fresh VirtualWallet(deposit_usd) if `saved` is None (first-ever
        launch for this strategy, or one just reset) — otherwise resumes
        the balance write_snapshot() persisted for it, so a redeploy/crash/
        restart doesn't silently reset every strategy back to its starting
        deposit while the Analytics tab (backed by the separately, every-
        tick-persisted trades table) keeps remembering the full history.

        initial_balance is restored too (not re-read from config) so
        net_pnl/equity keep meaning "profit since this strategy's actual
        first run", even across a config.yaml edit to deposit_usd later —
        that's what the Reset button/`--reset-strategy` are for instead.

        Any `reserved` capital (stake locked in trades that were still
        open at the last snapshot) is folded back into balance rather than
        restored as reserved: those specific Trade objects only ever lived
        in memory and are gone after a restart, so nothing will ever
        settle them and return that money on its own — leaving it in
        `reserved` would just strand it there permanently. This is the
        same "give the stake back, we can't confirm the outcome" logic
        VirtualWallet.mark_unresolved() already uses for a settlement that
        times out."""
        if saved is None:
            return VirtualWallet(strategy=s_cfg.name, initial_balance=s_cfg.deposit_usd)
        wallet = VirtualWallet(strategy=s_cfg.name, initial_balance=saved["initial_balance"])
        wallet.balance = saved["balance"] + saved["reserved"]
        if saved["reserved"]:
            logger.warning(
                "Strategy '%s': restarted with $%.2f still reserved in-flight at the last "
                "snapshot — refunded to balance (its open trade(s) can't be resumed across a restart).",
                s_cfg.name, saved["reserved"],
            )
        return wallet

    # -- lifecycle ---------------------------------------------------------------
    def stop(self) -> None:
        self._running = False

    async def aclose_strategies(self) -> None:
        """Release resources held by the CURRENT strategy instances (e.g.
        ai_prompt's HTTP session). Call once during final shutdown."""
        await self._aclose_all(self.strategy_instances.values())

    def _close_strategies_soon(self, strategies) -> None:
        """Fire-and-forget cleanup for strategy instances `reset()`/
        `reset_strategy()` just replaced (e.g. an ai_prompt instance with
        an open aiohttp session). Those stay synchronous, so this just
        schedules the cleanup on the running loop rather than awaiting it
        inline."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return  # no running loop (e.g. a sync unit test) — nothing we can schedule
        asyncio.create_task(self._aclose_all(strategies))

    @staticmethod
    async def _aclose_all(strategies) -> None:
        for strategy in strategies:
            try:
                await strategy.aclose()
            except Exception:  # never let one strategy's cleanup blow up shutdown/reset
                logger.exception("Error closing strategy '%s'", strategy.name)

    def reset(self) -> None:
        """Wipe all wallets/trades/timing state back to a fresh start and
        clear persisted history (the trades/wallets tables in data/bot.db).
        Safe to call while the engine is running — it's synchronous and
        holds no `await` points, so it can't race a concurrent `tick()`
        under asyncio's single-threaded cooperative scheduling. Used by the
        web dashboard's Reset button (`POST /api/reset`) and
        `run.py --reset-data`.
        """
        old_strategies = list(self.strategy_instances.values())
        # storage.reset() MUST run before _build_wallets_and_strategies():
        # that method restores each wallet's balance from storage.
        # load_wallets() when a saved row exists (see
        # _restore_or_create_wallet) — wiping storage first is what makes
        # "fresh wallets" actually mean fresh, instead of it immediately
        # reloading the very balances this call is meant to erase.
        self.storage.reset()
        self._build_wallets_and_strategies()  # fresh wallets AND fresh strategy instances
        self._close_strategies_soon(old_strategies)  # e.g. ai_prompt's old HTTP session
        self._settlement_attempts.clear()
        self._last_settlement_check.clear()
        self._last_inst_id.clear()
        self._previous_outcome.clear()
        self._pending_outcome_inst_id.clear()
        self._pending_outcome_attempts.clear()
        self._last_outcome_check.clear()
        self._activity.clear()
        self._log_activity("*", "*", None, "no_signal", "База сброшена — журнал активности очищен")
        self.timing = EntryWindowManager()
        self._last_snapshot_write = 0.0
        logger.warning("Engine reset: all wallets/trades/timing cleared, storage wiped.")

    def reset_strategy(self, name: str) -> None:
        """Reset just ONE strategy's wallet/trades/timing-history — every
        other strategy's wallet, trades and persisted rows are left
        completely untouched. Used by the Settings tab's per-strategy
        Reset button. Raises ValueError for an unknown or disabled
        strategy name (there's nothing to reset for a strategy with no
        wallet)."""
        s_cfg = next((s for s in self.cfg.strategies if s.name == name), None)
        if s_cfg is None:
            raise ValueError(f"unknown strategy '{name}'")
        if not s_cfg.enabled:
            raise ValueError(f"strategy '{name}' is not enabled")

        old_strategy = self.strategy_instances.get(name)
        self.wallets[name] = VirtualWallet(strategy=name, initial_balance=s_cfg.deposit_usd)
        strat_cls = STRATEGY_REGISTRY[name]
        self.strategy_instances[name] = strat_cls(config=dict(s_cfg.extra))
        if old_strategy is not None:
            self._close_strategies_soon([old_strategy])

        self.timing.reset_strategy(name)
        self.storage.reset_strategy(name)
        logger.warning("Engine reset for strategy '%s' only — other strategies untouched.", name)

    def update_strategy_settings(self, updates: dict[str, dict]) -> None:
        """Apply partial per-strategy config changes — any of `enabled`,
        `deposit_usd`, `stake_fraction`, `max_coefficient`,
        `entry_windows_min`, `extra` (a dict merged into the strategy's own
        extra config) — then reset. A wallet/strategy mid-trade has no
        coherent way to be "resized" or have its behavior swapped in
        place, so applying new settings always starts every strategy
        fresh. Used by the web dashboard's Settings tab (`POST
        /api/settings`).

        Everything is validated BEFORE anything is applied, so a bad
        request never leaves config half-updated. Unknown strategy names
        are ignored (logged), not fatal — a stale/typo'd entry shouldn't
        block updating the rest.
        """
        by_name = {s_cfg.name: s_cfg for s_cfg in self.cfg.strategies}

        resulting_enabled = {s.name for s in self.cfg.strategies if s.enabled}
        for name, fields in updates.items():
            if name not in by_name:
                logger.warning("update_strategy_settings: ignoring unknown strategy '%s'", name)
                continue
            if "enabled" in fields:
                (resulting_enabled.add if fields["enabled"] else resulting_enabled.discard)(name)
            if "deposit_usd" in fields and fields["deposit_usd"] <= 0:
                raise ValueError(f"deposit_usd for '{name}' must be positive")
            if "stake_fraction" in fields and not (0 < fields["stake_fraction"] <= 1):
                raise ValueError(f"stake_fraction for '{name}' must be in (0, 1]")
            if "max_coefficient" in fields and not (0 < fields["max_coefficient"] <= 1):
                raise ValueError(f"max_coefficient for '{name}' must be in (0, 1]")
            if "entry_windows_min" in fields:
                windows = fields["entry_windows_min"]
                if not windows or any(w <= 0 for w in windows):
                    raise ValueError(f"entry_windows_min for '{name}' must be a non-empty list of positive numbers")
        if not resulting_enabled:
            raise ValueError("at least one strategy must stay enabled")

        for name, fields in updates.items():
            s_cfg = by_name.get(name)
            if s_cfg is None:
                continue
            if "enabled" in fields:
                s_cfg.enabled = bool(fields["enabled"])
            if "deposit_usd" in fields:
                s_cfg.deposit_usd = float(fields["deposit_usd"])
            if "stake_fraction" in fields:
                s_cfg.stake_fraction = float(fields["stake_fraction"])
            if "max_coefficient" in fields:
                s_cfg.max_coefficient = float(fields["max_coefficient"])
            if "entry_windows_min" in fields:
                s_cfg.entry_windows_min = [float(w) for w in fields["entry_windows_min"]]
            if fields.get("extra"):
                s_cfg.extra.update(fields["extra"])

        logger.warning("Strategy settings changed for %s — resetting.", list(updates.keys()))
        self.reset()

    async def run_forever(self) -> None:
        self._running = True
        while self._running:
            tick_start = time.time()
            try:
                await self.tick()
            except Exception:
                logger.exception("Unhandled error in engine tick — continuing (state untouched).")
            elapsed = time.time() - tick_start
            sleep_for = max(0.1, self.cfg.okx.poll_interval_sec - elapsed)
            await asyncio.sleep(sleep_for)

    async def tick(self) -> None:
        await self.provider.refresh()
        await self._update_previous_outcomes()
        await self._open_due_trades()
        await self._settle_expired_trades()
        self.timing.prune()
        self._maybe_persist()

    # -- prior-window outcome tracking (prior_window_momentum) -----------------------
    async def _update_previous_outcomes(self) -> None:
        """The moment a series rolls over to a new instId, look up the
        JUST-CLOSED window's settlement outcome — regardless of whether
        any strategy actually held a position in it. _settle_expired_trades
        only ever learns an outcome for a window some strategy traded;
        prior_window_momentum needs to know every window's outcome to bet
        on it, so this tracks it independently at the series level.

        Retries (throttled by settlement_poll_interval_sec, capped at
        settlement_poll_attempts — the same knobs _settle_expired_trades
        uses, since it's the same "OKX hasn't published outcome yet"
        situation) if the first lookup right at rollover comes back None.
        Giving up just leaves the previous reading in place rather than
        clearing it to None — a strategy checking two windows back is
        better than one that suddenly stops betting at all because of one
        transient lookup failure.
        """
        now = time.time()
        poll_interval = self.cfg.okx.settlement_poll_interval_sec
        active_markets = self.provider.active_markets()

        for series_id in self.cfg.okx.series_ids:
            market = active_markets.get(series_id)
            if market is None or not market.inst_id:
                continue

            pending_inst_id = self._pending_outcome_inst_id.get(series_id)
            if pending_inst_id is not None:
                last_check = self._last_outcome_check.get(series_id, 0.0)
                if now - last_check >= poll_interval:
                    self._last_outcome_check[series_id] = now
                    outcome = await self.provider.check_settlement(series_id, pending_inst_id)
                    if outcome is not None:
                        self._previous_outcome[series_id] = outcome
                        self._pending_outcome_inst_id.pop(series_id, None)
                        self._pending_outcome_attempts.pop(series_id, None)
                    else:
                        attempts = self._pending_outcome_attempts.get(series_id, 0) + 1
                        self._pending_outcome_attempts[series_id] = attempts
                        if attempts >= self.cfg.okx.settlement_poll_attempts:
                            logger.warning(
                                "prior_window_momentum: giving up on %s/%s's outcome after %d attempts — "
                                "keeping the previous reading (if any) until the next rollover.",
                                series_id, pending_inst_id, attempts,
                            )
                            self._pending_outcome_inst_id.pop(series_id, None)
                            self._pending_outcome_attempts.pop(series_id, None)

            last_inst_id = self._last_inst_id.get(series_id)
            if last_inst_id is not None and last_inst_id != market.inst_id:
                # Rollover detected this tick — try immediately (most
                # settlements are already known by the time a new window
                # opens), falling back to the retry loop above if not.
                outcome = await self.provider.check_settlement(series_id, last_inst_id)
                if outcome is not None:
                    self._previous_outcome[series_id] = outcome
                else:
                    self._pending_outcome_inst_id[series_id] = last_inst_id
                    self._pending_outcome_attempts[series_id] = 0
                    self._last_outcome_check[series_id] = now
            self._last_inst_id[series_id] = market.inst_id

    # -- opening trades ------------------------------------------------------------
    async def _open_due_trades(self) -> None:
        now = time.time()
        active_markets = self.provider.active_markets()

        for s_cfg in self.cfg.strategies:
            if not s_cfg.enabled:
                continue
            strategy = self.strategy_instances[s_cfg.name]
            wallet = self.wallets[s_cfg.name]

            for series_id in self.cfg.okx.series_ids:
                market = active_markets.get(series_id)
                if market is None:
                    continue
                remaining = market.remaining_sec(now)
                if remaining <= 0:
                    continue

                due_windows = self.timing.due_windows(
                    series_id, market.expiry_ts, s_cfg.name, remaining, s_cfg.entry_windows_min
                )
                for window_min in due_windows:
                    ctx = StrategyContext(
                        price_history=self.provider.btc_price_history(),
                        orderbook=self.provider.btc_orderbook(),
                        remaining_sec=remaining, window_min=window_min,
                        market=market, funding_rate=self.provider.funding_rate(),
                        previous_outcome=self._previous_outcome.get(series_id),
                    )
                    signal = await strategy.evaluate(ctx)
                    if signal is None:
                        logger.debug("%s: no signal at %dm-to-expiry for %s", s_cfg.name, window_min, series_id)
                        self._log_activity(s_cfg.name, series_id, window_min, "no_signal", "нет сигнала")
                        continue

                    inst_id = market.inst_id
                    if not inst_id:
                        logger.warning(
                            "%s wants to bet %s on %s but that market has no live "
                            "quote yet — skipping.",
                            s_cfg.name, signal.direction.value, series_id,
                        )
                        self._log_activity(
                            s_cfg.name, series_id, window_min, "rejected",
                            f"{signal.direction.value.upper()} — нет живой котировки ({signal.reason})",
                        )
                        continue

                    stake = round(wallet.balance * s_cfg.stake_fraction, 4)
                    if stake <= 0.01:
                        logger.warning("%s: wallet balance too low to stake ($%.2f) — skipping.", s_cfg.name, wallet.balance)
                        self._log_activity(
                            s_cfg.name, series_id, window_min, "rejected",
                            f"баланс слишком мал для стейка (${wallet.balance:.2f})",
                        )
                        continue

                    # Honest expected fill for actually committing THIS
                    # stake right now — walks the real order book for UP
                    # (verified on live data: routinely 40-1000%+ away from
                    # the naive last/mid price on these thin books), falls
                    # back to top-of-book for DOWN (no public depth to walk
                    # there) or to the naive price entirely if no book data
                    # came back this tick. See EventMarket.fill_price_for.
                    price = market.fill_price_for(signal.direction, stake)
                    if price is None:
                        logger.warning(
                            "%s wants to bet %s on %s but that market has no live "
                            "quote yet — skipping.",
                            s_cfg.name, signal.direction.value, series_id,
                        )
                        self._log_activity(
                            s_cfg.name, series_id, window_min, "rejected",
                            f"{signal.direction.value.upper()} — нет цены исполнения ({signal.reason})",
                        )
                        continue
                    if price > s_cfg.max_coefficient:
                        logger.debug(
                            "%s: signal on %s rejected, price %.3f > max_coefficient %.3f",
                            s_cfg.name, series_id, price, s_cfg.max_coefficient,
                        )
                        self._log_activity(
                            s_cfg.name, series_id, window_min, "rejected",
                            f"{signal.direction.value.upper()} @ {price:.3f} > лимит {s_cfg.max_coefficient:.2f} ({signal.reason})",
                        )
                        continue

                    # Paper-trading analog of the "max slippage" guard a
                    # real OKX order lets you set before it refuses to
                    # fill: reject if the honest price is more than
                    # max_slippage_pct WORSE than the naive quote, even if
                    # it's still under max_coefficient's absolute ceiling.
                    # None (default/unset) = no limit, old behavior.
                    if s_cfg.max_slippage_pct is not None:
                        naive_price = market.price_for(signal.direction)
                        if naive_price:
                            slippage_pct = (price - naive_price) / naive_price * 100
                            if slippage_pct > s_cfg.max_slippage_pct:
                                logger.debug(
                                    "%s: signal on %s rejected, slippage %.1f%% > max_slippage_pct %.1f%% "
                                    "(quoted %.4f, real fill %.4f)",
                                    s_cfg.name, series_id, slippage_pct, s_cfg.max_slippage_pct,
                                    naive_price, price,
                                )
                                self._log_activity(
                                    s_cfg.name, series_id, window_min, "rejected",
                                    f"{signal.direction.value.upper()} — проскальзывание {slippage_pct:.0f}% "
                                    f"> лимита {s_cfg.max_slippage_pct:.0f}% (котировка {naive_price:.3f} -> {price:.3f})",
                                )
                                continue

                    trade = Trade(
                        strategy=s_cfg.name, entry_window_min=window_min, series_id=series_id,
                        inst_id=inst_id, direction=signal.direction, entry_price=price,
                        stake_usd=stake, contracts=stake / price, opened_ts=now,
                        expiry_ts=market.expiry_ts, reason=signal.reason,
                    )
                    if wallet.open_trade(trade):
                        logger.info(
                            "OPEN  [%s|%dm] %s %s @ %.3f stake=$%.2f (%s)",
                            s_cfg.name, window_min, signal.direction.value.upper(), inst_id,
                            price, stake, signal.reason,
                        )
                        self._log_activity(
                            s_cfg.name, series_id, window_min, "opened",
                            f"{signal.direction.value.upper()} @ {price:.3f} стейк ${stake:.2f} — {signal.reason}",
                        )
                    else:
                        logger.warning("%s: could not afford stake $%.2f (balance $%.2f)", s_cfg.name, stake, wallet.balance)
                        self._log_activity(
                            s_cfg.name, series_id, window_min, "rejected",
                            f"не хватило средств на стейк ${stake:.2f} (баланс ${wallet.balance:.2f})",
                        )

    # -- settling trades --------------------------------------------------------------
    async def _settle_expired_trades(self) -> None:
        """Poll OKX for the real settlement outcome of every expired-but-
        still-open trade. Checks are throttled to at most one per trade
        every `settlement_poll_interval_sec` (previously this field was
        parsed from config.yaml but never actually used — a settlement
        check fired on every single engine tick regardless, so the real
        give-up window was `settlement_poll_attempts * poll_interval_sec`,
        not `* settlement_poll_interval_sec` as the config implied. Now it
        does what it says.) After `settlement_poll_attempts` checks with
        still no definitive outcome, the stake is refunded
        (`mark_unresolved`) rather than left distorting win/loss stats —
        see the "не засчиталась в статистику" conversation this was added
        for: that's what an UNRESOLVED trade looks like on the dashboard
        (0W/0L, $0 PnL, gone from Активные сделки)."""
        now = time.time()
        poll_interval = self.cfg.okx.settlement_poll_interval_sec
        for wallet in self.wallets.values():
            for trade in list(wallet.open_trades()):
                if now < trade.expiry_ts:
                    continue

                last_check = self._last_settlement_check.get(trade.id, 0.0)
                if now - last_check < poll_interval:
                    continue  # not due for another settlement check yet
                self._last_settlement_check[trade.id] = now

                winning_direction = await self.provider.check_settlement(trade.series_id, trade.inst_id)
                if winning_direction is None:
                    attempts = self._settlement_attempts.get(trade.id, 0) + 1
                    self._settlement_attempts[trade.id] = attempts
                    if attempts >= self.cfg.okx.settlement_poll_attempts:
                        logger.warning(
                            "Settlement unresolved after %d attempts (~%.0fs) for %s (%s) — "
                            "refunding stake so it doesn't distort stats.",
                            attempts, attempts * poll_interval, trade.id, trade.inst_id,
                        )
                        wallet.mark_unresolved(trade)
                        self._settlement_attempts.pop(trade.id, None)
                        self._last_settlement_check.pop(trade.id, None)
                        self._log_activity(
                            trade.strategy, trade.series_id, trade.entry_window_min, "unresolved",
                            f"{trade.direction.value.upper()} {trade.inst_id} — исход не подтверждён за "
                            f"{attempts} попыток, стейк ${trade.stake_usd:.2f} возвращён",
                        )
                    continue

                self._settlement_attempts.pop(trade.id, None)
                self._last_settlement_check.pop(trade.id, None)
                outcome = winning_direction == trade.direction
                wallet.settle_trade(trade, won=outcome)
                logger.info(
                    "%s [%s|%dm] trade=%s %s -> %s  pnl=$%.2f  balance=$%.2f",
                    "WIN " if outcome else "LOSS", trade.strategy, trade.entry_window_min,
                    trade.id, trade.direction.value.upper(), trade.status.value,
                    trade.pnl_usd or 0.0, wallet.balance,
                )
                self._log_activity(
                    trade.strategy, trade.series_id, trade.entry_window_min,
                    "won" if outcome else "lost",
                    f"{trade.direction.value.upper()} {trade.inst_id} -> pnl ${trade.pnl_usd or 0.0:+.2f}",
                )

    # -- live activity feed ----------------------------------------------------------
    def _log_activity(self, strategy: str, series_id: str, window_min: Optional[int], kind: str, message: str) -> None:
        self._activity_seq += 1
        self._activity.append(ActivityEvent(
            id=self._activity_seq, ts=time.time(), strategy=strategy,
            series_id=series_id, window_min=window_min, kind=kind, message=message,
        ))

    def activity_since(self, since_id: int = 0, strategy: Optional[str] = None, limit: int = 300) -> list[ActivityEvent]:
        """Events with id > since_id (oldest first), optionally filtered
        to one strategy — the dashboard polls this incrementally (passing
        back the highest id it's already rendered) instead of re-fetching
        the whole buffer every tick."""
        rows = [e for e in self._activity if e.id > since_id and (strategy is None or e.strategy == strategy)]
        return rows[-limit:]

    def latest_activity_id(self) -> int:
        return self._activity_seq

    # -- persistence ---------------------------------------------------------------
    def _maybe_persist(self) -> None:
        self.storage.append_closed_trades(self.wallets)
        now = time.time()
        if now - self._last_snapshot_write >= self.cfg.storage.snapshot_every_sec:
            self.storage.write_snapshot(self.wallets)
            self._last_snapshot_write = now

    # -- dashboard access -----------------------------------------------------------
    def snapshot(self) -> EngineSnapshot:
        combo_stats = build_combo_stats(self.wallets)
        leaderboard = build_leaderboard(combo_stats, min_sample_size=5)
        history = self.provider.btc_price_history()
        underlying_price = history[-1].price if history else None
        return EngineSnapshot(
            wallets=self.wallets,
            combo_stats=combo_stats,
            leaderboard=leaderboard,
            underlying_price=underlying_price,
            active_markets=dict(self.provider.active_markets()),
        )
