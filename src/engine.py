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
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

from .config import AppConfig
from .features import pct_change_over
from .market_data import MarketDataProvider
from .metrics import ComboStats, Leaderboard, build_combo_stats, build_leaderboard
from .models import Direction, Trade
from .storage import Storage
from .strategies import STRATEGY_REGISTRY, BaseStrategy, StrategyContext
from .strategies.fair_value_edge import (
    DEFAULT_MIN_SIGMA_PCT_PER_MIN, DEFAULT_UNFIXED_STRIKE_BASIS_PCT,
    basis_sigma_for_market, compute_barrier_stats, min_sigma_per_sec_from_pct,
)
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

    @staticmethod
    def _wallet_key(strategy_name: str, window_min: Optional[int] = None) -> str:
        """Every configured entry_windows_min checkpoint gets its OWN
        wallet — not just its own stats row (build_combo_stats already
        did that from trade data alone) but its own actual capital, so
        e.g. breakout_retest's "12 мин" checkpoint compounds completely
        independently of its "2 мин" one instead of both drawing stake
        from, and feeding wins back into, one shared pool. This composite
        string is the ONE place that format is decided — Storage persists
        it verbatim as the wallets table's primary key (see
        write_snapshot/load_wallets).

        window_min=None (only for a `dynamic_timing` strategy — see
        StrategyConfig.dynamic_timing) collapses this to the bare strategy
        name instead: such a strategy gets exactly ONE wallet regardless
        of how many checkpoints it's actually called at, since it places
        at most one trade per market no matter which checkpoint that
        happens on. Never collides with a per-checkpoint key, which always
        contains ':'."""
        return strategy_name if window_min is None else f"{strategy_name}:{window_min}"

    def wallet_for(self, strategy_name: str, window_min: Optional[int] = None) -> VirtualWallet:
        """Convenience accessor mirroring _wallet_key — mainly for tests
        and any future code that needs one specific checkpoint's wallet
        (or, for a dynamic_timing strategy, its one shared wallet) rather
        than iterating self.wallets directly."""
        return self.wallets[self._wallet_key(strategy_name, window_min)]

    def _build_wallets_and_strategies(self) -> None:
        enabled_names = self.cfg.enabled_strategy_names()
        if not enabled_names:
            raise RuntimeError("No strategies enabled in config.yaml — nothing to run.")
        # Full rebuild, not an incremental update — clear first so a
        # checkpoint removed from entry_windows_min (via the Settings tab)
        # doesn't leave a stale, no-longer-tradeable wallet lingering in
        # self.wallets forever after a reset() rebuild.
        self.wallets.clear()
        self.strategy_instances.clear()
        # One query up front rather than one per (strategy, checkpoint) —
        # cheap, and keeps _build_wallets_and_strategies() the single place
        # that decides "restore vs fresh start" for every wallet at once.
        saved_wallets = self.storage.load_wallets()
        for s_cfg in self.cfg.strategies:
            if not s_cfg.enabled:
                continue
            # Each strategy gets its OWN independent deposit (config.yaml ->
            # strategies.<name>.deposit_usd, default $100) — not a slice of
            # one shared pool. That's what makes a fair head-to-head
            # comparison possible: every strategy is judged on the same
            # starting bankroll, not a fraction that shrinks as you enable
            # more. Same fairness applied one level deeper here: EACH of its
            # configured entry checkpoints gets that same full deposit_usd
            # again, independently — not a further split of it — so "12
            # мин" and "2 мин" are judged on equal starting terms too.
            #
            # Exception: dynamic_timing strategies (see adaptive_timing)
            # place at most one trade per market no matter which of their
            # (usually many, densely-spaced) checkpoints it happens on —
            # splitting capital N ways there would leave N-1 wallets
            # permanently idle. They get exactly ONE wallet instead.
            if s_cfg.dynamic_timing:
                key = self._wallet_key(s_cfg.name)
                self.wallets[key] = self._restore_or_create_wallet(s_cfg, None, saved_wallets.get(key))
            else:
                for window_min in s_cfg.entry_windows_min:
                    key = self._wallet_key(s_cfg.name, window_min)
                    self.wallets[key] = self._restore_or_create_wallet(s_cfg, window_min, saved_wallets.get(key))

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
    def _restore_or_create_wallet(s_cfg, window_min: Optional[int], saved: Optional[dict]) -> VirtualWallet:
        """A fresh VirtualWallet(deposit_usd) for this ONE checkpoint if
        `saved` is None (first-ever launch, a checkpoint just added to
        entry_windows_min, or one just reset) — otherwise resumes the
        balance write_snapshot() persisted for it, so a redeploy/crash/
        restart doesn't silently reset every checkpoint back to its
        starting deposit while the Analytics tab (backed by the
        separately, every-tick-persisted trades table) keeps remembering
        the full history.

        initial_balance is restored too (not re-read from config) so
        net_pnl/equity keep meaning "profit since this checkpoint's actual
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
            return VirtualWallet(strategy=s_cfg.name, window_min=window_min, initial_balance=s_cfg.deposit_usd)
        wallet = VirtualWallet(
            strategy=s_cfg.name, window_min=window_min, initial_balance=saved["initial_balance"],
        )
        wallet.balance = saved["balance"] + saved["reserved"]
        if saved["reserved"]:
            window_label = f"{window_min} мин" if window_min is not None else "динамический тайминг"
            logger.warning(
                "Strategy '%s' (%s): restarted with $%.2f still reserved in-flight at the last "
                "snapshot — refunded to balance (its open trade(s) can't be resumed across a restart).",
                s_cfg.name, window_label, saved["reserved"],
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
        """Reset just ONE strategy's wallets/trades/timing-history — every
        other strategy's wallets, trades and persisted rows are left
        completely untouched. Used by the Settings tab's per-strategy
        Reset button. Raises ValueError for an unknown or disabled
        strategy name (there's nothing to reset for a strategy with no
        wallet).

        Rebuilds ALL of this strategy's checkpoint wallets fresh — first
        dropping every existing "name:*" key (not just the ones in the
        CURRENT entry_windows_min list) so a checkpoint removed just
        before this reset doesn't leave a stale wallet behind, mirroring
        the same cleanup _build_wallets_and_strategies() does on a full
        reset()."""
        s_cfg = next((s for s in self.cfg.strategies if s.name == name), None)
        if s_cfg is None:
            raise ValueError(f"unknown strategy '{name}'")
        if not s_cfg.enabled:
            raise ValueError(f"strategy '{name}' is not enabled")

        old_strategy = self.strategy_instances.get(name)
        prefix = f"{name}:"
        for key in [k for k in self.wallets if k == name or k.startswith(prefix)]:
            del self.wallets[key]
        if s_cfg.dynamic_timing:
            key = self._wallet_key(name)
            self.wallets[key] = VirtualWallet(strategy=name, window_min=None, initial_balance=s_cfg.deposit_usd)
        else:
            for window_min in s_cfg.entry_windows_min:
                key = self._wallet_key(name, window_min)
                self.wallets[key] = VirtualWallet(strategy=name, window_min=window_min, initial_balance=s_cfg.deposit_usd)
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

    # -- ML feature logging ----------------------------------------------------------
    def _record_checkpoint_features(
        self, s_cfg, series_id: str, ctx: StrategyContext, signal, decision: str,
        fill_price: Optional[float] = None, stake_usd: Optional[float] = None,
        trade_id: Optional[str] = None,
    ) -> None:
        """One feature snapshot per (strategy, checkpoint) EVALUATION,
        written to data/bot.db's checkpoint_features table regardless of
        `decision` (no_signal/rejected-for-whatever-reason/opened alike)
        — see storage.py's module docstring for why a dataset needs the
        negative examples too, not just executed trades, to be any use
        for future ML work. Reuses the same barrier-model machinery
        fair_value_edge/ai_prompt already compute (z_score/base_prob/
        sigma_horizon, with the volatility floor + basis-risk terms) so
        the logged numbers mean the same thing everywhere they appear,
        not a second, subtly different calculation.

        Best-effort: wrapped in try/except so a logging failure can NEVER
        take down real trading logic, matching this project's standing
        rule for every strategy (ai_prompt's docstring states it most
        explicitly, but it applies here just as much — this is pure
        instrumentation, not something a trade decision should ever
        depend on).
        """
        try:
            market = ctx.market
            points = list(ctx.price_history)
            now = points[-1].ts if points else time.time()
            spot = points[-1].price if points else None
            barrier = (
                compute_barrier_stats(
                    points, spot, market.floor_strike, ctx.remaining_sec,
                    min_sigma_per_sec=min_sigma_per_sec_from_pct(DEFAULT_MIN_SIGMA_PCT_PER_MIN),
                    basis_sigma=basis_sigma_for_market(market, DEFAULT_UNFIXED_STRIKE_BASIS_PCT),
                )
                if spot is not None and market.floor_strike is not None else None
            )
            ob = ctx.orderbook
            row = {
                "id": uuid.uuid4().hex[:12],
                "ts": now,
                "strategy": s_cfg.name,
                "series_id": series_id,
                "inst_id": market.inst_id,
                "window_min": ctx.window_min,
                "remaining_sec": ctx.remaining_sec,
                "market_method": market.method,
                "up_price": market.up_price,
                "floor_strike": market.floor_strike,
                "strike_is_fixed": None if market.strike_is_fixed is None else int(market.strike_is_fixed),
                "spot": spot,
                "drift_5m_pct": pct_change_over(points, now, 300),
                "mom_1m_pct": pct_change_over(points, now, 60),
                "z_score": barrier.z_score if barrier else None,
                "base_prob": barrier.base_prob if barrier else None,
                "sigma_horizon_pct": barrier.sigma_horizon_pct if barrier else None,
                "orderbook_bid_vol": ob.bid_volume(10) if ob else None,
                "orderbook_ask_vol": ob.ask_volume(10) if ob else None,
                "funding_rate": ctx.funding_rate,
                "previous_outcome": ctx.previous_outcome.value if ctx.previous_outcome else None,
                "signal_direction": signal.direction.value if signal else None,
                "signal_confidence": signal.confidence if signal else None,
                "signal_reason": signal.reason if signal else None,
                "decision": decision,
                "fill_price": fill_price,
                "stake_usd": stake_usd,
                "trade_id": trade_id,
            }
            self.storage.log_checkpoint_features(row)
        except Exception:
            logger.exception(
                "Failed to log checkpoint features for %s/%s (decision=%s) — continuing.",
                s_cfg.name, series_id, decision,
            )

    # -- opening trades ------------------------------------------------------------
    async def _open_due_trades(self) -> None:
        now = time.time()
        active_markets = self.provider.active_markets()

        for s_cfg in self.cfg.strategies:
            if not s_cfg.enabled:
                continue
            strategy = self.strategy_instances[s_cfg.name]

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
                    # Each entry checkpoint has its own wallet (see
                    # _wallet_key) — looked up per window_min, not once per
                    # strategy, since that's exactly the isolation this
                    # split is for: "12 мин" stakes/wins/loses out of its
                    # own pool, completely independent of "2 мин"'s.
                    # dynamic_timing strategies collapse to their one
                    # shared wallet instead (see _wallet_key).
                    wallet_key_window = None if s_cfg.dynamic_timing else window_min
                    wallet = self.wallets[self._wallet_key(s_cfg.name, wallet_key_window)]
                    already_open_this_market = any(
                        t.series_id == series_id and t.expiry_ts == market.expiry_ts
                        for t in wallet.open_trades()
                    )
                    ctx = StrategyContext(
                        price_history=self.provider.btc_price_history(),
                        orderbook=self.provider.btc_orderbook(),
                        remaining_sec=remaining, window_min=window_min,
                        market=market, funding_rate=self.provider.funding_rate(),
                        previous_outcome=self._previous_outcome.get(series_id),
                        already_open_this_market=already_open_this_market,
                    )
                    signal = await strategy.evaluate(ctx)
                    if signal is None:
                        logger.debug("%s: no signal at %dm-to-expiry for %s", s_cfg.name, window_min, series_id)
                        self._log_activity(s_cfg.name, series_id, window_min, "no_signal", "нет сигнала")
                        self._record_checkpoint_features(s_cfg, series_id, ctx, None, "no_signal")
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
                        self._record_checkpoint_features(s_cfg, series_id, ctx, signal, "rejected_no_quote")
                        continue

                    stake = round(wallet.balance * s_cfg.stake_fraction, 4)
                    if stake <= 0.01:
                        logger.warning("%s: wallet balance too low to stake ($%.2f) — skipping.", s_cfg.name, wallet.balance)
                        self._log_activity(
                            s_cfg.name, series_id, window_min, "rejected",
                            f"баланс слишком мал для стейка (${wallet.balance:.2f})",
                        )
                        self._record_checkpoint_features(
                            s_cfg, series_id, ctx, signal, "rejected_low_balance", stake_usd=stake,
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
                        self._record_checkpoint_features(
                            s_cfg, series_id, ctx, signal, "rejected_no_fill_price", stake_usd=stake,
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
                        self._record_checkpoint_features(
                            s_cfg, series_id, ctx, signal, "rejected_max_coefficient",
                            fill_price=price, stake_usd=stake,
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
                                self._record_checkpoint_features(
                                    s_cfg, series_id, ctx, signal, "rejected_max_slippage",
                                    fill_price=price, stake_usd=stake,
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
                        self._record_checkpoint_features(
                            s_cfg, series_id, ctx, signal, "opened",
                            fill_price=price, stake_usd=stake, trade_id=trade.id,
                        )
                    else:
                        logger.warning("%s: could not afford stake $%.2f (balance $%.2f)", s_cfg.name, stake, wallet.balance)
                        self._log_activity(
                            s_cfg.name, series_id, window_min, "rejected",
                            f"не хватило средств на стейк ${stake:.2f} (баланс ${wallet.balance:.2f})",
                        )
                        self._record_checkpoint_features(
                            s_cfg, series_id, ctx, signal, "rejected_insufficient_funds",
                            fill_price=price, stake_usd=stake,
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
