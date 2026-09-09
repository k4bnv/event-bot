"""
Entry-window scheduler.

For every (series, expiry, strategy) triple we fire each configured
"minutes-to-expiry" checkpoint (e.g. 12 / 7 / 2) at most once, and only on a
genuine downward CROSSING of that threshold — the window's own remaining
time has to have actually been above the checkpoint at some point during
this window's life, not just already at-or-below it the very first time we
looked. That distinction matters as soon as one strategy's entry_windows_min
list is shared across series of different lengths (config.yaml's default
does this on purpose, e.g. breakout_retest trades both a 5-minute and a
15-minute series with the same [12, 7, 2] list): on the 15-minute series all
three are real crossings (remaining_min starts at 15 and genuinely passes
through 12, 7, then 2). On the 5-minute series remaining_min never exceeds
~5 at all — so a naive "remaining_min <= w and not fired yet" check saw 12
and 7 BOTH already satisfied on the very first poll of every single 5-minute
window, and fired them together in one due_windows() call. The caller
(engine.py's per-strategy loop) then dutifully evaluated the strategy twice
for that one call and opened two near-simultaneous trades on the exact same
instrument — same direction, same price, stakes a cent apart only because
the first trade's stake had already shifted the wallet balance the second
was sized from. Real bug, found via a live trade with a duplicate
"12 мин"/"7 мин" pair on one 5-minute window — see tests for the exact
regression case.
"""
from __future__ import annotations

import time
from typing import Iterable


class EntryWindowManager:
    def __init__(self, stale_after_sec: float = 3600.0):
        self._fired: dict[tuple, set[int]] = {}
        self._last_seen: dict[tuple, float] = {}
        # The remaining_min value the FIRST due_windows() call for a given
        # key ever saw — a proxy for "how much time this window actually
        # had" (we don't track each window's own open_ts elsewhere). A
        # checkpoint only counts as a real crossing, and is only eligible
        # to ever fire, if it's strictly below this starting value.
        self._window_start_min: dict[tuple, float] = {}
        self.stale_after_sec = stale_after_sec

    def due_windows(
        self,
        series_id: str,
        expiry_ts: float,
        strategy_name: str,
        remaining_sec: float,
        configured_windows_min: Iterable[int],
    ) -> list[int]:
        key = (series_id, expiry_ts, strategy_name)
        fired = self._fired.setdefault(key, set())
        self._last_seen[key] = time.time()

        remaining_min = remaining_sec / 60.0
        window_start_min = self._window_start_min.setdefault(key, remaining_min)

        due = []
        for w in sorted(set(configured_windows_min), reverse=True):
            if (
                w not in fired and remaining_sec > 0
                and w < window_start_min  # a real crossing was possible for this window at all
                and remaining_min <= w    # ...and it has now actually happened
            ):
                fired.add(w)
                due.append(w)
        return due

    def reset_strategy(self, strategy_name: str) -> None:
        """Drop fired-checkpoint bookkeeping for ONE strategy only, so a
        per-strategy reset doesn't leave stale "already fired" entries
        blocking checkpoints that should fire again on its fresh wallet.
        Other strategies' keys (same series/expiry, different name) are
        untouched."""
        stale = [k for k in self._fired if k[2] == strategy_name]
        for k in stale:
            self._fired.pop(k, None)
            self._last_seen.pop(k, None)
            self._window_start_min.pop(k, None)

    def prune(self) -> None:
        """Drop bookkeeping for expiries we haven't touched in a while, so
        long-running processes don't leak memory."""
        now = time.time()
        stale = [k for k, last in self._last_seen.items() if now - last > self.stale_after_sec]
        for k in stale:
            self._fired.pop(k, None)
            self._last_seen.pop(k, None)
            self._window_start_min.pop(k, None)
