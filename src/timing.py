"""
Entry-window scheduler.

For every (series, expiry, strategy) triple we fire each configured
"minutes-to-expiry" checkpoint (e.g. 12 / 7 / 2) at most once. Because the
engine polls every few seconds, `remaining_minutes` crosses each threshold
from above exactly once per window — the first poll where it does, we mark
that checkpoint fired and hand it back to the caller.
"""
from __future__ import annotations

import time
from typing import Iterable


class EntryWindowManager:
    def __init__(self, stale_after_sec: float = 3600.0):
        self._fired: dict[tuple, set[int]] = {}
        self._last_seen: dict[tuple, float] = {}
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
        due = []
        for w in sorted(set(configured_windows_min), reverse=True):
            if w not in fired and remaining_min <= w and remaining_sec > 0:
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

    def prune(self) -> None:
        """Drop bookkeeping for expiries we haven't touched in a while, so
        long-running processes don't leak memory."""
        now = time.time()
        stale = [k for k, last in self._last_seen.items() if now - last > self.stale_after_sec]
        for k in stale:
            self._fired.pop(k, None)
            self._last_seen.pop(k, None)
