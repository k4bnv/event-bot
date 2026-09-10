"""
Per-strategy decision breakdown — reads checkpoint_features (every
strategy, not just absorption_reversal — see analyze_absorption_funnel.py
for that strategy's own deeper phase-by-phase funnel) and reports how
many evaluations landed on each `decision` value:

    no_signal                  the strategy's own entry logic never fired
    skipped_already_positioned dynamic_timing strategy already has a bet in this market
    rejected_no_quote          signal fired but market had no live quote yet
    rejected_low_balance       wallet balance too small to stake anything
    rejected_no_fill_price     signal fired but no fill price available
    rejected_max_coefficient   signal fired, but priced above s_cfg.max_coefficient
    rejected_max_slippage      signal fired, quote moved too much before fill
    rejected_insufficient_funds  stake computed but wallet couldn't cover it
    opened                     an actual trade

This is the first thing to check before touching a strategy's
thresholds: a strategy stuck at 100% no_signal has a problem in its OWN
entry-condition logic (thresholds too strict, or a structural bug), while
one stuck at rejected_max_coefficient is finding real setups but pricing
them out — a completely different, usually much smaller, fix (raise
max_coefficient, or the strategy is chasing bets nobody would actually
take at that price).

Reads data/bot.db directly via sqlite3 — no dashboard/HTTP/auth
involved (see analyze_absorption_funnel.py's docstring for why that
matters when DASHBOARD_PASSWORD is set).

Usage:
    python3 scripts/decision_breakdown.py data/bot.db
    python3 scripts/decision_breakdown.py data/bot.db --strategy absorption_reversal
"""
from __future__ import annotations

import sqlite3
import sys
from collections import defaultdict


def load_counts(db_path: str, strategy: str | None = None) -> dict[str, dict[str, int]]:
    """{strategy: {decision: count}}, ordered by nothing in particular —
    the caller sorts however it wants to print."""
    conn = sqlite3.connect(db_path)
    try:
        if strategy:
            cur = conn.execute(
                "SELECT strategy, decision, COUNT(*) FROM checkpoint_features "
                "WHERE strategy = ? GROUP BY strategy, decision", (strategy,),
            )
        else:
            cur = conn.execute(
                "SELECT strategy, decision, COUNT(*) FROM checkpoint_features "
                "GROUP BY strategy, decision",
            )
        out: dict[str, dict[str, int]] = defaultdict(dict)
        for strat, decision, count in cur.fetchall():
            out[strat][decision or "(none)"] = count
        return dict(out)
    finally:
        conn.close()


def main(db_path: str, strategy: str | None = None) -> None:
    counts = load_counts(db_path, strategy)
    if not counts:
        print("No checkpoint_features rows found" + (f" for strategy={strategy}" if strategy else "") + ".")
        return

    for strat in sorted(counts, key=lambda s: -sum(counts[s].values())):
        by_decision = counts[strat]
        total = sum(by_decision.values())
        opened = by_decision.get("opened", 0)
        print(f"\n{strat}  (total evaluations: {total}, opened: {opened})")
        for decision, count in sorted(by_decision.items(), key=lambda kv: -kv[1]):
            pct = count / total * 100
            print(f"    {decision:32s} {count:6d}  {pct:5.1f}%")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    strategy_arg = None
    if "--strategy" in sys.argv:
        strategy_arg = sys.argv[sys.argv.index("--strategy") + 1]
        args = [a for a in args if a != strategy_arg]
    if len(args) != 1:
        print(f"Usage: python3 {sys.argv[0]} path/to/bot.db [--strategy NAME]")
        sys.exit(1)
    main(args[0], strategy_arg)
