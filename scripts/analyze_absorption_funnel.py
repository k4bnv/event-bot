"""
Funnel analysis for Strategy K (absorption_reversal) — reads the ML
export CSV (GET /api/features/export.csv?strategy=absorption_reversal,
or the full export filtered here) and reports how many checkpoint
evaluations reached each phase of the strategy, using the progressive
`extra_json` diagnostics absorption_reversal.evaluate() writes (see its
docstring — diag is filled in stage by stage, so a row that never made
it past e.g. the TFI gate simply has no `volume_multiple`/`residual_pct`/
etc. keys at all).

This answers "which gate is actually the bottleneck" from real data
instead of guessing which config knob to loosen.

Usage:
    python3 scripts/analyze_absorption_funnel.py path/to/export.csv

No third-party dependencies — safe to run directly inside the container
too, e.g.:
    docker compose exec okx-event-bot python3 scripts/analyze_absorption_funnel.py /app/data/export.csv
(after copying/exporting the CSV there), or just run it locally on your
own machine against a CSV downloaded via the dashboard's export button.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys

# Ordered stages, each keyed by the diagnostics field that first appears
# once evaluate() reaches that point — mirrors the gate order in
# absorption_reversal.py exactly.
STAGES = [
    ("spread checked", "spread_pct"),
    ("enough prints (>= min_prints)", "n_prints"),
    ("TFI computed", "tfi"),
    ("volume vs baseline computed", "volume_multiple"),
    ("residual computed", "residual_pct"),
    ("candidate direction found (Phase A passed)", "candidate_direction"),
    ("book replenishment checked (Phase B)", "replenish_ratio"),
    ("local-range breakout checked (Phase C)", "broke_local_range"),
]


def load_rows(csv_path: str, strategy: str = "absorption_reversal") -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("strategy") != strategy:
                continue
            extra = row.get("extra_json") or ""
            diag = json.loads(extra) if extra.strip() else {}
            rows.append({"decision": row.get("decision"), "diag": diag})
    return rows


def main(csv_path: str) -> None:
    rows = load_rows(csv_path)
    total = len(rows)
    if total == 0:
        print("No absorption_reversal rows found in this CSV.")
        return

    print(f"Total absorption_reversal checkpoint evaluations: {total}\n")
    print(f"{'Stage':45s} {'reached':>8s} {'% of total':>10s} {'% of prior':>10s}")

    prior = total
    for label, field in STAGES:
        reached = sum(1 for r in rows if field in r["diag"] and r["diag"][field] is not None)
        pct_total = reached / total * 100
        pct_prior = reached / prior * 100 if prior else 0.0
        print(f"{label:45s} {reached:8d} {pct_total:9.1f}% {pct_prior:9.1f}%")
        prior = reached if reached else prior  # avoid div-by-zero cascading to 0

    fired = sum(1 for r in rows if r["decision"] not in (None, "", "no_signal"))
    print(f"\nActual signals fired: {fired} ({fired / total * 100:.2f}% of all checkpoints)")

    # Distribution of the key gate values, for rows that got far enough to
    # compute them — helps judge HOW close near-misses were, not just
    # whether they passed.
    def _values(field: str) -> list[float]:
        return [r["diag"][field] for r in rows if isinstance(r["diag"].get(field), (int, float))]

    print("\nValue distributions (only rows where the field was computed):")
    for field in ("tfi", "volume_multiple", "residual_pct", "replenish_ratio"):
        vals = _values(field)
        if not vals:
            print(f"  {field:20s}: no data")
            continue
        abs_vals = [abs(v) for v in vals] if field in ("tfi", "residual_pct") else vals
        print(
            f"  {field:20s}: n={len(vals):5d}  "
            f"median={statistics.median(vals):+.4f}  "
            f"max|value|={max(abs_vals):.4f}  "
            f"mean={statistics.fmean(vals):+.4f}"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} path/to/export.csv")
        sys.exit(1)
    main(sys.argv[1])
