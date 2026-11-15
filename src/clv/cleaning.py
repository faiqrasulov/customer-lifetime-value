"""Turn raw invoice lines into clean, positive product purchases, logging rows at each step.

Order of the rules (each step reports rows in / rows removed):

1. ``missing_customer``  no CustomerID: cannot be attributed to a customer.
2. ``cancellations``     invoices starting with "C" are removed; before that, each cancellation
                         line is netted against the latest earlier purchase line of the same
                         customer and StockCode (returns). Fully returned lines are counted in
                         step 3, partially returned lines keep the remaining quantity.
3. ``fully_returned``    purchase lines whose whole quantity was returned.
4. ``non_positive``      quantity <= 0 or unit price <= 0 (free samples, adjustments).
5. ``non_product``       StockCodes that are not merchandise (postage, fees, manual
                         adjustments, discounts, gift vouchers ...): anything that is not five
                         digits plus an optional 1-2 letter variant suffix.
6. ``extreme_lines``     guard against data-entry outliers: lines with more than
                         ``max_line_quantity`` units or ``max_line_revenue`` GBP.
7. ``country``           keep one market (default United Kingdom).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

PRODUCT_CODE = r"^\d{5}[A-Za-z]{0,2}$"


@dataclass(frozen=True)
class CleaningConfig:
    """Thresholds for :func:`clean_transactions`; ``country=None`` keeps all countries."""

    country: str | None = "United Kingdom"
    max_line_quantity: int = 10_000
    max_line_revenue: float = 10_000.0


def _net_returns(purchases: pd.DataFrame, cancels: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Subtract each cancelled quantity from the latest earlier matching purchase line.

    A purchase line matches a cancellation line when CustomerID and StockCode agree, it is not
    later than the cancellation and it still has at least the cancelled quantity. Cancellations
    are processed in time order, so one purchase line can absorb several partial returns.
    """
    qty = purchases["Quantity"].to_numpy(dtype=np.int64).copy()
    when = purchases["InvoiceDate"].to_numpy()
    lines: dict[tuple, list[int]] = defaultdict(list)
    keys = zip(purchases["CustomerID"].to_numpy(), purchases["StockCode"].to_numpy(), strict=True)
    for pos, key in enumerate(keys):
        lines[key].append(pos)
    order = np.argsort(cancels["InvoiceDate"].to_numpy(), kind="stable")
    c_cust = cancels["CustomerID"].to_numpy()[order]
    c_code = cancels["StockCode"].to_numpy()[order]
    c_qty = -cancels["Quantity"].to_numpy(dtype=np.int64)[order]
    c_when = cancels["InvoiceDate"].to_numpy()[order]
    stats = {"full": 0, "partial": 0, "unmatched": 0}
    for cust, code, need, t in zip(c_cust, c_code, c_qty, c_when, strict=True):
        best = -1
        for pos in lines.get((cust, code), ()):
            if when[pos] <= t and qty[pos] >= need and (best < 0 or when[pos] >= when[best]):
                best = pos
        if best < 0:
            stats["unmatched"] += 1
            continue
        qty[best] -= need
        stats["full" if qty[best] == 0 else "partial"] += 1
    return qty, stats


def clean_transactions(
    raw: pd.DataFrame, config: CleaningConfig | None = None
) -> tuple[pd.DataFrame, list[dict]]:
    """Apply the cleaning rules; return clean lines (with ``Revenue``) and the step log."""
    cfg = config or CleaningConfig()
    steps: list[dict] = []
    df = raw

    def log(step: str, before: int, after: int, note: str) -> None:
        steps.append(
            {"step": step, "rows_before": before, "removed": before - after, "rows_after": after,
             "note": note}
        )

    steps.append({"step": "raw", "rows_before": len(df), "removed": 0, "rows_after": len(df),
                  "note": "invoice lines in the source file"})

    n = len(df)
    df = df[df["CustomerID"].notna()]
    log("missing_customer", n, len(df), "no CustomerID")

    n = len(df)
    is_cancel = df["InvoiceNo"].astype(str).str.startswith("C").to_numpy()
    cancels = df[is_cancel]
    df = df[~is_cancel]
    is_purchase = df["Quantity"].to_numpy() > 0
    net_qty, net = _net_returns(df[is_purchase], cancels)
    log("cancellations", n, len(df),
        f"invoices starting with 'C'; each line netted against the latest earlier purchase of "
        f"the same customer and StockCode: {net['full']} full returns, {net['partial']} partial, "
        f"{net['unmatched']} unmatched (bought before Dec 2010 or a different quantity)")

    n = len(df)
    qty = df["Quantity"].to_numpy(dtype=np.int64).copy()
    qty[is_purchase] = net_qty
    fully_returned = is_purchase & (qty == 0)
    df = df.assign(Quantity=qty)[~fully_returned]
    log("fully_returned", n, len(df),
        f"purchase lines whose whole quantity was cancelled later ({net['partial']} more lines "
        "keep their remaining quantity)")

    n = len(df)
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    log("non_positive", n, len(df), "quantity <= 0 or unit price <= 0")

    n = len(df)
    is_product = df["StockCode"].astype(str).str.match(PRODUCT_CODE).to_numpy()
    dropped_codes = sorted(df.loc[~is_product, "StockCode"].astype(str).unique())
    df = df[is_product]
    log("non_product", n, len(df), "non-merchandise StockCodes: " + ", ".join(dropped_codes))

    n = len(df)
    revenue = df["Quantity"].to_numpy() * df["UnitPrice"].to_numpy()
    keep = (df["Quantity"].to_numpy() <= cfg.max_line_quantity) & (
        revenue <= cfg.max_line_revenue
    )
    df = df[keep]
    log("extreme_lines", n, len(df),
        f"lines above {cfg.max_line_quantity:,} units or GBP {cfg.max_line_revenue:,.0f}")

    if cfg.country is not None:
        n = len(df)
        df = df[df["Country"] == cfg.country]
        log("country", n, len(df), f"keep {cfg.country} only")

    out = df.assign(
        CustomerID=df["CustomerID"].astype(np.int64),
        Revenue=df["Quantity"].to_numpy() * df["UnitPrice"].to_numpy(),
    ).reset_index(drop=True)
    return out, steps
