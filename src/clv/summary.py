"""Customer-level summaries in the Fader-Hardie convention (time unit: weeks).

A *transaction* is a customer purchase **day**: all invoices of one customer on one calendar day
are combined, so two invoices placed an hour apart count once. For the calibration period:

- ``frequency`` (x):   number of repeat purchase days (purchase days minus the first one)
- ``recency`` (t_x):   weeks between the first and the last purchase day (0 if x = 0)
- ``T``:               weeks between the first purchase day and the end of calibration
- ``monetary_value``:  mean spend of the repeat purchase days (0 if x = 0); the first purchase
                       is excluded, as in Fader, Hardie & Lee (2005)

Periods are half-open: ``[start, calibration_end)`` and ``[calibration_end, holdout_end)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DAYS_PER_WEEK = 7.0


def daily_transactions(
    transactions: pd.DataFrame,
    customer_col: str = "CustomerID",
    date_col: str = "InvoiceDate",
    value_col: str = "Revenue",
) -> pd.DataFrame:
    """Aggregate lines to one row per customer and purchase day: ``customer, date, spend``."""
    day = pd.to_datetime(transactions[date_col]).dt.normalize()
    out = (
        pd.DataFrame(
            {
                "customer": transactions[customer_col].to_numpy(),
                "date": day.to_numpy(),
                "spend": transactions[value_col].to_numpy(dtype=np.float64),
            }
        )
        .groupby(["customer", "date"], as_index=False, sort=True)["spend"]
        .sum()
    )
    return out


def customer_summary(
    daily: pd.DataFrame,
    calibration_end: str | pd.Timestamp,
    holdout_end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Per-customer calibration summary (+ holdout outcomes if ``holdout_end`` is given).

    Only customers with at least one purchase day before ``calibration_end`` are included.
    Extra columns: ``n_days`` (purchase days, = x + 1), ``total_spend`` (all calibration days),
    ``first``/``last`` purchase dates, ``days_since_last`` (for RFM) and, with a holdout,
    ``holdout_transactions`` and ``holdout_revenue``.
    """
    cal_end = pd.Timestamp(calibration_end)
    cal = daily[daily["date"] < cal_end]
    g = cal.groupby("customer", sort=True)
    first = g["date"].min()
    last = g["date"].max()
    n_days = g["date"].size()
    total = g["spend"].sum()
    first_spend = cal.loc[g["date"].idxmin(), ["customer", "spend"]].set_index("customer")["spend"]
    freq = (n_days - 1).astype(np.int64)
    repeat_spend = total - first_spend
    with np.errstate(invalid="ignore", divide="ignore"):
        monetary = np.where(freq > 0, repeat_spend / freq.clip(lower=1), 0.0)
    week = pd.Timedelta(days=1) * DAYS_PER_WEEK
    out = pd.DataFrame(
        {
            "frequency": freq,
            "recency": (last - first) / week,
            "T": (cal_end - first) / week,
            "monetary_value": monetary,
            "n_days": n_days.astype(np.int64),
            "total_spend": total,
            "first": first,
            "last": last,
            "days_since_last": ((cal_end - last) / pd.Timedelta(days=1)).astype(np.float64),
        }
    )
    out.index.name = "customer"
    if holdout_end is not None:
        hold_end = pd.Timestamp(holdout_end)
        hold = daily[(daily["date"] >= cal_end) & (daily["date"] < hold_end)]
        hg = hold.groupby("customer")
        out["holdout_transactions"] = hg.size().reindex(out.index, fill_value=0).astype(np.int64)
        out["holdout_revenue"] = hg["spend"].sum().reindex(out.index, fill_value=0.0)
    return out


def weeks_between(start: str | pd.Timestamp, end: str | pd.Timestamp) -> float:
    """Length of ``[start, end)`` in weeks."""
    return float((pd.Timestamp(end) - pd.Timestamp(start)) / pd.Timedelta(days=DAYS_PER_WEEK))
