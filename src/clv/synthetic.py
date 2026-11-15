"""Synthetic invoice lines in the Online Retail layout (for tests and offline demos).

Customers are drawn from the BG/NBD + Gamma-Gamma generative process, so the ground truth is
known; a few cancellations, a postage line, a missing CustomerID and a non-UK customer are added
so every cleaning rule has something to do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from clv.bgnbd import BGNBDParams
from clv.gamma_gamma import GammaGammaParams


def make_transactions(
    n_customers: int = 400,
    start: str = "2010-12-01",
    end: str = "2011-12-10",
    bg: BGNBDParams | None = None,
    gg: GammaGammaParams | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Raw-looking invoice lines between ``start`` and ``end`` (exclusive)."""
    rng = np.random.default_rng(seed)
    bg = bg or BGNBDParams(r=0.8, alpha=10.0, a=0.5, b=3.0)
    gg = gg or GammaGammaParams(p=3.0, q=4.0, gamma=400.0)
    t0, t_end = pd.Timestamp(start), pd.Timestamp(end)
    span_days = (t_end - t0).days
    rows: list[tuple] = []
    invoice = 536_000
    for cust in range(n_customers):
        birth = rng.uniform(0, 0.75 * span_days) / 7.0  # weeks
        lam = rng.gamma(bg.r, 1.0 / bg.alpha)
        p_drop = rng.beta(bg.a, bg.b)
        nu = rng.gamma(gg.q, 1.0 / gg.gamma)
        times = [birth]
        now = birth
        while True:
            now += rng.exponential(1.0 / lam)
            if now * 7.0 >= span_days:
                break
            times.append(now)
            if rng.random() < p_drop:
                break
        country = "France" if cust % 25 == 0 else "United Kingdom"
        for t in times:
            when = t0 + pd.Timedelta(days=int(t * 7.0), hours=10)
            spend = rng.gamma(gg.p, 1.0 / nu)
            n_lines = int(rng.integers(1, 4))
            invoice += 1
            for k in range(n_lines):
                price = round(float(rng.choice([0.85, 1.25, 2.95, 4.95])), 2)
                qty = max(1, int(round(spend / n_lines / price)))
                rows.append((str(invoice), f"{21000 + k}", "ITEM", qty, when, price,
                             float(12000 + cust), country))
    df = pd.DataFrame(rows, columns=["InvoiceNo", "StockCode", "Description", "Quantity",
                                     "InvoiceDate", "UnitPrice", "CustomerID", "Country"])
    # A few rows for the cleaning rules: a full return, postage, a line without a customer.
    first = df.iloc[0]
    extra = [
        ("C999001", first.StockCode, "ITEM", -int(first.Quantity),
         first.InvoiceDate + pd.Timedelta(hours=2), first.UnitPrice, first.CustomerID,
         first.Country),
        (first.InvoiceNo, "POST", "POSTAGE", 1, first.InvoiceDate, 18.0, first.CustomerID,
         first.Country),
        ("999002", "22000", "ITEM", 3, first.InvoiceDate, 1.25, np.nan, "United Kingdom"),
    ]
    df = pd.concat([df, pd.DataFrame(extra, columns=df.columns)], ignore_index=True)
    df["InvoiceNo"] = df["InvoiceNo"].astype(object)
    df["StockCode"] = df["StockCode"].astype(object)
    df["Description"] = df["Description"].astype(object)
    df["Country"] = df["Country"].astype(object)
    df["Quantity"] = df["Quantity"].astype(np.int64)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"]).astype("datetime64[ns]")
    df["UnitPrice"] = df["UnitPrice"].astype(np.float64)
    df["CustomerID"] = df["CustomerID"].astype(np.float64)
    return df
