"""Holdout validation: error metrics, baselines, ranking quality and the classic Fader plots."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from clv import bgnbd
from clv.bgnbd import BGNBDParams
from clv.rfm import rfm_table
from clv.summary import DAYS_PER_WEEK, customer_summary


def error_metrics(actual, predicted) -> dict[str, float]:
    """Per-customer MAE, RMSE, Pearson/Spearman correlation and aggregate totals."""
    y = np.asarray(actual, dtype=np.float64)
    f = np.asarray(predicted, dtype=np.float64)
    err = f - y
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "pearson": float(stats.pearsonr(y, f)[0]),
        "spearman": float(stats.spearmanr(y, f)[0]),
        "total_actual": float(y.sum()),
        "total_predicted": float(f.sum()),
        "total_error_pct": float(100.0 * (f.sum() - y.sum()) / y.sum()),
    }


def top_share_capture(score, outcome, share: float = 0.10) -> float:
    """Share of the total ``outcome`` held by the top ``share`` of customers ranked by ``score``.

    Ties at the cut-off are split pro rata (the expected value under random tie-breaking), so
    coarse scores such as the 3-15 RFM total are not flattered by an arbitrary tie order.
    """
    s = np.asarray(score, dtype=np.float64)
    y = np.asarray(outcome, dtype=np.float64)
    k = share * s.size
    order = np.argsort(-s, kind="stable")
    s_sorted, y_sorted = s[order], y[order]
    uniq, start = np.unique(-s_sorted, return_index=True)  # groups of equal score, best first
    ends = np.append(start[1:], s_sorted.size)
    taken, captured = 0.0, 0.0
    for lo, hi in zip(start, ends, strict=True):
        size = hi - lo
        frac = min(1.0, max(0.0, (k - taken) / size))
        captured += frac * y_sorted[lo:hi].sum()
        taken += frac * size
        if taken >= k:
            break
    return float(captured / y.sum())


def gain_curve(score, outcome, shares: np.ndarray) -> np.ndarray:
    """:func:`top_share_capture` evaluated at each share in ``shares``."""
    return np.array([top_share_capture(score, outcome, float(q)) for q in shares])


def historical_rate_baseline(summary: pd.DataFrame, horizon_weeks: float) -> dict[str, np.ndarray]:
    """Naive extrapolation: past repeat-purchase rate x/T and past mean spend per purchase day."""
    T = summary["T"].to_numpy(dtype=np.float64)
    x = summary["frequency"].to_numpy(dtype=np.float64)
    rate = np.divide(x, T, out=np.zeros_like(x), where=T > 0)
    transactions = rate * horizon_weeks
    spend = summary["total_spend"].to_numpy() / summary["n_days"].to_numpy()
    return {"transactions": transactions, "revenue": transactions * spend}


def rfm_lookup_baseline(
    daily: pd.DataFrame, calibration_end: str | pd.Timestamp, horizon_weeks: float,
    summary: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """RFM-cell means, learned without peeking at the holdout.

    RFM scores are computed at ``calibration_end - horizon`` and the mean number of purchase
    days and revenue in the following ``horizon`` (still inside calibration) is recorded for each
    RFM total score (3-15). Each customer's score at ``calibration_end`` is then mapped to that
    mean. Scores never seen in the inner window fall back to the overall inner mean.
    """
    cal_end = pd.Timestamp(calibration_end)
    inner_end = cal_end - pd.Timedelta(days=horizon_weeks * DAYS_PER_WEEK)
    inner = customer_summary(daily[daily["date"] < cal_end], inner_end, cal_end)
    inner_scores = rfm_table(inner)["RFM_score"]
    cols = ["holdout_transactions", "holdout_revenue"]
    table = inner.groupby(inner_scores.to_numpy())[cols].mean()
    scores = rfm_table(summary)["RFM_score"].to_numpy()
    out = {}
    for key, col in (("transactions", "holdout_transactions"), ("revenue", "holdout_revenue")):
        lookup = table[col].reindex(np.unique(scores)).fillna(inner[col].mean())
        out[key] = lookup.loc[scores].to_numpy(dtype=np.float64)
    return out


def frequency_buckets(x, actual, predictions: dict[str, np.ndarray], max_x: int = 7
                      ) -> pd.DataFrame:
    """Mean actual and predicted holdout purchases by calibration frequency (last: max_x+)."""
    bucket = np.minimum(np.asarray(x, dtype=np.int64), max_x)
    frame = pd.DataFrame({"bucket": bucket, "actual": np.asarray(actual, dtype=np.float64)})
    for name, pred in predictions.items():
        frame[name] = np.asarray(pred, dtype=np.float64)
    out = frame.groupby("bucket").agg(n=("actual", "size"), **{
        c: (c, "mean") for c in frame.columns if c != "bucket"
    })
    out.index = [f"{b}+" if b == max_x else str(b) for b in out.index]
    return out


def weekly_tracking(
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    params: BGNBDParams,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Weekly repeat purchases of the calibration cohort: actual vs BG/NBD expectation.

    Weeks are counted back from ``end`` so that the last week ends exactly at ``end`` (and a
    calibration cut-off a whole number of weeks earlier falls on a week boundary); the first
    week may start a few days before ``start``. The expectation for a customer first seen at
    time ``s`` (weeks) is the unconditional ``E[X(t - s)]`` for ``t > s``, summed over customers.
    """
    end = pd.Timestamp(end)
    n_weeks = int(np.ceil((end - pd.Timestamp(start)) / pd.Timedelta(days=DAYS_PER_WEEK)))
    start = end - pd.Timedelta(days=DAYS_PER_WEEK * n_weeks)
    cohort = summary.index
    first = summary["first"]
    rows = daily[daily["customer"].isin(cohort) & (daily["date"] < end)]
    is_repeat = rows["date"].to_numpy() > first.reindex(rows["customer"]).to_numpy()
    week = ((rows["date"] - start) / pd.Timedelta(days=DAYS_PER_WEEK)).to_numpy()
    actual = np.bincount(np.floor(week[is_repeat]).astype(np.int64), minlength=n_weeks)[:n_weeks]

    birth = ((first - start) / pd.Timedelta(days=DAYS_PER_WEEK)).to_numpy(dtype=np.float64)
    births, counts = np.unique(birth, return_counts=True)
    edges = np.arange(n_weeks + 1, dtype=np.float64)
    elapsed = np.clip(edges[None, :] - births[:, None], 0.0, None)
    cum = counts @ bgnbd.expected_transactions(params, elapsed)
    week_start = start + pd.to_timedelta(edges[:-1] * DAYS_PER_WEEK, unit="D")
    return pd.DataFrame(
        {
            "week_start": week_start,
            "actual": actual.astype(np.float64),
            "expected": np.diff(cum),
            "cum_actual": np.cumsum(actual).astype(np.float64),
            "cum_expected": cum[1:],
        }
    )
