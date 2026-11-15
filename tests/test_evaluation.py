import numpy as np
import pandas as pd
import pytest

from clv import bgnbd
from clv.cleaning import clean_transactions
from clv.evaluation import (
    error_metrics,
    frequency_buckets,
    historical_rate_baseline,
    top_share_capture,
    weekly_tracking,
)
from clv.summary import customer_summary, daily_transactions
from clv.synthetic import make_transactions


def test_error_metrics_known_values():
    m = error_metrics([0, 1, 2, 3], [0, 1, 2, 5])
    assert m["mae"] == pytest.approx(0.5) and m["rmse"] == pytest.approx(1.0)
    assert m["total_error_pct"] == pytest.approx(100 * 2 / 6)


def test_top_share_capture_ranks_and_splits_ties():
    outcome = np.array([10.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert top_share_capture(-np.arange(10), outcome, 0.1) == pytest.approx(0.5)
    assert top_share_capture(outcome, outcome, 0.3) == pytest.approx(1.0)
    # all scores tied: any share captures that share of the total in expectation
    assert top_share_capture(np.ones(10), outcome, 0.2) == pytest.approx(0.2)
    # top 10% with a 2-way tie at the top: half of the tied pair's outcome
    score = np.array([3, 3, 1, 1, 1, 1, 1, 1, 1, 0])
    assert top_share_capture(score, outcome, 0.1) == pytest.approx(5.0 / 20.0)


def test_frequency_buckets_and_historical_baseline():
    summary = pd.DataFrame({"frequency": [0, 1, 9, 12], "T": [10.0, 10.0, 20.0, 0.0],
                            "total_spend": [10.0, 40.0, 100.0, 5.0], "n_days": [1, 2, 10, 13]})
    base = historical_rate_baseline(summary, 13.0)
    np.testing.assert_allclose(base["transactions"], [0.0, 1.3, 5.85, 0.0])
    np.testing.assert_allclose(base["revenue"], [0.0, 26.0, 58.5, 0.0])
    tab = frequency_buckets(summary["frequency"], [0, 1, 2, 4], {"m": [1, 1, 1, 1]}, max_x=7)
    assert list(tab.index) == ["0", "1", "7+"]
    assert tab.loc["7+", "n"] == 2 and tab.loc["7+", "actual"] == pytest.approx(3.0)


def test_weekly_tracking_counts_every_repeat_purchase():
    daily = daily_transactions(clean_transactions(make_transactions(n_customers=150, seed=3))[0])
    summary = customer_summary(daily, "2011-09-10", "2011-12-10")
    params = bgnbd.BGNBDParams(0.8, 10.0, 0.5, 3.0)
    tr = weekly_tracking(daily, summary, params, "2010-12-01", "2011-12-10")
    assert tr["week_start"].iloc[-1] + pd.Timedelta(days=7) == pd.Timestamp("2011-12-10")
    in_cal = tr["week_start"] < pd.Timestamp("2011-09-10")
    assert tr.loc[in_cal, "actual"].sum() == summary["frequency"].sum()
    assert tr.loc[~in_cal, "actual"].sum() == summary["holdout_transactions"].sum()
    assert np.all(np.diff(tr["cum_expected"]) >= 0)
