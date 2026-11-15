import numpy as np
import pandas as pd
import pytest

from clv.summary import customer_summary, daily_transactions, weeks_between


@pytest.fixture()
def log():
    rows = [
        # customer 1: four invoices on three days in calibration, one holdout purchase
        (1, "2011-01-03 09:00", 10.0),
        (1, "2011-01-03 15:30", 5.0),  # same day: counts once
        (1, "2011-01-10 11:00", 20.0),
        (1, "2011-01-24 12:00", 30.0),
        (1, "2011-02-14 10:00", 40.0),  # holdout
        # customer 2: one purchase day (two invoices) at the very end of calibration
        (2, "2011-02-06 08:00", 7.0),
        (2, "2011-02-06 17:00", 3.0),
        # customer 3: first seen in the holdout -> excluded from the summary
        (3, "2011-02-20 10:00", 99.0),
        # customer 1 again after the holdout end -> ignored
        (1, "2011-03-20 10:00", 1000.0),
    ]
    return pd.DataFrame(rows, columns=["CustomerID", "InvoiceDate", "Revenue"]).assign(
        InvoiceDate=lambda d: pd.to_datetime(d["InvoiceDate"]))


def test_daily_aggregation_counts_days_not_invoices(log):
    daily = daily_transactions(log)
    assert len(daily) == 7
    assert daily.loc[daily["customer"] == 2, "spend"].tolist() == [10.0]


def test_fader_hardie_summary(log):
    s = customer_summary(daily_transactions(log), "2011-02-07", "2011-03-07")
    assert list(s.index) == [1, 2]
    c1, c2 = s.loc[1], s.loc[2]
    assert c1["frequency"] == 2  # 3 purchase days -> 2 repeats (4 invoices)
    assert c1["recency"] == pytest.approx(3.0)  # Jan 3 -> Jan 24
    assert c1["T"] == pytest.approx(5.0)  # Jan 3 -> Feb 7
    assert c1["monetary_value"] == pytest.approx(25.0)  # mean of repeat days (20, 30)
    assert c1["total_spend"] == pytest.approx(65.0)
    assert c1["days_since_last"] == pytest.approx(14.0)
    assert c1["holdout_transactions"] == 1 and c1["holdout_revenue"] == pytest.approx(40.0)
    assert c2["frequency"] == 0 and c2["recency"] == 0.0 and c2["monetary_value"] == 0.0
    assert c2["T"] == pytest.approx(1 / 7)
    assert c2["holdout_transactions"] == 0 and c2["holdout_revenue"] == 0.0
    assert s["frequency"].dtype == np.int64


def test_weeks_between():
    assert weeks_between("2011-09-10", "2011-12-10") == pytest.approx(13.0)
