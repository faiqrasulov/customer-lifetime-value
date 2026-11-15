import numpy as np
import pandas as pd
import pytest

from clv.cleaning import CleaningConfig, clean_transactions

COLS = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice",
        "CustomerID", "Country"]
UK = "United Kingdom"


@pytest.fixture()
def raw():
    t = pd.Timestamp("2011-03-01 10:00")
    h = pd.Timedelta(hours=1)
    rows = [
        ("500001", "85123A", "HEART", 10, t, 2.55, 1.0, UK),        # fully returned below
        ("500001", "71053", "LANTERN", 6, t, 3.39, 1.0, UK),        # partially returned below
        ("C500002", "85123A", "HEART", -10, t + h, 2.55, 1.0, UK),  # full return
        ("C500003", "71053", "LANTERN", -2, t + h, 3.39, 1.0, UK),  # partial return -> 4 left
        ("C500004", "22111", "OLD ITEM", -1, t, 1.0, 1.0, UK),      # no earlier purchase
        ("500005", "22423", "CAKESTAND", 2, t, 12.75, np.nan, UK),  # no customer
        ("500006", "22423", "CAKESTAND", 1, t, 0.0, 2.0, UK),       # zero price
        ("500007", "POST", "POSTAGE", 1, t, 18.0, 2.0, UK),         # non-product
        ("500007", "M", "Manual", 1, t, 5.0, 2.0, UK),              # non-product
        ("500008", "23843", "BIRDIE", 80995, t, 2.08, 3.0, UK),     # extreme line
        ("500009", "15056BL", "UMBRELLA", 3, t, 5.95, 2.0, UK),     # kept (letter suffix)
        ("500010", "21232", "TRINKET", 12, t, 1.25, 4.0, "France"),  # other country
    ]
    return pd.DataFrame(rows, columns=COLS)


def test_cleaning_steps_and_row_counts(raw):
    clean, steps = clean_transactions(raw)
    removed = {s["step"]: s["removed"] for s in steps}
    assert removed == {"raw": 0, "missing_customer": 1, "cancellations": 3, "fully_returned": 1,
                       "non_positive": 1, "non_product": 2, "extreme_lines": 1, "country": 1}
    assert steps[-1]["rows_after"] == len(clean) == 2
    for prev, nxt in zip(steps, steps[1:], strict=False):
        assert nxt["rows_before"] == prev["rows_after"]
    assert "2 full returns" not in steps[2]["note"] and "1 full returns" in steps[2]["note"]
    assert "1 unmatched" in steps[2]["note"]


def test_partial_return_keeps_net_quantity_and_revenue(raw):
    clean, _ = clean_transactions(raw)
    lantern = clean[clean["StockCode"] == "71053"].iloc[0]
    assert lantern["Quantity"] == 4
    assert lantern["Revenue"] == pytest.approx(4 * 3.39)
    assert set(clean["StockCode"]) == {"71053", "15056BL"}
    assert clean["CustomerID"].dtype == np.int64


def test_all_countries_and_return_must_not_precede_purchase(raw):
    late = raw.copy()
    late.loc[2, "InvoiceDate"] = pd.Timestamp("2011-02-01")  # "return" before the purchase
    clean, steps = clean_transactions(late, CleaningConfig(country=None))
    assert "country" not in [s["step"] for s in steps]
    assert 10 in clean.loc[clean["StockCode"] == "85123A", "Quantity"].tolist()
    assert "France" in set(clean["Country"])
