import urllib.error
from datetime import datetime

import numpy as np
import pytest

from clv import data


def test_download_fails_with_clear_message_when_offline(tmp_path, monkeypatch):
    def offline(*args, **kwargs):
        raise urllib.error.URLError("no network")

    monkeypatch.setattr(data.urllib.request, "urlopen", offline)
    with pytest.raises(RuntimeError, match="needs internet access"):
        data.download(tmp_path)
    assert not (tmp_path / "raw" / data.ZIP_NAME).exists()


def test_parse_xlsx_streams_mixed_types(tmp_path):
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.append(list(data.COLUMNS))
    ws.append([536365, "85123A", "HEART", 6, datetime(2010, 12, 1, 8, 26), 2.55, 17850, "UK"])
    ws.append(["C536379", "D", "Discount", -1, datetime(2010, 12, 1, 9, 41), 27.5, None, "UK"])
    ws.append([536366, 22633, None, 6, datetime(2010, 12, 1, 8, 28), 1.85, 17850, "UK"])
    path = tmp_path / "tiny.xlsx"
    wb.save(path)
    df = data.parse_xlsx(path, chunk_rows=2)
    assert len(df) == 3
    assert df["InvoiceNo"].tolist() == ["536365", "C536379", "536366"]
    assert df["StockCode"].tolist() == ["85123A", "D", "22633"]
    assert df["Quantity"].dtype == np.int64 and df["UnitPrice"].dtype == np.float64
    assert np.isnan(df["CustomerID"].iloc[1]) and df["CustomerID"].iloc[0] == 17850.0
    assert str(df["InvoiceDate"].dtype) == "datetime64[ns]"
