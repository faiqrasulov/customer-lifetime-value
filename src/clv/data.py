"""Download, parse and cache the UCI "Online Retail" transactions.

The official zip holds one ~23 MB xlsx (541,909 invoice lines). It is parsed once with
openpyxl in read-only (streaming) mode, in chunks, so peak memory stays a few hundred MB,
and the typed result is cached as ``data/online_retail.parquet``.
"""

from __future__ import annotations

import shutil
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
ZIP_NAME = "online_retail.zip"
XLSX_NAME = "Online Retail.xlsx"
PARQUET_NAME = "online_retail.parquet"
COLUMNS = (
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
)
N_ROWS_EXPECTED = 541_909


def download(data_home: str | Path = "data", url: str = URL, timeout: float = 120.0) -> Path:
    """Download the official zip into ``data_home/raw/`` (skipped when already cached)."""
    raw = Path(data_home) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    target = raw / ZIP_NAME
    if target.exists() and zipfile.is_zipfile(target):
        return target
    tmp = target.with_suffix(".part")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the UCI Online Retail data from {url}. The first run needs "
            f"internet access; afterwards the copy cached in '{raw}/' is used. Cause: {exc!r}"
        ) from exc
    if not zipfile.is_zipfile(tmp):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file from {url} is not a zip archive.")
    tmp.replace(target)
    return target


def _iter_xlsx_rows(xlsx: Path) -> Iterator[tuple]:
    from openpyxl import load_workbook

    wb = load_workbook(xlsx, read_only=True, data_only=True)
    try:
        rows = wb.worksheets[0].iter_rows(values_only=True)
        header = tuple(next(rows))
        if header[: len(COLUMNS)] != COLUMNS:
            raise ValueError(f"unexpected header {header}")
        for row in rows:
            if row and any(v is not None for v in row):
                yield row[: len(COLUMNS)]
    finally:
        wb.close()


def _chunk_frame(rows: list[tuple]) -> pd.DataFrame:
    cols = list(zip(*rows, strict=True))

    def text(values: tuple) -> np.ndarray:
        return np.array(["" if v is None else str(v).strip() for v in values], dtype=object)

    def when(values: tuple) -> np.ndarray:
        return np.array(
            [v if isinstance(v, datetime) else pd.Timestamp(v) for v in values],
            dtype="datetime64[ns]",
        )

    return pd.DataFrame(
        {
            "InvoiceNo": text(cols[0]),
            "StockCode": text(cols[1]),
            "Description": text(cols[2]),
            "Quantity": np.array(cols[3], dtype=np.int64),
            "InvoiceDate": when(cols[4]),
            "UnitPrice": np.array(cols[5], dtype=np.float64),
            "CustomerID": np.array(
                [np.nan if v is None else float(v) for v in cols[6]], dtype=np.float64
            ),
            "Country": text(cols[7]),
        }
    )


def parse_xlsx(xlsx: str | Path, chunk_rows: int = 50_000) -> pd.DataFrame:
    """Stream the workbook into a typed DataFrame (strings as object, ids as float64)."""
    frames: list[pd.DataFrame] = []
    buf: list[tuple] = []
    for row in _iter_xlsx_rows(Path(xlsx)):
        buf.append(row)
        if len(buf) == chunk_rows:
            frames.append(_chunk_frame(buf))
            buf = []
    if buf:
        frames.append(_chunk_frame(buf))
    return pd.concat(frames, ignore_index=True)


def load_online_retail(data_home: str | Path = "data") -> pd.DataFrame:
    """Return all 541,909 raw invoice lines, parsing the xlsx only on the first call."""
    data_home = Path(data_home)
    cache = data_home / PARQUET_NAME
    if not cache.exists():
        zpath = download(data_home)
        xlsx = data_home / "raw" / XLSX_NAME
        if not xlsx.exists():
            with zipfile.ZipFile(zpath) as zf:
                zf.extract(XLSX_NAME, path=data_home / "raw")
        df = parse_xlsx(xlsx)
        if len(df) != N_ROWS_EXPECTED:
            raise ValueError(f"expected {N_ROWS_EXPECTED} rows, parsed {len(df)}")
        df.to_parquet(cache, index=False)
    df = pd.read_parquet(cache)
    for col in ("InvoiceNo", "StockCode", "Description", "Country"):
        df[col] = df[col].astype(object)
    return df
