"""RFM scoring: quintile scores (1-5) and the usual named segments from the R and F scores.

R = days since the last purchase (fewer is better), F = number of purchase days, M = total
spend. Scores are percentile ranks cut into fifths, with ties sharing a score (about a third of
customers bought on a single day, so they all get F = 1).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# R-score digit followed by F-score digit -> segment (first match wins). This is the widely used
# RF grid of RFM practice; M only enters the total score.
SEGMENT_RULES: tuple[tuple[str, str], ...] = (
    (r"[1-2][1-2]", "Hibernating"),
    (r"[1-2][3-4]", "At risk"),
    (r"[1-2]5", "Can't lose them"),
    (r"3[1-2]", "About to sleep"),
    (r"33", "Need attention"),
    (r"[3-4][4-5]", "Loyal customers"),
    (r"41", "Promising"),
    (r"51", "New customers"),
    (r"[4-5][2-3]", "Potential loyalists"),
    (r"5[4-5]", "Champions"),
)
SEGMENTS: tuple[str, ...] = tuple(dict.fromkeys(name for _, name in SEGMENT_RULES))


def quintile_score(values: pd.Series | np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    """Score 1-5 from the average percentile rank (ties share a score)."""
    s = pd.Series(np.asarray(values, dtype=np.float64))
    if not higher_is_better:
        s = -s
    pct = s.rank(method="average", pct=True).to_numpy()
    return np.clip(np.ceil(pct * 5.0), 1, 5).astype(np.int64)


def segment_name(r_score: int, f_score: int) -> str:
    """Named RFM segment for one (R, F) score pair."""
    key = f"{r_score}{f_score}"
    for pattern, name in SEGMENT_RULES:
        if re.fullmatch(pattern, key):
            return name
    raise ValueError(f"no segment for R={r_score}, F={f_score}")


def rfm_table(summary: pd.DataFrame) -> pd.DataFrame:
    """R/F/M scores, total score (3-15) and segment for a :func:`customer_summary` frame."""
    r = quintile_score(summary["days_since_last"], higher_is_better=False)
    f = quintile_score(summary["n_days"])
    m = quintile_score(summary["total_spend"])
    lookup = {(i, j): segment_name(i, j) for i in range(1, 6) for j in range(1, 6)}
    seg = [lookup[(i, j)] for i, j in zip(r, f, strict=True)]
    return pd.DataFrame(
        {"R": r, "F": f, "M": m, "RFM_score": r + f + m, "segment": seg}, index=summary.index
    )
