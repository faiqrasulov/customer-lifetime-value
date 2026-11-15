import numpy as np
import pandas as pd
import pytest

from clv.rfm import SEGMENTS, quintile_score, rfm_table, segment_name


def test_quintile_scores_and_ties():
    assert quintile_score(np.arange(10)).tolist() == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert quintile_score(np.arange(10), higher_is_better=False).tolist() == [
        5, 5, 4, 4, 3, 3, 2, 2, 1, 1]
    tied = quintile_score([1, 1, 1, 1, 1, 1, 2, 3, 4, 5])
    assert len(set(tied[:6])) == 1  # ties share a score


@pytest.mark.parametrize("r, f, name", [(5, 5, "Champions"), (1, 1, "Hibernating"),
                                        (5, 1, "New customers"), (4, 1, "Promising"),
                                        (1, 5, "Can't lose them"), (2, 3, "At risk"),
                                        (3, 3, "Need attention"), (4, 5, "Loyal customers"),
                                        (3, 1, "About to sleep"),
                                        (4, 2, "Potential loyalists")])
def test_segment_map(r, f, name):
    assert segment_name(r, f) == name


def test_every_score_pair_has_a_segment():
    names = {segment_name(r, f) for r in range(1, 6) for f in range(1, 6)}
    assert names == set(SEGMENTS)


def test_rfm_table():
    summary = pd.DataFrame({"days_since_last": [1, 200, 30, 90, 5],
                            "n_days": [20, 1, 5, 2, 9],
                            "total_spend": [5000.0, 50.0, 800.0, 120.0, 2000.0]},
                           index=pd.Index([10, 11, 12, 13, 14], name="customer"))
    t = rfm_table(summary)
    assert t.loc[10, "segment"] == "Champions" and t.loc[11, "segment"] == "Hibernating"
    assert t["RFM_score"].between(3, 15).all()
    assert t.loc[10, "RFM_score"] == 15
