import json

import numpy as np

from clv.pipeline import run
from clv.synthetic import make_transactions


def test_pipeline_end_to_end_on_synthetic_transactions(tmp_path):
    raw = make_transactions(n_customers=500, seed=1)
    metrics = run(tmp_path / "data", tmp_path / "reports", raw=raw)

    saved = json.loads((tmp_path / "reports" / "metrics.json").read_text())
    assert saved == metrics
    for name in ("fader_frequency_buckets", "tracking", "value_concentration", "p_alive_heatmap"):
        assert (tmp_path / "reports" / "figures" / f"{name}.png").stat().st_size > 10_000
    assert (tmp_path / "reports" / "segments.csv").exists()

    steps = {s["step"]: s["removed"] for s in metrics["data"]["cleaning_steps"]}
    assert steps["missing_customer"] == 1 and steps["non_product"] == 1
    assert steps["cancellations"] == 1 and steps["fully_returned"] == 1
    main = metrics["validation"]["main"]
    assert main["bgnbd"]["converged"] and main["gamma_gamma"]["converged"]
    tx = main["holdout_transactions"]["bgnbd"]
    assert np.isfinite(tx["mae"]) and tx["pearson"] > 0.3
    assert 0.1 <= main["top_decile_revenue_capture"]["bgnbd_gg_clv"] <= 1.0
    business = metrics["full_data_model"]
    assert sum(t["customers"] for t in business["value_tiers"]) == business["n_customers"]
    assert abs(sum(s["clv_share"] for s in business["segments"]) - 1.0) < 1e-4  # 6-dp rounding
