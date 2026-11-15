"""End-to-end analysis: clean -> summarise -> validate on a holdout -> refit -> CLV segments.

``run()`` reproduces every number in the README and writes ``reports/metrics.json``,
``reports/segments.csv`` and the figures.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from clv import bgnbd, gamma_gamma
from clv.cleaning import CleaningConfig, clean_transactions
from clv.evaluation import (
    error_metrics,
    frequency_buckets,
    gain_curve,
    historical_rate_baseline,
    rfm_lookup_baseline,
    top_share_capture,
    weekly_tracking,
)
from clv.rfm import SEGMENTS, rfm_table
from clv.summary import customer_summary, daily_transactions, weeks_between
from clv.value import ValueAssumptions, customer_lifetime_value

DATA_START = "2010-12-01"
DATA_END = "2011-12-10"  # exclusive: the last invoice is on 2011-12-09


@dataclass(frozen=True)
class Split:
    """Half-open calibration ``[DATA_START, calibration_end)`` and holdout periods."""

    name: str
    calibration_end: str
    holdout_end: str
    note: str


SPLITS = (
    Split("main", "2011-09-10", DATA_END,
          "first ~9.3 months calibrate; the last 13 weeks (Sep 10 - Dec 9, 2011) are held out"),
    Split("summer", "2011-06-11", "2011-09-10",
          "sensitivity: same 13-week holdout length, but outside the Q4 gift season"),
)
TOP_SHARE = 0.10


def _fits(summary: pd.DataFrame) -> tuple[bgnbd.BGNBDParams, gamma_gamma.GammaGammaParams, dict]:
    bg_res = bgnbd.fit(summary["frequency"], summary["recency"], summary["T"])
    repeat = summary[summary["frequency"] > 0]
    gg_res = gamma_gamma.fit(repeat["frequency"], repeat["monetary_value"])
    info = {
        "bgnbd": {"params": bg_res.as_dict(), "loglik": bg_res.loglik,
                  "converged": bg_res.converged, "n_customers": bg_res.n_obs},
        "gamma_gamma": {"params": gg_res.as_dict(), "loglik": gg_res.loglik,
                        "converged": gg_res.converged, "n_repeat_customers": gg_res.n_obs},
        "frequency_spend_correlation": gamma_gamma.frequency_spend_correlation(
            repeat["frequency"], repeat["monetary_value"]),
    }
    bg, gg = bgnbd.params_from(bg_res), gamma_gamma.params_from(gg_res)
    info["gamma_gamma"]["population_mean_spend"] = gg.population_mean_spend
    info["gamma_gamma"]["observed_mean_spend"] = float(repeat["monetary_value"].mean())
    info["bgnbd"]["mean_dropout_probability"] = bg.a / (bg.a + bg.b)
    nbd = bgnbd.fit_no_dropout(summary["frequency"], summary["T"])
    info["bgnbd"]["no_dropout_model"] = {
        "params": nbd.as_dict(),
        "loglik": nbd.loglik,
        "loglik_gain_from_dropout": bg_res.loglik - nbd.loglik,
    }
    return bg, gg, info


def _columns(summary: pd.DataFrame) -> tuple[np.ndarray, ...]:
    return tuple(summary[c].to_numpy(dtype=np.float64)
                 for c in ("frequency", "recency", "T", "monetary_value"))


def _p_alive_stats(p_alive: np.ndarray, x: np.ndarray) -> dict:
    repeat = x > 0
    return {
        "mean": float(p_alive.mean()),
        "min": float(p_alive.min()),
        "p01_repeat_buyers": float(np.quantile(p_alive[repeat], 0.01)),
        "n_below_0_9": int((p_alive < 0.9).sum()),
        "n_below_0_5": int((p_alive < 0.5).sum()),
    }


def _spend_check(summary: pd.DataFrame, gg: gamma_gamma.GammaGammaParams) -> dict:
    """Gamma-Gamma alone: E[spend | x, m] vs the actual mean spend per holdout purchase day.

    Restricted to repeat customers (x >= 1) who bought in the holdout; the baseline is the
    customer's own mean repeat spend in calibration.
    """
    keep = (summary["frequency"] > 0) & (summary["holdout_transactions"] > 0)
    s = summary[keep]
    actual = (s["holdout_revenue"] / s["holdout_transactions"]).to_numpy(dtype=np.float64)
    model = gamma_gamma.conditional_expected_spend(gg, s["frequency"], s["monetary_value"])
    return {
        "n_customers": int(keep.sum()),
        "gamma_gamma": error_metrics(actual, model),
        "own_past_mean": error_metrics(actual, s["monetary_value"].to_numpy(dtype=np.float64)),
    }


def _segment_validation(summary: pd.DataFrame, pred_n: np.ndarray, pred_rev: np.ndarray
                        ) -> list[dict]:
    """Per RFM segment (scored at the end of calibration): predicted vs actual holdout."""
    frame = pd.DataFrame({
        "segment": rfm_table(summary)["segment"].to_numpy(),
        "active": summary["holdout_transactions"].to_numpy() > 0,
        "actual_purchases": summary["holdout_transactions"].to_numpy(dtype=np.float64),
        "predicted_purchases": pred_n,
        "actual_revenue": summary["holdout_revenue"].to_numpy(dtype=np.float64),
        "predicted_revenue": pred_rev,
    })
    out = frame.groupby("segment").agg(
        customers=("active", "size"),
        share_active_in_holdout=("active", "mean"),
        actual_purchases=("actual_purchases", "mean"),
        predicted_purchases=("predicted_purchases", "mean"),
        actual_revenue=("actual_revenue", "mean"),
        predicted_revenue=("predicted_revenue", "mean"),
    )
    out = out.reindex([s for s in SEGMENTS if s in out.index])
    return out.reset_index(names="segment").to_dict(orient="records")


def validate(daily: pd.DataFrame, split: Split, assumptions: ValueAssumptions,
             details: bool = False) -> dict:
    """Fit on the calibration period, score the holdout, compare with the baselines."""
    summary = customer_summary(daily, split.calibration_end, split.holdout_end)
    horizon = weeks_between(split.calibration_end, split.holdout_end)
    bg, gg, info = _fits(summary)
    x, t_x, T, m = _columns(summary)
    actual_n = summary["holdout_transactions"].to_numpy(dtype=np.float64)
    actual_rev = summary["holdout_revenue"].to_numpy(dtype=np.float64)

    pred_n = bgnbd.conditional_expected_transactions(bg, horizon, x, t_x, T)
    pred_rev = pred_n * gamma_gamma.conditional_expected_spend(gg, x, m)
    hist = historical_rate_baseline(summary, horizon)
    rfm_base = rfm_lookup_baseline(daily, split.calibration_end, horizon, summary)
    clv = customer_lifetime_value(bg, gg, x, t_x, T, m, assumptions)["clv"]
    rfm_score = rfm_table(summary)["RFM_score"].to_numpy(dtype=np.float64)
    past_spend = summary["total_spend"].to_numpy(dtype=np.float64)
    rankers = {"bgnbd_gg_clv": clv, "rfm_score": rfm_score, "past_spend": past_spend,
               "oracle": actual_rev}

    result = {
        "split": asdict(split),
        "calibration_weeks": weeks_between(DATA_START, split.calibration_end),
        "holdout_weeks": horizon,
        "n_customers": int(len(summary)),
        "n_repeat_customers": int((summary["frequency"] > 0).sum()),
        "share_active_in_holdout": float((actual_n > 0).mean()),
        **info,
        "holdout_transactions": {
            "bgnbd": error_metrics(actual_n, pred_n),
            "historical_rate": error_metrics(actual_n, hist["transactions"]),
            "rfm_cell_mean": error_metrics(actual_n, rfm_base["transactions"]),
        },
        "holdout_revenue": {
            "bgnbd_gg": error_metrics(actual_rev, pred_rev),
            "historical_rate": error_metrics(actual_rev, hist["revenue"]),
            "rfm_cell_mean": error_metrics(actual_rev, rfm_base["revenue"]),
        },
        "holdout_spend": _spend_check(summary, gg),
        "top_decile_revenue_capture": {
            k: top_share_capture(v, actual_rev, TOP_SHARE) for k, v in rankers.items()
        },
        "top_quintile_revenue_capture": {
            k: top_share_capture(v, actual_rev, 0.20) for k, v in rankers.items()
        },
        "frequency_buckets": frequency_buckets(
            x, actual_n, {"bgnbd": pred_n, "historical_rate": hist["transactions"]}
        ).reset_index(names="frequency").to_dict(orient="records"),
        "p_alive": _p_alive_stats(bgnbd.p_alive(bg, x, t_x, T), x),
        "rfm_segments": _segment_validation(summary, pred_n, pred_rev),
    }
    if details:
        shares = np.linspace(0.0, 1.0, 101)
        result["_tracking"] = weekly_tracking(daily, summary, bg, DATA_START, split.holdout_end)
        result["_gain"] = {"share": shares,
                           **{k: gain_curve(v, actual_rev, shares) for k, v in rankers.items()}}
        result["_buckets"] = frequency_buckets(
            x, actual_n, {"bgnbd": pred_n, "historical_rate": hist["transactions"]})
        tr = result["_tracking"]
        hold = tr["week_start"] >= pd.Timestamp(split.calibration_end)
        result["tracking_summary"] = {
            "calibration_actual": float(tr.loc[~hold, "actual"].sum()),
            "calibration_expected": float(tr.loc[~hold, "expected"].sum()),
            "holdout_actual": float(tr.loc[hold, "actual"].sum()),
            "holdout_expected": float(tr.loc[hold, "expected"].sum()),
            "holdout_peak_week_actual": float(tr.loc[hold, "actual"].max()),
            "holdout_expected_per_week": float(tr.loc[hold, "expected"].mean()),
        }
    return result


def business_view(daily: pd.DataFrame, assumptions: ValueAssumptions) -> dict:
    """Refit on all data and score every customer for the next 12 months."""
    summary = customer_summary(daily, DATA_END)
    bg, gg, info = _fits(summary)
    x, t_x, T, m = _columns(summary)
    val = customer_lifetime_value(bg, gg, x, t_x, T, m, assumptions)
    rfm = rfm_table(summary)
    scores = summary.assign(**val, segment=rfm["segment"].to_numpy())
    clv = scores["clv"].to_numpy()
    total_clv = float(clv.sum())

    by_seg = scores.groupby("segment").agg(
        customers=("clv", "size"),
        p_alive=("p_alive", "mean"),
        expected_purchases_12m=("expected_transactions", "mean"),
        expected_spend=("expected_spend", "mean"),
        clv_12m_mean=("clv", "mean"),
        clv_12m_total=("clv", "sum"),
        past_spend_mean=("total_spend", "mean"),
    ).reindex([s for s in SEGMENTS if s in set(scores["segment"])])
    by_seg["customer_share"] = by_seg["customers"] / len(scores)
    by_seg["clv_share"] = by_seg["clv_12m_total"] / total_clv
    by_seg = by_seg.sort_values("clv_12m_mean", ascending=False)

    # Value tiers by predicted CLV (rank-based, so independent of the assumed margin).
    rank = pd.Series(clv).rank(method="first", ascending=False).to_numpy()
    tier_edges = ((0.0, 0.10, "top 10%"), (0.10, 0.20, "next 10%"), (0.20, 0.50, "next 30%"),
                  (0.50, 1.00, "bottom 50%"))
    tiers = []
    for lo, hi, name in tier_edges:
        mask = (rank > int(lo * len(clv))) & (rank <= int(hi * len(clv)))
        tiers.append({
            "tier": name,
            "customers": int(mask.sum()),
            "clv_12m_mean": float(clv[mask].mean()),
            "clv_share": float(clv[mask].sum() / total_clv),
            "expected_purchases_12m": float(val["expected_transactions"][mask].mean()),
            "expected_spend": float(val["expected_spend"][mask].mean()),
            "p_alive_mean": float(val["p_alive"][mask].mean()),
            "one_time_buyer_share": float((x[mask] == 0).mean()),
            "past_spend_share": float(scores["total_spend"].to_numpy()[mask].sum()
                                          / scores["total_spend"].sum()),
        })

    # A newly acquired customer: first order now, repeat purchases from the model (x = 0, T = 0).
    first_order = daily.sort_values("date").groupby("customer")["spend"].first()
    new_val = customer_lifetime_value(bg, gg, [0.0], [0.0], [0.0], [0.0], assumptions)
    new_customer = {
        "mean_first_order_value": float(first_order.mean()),
        "expected_repeat_purchases_12m": float(new_val["expected_transactions"][0]),
        "expected_spend_per_purchase": float(new_val["expected_spend"][0]),
        "repeat_clv_12m": float(new_val["clv"][0]),
        "first_order_margin": float(assumptions.margin * first_order.mean()),
    }
    new_customer["total_margin_12m"] = new_customer["repeat_clv_12m"] + new_customer[
        "first_order_margin"]

    order = np.sort(clv)[::-1]
    lorenz = np.concatenate([[0.0], np.cumsum(order) / total_clv])
    p_alive = val["p_alive"]
    return {
        "as_of": DATA_END,
        "n_customers": int(len(scores)),
        **info,
        "clv_12m": {
            "total": total_clv,
            "mean": float(clv.mean()),
            "median": float(np.median(clv)),
            "top_10pct_share": float(lorenz[int(0.10 * len(order))]),
            "top_20pct_share": float(lorenz[int(0.20 * len(order))]),
            # Concentration curve (top customers first) lies above the diagonal: G = 2*area - 1.
            "gini": float((lorenz[:-1] + lorenz[1:]).sum() / len(order) - 1.0),
        },
        "p_alive": _p_alive_stats(p_alive, x),
        "one_time_buyers": {
            "customers": int((x == 0).sum()),
            "customer_share": float((x == 0).mean()),
            "clv_share": float(clv[x == 0].sum() / total_clv),
            "clv_12m_mean": float(clv[x == 0].mean()),
        },
        "segments": by_seg.reset_index(names="segment").to_dict(orient="records"),
        "value_tiers": tiers,
        "new_customer": new_customer,
        "_scores": scores,
        "_lorenz": lorenz,
        "_bg": bg,
    }


def _strip(obj):
    """Drop private ``_`` keys and convert numpy scalars for JSON."""
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(round(float(obj), 6))
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def run(data_home: str | Path = "data", out: str | Path = "reports",
        raw: pd.DataFrame | None = None) -> dict:
    """Full reproduction: writes metrics.json, segments.csv and figures; returns the metrics.

    ``raw`` (invoice lines in the Online Retail layout) skips the download; tests use it.
    """
    from clv.data import load_online_retail
    from clv.figures import make_figures

    t0 = time.perf_counter()
    out = Path(out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    Path(data_home).mkdir(parents=True, exist_ok=True)
    assumptions = ValueAssumptions()
    if raw is None:
        raw = load_online_retail(data_home)
    clean, steps = clean_transactions(raw, CleaningConfig())
    clean_all, steps_all = clean_transactions(raw, CleaningConfig(country=None))
    del raw
    clean.to_parquet(Path(data_home) / "online_retail_clean_uk.parquet", index=False)
    daily = daily_transactions(clean)
    daily_all = daily_transactions(clean_all)

    validation = {s.name: validate(daily, s, assumptions, details=(s.name == "main"))
                  for s in SPLITS}
    validation["main_all_countries"] = validate(daily_all, SPLITS[0], assumptions)
    business = business_view(daily, assumptions)

    metrics = {
        "data": {
            "source": "UCI Online Retail (id 352), Chen, Sain & Guo (2012), CC BY 4.0",
            "period": [DATA_START, "2011-12-09"],
            "cleaning_steps": steps,
            "cleaning_all_countries_last_step": steps_all[-1],
            "clean_lines_uk": int(len(clean)),
            "customers_uk": int(daily["customer"].nunique()),
            "purchase_days_uk": int(len(daily)),
            "revenue_uk": float(clean["Revenue"].sum()),
            "largest_clean_line_gbp": float(clean["Revenue"].max()),
            "top5_customer_spend_gbp": sorted(
                daily.groupby("customer")["spend"].sum().nlargest(5).round(0).tolist()),
            "customers_all_countries": int(daily_all["customer"].nunique()),
        },
        "assumptions": {**asdict(assumptions),
                        "monthly_discount_rate": assumptions.monthly_discount_rate,
                        "time_unit": "week", "transaction": "customer purchase day"},
        "validation": validation,
        "full_data_model": business,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "pandas": pd.__version__},
    }
    make_figures(validation["main"], business, out / "figures")
    seg = pd.DataFrame(business["segments"])
    seg.round(4).to_csv(out / "segments.csv", index=False)
    metrics["runtime_seconds"] = round(time.perf_counter() - t0, 1)
    clean_metrics = _strip(metrics)
    (out / "metrics.json").write_text(json.dumps(clean_metrics, indent=2) + "\n")
    return clean_metrics
