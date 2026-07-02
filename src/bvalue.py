"""
Gutenberg-Richter b-value estimation -- the "consensus strain index".

Gutenberg-Richter:  log10 N(>=M) = a - b*M
The b-value is the slope: high b => many small events relative to large ones
(a "creeping" fault); low b => relatively more large events (a "locked" fault
accumulating strain). Seismology's central empirical claim is that b DROPS in
the run-up to a large earthquake as stress concentrates.

The consensus-seismology analog: compute a rolling b-value from a stream of
reorg/fork "magnitudes". Prediction: b falls before the largest fork bursts.

We use the Aki (1965) / Utsu maximum-likelihood estimator, which for
magnitudes with completeness threshold Mc and binning dM is:

    b = log10(e) / (mean(M) - (Mc - dM/2))

with the Shi & Bolt (1982) standard error. For integer "depth"/"size"
magnitudes we set dM=1.
"""
import numpy as np


LOG10E = np.log10(np.e)


def aki_b_value(mags, mc=None, dm=1.0):
    """
    Maximum-likelihood b-value (Aki-Utsu) for magnitudes >= mc.
    Returns dict with b, a, se(b), n.
    """
    mags = np.asarray(mags, dtype=float)
    if mc is None:
        mc = mags.min()
    m = mags[mags >= mc]
    n = len(m)
    if n < 2:
        return {"b": np.nan, "a": np.nan, "se": np.nan, "n": n, "mc": mc}
    mbar = m.mean()
    denom = mbar - (mc - dm / 2.0)
    if denom <= 0:
        return {"b": np.nan, "a": np.nan, "se": np.nan, "n": n, "mc": mc}
    b = LOG10E / denom
    # Shi & Bolt (1982) standard error
    var = np.sum((m - mbar) ** 2) / (n * (n - 1))
    se = 2.30 * b ** 2 * np.sqrt(var)
    # a-value from  log10 N(>=mc) = a - b*mc
    a = np.log10(n) + b * mc
    return {"b": float(b), "a": float(a), "se": float(se), "n": int(n),
            "mc": float(mc)}


def gr_curve(mags, mc=None):
    """Return (magnitude bins, log10 cumulative counts) for a GR plot."""
    mags = np.asarray(mags, dtype=float)
    if mc is not None:
        mags = mags[mags >= mc]
    vals = np.unique(mags)
    cum = np.array([(mags >= v).sum() for v in vals], dtype=float)
    return vals, np.log10(cum)


def rolling_b_value(times, mags, window, step, mc=None, dm=1.0, min_events=25):
    """
    Rolling b-value over a moving time window.

    times : event times (sorted ascending), same length as mags
    mags  : event magnitudes
    window: window width in the same units as times
    step  : advance between windows

    Returns dict of arrays: t_center, b, se, n.
    """
    times = np.asarray(times, dtype=float)
    mags = np.asarray(mags, dtype=float)
    order = np.argsort(times)
    times, mags = times[order], mags[order]
    if mc is None:
        mc = mags.min()
    t0, t1 = times[0], times[-1]
    centers, bs, ses, ns = [], [], [], []
    left = t0
    while left + window <= t1 + step:
        right = left + window
        sel = (times >= left) & (times < right)
        m = mags[sel]
        if len(m) >= min_events:
            r = aki_b_value(m, mc=mc, dm=dm)
            centers.append(left + window / 2)
            bs.append(r["b"]); ses.append(r["se"]); ns.append(r["n"])
        left += step
    return {"t_center": np.array(centers), "b": np.array(bs),
            "se": np.array(ses), "n": np.array(ns), "mc": mc}


def foreshock_b_drop(times, mags, big_event_times, pre_window, ref_window,
                     mc=None, dm=1.0, min_events=20):
    """
    For each large event, compare the b-value in the pre_window immediately
    before it against a reference (the ref_window before the pre_window).
    Returns per-event (b_pre, b_ref, delta) and the aggregate mean delta.

    Prediction (theory): b_pre < b_ref (b drops before big ruptures) => delta<0.
    """
    times = np.asarray(times, dtype=float)
    mags = np.asarray(mags, dtype=float)
    if mc is None:
        mc = mags.min()
    rows = []
    for te in big_event_times:
        pre = (times >= te - pre_window) & (times < te)
        ref = (times >= te - pre_window - ref_window) & (times < te - pre_window)
        mp, mr = mags[pre], mags[ref]
        if len(mp) < min_events or len(mr) < min_events:
            continue
        bp = aki_b_value(mp, mc=mc, dm=dm)["b"]
        br = aki_b_value(mr, mc=mc, dm=dm)["b"]
        if np.isfinite(bp) and np.isfinite(br):
            rows.append((te, bp, br, bp - br))
    rows = np.array(rows)
    if len(rows) == 0:
        return {"rows": rows, "mean_delta": np.nan, "n": 0}
    deltas = rows[:, 3]
    # one-sample sign test / t-test that mean delta < 0
    from scipy.stats import ttest_1samp, wilcoxon
    t_p = ttest_1samp(deltas, 0.0).pvalue if len(deltas) > 1 else np.nan
    try:
        w_p = wilcoxon(deltas).pvalue if len(deltas) > 1 else np.nan
    except ValueError:
        w_p = np.nan
    return {"rows": rows, "mean_delta": float(deltas.mean()),
            "frac_negative": float(np.mean(deltas < 0)),
            "ttest_p": float(t_p), "wilcoxon_p": float(w_p), "n": len(rows)}
