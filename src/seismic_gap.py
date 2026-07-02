"""
Seismic-gap / recurrence analysis for consensus strain release.

The seismic-gap hypothesis (McCann et al. 1979; Fedotov): a fault segment that
has produced large events historically but has gone quiet is *more* likely to
rupture next, because strain keeps accumulating while it is locked. Its temporal
signatures are:

  (1) Quasi-periodic recurrence of large events on a given fault -- inter-event
      times are more regular than a memory-less Poisson process. Measured by the
      coefficient of variation CV = std/mean of recurrence intervals: CV<1 =>
      more regular than Poisson (gap-like); CV=1 => Poisson; CV>1 => clustered.

  (2) Quiescence: a measurable drop in activity on a fault before its large
      event (the flip side of the b-value drop).

For a single blockchain we cannot attribute reorgs to distinct latency
boundaries, so (1)/(2) are tested per-fault only in the multi-fault model. On
real single-chain data we test the *aggregate* recurrence regularity of the
largest events, which distinguishes a gap/creep process (CV<1) from pure
aftershock clustering (CV>1).
"""
import numpy as np


def recurrence_cv(event_times):
    """Coefficient of variation of inter-event times. CV<1 regular, >1 clustered."""
    t = np.sort(np.asarray(event_times, dtype=float))
    if len(t) < 3:
        return {"cv": np.nan, "mean": np.nan, "n": len(t)}
    dt = np.diff(t)
    cv = dt.std(ddof=1) / dt.mean() if dt.mean() > 0 else np.nan
    return {"cv": float(cv), "mean_interval": float(dt.mean()),
            "n_intervals": int(len(dt))}


def poisson_cv_null(n_intervals, n_boot=2000, seed=0):
    """
    Null distribution of CV for a Poisson process (exponential intervals) with
    the same count, to test whether an observed CV is significantly <1 or >1.
    """
    rng = np.random.default_rng(seed)
    cvs = np.empty(n_boot)
    for i in range(n_boot):
        dt = rng.exponential(1.0, size=n_intervals)
        cvs[i] = dt.std(ddof=1) / dt.mean()
    return cvs


def gap_test(event_times, n_boot=2000, seed=0):
    """
    Test observed recurrence CV against the Poisson null.
    Returns CV, the null 5-95% band, and a two-sided p-value.
    """
    rc = recurrence_cv(event_times)
    if not np.isfinite(rc["cv"]):
        return {**rc, "p": np.nan}
    null = poisson_cv_null(rc["n_intervals"], n_boot=n_boot, seed=seed)
    p_low = np.mean(null <= rc["cv"])
    p_high = np.mean(null >= rc["cv"])
    p = 2 * min(p_low, p_high)
    interp = ("regular/gap-like (CV<1)" if rc["cv"] < 1 else
              "clustered/aftershock-like (CV>1)")
    return {**rc, "null_p05": float(np.percentile(null, 5)),
            "null_p95": float(np.percentile(null, 95)),
            "p": float(min(p, 1.0)), "interpretation": interp}


def quiescence_before_events(rate_t, rate_v, big_event_times,
                             quiet_window, ref_window):
    """
    For each big event, compare the mean activity rate in the quiet_window just
    before it vs the ref_window before that. Negative ratio-1 => quiescence.
    """
    rate_t = np.asarray(rate_t, dtype=float)
    rate_v = np.asarray(rate_v, dtype=float)
    rows = []
    for te in big_event_times:
        q = (rate_t >= te - quiet_window) & (rate_t < te)
        r = (rate_t >= te - quiet_window - ref_window) & (rate_t < te - quiet_window)
        if q.sum() < 3 or r.sum() < 3:
            continue
        mq, mr = rate_v[q].mean(), rate_v[r].mean()
        if mr > 0:
            rows.append((te, mq, mr, mq / mr - 1.0))
    rows = np.array(rows)
    if len(rows) == 0:
        return {"n": 0, "mean_change": np.nan}
    ch = rows[:, 3]
    return {"n": len(rows), "mean_change": float(ch.mean()),
            "frac_quiescent": float(np.mean(ch < 0)), "rows": rows}
