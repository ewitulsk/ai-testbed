"""
Omori-Utsu aftershock decay fitting.

The modified Omori law (Utsu 1961) for the aftershock rate at time t after a
mainshock:

    n(t) = K / (t + c)^p

with p typically near 1. The consensus-seismology prediction: after a large
fork/reorg burst, the *excess* stale-block (uncle/orphan) rate relaxes as a
power law in time rather than dropping instantly -- because the strain release
perturbs mempool/propagation synchronization, which re-equilibrates gradually.

Falsification: if the post-event excess rate shows no power-law decay (p ~ 0,
i.e. flat, or a pure exponential relaxation fits far better), Omori fails.

We fit via maximum likelihood on the aftershock event times (the proper method
for point-process rate estimation; Ogata 1983), and also provide a binned
least-squares fit for plotting and an exponential-relaxation comparison.
"""
import numpy as np
from scipy.optimize import minimize


def omori_rate(t, K, c, p):
    return K / np.power(t + c, p)


def _omori_negloglik(params, t, T):
    """
    Point-process negative log-likelihood for the modified Omori law on
    aftershock times t in (0, T]. (Ogata 1983.)
        loglik = sum ln n(t_i) - integral_0^T n(t) dt
    """
    K, c, p = params
    if K <= 0 or c <= 0 or p <= 0:
        return 1e18
    ll = np.sum(np.log(K) - p * np.log(t + c))
    if abs(p - 1.0) < 1e-8:
        integ = K * (np.log(T + c) - np.log(c))
    else:
        integ = K / (1 - p) * (np.power(T + c, 1 - p) - np.power(c, 1 - p))
    return -(ll - integ)


def fit_omori_mle(event_times, T=None):
    """
    MLE fit of modified Omori law to aftershock inter-event times.
    event_times : times of aftershocks measured from the mainshock (>0).
    Returns dict with K, c, p and the fit quality.
    """
    t = np.sort(np.asarray(event_times, dtype=float))
    t = t[t > 0]
    if len(t) < 10:
        return {"K": np.nan, "c": np.nan, "p": np.nan, "n": len(t)}
    if T is None:
        T = t.max()
    best = None
    for p0 in (0.8, 1.0, 1.2):
        for c0 in (0.01 * T, 0.05 * T, 0.2 * T):
            x0 = [len(t) / T, max(c0, 1e-6), p0]
            try:
                res = minimize(_omori_negloglik, x0, args=(t, T),
                               method="Nelder-Mead",
                               options={"xatol": 1e-6, "fatol": 1e-6,
                                        "maxiter": 5000})
                if best is None or res.fun < best.fun:
                    best = res
            except Exception:
                continue
    if best is None:
        return {"K": np.nan, "c": np.nan, "p": np.nan, "n": len(t)}
    K, c, p = best.x
    return {"K": float(K), "c": float(c), "p": float(p),
            "negloglik": float(best.fun), "n": int(len(t)), "T": float(T)}


def _omori_bg_negloglik(params, t, T):
    """
    Point-process NLL for background + modified Omori:
        n(t) = mu + K/(t+c)^p     on (0, T].
    (Reasenberg-Jones / ETAS single-mainshock form.) `mu` absorbs the flat
    background of unrelated forks so `p` measures the true aftershock decay.
    """
    mu, K, c, p = params
    if mu < 0 or K <= 0 or c <= 0 or not (0.05 < p < 4.0):
        return 1e18
    rate = mu + K / np.power(t + c, p)
    if np.any(rate <= 0):
        return 1e18
    ll = np.sum(np.log(rate))
    if abs(p - 1.0) < 1e-8:
        integ_omori = K * (np.log(T + c) - np.log(c))
    else:
        integ_omori = K / (1 - p) * (np.power(T + c, 1 - p) - np.power(c, 1 - p))
    integ = mu * T + integ_omori
    return -(ll - integ)


def fit_omori_background(event_times, T=None, mu_hint=None):
    """
    MLE fit of n(t)=mu + K/(t+c)^p. Returns mu,K,c,p and AIC. Compares against a
    background-only (flat) model to confirm an aftershock excess exists at all.
    """
    from scipy.optimize import minimize
    t = np.sort(np.asarray(event_times, dtype=float))
    t = t[t > 0]
    if len(t) < 20:
        return {"error": "insufficient", "n": len(t)}
    if T is None:
        T = t.max()
    mu0 = (mu_hint if mu_hint is not None else len(t) / T)
    best = None
    for p0 in (0.7, 1.0, 1.3):
        for c0 in (0.01 * T, 0.05 * T, 0.2 * T):
            for kfac in (0.5, 2.0):
                x0 = [mu0 * 0.5, mu0 * kfac * c0, max(c0, 1e-6), p0]
                res = minimize(_omori_bg_negloglik, x0, args=(t, T),
                               method="Nelder-Mead",
                               options={"maxiter": 8000, "xatol": 1e-7,
                                        "fatol": 1e-7})
                if best is None or res.fun < best.fun:
                    best = res
    mu, K, c, p = best.x
    aic_full = 2 * 4 + 2 * best.fun
    # flat-only model: constant rate = n/T
    nll_flat = -(len(t) * np.log(len(t) / T) - len(t))
    aic_flat = 2 * 1 + 2 * nll_flat
    return {"mu": float(mu), "K": float(K), "c": float(c), "p": float(p),
            "negloglik": float(best.fun), "aic": float(aic_full),
            "aic_flat": float(aic_flat),
            "delta_aic_vs_flat": float(aic_flat - aic_full),  # >0 => aftershocks real
            "n": int(len(t)), "T": float(T)}


def binned_rate(event_times, T, nbins=20, log=True):
    """Empirical aftershock rate in bins, for plotting/LS. Returns t_mid, rate."""
    t = np.asarray(event_times, dtype=float)
    t = t[(t > 0) & (t <= T)]
    if log:
        edges = np.logspace(np.log10(max(t.min(), 1e-6)), np.log10(T), nbins + 1)
    else:
        edges = np.linspace(0, T, nbins + 1)
    counts, _ = np.histogram(t, bins=edges)
    width = np.diff(edges)
    mid = np.sqrt(edges[:-1] * edges[1:]) if log else 0.5 * (edges[:-1] + edges[1:])
    rate = counts / width
    keep = counts > 0
    return mid[keep], rate[keep]


def compare_omori_vs_exponential(event_times, T=None):
    """
    Compare the modified Omori (power-law) decay against an exponential
    relaxation n(t)=A e^{-t/tau} using AIC on the point-process likelihood.
    Returns both fits and which wins (lower AIC).
    """
    t = np.sort(np.asarray(event_times, dtype=float))
    t = t[t > 0]
    if len(t) < 10:
        return {"error": "insufficient", "n": len(t)}
    if T is None:
        T = t.max()

    om = fit_omori_mle(t, T=T)
    aic_omori = 2 * 3 + 2 * om["negloglik"]

    # exponential relaxation point process: n(t)=A e^{-t/tau}
    def exp_negloglik(params):
        A, tau = params
        if A <= 0 or tau <= 0:
            return 1e18
        ll = np.sum(np.log(A) - t / tau)
        integ = A * tau * (1 - np.exp(-T / tau))
        return -(ll - integ)

    best = None
    for tau0 in (0.05 * T, 0.2 * T, 0.5 * T, T):
        res = minimize(exp_negloglik, [len(t) / T, tau0],
                       method="Nelder-Mead",
                       options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6})
        if best is None or res.fun < best.fun:
            best = res
    aic_exp = 2 * 2 + 2 * best.fun

    winner = "omori (power-law)" if aic_omori < aic_exp else "exponential"
    return {
        "omori": om, "aic_omori": float(aic_omori),
        "exp_A": float(best.x[0]), "exp_tau": float(best.x[1]),
        "aic_exp": float(aic_exp),
        "delta_aic": float(aic_exp - aic_omori),  # >0 favours Omori
        "winner": winner,
        "p": om["p"], "n": int(len(t)),
    }


def stack_aftershocks(event_times, mainshock_times, max_lag, min_lag=0.0):
    """
    Superposed-epoch stacking: collect aftershock lags relative to each
    mainshock (used when individual sequences are too sparse to fit alone).
    Returns concatenated lags in (min_lag, max_lag].

    `min_lag` excludes the mainshock's own event window so that autocorrelation
    of the mainshock cluster itself is not mistaken for aftershock decay.
    """
    event_times = np.sort(np.asarray(event_times, dtype=float))
    lags = []
    for ms in mainshock_times:
        idx = np.searchsorted(event_times, ms + min_lag, side="right")
        j = idx
        while j < len(event_times) and event_times[j] - ms <= max_lag:
            lag = event_times[j] - ms
            if lag > min_lag:
                lags.append(lag)
            j += 1
    return np.array(lags)


def aftershock_excess(event_times, mainshock_times, min_lag, max_lag):
    """
    Model-free Omori check: mean event rate in (min_lag, max_lag] after
    mainshocks, divided by the global background rate. ~1.0 => no aftershock
    excess (Omori absent); >>1 with a decaying profile => Omori present.
    """
    event_times = np.sort(np.asarray(event_times, dtype=float))
    bg = len(event_times) / (event_times.max() - event_times.min())
    lags = stack_aftershocks(event_times, mainshock_times, max_lag, min_lag)
    span = (max_lag - min_lag) * len(mainshock_times)
    obs_rate = len(lags) / span if span > 0 else np.nan
    return {"excess_ratio": float(obs_rate / bg) if bg > 0 else np.nan,
            "bg_rate": float(bg), "obs_rate": float(obs_rate),
            "n_after": int(len(lags)), "n_main": int(len(mainshock_times))}
