"""
Power-law fitting and model comparison for discrete data, following
Clauset, Shalizi & Newman (2009), "Power-law distributions in empirical data",
SIAM Review 51(4):661-703.

This module is the falsifiability core of the consensus-seismology theory.
Gutenberg-Richter predicts event *magnitudes* are power-law distributed. The
null we must reject is that they are merely exponential (a memory-less,
non-critical process). We therefore:

  1. MLE-fit a discrete power law  p(x) ~ x^{-alpha}  for x >= xmin.
  2. Choose xmin by minimizing the KS distance (Clauset method).
  3. Bootstrap a goodness-of-fit p-value for the power law.
  4. Fit the competing exponential and compute Vuong's normalized
     log-likelihood ratio R (with p-value) to say which fits better.

A theory-supporting result is: power law not rejected (p_fit high) AND
loglik-ratio favors power law over exponential (R>0, significant).
A theory-killing result is: exponential strongly preferred, or power law
rejected outright.
"""
import numpy as np
from scipy.special import zeta
from scipy.optimize import brentq


# ----------------------------------------------------------------------------
# Discrete power law  p(x) = x^{-alpha} / zeta(alpha, xmin),  x = xmin, xmin+1...
# ----------------------------------------------------------------------------
def _hurwitz_zeta(alpha, xmin):
    # scipy.special.zeta(a, q) = sum_{n=0}^inf (q+n)^{-a} = Hurwitz zeta
    return zeta(alpha, xmin)


def discrete_pl_negloglik(alpha, x, xmin):
    n = len(x)
    return n * np.log(_hurwitz_zeta(alpha, xmin)) + alpha * np.sum(np.log(x))


def fit_discrete_powerlaw_alpha(x, xmin):
    """MLE of alpha for discrete power law with lower cutoff xmin."""
    x = np.asarray(x)
    x = x[x >= xmin]
    if len(x) < 2:
        return np.nan
    # Objective derivative: minimize negloglik over alpha in (1, ~6]
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(
        lambda a: discrete_pl_negloglik(a, x, xmin),
        bounds=(1.01, 6.0), method="bounded")
    return float(res.x)


def _discrete_pl_cdf(alpha, xmin, xmax):
    xs = np.arange(xmin, xmax + 1)
    pmf = xs.astype(float) ** (-alpha) / _hurwitz_zeta(alpha, xmin)
    return xs, np.cumsum(pmf)


def ks_distance_discrete(x, alpha, xmin):
    """KS distance between empirical and fitted power-law CDF, x>=xmin."""
    x = np.sort(np.asarray(x))
    x = x[x >= xmin]
    if len(x) < 2:
        return np.inf
    xmax = int(x.max())
    xs, cdf_model = _discrete_pl_cdf(alpha, xmin, xmax)
    cdf_map = dict(zip(xs, cdf_model))
    # empirical CDF evaluated at each distinct value
    vals, counts = np.unique(x, return_counts=True)
    emp = np.cumsum(counts) / len(x)
    mod = np.array([cdf_map[v] for v in vals])
    return float(np.max(np.abs(emp - mod)))


def fit_powerlaw(x, xmin_range=None):
    """
    Full Clauset fit: pick xmin minimizing KS distance, return alpha, xmin, D, n.
    """
    x = np.asarray(x)
    x = x[x >= 1]
    uniq = np.unique(x)
    if xmin_range is None:
        # candidate xmins: all distinct values except the largest few
        xmin_range = uniq[uniq <= np.percentile(uniq, 95)]
        if len(xmin_range) == 0:
            xmin_range = uniq[:1]
    best = None
    for xmin in xmin_range:
        xtail = x[x >= xmin]
        if len(xtail) < 10:
            continue
        alpha = fit_discrete_powerlaw_alpha(xtail, xmin)
        if not np.isfinite(alpha):
            continue
        D = ks_distance_discrete(xtail, alpha, xmin)
        if best is None or D < best["D"]:
            best = {"alpha": alpha, "xmin": int(xmin), "D": D,
                    "n_tail": int(len(xtail))}
    return best


# ----------------------------------------------------------------------------
# Goodness-of-fit bootstrap p-value (Clauset section 4)
# ----------------------------------------------------------------------------
def _sample_discrete_pl(alpha, xmin, size, rng, xmax=100000):
    xs, cdf = _discrete_pl_cdf(alpha, xmin, xmax)
    u = rng.random(size)
    idx = np.searchsorted(cdf, u)
    idx = np.clip(idx, 0, len(xs) - 1)
    return xs[idx]


def gof_pvalue(x, fit, n_boot=200, seed=0):
    """
    Fraction of synthetic datasets (drawn from the fitted power law) whose KS
    distance exceeds the empirical KS distance. p >= 0.1 => power law is a
    plausible fit (not rejected).
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    xmin, alpha, D_emp = fit["xmin"], fit["alpha"], fit["D"]
    ntail = np.sum(x >= xmin)
    n_nontail = np.sum(x < xmin)
    p_tail = ntail / len(x)
    nontail_vals = x[x < xmin]
    count = 0
    for _ in range(n_boot):
        # mixture: with prob p_tail draw from PL tail, else resample non-tail
        n = len(x)
        is_tail = rng.random(n) < p_tail
        n_t = int(np.sum(is_tail))
        synth = np.empty(n)
        if n_t > 0:
            synth[:n_t] = _sample_discrete_pl(alpha, xmin, n_t, rng)
        if n - n_t > 0:
            if len(nontail_vals) > 0:
                synth[n_t:] = rng.choice(nontail_vals, size=n - n_t, replace=True)
            else:
                synth[n_t:] = xmin - 1
        sfit_alpha = fit_discrete_powerlaw_alpha(synth[synth >= xmin], xmin)
        if not np.isfinite(sfit_alpha):
            continue
        D_synth = ks_distance_discrete(synth, sfit_alpha, xmin)
        if D_synth >= D_emp:
            count += 1
    return count / n_boot


# ----------------------------------------------------------------------------
# Exponential alternative + Vuong likelihood-ratio test (Clauset section 5)
# ----------------------------------------------------------------------------
def fit_discrete_exponential_lambda(x, xmin):
    """MLE for discrete (geometric-type) exponential p(x) ~ e^{-lambda x}, x>=xmin."""
    x = np.asarray(x)
    x = x[x >= xmin]
    # For discrete exp on {xmin, xmin+1, ...}: mean = xmin + 1/(e^lambda - 1)
    m = x.mean()
    # solve  m = xmin + 1/(e^l - 1)
    denom = m - xmin
    if denom <= 0:
        return np.nan
    lam = np.log(1 + 1.0 / denom)
    return float(lam)


def _loglik_pl_pointwise(x, alpha, xmin):
    return -alpha * np.log(x) - np.log(_hurwitz_zeta(alpha, xmin))


def _loglik_exp_pointwise(x, lam, xmin):
    # normalized discrete exponential on x>=xmin:
    # p(x) = (1 - e^{-lam}) e^{-lam (x - xmin)}
    return np.log(1 - np.exp(-lam)) - lam * (x - xmin)


def vuong_test(x, fit):
    """
    Vuong's test comparing power law vs exponential on the tail x>=xmin.
    Returns R (normalized loglik ratio; >0 favors power law) and two-sided p.
    """
    from scipy.stats import norm
    x = np.asarray(x)
    xmin, alpha = fit["xmin"], fit["alpha"]
    xt = x[x >= xmin].astype(float)
    lam = fit_discrete_exponential_lambda(xt, xmin)
    if not np.isfinite(lam):
        return {"R": np.nan, "p": np.nan, "lambda": np.nan}
    ll_pl = _loglik_pl_pointwise(xt, alpha, xmin)
    ll_exp = _loglik_exp_pointwise(xt, lam, xmin)
    diff = ll_pl - ll_exp
    n = len(diff)
    lr = diff.sum()
    sigma = diff.std(ddof=1)
    if sigma == 0:
        return {"R": np.sign(lr) * np.inf, "p": 0.0, "lambda": lam}
    R = lr / (np.sqrt(n) * sigma)
    p = 2 * norm.sf(abs(R))
    return {"R": float(R), "p": float(p), "lambda": float(lam),
            "loglik_pl": float(ll_pl.sum()), "loglik_exp": float(ll_exp.sum())}


def full_report(x, n_boot=200, seed=0, label=""):
    """Convenience: fit PL, GoF p-value, exponential comparison, verdict."""
    x = np.asarray(x)
    x = x[x >= 1]
    fit = fit_powerlaw(x)
    if fit is None:
        return {"label": label, "error": "insufficient data", "n": int(len(x))}
    p_fit = gof_pvalue(x, fit, n_boot=n_boot, seed=seed)
    vt = vuong_test(x, fit)
    pl_plausible = p_fit >= 0.10
    exp_preferred = (vt["R"] < 0) and (vt["p"] < 0.05)
    pl_preferred = (vt["R"] > 0) and (vt["p"] < 0.05)
    if pl_plausible and pl_preferred:
        verdict = "SUPPORTS power law (GR-consistent)"
    elif not pl_plausible and exp_preferred:
        verdict = "REJECTS power law (favours exponential -> theory fails)"
    elif exp_preferred:
        verdict = "leans exponential"
    elif pl_preferred:
        verdict = "leans power law"
    else:
        verdict = "inconclusive"
    return {
        "label": label, "n": int(len(x)),
        "alpha": fit["alpha"], "xmin": fit["xmin"], "D": fit["D"],
        "n_tail": fit["n_tail"], "p_fit": p_fit,
        "vuong_R": vt["R"], "vuong_p": vt["p"], "exp_lambda": vt.get("lambda"),
        "verdict": verdict,
    }
