"""
Theory-coherence test: run the mechanistic models through the full
seismological pipeline and check whether the predicted signatures EMERGE.

We test two regimes:
  * block-race ("creeping fault") -- latency-shock driven, expected to show
    Gutenberg-Richter depths + Omori aftershock clustering.
  * stick-slip ("locked fault")  -- strain-accumulate-and-release, expected to
    show quasi-periodic (seismic-gap) recurrence.

Outputs figures + results/model_metrics.json.
"""
import numpy as np
from common import (save, dump_json, ccdf, plt, C_DATA, C_MODEL, C_ALT,
                    C_ACCENT, C_STRAIN)
import powerlaw as pl
import bvalue as bv
import omori as om
import seismic_gap as sg
from model import (simulate_block_race, simulate_stick_slip,
                   uncle_series_from_reorgs)


def analyse_depths(depths, label):
    rep = pl.full_report(depths, n_boot=150, seed=1, label=label)
    return rep


def plot_depth_ccdf(depths, rep, title, fname):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    vals, surv = ccdf(depths)
    ax.loglog(vals, surv, "o", ms=4, color=C_DATA, label="empirical CCDF")
    # fitted power-law tail
    xmin, alpha = rep["xmin"], rep["alpha"]
    xt = np.arange(xmin, vals.max() + 1)
    # normalise PL CCDF to empirical survival at xmin
    s_at_xmin = surv[np.searchsorted(vals, xmin)]
    from scipy.special import zeta
    pl_ccdf = np.array([zeta(alpha, x) for x in xt]) / zeta(alpha, xmin)
    ax.loglog(xt, pl_ccdf * s_at_xmin, "-", color=C_MODEL, lw=2,
              label=f"power law α={alpha:.2f} (x≥{xmin})")
    # exponential comparison
    lam = rep.get("exp_lambda")
    if lam:
        exp_ccdf = np.exp(-lam * (xt - xmin))
        ax.loglog(xt, exp_ccdf * s_at_xmin, "--", color=C_ALT, lw=1.8,
                  label=f"exponential λ={lam:.2f}")
    ax.set_xlabel("reorg depth  d  (blocks orphaned)")
    ax.set_ylabel("P(D ≥ d)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    save(fig, fname)


def omori_analysis(reorgs, depth_thresh, max_lag, label, fname):
    """Stack aftershocks after large reorgs and fit the modified Omori law."""
    times = reorgs["time"]
    depths = reorgs["depth"]
    big = times[depths >= depth_thresh]
    # avoid counting a mainshock's own cluster members as separate mainshocks:
    # keep mainshocks separated by at least max_lag
    keep = []
    last = -np.inf
    for t in np.sort(big):
        if t - last >= max_lag:
            keep.append(t); last = t
    big = np.array(keep)
    lags = om.stack_aftershocks(times, big, max_lag)
    if len(lags) < 30:
        return {"error": "too few aftershocks", "n_main": len(big),
                "n_after": len(lags)}
    # background rate per mainshock per block from the global fork rate
    bg_per_main = len(times) / (times.max() - times.min())
    fit = om.fit_omori_background(lags, T=max_lag, mu_hint=bg_per_main * len(big))
    mid, rate = om.binned_rate(lags, max_lag, nbins=20, log=True)
    rate_pm = rate / len(big)
    excess = np.maximum(rate_pm - bg_per_main, 1e-9)

    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    ax.loglog(mid, excess, "o", ms=5, color=C_DATA,
              label="excess rate (above background)")
    tt = np.logspace(np.log10(mid.min()), np.log10(max_lag), 100)
    ax.loglog(tt, (fit["K"] / np.power(tt + fit["c"], fit["p"])) / len(big),
              "-", color=C_MODEL, lw=2, label=f"Omori p={fit['p']:.2f}")
    ax.axhline(bg_per_main, color=C_ALT, ls=":", lw=1.2,
               label="background level (subtracted)")
    ax.set_xlabel("time since large reorg  (blocks)")
    ax.set_ylabel("excess aftershock rate / mainshock")
    ax.set_title(f"{label}: post-reorg fork-rate decay (Omori)")
    ax.legend(frameon=False, fontsize=9)
    save(fig, fname)
    return {"n_main": int(len(big)), "n_after": int(len(lags)),
            "omori_p": fit["p"], "omori_c": fit["c"],
            "delta_aic_vs_flat": fit["delta_aic_vs_flat"],
            "aftershocks_significant": bool(fit["delta_aic_vs_flat"] > 10)}


def bvalue_strain_analysis(res, window, step, fname, label):
    reorgs = res["reorgs"]
    times, depths = reorgs["time"], reorgs["depth"]
    mags = np.log10(depths.astype(float)) + 0.0  # magnitude = log10(depth)
    # rolling b (use raw depth as integer magnitude; mc=1)
    roll = bv.rolling_b_value(times, depths.astype(float), window, step,
                              mc=1.0, dm=1.0, min_events=40)
    # strain sampled at rolling centers
    strain_interp = np.interp(roll["t_center"], res["strain_t"], res["strain_v"])
    # correlation between b and strain
    good = np.isfinite(roll["b"]) & np.isfinite(strain_interp)
    if good.sum() > 5:
        from scipy.stats import pearsonr, spearmanr
        r_p, p_p = pearsonr(roll["b"][good], strain_interp[good])
        r_s, p_s = spearmanr(roll["b"][good], strain_interp[good])
    else:
        r_p = p_p = r_s = p_s = np.nan

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True)
    a1.plot(roll["t_center"], roll["b"], color=C_MODEL, lw=1.4)
    a1.fill_between(roll["t_center"], roll["b"] - roll["se"],
                    roll["b"] + roll["se"], color=C_MODEL, alpha=0.15)
    a1.set_ylabel("consensus b-value")
    a1.set_title(f"{label}: rolling b-value vs strain  "
                 f"(Spearman r={r_s:.2f}, p={p_s:.1e})")
    a2.plot(res["strain_t"], res["strain_v"], color=C_STRAIN, lw=0.8)
    a2.set_ylabel("strain (mean cross-cluster\nlatency, block-times)")
    a2.set_xlabel("simulated time (blocks)")
    save(fig, fname)
    return {"pearson_r": float(r_p), "pearson_p": float(p_p),
            "spearman_r": float(r_s), "spearman_p": float(p_s),
            "n_windows": int(good.sum()),
            "mean_b": float(np.nanmean(roll["b"]))}


def main():
    metrics = {}

    # ---------------- block-race (creeping fault) ----------------
    print("[block-race] simulating...")
    br = simulate_block_race(duration=3_000_000, lam=1.0, f=0.5,
                             base_delay=0.15, shock_rate=2e-3, amp_mean=1.2,
                             tau_min=5, tau_max=3000, seed=7)
    d = br["reorgs"]["depth"]
    print(f"  n_blocks={br['n_blocks']} n_reorgs={len(d)} "
          f"rate={100*len(d)/br['n_blocks']:.2f}% maxdepth={d.max()}")
    rep = analyse_depths(d, "block-race depths")
    metrics["block_race_depths"] = rep
    print(f"  GR fit: alpha={rep['alpha']:.2f} verdict={rep['verdict']}")
    plot_depth_ccdf(d, rep, "Creeping fault (block-race): reorg-depth distribution",
                    "model_blockrace_depth_ccdf.png")

    metrics["block_race_omori"] = omori_analysis(
        br["reorgs"], depth_thresh=max(3, int(np.percentile(d, 99))),
        max_lag=1500, label="Creeping fault",
        fname="model_blockrace_omori.png")
    print(f"  Omori: {metrics['block_race_omori']}")

    metrics["block_race_bvalue"] = bvalue_strain_analysis(
        br, window=60000, step=15000,
        fname="model_blockrace_bvalue.png", label="Creeping fault")
    print(f"  b-vs-strain: {metrics['block_race_bvalue']}")

    metrics["block_race_gap"] = sg.gap_test(
        br["reorgs"]["time"][d >= max(3, int(np.percentile(d, 99)))])
    print(f"  recurrence: {metrics['block_race_gap']}")

    # ---------------- stick-slip (locked fault) ----------------
    print("[stick-slip] simulating...")
    ss = simulate_stick_slip(n_steps=2_000_000, load_rate=1.0, noise=0.15,
                             threshold=120.0, release_frac=0.9,
                             depth_scale=0.12, seed=3)
    d2 = ss["reorgs"]["depth"]
    print(f"  n_reorgs={len(d2)} maxdepth={d2.max()}")
    rep2 = analyse_depths(d2, "stick-slip depths")
    metrics["stick_slip_depths"] = rep2
    plot_depth_ccdf(d2, rep2, "Locked fault (stick-slip): reorg-depth distribution",
                    "model_stickslip_depth_ccdf.png")
    metrics["stick_slip_gap"] = sg.gap_test(ss["reorgs"]["time"])
    print(f"  recurrence: {metrics['stick_slip_gap']}")

    # contrast recurrence plot
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for res, lab, col in [(br, "creeping (block-race)", C_DATA),
                          (ss, "locked (stick-slip)", C_ACCENT)]:
        dd = res["reorgs"]["depth"]
        thr = max(3, int(np.percentile(dd, 99))) if lab.startswith("creep") else 1
        tt = np.sort(res["reorgs"]["time"][dd >= thr])
        if len(tt) > 3:
            dti = np.diff(tt)
            cv = dti.std(ddof=1) / dti.mean()
            ax.hist(dti / dti.mean(), bins=40, histtype="step", lw=2,
                    color=col, density=True, label=f"{lab}  CV={cv:.2f}")
    xx = np.linspace(0, 5, 100)
    ax.plot(xx, np.exp(-xx), "k--", lw=1, alpha=0.6, label="Poisson (CV=1)")
    ax.set_xlabel("normalised inter-event time  Δt / ⟨Δt⟩")
    ax.set_ylabel("density")
    ax.set_title("Recurrence of large reorgs: clustered vs gap-like")
    ax.legend(frameon=False, fontsize=9)
    save(fig, "model_recurrence_contrast.png")

    dump_json(metrics, "model_metrics.json")
    print("\n[model] done.")


if __name__ == "__main__":
    main()
