"""
Confront the consensus-seismology theory with real pre-merge Ethereum data.

Data: ~1M canonical blocks (12.4M-13.4M, ~Apr-Oct 2021) and every uncle
(stale block) they reference. Uncles are organic forks that resolved.

Tests:
  P1 (Gutenberg-Richter): are fork "magnitudes" power-law or exponential?
     Magnitude proxies: (a) uncle inclusion distance (depth analog, capped at 6);
     (b) uncle-burst size = number of stale blocks in a short sliding window
     (a strain-release cluster analog with real dynamic range).
  P2 (Omori): after large uncle bursts, does the excess uncle rate decay ~1/t^p?
  P3 (b-value strain index): rolling b-value; does it drop before big bursts?
     Recurrence CV of large bursts: gap-like (<1) or aftershock-clustered (>1)?
  Natural experiment: the London hard fork / EIP-1559 (block 12,965,000) changed
     block economics -- did the consensus dynamics measurably shift across it?
"""
import numpy as np
import pandas as pd
from common import (save, dump_json, ccdf, load_eth_blocks, load_eth_uncles,
                    plt, C_DATA, C_MODEL, C_ALT, C_ACCENT, C_STRAIN)
import powerlaw as pl
import bvalue as bv
import omori as om
import seismic_gap as sg

LONDON_BLOCK = 12_965_000


def burst_series(block_num, ts, n_uncles, win_blocks=50):
    """
    Uncle-burst size: total uncles in a sliding window of `win_blocks`, sampled
    once per non-overlapping window. Timestamps taken at window centre.
    Returns (window_center_time, window_center_block, burst_size).
    """
    order = np.argsort(block_num)
    block_num, ts, n_uncles = block_num[order], ts[order], n_uncles[order]
    b0, b1 = block_num.min(), block_num.max()
    edges = np.arange(b0, b1 + win_blocks, win_blocks)
    idx = np.searchsorted(edges, block_num, side="right") - 1
    nb = len(edges) - 1
    burst = np.zeros(nb, dtype=int)
    tsum = np.zeros(nb); tcnt = np.zeros(nb)
    for i, w in enumerate(idx):
        if 0 <= w < nb:
            burst[w] += n_uncles[i]
            tsum[w] += ts[i]; tcnt[w] += 1
    keep = tcnt > 0
    tc = tsum[keep] / tcnt[keep]
    bc = (edges[:-1] + win_blocks // 2)[keep]
    return tc, bc, burst[keep]


def main():
    metrics = {"data": {}}
    blocks = load_eth_blocks()
    uncles = load_eth_uncles()
    bn = blocks["number"].to_numpy()
    ts = blocks["timestamp"].to_numpy()
    nu = blocks["n_uncles"].to_numpy()
    metrics["data"] = {
        "n_blocks": int(len(blocks)),
        "block_range": [int(bn.min()), int(bn.max())],
        "date_range": [pd.to_datetime(ts.min(), unit="s").isoformat(),
                       pd.to_datetime(ts.max(), unit="s").isoformat()],
        "n_uncles": int(nu.sum()),
        "uncle_rate_pct": float(100 * nu.sum() / len(blocks)),
        "mean_block_time_s": float(np.diff(np.sort(ts)).mean()),
    }
    print("data:", metrics["data"])

    # ---------------- P1a: uncle inclusion distance ----------------
    dist = uncles["distance"].to_numpy()
    dist = dist[dist >= 1]
    vals, cnts = np.unique(dist, return_counts=True)
    metrics["P1a_distance_hist"] = dict(zip(vals.tolist(), cnts.tolist()))
    rep_dist = pl.full_report(dist, n_boot=150, seed=1, label="uncle distance")
    metrics["P1a_distance_fit"] = rep_dist
    print("P1a uncle-distance:", rep_dist["verdict"],
          "alpha=%.2f" % rep_dist["alpha"], "(capped at 6 by protocol)")

    # ---------------- P1b: uncle-burst size ----------------
    tc, bc, burst = burst_series(bn, ts, nu, win_blocks=50)
    burst_nz = burst[burst >= 1]
    rep_burst = pl.full_report(burst_nz, n_boot=150, seed=2,
                               label="uncle-burst size")
    metrics["P1b_burst_fit"] = rep_burst
    metrics["P1b_burst_max"] = int(burst.max())
    print("P1b uncle-burst:", rep_burst["verdict"],
          "alpha=%.2f xmin=%d maxburst=%d" %
          (rep_burst["alpha"], rep_burst["xmin"], burst.max()))

    # figure: both magnitude CCDFs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, data, rep, name in [
        (ax1, dist, rep_dist, "uncle inclusion distance (depth analog)"),
        (ax2, burst_nz, rep_burst, "uncle-burst size (50-block window)")]:
        v, s = ccdf(data)
        ax.loglog(v, s, "o", ms=4, color=C_DATA, label="empirical CCDF")
        xmin, alpha = rep["xmin"], rep["alpha"]
        from scipy.special import zeta
        xt = np.arange(xmin, v.max() + 1)
        s_at = s[np.searchsorted(v, xmin)]
        plc = np.array([zeta(alpha, x) for x in xt]) / zeta(alpha, xmin)
        ax.loglog(xt, plc * s_at, "-", color=C_MODEL, lw=2,
                  label=f"power law α={alpha:.2f}")
        lam = rep.get("exp_lambda")
        if lam:
            ax.loglog(xt, np.exp(-lam * (xt - xmin)) * s_at, "--",
                      color=C_ALT, lw=1.8, label=f"exponential λ={lam:.2f}")
        ax.set_xlabel(name); ax.set_ylabel("P(X ≥ x)")
        ax.legend(frameon=False, fontsize=8.5)
        ax.set_title(rep["verdict"], fontsize=9.5)
    fig.suptitle("P1 Gutenberg-Richter test: fork magnitude distribution (real ETH)",
                 y=1.02, fontsize=11)
    save(fig, "eth_P1_magnitude_ccdf.png")

    # ---------------- P2: Omori aftershocks ----------------
    # All times are absolute epoch seconds; Omori uses lags (base-invariant).
    # events = individual uncles placed at their nephew's timestamp
    u_neph = uncles["nephew"].to_numpy()
    ts_by_block = dict(zip(bn.tolist(), ts.tolist()))
    u_time = np.array([ts_by_block.get(int(b), np.nan) for b in u_neph])
    u_time = np.sort(u_time[np.isfinite(u_time)])

    # mainshocks = large uncle bursts (burst-series peaks), absolute times.
    thr = np.percentile(burst[burst > 0], 99)
    thr = max(thr, 3)
    big_mask = burst >= thr
    big_times_raw = np.sort(tc[big_mask])   # for recurrence (no artefacts)
    # de-cluster mainshocks so aftershock windows don't overlap the next one
    decluster_sep = 12 * 3600.0  # 12 hours (used ONLY for Omori stacking)
    keep = []; last = -np.inf
    for t in big_times_raw:
        if t - last >= decluster_sep:
            keep.append(t); last = t
    big_times = np.array(keep)

    # Exclude the mainshock's own burst window (~50 blocks) so we measure the
    # *aftershock* rate, not the burst's internal autocorrelation. Use a long
    # 12 h horizon to see any relaxation.
    max_lag = 12 * 3600.0
    burst_window_s = 50 * metrics["data"]["mean_block_time_s"]
    bg_rate = len(u_time) / (u_time.max() - u_time.min())  # uncles/sec
    exc = om.aftershock_excess(u_time, big_times, min_lag=burst_window_s,
                               max_lag=max_lag)
    lags = om.stack_aftershocks(u_time, big_times, max_lag, min_lag=burst_window_s)
    fito = om.fit_omori_background(lags, T=max_lag,
                                   mu_hint=bg_rate * len(big_times))
    mid, rate = om.binned_rate(lags, max_lag, nbins=16, log=True)
    rate_pm = rate / len(big_times)
    # model-free decay slope of the post-burst rate
    m = rate_pm > 0
    slope = float(np.polyfit(np.log(mid[m]), np.log(rate_pm[m]), 1)[0]) \
        if m.sum() > 4 else np.nan
    omori_present = (exc["excess_ratio"] > 1.3) and (slope < -0.3)
    metrics["P2_omori"] = {
        "n_mainshocks": int(len(big_times)), "burst_threshold": float(thr),
        "n_aftershocks": int(len(lags)),
        "excess_ratio_over_background": exc["excess_ratio"],
        "post_burst_rate_slope": slope,
        "omori_p_fit": fito.get("p"),
        "verdict": ("Omori decay present" if omori_present else
                    "NO Omori decay -- rate returns to background (P2 falsified)"),
    }
    print("P2 Omori:", metrics["P2_omori"])

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.semilogx(mid, rate_pm, "o-", ms=5, color=C_DATA,
                label="post-burst uncle rate")
    ax.axhline(bg_rate, color=C_MODEL, ls="--", lw=1.6,
               label=f"background rate ({bg_rate*3600:.1f}/h)")
    # what an Omori aftershock sequence WOULD look like (illustrative p=1)
    tt = np.logspace(np.log10(mid.min()), np.log10(max_lag), 100)
    ax.semilogx(tt, bg_rate * (1 + 3 * (tt / burst_window_s) ** -1.0), ":",
                color=C_ACCENT, lw=1.6, label="Omori p=1 (would-be, illustrative)")
    ax.set_xlabel("time since large uncle burst  (s)")
    ax.set_ylabel("uncle rate / mainshock  (1/s)")
    ax.set_title(f"P2: post-burst rate returns to background "
                 f"(excess×{exc['excess_ratio']:.2f}, slope {slope:+.2f})")
    ax.legend(frameon=False, fontsize=8.5)
    save(fig, "eth_P2_omori.png")

    # ---------------- P3: rolling b-value strain index ----------------
    # magnitude = burst size; time = burst-window center time
    order = np.argsort(tc)
    tct, bcb, burb = tc[order], bc[order], burst[order]
    mag_t = tct[burb >= 1]                 # absolute epoch seconds
    mag = burb[burb >= 1].astype(float)
    day = 86400.0
    roll = bv.rolling_b_value(mag_t, mag, window=7 * day, step=1 * day,
                              mc=1.0, dm=1.0, min_events=60)
    metrics["P3_bvalue"] = {
        "mean_b": float(np.nanmean(roll["b"])),
        "std_b": float(np.nanstd(roll["b"])),
        "n_windows": int(len(roll["b"])),
    }

    # foreshock b-drop before the largest bursts (all absolute epoch seconds)
    fb = bv.foreshock_b_drop(mag_t, mag, big_times,
                             pre_window=3 * day, ref_window=10 * day,
                             mc=1.0, dm=1.0, min_events=40)
    metrics["P3_foreshock_bdrop"] = {k: (v if not isinstance(v, np.ndarray)
                                         else None)
                                     for k, v in fb.items() if k != "rows"}
    print("P3 b-value:", metrics["P3_bvalue"])
    print("P3 foreshock b-drop:", metrics["P3_foreshock_bdrop"])

    # recurrence CV of large bursts -- use RAW (undeclustered) times, since
    # declustering imposes a minimum spacing that artificially lowers CV.
    gap = sg.gap_test(big_times_raw)
    metrics["P3_recurrence"] = {k: v for k, v in gap.items() if k != "rows"}
    print("P3 recurrence (raw bursts):", metrics["P3_recurrence"])

    # figure: strain index over time (convert absolute epoch -> days from start)
    t0 = ts.min()
    roll_days = (roll["t_center"] - t0) / day
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    a1.plot(roll_days, roll["b"], color=C_STRAIN, lw=1.5)
    a1.fill_between(roll_days, roll["b"] - roll["se"], roll["b"] + roll["se"],
                    color=C_STRAIN, alpha=0.15)
    a1.set_ylabel("consensus b-value\n(7-day window)")
    a1.set_title("P3 Consensus strain index (rolling Gutenberg-Richter b), real ETH")
    london_day = (ts_by_block.get(LONDON_BLOCK) - t0) / day \
        if LONDON_BLOCK in ts_by_block else None
    for te in big_times:
        a2.axvline((te - t0) / day, color=C_ACCENT, alpha=0.35, lw=0.8)
    # daily uncle rate
    dfb = pd.DataFrame({"day": ((ts - t0) // day).astype(int), "nu": nu})
    daily = dfb.groupby("day")["nu"].mean() * (86400 / metrics["data"]["mean_block_time_s"])
    a2.plot(daily.index, daily.values, color=C_DATA, lw=1.0)
    a2.set_ylabel("uncles / day")
    a2.set_xlabel("days since %s" % pd.to_datetime(t0, unit="s").date())
    for ax in (a1, a2):
        if london_day is not None:
            ax.axvline(london_day, color=C_MODEL, ls="--", lw=1.6)
    if london_day is not None:
        a1.text(london_day, np.nanmax(roll["b"]), " London/EIP-1559",
                color=C_MODEL, fontsize=8.5, va="top")
    save(fig, "eth_P3_strain_index.png")

    # ---------------- Natural experiment: London hard fork ----------------
    if LONDON_BLOCK in ts_by_block:
        tL = ts_by_block[LONDON_BLOCK]
        pre = (ts < tL) & (ts >= tL - 30 * day)
        post = (ts >= tL) & (ts < tL + 30 * day)
        pre_rate = 100 * nu[pre].sum() / max(pre.sum(), 1)
        post_rate = 100 * nu[post].sum() / max(post.sum(), 1)
        # b-value pre/post using bursts
        preb = (mag_t < tL) & (mag_t >= tL - 30 * day)
        postb = (mag_t >= tL) & (mag_t < tL + 30 * day)
        b_pre = bv.aki_b_value(mag[preb], mc=1.0)["b"] if preb.sum() > 40 else np.nan
        b_post = bv.aki_b_value(mag[postb], mc=1.0)["b"] if postb.sum() > 40 else np.nan
        metrics["london_experiment"] = {
            "london_block": LONDON_BLOCK,
            "uncle_rate_pre_pct": float(pre_rate),
            "uncle_rate_post_pct": float(post_rate),
            "b_pre": float(b_pre), "b_post": float(b_post),
        }
        print("London experiment:", metrics["london_experiment"])

    dump_json(metrics, "eth_metrics.json")
    print("\n[ethereum] done.")


if __name__ == "__main__":
    main()
