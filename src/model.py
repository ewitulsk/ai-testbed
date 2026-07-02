"""
A mechanistic generative model of consensus strain and its release as reorgs.

We do NOT hard-code Gutenberg-Richter or Omori. We build the minimal physical
mechanism the theory posits -- two validator clusters separated by a latency
boundary, racing to extend the longest chain -- and then MEASURE whether the
seismological signatures emerge. If they do, the theory is mechanistically
coherent; if a plausible mechanism cannot produce them, the analogy is empty.

Mechanism
---------
* Blocks are produced network-wide by a Poisson process (rate `lam`, one block
  per `1/lam` time units on average). Each block is produced by cluster A with
  probability `f`, else cluster B (the mining-power split).
* A block produced in a cluster instantly extends that cluster's local tip.
  The *other* cluster only learns of it after a cross-boundary delay `d(t)`.
* While a cluster is unaware of a longer competing branch, it keeps mining its
  own branch -> the branches diverge. When a cluster learns the other branch is
  strictly longer, it re-orgs onto it, orphaning its own divergent blocks. The
  number orphaned is the **reorg depth** (our event magnitude).
* `d(t)` is the consensus-strain state: a baseline latency plus a superposition
  of decaying "latency shocks" (congestion / partition episodes). Each shock
  relaxes on its own timescale drawn from a broad (log-uniform) range. This
  heterogeneous relaxation is the mechanism for aftershock clustering.

Emergent, not assumed:
  - depth distribution shape (power law vs exponential),
  - how the small/large event ratio (b-value) moves with strain,
  - the temporal decay of the post-event fork rate (Omori or not).
"""
import heapq
import numpy as np


class StrainProcess:
    """
    d(t) = base + sum_k A_k * exp(-(t - t_k)/tau_k)  for shocks k with t_k <= t.
    Shocks arrive as a Poisson process; amplitudes ~ Exp; timescales log-uniform
    over [tau_min, tau_max] (the broad range that yields power-law relaxation).
    """
    def __init__(self, base, shock_rate, amp_mean, tau_min, tau_max, rng):
        self.base = base
        self.shock_rate = shock_rate
        self.amp_mean = amp_mean
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.rng = rng
        self.shocks = []  # (t_k, A_k, tau_k)
        self._next_shock = rng.exponential(1.0 / shock_rate)

    def add_trigger(self, tk, Ak, tau):
        """Register a reorg-triggered latency bump (mempool-desync relaxation)."""
        self.shocks.append((tk, Ak, tau))

    def advance_to(self, t):
        while self._next_shock <= t:
            tk = self._next_shock
            Ak = self.rng.exponential(self.amp_mean)
            tau = np.exp(self.rng.uniform(np.log(self.tau_min),
                                          np.log(self.tau_max)))
            self.shocks.append((tk, Ak, tau))
            self._next_shock += self.rng.exponential(1.0 / self.shock_rate)
        # prune fully-decayed shocks (>15 timescales old)
        self.shocks = [(tk, Ak, tau) for (tk, Ak, tau) in self.shocks
                       if t - tk < 15 * tau]

    def value(self, t):
        d = self.base
        for (tk, Ak, tau) in self.shocks:
            if tk <= t:
                d += Ak * np.exp(-(t - tk) / tau)
        return d


def simulate_block_race(
    duration=2.0e6,      # simulated time units
    lam=1.0,             # block production rate (blocks per unit time)
    f=0.5,               # fraction of blocks produced by cluster A
    base_delay=0.15,     # baseline cross-cluster delay (in units of 1/lam)
    shock_rate=1.0e-3,   # latency-shock arrival rate
    amp_mean=0.6,        # mean shock amplitude (added delay)
    tau_min=5.0,         # min shock relaxation timescale
    tau_max=2000.0,      # max shock relaxation timescale
    mempool_relax=0.0,   # reorg-triggered latency bump amplitude (per depth);
                         # models strain release perturbing mempool sync
    relax_tau_min=3.0,   # heterogeneous re-sync timescales (broad => Omori 1/t)
    relax_tau_max=800.0,
    seed=0,
):
    """
    Discrete-event simulation of the two-cluster longest-chain race.

    Returns dict:
      reorgs   : structured array (time, depth) for every reorg event
      strain_t, strain_v : sampled strain time series
      n_blocks : total blocks produced
    """
    rng = np.random.default_rng(seed)
    strain = StrainProcess(base_delay / lam, shock_rate, amp_mean / lam,
                           tau_min / lam, tau_max / lam, rng)

    # Branch state relative to the last common ancestor.
    # lead_A, lead_B = blocks each cluster has built past the common ancestor
    # on its own private branch. At most one is nonzero once reconciled; both
    # can be nonzero mid-divergence.
    lead_A = 0
    lead_B = 0

    # Event queue holds ("block", cluster) productions and ("deliver", cluster,
    # height_snapshot) messages informing the OTHER cluster of a branch length.
    # We use a simpler, exact bookkeeping: each block schedules a delivery to
    # the peer after d(t). The peer, on delivery, compares lengths and reorgs.
    evq = []  # (time, seq, kind, payload)
    seq = 0

    def push(t, kind, payload):
        nonlocal seq
        heapq.heappush(evq, (t, seq, kind, payload))
        seq += 1

    # schedule first block
    t = 0.0
    push(rng.exponential(1.0 / lam), "block", None)

    reorg_times = []
    reorg_depths = []
    strain_sample_t = []
    strain_sample_v = []
    next_strain_sample = 0.0
    strain_dt = duration / 4000.0
    n_blocks = 0

    while evq:
        t, _, kind, payload = heapq.heappop(evq)
        if t > duration:
            break
        strain.advance_to(t)

        if next_strain_sample <= t:
            strain_sample_t.append(t)
            strain_sample_v.append(strain.value(t) * lam)  # in block-time units
            next_strain_sample += strain_dt

        if kind == "block":
            n_blocks += 1
            d = max(strain.value(t), 0.0)
            producer_A = rng.random() < f
            if producer_A:
                lead_A += 1
                # peer B will learn A's branch length after delay d
                push(t + d, "deliver", ("A", lead_A))
            else:
                lead_B += 1
                push(t + d, "deliver", ("B", lead_B))
            # schedule next block
            push(t + rng.exponential(1.0 / lam), "block", None)

        elif kind == "deliver":
            src, length_at_send = payload
            # The peer compares the delivered branch length against its own
            # current lead. If the delivered (competing) branch is strictly
            # longer, the peer reorgs onto it, orphaning its own lead.
            if src == "A":
                # B receives news of A's branch of length `length_at_send`
                if length_at_send > lead_B and lead_B > 0:
                    depth = lead_B  # B orphans its private branch
                    reorg_times.append(t)
                    reorg_depths.append(depth)
                    if mempool_relax > 0:
                        rtau = np.exp(rng.uniform(np.log(relax_tau_min / lam),
                                                  np.log(relax_tau_max / lam)))
                        strain.add_trigger(t, mempool_relax * depth / lam, rtau)
                    # branches reconcile: common ancestor advances to the
                    # shorter merge point; A keeps its lead, B resets.
                    lead_A = max(lead_A - length_at_send, 0)
                    lead_B = 0
                elif length_at_send >= lead_B:
                    # A is ahead or tied and B had no divergent blocks:
                    # collapse the common ancestor forward (no orphan).
                    lead_A = max(lead_A - length_at_send, 0)
                    if lead_A == 0:
                        lead_B = 0
            else:  # src == "B"
                if length_at_send > lead_A and lead_A > 0:
                    depth = lead_A
                    reorg_times.append(t)
                    reorg_depths.append(depth)
                    if mempool_relax > 0:
                        rtau = np.exp(rng.uniform(np.log(relax_tau_min / lam),
                                                  np.log(relax_tau_max / lam)))
                        strain.add_trigger(t, mempool_relax * depth / lam, rtau)
                    lead_B = max(lead_B - length_at_send, 0)
                    lead_A = 0
                elif length_at_send >= lead_A:
                    lead_B = max(lead_B - length_at_send, 0)
                    if lead_B == 0:
                        lead_A = 0

    reorgs = np.zeros(len(reorg_times),
                      dtype=[("time", float), ("depth", int)])
    reorgs["time"] = reorg_times
    reorgs["depth"] = reorg_depths
    return {
        "reorgs": reorgs,
        "strain_t": np.array(strain_sample_t),
        "strain_v": np.array(strain_sample_v),
        "n_blocks": n_blocks,
        "params": dict(duration=duration, lam=lam, f=f, base_delay=base_delay,
                       shock_rate=shock_rate, amp_mean=amp_mean,
                       tau_min=tau_min, tau_max=tau_max, seed=seed),
    }


def simulate_stick_slip(
    n_steps=2_000_000,
    load_rate=1.0,       # strain loaded per step (tectonic drive)
    noise=0.15,          # multiplicative noise on the failure threshold
    threshold=100.0,     # mean strain at which the fault slips
    release_frac=0.9,    # fraction of accumulated strain released per slip
    depth_scale=0.15,    # reorg depth per unit strain released
    seed=0,
):
    """
    A stick-slip / strain-release model of a *locked* consensus fault.

    Strain S accumulates deterministically (S += load_rate) until it crosses a
    noisy failure threshold, at which point the fault slips: it releases a
    fraction of S as a reorg whose depth scales with the strain dropped, and S
    partially resets. Because a big slip empties the fault, the next big slip is
    delayed -- producing quasi-periodic ("characteristic earthquake") recurrence
    (CV<1), the seismic-gap signature. Contrast with simulate_block_race, whose
    externally-driven latency shocks give aftershock clustering (CV>1).

    Returns the same reorg/strain structure as simulate_block_race so the same
    analysis pipeline applies.
    """
    rng = np.random.default_rng(seed)
    S = 0.0
    reorg_times, reorg_depths = [], []
    strain_t, strain_v = [], []
    sample_every = max(1, n_steps // 4000)
    for step in range(n_steps):
        S += load_rate
        thr = threshold * (1.0 + noise * rng.standard_normal())
        if S >= thr:
            released = release_frac * S
            depth = max(1, int(round(depth_scale * released
                                     * rng.uniform(0.5, 1.5))))
            reorg_times.append(float(step))
            reorg_depths.append(depth)
            S -= released
        if step % sample_every == 0:
            strain_t.append(float(step))
            strain_v.append(S)
    reorgs = np.zeros(len(reorg_times), dtype=[("time", float), ("depth", int)])
    reorgs["time"] = reorg_times
    reorgs["depth"] = reorg_depths
    return {"reorgs": reorgs, "strain_t": np.array(strain_t),
            "strain_v": np.array(strain_v), "n_blocks": n_steps,
            "params": dict(kind="stick_slip", n_steps=n_steps,
                           load_rate=load_rate, threshold=threshold,
                           release_frac=release_frac, seed=seed)}


def uncle_series_from_reorgs(reorgs, duration, nbins=4000):
    """Bin reorg events into a fork-rate time series (the 'seismicity rate')."""
    edges = np.linspace(0, duration, nbins + 1)
    counts, _ = np.histogram(reorgs["time"], bins=edges)
    mid = 0.5 * (edges[:-1] + edges[1:])
    return mid, counts
