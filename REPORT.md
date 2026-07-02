# Consensus Seismology — Findings

*A Gutenberg–Richter theory of blockchain chain reorganizations, built out and
tested against a mechanistic model and 1,000,000 blocks of real pre-merge
Ethereum data.*

---

## TL;DR

The seismological analogy is **mechanistically coherent but empirically
unsupported on organic Ethereum data**. A physically-motivated model of two
validator clusters racing across a latency boundary *does* spontaneously
generate earthquake-like statistics (power-law reorg depths, aftershock
clustering, a strain-sensitive b-value). But the real Ethereum uncle record
does *not* show the key scaling laws: fork magnitudes are **exponential, not
power-law** (P1 fails), and post-fork orphan rates show **no Omori decay** (P2
fails). Only a faint version of the b-value/strain coupling survives (P3
marginal). 

The most useful result is the *reason* for the failure, and it vindicates the
theory's own engineering prescription: **Ethereum behaves like a strongly
sub-critical "creeping fault"** — enormous numbers of tiny depth-1 forks, no
large ones, and a near-memoryless release process. That is exactly the regime a
protocol designer *wants* (faults that creep, never rupture), but it is also
precisely the regime in which the critical-phenomena scaling laws (Gutenberg–
Richter, Omori) do **not** appear. The earthquake mathematics describes a
near-critical system; well-designed chains are engineered to stay far from it.

### Verdict table

| Prediction | Mechanistic model | Real Ethereum (1M blocks) |
|---|---|---|
| **P1** Gutenberg–Richter power-law depths | **Emerges** (α≈3.85, power law ≫ exponential, Vuong R=+15) | **Falsified** — exponential preferred (distance: Vuong R=−24.9, p≈1e-137; burst: R=−10.5) |
| **P2** Omori aftershock decay | Clustering emerges (CV=2.4) but decay is shallow, not clean 1/t | **Falsified** — post-burst rate = background ×1.02, slope +0.02 (flat) |
| **P3a** b-value drops with strain | **Supported** (Spearman r=−0.17, p=0.019) | **Marginal** — b drops before large bursts (t-test p=0.032, Wilcoxon p=0.041; ~1% effect) |
| **P3b** seismic-gap / quasi-periodic recurrence | Reproducible in a *locked-fault* model (CV=0.11) | **Not supported** — bursts mildly clustered (CV=1.20, p=0.006), not gap-like |
| Natural experiment: London/EIP-1559 | — | **Null** — uncle rate 4.76%→4.75%, b 0.207→0.204 (no shift) |

---

## 1. Data and methods

**Real data.** 1,000,000 consecutive pre-merge Ethereum blocks,
`#12,400,000`–`#13,399,999` (2021-05-09 → 2021-10-11), fetched via public
JSON-RPC (concurrent, multi-endpoint; `src/fetch_ethereum.py`). Per block we
record `(number, timestamp, n_uncles)`; for every one of the **48,704 uncles**
we record its inclusion distance `nephew_height − uncle_height`. Uncles (stale
blocks) are the densest public record of *organic* forks on a major PoW chain.
Mean block time 13.43 s; uncle rate **4.87 %**. The window deliberately brackets
the London hard fork / EIP-1559 (`#12,965,000`) as a natural experiment.

**Statistical machinery** (`src/`), all standard seismology/heavy-tail methods:

- **`powerlaw.py`** — Clauset–Shalizi–Newman: discrete power-law MLE, KS-optimal
  `xmin`, bootstrap goodness-of-fit, and **Vuong's likelihood-ratio test vs a
  fitted exponential**. This is the falsifiability core: it can *reject* a power
  law in favour of an exponential. Validated on synthetic data — it recovers
  α=2.5 from a true power law and correctly favours the exponential (R=−2.8,
  p=0.005) on wide-range exponential data. (On short-range discrete data its
  discriminating power is genuinely limited — relevant to P1 below.)
- **`bvalue.py`** — Aki–Utsu maximum-likelihood b-value with Shi–Bolt error;
  rolling "consensus strain index"; a foreshock b-drop test.
- **`omori.py`** — modified-Omori (Utsu) MLE with an **ETAS background term**
  `n(t)=μ+K/(t+c)^p`, plus a model-free "excess-over-background" statistic.
- **`seismic_gap.py`** — recurrence coefficient-of-variation vs a Poisson null.

**Mechanistic model** (`src/model.py`). We do **not** hard-code the
seismological laws. A discrete-event simulator races two validator clusters
(mining split *f*) that extend a longest chain; a block a cluster mines is
learned by the other only after a cross-boundary delay `d(t)`. While unaware of
a longer branch a cluster keeps mining its own → branches diverge → on
reconciliation the shorter branch re-orgs, orphaning `depth` blocks. `d(t)` is
the **consensus-strain state**: baseline latency plus decaying "latency shocks"
with a broad (log-uniform) timescale distribution. We then *measure* whether the
signatures emerge. A complementary **stick-slip** model (strain accumulates to a
noisy threshold, then slips) represents a *locked* fault.

---

## 2. Is the analogy coherent? (model results)

**Yes — the mechanism generates the statistics without being told to.**

- **P1 emerges.** Exponentially-distributed latency shocks produce
  **heavy-tailed reorg depths** spanning 1–42 blocks with α≈3.85, and a power
  law is decisively preferred over an exponential (Vuong **R=+15.2**,
  p≈5e-52). *(The pure power law is formally rejected by the bootstrap GoF only
  because n=227k is huge — Clauset's KS test rejects near-power-laws at large
  n; the shape is unambiguously heavier-than-exponential.)* See
  `results/model_blockrace_depth_ccdf.png`.
- **b-value tracks strain (P3a).** The rolling consensus b-value is
  **anti-correlated with latency strain** (Spearman **r=−0.17, p=0.019**): when
  the network desynchronizes, relatively more large reorgs appear and b falls —
  the seismological foreshock signature. See `model_blockrace_bvalue.png`.
- **Aftershock clustering (partial P2).** Large reorgs are followed by
  significant excess activity (ΔAIC=532 vs a flat rate; recurrence CV=2.40 ≫ 1),
  **but** the decay is shallow (p≈0.15), not a clean Omori 1/t. *Honest finding:
  latency-driven forking alone yields clustering but not a canonical Omori law;
  a genuine relaxation process is required. The theory names the right one —
  "strain release perturbs mempool synchronization, which relaxes gradually" —
  and the model exposes an explicit `mempool_relax` hook for it, but tuning it
  to p≈1 is delicate; Omori is better adjudicated on real data.*
- **Creeping vs locked faults (P3b).** The latency model is **clustered**
  (CV=2.40); the stick-slip model is **quasi-periodic** (CV=0.11), the classic
  seismic-gap / characteristic-earthquake signature. Its depths are *not*
  power-law (α→6, exponential preferred) — as expected, locked faults produce
  characteristic sizes, not Gutenberg–Richter. See
  `model_recurrence_contrast.png`.

**Conclusion:** the earthquake analogy is not vacuous. A plausible consensus
mechanism spontaneously reproduces GR depths and b-value/strain coupling, and
the creeping-vs-locked distinction is real and controllable.

---

## 3. Does reality agree? (Ethereum results)

**Largely no — and instructively so.**

### P1 — Gutenberg–Richter: **falsified**
Two magnitude proxies, both **reject the power law in favour of an
exponential**:

- *Uncle inclusion distance* (depth analog): histogram
  `{1:38058, 2:5678, 3:295, 4:77, 5:35, 6:23}`. Vuong **R=−24.9** (p≈1e-137).
- *Uncle-burst size* (uncles per 50-block window, max=12): Vuong **R=−10.5**
  (p≈1e-25).

`results/eth_P1_magnitude_ccdf.png` shows both empirical tails bending *below*
the power law and hugging the exponential. **Major caveat:** the available
dynamic range is tiny — Ethereum caps uncle distance at 6 by protocol, and the
largest 50-block burst was only 12. Organic reorgs on major chains have almost
no magnitude range, so P1 is as much *untestable* as *falsified*: there simply
are no large events to populate a power-law tail. Within the range that exists,
the data are exponential.

### P2 — Omori aftershocks: **falsified**
After a large uncle burst (≥7 uncles/50 blocks, 123 declustered mainshocks),
the excess uncle rate over the following 12 h, *excluding the burst window
itself*, is **1.02× background** with a decay slope of **+0.02** — i.e. flat.
`results/eth_P2_omori.png` shows the post-burst rate sitting exactly on the
background line, nowhere near an Omori p=1 curve. The orphan rate returns to
baseline **immediately**; there is no gradual relaxation.
*(An earlier, naïve fit returned p≈9.9 — a degenerate artifact of including the
mainshock's own uncles at tiny lag. Excluding the burst window and using a
model-free excess ratio removes the illusion. Method matters.)*

### P3 — b-value strain index: **marginal / mixed**
- **Foreshock b-drop:** the rolling b *does* dip before the largest bursts —
  56 % of events show a drop, mean Δb<0, **t-test p=0.032, Wilcoxon p=0.041**.
  Statistically real but a ~1 % effect, barely under 0.05, and dwarfed by the
  clean model version. `results/eth_P3_strain_index.png` shows a visible b
  decline (0.21→0.175) over the final ~3 weeks as uncle rate rose.
- **Recurrence (seismic gap):** large bursts are **mildly clustered**
  (CV=1.20, p=0.006), *not* quasi-periodic. The gap/characteristic-earthquake
  signature (CV<1) is absent. *(We test on raw burst times: declustering imposes
  a minimum spacing that mechanically drives CV below 1 — a trap that would
  otherwise manufacture false support for P3b.)*

### Natural experiment — London / EIP-1559: **null**
Across the fork the uncle rate barely moved (4.76 %→4.75 %) and b was flat
(0.207→0.204). EIP-1559 overhauled fee economics but left the *propagation*
dynamics that drive uncle production untouched — consistent with forks being a
network-latency phenomenon, not a fee phenomenon.

---

## 4. Synthesis: Ethereum is a sub-critical creeping fault

The failures are consistent and have one explanation. Real Ethereum consensus
runs **far below criticality**:

- Block time (~13 s) ≫ propagation latency, so a fork is almost always resolved
  within a single block. Depth-1 forks dominate (86 % of uncles); the protocol
  caps distance at 6 regardless.
- With no room for branches to grow, there is **no magnitude range** for a
  power-law tail (P1), and **no accumulated strain to release gradually**, so
  the orphan rate snaps back to background instead of relaxing (P2).
- The release process is close to **memoryless** (recurrence CV≈1.2, only mildly
  clustered by day-scale network conditions), with only a faint strain-dependent
  modulation of the small/large ratio (P3a).

This is the *creeping* regime of our own model (CV≫1, tiny events) — not the
*locked* regime (CV<1, characteristic ruptures) where seismology's scaling laws
would bite. **The theory's engineering prescription is therefore vindicated even
as its empirical predictions fail:** Ethereum is already the "fault that creeps"
one would design for, and that is exactly why it does not obey Gutenberg–Richter
or Omori. The earthquake statistics are a signature of *near-critical* systems;
a healthy chain is built to avoid that.

---

## 5. What would actually be useful

- **A live consensus-strain index is still worth shipping** — but as a
  b-value/uncle-rate *creep monitor*, not an earthquake predictor. The rolling
  b is stable and measurable (0.20±0.008) and its multi-week decline (§3, P3a)
  is a real, cheap signal of rising desynchronization. Bridges/exchanges could
  raise confirmation counts when the creep index drifts, without any claim to
  forecast a specific reorg.
- **The falsification generalizes.** Any chain engineered to keep block time ≫
  propagation latency (i.e. any chain trying to be safe) will sit in this
  sub-critical regime and fail P1/P2 the same way. The theory becomes testable
  only where chains run *near* criticality — small/fast/contentious networks,
  testnets under stress, or adversarial 51 %-attack reorgs (which are
  strain-*injection*, not organic release, and out of scope here).
- **The model is the reusable artifact.** It quantifies the trade-off the theory
  cares about: tune latency/mining-split to move a chain along the
  creeping↔locked axis, and read off the resulting depth distribution and
  recurrence CV before shipping a parameter change.

---

## 6. Limitations (honest)

1. **No true deep reorgs.** Organic depth on Bitcoin/Ethereum is essentially
   1–2 (Bitcoin's worst organic events are ~1–2 blocks; the 2013 depth-24 fork
   was a software bug, not strain release). P1 cannot be strongly tested where
   the phenomenon it predicts barely occurs. This is a limitation of *reality*,
   not just our data slice.
2. **Uncles ≠ reorgs exactly.** An uncle is a fork that lost and was referenced;
   it is a faithful but not perfect proxy for a canonical-chain reorg.
3. **Single 5-month window on one chain.** We did not scan all of Ethereum
   history or other chains; results are representative of a busy 2021 period,
   not proven universal.
4. **Model Omori is inconclusive.** The mechanistic model shows clustering but
   we did not pin a clean p≈1; that piece rests on the (falsified) real-data
   test rather than a clinched simulation.

---

## 7. Reproduce

```bash
pip install -r requirements.txt
python src/fetch_ethereum.py --start 12400000 --end 13400000   # ~20 min, resumable
python analysis/run_model.py        # -> results/model_*.png, model_metrics.json
python analysis/run_ethereum.py     # -> results/eth_*.png,   eth_metrics.json
```

All numbers in this report are emitted to `results/eth_metrics.json` and
`results/model_metrics.json`; all figures to `results/`.
