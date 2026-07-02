# Consensus Seismology

**A Gutenberg–Richter theory of blockchain chain reorganizations.**

This repository builds out and empirically tests the hypothesis that a
distributed ledger is a *strain-accumulating system* analogous to a geological
fault: "consensus strain" (propagation-latency gradients, validator clustering,
mempool divergence) accumulates silently and releases as chain
reorganizations, whose statistics should mirror earthquakes.

The theory makes three falsifiable predictions:

| # | Seismology | Consensus-seismology prediction | Falsified if… |
|---|------------|--------------------------------|---------------|
| **P1** | Gutenberg–Richter | Reorg *magnitudes* follow a **power law**, with a "consensus b-value" | magnitudes are **exponential** |
| **P2** | Omori's law | Fork/orphan rate after a large reorg decays as **~1/tᵖ**, p≈1 | no power-law decay (flat / pure exponential) |
| **P3** | b-value drop / seismic gap | b **drops before** deep reorgs; quiet-but-active latency boundaries rupture next | b is stationary / uncorrelated with strain |

We test these two ways:

1. **Mechanistic model** (`src/model.py`) — a two-cluster longest-chain race
   across a latency boundary. We do *not* hard-code the seismological laws; we
   build the physical mechanism and measure whether the signatures emerge.
   This tests whether the analogy is even *coherent*.
2. **Real data** (`src/fetch_ethereum.py` → `analysis/run_ethereum.py`) —
   ~1,000,000 pre-merge Ethereum blocks (12.4M–13.4M, ≈ Apr–Oct 2021) and every
   *uncle* (stale block) they reference. Uncles are the densest public record of
   organic forks on a major proof-of-work chain.

## Layout

```
src/
  powerlaw.py     Clauset–Shalizi–Newman power-law MLE + Vuong exp. comparison (P1)
  bvalue.py       Aki–Utsu Gutenberg–Richter b-value + rolling strain index (P3)
  omori.py        Modified-Omori (Utsu) MLE with ETAS background term (P2)
  seismic_gap.py  recurrence-CV / quiescence analysis (P3)
  model.py        mechanistic block-race + stick-slip generative models
  fetch_ethereum.py  concurrent multi-endpoint RPC fetch of block/uncle data
analysis/
  run_model.py    theory-coherence test on the generative models
  run_ethereum.py confront the theory with real Ethereum uncle data
results/          figures + metrics JSON (generated)
REPORT.md         full findings, verdicts, and honest limitations
```

## Reproduce

```bash
pip install -r requirements.txt
python src/fetch_ethereum.py --start 12400000 --end 13400000   # ~20 min
python analysis/run_model.py       # generative-model coherence test
python analysis/run_ethereum.py    # real-data confrontation
```

Public RPC endpoints (no key) are used for data; the fetch is concurrent and
resumable. See **REPORT.md** for the results and what they do and do not
establish.
