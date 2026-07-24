# Is (N,D) = (8,3) border-achievable? — YES (verified exactly)

**Conclusion: (8,3) IS border-achievable: inf cost = 0, monochromatic-fidelity
supremum = 1 for GHZ(8,3).** An explicit one-parameter family W(t) satisfies all
3^8 = 6561 equations exactly except **two**, each with residual t^{-1} → 0
(least-squares cost exactly t^{-2}). This is verified symbolically (sympy, full
definition, all 6561 colorings) and numerically (the validated `mqg.py` fast
path). It **supersedes the preliminary hypothesis in REPORT.md §(b2)** — the
LM cost-0.5 floor there was a basin-of-attraction artifact, not evidence of
sup-fidelity < 1.

## The explicit border family

Vertices 0..7. Support = three pairwise-disjoint perfect matchings, each
monochromatic in one color; every edge is a single-entry matrix
W = w·e_c e_cᵀ (entry (c,c) only):

| matching | color | edges (weight) |
|---|---|---|
| M₀ | 0 | (0,1): **t²**, (2,3): 1, (4,5): **t⁻¹**, (6,7): **t⁻¹** |
| M₁ | 1 | (0,2): 1, (1,4): 1, (3,6): 1, (5,7): 1 |
| M₂ | 2 | (0,4): 1, (1,7): 1, (2,6): 1, (3,5): 1 |

Structure of the union cubic graph: **two triangles + an extra pair** —
triangles {0,1,4} (edges 01/c0, 14/c1, 04/c2) and {2,3,6} (23/c0, 36/c1,
26/c2), extra vertices {5,7}, with six connector edges 45/c0, 67/c0, 02/c1,
57/c1, 17/c2, 35/c2. This is the natural generalization of Krenn's (6,3)
two-triangle family (which this study first reconstructed in single-entry form
and re-verified: exactly one bad equation, residual t⁻⁶, at coloring
(0,1,2,0,1,2) for triangles {0,1,2},{3,4,5}, cross (c,c+3)).

Exact residual analysis (sympy, `border83_verify.py`):
- all 3 constant colorings: pmSum = 1 **exactly** (only M_c fires const-c);
- 6559 non-constant colorings: 6557 are **exactly 0**; the two colorings
  (1,1,1,2,1,2,0,0) and (1,2,1,1,0,0,1,2) get residual **t⁻¹**, from the
  degree −1 bad matchings {02,14,35,67} and {02,36,45,17} respectively.
- Numeric (mqg.py fast path, independent of the LP model):
  cost = 1.000e-2 / 1.000e-4 / 1.000e-6 at t = 10 / 100 / 1000; exactly 2
  nonzero residuals each time. (Decay exponent is meaningless up to the
  reparametrization t → t^k; what matters is cost → 0.)

Saved weight vectors (search.py x-format, [Re; Im] of W.ravel(), scratchpad
`/tmp/claude-0/-home-user-ai-testbed/e8d6bc0e-c2d6-5089-9811-4145f3a745b6/scratchpad/`):
- `border83_t10.npy` (cost 1e-2), `border83_t100.npy` (1e-4),
  `border83_t1000.npy` (1e-6), `border83_lm_polished.npy` (LM-polished from t=10).

## Methodology

### 1. Reduction theorem (strict monomial single-entry class)

Call a family *strict monomial single-entry* if every nonzero entry is w·t^d
with one entry per (edge, color-pair) label, no cancellation used among
same-degree bad terms. Every support-PM then fires on exactly **one** coloring
(each vertex's color is forced by its unique matching edge). Border ⟺
(i) for each color c some const-c-firing PM has degree 0 with coefficients
summing to 1; (ii) every non-constant-firing PM has degree < 0.

*Reduction:* a const-c-firing PM uses only (c,c)-labeled edges. Pick one such
M*_c per color and restrict the support to M*₀ ∪ M*₁ ∪ M*₂: every PM of the
sub-support is a PM of the full support (constraints inherited), and the only
const-c-firing PM in the sub-support is M*_c itself. **So WLOG the support is
a union of three color-labeled PMs**, and existence in this class is exactly an
LP feasibility problem in the 12 edge-degrees d_e:

    Σ_{e∈M_c} d_e = 0 (c = 0,1,2);   Σ_{e∈P} d_e ≤ −1 for every other
    labeled PM P of the support multigraph.

This also covers multi-parameter monomial families (t,s): a 2D degree grading
with bad terms → 0 along some direction projects to a 1D rational grading.

*Structure lemma* (proved via Farkas pairs, confirmed empirically below): the
three matchings must be pairwise disjoint as vertex-pair sets, and each
pairwise union M_c ∪ M_c' must be a single Hamiltonian cycle. (If a pair is
shared, the two label-swapped matchings B, B' are bad with
deg B + deg B' = 0; if M_c Δ M_c' has ≥ 2 alternating cycles, the two mixed
matchings satisfy 1_B + 1_{B'} = 1_{M_c} + 1_{M_c'} — either way two bad PMs
whose degrees sum to 0, contradicting both ≤ −1.)

### 2. Exhaustive LP search (`border_lp.py`)

Fix M₀ = {(0,1),(2,3),(4,5),(6,7)} (WLOG by vertex relabeling), enumerate all
(M₁, M₂) ∈ PMs(K₈)² with M₁ ≤ M₂ (color-swap symmetry): 5565 pairs, LP each.

- **Calibration N=6**: 120 pairs → 12 feasible, exactly the two-triangle
  (prism) configurations; recovers the known family. ✓
- **N=8**: 5565 pairs → **288 feasible**. Every feasible triple is pairwise
  disjoint with all three pairwise unions Hamiltonian (structure lemma ✓),
  and every feasible union cubic graph contains **exactly two triangles** —
  the two-triangles+pair structure is universal among feasible configurations
  (with M₀ fixed; 288 counts (M₁,M₂) pairs, not isomorphism classes).
- First feasible candidate verified exactly end-to-end (see above).

So within this natural class the question is *completely decided*: border
families exist, and all of them are of two-triangles+extra-pair type.

### 3. Multistart LM numerics (`search.py`, N=8 D=3)

Machine constraint: 3 concurrent SAT solvers; run executed as a single
`nice -n 15`, single-BLAS-thread process. 12 random starts, seed 7
(504 real unknowns, 13122 real equations), plus the one start recorded
earlier (REPORT.md (b2), cost 0.500000).

Results (as recorded during this session):
- Prior session, seed 1, first converged start: cost **0.5000000004** (the
  drop-one-equation floor).
- This run, seed 7: start 0 → cost **0.5000000004**; after ~44 CPU-minutes
  (est. 8–11 of 12 starts complete under heavy SAT-solver contention), **no
  start improved on 0.5** (the best-so-far JSONL `best_n8d3_run2.jsonl`
  records improvements; none after start 0). Median across completed random
  starts is therefore ≥ 0.5. The run was left to finish; its final quartile
  line appends to `log_n8d3_run2.txt`.
- So every random start observed lands on the ≈ 0.5 floor, in sharp contrast
  to (6,3) where random starts reach ~1e-10 routinely.

Interpretation: random-start LM lands on the cost ≈ 0.5 drop-one-equation
floor because the border manifold requires a coordinated escape to infinity
(t² together with t⁻¹ entries); its basin is thin at N=8, unlike N=6 where
random starts find the border family easily. Seeded LM demonstrates the floor
is an artifact: initialized at the explicit family's t = 10 point
(cost 1.000e-2), LM descends to **5.379e-8** in 300 iterations with weights
growing to max|W| ≈ 441 — riding the border manifold toward cost 0
(`border83_seeded_lm.py`; polished vector `border83_lm_polished.npy`).
Lesson for the wider project: a multistart-LM floor of 0.5 is NOT reliable
evidence that sup-fidelity < 1; the (8,3) case is now a concrete
counterexample to that inference.

## Ansätze considered on the way (with exact residual analysis)

1. **Two K4 blocks, each carrying the exact (4,3) witness, + cross edges.**
   Block PMs give the bi-constant failure: coloring constant c₁ on one block,
   c₂ ≠ c₁ on the other receives the full block-product weight at the leading
   degree. Grading analysis: with block edges at degree p and cross at r, the
   three matching types have degrees 4p, 2p+2r, 4r; making the good type 0
   forces either a positive-degree bad type (blow-up) or reduces the entire
   burden to an **exact** monochromatic quantum solution on K_{4,4} at degree
   0 (deg-0 bad terms can only cancel against deg-0 terms). So even-even
   splits give no leverage — consistent with the LP outcome that all feasible
   supports are odd-split (two triangles) based.
2. **(6,3) family ⊕ extra pair {6,7} with diag(1,1,1).** Coupling failure:
   colorings (const c on 0..5, e ≠ c on {6,7}) receive weight 1 at degree 0
   from M_c ∪ {(6,7)}. Killing these requires the pair to be color-coupled
   into the rest — exactly what the verified family's connector edges
   (45, 67, 02, 57, 17, 35) do: the extra pair {5,7} never carries a
   full-diagonal edge; each of 45/67/57/17/35 is single-entry, so every
   matching through {5,7} forces its colors.
3. **3+5 split, triangle + (triangle+pair) with per-color routing** (hand
   design, y_c-attachment): subsumed by the exhaustive search — the verified
   family is of this type (cross edge 02 links the triangles; the pair is
   reached by per-color connectors).
4. **Parity obstruction attempt for n ≡ 0 (mod 4).** The (6,3) proof-shape
   "two odd blocks so every PM uses an odd number of cross edges" does not
   extend to an impossibility proof: 8 = 3+5 is an odd split too, and the LP
   shows the obstruction simply does not exist. The structure lemma above is
   the true constraint (pairwise-Hamiltonian 1-factors), and it is satisfiable
   on 8 vertices.

## Class limitations (honesty about scope)

The exhaustive result is complete for strict monomial single-entry families.
Not covered: families using coefficient cancellation among same-degree bad
terms, non-monomial t-dependence, or multi-entry edges with interference.
Since the strict class already succeeds, these larger classes are moot for
the border-achievability question (they can only make it easier).

## Consequences

- GHZ(8,3) has monochromatic-fidelity supremum 1 (unattained by this family;
  whether an *exact* solution exists remains open, as for (6,3)).
- REPORT.md §(b2) is corrected: (8,3) is **not** a candidate for a
  closed-set separation / sup-fidelity < 1 attack. Both (6,3) and (8,3) sit
  in the hard "empty-but-approximable" regime; any nonexistence proof must be
  exact/algebraic for N=8 too.
- The construction pattern iterates: random LP sampling at **N=10** found
  feasible strict configurations quickly (3 hits in ~1100 samples;
  `border_lp_n10_sample.py`, `log_border_lp_n10.txt`), e.g.
  M₁ = {02,15,36,49,78}, M₂ = {09,14,27,35,68} with
  d_{M₀} = (0,3,−1,−2,0), d_{M₁} = (1,0,0,−1,0), d_{M₂} = 0. This is
  LP-model-level evidence (the model was verified end-to-end at N=8);
  it strongly suggests **all even N ≥ 6, D=3 are border-achievable**, i.e.
  sup-fidelity = 1 throughout the conjecture's range — the whole conjecture
  lives in the "empty-but-approximable" regime.

## Files

- `/home/user/ai-testbed/krenn/N8D3_STUDY.md` — this file.
- Scratchpad (`/tmp/claude-0/-home-user-ai-testbed/e8d6bc0e-c2d6-5089-9811-4145f3a745b6/scratchpad/`):
  `border_lp.py` (reduction + exhaustive LP), `border83_verify.py` (exact
  sympy verification), `border63_verify.py` ((6,3) reconstruction),
  `border83_analyze.py` (numerics + structure stats),
  `border83_seeded_lm.py`, `border_lp_feas_n8.npy` (all 288 feasible pairs),
  `border83_t{10,100,1000}.npy`, `border83_lm_polished.npy`,
  `log_border_lp_n8.txt`, `log_n8d3_run2.txt`.

**Confidence: very high** for border-achievability (exact symbolic
verification of an explicit family against the raw definition, independently
cross-checked numerically on a harness validated against the Lean witnesses).
The classification "all strict single-entry families are two-triangles+pair"
is exhaustive within its class (high confidence, modulo code correctness of
the PM enumeration, which was calibrated on N=6).
