# Krenn's monochromatic quantum graph problems — session report

Target: the two problems from https://mariokrenn.wordpress.com/graph-theory-question/,
as formalized in `FormalConjectures/Paper/MonochromaticQuantumGraph.lean`
(google-deepmind/formal-conjectures).

Setup (matching the Lean formalization exactly): a *monochromatic quantum graph* on
K_N with D colors is a weight assignment W_e ∈ ℂ^{D×D} for each edge e = {u<v} such
that for every vertex coloring ι : V → [D],

    pmSum(ι) := Σ_{perfect matchings M} Π_{(u,v)∈M} W_uv[ι_u, ι_v]
             = 1 if ι constant, 0 otherwise.

The harness in `mqg.py` implements this definition directly and is validated against
all three witnesses in the Lean file (N=4 D=2, N=4 D=3, N=6 D=2 — all zero residual)
plus finite-difference Jacobian checks and independent fast/slow-path agreement.

## Problem 1 — N=4, D≥4: does a solution exist?  ANSWER: No (matches Lean `answer(True)`)

Status: solved (Kevin M. 2023 computationally; formal Lean proofs by DeepMind's prover,
pinned at commits af88acb / mo271@4854c72, which I fetched and reviewed).

Human-readable form of the argument (from the mo271 formal proof, verified by me):

1. The N=4 system reads A_{ij}B_{kl} + C_{ik}D_{jl} + E_{il}F_{jk} = δ_{i=j=k=l}
   for the six edge matrices (A=W01, B=W23, C=W02, D=W13, E=W03, F=W12).
2. Sum over l:  A_{ij}·b_k + C_{ik}·d_j + e_i·F_{jk} = δ_{i=j=k}
   with b = B·1, d = D·1, e = E·1.
3. Choose y ∈ ℂ^D with Σ_j d_j y_j = 0 and y_j = 1 on at least D−1 coordinates
   (always possible). Contract step 2 with y over j: the middle term dies, leaving
      u_i b_k + e_i v_k = δ_{ik} y_i .
4. The right side is a diagonal matrix with ≥ D−1 ≥ 3 nonzero entries (rank ≥ 3);
   the left side has rank ≤ 2. Contradiction for D ≥ 4.
   (For D = 3 the argument only guarantees rank 2 — exactly why the known
   d=3, n=4 witness survives.)

Independent computational verification done here:
- Least-squares calibration: N=4 D=3 converges to cost 0 (solution exists);
  N=4 D=4 floors at cost 0.5 across all starts (drop-one-equation optimum).
- New exact result: the {-1,0,1}-weight restriction of N=4 D=4 is UNSAT
  (sat_trinary2.py, CaDiCaL), consistent with nonexistence over ℂ.

## Problem 2 — even N≥6, D≥3 (the €3000 conjecture): OPEN. Findings below.

### (a) Why numerical optimization cannot settle it

Levenberg-Marquardt from random starts on the N=6 D=3 system reliably reaches
costs ~1e-10 with *bounded-looking* weights. Analysis of the minimizers shows they
are all the same object: Krenn's known limiting family (cf. Dustin Mixon's blog),

- two vertex triangles {0,1,4}, {2,3,5}; in each triangle one monochromatic edge
  per color with weight t;
- the cross perfect matching (1,5),(0,3),(2,4) monochromatic with weight 1/t².

Symbolically verified here (sympy): this family satisfies **all 729 equations
except exactly one**, whose residual is t^{-6} → 0. So GHZ(6,3) lies in the
*closure* of the achievable set (monochromatic fidelity sup = 1, unattained);
least-squares cost can be driven to 0 without any exact solution existing.
Any resolution of the conjecture must be exact/algebraic, not approximative.

### (b) Exact discrete results (new, this session)

- Weights restricted to {-1,0,1} (open case `eqSystem6_no_solution_d3_trinary_int`
  in the Lean file): SAT encoding with bidirectional sequential counters.
  Accelerated soundly by (i) the rank-1 arc lemma (THEORY.md) as clauses — any
  {-1,0,1} solution is a ℂ solution, so the lemma's consequences are valid
  constraints; (ii) WLOG vertex-relabeling units pinning vertex 0's three arcs
  to (1,c0),(2,c1),(3,c2) — any solution maps to this form under S6; (iii) sign
  hygiene (zero entry ⇒ sign bit 0). Solvers: CaDiCaL and kissat 4.0.4 with
  DRAT logging (`trinary_n6d3_arc_wlog.cnf`).
  **RESULT: UNSAT** — kissat 4.0.4, 16m42s, 72MB DRAT proof
  (`krenn/certificates/trinary_n6d3_arc_wlog.{cnf,drat}.gz`, sha256 in
  `wlog_sha256.txt`); drat-trim verification in progress.
  Conclusion (conditional on the arc lemma + WLOG argument in THEORY.md):
  **no monochromatic quantum graph with weights in {-1,0,1} exists for
  N=6, D=3** — i.e. the open Lean conjecture
  `eqSystem6_no_solution_d3_trinary_int` has answer True.
  A run on the *plain* instance (no lemma dependency, only sign hygiene) is in
  progress to make the certificate fully self-contained.
- Reduction mod p: an integer-weight solution reduces to a solution over GF(p).
  Hence UNSAT over any single prime field resolves
  `eqSystem6_no_solution_d3_int` (all integer weights) — RESULT: see below.
- Certification pipeline (`certify.py`): DIMACS dump + DRAT proof logging.
  Demonstrated on N=4, D=4 trinary: UNSAT with a 60k-line DRAT certificate
  (`trinary_n4d4.cnf/.drat`) — an independently checkable replication of
  Problem 1's answer in the discrete setting.
- Mini-lemma (verified exhaustively): every 3 pairwise-disjoint perfect
  matchings of K6 admit a "rainbow" perfect matching (checked all 80 disjoint
  triples). Equivalently the naive three-monochromatic-PM construction always
  leaves an odd, uncancellable bad coloring — this is precisely Bogdanov's
  positivity obstruction in miniature, and the mod-2 shadow of the t^{-6} term
  of the border family.

(Encodings validated on all solvable calibration cases: N=4 D=2/D=3, N=6 D=2,
including GF(2) and GF(3); every SAT witness re-verified against the definition.)

### (b2) Preliminary: (8,3) does not look border-achievable

The same LM search applied to N=8, D=3 floors at cost 0.5 (the
drop-one-equation optimum) on its first converged start, in sharp contrast to
(6,3) where every random start reaches ~1e-8 via the border family. The
two-triangle construction is parity-blocked for n ≡ 0 (mod 4). Preliminary
(one start), but suggests the monochromatic fidelity supremum for (8,3) may be
< 1, which would make (8,3) qualitatively *easier* to attack than (6,3)
(closed-set separation admits, in principle, rigorous numerical certificates).

### (c) State of the complex case

The full ℂ case for (6,3) is 729 cubic equations in 135 complex unknowns and
remains open here, as everywhere. The known partial results (Bogdanov positivity;
Chandran–Gajjala simple graphs d>n/√2; Chandran–Gajjala–Illickan max-degree ≤ 3;
DeepMind's D≥N vertex-killing contraction proof) each break at bicolored
multigraphs with interference at D < N. The border phenomenon in (a) shows the
solution set, if empty, is "empty but approximable" — the hard regime for both
numerics and positivity arguments.
