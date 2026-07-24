# Arc-configuration case analysis for the (6,3) case of Krenn's conjecture

Companion to `THEORY.md` (the rank-1 arc lemma) and `arc_cases.py` (all code).
Everything below is over ℂ, for the exact system

    pmSum(ι) = Σ_{PM M of K_6} Π_{(u,v)∈M} W_uv[ι_u, ι_v] = [ι constant],   (★)

with D = 3 colors. By the arc lemma, any solution induces an **arc
configuration**: for every vertex v an injective map arcs[v] : {0,1,2} → V\{v},
where arcs[v][c] = u means the edge {v,u} has *all entries with u-side index
≠ c equal to zero* and W_{vu} ≠ 0. This file pushes a case analysis over all
arc configurations toward a proof that no (6,3) solution exists.

## 0. Normalization and symmetry

* WLOG (vertex relabeling) vertex 0's arcs are 0→1 (color 0), 0→2 (color 1),
  0→3 (color 2). Raw normalized space: each of vertices 1..5 picks one of
  5·4·3 = 60 injections → **60^5 = 777,600,000 normalized configurations**.
* The full symmetry group is S6 × S3 (order 4320), acting by
  A'[π(v)][σ(c)] = π(A[v][c]). By Burnside (exact count, `burnside_total()`):
  **10,806,023 configurations up to symmetry** (before any pruning).
* Canonical forms: exactly 72 group elements map a normalized configuration to
  a normalized one (choose π⁻¹(0): 6, choose σ: 6 — this forces π on the three
  arc-targets of the new vertex 0 — choose π on the remaining 2 vertices: 2).
  The canonical form is the minimum encoding over these 72 images; two
  normalized configurations are S6×S3-equivalent iff their canonical forms
  agree. (Validated against a brute-force orbit computation on random
  configurations.)

## 1. Sound pruning rules

### Rule A — every color-diagonal graph has a perfect matching

An arc x→y labeled c′ forces the y-side of {x,y} to color c′, so the [c,c]
entry of an edge is structurally zero unless **every** arc on the edge is
labeled c. Let G_c = { edges all of whose arcs are labeled c } (free edges,
with no arcs, belong to every G_c). Every perfect matching alive under the
monochromatic coloring ι ≡ c lies in G_c, and pmSum(ι≡c) = 1 ≠ 0 needs an
alive matching. Hence **G_c contains a perfect matching for each c**.
Exclusions only grow as arcs are added, so partial assignments can be pruned
during the DFS (a 2^15-entry table `hasPM[mask]` makes the check O(1)).

### Rule B — no isolated forced matching

A doubly-arced edge (arcs in both directions, labels c_u at the u side, c_v at
the v side) has a single structurally-allowed entry W = w·e_{c_v} e_{c_u}^T,
and w ≠ 0 because the arc lemma gives a *nonzero* matrix in both directions.
Call an edge *alive* under ι if no arc on it is violated (arc x→y labeled c
requires ι_y = c). If a perfect matching S has all three edges doubly-arced, S
forces a coloring ι_S of all six vertices. If ι_S is **non-constant** and **no
other** perfect matching of K_6 is alive under ι_S, then
pmSum(ι_S) = Π_{e∈S} w_e ≠ 0, contradicting (★). This is exactly the t⁻⁶
obstruction that kills Krenn's limiting family.

Completeness note: rule B is the *full* strength of the criterion "some
non-constant ι has exactly one alive PM whose entries are all guaranteed
nonzero", because an alive all-doubly-arced PM forces ι on all six vertices,
so such an ι is necessarily the coloring induced by that PM.

### Rule C — zero/nonzero propagation (root fixpoint)

Working over the structural term table (for each of the 729 colorings ι, the
alive matchings and their entry monomials; doubled weights w are guaranteed
nonzero, all other symbols — free-edge entries and components of singly-arced
β vectors — may vanish), iterate to a fixpoint:

* eq (=1) with no alive term ......................... contradiction;
* eq (=0) with one alive term, all factors proven ≠ 0 . contradiction
  (subsumes rule B; also subsumes rule A via the =1 case);
* eq (=1) with one alive term ........................ all its factors ≠ 0;
* eq (=0) with one alive term, exactly one factor not
  yet proven ≠ 0 ..................................... that factor = 0
  (kills the term everywhere, reshaping other equations).

Every step is an elementary consequence of (★) and the arc lemma, so a derived
contradiction soundly kills the configuration *class*.

### Rule D — branching (case split) with Groebner endgame

When propagation stalls, an eq (=0) with a single alive term and k ≥ 2
undetermined factors yields a sound k-way split ("one of them vanishes").
Optionally a DPLL split on a single symbol (zero / nonzero) is added. A class
is killed only if **every** branch reaches a contradiction. At stalled leaves,
the subsystem of equations whose alive terms contain only proven-nonzero
symbols is passed to a Groebner-basis check over ℚ[vars, t] with the
saturation t·Π(vars) − 1 (1 in the ideal ⇒ no solution over ℂ, by the
Nullstellensatz); optionally the *full* reduced system (undetermined symbols
unsaturated) is checked the same way.

The term table underlying rules C/D was validated symbol-by-symbol against an
independent sympy construction of pmSum on random configurations.

## 2. Counts at each pruning stage

| stage | raw normalized (60^5 space) | up to S6×S3 |
|---|---|---|
| all arc configurations | 777,600,000 | 10,806,023 (Burnside) |
| after rule A (DFS-pruned) | 133,246,310 | CANON_A |
| after rule B | — | CANON_B |
| after rule C (root propagation) | — | CANON_C |
| after rule D (depth 12, budget 3000 nodes) | — | CANON_D |

DFS depth-pass counts (vertices 1..5): 60, 3,529, 154,505, 5,325,728,
133,246,310 — rule A alone eliminates 82.9% of the raw space.

SURVIVOR_BREAKDOWN

## 3. The all-doubled classes (the maximally constrained border family)

If all 18 arcs sit on 9 doubly-arced edges, every vertex's arc targets are
exactly its neighbors in a **cubic support graph**, and the only cubic graphs
on 6 vertices are the triangular prism K3□K2 and K3,3. Direct enumeration
(`all_doubled_classes()`):

* 7 labeled cubic support graphs compatible with the normalization;
  50,978 normalized all-doubled configurations pass rule A;
* **796 classes up to symmetry: 718 with prism support, 78 with K3,3 support**;
* rule B kills none of them. This is forced: in an all-doubled class, for any
  support matching S and any edge (p,q) ∈ S, the matching consisting of (p,q)
  and two *free* (non-support) edges on the remaining four vertices is alive
  under ι_S whenever it exists — for K3,3 support the remaining pairs lie
  inside the free triangles, and for prism support the two perfect matchings
  of the free complement 6-cycle are alive under **every** coloring. So an
  isolated forced matching cannot occur.

### 3a. K3,3-support classes: 75 of 78 killed

Running the rule C/D engine (depth ≤ 25, node budget 3·10^5, Groebner
endgames): **75 of the 78 K3,3 classes are proven infeasible** (74 by
propagation + branching alone, one more by the full-system Groebner endgame at
stalled leaves). Structure of the K3,3 case: with parts P = {0,4,5},
Q = {1,2,3}, the free edges are the two triangles inside P and Q, which have
no perfect matching, so *every* alive matching must use a support edge whose
two forced colors match ι — this produces a large supply of single-alive-term
equations w·x·y = 0 (x, y free triangle entries), which is exactly what the
engine consumes.

The 3 surviving K3,3 classes (canonical encodings 111581610654, 111581610684,
122717722884) resisted budgets of 5·10^5 nodes, DPLL zero/nonzero splitting
on every entry, and saturated-Groebner endgames at fully decided leaves: they
possess consistent zero/nonzero patterns whose guaranteed-nonzero subsystems
are solvable — so, like the prism classes, they cannot be killed by
zero-pattern logic at all and need quantitative algebra. The last one is the
fully symmetric **diagonal class**: all nine support edges carry *diagonal*
forced pairs (c,c), forming three monochromatic perfect matchings of K3,3
(color-c matching σ_c), i.e. the exact 6-vertex analogue of the known N=4,
D=3 solution (three monochromatic matchings) — with two free triangles on
top. These three classes are the K3,3 hard core.

### 3b. Prism-support classes and Krenn's border family

The prism classes contain Krenn's limiting family: support = triangles
{0,1,2}, {3,4,5} (weight t) + cross matching 03,14,25 (weight t⁻²), forced
pairs: triangle edge {i,j} ↦ (k,k) (k the opposite vertex), cross edge
{i,i+3} ↦ (i,i). Its canonical class is **127091349204**; the family's max
residual is *exactly* t⁻⁶ (verified numerically at t = 2, 4, 8), concentrated
on ι* = (0,1,2,0,1,2), whose only alive matching (when all free entries are 0)
is the cross matching.

No prism class can be killed by rules B–D's zero-pattern logic alone: the two
all-free matchings of the complement 6-cycle are alive under **every**
coloring, so no equation ever has a single alive term at the root. This is
the structural reason the border family is hard — every "isolated bad
matching" can be escorted by the free 6-cycle.

Exact system for the border class (triangle weights gauged to 1 — see gauge
remark below; cross weights u_0, u_1, u_2 ≠ 0; free 6-cycle edges
F04, F05, F13, F15, F23, F24 ∈ ℂ^{3×3}):

* mono-c (c = 0):
  u_0 + u_0·F15[0,0]·F24[0,0] + F04[0,0]·F15[0,0]·F23[0,0]
      + F05[0,0]·F13[0,0]·F24[0,0] = 1, and cyclically for c = 1, 2;
* the bad coloring ι* = (0,1,2,0,1,2), with a = F04[0,1], b = F15[1,2],
  c = F23[2,0], d = F05[0,2], e = F13[1,0], f = F24[2,1]:
  u_0u_1u_2 + u_0·b·f + u_1·d·c + u_2·a·e + a·b·c + d·e·f = 0
  (Krenn's family is a..f = 0, leaving u_0u_1u_2 = t⁻⁶ ≠ 0);
* plus 723 further equations coupling the 54 free entries.

First rigorous cascade step (THEORY.md's "patching" made concrete): if all six
of a,…,f vanish, the ι*-equation reads u_0u_1u_2 = 0, impossible; so **any
exact solution in the border class must have a nonzero free-edge entry among
the six ι*-entries** — the free 6-cycle must genuinely participate, and then
the equations in which that entry occurs as a single alive term force further
structure. The rule C/D engine automates exactly this cascade; its failure to
close the border class shows the cascade admits consistent zero/nonzero
patterns, i.e. the obstruction here is quantitative, not combinatorial.

Gauge remark: the transformations W_uv[a,b] ↦ λ_u(a)λ_v(b)·W_uv[a,b] with
Π_v λ_v(c) = 1 (c = 0,1,2) map solutions to solutions (they scale pmSum(ι) by
Π_v λ_v(ι_v), which is 1 on constant ι and nonzero always). For the border
class the gauge action on the 9 log-weights has rank 6 inside the constraint
subspace, so WLOG the six triangle weights are 1 and only the three
gauge-invariant products m_c = (product of the mono-c support matching) = u_c
remain, together with a 9-dimensional residual gauge on the free entries:
**48 essential unknowns, 729 multilinear equations**. Killing this class
(and its 717 prism siblings) needs genuinely quantitative algebra — this is
the precise remaining frontier of the (6,3) case.

## 4. What is proven

1. (Rule A formalized + counted) Any (6,3) solution's arc configuration lies
   in one of CANON_A symmetry classes (of 10,806,023).
2. (Rule B) ... CANON_B classes.
3. (Rules C/D, machine-checked sound derivations + Nullstellensatz
   certificates via Groebner bases over ℚ) ... **CANON_D classes**.
4. **No (6,3) solution has all-doubled K3,3 support except possibly in 3 of
   the 78 such classes**; in particular 75 all-doubled K3,3 classes are dead.
5. The all-doubled survivors are exactly characterized: 718 prism classes
   (Krenn border family and siblings) + 3 K3,3 classes (including the
   diagonal three-monochromatic-matchings class).

## 5. What remains / suggested next steps

* The prism border family requires quantitative (not zero-pattern) algebra;
  the gauge-reduced 48-unknown system above is the concrete target. A
  valuation/tropical argument quantifying "the ι*-equation forces a new bad
  coloring" (the cascade described in THEORY.md) is the most promising route.
* The 3 surviving K3,3 classes are small enough for a dedicated Groebner
  attack with a stronger engine (msolve/Singular) on the 63-variable system,
  or a bilinear-structure argument (the K3,3 system is bilinear in the two
  free triangles).
* Scale rule D with larger budgets over the CANON_D remaining classes and
  classify survivors by (#arc edges, #doubly-arced edges) to isolate further
  finite hard cores.

## Appendix: an example machine-found kill trace (rule C)

Configuration (normalized encoding 139013987088), arcs
A = ((1,2,3),(0,2,3),(5,0,1),(0,2,1),(0,1,5),(3,4,1)); doubly-arced edges
01:(0,0), 02:(1,1), 03:(0,2), 12:(2,1), 13:(2,2), 45:(1,2); singly-arced
04,14,15,23,25,35; free 05,24,34. The engine derives:

1. pmSum(001112) = 0 has a single alive matching term w01·w45·W23[1,1]
   (every other matching is structurally dead under ι = (0,0,1,1,1,2)); w01
   and w45 are nonzero, hence **W23[1,1] = 0**.
2. Similarly pmSum(001212) = 0 forces **W23[1,2] = 0**.
3. Under the monochromatic coloring ι ≡ 1, every one of the 15 matchings now
   contains a structurally-zero or derived-zero entry — the only escapes ran
   through W23[1,1] — so pmSum(1,…,1) = 0 ≠ 1. **Contradiction.**

Each step is elementary and human-checkable; rule-D kills additionally carry
Groebner certificates (1 ∈ saturated ideal over ℚ).

## Reproduction

```
python3 arc_cases.py burnside   # 10,806,023
python3 arc_cases.py stage1     # DFS + rule A -> survivorsA.npy (133,246,310)
python3 arc_cases.py stage2     # canonicalization -> canonicalA.npy
python3 arc_cases.py stage2b    # rule B on canonical reps
# tiered rules C/D: scratchpad/stage_bc.py; all-doubled analysis:
# arc_cases.all_doubled_classes(), kill_by_branching(...)
```
All computations single-process under `nice -n 15`; canonicalization is
vectorized (numpy); Groebner bases via sympy over ℚ with saturation.
