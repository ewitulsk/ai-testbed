# A structural necessary condition for d=3 monochromatic quantum graphs

Setting as in the Lean formalization (`MonochromaticQuantumGraph.lean`): K_N (even
N ≥ 6), D = 3 colors, each edge {u,v} (u<v) carries W_uv ∈ ℂ^{3×3}; for every
ι : V → {0,1,2},

    pmSum(ι) = Σ_{PM M} Π_{(u,v)∈M} W_uv[ι_u, ι_v]  =  [ι constant].       (★)

Notation: for a fixed vertex v0 and neighbor u, write M_u for the weight matrix
of edge {v0,u} *oriented from v0*: (M_u)[a,b] with a the v0-side color, b the
u-side color (transpose the stored matrix when v0 > u).

## Lemma (rank-1 arc lemma)

If (★) holds with D=3 and even N ≥ 4, then for every vertex v0 and every color c
there exists a neighbor u with

    M_u = β e_cᵀ  for some β ∈ ℂ³ \ {0},

i.e. the full weight matrix of the edge {v0,u} is supported on the single u-side
color c ("the edge is monochromatic at its far end, in color c"). The three
neighbors u_0, u_1, u_2 obtained for c = 0,1,2 are distinct.

### Proof

Contract the tensor identity (★) with a product vector: choose x ∈ ℂ³ at v0 and
v_u ∈ ℂ³ at each u ≠ v0. Summing (★)·x(ι_{v0})·Π_u v_u(ι_u) over all ι gives

    Σ_M Π_{(u,w)∈M} (contracted edge factors) = Σ_c x_c Π_{u≠v0} v_u(c).     (1)

Every perfect matching contains exactly one edge at v0, whose contracted factor
is xᵀ M_u v_u = ⟨A_u, v_u⟩ where A_u := M_uᵀ x ∈ ℂ³. Hence if v_u ⊥ A_u for all
u ≠ v0, the left side of (1) vanishes, so

    Σ_c x_c Π_{u≠v0} v_u(c) = 0   whenever v_u ∈ H_u := A_u^⊥.             (2)

Fix x with all x_c ≠ 0. Let x_c^{(u)} ∈ H_u^* denote the restriction of the
coordinate functional e_c^* to H_u. Condition (2) says the tensor

    T(x) := Σ_{c=0}^{2} x_c · x_c^{(1)} ⊗ ... ⊗ x_c^{(N-1)}  =  0           (3)

in H_1^* ⊗ ... ⊗ H_{N-1}^* (slots indexed by the N−1 ≥ 3 neighbors).

Key fact about the factor families: a linear combination Σ_c λ_c x_c^{(u)}
vanishes iff Σ_c λ_c e_c ∈ ℂ·A_u; so if A_u ≠ 0 the only relation (up to scale)
is λ ∝ A_u, and if A_u = 0 there is none.

(3) is a vanishing sum of ≤ 3 decomposable tensors. Standard Segre geometry:
- If all three summands are nonzero, a vanishing sum of three decomposables
  forces all three to lie on a line of the Segre variety, which varies in one
  slot only: in every other slot u, x_0^{(u)}, x_1^{(u)}, x_2^{(u)} are pairwise
  proportional. Pairwise proportionality of all three gives two independent
  relations with 2-element supports {c,c'} — impossible for a single relation
  direction A_u, unless one of the restrictions vanishes.
- If exactly one summand is nonzero, it must vanish — contradiction; so at least
  two summands vanish or a two-term cancellation occurs.
- A two-term cancellation x_c(...)⊗ = −x_{c'}(...)⊗ forces x_c^{(u)} ∝ x_{c'}^{(u)}
  in *every* slot u, i.e. A_u ∈ span(e_c, e_{c'}) for all u ≠ v0.

Thus for each x (with all coordinates nonzero, off a proper closed locus) one of:
  (a) for every color c some u has x_c^{(u)} = 0, i.e. A_u = M_uᵀ x ∈ ℂe_c \ 0;
  (b) there is a pair {c,c'} with M_uᵀ x ∈ span(e_c, e_{c'}) for all u ≠ v0.

Branch (b) on a Zariski-dense set of x: {x : M_uᵀx ∈ span(e_c,e_{c'}) ∀u} is a
linear subspace; dense ⇒ everything ⇒ every edge at v0 has zero u-side color-c''
entries (c'' the third color) ⇒ the monochromatic-c'' equation reads 0 = 1.
Contradiction. Hence branch (a) holds for all x in a dense set.

Covering argument: fix c. X_{u,c} := {x : M_uᵀ x ∈ ℂe_c} is a linear subspace,
and a dense subset of ℂ³ is contained in ⋃_u X_{u,c}; taking closures,
ℂ³ = ⋃_u X_{u,c}, and a vector space over ℂ is not a finite union of proper
subspaces, so X_{u,c} = ℂ³ for some u; moreover the kill at generic x requires
M_uᵀ x ≠ 0 there, so M_uᵀ ≠ 0. X_{u,c} = ℂ³ means im(M_uᵀ) ⊆ ℂe_c, i.e.
M_u = β e_cᵀ, β ≠ 0. Distinctness: a matrix β e_cᵀ ≠ 0 determines c. ∎

Remarks.
- For D = 2 the argument is vacuous exactly where it should be: branch (b)'s
  "coordinate plane" is the whole ℂ², so no contradiction arises — consistent
  with the known d=2 solutions (cycles).
- The N=4, D=3 witness (three monochromatic perfect matchings) satisfies the
  lemma: at each vertex the three incident edges are e_c e_cᵀ.
- Krenn's limiting family for (6,3) (two triangles with weight t, cross matching
  with weight t⁻²) satisfies the lemma "in structure": all nine support edges are
  single-entry — each is a rank-1 far-end-monochromatic edge in both directions.

## Consequences and proposed finishing strategy for (6,3)

Counting: the lemma yields 3 out-arcs per vertex = 18 arcs on ≤ 15 edges, so at
least 3 edges carry arcs in both directions; a doubly-arced edge has a single
nonzero entry W = w·e_{c'} e_cᵀ (v-side mono c', u-side mono c).

If S ⊆ E is a perfect matching all of whose edges are doubly-arced (single
entry, entry (c_e, c'_e)), let ι be the coloring that assigns each vertex the
color forced by its S-edge. Then S contributes Π_{e∈S} w_e to pmSum(ι), and w_e ≠ 0
(β ≠ 0 componentwise is not guaranteed, but w_e ≠ 0 holds for arc edges by the
lemma's nonvanishing statement applied in both directions). If ι is not
monochromatic and *no other perfect matching is alive under ι*, then
pmSum(ι) = Π w_e ≠ 0 contradicts (★). This is exactly the t⁻⁶ obstruction that
kills Krenn's limiting family; an exact solution must therefore cancel every
such "isolated bad matching" with other alive matchings, which requires
additional nonzero entries, which (empirically) create new bad colorings.
Making this cascade rigorous — e.g. by a discharging/patching argument over the
finitely many arc configurations (up to S6 × S3 symmetry) — is a plausible route
to a full proof of the (6,3) case, and the lemma sharply restricts the
configuration space. Not completed here.

## What the lemma does NOT do

It does not by itself resolve (6,3): free (non-arc) edges may carry arbitrary
3×3 matrices, and cancellations among matchings through free edges evade the
isolated-bad-matching argument. The lemma is a necessary condition narrowing
the search/case space.
