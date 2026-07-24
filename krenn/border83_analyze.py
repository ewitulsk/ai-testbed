"""Analyze the verified (8,3) border family + all feasible LP configurations.

1. Numeric cost of the explicit family at t = 10, 100, 1000 via the mqg fast path.
2. Save weight vectors as .npy in search.py x-format ([Re; Im]).
3. Check the structure lemma on all 288 feasible pairs: pairwise disjoint,
   pairwise unions Hamiltonian; count triangle structures.
"""
import sys, itertools
import numpy as np
sys.path.insert(0, '/tmp/claude-0/-home-user-ai-testbed/e8d6bc0e-c2d6-5089-9811-4145f3a745b6/scratchpad')
from mqg import MQG, perfect_matchings

SCRATCH = '/tmp/claude-0/-home-user-ai-testbed/e8d6bc0e-c2d6-5089-9811-4145f3a745b6/scratchpad'

M0 = ((0, 1), (2, 3), (4, 5), (6, 7))
M1 = ((0, 2), (1, 4), (3, 6), (5, 7))
M2 = ((0, 4), (1, 7), (2, 6), (3, 5))
d = {(*M0[0], 0): 2, (*M0[1], 0): 0, (*M0[2], 0): -1, (*M0[3], 0): -1,
     **{(*p, 1): 0 for p in M1}, **{(*p, 2): 0 for p in M2}}

g = MQG(8, 3)
g.build_index_tables()

for tval in [10.0, 100.0, 1000.0]:
    W = np.zeros((g.nE, 3, 3), dtype=complex)
    for c, M in enumerate((M0, M1, M2)):
        for pr in M:
            W[g.eidx[pr], c, c] = tval ** d[(*pr, c)]
    r = g.residuals_fast(W.ravel())
    cost = 0.5 * np.sum(np.abs(r) ** 2)
    print(f"t={tval:g}: cost = {cost:.3e}, max|r| = {np.abs(r).max():.3e}, "
          f"nonzero residuals = {(np.abs(r) > 1e-12).sum()}")
    x = np.concatenate([W.ravel().real, W.ravel().imag])
    np.save(f"{SCRATCH}/border83_t{int(tval)}.npy", x)

# --- structure analysis of all feasible pairs ---
pms = perfect_matchings(8)
M0fix = tuple((2 * i, 2 * i + 1) for i in range(4))
feas = np.load(f"{SCRATCH}/border_lp_feas_n8.npy")
print(f"\nfeasible pairs: {len(feas)}")

def is_ham_union(A, B):
    """A,B disjoint PMs; is A u B a single 8-cycle?"""
    adj = {}
    for (u, v) in list(A) + list(B):
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    seen = {0}
    prev, cur = None, 0
    for _ in range(8):
        nxts = [w for w in adj[cur] if w != prev]
        if not nxts:
            return False
        prev, cur = cur, nxts[0]
        seen.add(cur)
    return cur == 0 and len(seen) == 8

def triangles(edges):
    es = set(map(tuple, map(sorted, edges)))
    tri = []
    for a, b, c in itertools.combinations(range(8), 3):
        if ((a,b) in es) and ((b,c) in es) and ((a,c) in es):
            tri.append((a, b, c))
    return tri

stats = {'disjoint': 0, 'all_ham': 0}
tri_counts = {}
for (i1, i2) in feas:
    A, B, C = M0fix, pms[i1], pms[i2]
    sets = [set(A), set(B), set(C)]
    disj = not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    stats['disjoint'] += disj
    if disj:
        ham = (is_ham_union(A, B) and is_ham_union(A, C) and is_ham_union(B, C))
        stats['all_ham'] += ham
    ntri = len(triangles(list(A) + list(B) + list(C)))
    tri_counts[ntri] = tri_counts.get(ntri, 0) + 1
print("pairwise disjoint:", stats['disjoint'], "/", len(feas))
print("all pairwise unions Hamiltonian:", stats['all_ham'], "/", len(feas))
print("triangle-count distribution of union cubic graph:", tri_counts)
