"""Reconstruct the (6,3) border family with single-entry edges and verify exactly.

Family: T1={0,1,2}, T2={3,4,5} triangles. T1 edge of color c = T1\{c},
T2 edge of color c = T2\{c+3}. Cross matching pairs leftover vertices:
cross_c = (c, c+3), single entry (c,c), weight t^-2. Triangle edges weight t,
single entry (c,c).

Good matching M_c = {T1_c, T2_c, cross_c}, weight t*t*t^-2 = 1, fires const-c.
Bad matching B = {cross_0, cross_1, cross_2}, weight t^-6, fires coloring
(0,1,2,0,1,2) -- non-constant.

Check: exact residuals over Q(t) with sympy for symbolic t.
"""
import itertools, sympy as sp
import sys
sys.path.insert(0, '/tmp/claude-0/-home-user-ai-testbed/e8d6bc0e-c2d6-5089-9811-4145f3a745b6/scratchpad')
from mqg import perfect_matchings

t = sp.symbols('t', positive=True)
n, d = 6, 3
edges = [(u, v) for u in range(n) for v in range(u+1, n)]
eidx = {e: k for k, e in enumerate(edges)}
W = {}  # (edge, cu, cv) -> weight
for c in range(3):
    T1c = tuple(sorted(set([0,1,2]) - {c}))
    T2c = tuple(sorted(set([3,4,5]) - {c+3}))
    W[(T1c, c, c)] = t
    W[(T2c, c, c)] = t
    W[((c, c+3), c, c)] = t**-2

pms = perfect_matchings(n)
bad = []
for iota in itertools.product(range(3), repeat=n):
    s = sp.Integer(0)
    for m in pms:
        p = sp.Integer(1)
        for (u, v) in m:
            w = W.get(((u, v), iota[u], iota[v]), 0)
            if w == 0:
                p = 0; break
            p *= w
        s += p
    tgt = 1 if all(x == iota[0] for x in iota) else 0
    r = sp.simplify(s - tgt)
    if r != 0:
        bad.append((iota, r))

print(f"nonzero residuals: {len(bad)}")
for io, r in bad:
    print("  coloring", io, "residual", r)
