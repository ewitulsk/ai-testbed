"""Exact verification of a candidate (8,3) border family found by border_lp.py.

Support = M0 u M1 u M2, M_c single-entry (c,c), weight t^{d_e}.
Checks all 3^8 = 6561 equations symbolically: constant colorings must give
exactly 1 + (negative powers of t); non-constant colorings only negative
powers of t. Prints the full list of nonzero residuals.
"""
import itertools, sys
import sympy as sp
sys.path.insert(0, '/tmp/claude-0/-home-user-ai-testbed/e8d6bc0e-c2d6-5089-9811-4145f3a745b6/scratchpad')
from mqg import perfect_matchings

t = sp.symbols('t', positive=True)
n = 8

M0 = ((0, 1), (2, 3), (4, 5), (6, 7))
M1 = ((0, 2), (1, 4), (3, 6), (5, 7))
M2 = ((0, 4), (1, 7), (2, 6), (3, 5))
d = [2, 0, -1, -1,  0, 0, 0, 0,  0, 0, 0, 0]

W = {}
for c, (M, dd) in enumerate([(M0, d[0:4]), (M1, d[4:8]), (M2, d[8:12])]):
    for pr, de in zip(M, dd):
        assert (pr, c, c) not in W
        W[(pr, c, c)] = t**de

pms = perfect_matchings(n)
nonzero = []
ok_const = 0
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
    const = all(x == iota[0] for x in iota)
    tgt = 1 if const else 0
    r = sp.expand(s - tgt)
    if r != 0:
        lead = sp.limit(r, t, sp.oo)
        nonzero.append((iota, r, lead))
    elif const:
        ok_const += 1

print(f"constant colorings exactly satisfied: {ok_const}/3")
print(f"nonzero residuals: {len(nonzero)}")
bad_limits = [(io, r, L) for io, r, L in nonzero if L != 0]
print(f"residuals NOT vanishing as t->oo: {len(bad_limits)}")
for io, r, L in nonzero[:30]:
    print(" ", io, "res", r, "limit", L)
if not bad_limits and ok_const == 3:
    slowest = max((sp.Rational(str(sp.nsimplify(sp.limit(sp.log(sp.Abs(r))/sp.log(t), t, sp.oo))))
                   for _, r, _ in nonzero), default=None)
    print("VERIFIED: genuine border family for (8,3); slowest residual decay t^", slowest)
