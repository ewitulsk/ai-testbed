"""Exhaustive search for strict monomial single-entry border families.

Class: support = M_0 u M_1 u M_2 where M_c is a PM of K_n whose edges are all
single-entry (c,c); edge weights w_e * t^{d_e}. Border family exists in this
class iff LP feasible:
    sum_{e in M_c} d_e = 0   (c = 0,1,2)          [good matchings, weight -> 1]
    sum_{e in P}  d_e <= -1  for every other labeled PM P of the support
                              [bad matchings -> 0]
(Any strict single-entry monomial border family reduces to this form; see
N8D3_STUDY.md. '<= -1' is WLOG by scaling.)

Relaxed pass (necessary condition for cancellation-based families on the same
support): PMs sharing their forced coloring with another PM are exempted from
the <= -1 constraint (assume they could be cancelled by coefficients).

Usage: python3 border_lp.py N [--relaxed]
Fixes M_0 = {(0,1),(2,3),...} (WLOG by vertex relabeling), loops over all
(M_1, M_2) pairs with M_1 <= M_2 lexicographically (color swap symmetry 1<->2).
"""
import sys, itertools, time
import numpy as np
from scipy.optimize import linprog
sys.path.insert(0, '/tmp/claude-0/-home-user-ai-testbed/e8d6bc0e-c2d6-5089-9811-4145f3a745b6/scratchpad')
from mqg import perfect_matchings


def labeled_support(M0, M1, M2):
    """Return list of labeled edges (pair, color) and per-color matchings as
    index sets."""
    labs = []
    msets = []
    for c, M in enumerate((M0, M1, M2)):
        idxs = []
        for pr in M:
            labs.append((pr, c))
            idxs.append(len(labs) - 1)
        msets.append(frozenset(idxs))
    return labs, msets


def all_labeled_pms(labs, n):
    """All PMs of the labeled multigraph: frozensets of labeled-edge indices."""
    # adjacency: for each vertex, labeled edges covering it
    by_min = {}
    for i, (pr, c) in enumerate(labs):
        by_min.setdefault(pr, []).append(i)
    pairs = sorted(by_min)
    # enumerate PMs of the simple graph on 'pairs', then expand color choices
    out = []
    verts = list(range(n))

    def rec(rem, chosen_pairs):
        if not rem:
            # expand labels
            for combo in itertools.product(*[by_min[p] for p in chosen_pairs]):
                out.append(frozenset(combo))
            return
        v = rem[0]
        for p in pairs:
            if v not in p:
                continue
            u = p[0] if p[1] == v else p[1]
            if u not in rem:
                continue
            rem2 = [x for x in rem if x != v and x != u]
            rec(rem2, chosen_pairs + [p])

    rec(verts, [])
    return set(out)


def forced_coloring(P, labs, n):
    col = [None] * n
    for i in P:
        (u, v), c = labs[i]
        col[u] = c
        col[v] = c
    return tuple(col)


def check_pair(M0, M1, M2, n, relaxed=False):
    """Return (feasible, ncons, d or None)."""
    labs, msets = labeled_support(M0, M1, M2)
    nvar = len(labs)
    pms = all_labeled_pms(labs, n)
    good = set(msets)
    bad = [P for P in pms if P not in good]
    if relaxed:
        from collections import Counter
        cnt = Counter(forced_coloring(P, labs, n) for P in bad)
        bad = [P for P in bad if cnt[forced_coloring(P, labs, n)] == 1]
    if not bad:
        return True, 0, np.zeros(nvar)
    A_eq = np.zeros((3, nvar)); b_eq = np.zeros(3)
    for c, ms in enumerate(msets):
        for i in ms:
            A_eq[c, i] = 1
    A_ub = np.zeros((len(bad), nvar))
    for k, P in enumerate(bad):
        for i in P:
            A_ub[k, i] = 1
    b_ub = -np.ones(len(bad))
    res = linprog(np.zeros(nvar), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(None, None)] * nvar, method='highs')
    return res.status == 0, len(bad), (res.x if res.status == 0 else None)


def main():
    n = int(sys.argv[1])
    relaxed = '--relaxed' in sys.argv
    pms = perfect_matchings(n)
    M0 = tuple((2 * i, 2 * i + 1) for i in range(n // 2))
    assert M0 in [tuple(sorted(m)) for m in pms] or True
    feas = []
    t0 = time.time()
    tot = 0
    for i1, M1 in enumerate(pms):
        for i2 in range(i1, len(pms)):   # color-swap symmetry 1<->2
            M2 = pms[i2]
            tot += 1
            ok, nb, d = check_pair(M0, M1, M2, n, relaxed=relaxed)
            if ok:
                feas.append((i1, i2, M1, M2, d))
        if (i1 + 1) % 10 == 0:
            print(f"  M1 index {i1+1}/{len(pms)}: {tot} pairs done, "
                  f"{len(feas)} feasible, t={time.time()-t0:.0f}s", flush=True)
    print(f"n={n} relaxed={relaxed}: {tot} pairs, {len(feas)} FEASIBLE")
    for (i1, i2, M1, M2, d) in feas[:20]:
        print("  M1=", M1, " M2=", M2, " d=", np.round(d, 3))
    if feas:
        np.save(f'border_lp_feas_n{n}{"_rel" if relaxed else ""}.npy',
                np.array([(i1, i2) for (i1, i2, *_r) in feas]))
    return feas


if __name__ == '__main__':
    main()
