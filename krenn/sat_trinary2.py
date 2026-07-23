"""SAT encoding v2 for trinary-weight monochromatic quantum graphs.

Sound bidirectional sequential counters for exact equality #P - #M = target.
Usage: python3 sat_trinary2.py N D [assumefile]
"""
import sys
import numpy as np
from pysat.formula import IDPool
from pysat.solvers import Cadical195
from mqg import MQG


class Enc:
    def __init__(self):
        self.pool = IDPool()
        self.clauses = []

    def newvar(self, tag):
        return self.pool.id(tag)

    def counter(self, lits, tag):
        """Full (biconditional) sequential counter.
        Returns dict j -> literal meaning (#true(lits) >= j), j=1..len(lits)."""
        m = len(lits)
        # c[i][j] for i terms, j>=1
        prev = {}  # j -> var, for i-1 ; c_{0,j}=False all j>=1
        for i in range(1, m + 1):
            cur = {}
            li = lits[i - 1]
            for j in range(1, i + 1):
                x = self.newvar((tag, "c", i, j))
                A = prev.get(j)          # c_{i-1,j}
                B = prev.get(j - 1)      # c_{i-1,j-1}; j-1=0 -> True
                # x <-> A or (B and li)
                if j == 1:
                    # B is True: x <-> A or li
                    if A is None:
                        # x <-> li
                        self.clauses += [[-x, li], [x, -li]]
                    else:
                        self.clauses += [[-x, A, li], [x, -A], [x, -li]]
                else:
                    if A is None:
                        # c_{i-1,j} False (j > i-1): x <-> B and li
                        assert B is not None or j - 1 <= i - 1
                        self.clauses += [[-x, B], [-x, li], [x, -B, -li]]
                    else:
                        self.clauses += [[-x, A, B], [-x, A, li], [x, -A], [x, -B, -li]]
                cur[j] = x
            prev = cur
        return prev  # j -> (#>=j)


def build(n, d):
    g = MQG(n, d)
    g.build_index_tables()
    e = Enc()
    W = g.nE * d * d
    nvar = [e.newvar(("n", w)) for w in range(W)]
    svar = [e.newvar(("s", w)) for w in range(W)]

    half = n // 2
    term_p, term_m = {}, {}
    for ci in range(len(g.colorings)):
        for mi in range(len(g.pms)):
            wids = tuple(sorted(int(g.idx[ci, mi, k]) for k in range(half)))
            if wids in term_p:
                continue
            ns = [nvar[w] for w in wids]
            ss = [svar[w] for w in wids]
            a = e.newvar(("and", wids))
            for nv_ in ns:
                e.clauses.append([-a, nv_])
            e.clauses.append([a] + [-x for x in ns])
            x = ss[0]
            for k in range(1, half):
                y = e.newvar(("xor", wids, k))
                b = ss[k]
                e.clauses += [[-y, x, b], [-y, -x, -b], [y, -x, b], [y, x, -b]]
                x = y
            p = e.newvar(("p", wids))
            mv = e.newvar(("m", wids))
            e.clauses += [[-p, a], [-p, -x], [p, -a, x]]
            e.clauses += [[-mv, a], [-mv, x], [mv, -a, -x]]
            term_p[wids] = p
            term_m[wids] = mv

    for ci, io in enumerate(g.colorings):
        t = 1 if all(c == io[0] for c in io) else 0
        keys = [tuple(sorted(int(g.idx[ci, mi, k]) for k in range(half)))
                for mi in range(len(g.pms))]
        plits = [term_p[k] for k in keys]
        mlits = [term_m[k] for k in keys]
        cP = e.counter(plits, ("P", ci))
        cM = e.counter(mlits, ("M", ci))
        m = len(plits)
        # exact: #P = #M + t
        for j in range(1, m + 1):
            jp = j + t
            if jp <= m:
                e.clauses += [[-cM[j], cP[jp]], [cM[j], -cP[jp]]]
            else:
                e.clauses.append([-cM[j]])
        if t == 1:
            e.clauses.append([cP[1]])
    return g, e, nvar, svar


def main():
    n, d = int(sys.argv[1]), int(sys.argv[2])
    g, e, nvar, svar = build(n, d)
    print(f"n={n} d={d}: {e.pool.top} vars, {len(e.clauses)} clauses", flush=True)
    with Cadical195(bootstrap_with=e.clauses) as sol:
        sat = sol.solve()
        print("SAT" if sat else "UNSAT", flush=True)
        if sat:
            model = set(l for l in sol.get_model() if l > 0)
            Wf = np.zeros(g.nE * d * d, dtype=complex)
            for w in range(g.nE * d * d):
                if nvar[w] in model:
                    Wf[w] = -1.0 if svar[w] in model else 1.0
            r = g.residuals_fast(Wf)
            print("witness max |residual| =", np.abs(r).max())
            np.save(f"sat_witness_n{n}_d{d}.npy", Wf)


def add_arc_lemma(n, d, g, e, nvar):
    """Rank-1 arc lemma (proved over C, hence valid for {-1,0,1} solutions):
    for every vertex v0 and color c, some incident edge has far-side support
    only color c, with the far-side-c column nonzero."""
    def eidx_entry(u, v, i, j):
        e_ = (min(u, v), max(u, v))
        return (g.eidx[e_] * d + i) * d + j
    for v0 in range(n):
        for c in range(d):
            arc_lits = []
            for u in range(n):
                if u == v0:
                    continue
                if v0 < u:
                    far_entries_bad = [eidx_entry(v0, u, i, j)
                                       for i in range(d) for j in range(d) if j != c]
                    far_entries_c = [eidx_entry(v0, u, i, c) for i in range(d)]
                else:
                    far_entries_bad = [eidx_entry(v0, u, i, j)
                                       for i in range(d) for j in range(d) if i != c]
                    far_entries_c = [eidx_entry(v0, u, c, j) for j in range(d)]
                a = e.newvar(("arc", v0, u, c))
                for w in far_entries_bad:
                    e.clauses.append([-a, -nvar[w]])
                e.clauses.append([-a] + [nvar[w] for w in far_entries_c])
                arc_lits.append(a)
            e.clauses.append(arc_lits)


def main2():
    n, d = int(sys.argv[1]), int(sys.argv[2])
    g, e, nvar, svar = build(n, d)
    add_arc_lemma(n, d, g, e, nvar)
    print(f"n={n} d={d} +arc-lemma: {e.pool.top} vars, {len(e.clauses)} clauses", flush=True)
    with Cadical195(bootstrap_with=e.clauses) as sol:
        sat = sol.solve()
        print("ARC-CONSTRAINED:", "SAT" if sat else "UNSAT", flush=True)
        if sat:
            model = set(l for l in sol.get_model() if l > 0)
            Wf = np.zeros(g.nE * d * d, dtype=complex)
            for w in range(g.nE * d * d):
                if nvar[w] in model:
                    Wf[w] = -1.0 if svar[w] in model else 1.0
            r = g.residuals_fast(Wf)
            print("witness max |residual| =", np.abs(r).max())
            np.save(f"sat_witness_arc_n{n}_d{d}.npy", Wf)


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[3] == "arc":
        main2()
    else:
        main()
