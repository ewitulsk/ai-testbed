"""Monochromatic quantum graph harness.

Mirrors the Lean formalization in FormalConjectures/Paper/MonochromaticQuantumGraph.lean:
  - K_N with edges {u<v}, each edge has a DxD complex weight matrix W_e[i,j]
    (i = color at u, j = color at v).
  - pmSum(iota) = sum over perfect matchings M of K_N of prod_{(u,v) in M} W_uv[iota_u, iota_v]
  - EqSystem: pmSum(iota) == 1 if iota constant else 0.
"""
import itertools
import numpy as np


def perfect_matchings(n):
    """All perfect matchings of K_n as tuples of (u,v) pairs with u<v."""
    verts = list(range(n))
    def rec(vs):
        if not vs:
            yield []
            return
        v = vs[0]
        for k in range(1, len(vs)):
            u = vs[k]
            rest = vs[1:k] + vs[k+1:]
            for m in rec(rest):
                yield [(v, u)] + m
    return [tuple(m) for m in rec(verts)]


class MQG:
    def __init__(self, n, d):
        self.n, self.d = n, d
        self.edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        self.eidx = {e: k for k, e in enumerate(self.edges)}
        self.pms = perfect_matchings(n)          # (n-1)!! matchings
        self.colorings = list(itertools.product(range(d), repeat=n))
        self.nE = len(self.edges)
        # W is stored as array of shape (nE, d, d)

    def target(self, iota):
        return 1.0 if all(c == iota[0] for c in iota) else 0.0

    def pm_sum(self, W, iota):
        s = 0.0 + 0.0j
        for m in self.pms:
            p = 1.0 + 0.0j
            for (u, v) in m:
                p *= W[self.eidx[(u, v)], iota[u], iota[v]]
                if p == 0:
                    break
            s += p
        return s

    def residuals(self, W):
        """Vector of pmSum(iota) - target(iota) over all colorings."""
        return np.array([self.pm_sum(W, io) - self.target(io) for io in self.colorings])

    # ---- vectorized version ----
    def build_index_tables(self):
        """Precompute for each coloring and each PM the flat indices into W."""
        nC, nM = len(self.colorings), len(self.pms)
        # idx[c, m, k] = flat index into W.ravel() for k-th edge of matching m under coloring c
        half = self.n // 2
        idx = np.empty((nC, nM, half), dtype=np.int64)
        d = self.d
        for ci, io in enumerate(self.colorings):
            for mi, m in enumerate(self.pms):
                for k, (u, v) in enumerate(m):
                    e = self.eidx[(u, v)]
                    idx[ci, mi, k] = (e * d + io[u]) * d + io[v]
        self.idx = idx
        self.tgt = np.array([self.target(io) for io in self.colorings])
        return idx

    def residuals_fast(self, Wflat):
        """Wflat: complex array of length nE*d*d."""
        terms = Wflat[self.idx]                  # (nC, nM, half)
        prods = terms.prod(axis=2)               # (nC, nM)
        return prods.sum(axis=1) - self.tgt

    def jacobian_fast(self, Wflat):
        """d residual_c / d Wflat_x  (complex Jacobian, holomorphic)."""
        nC, nM, half = self.idx.shape
        terms = Wflat[self.idx]                  # (nC, nM, half)
        J = np.zeros((nC, Wflat.size), dtype=complex)
        for k in range(half):
            # product of the other edges
            others = np.ones((nC, nM), dtype=complex)
            for k2 in range(half):
                if k2 != k:
                    others *= terms[:, :, k2]
            # scatter-add into J at column idx[:, :, k]
            for mi in range(nM):
                cols = self.idx[:, mi, k]
                J[np.arange(nC), cols] += others[:, mi]
        return J


def check_witness():
    """Sanity checks against the Lean witnesses."""
    # N=4, D=2 witness
    g = MQG(4, 2)
    W = np.zeros((g.nE, 2, 2), dtype=complex)
    W[g.eidx[(0, 1)], 0, 0] = 1
    W[g.eidx[(2, 3)], 0, 0] = 1
    W[g.eidx[(0, 2)], 1, 1] = 1
    W[g.eidx[(1, 3)], 1, 1] = 1
    r = g.residuals(W)
    assert np.allclose(r, 0), f"N4D2 witness failed: {np.abs(r).max()}"

    # N=4, D=3 witness
    g = MQG(4, 3)
    W = np.zeros((g.nE, 3, 3), dtype=complex)
    for (e, c) in [((0, 1), 0), ((2, 3), 0), ((0, 2), 1), ((1, 3), 1), ((0, 3), 2), ((1, 2), 2)]:
        W[g.eidx[e], c, c] = 1
    r = g.residuals(W)
    assert np.allclose(r, 0), f"N4D3 witness failed: {np.abs(r).max()}"

    # N=6, D=2 witness (cycle 0-1-2-3-4-5)
    g = MQG(6, 2)
    W = np.zeros((g.nE, 2, 2), dtype=complex)
    for (e, c) in [((0, 1), 0), ((2, 3), 0), ((4, 5), 0), ((0, 5), 1), ((1, 2), 1), ((3, 4), 1)]:
        W[g.eidx[e], c, c] = 1
    r = g.residuals(W)
    assert np.allclose(r, 0), f"N6D2 witness failed: {np.abs(r).max()}"

    # fast path agrees with slow path on random W
    g = MQG(6, 3)
    g.build_index_tables()
    rng = np.random.default_rng(0)
    W = rng.normal(size=(g.nE, 3, 3)) + 1j * rng.normal(size=(g.nE, 3, 3))
    r1 = g.residuals(W)
    r2 = g.residuals_fast(W.ravel())
    assert np.allclose(r1, r2), f"fast/slow mismatch {np.abs(r1-r2).max()}"

    # jacobian check by finite differences
    Wf = W.ravel().copy()
    J = g.jacobian_fast(Wf)
    h = 1e-7
    for t in [0, 17, 100]:
        Wp = Wf.copy(); Wp[t] += h
        num = (g.residuals_fast(Wp) - g.residuals_fast(Wf)) / h
        assert np.allclose(num, J[:, t], atol=1e-4), f"jac mismatch at {t}"
    print("all sanity checks passed;",
          f"N=6: {len(g.pms)} PMs, {len(g.colorings)} colorings, {g.nE * 9} complex unknowns")


if __name__ == "__main__":
    check_witness()
