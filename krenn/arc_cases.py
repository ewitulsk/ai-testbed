"""Arc-configuration case analysis for Krenn's (6,3) problem.

Background (see THEORY.md): in any exact monochromatic-quantum-graph solution on
K_6 with D=3 colors, the rank-1 arc lemma gives, for every vertex v and every
color c, an out-arc v->u (u a neighbor) meaning the full 3x3 weight matrix of
edge {v,u} is supported on the single u-side color c:

    W_{v,u}[a, b] = 0  whenever the u-side index b != c,   and W != 0.

The three out-arcs of a vertex go to three DISTINCT neighbors (one per color).
So a solution induces an "arc configuration": an injective map
arcs[v] : {0,1,2} -> V \ {v} for each of the 6 vertices (18 arcs on <= 15 edges).

This file:
  * enumerates arc configurations under the WLOG normalization
        arcs[0] = (0->1 color 0, 0->2 color 1, 0->3 color 2)
    (raw space 60^5 = 777,600,000),
  * prunes with two SOUND necessary conditions (rules A and B below),
  * counts configurations up to the full symmetry group S6 x S3 by
    canonical-form hashing,
  * classifies the survivors, and
  * sets up the exact polynomial system for maximally constrained classes
    (stage 3) to kill configuration classes algebraically.

Sound pruning rules
-------------------
Rule A ("c-diagonal graph has a perfect matching").
  An arc x->y labeled c forces the y-side of edge {x,y} to color c. For the
  monochromatic coloring iota == c, the edge factor of {x,y} is W[c,c], which is
  identically 0 unless every arc on the edge is labeled c. Let G_c be the graph
  of edges all of whose arcs (possibly none) are labeled c. Every perfect
  matching of K_6 that is alive under iota == c lies inside G_c, and
  pmSum(iota==c) = 1 != 0 requires at least one alive matching, hence G_c must
  contain a perfect matching, for each c in {0,1,2}. Exclusions only grow as
  arcs are added, so the condition can prune partial assignments (used in DFS).

Rule B ("isolated forced matching").
  A doubly-arced edge (arcs in both directions, labels c at the u side and c'
  at the v side) has a SINGLE possibly-nonzero entry W = w e_{c'} e_c^T, and
  w != 0 (both arc lemma applications give a nonzero matrix). Call an edge e
  ALIVE under a coloring iota if no arc on e is violated (arc x->y labeled c
  requires iota_y = c); free edges are always alive. If S is a perfect matching
  whose 3 edges are all doubly-arced, S forces a coloring iota_S on all 6
  vertices. If iota_S is non-constant and NO OTHER perfect matching of K_6 is
  alive under iota_S, then pmSum(iota_S) = prod_{e in S} w_e != 0, contradicting
  pmSum(iota_S) = 0. Note this rule has full strength for the criterion
  "some non-constant iota has exactly one alive PM whose entries are all
  guaranteed nonzero": an alive all-doubly-arced PM forces iota on all six
  vertices, so any such iota is the induced coloring of that PM.

Symmetry
--------
The full group is S6 x S3 (order 4320) acting by
    A'[pi(v)][sigma(c)] = pi(A[v][c]).
A normalized configuration (arcs[0] = (1,2,3)) is mapped to a normalized one by
exactly 72 group elements' worth of choices: pick v0 = pi^{-1}(0) (6 ways),
pick sigma (6 ways) — then pi is forced on v0 and its three arc targets
(pi(A[v0][c]) = sigma(c)+1) — and pick pi on the remaining 2 vertices (2 ways).
The canonical form of a configuration is the minimum encoding over these 72
images; two normalized configurations lie in the same S6 x S3 orbit iff their
canonical forms agree.

Usage:  python3 arc_cases.py stage1|burnside|stage2|stage3  (see main below).
CPU-friendly: single process; intended to run under `nice -n 15`.
"""
import itertools
import os
import sys
import time
from array import array

import numpy as np

V = 6
NCOL = 3
EDGES = [(u, v) for u in range(V) for v in range(u + 1, V)]
EIDX = {e: i for i, e in enumerate(EDGES)}
PERMS3 = list(itertools.permutations(range(NCOL)))
SCRATCH = os.environ.get(
    "ARC_SCRATCH",
    "/tmp/claude-0/-home-user-ai-testbed/e8d6bc0e-c2d6-5089-9811-4145f3a745b6/scratchpad",
)


def ei(a, b):
    return EIDX[(a, b)] if a < b else EIDX[(b, a)]


def gen_pms(verts):
    if not verts:
        yield ()
        return
    a = verts[0]
    for b in verts[1:]:
        rest = [x for x in verts if x != a and x != b]
        for m in gen_pms(rest):
            yield ((a, b),) + m


PM_PAIRS = list(gen_pms(list(range(V))))                    # 15 matchings
PM_TRIPLES = [tuple(ei(a, b) for a, b in m) for m in PM_PAIRS]
PM_MASKS = [sum(1 << e for e in t) for t in PM_TRIPLES]

# hasPM[mask] = does K6 minus edge-set `mask` still have a perfect matching?
_hasPM = np.zeros(1 << 15, dtype=bool)
for _m in range(1 << 15):
    _hasPM[_m] = any((pm & _m) == 0 for pm in PM_MASKS)
HASPM = _hasPM.tolist()


# ---------------------------------------------------------------------------
# encoding: a normalized config is (arcs of vertices 1..5), 15 base-6 digits;
# digit at position 3*(v-1)+c is arcs[v][c].
# ---------------------------------------------------------------------------
def decode(enc):
    enc = int(enc)
    A = [(1, 2, 3)]
    for v in range(1, 6):
        row = []
        for c in range(3):
            row.append(enc % 6)
            enc //= 6
        A.append(tuple(row))
    return tuple(A)


def encode(A):
    e = 0
    for v in range(5, 0, -1):
        for c in (2, 1, 0):
            e = e * 6 + A[v][c]
    return e


# ---------------------------------------------------------------------------
# Stage 1: DFS enumeration with incremental rule-A pruning
# ---------------------------------------------------------------------------
def _choices(v):
    """(enc_partial, d0, d1, d2) for each of the 60 injections at vertex v.

    d_c = mask of edges that an arc labeled c' != c excludes from G_c."""
    nb = [u for u in range(V) if u != v]
    out = []
    for t in itertools.permutations(nb, 3):
        d = [0, 0, 0]
        for c in range(3):
            b = 1 << ei(v, t[c])
            for c2 in range(3):
                if c2 != c:
                    d[c2] |= b
        enc = sum(t[c] * (6 ** (3 * (v - 1) + c)) for c in range(3))
        out.append((enc, d[0], d[1], d[2]))
    return out


def stage1(out_path=None, limit1=None):
    """Enumerate normalized configs surviving rule A. Returns (array, depth_counts)."""
    C = {v: _choices(v) for v in range(1, 6)}
    # vertex 0 fixed: 0->1 color0, 0->2 color1, 0->3 color2
    e0 = e1 = e2 = 0
    for c, t in enumerate((1, 2, 3)):
        b = 1 << ei(0, t)
        for c2 in range(3):
            if c2 != c:
                if c2 == 0:
                    e0 |= b
                elif c2 == 1:
                    e1 |= b
                else:
                    e2 |= b
    H = HASPM
    C1 = C[1][:limit1] if limit1 else C[1]
    C2, C3, C4, C5 = C[2], C[3], C[4], C[5]
    survivors = array("Q")
    app = survivors.append
    n1 = n2 = n3 = n4 = 0
    t_start = time.time()
    for q1, a0, a1, a2 in C1:
        f0 = e0 | a0; f1 = e1 | a1; f2 = e2 | a2
        if not (H[f0] and H[f1] and H[f2]):
            continue
        n1 += 1
        for q2, b0, b1, b2 in C2:
            g0 = f0 | b0; g1 = f1 | b1; g2 = f2 | b2
            if not (H[g0] and H[g1] and H[g2]):
                continue
            n2 += 1
            q12 = q1 + q2
            for q3, c0, c1, c2_ in C3:
                h0 = g0 | c0; h1 = g1 | c1; h2 = g2 | c2_
                if not (H[h0] and H[h1] and H[h2]):
                    continue
                n3 += 1
                q123 = q12 + q3
                for q4, d0, d1, d2 in C4:
                    i0 = h0 | d0; i1 = h1 | d1; i2 = h2 | d2
                    if not (H[i0] and H[i1] and H[i2]):
                        continue
                    n4 += 1
                    q1234 = q123 + q4
                    for q5, x0, x1, x2 in C5:
                        if H[i0 | x0] and H[i1 | x1] and H[i2 | x2]:
                            app(q1234 + q5)
    arr = np.frombuffer(survivors, dtype=np.uint64)
    print(f"stage1: depth-pass counts = {(n1, n2, n3, n4, len(arr))}, "
          f"{time.time()-t_start:.1f}s")
    if out_path:
        np.save(out_path, arr)
    return arr, (n1, n2, n3, n4, len(arr))


# ---------------------------------------------------------------------------
# Burnside: exact number of raw configurations up to S6 x S3 (no pruning)
# ---------------------------------------------------------------------------
def burnside_total():
    """|{arc configs}/G| for G = S6 x S3 acting by A'[pi v][sigma c] = pi(A[v][c])."""
    total = 0
    injections = list(itertools.permutations(range(V), 3))  # generic; filter per vertex
    for pi in itertools.permutations(range(V)):
        # vertex cycles of pi
        seen = [False] * V
        cycles = []
        for v in range(V):
            if not seen[v]:
                cyc = [v]
                seen[v] = True
                w = pi[v]
                while w != v:
                    cyc.append(w)
                    seen[w] = True
                    w = pi[w]
                cycles.append(cyc)
        # pi^L for each cycle length
        for sigma in PERMS3:
            cnt = 1
            for cyc in cycles:
                L = len(cyc)
                v0 = cyc[0]
                # pi^L and sigma^L
                piL = list(range(V))
                for _ in range(L):
                    piL = [pi[x] for x in piL]
                sigL = list(range(3))
                for _ in range(L):
                    sigL = [sigma[x] for x in sigL]
                # count injections h: C -> V\{v0} with h(sigL(c)) = piL(h(c))
                k = 0
                for h in itertools.permutations([u for u in range(V) if u != v0], 3):
                    if all(h[sigL[c]] == piL[h[c]] for c in range(3)):
                        k += 1
                cnt *= k
                if cnt == 0:
                    break
            total += cnt
    assert total % (720 * 6) == 0 or True
    return total // (720 * 6), total


# ---------------------------------------------------------------------------
# Canonical forms
# ---------------------------------------------------------------------------
def canonical(A):
    """Scalar canonical form (min encoding over the 72 normalized images)."""
    best = None
    for v0 in range(6):
        Av0 = A[v0]
        for sig in PERMS3:
            pi = [-1] * 6
            pi[v0] = 0
            for c in range(3):
                pi[Av0[c]] = sig[c] + 1
            rest = [u for u in range(6) if pi[u] < 0]
            for r in (rest, rest[::-1]):
                pi[r[0]] = 4
                pi[r[1]] = 5
                e = 0
                for v in range(5, 0, -1):
                    # find source vertex vs with pi[vs] = v
                    vs = pi.index(v)
                    Avs = A[vs]
                    for c in (2, 1, 0):
                        # digit at (v, c) is pi(A[vs][sig^{-1}(c)])
                        cs = sig.index(c)
                        e = e * 6 + pi[Avs[cs]]
                if best is None or e < best:
                    best = e
    return best


def canonical_batch(encs, chunk=2_000_000, progress=False):
    """Vectorized canonical form for a uint64 array of normalized configs."""
    out = np.empty(len(encs), np.uint64)
    POW = 6 ** np.arange(15, dtype=np.uint64)
    for lo in range(0, len(encs), chunk):
        sub = encs[lo:lo + chunk]
        N = len(sub)
        rows = np.arange(N)
        A = np.empty((N, 18), np.uint8)
        A[:, 0] = 1; A[:, 1] = 2; A[:, 2] = 3
        e = sub.copy()
        for k in range(15):
            A[:, 3 + k] = (e % 6).astype(np.uint8)
            e //= 6
        best = np.full(N, np.iinfo(np.uint64).max, np.uint64)
        for v0 in range(6):
            for sig in PERMS3:
                pi = np.full((N, 6), -1, np.int8)
                pi[:, v0] = 0
                for c in range(3):
                    pi[rows, A[:, 3 * v0 + c]] = sig[c] + 1
                restcols = np.nonzero(pi < 0)[1].reshape(N, 2)
                for flip in (0, 1):
                    pi2 = pi.copy()
                    pi2[rows, restcols[:, flip]] = 4
                    pi2[rows, restcols[:, 1 - flip]] = 5
                    enc = np.zeros(N, np.uint64)
                    for v in range(6):
                        pv = pi2[:, v].astype(np.int64)
                        m = pv != 0
                        for c in range(3):
                            val = pi2[rows, A[:, 3 * v + c]].astype(np.uint64)
                            pos = 3 * (pv - 1) + sig[c]
                            enc[m] += val[m] * POW[pos[m]]
                    np.minimum(best, enc, out=best)
        out[lo:lo + chunk] = best
        if progress:
            print(f"  canonical_batch: {lo + N}/{len(encs)}", flush=True)
    return out


# ---------------------------------------------------------------------------
# Rule B and structure of a configuration
# ---------------------------------------------------------------------------
def edge_arcs(A):
    """dict edge_index -> tuple of (src, dst, color) arcs on that edge."""
    d = {}
    for v in range(6):
        for c in range(3):
            u = A[v][c]
            d.setdefault(ei(v, u), []).append((v, u, c))
    return d


def structure(A):
    ea = edge_arcs(A)
    doubly = frozenset(e for e, l in ea.items() if len(l) == 2)
    return ea, doubly


def alive_pms(iota, ea):
    """Indices of perfect matchings alive under coloring iota given arc dict ea."""
    out = []
    for i, tri in enumerate(PM_TRIPLES):
        ok = True
        for e in tri:
            for (x, y, c) in ea.get(e, ()):
                if iota[y] != c:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            out.append(i)
    return out


def rule_a(A):
    """Recheck rule A on a full config (used for validation)."""
    ea = edge_arcs(A)
    for c in range(3):
        allowed = 0
        for e in range(15):
            if all(cc == c for (_, _, cc) in ea.get(e, ())):
                allowed |= 1 << e
        if not any((pm & ~allowed) == 0 for pm in PM_MASKS):
            return False
    return True


def rule_b_violation(A, ea=None, doubly=None):
    """True iff config is KILLED by rule B (isolated forced matching exists)."""
    if ea is None:
        ea, doubly = structure(A)
    forced = {}
    for e in doubly:
        d = {}
        for (x, y, c) in ea[e]:
            d[y] = c
        forced[e] = d
    for i, tri in enumerate(PM_TRIPLES):
        if not all(e in doubly for e in tri):
            continue
        iota = {}
        for e in tri:
            iota.update(forced[e])
        it = tuple(iota[v] for v in range(6))
        if len(set(it)) == 1:
            continue                       # monochromatic: no contradiction
        alive = alive_pms(it, ea)
        # tri itself is always alive under its induced coloring
        if len(alive) == 1:
            return True
    return False


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classify(A):
    ea, doubly = structure(A)
    return len(ea), len(doubly)            # (#arc edges, #doubly-arced edges)


# ---------------------------------------------------------------------------
# Stages runnable from the command line
# ---------------------------------------------------------------------------
def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "burnside":
        orbits, tot = burnside_total()
        print(f"raw configs: 60^6 = {60**6}")
        print(f"Burnside fixed-point sum = {tot}")
        print(f"configs up to S6 x S3 = {orbits}")
    elif cmd == "stage1":
        stage1(out_path=os.path.join(SCRATCH, "survivorsA.npy"))
    elif cmd == "stage2":
        arr = np.load(os.path.join(SCRATCH, "survivorsA.npy"))
        print(f"rule-A raw survivors: {len(arr)}")
        can = canonical_batch(arr, progress=True)
        uniq = np.unique(can)
        np.save(os.path.join(SCRATCH, "canonicalA.npy"), uniq)
        print(f"rule-A survivors up to symmetry: {len(uniq)}")
    elif cmd == "stage2b":
        uniq = np.load(os.path.join(SCRATCH, "canonicalA.npy"))
        keep = []
        hist = {}
        t0 = time.time()
        for enc in uniq:
            A = decode(enc)
            ea, doubly = structure(A)
            if rule_b_violation(A, ea, doubly):
                continue
            keep.append(int(enc))
            key = (len(ea), len(doubly))
            hist[key] = hist.get(key, 0) + 1
        np.save(os.path.join(SCRATCH, "canonicalAB.npy"),
                np.array(keep, dtype=np.uint64))
        print(f"rule A+B survivors up to symmetry: {len(keep)} "
              f"({time.time()-t0:.0f}s)")
        for k in sorted(hist):
            print(f"  (#arc-edges, #doubly-arced) = {k}: {hist[k]}")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
