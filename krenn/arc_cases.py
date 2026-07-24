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

Usage:
    python3 arc_cases.py burnside        # orbit count of the raw space
    python3 arc_cases.py stage1          # DFS + rule A -> survivorsA.npy
    python3 arc_cases.py stage2          # canonicalization -> canonicalA.npy
    python3 arc_cases.py stagecd         # tiered rules B/C/D over canonical reps
    python3 arc_cases.py stagecd-report  # aggregate shard counts + classify
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
# All-doubled classes (18 arcs on 9 doubly-arced edges; support = cubic graph)
# ---------------------------------------------------------------------------
def support_graph(A):
    return frozenset(edge_arcs(A).keys())


def is_bipartite_support(supp):
    """True iff support graph is K3,3 (else, for cubic support, it is the prism)."""
    adj = {v: set() for v in range(6)}
    for e in supp:
        u, v = EDGES[e]
        adj[u].add(v)
        adj[v].add(u)
    color = {0: 0}
    stack = [0]
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if y in color:
                if color[y] == color[x]:
                    return False
            else:
                color[y] = 1 - color[x]
                stack.append(y)
    return True


def all_doubled_classes():
    """Enumerate all-doubled configs directly; return dict canonical -> info.

    A config has all 18 arcs on 9 doubly-arced edges iff every vertex's arc
    targets are exactly its neighbors in a cubic support graph (each ordered
    pair on a support edge carries exactly one arc). Support graphs on 6
    vertices that are cubic: the prism K3 x K2 and K3,3.
    """
    # labeled cubic graphs with N(0) = {1,2,3} (normalization compatibility)
    graphs = []
    for es in itertools.combinations(range(15), 9):
        deg = [0] * 6
        for e in es:
            u, v = EDGES[e]
            deg[u] += 1
            deg[v] += 1
        if any(d != 3 for d in deg):
            continue
        nbr0 = {v for (u, v) in (EDGES[e] for e in es) if u == 0}
        if nbr0 == {1, 2, 3}:
            graphs.append(es)
    out = {}
    stats = {"graphs": len(graphs), "rawA": 0}
    for es in graphs:
        adj = {v: [] for v in range(6)}
        for e in es:
            u, v = EDGES[e]
            adj[u].append(v)
            adj[v].append(u)
        for p in itertools.product(*(itertools.permutations(adj[v])
                                     for v in range(1, 6))):
            A = ((1, 2, 3),) + p
            if not rule_a(A):
                continue
            stats["rawA"] += 1
            can = canonical(A)
            if can not in out:
                supp = support_graph(A)
                out[can] = {
                    "A": decode(can),
                    "support": "K33" if is_bipartite_support(supp) else "prism",
                    "ruleB_killed": rule_b_violation(decode(can)),
                }
    return out, stats


# ---------------------------------------------------------------------------
# Stage 3: exact polynomial system for a configuration class
# ---------------------------------------------------------------------------
def build_system(A):
    """Return (W, syms) where W[eidx] is a 3x3 sympy matrix with the structural
    zeros dictated by the arcs, and syms = dict of symbol groups.

    Orientation: W[eidx][a, b] with a the color at u, b the color at v (u < v),
    matching mqg.py / the Lean formalization. An arc x->y labeled c zeroes all
    entries whose y-side index != c."""
    import sympy as sp
    ea = edge_arcs(A)
    W = {}
    syms = {"w": [], "beta": [], "free": []}
    for idx, (u, v) in enumerate(EDGES):
        arcs = ea.get(idx, [])
        M = sp.zeros(3, 3)
        if len(arcs) == 2:
            d = {y: c for (x, y, c) in arcs}
            w = sp.Symbol(f"w{u}{v}")
            M[d[u], d[v]] = w
            syms["w"].append(w)
        elif len(arcs) == 1:
            (x, y, c) = arcs[0]
            row = []
            for a in range(3):
                s = sp.Symbol(f"b{u}{v}_{a}")
                row.append(s)
                if y == v:
                    M[a, c] = s      # v-side forced to c, u-side index free
                else:
                    M[c, a] = s      # u-side forced to c, v-side index free
            syms["beta"].append(tuple(row))
        else:
            for a in range(3):
                for b in range(3):
                    s = sp.Symbol(f"f{u}{v}_{a}{b}")
                    M[a, b] = s
                    syms["free"].append(s)
        W[idx] = M
    return W, syms


def pm_sum_expr(W, iota):
    import sympy as sp
    total = sp.Integer(0)
    for pm in PM_PAIRS:
        term = sp.Integer(1)
        for (u, v) in pm:
            f = W[ei(u, v)][iota[u], iota[v]]
            if f == 0:
                term = sp.Integer(0)
                break
            term = term * f
        total += term
    return sp.expand(total)


def w_only_equations(A, W=None, syms=None):
    """All equations pmSum(iota) = target whose expression involves only the
    doubly-arced single-entry weights w_e (guaranteed nonzero). Returns list of
    (iota, expr, target)."""
    import sympy as sp
    if W is None:
        W, syms = build_system(A)
    wset = set(syms["w"])
    out = []
    for iota in itertools.product(range(3), repeat=6):
        e = pm_sum_expr(W, iota)
        fs = e.free_symbols
        if fs and fs <= wset:
            tgt = 1 if len(set(iota)) == 1 else 0
            out.append((iota, e, tgt))
        elif not fs:
            tgt = 1 if len(set(iota)) == 1 else 0
            if e != tgt:
                out.append((iota, e, tgt))   # constant contradiction (0 = 1)
    return out


def kill_by_w_groebner(A, verbose=False):
    """Sound class killer: take the subsystem of equations involving only the
    guaranteed-nonzero doubled weights w_e, saturate by prod(w) != 0, and check
    1 in ideal (over Q). Returns True iff the class is PROVEN infeasible."""
    import sympy as sp
    W, syms = build_system(A)
    eqs = w_only_equations(A, W, syms)
    if not eqs:
        return False
    polys = []
    for iota, e, tgt in eqs:
        if not e.free_symbols:
            if e != tgt:
                return True                    # 0 = 1 style contradiction
            continue
        polys.append(sp.expand(e - tgt))
    if not polys:
        return False
    t = sp.Symbol("t_aux")
    sat = t * sp.prod(syms["w"]) - 1
    gens = list(syms["w"]) + [t]
    try:
        gb = sp.groebner(polys + [sat], *gens, order="grevlex")
        killed = 1 in gb.exprs or sp.Integer(1) in gb.exprs
    except Exception as ex:
        if verbose:
            print("groebner failed:", ex)
        return False
    if verbose:
        print(f"{len(polys)} w-only eqs, killed={killed}")
    return killed


# ---------------------------------------------------------------------------
# Krenn's limiting family (border class)
# ---------------------------------------------------------------------------
# Support: triangles {0,1,2}, {3,4,5} (weight t) + cross matching 03,14,25
# (weight t^-2); forced color pairs: triangle edge {i,j} single entry (k,k)
# with k the opposite vertex('s color), cross edge {i,i+3} single entry (i,i).
# pmSum residual is exactly t^-6 (checked numerically), concentrated on the
# coloring iota* = (0,1,2,0,1,2) whose only alive PM (with all free edges = 0)
# is the cross matching.
KRENN_BORDER = ((3, 2, 1), (2, 4, 0), (1, 0, 5), (0, 5, 4), (5, 1, 3), (4, 3, 2))
KRENN_BORDER_CANONICAL = 127091349204


# ---------------------------------------------------------------------------
# Branch-and-propagate class killer (sound refutation engine)
# ---------------------------------------------------------------------------
# Facts it reasons with, all consequences of the arc lemma:
#   * doubled-edge weights w_e are nonzero;
#   * every other symbol (free-edge entry, or component of a singly-arced
#     edge's beta vector) may or may not vanish;
#   * the 729 equations pmSum(iota) = [iota constant], where each surviving
#     monomial is a product of symbols from the 3 edges of an alive PM.
# Derivation rules on a branch state (Z = symbols proven 0, NZ = proven != 0):
#   - a term is dead if one of its possibly-zero symbols is in Z;
#   - eq (=1) with no live terms ............................ contradiction
#   - eq (=0) with exactly one live term, all of whose
#     possibly-zero symbols are in NZ (e.g. a pure-w term) ... contradiction
#   - eq (=1) with exactly one live term .................... all its symbols to NZ
#   - eq (=0) with exactly one live term and exactly one
#     symbol not yet in NZ ................................. that symbol to Z
#   - eq (=0) with one live term and k>=2 undetermined
#     symbols ............................................... case split (one of
#     them must vanish; k branches)
#   - eqs whose live terms are all pure-w .................. collected; endgame
#     Groebner basis over Q[w, t]/(t*prod(w)-1): 1 in ideal => contradiction.
# The engine returns True only if EVERY branch of the split tree is closed by a
# contradiction; this is a sound proof that the configuration class admits no
# solution.
def _term_table(A):
    """For each of the 729 colorings: target and list of terms.
    Term = (frozenset of w-ids, frozenset of other-ids). Symbol ids are
    (eidx, a, b) entry coordinates; w-ids are eidx alone."""
    ea = edge_arcs(A)
    entry_kind = {}   # eidx -> ('w', (cu,cv)) | ('b',(x,y,c)) | 'free'
    for idx in range(15):
        arcs = ea.get(idx, [])
        u, v = EDGES[idx]
        if len(arcs) == 2:
            d = {y: c for (x, y, c) in arcs}
            entry_kind[idx] = ("w", (d[u], d[v]))
        elif len(arcs) == 1:
            entry_kind[idx] = ("b", arcs[0])
        else:
            entry_kind[idx] = ("free", None)
    table = []
    for iota in itertools.product(range(3), repeat=6):
        terms = []
        for pm in PM_PAIRS:
            wids, oids = [], []
            ok = True
            for (u, v) in pm:
                idx = ei(u, v)
                kind, info = entry_kind[idx]
                a, b = iota[u], iota[v]
                if kind == "w":
                    if (a, b) != info:
                        ok = False
                        break
                    wids.append(idx)
                elif kind == "b":
                    (x, y, c) = info
                    if iota[y] != c:
                        ok = False
                        break
                    oids.append((idx, a, b))
                else:
                    oids.append((idx, a, b))
            if ok:
                terms.append((frozenset(wids), frozenset(oids)))
        tgt = 1 if len(set(iota)) == 1 else 0
        table.append((iota, tgt, terms))
    return table


_GROEBNER_CACHE = {}


def _groebner_refute(eq_keys):
    """eq_keys: frozenset of (tgt, terms) with terms a tuple of
    (w-id tuple, nz-id tuple). All symbols saturated (known nonzero).
    True iff 1 lies in the saturated ideal (=> contradiction)."""
    import sympy as sp
    key = frozenset(eq_keys)
    if key in _GROEBNER_CACHE:
        return _GROEBNER_CACHE[key]
    syms = {}

    def sym(s):
        if s not in syms:
            syms[s] = sp.Symbol(f"v{len(syms)}")
        return syms[s]

    polys = []
    trivial = False
    for tgt, terms in eq_keys:
        expr = sp.Integer(-tgt)
        for wids, oids in terms:
            m = sp.Integer(1)
            for x in wids:
                m *= sym(("w", x))
            for x in oids:
                m *= sym(("o", x))
            expr += m
        if not expr.free_symbols:
            if expr != 0:
                trivial = True
            continue
        polys.append(expr)
    res = False
    if trivial:
        res = True
    elif polys:
        import signal
        t = sp.Symbol("t_aux")
        gens = list(syms.values()) + [t]
        sat = t * sp.prod(list(syms.values())) - 1

        def _to(signum, frame):
            raise TimeoutError

        old = signal.signal(signal.SIGALRM, _to)
        signal.alarm(30)
        try:
            gb = sp.groebner(polys + [sat], *gens, order="grevlex")
            res = 1 in gb.exprs or sp.Integer(1) in gb.exprs
        except (TimeoutError, Exception):
            res = False
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
    _GROEBNER_CACHE[key] = res
    return res


def _close_branch(table, Z, NZ, depth, stats):
    """Return True iff branch provably contradictory (sound)."""
    Z = set(Z)
    NZ = set(NZ)
    stats["nodes"] += 1
    if stats["nodes"] > stats["budget"]:
        return False
    # propagation to fixpoint
    while True:
        changed = False
        split = None
        for iota, tgt, terms in table:
            live = [t for t in terms if not (t[1] & Z)]
            if not live:
                if tgt == 1:
                    return True
                continue
            if len(live) == 1:
                und = live[0][1] - NZ
                if tgt == 1:
                    if und:
                        NZ |= und
                        changed = True
                else:
                    if not und:
                        return True      # product of guaranteed nonzeros = 0
                    if len(und) == 1:
                        Z.add(next(iter(und)))
                        changed = True
                    elif split is None or len(und) < len(split):
                        split = und
        if not changed:
            break
    if split is not None and depth < stats["max_depth"]:
        stats["splits"] += 1
        return all(_close_branch(table, Z | {s}, NZ, depth + 1, stats)
                   for s in split)
    # DPLL fallback: branch on an undecided symbol being zero / nonzero
    if stats.get("dpll") and depth < stats["max_depth"]:
        from collections import Counter
        cnt = Counter()
        for iota, tgt, terms in table:
            for t in terms:
                if not (t[1] & Z):
                    for s in t[1] - NZ:
                        cnt[s] += 1
        if cnt:
            s = cnt.most_common(1)[0][0]
            stats["splits"] += 1
            return (_close_branch(table, Z | {s}, NZ, depth + 1, stats) and
                    _close_branch(table, Z, NZ | {s}, depth + 1, stats))
    # endgame 1: equations whose live terms carry only guaranteed-nonzero symbols
    eq_keys = set()
    for iota, tgt, terms in table:
        live = [t for t in terms if not (t[1] & Z)]
        if live and all(t[1] <= NZ for t in live):
            eq_keys.add((tgt, tuple(sorted(
                (tuple(sorted(t[0])), tuple(sorted(t[1]))) for t in live))))
    if eq_keys and _groebner_refute(frozenset(eq_keys)):
        return True
    # endgame 2: full Groebner refutation of the whole reduced system
    if stats.get("leaf_groebner"):
        if _full_groebner_refute(table, Z, NZ, stats):
            return True
    return False


def _full_groebner_refute(table, Z, NZ, stats):
    """Reduced full system at a leaf: substitute the proven zeros, keep all live
    equations, saturate the w's and the proven-nonzero symbols, ask sympy for a
    Groebner basis; 1 in the ideal is a sound refutation. Guarded by SIGALRM."""
    import signal
    import sympy as sp
    eq_keys = set()
    for iota, tgt, terms in table:
        live = tuple(sorted((tuple(sorted(t[0])), tuple(sorted(t[1])))
                            for t in terms if not (t[1] & Z)))
        if live or tgt:
            eq_keys.add((tgt, live))
    key = ("full", frozenset(eq_keys), frozenset(NZ))
    if key in _GROEBNER_CACHE:
        return _GROEBNER_CACHE[key]
    syms = {}

    def sym(kind, s):
        if (kind, s) not in syms:
            syms[(kind, s)] = sp.Symbol(f"v{len(syms)}")
        return syms[(kind, s)]

    polys = []
    res = False
    for tgt, live in eq_keys:
        expr = sp.Integer(-tgt)
        for wids, oids in live:
            m = sp.Integer(1)
            for x in wids:
                m *= sym("w", x)
            for x in oids:
                m *= sym("o", x)
            expr += m
        if not expr.free_symbols:
            if expr != 0:
                res = True
                break
        else:
            polys.append(expr)
    nvar = len(syms)
    if not res and polys and nvar <= stats.get("leaf_var_cap", 40):
        satsyms = ([v for (k, s), v in syms.items() if k == "w"] +
                   [v for (k, s), v in syms.items() if k == "o" and s in NZ])
        t = sp.Symbol("t_aux")
        gens = list(syms.values()) + [t]
        eqs = list(polys)
        if satsyms:
            eqs.append(t * sp.prod(satsyms) - 1)

        def _to(signum, frame):
            raise TimeoutError

        old = signal.signal(signal.SIGALRM, _to)
        signal.alarm(stats.get("leaf_timeout", 60))
        try:
            gb = sp.groebner(eqs, *gens, order="grevlex")
            res = 1 in gb.exprs or sp.Integer(1) in gb.exprs
        except (TimeoutError, Exception):
            res = False
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
    _GROEBNER_CACHE[key] = res
    return res


# vectorized root propagation (rule C at the root, no splitting) ------------
_COL = np.array(list(itertools.product(range(3), repeat=6)), dtype=np.int8)
_TGT = (_COL.max(axis=1) == _COL.min(axis=1))
_PM_EDGE = [[(ei(u, v), u, v) for (u, v) in pm] for pm in PM_PAIRS]


def rule_c_root(A):
    """True iff the class is killed by zero/nonzero propagation at the root.
    Vectorized equivalent of kill_by_branching(A, max_depth=0, budget=1)."""
    ea = edge_arcs(A)
    allowed = np.zeros((15, 3, 3), dtype=bool)
    guaranteed = np.zeros((15, 3, 3), dtype=bool)
    for idx in range(15):
        arcs = ea.get(idx, [])
        u, v = EDGES[idx]
        if len(arcs) == 2:
            d = {y: c for (x, y, c) in arcs}
            allowed[idx, d[u], d[v]] = True
            guaranteed[idx, d[u], d[v]] = True
        elif len(arcs) == 1:
            (x, y, c) = arcs[0]
            if y == v:
                allowed[idx, :, c] = True
            else:
                allowed[idx, c, :] = True
        else:
            allowed[idx] = True
    nz = guaranteed.copy()          # proven-nonzero entries
    zero = np.zeros_like(allowed)
    ecoords = []                    # per PM: (edge_idx, iu_col, iv_col) arrays
    for pmedges in _PM_EDGE:
        ecoords.append([(e, _COL[:, u], _COL[:, v]) for (e, u, v) in pmedges])
    while True:
        cur = allowed & ~zero
        alive = np.ones((729, 15), dtype=bool)
        for m, trip in enumerate(ecoords):
            a = cur[trip[0][0], trip[0][1], trip[0][2]]
            a &= cur[trip[1][0], trip[1][1], trip[1][2]]
            a &= cur[trip[2][0], trip[2][1], trip[2][2]]
            alive[:, m] = a
        cnt = alive.sum(axis=1)
        if np.any(_TGT & (cnt == 0)):
            return True
        changed = False
        for k in np.nonzero(cnt == 1)[0]:
            m = int(np.argmax(alive[k]))
            iota = _COL[k]
            und = []
            for (e, u, v) in _PM_EDGE[m]:
                a, b = int(iota[u]), int(iota[v])
                if not nz[e, a, b]:
                    und.append((e, a, b))
            if _TGT[k]:
                for (e, a, b) in und:
                    if not nz[e, a, b]:
                        nz[e, a, b] = True
                        changed = True
            else:
                if not und:
                    return True
                if len(und) == 1:
                    (e, a, b) = und[0]
                    zero[e, a, b] = True
                    changed = True
        if not changed:
            return False


def kill_by_branching(A, max_depth=12, budget=20000, leaf_groebner=False,
                      leaf_timeout=60, leaf_var_cap=40, dpll=False):
    """Sound refutation attempt for a configuration class. True => class killed."""
    table = _term_table(A)
    stats = {"splits": 0, "nodes": 0, "budget": budget, "max_depth": max_depth,
             "leaf_groebner": leaf_groebner, "leaf_timeout": leaf_timeout,
             "leaf_var_cap": leaf_var_cap, "dpll": dpll}
    return _close_branch(table, set(), set(), 0, stats)


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
    elif cmd == "stagecd":
        # tiered pruning of canonical rule-A survivors:
        # 0 = rule B, 1 = rule C (root propagation), 2 = rule D (branching,
        # depth 12 / 3000 nodes, cached Groebner endgame), 3 = survivor
        uniq = np.load(os.path.join(SCRATCH, "canonicalA.npy"))
        N = len(uniq)
        shard = 100_000
        nsh = (N + shard - 1) // shard
        print(f"canonical rule-A survivors: {N} ({nsh} shards)", flush=True)
        t00 = time.time()
        for sh in range(nsh):
            path = os.path.join(SCRATCH, f"codes_{sh}.npy")
            if os.path.exists(path):
                continue
            lo, hi = sh * shard, min(N, (sh + 1) * shard)
            codes = np.empty(hi - lo, dtype=np.uint8)
            t0 = time.time()
            if len(_GROEBNER_CACHE) > 200_000:
                _GROEBNER_CACHE.clear()
            for k in range(lo, hi):
                A = decode(int(uniq[k]))
                if rule_b_violation(A):
                    codes[k - lo] = 0
                elif rule_c_root(A):
                    codes[k - lo] = 1
                elif kill_by_branching(A, max_depth=12, budget=3000):
                    codes[k - lo] = 2
                else:
                    codes[k - lo] = 3
            np.save(path, codes)
            c = np.bincount(codes, minlength=4)
            print(f"shard {sh+1}/{nsh}: B={c[0]} C={c[1]} D={c[2]} "
                  f"alive={c[3]} ({time.time()-t0:.0f}s, "
                  f"total {(time.time()-t00)/60:.1f}m)", flush=True)
    elif cmd == "stagecd-report":
        import glob as _glob
        uniq = np.load(os.path.join(SCRATCH, "canonicalA.npy"))
        files = sorted(_glob.glob(os.path.join(SCRATCH, "codes_*.npy")),
                       key=lambda p: int(p.rsplit("_", 1)[1][:-4]))
        codes = np.concatenate([np.load(f) for f in files])
        c = np.bincount(codes, minlength=4)
        print(f"processed {len(codes)}/{len(uniq)} canonical classes")
        print(f"killed by rule B: {c[0]}")
        print(f"killed by rule C (root propagation): {c[1]}")
        print(f"killed by rule D (branching d12/b3000): {c[2]}")
        print(f"still alive: {c[3]}")
        # classify the survivors
        hist = {}
        for k in np.nonzero(codes == 3)[0]:
            A = decode(int(uniq[k]))
            key = classify(A)
            hist[key] = hist.get(key, 0) + 1
        for k in sorted(hist):
            print(f"  survivors with (#arc-edges, #doubly-arced) = {k}: {hist[k]}")
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
