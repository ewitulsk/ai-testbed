"""GF(2) N=6 D=3 with partial symmetry breaking (lex-leader over S6 x S3 generators)."""
import sys
import numpy as np
from pycryptosat import Solver
from mqg import MQG


def entry_index(g, d, u, v, i, j):
    """Weight bit index (1-based) for edge {u,v}, color i at u, j at v."""
    if u > v:
        u, v, i, j = v, u, j, i
    return (g.eidx[(u, v)] * d + i) * d + j + 1


def perm_action_vertex(g, d, pi):
    """Permutation of weight-bit indices induced by vertex permutation pi."""
    m = {}
    for (u, v) in g.edges:
        for i in range(d):
            for j in range(d):
                src = entry_index(g, d, u, v, i, j)
                dst = entry_index(g, d, pi[u], pi[v], i, j)
                m[src] = dst
    return m


def perm_action_color(g, d, sig):
    m = {}
    for (u, v) in g.edges:
        for i in range(d):
            for j in range(d):
                src = entry_index(g, d, u, v, i, j)
                dst = entry_index(g, d, u, v, sig[i], sig[j])
                m[src] = dst
    return m


def add_lex(s, nxt, order, m):
    """Add lex constraint: bits (in fixed order) <=_lex permuted bits.
    order: list of var indices; m: permutation dict on vars."""
    # e_0 = True; e_k <-> e_{k-1} & (x_k == y_k); e_{k-1} -> (~x_k | y_k)
    prev = None
    for k, x in enumerate(order):
        y = m[x]
        if x == y:
            continue
        if prev is None:
            s.add_clause([-x, y])          # x1 <= y1
            e = nxt; nxt += 1
            # e <-> (x == y)
            s.add_clause([-e, -x, y]); s.add_clause([-e, x, -y])
            s.add_clause([e, x, y]); s.add_clause([e, -x, -y])
            prev = e
        else:
            s.add_clause([-prev, -x, y])
            e = nxt; nxt += 1
            eq = nxt; nxt += 1
            s.add_clause([-eq, -x, y]); s.add_clause([-eq, x, -y])
            s.add_clause([eq, x, y]); s.add_clause([eq, -x, -y])
            # e <-> prev & eq
            s.add_clause([-e, prev]); s.add_clause([-e, eq]); s.add_clause([e, -prev, -eq])
            prev = e
    return nxt


def main(n, d):
    g = MQG(n, d)
    g.build_index_tables()
    W = g.nE * d * d
    half = n // 2
    nxt = W + 1
    s = Solver(threads=1)
    terms = {}
    for ci in range(len(g.colorings)):
        for mi in range(len(g.pms)):
            wids = tuple(sorted(int(g.idx[ci, mi, k]) + 1 for k in range(half)))
            if wids in terms:
                continue
            a = nxt; nxt += 1
            for b in wids:
                s.add_clause([-a, b])
            s.add_clause([a] + [-b for b in wids])
            terms[wids] = a
    for ci, io in enumerate(g.colorings):
        t = all(c == io[0] for c in io)
        tl = [terms[tuple(sorted(int(g.idx[ci, mi, k]) + 1 for k in range(half)))]
              for mi in range(len(g.pms))]
        s.add_xor_clause(tl, bool(t))
    # symmetry breaking
    order = list(range(1, W + 1))
    gens = []
    for k in range(n - 1):  # adjacent vertex transpositions
        pi = list(range(n)); pi[k], pi[k + 1] = pi[k + 1], pi[k]
        gens.append(perm_action_vertex(g, d, pi))
    pi = list(range(1, n)) + [0]  # n-cycle
    gens.append(perm_action_vertex(g, d, pi))
    for k in range(d - 1):
        sig = list(range(d)); sig[k], sig[k + 1] = sig[k + 1], sig[k]
        gens.append(perm_action_color(g, d, sig))
    sig = list(range(1, d)) + [0]
    gens.append(perm_action_color(g, d, sig))
    for m in gens:
        nxt = add_lex(s, nxt, order, m)
    print(f"SYM GF(2) n={n} d={d}: ~{nxt-1} vars", flush=True)
    sat, model = s.solve()
    print("SYM GF(2):", "SAT" if sat else "UNSAT", flush=True)
    if sat:
        Wv = np.array([1 if model[w + 1] else 0 for w in range(W)])
        bad = 0
        for ci, io in enumerate(g.colorings):
            ssum = 0
            for mi in range(len(g.pms)):
                p = 1
                for k in range(half):
                    p *= Wv[int(g.idx[ci, mi, k])]
                ssum += p
            t = 1 if all(c == io[0] for c in io) else 0
            if ssum % 2 != t:
                bad += 1
        print("witness verified, bad:", bad)
        np.save(f"gf2sym_witness_n{n}_d{d}.npy", Wv)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
