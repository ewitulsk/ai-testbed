"""Solve the monochromatic quantum graph system over a prime field GF(p).

If for some prime p the system is UNSAT over GF(p), then no solution over ZZ
exists (reduction mod p), resolving eqSystemN_no_solution_int for that (N, D).

GF(2): boolean vars per weight entry; term = AND of the 3 entries;
equation = XOR of the 15 terms == target bit. Pure SAT.

GF(p), p odd: each entry is a value in {0..p-1}, one-hot encoded; each term's
value tracked via sequential composition; equation via DFA over partial sums
mod p. Solved with Cadical.

Usage: python3 sat_gf.py N D p
"""
import sys
import numpy as np
from pysat.formula import IDPool
from pysat.solvers import Cadical195
from mqg import MQG


def solve_gf2(n, d, return_witness=True):
    g = MQG(n, d)
    g.build_index_tables()
    pool = IDPool()
    W = g.nE * d * d
    bvar = [pool.id(("b", w)) for w in range(W)]
    clauses = []
    half = n // 2
    terms = {}
    for ci in range(len(g.colorings)):
        for mi in range(len(g.pms)):
            wids = tuple(sorted(int(g.idx[ci, mi, k]) for k in range(half)))
            if wids in terms:
                continue
            a = pool.id(("and", wids))
            bs = [bvar[w] for w in wids]
            for b in bs:
                clauses.append([-a, b])
            clauses.append([a] + [-b for b in bs])
            terms[wids] = a
    for ci, io in enumerate(g.colorings):
        t = 1 if all(c == io[0] for c in io) else 0
        tlits = [terms[tuple(sorted(int(g.idx[ci, mi, k]) for k in range(half)))]
                 for mi in range(len(g.pms))]
        # XOR chain over tlits == t
        x = tlits[0]
        for k in range(1, len(tlits)):
            y = pool.id(("xor", ci, k))
            b = tlits[k]
            clauses += [[-y, x, b], [-y, -x, -b], [y, -x, b], [y, x, -b]]
            x = y
        clauses.append([x] if t == 1 else [-x])
    print(f"GF(2) n={n} d={d}: {pool.top} vars, {len(clauses)} clauses", flush=True)
    with Cadical195(bootstrap_with=clauses) as sol:
        sat = sol.solve()
        print("GF(2):", "SAT" if sat else "UNSAT", flush=True)
        if sat and return_witness:
            model = set(l for l in sol.get_model() if l > 0)
            Wv = np.array([1 if bvar[w] in model else 0 for w in range(W)])
            # verify mod 2
            bad = 0
            for ci, io in enumerate(g.colorings):
                s = 0
                for mi in range(len(g.pms)):
                    p = 1
                    for k in range(half):
                        p *= Wv[int(g.idx[ci, mi, k])]
                    s += p
                t = 1 if all(c == io[0] for c in io) else 0
                if s % 2 != t:
                    bad += 1
            print("GF(2) witness verified, bad equations:", bad)
            np.save(f"gf2_witness_n{n}_d{d}.npy", Wv)
        return sat


def solve_gfp(n, d, p):
    """One-hot value encoding over GF(p), DFA for running sums and products."""
    g = MQG(n, d)
    g.build_index_tables()
    pool = IDPool()
    W = g.nE * d * d
    # val[w][v] : entry w has value v  (one-hot)
    val = [[pool.id(("v", w, v)) for v in range(p)] for w in range(W)]
    clauses = []
    for w in range(W):
        clauses.append(val[w])                        # at least one
        for a in range(p):
            for b in range(a + 1, p):
                clauses.append([-val[w][a], -val[w][b]])  # at most one
    half = n // 2
    # term product one-hot per (matching, entries)
    terms = {}
    for ci in range(len(g.colorings)):
        for mi in range(len(g.pms)):
            wids = tuple(sorted(int(g.idx[ci, mi, k]) for k in range(half)))
            if wids in terms:
                continue
            # sequential product: prod after first entry = entry
            cur = val[wids[0]]
            for k in range(1, half):
                nxt = [pool.id(("prod", wids, k, v)) for v in range(p)]
                clauses.append(nxt)
                for a in range(p):
                    for b in range(a + 1, p):
                        clauses.append([-nxt[a], -nxt[b]])
                for a in range(p):
                    for b in range(p):
                        c = (a * b) % p
                        # cur=a & val=b -> nxt=c
                        clauses.append([-cur[a], -val[wids[k]][b], nxt[c]])
                cur = nxt
            terms[wids] = cur
    # equations: sum over 15 terms mod p == target
    for ci, io in enumerate(g.colorings):
        t = 1 if all(c == io[0] for c in io) else 0
        tvals = [terms[tuple(sorted(int(g.idx[ci, mi, k]) for k in range(half)))]
                 for mi in range(len(g.pms))]
        cur = tvals[0]
        for k in range(1, len(tvals)):
            nxt = [pool.id(("sum", ci, k, v)) for v in range(p)]
            clauses.append(nxt)
            for a in range(p):
                for b in range(a + 1, p):
                    clauses.append([-nxt[a], -nxt[b]])
            for a in range(p):
                for b in range(p):
                    c = (a + b) % p
                    clauses.append([-cur[a], -tvals[k][b], nxt[c]])
            cur = nxt
        clauses.append([cur[t]])
    print(f"GF({p}) n={n} d={d}: {pool.top} vars, {len(clauses)} clauses", flush=True)
    with Cadical195(bootstrap_with=clauses) as sol:
        sat = sol.solve()
        print(f"GF({p}):", "SAT" if sat else "UNSAT", flush=True)
        if sat:
            model = set(l for l in sol.get_model() if l > 0)
            Wv = np.array([next(v for v in range(p) if val[w][v] in model)
                           for w in range(W)])
            bad = 0
            for ci, io in enumerate(g.colorings):
                s = 0
                for mi in range(len(g.pms)):
                    pr = 1
                    for k in range(half):
                        pr = (pr * Wv[int(g.idx[ci, mi, k])]) % p
                    s = (s + pr) % p
                t = 1 if all(c == io[0] for c in io) else 0
                if s != t:
                    bad += 1
            print(f"GF({p}) witness verified, bad equations:", bad)
            np.save(f"gf{p}_witness_n{n}_d{d}.npy", Wv)
        return sat


if __name__ == "__main__":
    n, d, p = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    if p == 2:
        solve_gf2(n, d)
    else:
        solve_gfp(n, d, p)
