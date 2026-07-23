"""GF(2) monochromatic quantum graph system via CryptoMiniSat native XOR.

Vars 1..135: weight bits. AND-vars for each (matching, entry-triple).
Equations: XOR of the 15 AND-vars == target bit, as native xor clauses.
Usage: python3 gf2_cms.py N D
"""
import sys
import numpy as np
from pycryptosat import Solver
from mqg import MQG


def main(n, d):
    g = MQG(n, d)
    g.build_index_tables()
    W = g.nE * d * d
    half = n // 2
    nxt = W + 1
    terms = {}
    clauses = []
    for ci in range(len(g.colorings)):
        for mi in range(len(g.pms)):
            wids = tuple(sorted(int(g.idx[ci, mi, k]) + 1 for k in range(half)))
            if wids in terms:
                continue
            a = nxt; nxt += 1
            for b in wids:
                clauses.append([-a, b])
            clauses.append([a] + [-b for b in wids])
            terms[wids] = a
    s = Solver(threads=2)
    for c in clauses:
        s.add_clause(c)
    for ci, io in enumerate(g.colorings):
        t = all(c == io[0] for c in io)
        tl = [terms[tuple(sorted(int(g.idx[ci, mi, k]) + 1 for k in range(half)))]
              for mi in range(len(g.pms))]
        s.add_xor_clause(tl, bool(t))
    print(f"CMS GF(2) n={n} d={d}: {nxt-1} vars, {len(clauses)} clauses + {len(g.colorings)} xors", flush=True)
    sat, model = s.solve()
    print("CMS GF(2):", "SAT" if sat else "UNSAT", flush=True)
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
        np.save(f"gf2cms_witness_n{n}_d{d}.npy", Wv)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
