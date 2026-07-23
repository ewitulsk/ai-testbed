"""Certified rerun: dump DIMACS CNF + solve with proof logging (DRAT).

Usage:
  python3 certify.py trinary N D [arc]   -> cnf + drat for sat_trinary2 encoding
  python3 certify.py gf2 N D             -> cnf + drat for GF(2) cadical encoding
"""
import sys
from pysat.formula import CNF
from pysat.solvers import Cadical195


def run(tag, clauses, nvars):
    cnf = CNF(from_clauses=clauses)
    cnf.to_file(f"{tag}.cnf")
    print(f"{tag}: wrote {tag}.cnf ({nvars} vars, {len(clauses)} clauses)", flush=True)
    with Cadical195(bootstrap_with=clauses, with_proof=True) as sol:
        sat = sol.solve()
        print(f"{tag}:", "SAT" if sat else "UNSAT", flush=True)
        if not sat:
            proof = sol.get_proof()
            with open(f"{tag}.drat", "w") as f:
                f.write("\n".join(proof) + "\n")
            print(f"{tag}: DRAT proof written, {len(proof)} lines", flush=True)
        return sat


if __name__ == "__main__":
    mode, n, d = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    if mode == "trinary":
        import sat_trinary2 as st
        g, e, nvar, svar = st.build(n, d)
        tag = f"trinary_n{n}d{d}"
        if len(sys.argv) > 4 and sys.argv[4] == "arc":
            st.add_arc_lemma(n, d, g, e, nvar)
            tag += "_arc"
        run(tag, e.clauses, e.pool.top)
    elif mode == "gf2":
        # rebuild gf2 clauses (cadical encoding from sat_gf)
        import sat_gf
        # reproduce build portion of solve_gf2 without solving
        from pysat.formula import IDPool
        from mqg import MQG
        g = MQG(n, d); g.build_index_tables()
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
            x = tlits[0]
            for k in range(1, len(tlits)):
                y = pool.id(("xor", ci, k))
                b = tlits[k]
                clauses += [[-y, x, b], [-y, -x, -b], [y, -x, b], [y, x, -b]]
                x = y
            clauses.append([x] if t == 1 else [-x])
        run(f"gf2_n{n}d{d}", clauses, pool.top)
