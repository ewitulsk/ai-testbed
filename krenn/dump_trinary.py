"""Dump DIMACS for the trinary {-1,0,1} MQG system, PLAIN encoding:
sat_trinary2.build() only (the rank-1 arc lemma is NOT proven for D>=4, so no
arc clauses), plus sign hygiene (zero entry => sign bit 0):
    (n_w OR NOT s_w) for every weight entry w.

Usage: python3 dump_trinary.py N D out.cnf
"""
import sys
from sat_trinary2 import build

n, d, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
g, e, nvar, svar = build(n, d)
W = g.nE * d * d
for w in range(W):
    e.clauses.append([nvar[w], -svar[w]])   # sign hygiene
nv = e.pool.top
print(f"trinary n={n} d={d} plain+hygiene: {nv} vars, {len(e.clauses)} clauses",
      flush=True)
with open(out, "w") as fh:
    fh.write(f"p cnf {nv} {len(e.clauses)}\n")
    for c in e.clauses:
        fh.write(" ".join(map(str, c)) + " 0\n")
print(f"wrote {out}", flush=True)
