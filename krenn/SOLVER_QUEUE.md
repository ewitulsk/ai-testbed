# Solver queue — dumped CNFs for the (6,D) monochromatic quantum graph attack

All CNFs live in the session scratchpad:
`/tmp/claude-0/-home-user-ai-testbed/e8d6bc0e-c2d6-5089-9811-4145f3a745b6/scratchpad/`
Kissat binary: `<scratchpad>/kissat/build/kissat` (4.0.4).

Generators (reproducible):
- `sat_gfq.py` — GF(q) one-hot encoding for any prime power q; the field
  add/mul tables are built from an irreducible polynomial and **all field
  axioms are verified exhaustively** at build time; encoding calibrated SAT
  on GF(4)/(6,2), GF(4)/(4,3), GF(5)/(4,3) with witnesses re-verified.
  Rebuild: `python3 sat_gfq.py dump <q> 6 3 <out.cnf>`.
- `dump_trinary.py` — plain trinary {-1,0,1} encoding (`sat_trinary2.build`)
  plus sign hygiene `(n_w OR NOT s_w)`; **no arc-lemma clauses** (the rank-1
  arc lemma is only proven for the D=3 argument, so D>=4 instances must stay
  lemma-free). Rebuild: `python3 dump_trinary.py 6 <d> <out.cnf>`.

Each entry got a 60 s kissat probe on 2026-07-24; all returned `s UNKNOWN`
(no instance is easy). Do NOT re-run GF(2)/GF(3) for (6,3) — already running.

| # | CNF | vars | clauses | file size | 60s probe |
|---|-----|------|---------|-----------|-----------|
| 1 | `gfq_4_n6d3.cnf` | 128,844 | 739,422 | 14 MB | UNKNOWN |
| 2 | `gfq_5_n6d3.cnf` | 161,055 | 1,156,950 | 22 MB | UNKNOWN |
| 3 | `gfq_7_n6d3.cnf` | 225,477 | 2,281,095 | 44 MB | UNKNOWN |
| 4 | `trinary_n6d4.cnf` | 1,290,720 | 4,915,440 | 100 MB | UNKNOWN |
| 5 | `trinary_n6d5.cnf` | 4,922,625 | 18,750,375 | 409 MB | UNKNOWN |

## What each result would mean

### 1. `gfq_4_n6d3.cnf` — (6,3) over GF(4) = GF(2²), modulus X²+X+1
- **UNSAT**: no (6,3) solution over GF(4). Since GF(2) ⊂ GF(4), this *also*
  implies UNSAT over GF(2) and hence (by reduction mod 2 of any integer
  solution) **proves the open Lean conjecture `eqSystem6_no_solution_d3_int`**.
  Additionally it is spreading-out evidence toward the ℂ conjecture: a
  ℂ-solution yields solutions over the algebraic closure of F_p for almost
  all p, and GF(4) rules out all degree-≤2 elements over F_2.
- **SAT**: a genuinely non-prime-field solution exists; implies nothing for
  ℤ or ℂ directly, but the witness structure would be highly informative
  (first known finite-field (6,3) solution, if GF(2)/GF(3) come back UNSAT).

### 2. `gfq_5_n6d3.cnf` — (6,3) over GF(5)
- **UNSAT**: any integer solution reduces mod 5 to a GF(5) solution, so this
  alone **proves `eqSystem6_no_solution_d3_int`** (one prime suffices).
  Also spreading-out evidence toward ℂ (rules out F_5-rational solutions).
- **SAT**: integers not ruled out via p=5; witness worth analyzing for
  structure that might lift.

### 3. `gfq_7_n6d3.cnf` — (6,3) over GF(7)
- Same logic as GF(5) with p=7: **UNSAT alone proves
  `eqSystem6_no_solution_d3_int`**; multiple prime UNSATs together
  strengthen the spreading-out evidence toward the ℂ conjecture.

### 4. `trinary_n6d4.cnf` — (6,4) with weights in {-1,0,1} (plain + sign hygiene)
- **UNSAT**: no monochromatic quantum graph on K6 with D=4 and trinary
  weights — settles the trinary integer restriction of the open d=4 case
  (analog of `eqSystem6_no_solution_d3_trinary_int` at d=4). Self-contained:
  no arc-lemma or WLOG dependency, so the DRAT certificate stands alone.
- **SAT**: a trinary witness IS an exact ℂ solution — this would
  **disprove Krenn's (6,D≥3) conjecture outright** (€3000). Verify any
  model with `mqg.py` residuals before believing it.

### 5. `trinary_n6d5.cnf` — (6,5) with weights in {-1,0,1} (plain + sign hygiene)
- Same as #4 at d=5: **UNSAT** settles the trinary d=5 case (self-contained
  certificate); **SAT** would be a counterexample to the full conjecture.
- Warning: 4.9 M vars / 18.8 M clauses — expect multi-GB DRAT; consider
  running without proof first, and only re-run with proof logging if UNSAT.

## Suggested commands

```sh
SP=/tmp/claude-0/-home-user-ai-testbed/e8d6bc0e-c2d6-5089-9811-4145f3a745b6/scratchpad
K=$SP/kissat/build/kissat

# priority order: 2, 3, 1 (any single prime UNSAT closes the integer case;
# GF(4) additionally needs GF(2)-SAT context to interpret), then 4, then 5.
nice -n 15 $K -q $SP/gfq_5_n6d3.cnf   $SP/gfq_5_n6d3.drat   > $SP/log_kissat_gfq5.txt
nice -n 15 $K -q $SP/gfq_7_n6d3.cnf   $SP/gfq_7_n6d3.drat   > $SP/log_kissat_gfq7.txt
nice -n 15 $K -q $SP/gfq_4_n6d3.cnf   $SP/gfq_4_n6d3.drat   > $SP/log_kissat_gfq4.txt
nice -n 15 $K -q $SP/trinary_n6d4.cnf $SP/trinary_n6d4.drat > $SP/log_kissat_t64.txt
nice -n 15 $K -q $SP/trinary_n6d5.cnf                        > $SP/log_kissat_t65.txt

# after any UNSAT: verify the certificate
# drat-trim <cnf> <drat>   (expect "s VERIFIED"), then gzip + sha256 both files
```

Scheduling note: only 4 CPUs; 3 solvers (GF(2), GF(3), trinary-plain d=3) may
still be running — check with `ps` before launching more than one of these.
