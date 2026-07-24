"""SAT encoding of the monochromatic quantum graph system over any finite
field GF(q), q = p^k.

Field elements are integers 0..q-1 whose base-p digits are the coefficients
of a polynomial over GF(p) (little-endian: element x represents
sum_j digit_j(x) * X^j).  Arithmetic is polynomial arithmetic modulo a monic
irreducible polynomial of degree k, found by brute force and certified
irreducible by trial division.  The resulting addition/multiplication tables
are verified against ALL field axioms exhaustively before use.

Integer 0 is the field zero, integer 1 the field one, so the MQG targets
(0 / 1) need no translation.

Soundness use:
  * an integer-weight MQG solution reduces mod p to a GF(p) solution, so
    UNSAT over GF(p) (k=1) proves the integer case;
  * a C-solution spreads out to solutions over \bar{F_p} for almost all p,
    so UNSAT over GF(p^k) for many (p, k) is evidence toward the C case.

Usage:
  python3 sat_gfq.py validate                 # field axioms + SAT calibration
  python3 sat_gfq.py dump  q n d out.cnf      # build + dump DIMACS, no solve
  python3 sat_gfq.py solve q n d              # build + solve with Cadical
"""
import sys
import numpy as np
from pysat.formula import IDPool
from mqg import MQG


# ---------------------------------------------------------------- field maker

def factor_prime_power(q):
    """Return (p, k) with q = p^k, p prime. Raises if q is not a prime power."""
    for p in range(2, q + 1):
        if q % p == 0:
            k, m = 0, q
            while m % p == 0:
                m //= p
                k += 1
            if m != 1:
                raise ValueError(f"{q} is not a prime power")
            # p is the smallest divisor > 1, hence prime
            return p, k
    raise ValueError(f"bad q={q}")


def poly_divmod(a, b, p):
    """Polynomial division a = q*b + r over GF(p); polys are little-endian
    coefficient lists. Returns (quot, rem) with trailing zeros stripped."""
    a = list(a)
    db, inv_lead = len(b) - 1, pow(b[-1], p - 2, p) if p > 2 else b[-1]
    quot = [0] * max(len(a) - db, 1)
    while len(a) - 1 >= db and any(a):
        da = len(a) - 1
        c = (a[-1] * inv_lead) % p
        quot[da - db] = c
        for i in range(db + 1):
            a[da - db + i] = (a[da - db + i] - c * b[i]) % p
        while len(a) > 1 and a[-1] == 0:
            a.pop()
    while len(a) > 1 and a[-1] == 0:
        a.pop()
    return quot, a


def is_irreducible(f, p):
    """Trial division by all monic polynomials of degree 1..deg(f)//2."""
    k = len(f) - 1
    for deg in range(1, k // 2 + 1):
        # enumerate monic polys of degree `deg`: coeffs c0..c_{deg-1} free
        for x in range(p ** deg):
            g = [(x // p ** j) % p for j in range(deg)] + [1]
            _, r = poly_divmod(f, g, p)
            if r == [0]:
                return False
    return True


def find_irreducible(p, k):
    """Smallest (in digit order) monic irreducible polynomial of degree k."""
    if k == 1:
        return [0, 1]  # X  (unused for arithmetic when k=1, but valid)
    for x in range(p ** k):
        f = [(x // p ** j) % p for j in range(k)] + [1]
        if f[0] == 0:
            continue  # divisible by X
        if is_irreducible(f, p):
            return f
    raise RuntimeError("no irreducible found (impossible)")


def make_field(q):
    """Return (p, k, modpoly, ADD, MUL) with q x q int tables."""
    p, k = factor_prime_power(q)
    f = find_irreducible(p, k)

    def to_poly(x):
        return [(x // p ** j) % p for j in range(k)]

    def from_poly(c):
        return sum((c[j] % p) * p ** j for j in range(min(len(c), k)))

    ADD = [[0] * q for _ in range(q)]
    MUL = [[0] * q for _ in range(q)]
    for a in range(q):
        pa = to_poly(a)
        for b in range(q):
            pb = to_poly(b)
            ADD[a][b] = from_poly([(pa[j] + pb[j]) % p for j in range(k)])
            # multiply then reduce mod f
            prod = [0] * (2 * k - 1)
            for i in range(k):
                for j in range(k):
                    prod[i + j] = (prod[i + j] + pa[i] * pb[j]) % p
            _, r = poly_divmod(prod, f, p) if k > 1 else (None, [(a * b) % p])
            r = r + [0] * (k - len(r))
            MUL[a][b] = from_poly(r)
    check_field_axioms(q, ADD, MUL)
    return p, k, f, ADD, MUL


def check_field_axioms(q, ADD, MUL):
    """Exhaustively verify all field axioms. Raises AssertionError on failure."""
    R = range(q)
    for a in R:
        assert ADD[a][0] == a and ADD[0][a] == a, "additive identity"
        assert MUL[a][1] == a and MUL[1][a] == a, "multiplicative identity"
        assert MUL[a][0] == 0 and MUL[0][a] == 0, "mult by zero"
        assert any(ADD[a][b] == 0 for b in R), "additive inverse"
        if a != 0:
            assert any(MUL[a][b] == 1 for b in R), "multiplicative inverse"
        for b in R:
            assert ADD[a][b] == ADD[b][a], "add commutativity"
            assert MUL[a][b] == MUL[b][a], "mul commutativity"
            if a != 0 and b != 0:
                assert MUL[a][b] != 0, "zero divisors"
            for c in R:
                assert ADD[ADD[a][b]][c] == ADD[a][ADD[b][c]], "add assoc"
                assert MUL[MUL[a][b]][c] == MUL[a][MUL[b][c]], "mul assoc"
                assert MUL[a][ADD[b][c]] == ADD[MUL[a][b]][MUL[a][c]], "distributivity"
    # additive group has exponent p, i.e. characteristic is prime: implied by
    # construction; also verify the multiplicative group is cyclic of order q-1
    # implicitly via inverses + no zero divisors (finite integral domain).


# --------------------------------------------------------------- SAT encoding

def build(n, d, q):
    """One-hot GF(q) encoding of the (n, d) MQG system.

    Returns (g, nvars, clauses, val) where val[w][v] is the DIMACS variable
    asserting entry w equals field element v."""
    p, k, f, ADD, MUL = make_field(q)
    print(f"GF({q}) = GF({p}^{k}), modulus poly (little-endian) {f}", flush=True)
    g = MQG(n, d)
    g.build_index_tables()
    pool = IDPool()
    W = g.nE * d * d
    val = [[pool.id(("v", w, v)) for v in range(q)] for w in range(W)]
    clauses = []
    for w in range(W):
        clauses.append(list(val[w]))                       # at least one value
        for a in range(q):
            for b in range(a + 1, q):
                clauses.append([-val[w][a], -val[w][b]])   # at most one value
    half = n // 2

    def onehot(tag):
        vs = [pool.id(tag + (v,)) for v in range(q)]
        clauses.append(list(vs))
        for a in range(q):
            for b in range(a + 1, q):
                clauses.append([-vs[a], -vs[b]])
        return vs

    # term products, deduplicated on the (sorted) entry-index triple —
    # sound because field multiplication is commutative
    terms = {}
    for ci in range(len(g.colorings)):
        for mi in range(len(g.pms)):
            wids = tuple(sorted(int(g.idx[ci, mi, k_]) for k_ in range(half)))
            if wids in terms:
                continue
            cur = val[wids[0]]
            for k_ in range(1, half):
                nxt = onehot(("prod", wids, k_))
                for a in range(q):
                    for b in range(q):
                        clauses.append([-cur[a], -val[wids[k_]][b], nxt[MUL[a][b]]])
                cur = nxt
            terms[wids] = cur

    # equations: running sum over the (n-1)!! matchings == target
    for ci, io in enumerate(g.colorings):
        t = 1 if all(c == io[0] for c in io) else 0
        tvals = [terms[tuple(sorted(int(g.idx[ci, mi, k_]) for k_ in range(half)))]
                 for mi in range(len(g.pms))]
        cur = tvals[0]
        for k_ in range(1, len(tvals)):
            nxt = onehot(("sum", ci, k_))
            for a in range(q):
                for b in range(q):
                    clauses.append([-cur[a], -tvals[k_][b], nxt[ADD[a][b]]])
            cur = nxt
        clauses.append([cur[t]])

    print(f"GF({q}) n={n} d={d}: {pool.top} vars, {len(clauses)} clauses", flush=True)
    return g, pool.top, clauses, val, (ADD, MUL)


def verify_witness(g, q, tables, Wv):
    """Check a value assignment against the MQG equations using the tables."""
    ADD, MUL = tables
    half = g.n // 2
    bad = 0
    for ci, io in enumerate(g.colorings):
        s = 0
        for mi in range(len(g.pms)):
            pr = 1
            for k_ in range(half):
                pr = MUL[pr][Wv[int(g.idx[ci, mi, k_])]]
            s = ADD[s][pr]
        t = 1 if all(c == io[0] for c in io) else 0
        if s != t:
            bad += 1
    return bad


def dump_dimacs(path, nvars, clauses):
    with open(path, "w") as fh:
        fh.write(f"p cnf {nvars} {len(clauses)}\n")
        fh.write("".join(" ".join(map(str, c)) + " 0\n" for c in clauses))


def solve(n, d, q, save_prefix=None):
    from pysat.solvers import Cadical195
    g, nv, clauses, val, tables = build(n, d, q)
    with Cadical195(bootstrap_with=clauses) as sol:
        sat = sol.solve()
        print(f"GF({q}) n={n} d={d}:", "SAT" if sat else "UNSAT", flush=True)
        if sat:
            model = set(l for l in sol.get_model() if l > 0)
            Wv = [next(v for v in range(q) if val[w][v] in model)
                  for w in range(len(val))]
            bad = verify_witness(g, q, tables, Wv)
            print(f"witness verified against tables, bad equations: {bad}", flush=True)
            assert bad == 0, "encoding bug: SAT model violates field equations"
            if save_prefix:
                np.save(f"{save_prefix}.npy", np.array(Wv))
        return sat


def validate():
    # field-axiom self-tests on every field we will use (+ GF(8), GF(9) as extras)
    for q in (2, 3, 4, 5, 7, 8, 9):
        make_field(q)
        print(f"GF({q}): all field axioms verified exhaustively", flush=True)
    # GF(4) must reproduce the known SAT cases
    assert solve(6, 2, 4) is True, "GF(4) (6,2) should be SAT (alternating cycle)"
    assert solve(4, 3, 4) is True, "GF(4) (4,3) should be SAT"
    # cross-check the prime-field path against sat_gf.py's known results
    assert solve(4, 3, 5) is True, "GF(5) (4,3) should be SAT"
    print("validation complete: field axioms + SAT calibration all pass", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "validate":
        validate()
    elif sys.argv[1] == "dump":
        q, n, d, out = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
        g, nv, clauses, val, tables = build(n, d, q)
        dump_dimacs(out, nv, clauses)
        print(f"wrote {out}: {nv} vars, {len(clauses)} clauses", flush=True)
    elif sys.argv[1] == "solve":
        q, n, d = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        solve(n, d, q, save_prefix=f"gfq{q}_witness_n{sys.argv[3]}_d{sys.argv[4]}")
    else:
        raise SystemExit("usage: sat_gfq.py validate | dump q n d out.cnf | solve q n d")
