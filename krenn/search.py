"""Multistart Levenberg-Marquardt search for monochromatic quantum graphs.

Usage: python3 search.py N D n_starts seed [out.jsonl]
Minimizes sum |pmSum(iota) - delta|^2 over complex weights.
"""
import sys, json, time
import numpy as np
from scipy.optimize import least_squares
from mqg import MQG


def make_funcs(g):
    nv = g.nE * g.d * g.d

    def resid_real(x):
        W = x[:nv] + 1j * x[nv:]
        r = g.residuals_fast(W)
        return np.concatenate([r.real, r.imag])

    def jac_real(x):
        W = x[:nv] + 1j * x[nv:]
        J = g.jacobian_fast(W)
        return np.block([[J.real, -J.imag], [J.imag, J.real]])

    return resid_real, jac_real, nv


def run(n, d, n_starts, seed, out_path=None):
    g = MQG(n, d)
    g.build_index_tables()
    resid, jac, nv = make_funcs(g)
    rng = np.random.default_rng(seed)
    best = np.inf
    results = []
    t0 = time.time()
    fout = open(out_path, "a") if out_path else None
    for s in range(n_starts):
        scale = rng.choice([0.3, 0.7, 1.0, 1.5])
        x0 = rng.normal(scale=scale, size=2 * nv)
        # occasionally sparse starts (many exact zeros) to bias toward structured solutions
        if s % 3 == 1:
            mask = rng.random(2 * nv) < 0.7
            x0[mask] = 0.0
        try:
            sol = least_squares(resid, x0, jac=jac, method="lm", xtol=1e-14, ftol=1e-14,
                                gtol=1e-14, max_nfev=400)
        except Exception as ex:
            continue
        f = 0.5 * np.sum(sol.fun ** 2)
        results.append(f)
        if f < best:
            best = f
            if fout:
                rec = {"n": n, "d": d, "start": s, "cost": float(f),
                       "x": sol.x.tolist() if f < 1e-3 else None}
                fout.write(json.dumps(rec) + "\n"); fout.flush()
        if f < 1e-16:
            print(f"SOLUTION FOUND at start {s}: cost={f:.3e}")
            np.save(f"solution_n{n}_d{d}_seed{seed}_{s}.npy", sol.x)
            break
        if (s + 1) % 10 == 0:
            arr = np.array(results)
            print(f"[n={n} d={d} seed={seed}] {s+1}/{n_starts} starts, "
                  f"best={best:.6f}, median={np.median(arr):.4f}, t={time.time()-t0:.0f}s",
                  flush=True)
    arr = np.array(results)
    print(f"DONE n={n} d={d} seed={seed}: starts={len(arr)}, best={best:.8f}, "
          f"quartiles={np.percentile(arr, [0, 25, 50, 75, 100]).round(4).tolist()}")
    if fout:
        fout.close()
    return best


if __name__ == "__main__":
    n, d, ns, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    out = sys.argv[5] if len(sys.argv) > 5 else None
    run(n, d, ns, seed, out)
