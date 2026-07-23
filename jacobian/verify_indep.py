from fractions import Fraction
from itertools import product

# Polynomials as dict: (i,j,k) exponents of x,y,z -> Fraction coeff
def pmul(a, b):
    r = {}
    for ka, va in a.items():
        for kb, vb in b.items():
            k = (ka[0]+kb[0], ka[1]+kb[1], ka[2]+kb[2])
            r[k] = r.get(k, Fraction(0)) + va*vb
    return {k: v for k, v in r.items() if v != 0}

def padd(*ps):
    r = {}
    for p in ps:
        for k, v in p.items():
            r[k] = r.get(k, Fraction(0)) + v
    return {k: v for k, v in r.items() if v != 0}

def pscale(a, c):
    return {k: v*Fraction(c) for k, v in a.items() if v*Fraction(c) != 0}

def mono(c, i, j, k):
    return {(i,j,k): Fraction(c)}

def ppow(a, n):
    r = mono(1,0,0,0)
    for _ in range(n):
        r = pmul(r, a)
    return r

def pdiff(a, var):  # var 0=x,1=y,2=z
    r = {}
    for k, v in a.items():
        e = k[var]
        if e > 0:
            nk = list(k); nk[var] = e-1
            r[tuple(nk)] = r.get(tuple(nk), Fraction(0)) + v*e
    return {k: v for k, v in r.items() if v != 0}

u = padd(mono(1,0,0,0), mono(1,1,1,0))  # 1+xy
# P = u^3 z + y^2 u (4+3xy)
P = padd(pmul(ppow(u,3), mono(1,0,0,1)),
         pmul(pmul(mono(1,0,2,0), u), padd(mono(4,0,0,0), mono(3,1,1,0))))
# Q = y + 3x u^2 z + 3x y^2 (4+3xy)
Q = padd(mono(1,0,1,0),
         pmul(pmul(mono(3,1,0,1), ppow(u,2)), mono(1,0,0,0)),
         pmul(mono(3,1,2,0), padd(mono(4,0,0,0), mono(3,1,1,0))))
# R = 2x - 3x^2 y - x^3 z
R = padd(mono(2,1,0,0), mono(-3,2,1,0), mono(-1,3,0,1))

J = [[pdiff(F, v) for v in range(3)] for F in (P, Q, R)]
# det via cofactor expansion
def det3(m):
    t1 = pmul(m[0][0], padd(pmul(m[1][1], m[2][2]), pscale(pmul(m[1][2], m[2][1]), -1)))
    t2 = pmul(m[0][1], padd(pmul(m[1][0], m[2][2]), pscale(pmul(m[1][2], m[2][0]), -1)))
    t3 = pmul(m[0][2], padd(pmul(m[1][0], m[2][1]), pscale(pmul(m[1][1], m[2][0]), -1)))
    return padd(t1, pscale(t2, -1), t3)

D = det3(J)
print("det J (independent exact arithmetic):", D)

# numeric finite-difference check at random points
import random
def ev(p, x, y, z):
    return sum(float(v)*x**k[0]*y**k[1]*z**k[2] for k, v in p.items())
def F(x, y, z):
    return (ev(P,x,y,z), ev(Q,x,y,z), ev(R,x,y,z))
random.seed(1)
for _ in range(3):
    x0, y0, z0 = [random.uniform(-2,2) for _ in range(3)]
    h = 1e-6
    import numpy as np
    Jn = np.zeros((3,3))
    for j, dp in enumerate([(h,0,0),(0,h,0),(0,0,h)]):
        fp = F(x0+dp[0], y0+dp[1], z0+dp[2]); fm = F(x0-dp[0], y0-dp[1], z0-dp[2])
        for i in range(3):
            Jn[i,j] = (fp[i]-fm[i])/(2*h)
    print(f"numeric det at ({x0:.3f},{y0:.3f},{z0:.3f}) =", np.linalg.det(Jn))
