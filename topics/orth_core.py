"""⑦ 特殊関数 第2章 直交関数系:ページと演習で共通に使う計算

多項式は昇べきの係数のリスト(Fraction)。重みつき内積はモーメント m_k = ∫ x^k w(x) dx から厳密に計算する
(チェビシェフは π, エルミートは √π をくくり出したモーメントを使う)。
"""
from fractions import Fraction
from math import comb, factorial

import numpy as np

F = Fraction


# ------------------------------------------------------------
# 多項式(昇べきの係数)
# ------------------------------------------------------------
def p_add(a, b):
    n = max(len(a), len(b))
    return trim([(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) for i in range(n)])


def p_scale(a, c):
    return trim([c * x for x in a])


def p_mulx(a):
    return [F(0)] + list(a)


def p_mul(a, b):
    out = [F(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return trim(out)


def p_der(a):
    return trim([i * a[i] for i in range(1, len(a))]) or [F(0)]


def p_val(a, x):
    v = F(0)
    for c in reversed(a):
        v = v * x + c
    return v


def trim(a):
    a = [F(x) for x in a]
    while len(a) > 1 and a[-1] == 0:
        a.pop()
    return a or [F(0)]


def p_tex(a, var="x"):
    """昇べきの係数 → TeX(高次から)"""
    out = ""
    for k in range(len(a) - 1, -1, -1):
        c = F(a[k])
        if c == 0:
            continue
        body = "" if k == 0 else (var if k == 1 else f"{var}^{{{k}}}")
        m = abs(c)
        if m == 1 and body:
            coef = ""
        elif m.denominator == 1:
            coef = str(m.numerator)
        else:
            coef = rf"\frac{{{m.numerator}}}{{{m.denominator}}}"
        sgn = "-" if c < 0 else ("+" if out else "")
        out += sgn + coef + body
    return out or "0"


# ------------------------------------------------------------
# 4つの古典直交多項式(厳密な係数)
# ------------------------------------------------------------
def legendre(n):
    """(n+1)P_{n+1} = (2n+1)x P_n − n P_{n−1}"""
    p0, p1 = [F(1)], [F(0), F(1)]
    if n == 0:
        return p0
    for k in range(1, n):
        p0, p1 = p1, p_scale(p_add(p_scale(p_mulx(p1), 2 * k + 1), p_scale(p0, -k)), F(1, k + 1))
    return p1


def chebyshev(n):
    """T_{n+1} = 2x T_n − T_{n−1}"""
    p0, p1 = [F(1)], [F(0), F(1)]
    if n == 0:
        return p0
    for _ in range(1, n):
        p0, p1 = p1, p_add(p_scale(p_mulx(p1), 2), p_scale(p0, -1))
    return p1


def hermite(n):
    """H_{n+1} = 2x H_n − 2n H_{n−1}(物理学の規約)"""
    p0, p1 = [F(1)], [F(0), F(2)]
    if n == 0:
        return p0
    for k in range(1, n):
        p0, p1 = p1, p_add(p_scale(p_mulx(p1), 2), p_scale(p0, -2 * k))
    return p1


def laguerre(n):
    """L_n(x) = Σ_k C(n,k) (−1)^k x^k / k!"""
    return [F((-1) ** k * comb(n, k), factorial(k)) for k in range(n + 1)]


POLY = {"legendre": legendre, "chebyshev": chebyshev, "hermite": hermite, "laguerre": laguerre}


# ------------------------------------------------------------
# 重み(モーメント)
# ------------------------------------------------------------
def dfact(n):
    """二重階乗(n ≤ 0 では 1)"""
    r = 1
    while n > 1:
        r *= n
        n -= 2
    return r


# name: (表示名, 区間 TeX, 重み TeX, モーメント m_k(定数を除く), くくり出した定数 TeX)
WEIGHTS = {
    "legendre": ("ルジャンドル", r"[-1,\,1]", "1", lambda k: F(2, k + 1) if k % 2 == 0 else F(0), ""),
    "chebyshev": ("チェビシェフ", r"[-1,\,1]", r"\dfrac{1}{\sqrt{1-x^2}}", lambda k: F(dfact(k - 1), dfact(k)) if k % 2 == 0 else F(0), r"\pi"),
    "hermite": ("エルミート", r"(-\infty,\,\infty)", r"e^{-x^2}", lambda k: F(dfact(k - 1), 2 ** (k // 2)) if k % 2 == 0 else F(0), r"\sqrt{\pi}"),
    "laguerre": ("ラゲール", r"[0,\,\infty)", r"e^{-x}", lambda k: F(factorial(k)), ""),
    "unit01": ("区間 [0, 1], 重み 1", r"[0,\,1]", "1", lambda k: F(1, k + 1), ""),
    "x01": ("区間 [0, 1], 重み x", r"[0,\,1]", "x", lambda k: F(1, k + 2), ""),
}


def inner(a, b, wname):
    """(a, b)_w をモーメントで厳密に(定数 WEIGHTS[..][4] を除く)"""
    m = WEIGHTS[wname][3]
    return sum((a[i] * b[j] * m(i + j) for i in range(len(a)) for j in range(len(b))), F(0))


def gram_schmidt(n, wname):
    """1, x, …, x^n を直交化した最高次の係数 1 の多項式 [p_0, …, p_n]"""
    ps = []
    for k in range(n + 1):
        p = [F(0)] * k + [F(1)]
        for q in ps:
            p = p_add(p, p_scale(q, -inner(p, q, wname) / inner(q, q, wname)))
        ps.append(p)
    return ps


def norm2(fam, n):
    """標準の規格化での (p_n, p_n)_w(定数を除く):ルジャンドル 2/(2n+1), チェビシェフ π(n=0)・π/2, エルミート √π 2^n n!, ラゲール 1"""
    return {"legendre": F(2, 2 * n + 1), "chebyshev": F(1) if n == 0 else F(1, 2),
            "hermite": F(2 ** n * factorial(n)), "laguerre": F(1)}[fam]


def expand_coef(f_poly, fam, n):
    """多項式 f を p_0, p_1, … で展開したときの p_n の係数 (f, p_n)_w / (p_n, p_n)_w"""
    return inner(f_poly, POLY[fam](n), fam) / norm2(fam, n)


def expansion(f_poly, fam):
    """多項式 f = Σ c_n p_n の係数(n ≤ deg f)"""
    return [expand_coef(f_poly, fam, n) for n in range(len(f_poly))]


# ------------------------------------------------------------
# 数値(図)
# ------------------------------------------------------------
def p_eval(fam, n, x):
    c = np.zeros(n + 1)
    c[n] = 1.0
    P = np.polynomial
    return {"legendre": P.legendre.legval, "chebyshev": P.chebyshev.chebval,
            "hermite": P.hermite.hermval, "laguerre": P.laguerre.lagval}[fam](x, c)


def hermite_functions(nmax, x):
    """正規直交なエルミート関数 φ_n = H_n e^{−x²/2} / √(√π 2^n n!)(漸化式で桁あふれを避ける)"""
    out = np.zeros((nmax + 1, len(x)))
    out[0] = np.pi ** -0.25 * np.exp(-x ** 2 / 2)
    if nmax >= 1:
        out[1] = np.sqrt(2.0) * x * out[0]
    for n in range(1, nmax):
        out[n + 1] = np.sqrt(2.0 / (n + 1)) * x * out[n] - np.sqrt(n / (n + 1)) * out[n - 1]
    return out


def laguerre_functions(nmax, x):
    """正規直交なラゲール関数 ℓ_n = e^{−x/2} L_n(x)"""
    out = np.zeros((nmax + 1, len(x)))
    L0, L1 = np.ones_like(x), 1 - x
    out[0] = L0
    if nmax >= 1:
        out[1] = L1
    for n in range(1, nmax):
        L0, L1 = L1, ((2 * n + 1 - x) * L1 - n * L0) / (n + 1)
        out[n + 1] = L1
    return out * np.exp(-x / 2)


# ------------------------------------------------------------
# ウォルシュ関数
# ------------------------------------------------------------
def hadamard(N):
    """シルベスター型のアダマール行列(N = 2^m)"""
    H = np.array([[1]])
    while H.shape[0] < N:
        H = np.block([[H, H], [H, -H]])
    return H


def sign_changes(row):
    return int(np.sum(row[1:] != row[:-1]))


def walsh_matrix(N):
    """符号の変わる回数(交番数)の順に並べたウォルシュ関数(各行が [0,1) を N 等分した値)"""
    H = hadamard(N)
    return H[np.argsort([sign_changes(r) for r in H], kind="stable")]


def walsh_coefs(values):
    """N 等分した階段関数の値 → ウォルシュ係数 c_n = ∫_0^1 f w_n dx = (1/N) Σ f_k w_n(k)(厳密)"""
    N = len(values)
    W = walsh_matrix(N)
    return [sum((int(W[n, k]) * F(values[k]) for k in range(N)), F(0)) / N for n in range(N)]
