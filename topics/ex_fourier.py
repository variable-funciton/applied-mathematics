"""数学IV フーリエ解析 第1章 フーリエ級数:演習の問題生成(実数形・複素形)

規約(アプリの本文と同じ):周期 2L の f について
    f(x) ~ a_0/2 + Σ (a_n cos(nπx/L) + b_n sin(nπx/L)),  a_n = (1/L)∫_{-L}^{L} f cos(nπx/L) dx,  b_n も同様
    パーセバルの等式(L = π):(1/π)∫_{-π}^{π} f² dx = a_0²/2 + Σ (a_n² + b_n²)
型:
  基本:周期 2π のフーリエ級数 / 収束定理で級数が収束する値
  標準:一般周期のフーリエ係数 / 収束定理による級数の和 / パーセバルの等式(係数の2乗和)/ パーセバルの等式による級数の和
関数は c x, c|x|, c x², 矩形波, 半分だけの x(のこぎりの半分)。係数はすべて「整数 × (−1)^n などの形 / (n^p π^q)」で出る。
"""
import math
from fractions import Fraction

from topics.ex_core import Choice, ExamItem, Part, Problem, ProblemType, Retry, render, tex_fr

F = Fraction
PI = math.pi
CONV = (r"$f(x)\sim\dfrac{a_0}{2}+\displaystyle\sum_{n=1}^{\infty}\left(a_n\cos\frac{n\pi x}{L}+b_n\sin\frac{n\pi x}{L}\right)$, "
        r"$a_n=\dfrac1L\displaystyle\int_{-L}^{L}f(x)\cos\frac{n\pi x}{L}dx$, $b_n=\dfrac1L\displaystyle\int_{-L}^{L}f(x)\sin\frac{n\pi x}{L}dx$")
CONV_PI = (r"$f(x)\sim\dfrac{a_0}{2}+\displaystyle\sum_{n=1}^{\infty}(a_n\cos nx+b_n\sin nx)$, "
           r"$a_n=\dfrac1\pi\displaystyle\int_{-\pi}^{\pi}f(x)\cos nx\,dx$, $b_n=\dfrac1\pi\displaystyle\int_{-\pi}^{\pi}f(x)\sin nx\,dx$")


# ------------------------------------------------------------
# π を含む数:{π の指数: 有理数} の和
# ------------------------------------------------------------
class PiNum:
    def __init__(self, terms=None):
        self.t = {k: F(v) for k, v in (terms or {}).items() if v != 0}

    def __add__(self, o):
        t = dict(self.t)
        for k, v in o.t.items():
            t[k] = t.get(k, 0) + v
        return PiNum(t)

    def __sub__(self, o):
        return self + o.scale(-1)

    def scale(self, c):
        return PiNum({k: v * c for k, v in self.t.items()})

    def __mul__(self, o):
        t = {}
        for k1, v1 in self.t.items():
            for k2, v2 in o.t.items():
                t[k1 + k2] = t.get(k1 + k2, 0) + v1 * v2
        return PiNum(t)

    def __float__(self):
        return float(sum(float(v) * PI ** k for k, v in self.t.items()))

    def key(self):
        return tuple(sorted(self.t.items()))

    def tex(self):
        if not self.t:
            return "0"
        out = ""
        for k in sorted(self.t, reverse=True):
            v = self.t[k]
            sgn = "-" if v < 0 else ("+" if out else "")
            a = abs(v)
            pi = "" if k == 0 else (r"\pi" if k == 1 else rf"\pi^{{{k}}}")
            if not pi:
                body = tex_fr(a)
            elif a.denominator == 1:
                body = ("" if a == 1 else str(a.numerator)) + pi
            else:
                body = rf"\frac{{{'' if a.numerator == 1 else a.numerator}{pi}}}{{{a.denominator}}}"
            out += sgn + body
        return out


def P(c=0, k=0):
    """c π^k"""
    return PiNum({k: c})


# ------------------------------------------------------------
# 係数の式 R·L^m·pat(n) / (n^p π^q)
# ------------------------------------------------------------
PATS = {"alt+": (r"(-1)^{n+1}", lambda n: (-1) ** (n + 1)), "alt": (r"(-1)^n", lambda n: (-1) ** n),
        "odd": (r"\{1-(-1)^n\}", lambda n: 1 - (-1) ** n), "oddm": (r"\{(-1)^n-1\}", lambda n: (-1) ** n - 1)}
FLIP = {"alt+": "alt", "alt": "alt+", "odd": "oddm", "oddm": "odd"}


class Coef:
    def __init__(self, R, pat, p, q):
        self.R, self.pat, self.p, self.q = F(R), pat, p, q   # q は π の指数(負なら分子に π)
        if self.R < 0:                                       # 符号は (-1)^n の側に入れる(Σ の直後に − が来ないように)
            self.R, self.pat = -self.R, FLIP[pat]

    def __call__(self, n):
        return float(self.R) * PATS[self.pat][1](n) / (n ** self.p * PI ** self.q)

    def key(self):
        return (self.R, self.pat, self.p, self.q)

    def tex(self):
        if self.R == 0:
            return "0"
        sgn = "-" if self.R < 0 else ""
        a = abs(self.R)
        num = ("" if a.numerator == 1 else str(a.numerator)) + PATS[self.pat][0]
        if self.q < 0:
            num += r"\pi" if self.q == -1 else rf"\pi^{{{-self.q}}}"
        den = ("" if a.denominator == 1 else str(a.denominator)) + ("n" if self.p == 1 else f"n^{{{self.p}}}")
        if self.q > 0:
            den += r"\pi" if self.q == 1 else rf"\pi^{{{self.q}}}"
        return sgn + rf"\frac{{{num}}}{{{den}}}"


def with_L(R, m, p, q, pat, L):
    """係数 R L^m pat/(n^p π^q) を、L(整数か "pi")に合わせて Coef にする"""
    if L == "pi":
        return Coef(R, pat, p, q - m)
    return Coef(F(R) * L ** m, pat, p, q)


def Lval(L):
    return P(1, 1) if L == "pi" else P(L)


def Ltex(L):
    return r"\pi" if L == "pi" else str(L)


def arg_tex(L):
    """cos の中身:nx(L = π)、nπx(L = 1)、\\frac{nπx}{L}"""
    if L == "pi":
        return "nx"
    if L == 1:
        return r"n\pi x"
    return rf"\frac{{n\pi x}}{{{L}}}"


# ------------------------------------------------------------
# 関数の族
# ------------------------------------------------------------
def coef_term(c, s):
    return ("" if c == 1 else ("-" if c == -1 else str(c))) + s


def family(kind, L, c=1, A=None, B=None):
    """戻り値 dict(tex, f, a0(PiNum), an(Coef or None), bn(Coef or None), left/right(x0) など)"""
    Lt = Ltex(L)
    Lf = PI if L == "pi" else float(L)
    dom = rf"(-{Lt}<x\le {Lt})"
    if kind == "x":
        return dict(kind=kind, tex=rf"f(x)={coef_term(c, 'x')}\ {dom}", f=lambda x: c * x, a0=P(),
                    an=None, bn=with_L(2 * c, 1, 1, 1, "alt+", L), c=c, L=L,
                    sq=Lval(L) * Lval(L).scale(F(2 * c * c, 3)) if L != "pi" else P(F(2 * c * c, 3), 2))
    if kind == "abs":
        return dict(kind=kind, tex=rf"f(x)={coef_term(c, '|x|')}\ {dom}", f=lambda x: c * abs(x), a0=Lval(L).scale(c),
                    an=with_L(2 * c, 1, 2, 2, "oddm", L), bn=None, c=c, L=L,
                    sq=P(F(2 * c * c, 3), 2) if L == "pi" else P(F(2 * c * c * L * L, 3)))
    if kind == "sq":
        return dict(kind=kind, tex=rf"f(x)={coef_term(c, 'x^2')}\ {dom}", f=lambda x: c * x * x,
                    a0=(P(F(2 * c, 3), 2) if L == "pi" else P(F(2 * c * L * L, 3))),
                    an=with_L(4 * c, 2, 2, 2, "alt", L), bn=None, c=c, L=L,
                    sq=P(F(2 * c * c, 5), 4) if L == "pi" else P(F(2 * c * c * L ** 4, 5)))
    if kind == "step":
        tex = (rf"f(x)=\begin{{cases}}{B}&(-{Lt}<x<0)\\{A}&(0\le x\le {Lt})\end{{cases}}")
        return dict(kind=kind, tex=tex, f=lambda x: A if x >= 0 else B, a0=P(A + B), an=None,
                    bn=Coef(A - B, "odd", 1, 1), A=A, B=B, L=L, sq=P(A * A + B * B))
    if kind == "half":
        tex = (rf"f(x)=\begin{{cases}}0&(-{Lt}<x<0)\\{coef_term(c, 'x')}&(0\le x\le {Lt})\end{{cases}}")
        return dict(kind=kind, tex=tex, f=lambda x: c * x if x >= 0 else 0.0, a0=Lval(L).scale(F(c, 2)),
                    an=with_L(c, 1, 2, 2, "oddm", L), bn=with_L(c, 1, 1, 1, "alt+", L), c=c, L=L,
                    sq=P(F(c * c, 3), 2) if L == "pi" else P(F(c * c * L * L, 3)))
    raise ValueError(kind)


def series_tex(fam, an=None, bn=None, a0h=None, swap=False, L=None):
    """フーリエ級数の式。swap=True なら cos と sin を入れかえた誤答。L を指定すると cos の中身だけ変える(誤答用)"""
    L = fam["L"] if L is None else L
    an = fam["an"] if an is None else an
    bn = fam["bn"] if bn is None else bn
    a0h = fam["a0"].scale(F(1, 2)) if a0h is None else a0h
    cs, sn = (r"\sin ", r"\cos ") if swap else (r"\cos ", r"\sin ")
    arg = arg_tex(L)
    parts = []
    if an is not None and an.R != 0:
        parts.append(an.tex() + cs + arg)
    if bn is not None and bn.R != 0:
        parts.append(bn.tex() + sn + arg)
    body = "+".join(parts).replace("+-", "-")
    head = "" if not a0h.t else a0h.tex() + "+"
    return head + r"\sum_{n=1}^{\infty}\left(" + body + r"\right)" if len(parts) > 1 else \
        head + r"\sum_{n=1}^{\infty}" + body


def fam_text(fam):
    L = fam["L"]
    per = r"2\pi" if L == "pi" else str(2 * L)
    return rf"${fam['tex']}$, $f(x+{per})=f(x)$"


def pick_family(rng, kinds, L):
    kind = rng.choice(kinds)
    c = rng.choice([1, 1, 2, 3, -1, -2])
    if kind == "step":
        A, B = rng.sample([-2, -1, 0, 1, 2, 3], 2)
        return family("step", L, A=A, B=B)
    return family(kind, L, c=c)


# ============================================================
# 基本:周期 2π のフーリエ級数
# ============================================================
def coef_solution(fam):
    L = fam["L"]
    out = []
    k = fam["kind"]
    if k in ("x",):
        out.append("$f$ は奇関数なので $a_n=0$($n=0,1,2,\\ldots$)。")
    if k in ("abs", "sq"):
        out.append("$f$ は偶関数なので $b_n=0$。")
    if k == "step":
        out.append(r"$a_n\ (n\ge1)$ は $\cos$ の積分が $0$ になるので $0$。")
    if fam["a0"].t or k in ("abs", "sq", "half", "step"):
        out.append(rf"$a_0={fam['a0'].tex()}$ より, 定数項は $\dfrac{{a_0}}{{2}}={fam['a0'].scale(F(1, 2)).tex()}$。")
    if fam["an"] is not None:
        out.append(rf"部分積分により $a_n={fam['an'].tex()}$。")
    if fam["bn"] is not None:
        out.append(rf"部分積分により $b_n={fam['bn'].tex()}$。")
    out.append(rf"$$f(x)\sim {series_tex(fam)}$$")
    return out


def g_series_2pi(rng):
    fam = pick_family(rng, ["x", "abs", "sq", "step"], "pi")
    an, bn = fam["an"], fam["bn"]
    ok = series_tex(fam)
    cands = []
    main = an if an is not None else bn
    flip = Coef(main.R, FLIP[main.pat], main.p, main.q)
    cands.append((series_tex(fam, an=flip if an is not None else None, bn=flip if bn is not None else None),
                  "符号の形を確かめよう:$\\cos n\\pi=(-1)^n$。"))
    cands.append((series_tex(fam, swap=True), "偶関数のフーリエ級数は $\\cos$ だけ、奇関数は $\\sin$ だけ(定数項を除く)。"))
    half = Coef(main.R / 2, main.pat, main.p, main.q)
    cands.append((series_tex(fam, an=half if an is not None else None, bn=half if bn is not None else None),
                  "係数は $\\dfrac1\\pi\\int_{-\\pi}^{\\pi}$。偶関数・奇関数なら $\\dfrac2\\pi\\int_0^{\\pi}$ になる。"))
    if fam["a0"].t:
        cands.append((series_tex(fam, a0h=fam["a0"]), "定数項は $\\dfrac{a_0}{2}$($a_0$ ではない)。"))
    pw = Coef(main.R, main.pat, 3 - main.p, main.q)
    cands.append((series_tex(fam, an=pw if an is not None else None, bn=pw if bn is not None else None),
                  "部分積分の回数を確かめよう($n$ の次数)。"))
    choices = [Choice(ok, True, value=ok)]
    for t, why in cands:
        if t != ok and all(c.tex != t for c in choices) and len(choices) < 4:
            choices.append(Choice(t, why=why, value=t))
    if len(choices) < 4:
        raise Retry
    return Problem("周期 2π のフーリエ級数", "基本",
                   rf"周期 $2\pi$ の関数 {fam_text(fam)} のフーリエ級数を求めよ。",
                   choices, hints=["規約:" + CONV_PI, r"偶関数なら $b_n=0$、奇関数なら $a_n=0$。$\cos n\pi=(-1)^n$, $\sin n\pi=0$。"],
                   solution=coef_solution(fam), meta=dict(fam=fam))


# ============================================================
# 標準:一般周期のフーリエ係数
# ============================================================
def g_coef_general(rng):
    L = rng.choice([1, 2, 3, 4])
    fam = pick_family(rng, ["x", "abs", "sq", "half"], L)
    which = "an" if fam["an"] is not None and (fam["bn"] is None or rng.random() < 0.5) else "bn"
    co = fam[which]
    name = "a_n" if which == "an" else "b_n"
    R_ = co.R / (F(L) ** {"x": 1, "abs": 1, "sq": 2, "half": 1}[fam["kind"]])     # L を除いた係数
    m = {"x": 1, "abs": 1, "sq": 2, "half": 1}[fam["kind"]]
    cands = [(Coef(R_, co.pat, co.p, co.q), f"$a_n=\\dfrac1L\\int_{{-L}}^{{L}}$ で $x=\\dfrac{{Lu}}{{\\pi}}$ と置換すると $L^{{{m}}}$ が出る。"),
             (Coef(co.R, FLIP[co.pat], co.p, co.q), "符号の形を確かめよう:$\\cos n\\pi=(-1)^n$。"),
             (Coef(co.R, co.pat, co.p, co.q - 1), "$\\dfrac{n\\pi x}{L}$ の積分で出る $\\pi$ を確かめよう。"),
             (Coef(co.R * L, co.pat, co.p, co.q), "$\\dfrac1L$ をかけ忘れていないか。"),
             (Coef(co.R / 2, co.pat, co.p, co.q), "係数は $\\dfrac1L\\int_{-L}^{L}$(偶関数・奇関数なら $\\dfrac2L\\int_0^L$)。")]
    choices = [Choice(rf"{name}={co.tex()}", True, value=co.key())]
    for c, why in cands:
        t = rf"{name}={c.tex()}"
        if c.key() != co.key() and all(x.tex != t for x in choices) and len(choices) < 4:
            choices.append(Choice(t, why=why, value=c.key()))
    if len(choices) < 4:
        raise Retry
    return Problem("一般周期のフーリエ係数", "標準",
                   rf"周期 ${2 * L}$ の関数 {fam_text(fam)} のフーリエ係数 ${name}\ (n\ge1)$ を求めよ。",
                   choices, hints=["規約:" + CONV, rf"$L={L}$。偶関数・奇関数かどうかを先に調べ, 部分積分を使う。"],
                   solution=coef_solution(fam), meta=dict(fam=fam, which=which))


# ============================================================
# 基本:収束定理で、フーリエ級数が収束する値
# ============================================================
POINTS = [(F(0), "0"), (F(1), r"\pi"), (F(-1), r"-\pi"), (F(1, 2), r"\frac{\pi}{2}"), (F(-1, 2), r"-\frac{\pi}{2}"),
          (F(2), r"2\pi"), (F(3), r"3\pi")]


def fval_pi(fam, t, side):
    """x = tπ での片側極限(side = +1 右, −1 左)を PiNum で(周期 2π で戻す)"""
    s = t
    while s > 1 or (s == 1 and side > 0):
        s -= 2
    while s < -1 or (s == -1 and side < 0):
        s += 2
    k = fam["kind"]
    if k == "step":
        return P(fam["A"] if (s > 0 or (s == 0 and side > 0)) else fam["B"])
    if k == "lin":
        return P(fam["c"] * s, 1) + P(fam["k"])
    if k == "half":
        return P(fam["c"] * s, 1) if (s > 0 or (s == 0 and side > 0)) else P()
    if k == "sq":
        return P(fam["c"] * s * s, 2) + P(fam["k"])
    raise ValueError(k)


def g_conv_value(rng):
    kind = rng.choice(["step", "lin", "half", "sq"])
    c = rng.choice([1, 2, -1, 3])
    k = rng.choice([0, 1, -1, 2])
    if kind == "step":
        A, B = rng.sample([-2, -1, 0, 1, 2, 3], 2)
        fam = family("step", "pi", A=A, B=B)
    elif kind == "lin":
        fam = dict(kind="lin", c=c, k=k, L="pi", tex=rf"f(x)={coef_term(c, 'x')}{'+' if k > 0 else ''}{k if k else ''}\ (-\pi<x\le\pi)".replace("+-", "-"))
    elif kind == "half":
        fam = family("half", "pi", c=c)
    else:
        fam = dict(kind="sq", c=c, k=k, L="pi", tex=rf"f(x)={coef_term(c, 'x^2')}{'+' if k > 0 else ''}{k if k else ''}\ (-\pi<x\le\pi)")
    t, ttex = rng.choice(POINTS)
    lft, rgt = fval_pi(fam, t, -1), fval_pi(fam, t, 1)
    ok = (lft + rgt).scale(F(1, 2))
    cands = [(rgt, "不連続点では、右側の極限ではなく左右の極限の平均に収束する。"),
             (lft, "不連続点では、左側の極限ではなく左右の極限の平均に収束する。"),
             (P(), "収束定理:$\\dfrac{f(x_0+0)+f(x_0-0)}{2}$ に収束する。周期性で $(-\\pi,\\pi]$ に戻して考える。"),
             (lft + rgt, "左右の極限の**平均**($\\frac12$ をかける)。"),
             (P(fam.get("c", 1), 1), "周期 $2\\pi$ で $(-\\pi,\\pi]$ に戻してから値を求めよう。")]
    choices = [Choice(ok.tex(), True, value=float(ok))]
    for v, why in cands:
        if all(abs(float(v) - x.value) > 1e-9 for x in choices) and len(choices) < 4:
            choices.append(Choice(v.tex(), why=why, value=float(v)))
    if len(choices) < 4:
        raise Retry
    jump = abs(float(lft) - float(rgt)) > 1e-12
    sol = [rf"周期 $2\pi$ なので $x={ttex}$ の左右の極限は $f(x-0)={lft.tex()}$, $f(x+0)={rgt.tex()}$。",
           (r"不連続点なので, 収束定理より左右の極限の平均" if jump else r"$f$ はこの点で連続なので, その値") +
           rf" $$\frac{{f(x-0)+f(x+0)}}{{2}}={ok.tex()}$$ に収束する。"]
    return Problem("収束定理(級数が収束する値)", "基本",
                   rf"周期 $2\pi$ の関数 ${fam['tex']}$, $f(x+2\pi)=f(x)$ のフーリエ級数は, $x={ttex}$ でどんな値に収束するか。",
                   choices, hints=[r"収束定理:区分的に滑らかな $f$ のフーリエ級数は, $\dfrac{f(x+0)+f(x-0)}{2}$ に収束する。",
                                   r"周期性を使って, $(-\pi,\ \pi]$ の中の点に戻してから考える。"],
                   solution=sol, meta=dict(fam=fam, t=t, value=float(ok)))


# ============================================================
# 標準:収束定理による級数の和
# ============================================================
SUMS = {"z2": (r"\sum_{n=1}^{\infty}\frac{1}{n^2}", P(F(1, 6), 2)),
        "a2": (r"\sum_{n=1}^{\infty}\frac{(-1)^{n+1}}{n^2}", P(F(1, 12), 2)),
        "o2": (r"\sum_{n=1}^{\infty}\frac{1}{(2n-1)^2}", P(F(1, 8), 2)),
        "lb": (r"1-\frac13+\frac15-\frac17+\cdots", P(F(1, 4), 1)),
        "z4": (r"\sum_{n=1}^{\infty}\frac{1}{n^4}", P(F(1, 90), 4)),
        "o4": (r"\sum_{n=1}^{\infty}\frac{1}{(2n-1)^4}", P(F(1, 96), 4))}
WRONG_SUMS = [P(F(1, 6), 2), P(F(1, 8), 2), P(F(1, 12), 2), P(F(1, 4), 1), P(F(1, 24), 2), P(F(1, 2), 1),
              P(F(1, 3), 2), P(F(1, 90), 4), P(F(1, 96), 4), P(F(1, 48), 4)]


def pi_uni(v):
    """PiNum(整数係数の単項)を目盛り用の文字に:2π², −π, 3"""
    if not v.t:
        return "0"
    (k, c), = v.t.items()
    sg = "−" if c < 0 else ""
    a = abs(c)
    if k == 0:
        return sg + str(a)
    return sg + ("" if a == 1 else str(a)) + "π" + ("" if k == 1 else "²³⁴"[k - 2])


def wrap_pi(x):
    """(−π, π] に戻す"""
    return x - 2 * PI * math.ceil((x - PI) / (2 * PI) - 1e-12)


def fgraph(fam, mark=None, K=3):
    """周期 2π に拡張した f のグラフ(ex_fig の fgraph 仕様)。不連続点は白丸・黒丸で示す"""
    f = fam["f"]
    pieces, op, cl = [], [], []
    for k in range(-K, K):
        xs = [k * PI + (PI) * j / 60 for j in range(61)]
        xs[0] += 1e-9
        xs[-1] -= 1e-9
        pieces.append([xs, [float(f(wrap_pi(x))) for x in xs]])
    for k in range(-K + 1, K):
        xb = k * PI
        lft, rgt, val = float(f(wrap_pi(xb - 1e-9))), float(f(wrap_pi(xb + 1e-9))), float(f(wrap_pi(xb)))
        if abs(lft - rgt) > 1e-6:
            op += [(xb, v) for v in (lft, rgt) if abs(v - val) > 1e-6]
            cl.append((xb, val))
    xt = [(k * PI, (r"\pi" if k == 1 else (r"-\pi" if k == -1 else f"{k}\\pi")) if k else "0",
           ("π" if k == 1 else ("−π" if k == -1 else f"{k}π".replace("-", "−"))) if k else "0") for k in range(-K, K + 1)]
    kd = fam["kind"]
    c = fam.get("c", 1)
    if kd == "sq":
        yv = [P(c, 2)]
    elif kd == "abs":
        yv = [P(c, 1)]
    elif kd == "x":
        yv = [P(c, 1), P(-c, 1)]
    elif kd == "step":
        yv = [P(fam["A"]), P(fam["B"])]
    else:
        yv = []
    yt = [(float(v), v.tex(), pi_uni(v)) for v in yv if v.t]
    return dict(kind="fgraph", pieces=pieces, open=op, closed=cl, xt=xt, yt=yt, mark=mark)


X0NUM = {"0": 0.0, r"\pi": PI, r"\frac{\pi}{2}": PI / 2}


def conv_sum_parts(rng):
    case = rng.choice(["sq_pi", "sq_0", "abs_0", "x_half", "step_half"])
    c = rng.choice([1, 1, 2])
    if case.startswith("sq"):
        fam = family("sq", "pi", c=c)
    elif case == "abs_0":
        fam = family("abs", "pi", c=c)
    elif case == "x_half":
        fam = family("x", "pi", c=c)
    else:
        A = rng.choice([1, 2])
        fam = family("step", "pi", A=A, B=-A)
    key, x0 = {"sq_pi": ("z2", r"\pi"), "sq_0": ("a2", "0"), "abs_0": ("o2", "0"), "x_half": ("lb", r"\frac{\pi}{2}"),
               "step_half": ("lb", r"\frac{\pi}{2}")}[case]
    stex, val = SUMS[key]
    fx = {"sq_pi": P(c, 2), "sq_0": P(), "abs_0": P(), "x_half": P(F(c, 2), 1), "step_half": P(fam.get("A", 0))}[case]
    sub = {"sq_pi": rf"$\cos n\pi=(-1)^n$ より $(-1)^n\cos n\pi=1$ なので ${fx.tex()}={fam['a0'].scale(F(1, 2)).tex()}+{4 * c}{SUMS['z2'][0]}$",
           "sq_0": rf"$\cos 0=1$ なので $0={fam['a0'].scale(F(1, 2)).tex()}+{4 * c}\sum_{{n=1}}^{{\infty}}\frac{{(-1)^n}}{{n^2}}$",
           "abs_0": rf"$\{{(-1)^n-1\}}$ は $n$ が奇数のとき $-2$, 偶数のとき $0$ なので $0={fam['a0'].scale(F(1, 2)).tex()}-\frac{{{4 * c}}}{{\pi}}{SUMS['o2'][0]}$",
           "x_half": rf"$\sin\frac{{n\pi}}{{2}}$ は $1,0,-1,0,\ldots$ なので ${fx.tex()}={2 * c}\left(1-\frac13+\frac15-\cdots\right)$",
           "step_half": rf"$\sin\frac{{n\pi}}{{2}}$ は $1,0,-1,0,\ldots$ なので ${fx.tex()}=\frac{{{4 * fam.get('A', 1)}}}{{\pi}}\left(1-\frac13+\frac15-\cdots\right)$"}[case]
    sol = [rf"$f$ のフーリエ級数は $$f(x)\sim {series_tex(fam)}$$",
           rf"$f$ は $x={x0}$ で連続なので, 収束定理より級数は $f({x0})={fx.tex()}$ に収束する。",
           sub.replace("+-", "-"), rf"これを解いて $${stex}={val.tex()}$$"]
    return fam, x0, stex, val, sol


def g_conv_sum(rng):
    fam, x0, stex, val, sol = conv_sum_parts(rng)
    wrong = [w for w in WRONG_SUMS if w.key() != val.key()]
    rng.shuffle(wrong)
    choices = [Choice(val.tex(), True, value=float(val))] + [Choice(w.tex(), why="$x_0$ を代入した式を、求める級数について解き直そう。",
                                                                    value=float(w)) for w in wrong[:3]]
    return Problem("収束定理による級数の和", "標準",
                   rf"周期 $2\pi$ の関数 {fam_text(fam)}(グラフは下の図)のフーリエ級数は $f(x)\sim {series_tex(fam)}$ である。"
                   rf"収束定理を用いて, ${stex}$ の値を求めよ。",
                   choices, hints=[r"$f$ が連続な点 $x_0$ では, フーリエ級数は $f(x_0)$ に収束する。",
                                   rf"$x={x0}$ を代入して, 求める級数が出てくる形にする。"],
                   solution=sol, meta=dict(val=float(val), sum=stex, fam=fam, x0=X0NUM[x0]),
                   exam_statement=rf"周期 $2\pi$ の関数 {fam_text(fam)} のフーリエ級数は $f(x)\sim {series_tex(fam)}$ である. "
                   rf"収束定理を用いて, ${stex}={val.tex()}$ を示せ.", exam_answer="",
                   figure=fgraph(fam), answer_figure=conv_mark(fam, x0))


def conv_mark(fam, x0):
    """解答の図:代入する点 x0 に印をつける"""
    xv = X0NUM[x0]
    uni = "x=" + {"0": "0", r"\pi": "π", r"\frac{\pi}{2}": "π/2"}[x0]
    return fgraph(fam, mark=(xv, float(fam["f"](xv)), "x=" + x0, uni))


def guided_conv_sum(rng):
    fam, x0, stex, val, sol = conv_sum_parts(rng)
    return ExamItem("フーリエ級数と収束定理", rf"周期 $2\pi$ の関数 {fam_text(fam)} について, 次の問いに答えよ.",
                    [Part(r"$f$ のフーリエ級数を求めよ.", series_tex(fam), coef_solution(fam)),
                     Part(rf"収束定理を用いて, ${stex}$ の値を求めよ.", val.tex(), sol[1:])],
                    meta=dict(val=float(val)), figure=fgraph(fam), answer_figure=conv_mark(fam, x0))


# ============================================================
# 標準:パーセバルの等式(係数の2乗和)
# ============================================================
def parseval_parts(rng):
    fam = pick_family(rng, ["x", "abs", "sq", "step"], "pi")
    I = fam["sq"]                                     # (1/π)∫ f² dx
    a0sq = fam["a0"] * fam["a0"]
    ans = I - a0sq.scale(F(1, 2))
    return fam, I, a0sq, ans


def g_parseval(rng):
    fam, I, a0sq, ans = parseval_parts(rng)
    cands = [(I, "$\\dfrac{a_0^2}{2}$ の項を引き忘れていないか。"),
             (I - a0sq, "パーセバルの等式の定数項は $\\dfrac{a_0^2}{2}$($a_0^2$ ではない)。"),
             (I * P(1, 1) - a0sq.scale(F(1, 2)), "左辺は $\\dfrac1\\pi\\int_{-\\pi}^{\\pi}f^2dx$。$\\dfrac1\\pi$ を忘れていないか。"),
             (I.scale(F(1, 2)) - a0sq.scale(F(1, 2)), "左辺は $\\dfrac1\\pi\\int_{-\\pi}^{\\pi}$($\\dfrac{1}{2\\pi}$ ではない)。"),
             (ans.scale(2), "計算を確かめよう。")]
    choices = [Choice(ans.tex(), True, value=float(ans))]
    for v, why in cands:
        if all(abs(float(v) - x.value) > 1e-9 for x in choices) and len(choices) < 4:
            choices.append(Choice(v.tex(), why=why, value=float(v)))
    if len(choices) < 4:
        raise Retry
    return Problem("パーセバルの等式(係数の2乗和)", "標準",
                   rf"周期 $2\pi$ の関数 {fam_text(fam)} のフーリエ係数を $a_n,\ b_n$ とする。パーセバルの等式を用いて, "
                   r"$\displaystyle\sum_{n=1}^{\infty}(a_n^2+b_n^2)$ の値を求めよ。",
                   choices, hints=[r"パーセバルの等式:$\dfrac1\pi\displaystyle\int_{-\pi}^{\pi}f(x)^2dx=\frac{a_0^2}{2}+\sum_{n=1}^{\infty}(a_n^2+b_n^2)$。",
                                   r"$a_0=\dfrac1\pi\displaystyle\int_{-\pi}^{\pi}f(x)\,dx$ も求めておく。"],
                   solution=[rf"$\dfrac1\pi\displaystyle\int_{{-\pi}}^{{\pi}}f(x)^2dx={I.tex()}$, $a_0={fam['a0'].tex()}$ より",
                             rf"$$\sum_{{n=1}}^{{\infty}}(a_n^2+b_n^2)={I.tex()}-\frac{{1}}{{2}}\left({fam['a0'].tex()}\right)^2={ans.tex()}$$"],
                   meta=dict(fam=fam, value=float(ans)))


# ============================================================
# 標準:パーセバルの等式による級数の和
# ============================================================
def parseval_sum_parts(rng):
    case = rng.choice(["x", "sq", "abs", "step"])
    c = rng.choice([1, 1, 2])
    if case == "step":
        A = rng.choice([1, 2])
        fam = family("step", "pi", A=A, B=-A)
    else:
        fam = family(case, "pi", c=c)
    key = {"x": "z2", "sq": "z4", "abs": "o4", "step": "o2"}[case]
    stex, val = SUMS[key]
    I = fam["sq"]
    a0h2 = (fam["a0"] * fam["a0"]).scale(F(1, 2))
    co = fam["an"] if fam["an"] is not None else fam["bn"]
    rel = {"x": rf"{4 * c * c}{SUMS['z2'][0]}", "sq": rf"{16 * c * c}{SUMS['z4'][0]}",
           "abs": rf"\frac{{{16 * c * c}}}{{\pi^2}}{SUMS['o4'][0]}",
           "step": rf"\frac{{{16 * fam.get('A', 1) ** 2}}}{{\pi^2}}{SUMS['o2'][0]}"}[case]
    sol = [rf"フーリエ級数は $f(x)\sim {series_tex(fam)}$。",
           rf"$\dfrac1\pi\displaystyle\int_{{-\pi}}^{{\pi}}f(x)^2dx={I.tex()}$。"
           + (rf"$\dfrac{{a_0^2}}{{2}}={a0h2.tex()}$。" if a0h2.t else ""),
           rf"係数の2乗の和は $\displaystyle\sum(a_n^2+b_n^2)={rel}$ になるので, パーセバルの等式は "
           rf"$${I.tex()}={a0h2.tex() + '+' if a0h2.t else ''}{rel}$$".replace("+-", "-"),
           rf"これを解いて $${stex}={val.tex()}$$"]
    return fam, I, stex, val, sol


def g_parseval_sum(rng):
    fam, I, stex, val, sol = parseval_sum_parts(rng)
    wrong = [w for w in WRONG_SUMS if w.key() != val.key()]
    rng.shuffle(wrong)
    choices = [Choice(val.tex(), True, value=float(val))] + [Choice(w.tex(), why="パーセバルの等式の両辺を確かめ, 求める級数について解き直そう。",
                                                                    value=float(w)) for w in wrong[:3]]
    return Problem("パーセバルの等式による級数の和", "標準",
                   rf"周期 $2\pi$ の関数 {fam_text(fam)} のフーリエ級数は $f(x)\sim {series_tex(fam)}$ である。"
                   rf"パーセバルの等式を用いて, ${stex}$ の値を求めよ。",
                   choices, hints=[r"パーセバルの等式:$\dfrac1\pi\displaystyle\int_{-\pi}^{\pi}f(x)^2dx=\frac{a_0^2}{2}+\sum_{n=1}^{\infty}(a_n^2+b_n^2)$。",
                                   r"係数を2乗して足すと, 求める級数が出てくる。"],
                   solution=sol, meta=dict(val=float(val)),
                   exam_statement=rf"周期 $2\pi$ の関数 {fam_text(fam)} のフーリエ級数は $f(x)\sim {series_tex(fam)}$ である. "
                   rf"パーセバルの等式を用いて, ${stex}={val.tex()}$ を示せ.", exam_answer="")


def guided_parseval_sum(rng):
    fam, I, stex, val, sol = parseval_sum_parts(rng)
    return ExamItem("パーセバルの等式",
                    rf"周期 $2\pi$ の関数 {fam_text(fam)} のフーリエ級数は $f(x)\sim {series_tex(fam)}$ である. 次の問いに答えよ.",
                    [Part(r"$\dfrac1\pi\displaystyle\int_{-\pi}^{\pi}f(x)^2dx$ を求めよ.", I.tex(), [sol[1]]),
                     Part(rf"パーセバルの等式を用いて, ${stex}$ の値を求めよ.", val.tex(), sol[2:])],
                    meta=dict(val=float(val)))


# ============================================================
# 標準:一般周期のフーリエ級数(級数全体を選ぶ)
# ============================================================
def g_series_general(rng):
    L = rng.choice([1, 2, 3])
    fam = pick_family(rng, ["x", "abs", "sq", "step", "half"], L)
    an, bn = fam["an"], fam["bn"]
    ok = series_tex(fam)
    m = {"x": 1, "abs": 1, "sq": 2, "half": 1, "step": 0}[fam["kind"]]

    def mod(fn):
        return dict(an=fn(an) if an is not None else None, bn=fn(bn) if bn is not None else None)
    cands = [(series_tex(fam, L="pi"), r"周期 $2L$ では $\cos\dfrac{n\pi x}{L},\ \sin\dfrac{n\pi x}{L}$ を使う($\cos nx$ は周期 $2\pi$ のとき)。"),
             (series_tex(fam, **mod(lambda c: Coef(c.R / F(L) ** m, c.pat, c.p, c.q))),
              rf"$x=\dfrac{{Lu}}{{\pi}}$ と置換すると係数に $L^{{{m}}}$ が出る。"),
             (series_tex(fam, **mod(lambda c: Coef(c.R, FLIP[c.pat], c.p, c.q))), r"符号の形を確かめよう:$\cos n\pi=(-1)^n$。"),
             (series_tex(fam, **mod(lambda c: Coef(c.R, c.pat, c.p, c.q - 1))), r"$\dfrac{n\pi x}{L}$ の積分で出る $\pi$ の個数を確かめよう。")]
    if fam["a0"].t:
        cands.append((series_tex(fam, a0h=fam["a0"]), r"定数項は $\dfrac{a_0}{2}$($a_0$ ではない)。"))
    if an is None or bn is None:
        cands.append((series_tex(fam, swap=True), "偶関数のフーリエ級数は $\\cos$ だけ、奇関数は $\\sin$ だけ(定数項を除く)。"))
    choices = [Choice(ok, True, value=ok)]
    for t, why in cands:
        if all(c.tex != t for c in choices) and len(choices) < 4:
            choices.append(Choice(t, why=why, value=t))
    if len(choices) < 4:
        raise Retry
    return Problem("一般周期のフーリエ級数", "標準",
                   rf"周期 ${2 * L}$ の関数 {fam_text(fam)} のフーリエ級数を求めよ。",
                   choices, hints=["規約:" + CONV, rf"$L={L}$。偶関数・奇関数かどうかを先に調べ, 部分積分を使う。"],
                   solution=coef_solution(fam), meta=dict(fam=fam))


# ============================================================
# 標準:複素フーリエ係数
# ============================================================
CONV_C = (r"$f(x)\sim\displaystyle\sum_{n=-\infty}^{\infty}c_ne^{in\pi x/L}$, "
          r"$c_n=\dfrac{1}{2L}\displaystyle\int_{-L}^{L}f(x)e^{-in\pi x/L}dx$")


class CCoef:
    """複素数の係数 re + i·im(re, im は Coef か None)"""
    def __init__(self, re=None, im=None):
        self.re = re if re is not None and re.R != 0 else None
        self.im = im if im is not None and im.R != 0 else None

    def __call__(self, n):
        return (self.re(n) if self.re else 0.0) + 1j * (self.im(n) if self.im else 0.0)

    def key(self):
        return (self.re.key() if self.re else None, self.im.key() if self.im else None)

    def tex(self):
        parts = ([self.re.tex()] if self.re else []) + ([self.im.tex() + "i"] if self.im else [])
        return "+".join(parts) if parts else "0"


def ccoef(fam, conj=False, scale=F(1, 2)):
    """c_n = (a_n − i b_n)/2(conj=True なら (a_n + i b_n)/2 の誤答)"""
    an, bn = fam["an"], fam["bn"]
    re = Coef(an.R * scale, an.pat, an.p, an.q) if an is not None else None
    im = Coef((1 if conj else -1) * bn.R * scale, bn.pat, bn.p, bn.q) if bn is not None else None
    return CCoef(re, im)


def ccoef_solution(fam, cn):
    L = fam["L"]
    Lt = Ltex(L)
    out = [rf"$c_n=\dfrac{{1}}{{2L}}\displaystyle\int_{{-L}}^{{L}}f(x)e^{{-in\pi x/L}}dx$ で $e^{{-i\theta}}=\cos\theta-i\sin\theta$ を使うと, "
           rf"$c_n=\dfrac{{a_n-ib_n}}{{2}}$($n\ne0$), $c_0=\dfrac{{a_0}}{{2}}$($L={Lt}$)。"]
    if fam["an"] is not None:
        out.append(rf"$a_n={fam['an'].tex()}$")
    else:
        out.append(r"$a_n=0$($n\ge1$)")
    if fam["bn"] is not None:
        out.append(rf"$b_n={fam['bn'].tex()}$")
    else:
        out.append(r"$b_n=0$")
    out.append(rf"よって $$c_n={cn.tex()}\quad(n\ne0),\qquad c_0={fam['a0'].scale(F(1, 2)).tex()}$$")
    return out


def drop_i(cn):
    """誤答:i のつけ忘れ・つけすぎ"""
    if cn.re is None:
        return CCoef(cn.im, None)
    if cn.im is None:
        return CCoef(None, cn.re)
    return CCoef(cn.re, None)


def g_ccoef(rng):
    L = rng.choice(["pi", "pi", 1, 2])
    fam = pick_family(rng, ["x", "abs", "sq", "step", "half"], L)
    cn = ccoef(fam)
    cands = [(ccoef(fam, conj=True), r"$c_n$ は $e^{-in\pi x/L}$ をかけて積分する:$c_n=\dfrac{a_n-ib_n}{2}$。$i$ の前の符号を確かめよう。"),
             (ccoef(fam, scale=F(1)), r"$c_n$ の前の係数は $\dfrac{1}{2L}$($a_n$ の $\dfrac1L$ の半分)。"),
             (CCoef(*(Coef(c.R, FLIP[c.pat], c.p, c.q) if c is not None else None for c in (cn.re, cn.im))),
              r"符号の形を確かめよう:$e^{\pm in\pi}=(-1)^n$。"),
             (drop_i(cn), r"$e^{-i\theta}=\cos\theta-i\sin\theta$ の虚部($i$ の項)を確かめよう。")]
    name = r"c_n"
    choices = [Choice(rf"{name}={cn.tex()}", True, value=cn.key())]
    for c, why in cands:
        t = rf"{name}={c.tex()}"
        if c.key() != cn.key() and all(x.tex != t for x in choices) and len(choices) < 4:
            choices.append(Choice(t, why=why, value=c.key()))
    if len(choices) < 4:
        raise Retry
    per = r"2\pi" if L == "pi" else str(2 * L)
    return Problem("複素フーリエ係数", "標準",
                   rf"周期 ${per}$ の関数 {fam_text(fam)} の複素フーリエ係数 $c_n\ (n\ne0)$ を求めよ。",
                   choices, hints=["規約:" + CONV_C, r"$c_n=\dfrac{a_n-ib_n}{2}$ を使うか, $e^{-in\pi x/L}$ をかけて直接部分積分する。"
                                   r"$e^{\pm in\pi}=\cos n\pi=(-1)^n$。"],
                   solution=ccoef_solution(fam, cn), meta=dict(fam=fam, cn=cn))


# ------------------------------------------------------------
# 指数関数 e^{kx} の複素フーリエ係数:c_n = (−1)^n sinh(kπ) / (π(k − in))
# ------------------------------------------------------------
SINH = r"ただし $\sinh x=\dfrac{e^x-e^{-x}}{2}$, $\cosh x=\dfrac{e^x+e^{-x}}{2}$ とする。"


def ktex(k, s):
    return s if k == 1 else f"{k}{s}"


def exp_cn_tex(k, sign=-1, alt=True, half=False, cosh=False):
    num = (r"(-1)^n" if alt else "") + (r"\cosh " if cosh else r"\sinh ") + ktex(k, r"\pi")
    den = ("2" if half else "") + rf"\pi({k}{'-' if sign < 0 else '+'}in)"
    return rf"\frac{{{num}}}{{{den}}}"


def exp_parts(rng):
    k = rng.choice([1, 2, 3])
    ftex = rf"f(x)=e^{{{ktex(k, 'x')}}}\ (-\pi<x\le\pi)"
    fam = dict(kind="exp", k=k, L="pi", tex=ftex, f=lambda x: math.exp(k * x))
    cn = exp_cn_tex(k)
    kp = ktex(k, r"\pi")
    sol = [rf"$$c_n=\frac{{1}}{{2\pi}}\int_{{-\pi}}^{{\pi}}e^{{{ktex(k, 'x')}}}e^{{-inx}}dx=\frac{{1}}{{2\pi}}\left[\frac{{e^{{({k}-in)x}}}}{{{k}-in}}\right]_{{-\pi}}^{{\pi}}"
           rf"=\frac{{e^{{{kp}}}e^{{-in\pi}}-e^{{-{kp}}}e^{{in\pi}}}}{{2\pi({k}-in)}}$$",
           rf"$e^{{\pm in\pi}}=(-1)^n$ なので $$c_n=\frac{{(-1)^n(e^{{{kp}}}-e^{{-{kp}}})}}{{2\pi({k}-in)}}={cn}$$"]
    return fam, k, cn, sol


def g_exp_coef(rng):
    fam, k, cn, sol = exp_parts(rng)
    wrong = [(exp_cn_tex(k, sign=1), r"$e^{-inx}$ をかけるので, 指数は $(k-in)x$。$in$ の前の符号を確かめよう。"),
             (exp_cn_tex(k, alt=False), r"$e^{\pm in\pi}=(-1)^n$ を忘れていないか。"),
             (exp_cn_tex(k, half=True), r"$\dfrac{e^{a}-e^{-a}}{2}=\sinh a$ なので, 分母の $2$ は $\sinh$ に入る。"),
             (exp_cn_tex(k, cosh=True), r"$e^{a}-e^{-a}$ は $2\sinh a$($\cosh$ は和)。")]
    rng.shuffle(wrong)
    choices = [Choice(rf"c_n={cn}", True, value=cn)] + [Choice(rf"c_n={t}", why=w, value=t) for t, w in wrong[:3]]
    return Problem("複素フーリエ係数(指数関数)", "標準",
                   rf"周期 $2\pi$ の関数 ${fam['tex']}$, $f(x+2\pi)=f(x)$ の複素フーリエ係数 $c_n$ を求めよ。" + SINH,
                   choices, hints=["規約:" + CONV_C, r"$e^{kx}e^{-inx}=e^{(k-in)x}$ はそのまま積分できる(部分積分はいらない)。"],
                   solution=sol, meta=dict(fam=fam, k=k))


# ------------------------------------------------------------
# 誘導:複素フーリエ係数 → パーセバルの等式(複素形)
# ------------------------------------------------------------
PARS_C = r"パーセバルの等式(複素形):$\dfrac{1}{2\pi}\displaystyle\int_{-\pi}^{\pi}|f(x)|^2dx=\sum_{n=-\infty}^{\infty}|c_n|^2$"


def guided_ccoef(rng):
    c = rng.choice([1, 2, 3])
    fam = family("x", "pi", c=c)
    cn = ccoef(fam)
    I = P(F(c * c, 3), 2)       # (1/2π)∫ c²x² dx = c²π²/3
    p1 = ccoef_solution(fam, cn)
    p2 = [rf"$|c_n|^2=\dfrac{{{c * c}}}{{n^2}}\ (n\ne0)$, $c_0=0$, $\dfrac{{1}}{{2\pi}}\displaystyle\int_{{-\pi}}^{{\pi}}{coef_term(c * c, 'x^2')}dx={I.tex()}$。",
          rf"$|c_{{-n}}|=|c_n|$ なので $${I.tex()}={2 * c * c}\sum_{{n=1}}^{{\infty}}\frac{{1}}{{n^2}}\quad\text{{より}}\quad "
          r"\sum_{n=1}^{\infty}\frac{1}{n^2}=\frac{\pi^{2}}{6}$$"]
    return ExamItem("複素フーリエ級数とパーセバルの等式",
                    rf"周期 $2\pi$ の関数 {fam_text(fam)} について, 次の問いに答えよ. 複素フーリエ係数は "
                    r"$c_n=\dfrac{1}{2\pi}\displaystyle\int_{-\pi}^{\pi}f(x)e^{-inx}dx$ とする.",
                    [Part(r"複素フーリエ係数 $c_n\ (n\ne0)$ と $c_0$ を求めよ.", rf"c_n={cn.tex()}\ (n\ne0),\ c_0=0", p1),
                     Part(PARS_C.replace("パーセバルの等式(複素形):", "パーセバルの等式 ") + r" を用いて, "
                          r"$\displaystyle\sum_{n=1}^{\infty}\frac{1}{n^2}$ の値を求めよ.", r"\frac{\pi^{2}}{6}", p2)],
                    meta=dict(fam=fam, cn=cn, val=PI ** 2 / 6))


def guided_exp(rng):
    fam, k, cn, sol = exp_parts(rng)
    kp, k2p = ktex(k, r"\pi"), ktex(2 * k, r"\pi")
    ans = rf"\frac{{\pi\cosh {kp}}}{{{'' if k == 1 else k}\sinh {kp}}}"
    p2 = [rf"$|c_n|^2=\dfrac{{\sinh^2{kp}}}{{\pi^2({k * k}+n^2)}}$。",
          rf"$\dfrac{{1}}{{2\pi}}\displaystyle\int_{{-\pi}}^{{\pi}}e^{{{ktex(2 * k, 'x')}}}dx=\dfrac{{e^{{{k2p}}}-e^{{-{k2p}}}}}{{{4 * k}\pi}}"
          rf"=\dfrac{{\sinh {kp}\cosh {kp}}}{{{'' if k == 1 else k}\pi}}$(2倍角:$\sinh 2a=2\sinh a\cosh a$)。",
          rf"パーセバルの等式より $$\frac{{\sinh {kp}\cosh {kp}}}{{{'' if k == 1 else k}\pi}}=\frac{{\sinh^2{kp}}}{{\pi^2}}\sum_{{n=-\infty}}^{{\infty}}\frac{{1}}{{n^2+{k * k}}}"
          rf"\quad\text{{より}}\quad\sum_{{n=-\infty}}^{{\infty}}\frac{{1}}{{n^2+{k * k}}}={ans}$$"]
    return ExamItem("指数関数の複素フーリエ級数",
                    rf"周期 $2\pi$ の関数 ${fam['tex']}$, $f(x+2\pi)=f(x)$ について, 次の問いに答えよ. "
                    r"複素フーリエ係数は $c_n=\dfrac{1}{2\pi}\displaystyle\int_{-\pi}^{\pi}f(x)e^{-inx}dx$ とする. " + SINH.replace("。", "."),
                    [Part(r"複素フーリエ係数 $c_n$ を求めよ.", rf"c_n={cn}", sol),
                     Part(PARS_C.replace("パーセバルの等式(複素形):", "パーセバルの等式 ") + r" を用いて, "
                          rf"$\displaystyle\sum_{{n=-\infty}}^{{\infty}}\frac{{1}}{{n^2+{k * k}}}$ の値を求めよ.", ans, p2)],
                    meta=dict(k=k, val=PI * math.cosh(k * PI) / (k * math.sinh(k * PI))))


REGISTRY = [
    ProblemType("fs_2pi", "周期 2π のフーリエ級数", "基本", g_series_2pi),
    ProblemType("fs_conv", "収束定理(級数が収束する値)", "基本", g_conv_value),
    ProblemType("fs_gen", "一般周期のフーリエ係数", "標準", g_coef_general),
    ProblemType("fs_gens", "一般周期のフーリエ級数", "標準", g_series_general),
    ProblemType("fc_coef", "複素フーリエ係数", "標準", g_ccoef, guided=guided_ccoef),
    ProblemType("fc_exp", "複素フーリエ係数(指数関数)", "標準", g_exp_coef, guided=guided_exp),
    ProblemType("fs_sum", "収束定理による級数の和", "標準", g_conv_sum, guided=guided_conv_sum),
    ProblemType("fs_pars", "パーセバルの等式(係数の2乗和)", "標準", g_parseval),
    ProblemType("fs_psum", "パーセバルの等式による級数の和", "標準", g_parseval_sum, guided=guided_parseval_sum),
]
GUIDED_NAMES = {"fs_sum": "フーリエ級数 → 収束定理による級数の和", "fs_psum": "∫f² → パーセバルの等式による級数の和",
                "fc_coef": "複素フーリエ係数 → パーセバルの等式(複素形)で Σ1/n²",
                "fc_exp": "e^{kx} の複素フーリエ係数 → Σ1/(n²+k²)"}


def tab():
    render("fourier1", REGISTRY, "フーリエ解析 第1章 フーリエ級数")
