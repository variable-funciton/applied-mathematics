"""数学IV フーリエ解析 第1章 フーリエ級数:演習の問題生成

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


def series_tex(fam, an=None, bn=None, a0h=None, swap=False):
    """フーリエ級数の式。swap=True なら cos と sin を入れかえた誤答"""
    L = fam["L"]
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
                   rf"周期 $2\pi$ の関数 {fam_text(fam)} のフーリエ級数は $f(x)\sim {series_tex(fam)}$ である。"
                   rf"収束定理を用いて, ${stex}$ の値を求めよ。",
                   choices, hints=[r"$f$ が連続な点 $x_0$ では, フーリエ級数は $f(x_0)$ に収束する。",
                                   rf"$x={x0}$ を代入して, 求める級数が出てくる形にする。"],
                   solution=sol, meta=dict(val=float(val), sum=stex),
                   exam_statement=rf"周期 $2\pi$ の関数 {fam_text(fam)} のフーリエ級数は $f(x)\sim {series_tex(fam)}$ である. "
                   rf"収束定理を用いて, ${stex}={val.tex()}$ を示せ.", exam_answer="")


def guided_conv_sum(rng):
    fam, x0, stex, val, sol = conv_sum_parts(rng)
    return ExamItem("フーリエ級数と収束定理", rf"周期 $2\pi$ の関数 {fam_text(fam)} について, 次の問いに答えよ.",
                    [Part(r"$f$ のフーリエ級数を求めよ.", series_tex(fam), coef_solution(fam)),
                     Part(rf"収束定理を用いて, ${stex}$ の値を求めよ.", val.tex(), sol[1:])],
                    meta=dict(val=float(val)))


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


REGISTRY = [
    ProblemType("fs_2pi", "周期 2π のフーリエ級数", "基本", g_series_2pi),
    ProblemType("fs_conv", "収束定理(級数が収束する値)", "基本", g_conv_value),
    ProblemType("fs_gen", "一般周期のフーリエ係数", "標準", g_coef_general),
    ProblemType("fs_sum", "収束定理による級数の和", "標準", g_conv_sum, guided=guided_conv_sum),
    ProblemType("fs_pars", "パーセバルの等式(係数の2乗和)", "標準", g_parseval),
    ProblemType("fs_psum", "パーセバルの等式による級数の和", "標準", g_parseval_sum, guided=guided_parseval_sum),
]
GUIDED_NAMES = {"fs_sum": "フーリエ級数 → 収束定理による級数の和", "fs_psum": "∫f² → パーセバルの等式による級数の和"}


def tab():
    render("fourier1", REGISTRY, "フーリエ解析 第1章 フーリエ級数")
