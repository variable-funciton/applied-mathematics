"""数学IV 特殊関数(ガンマ関数・ベータ関数):演習の問題生成

前期演習問題10(学修単位対応)の10問の型に合わせている。
  ガンマ関数:∫x^{n/2}e^{-x}dx, ∫_a^∞(x-b)(x-a)^n e^{-x}dx, ∫x^k e^{-x²}dx, ∫_1^∞(log x)^n x^{-(c+1)}dx
  ベータ関数で表す:∫_0^1 x^p(1-x)^q dx, ∫_a^b (b-x)^m (x-a)^r dx, ∫_{-a}^a (√(a²-x²))^n dx,
                   ∫_0^∞ x^m/(x+c)^n dx, ∫_1^e (1-log x)^m (log x)^r / x dx, ∫_0^{π/2} sin^a x cos^b x dx
各選択肢の value は数値(テストで数値積分と照合する)。
"""
import math
from fractions import Fraction

from topics.ex_core import Choice, ExamItem, Part, Problem, ProblemType, Retry, render, sqrt_parts, tex_fr

F = Fraction
DEFS = (r"$\Gamma(s)=\displaystyle\int_0^\infty e^{-x}x^{s-1}dx\ (s>0)$, "
        r"$B(p,q)=\displaystyle\int_0^1x^{p-1}(1-x)^{q-1}dx\ (p,q>0)$")
FORMULAS = (r"$\Gamma(s+1)=s\Gamma(s)$, $\Gamma(n+1)=n!$, $\Gamma\left(\frac12\right)=\sqrt\pi$, "
            r"$B(p,q)=\dfrac{\Gamma(p)\Gamma(q)}{\Gamma(p+q)}$, "
            r"$\displaystyle\int_0^{\frac\pi2}\sin^ax\cos^bx\,dx=\frac12B\left(\frac{a+1}{2},\frac{b+1}{2}\right)$")


# ------------------------------------------------------------
# 厳密な値:r·√s·π^{k/2}(r は有理数、s は平方因子なしの整数、k = 0, 1, 2)
# ------------------------------------------------------------
class Num:
    def __init__(self, r, s=1, k=0):
        a, m = sqrt_parts(s)
        self.r, self.s, self.k = F(r) * a, m, k
        if self.k >= 2:                           # π^{k/2} の整数部分はそのまま π^... で持つ
            pass

    def __mul__(self, o):
        r = self.r * o.r
        s = self.s * o.s
        a, m = sqrt_parts(s)
        return Num(r * a, m, self.k + o.k)

    def __float__(self):
        return float(self.r) * math.sqrt(self.s) * math.pi ** (self.k / 2)

    def tex(self):
        r = self.r
        body = ("" if self.s == 1 else rf"\sqrt{{{self.s}}}")
        pi = {0: "", 1: r"\sqrt{\pi}", 2: r"\pi", 3: r"\pi\sqrt{\pi}", 4: r"\pi^2"}[self.k]
        tail = body + pi
        if not tail:
            return tex_fr(r)
        sgn = "-" if r < 0 else ""
        a = abs(r)
        if a.denominator == 1:
            return sgn + ("" if a == 1 else str(a.numerator)) + tail
        return sgn + rf"\frac{{{'' if a.numerator == 1 else a.numerator}{tail}}}{{{a.denominator}}}"


def gamma_exact(x):
    """x が正の整数か半整数のときの Γ(x)"""
    x = F(x)
    if x.denominator == 1:
        return Num(math.factorial(x.numerator - 1))
    if x.denominator != 2:
        return None
    r, y = F(1), F(1, 2)
    while y < x:
        r *= y
        y += 1
    return Num(r, 1, 1)


def beta_exact(p, q):
    gp, gq, gs = gamma_exact(p), gamma_exact(q), gamma_exact(F(p) + F(q))
    if None in (gp, gq, gs):
        return None
    num = gp * gq
    k = num.k - gs.k
    return Num(num.r / gs.r, 1, k)


def gamma_f(x):
    return math.gamma(float(x))


def beta_f(p, q):
    return math.exp(math.lgamma(float(p)) + math.lgamma(float(q)) - math.lgamma(float(p) + float(q)))


def fr(x):
    """B( , ) や Γ( ) の中に書く分数"""
    x = F(x)
    return str(x.numerator) if x.denominator == 1 else rf"\frac{{{x.numerator}}}{{{x.denominator}}}"


def btex(p, q):
    return rf"B\left({fr(p)},{fr(q)}\right)"


def coef_beta(c, p, q):
    """係数 c(Num)× B(p, q) の TeX と数値"""
    ct = c.tex()
    if ct == "1":
        t = btex(p, q)
    elif ct == "-1":
        t = "-" + btex(p, q)
    elif c.r.denominator != 1 and c.s == 1 and c.k == 0 and abs(c.r.numerator) == 1:
        t = ("-" if c.r < 0 else "") + rf"\dfrac{{{btex(p, q)}}}{{{c.r.denominator}}}"
    else:
        t = ct + btex(p, q)
    return t, float(c) * beta_f(p, q)


def power_num(L, e):
    """L^e(e は整数か半整数。負でもよい)"""
    e = F(e)
    n = math.floor(e)
    r = F(L) ** n
    return Num(r, L if e - n else 1)


def pw(base, e):
    """(base)^e の TeX(e = 1 なら指数なし)"""
    e = F(e)
    if e == 1:
        return base
    return rf"{base}^{{{fr(e)}}}"


def value_line(c, p, q):
    v = beta_exact(p, q)
    if v is None:
        return None
    return (c * v).tex()


def pick_choices(correct, cands):
    """正解と、数値が重ならない誤答3つ(足りなければ Retry)"""
    def big(t):                       # 値の分数は \dfrac で読みやすく(B( , ) の中はそのまま)
        return t if "B" in t else t.replace(r"\frac", r"\dfrac")
    correct = (big(correct[0]), correct[1])
    cands = [(big(t), v, why) for t, v, why in cands]
    choices = [Choice(correct[0], True, value=correct[1])]
    seen = [correct[1]]
    for t, v, why in cands:
        if v is None or not math.isfinite(v) or any(abs(v - s) < 1e-9 * max(1, abs(s)) for s in seen):
            continue
        if any(c.tex == t for c in choices):
            continue
        choices.append(Choice(t, why=why, value=v))
        seen.append(v)
        if len(choices) == 4:
            return choices
    raise Retry


# ============================================================
# ガンマ関数
# ============================================================
def g_gamma_half(rng):
    n = rng.choice([1, 3, 5, 7, 9])
    s = F(n, 2) + 1
    ok = gamma_exact(s)
    cands = [(gamma_exact(s - 1).tex(), float(gamma_exact(s - 1)),
              r"$\Gamma(s)=\int_0^\infty x^{s-1}e^{-x}dx$。$x$ の指数は $s-1$ なので $s$ は指数より $1$ 大きい。"),
             (gamma_exact(s + 1).tex(), float(gamma_exact(s + 1)), r"$x$ の指数 $=s-1$ から $s$ を決めよう。"),
             (Num(ok.r, 1, 2).tex(), float(Num(ok.r, 1, 2)), r"$\Gamma\left(\frac12\right)=\sqrt\pi$($\pi$ ではない)。")]
    steps = []
    y = s
    while y > F(1, 2):
        steps.append(fr(y - 1))
        y -= 1
    sol = [rf"$x^{{{fr(F(n, 2))}}}=x^{{s-1}}$ より $s={fr(s)}$。",
           rf"$$\int_0^\infty x^{{{fr(F(n, 2))}}}e^{{-x}}dx=\Gamma\left({fr(s)}\right)="
           + r"\cdot".join(steps) + r"\cdot\Gamma\left(\frac12\right)=" + ok.tex() + "$$"]
    return Problem("ガンマ関数の値", "基本",
                   rf"広義積分 $\displaystyle\int_0^\infty x^{{{fr(F(n, 2))}}}e^{{-x}}dx$ の値を求めよ。",
                   pick_choices((ok.tex(), float(ok)), cands),
                   hints=[DEFS, r"$\Gamma(s+1)=s\Gamma(s)$ をくり返し使い、最後に $\Gamma\left(\frac12\right)=\sqrt\pi$。"],
                   solution=sol, meta=dict(f=lambda x: x ** (n / 2) * math.exp(-x), a=0, b=math.inf))


def g_gamma_shift(rng):
    a = rng.choice([1, 2, 3, 4])
    b = rng.choice([x for x in range(-2, 8) if x != a])
    n = rng.choice([2, 3, 4, 5, 6])
    K = math.factorial(n) * (n + 1 + a - b)
    if K == 0:
        raise Retry

    def et(k):
        return rf"\frac{{{k}}}{{e^{{{a}}}}}" if k > 0 else rf"-\frac{{{-k}}}{{e^{{{a}}}}}"
    d = a - b
    xb = "x" if b == 0 else (f"x-{b}" if b > 0 else f"x+{-b}")
    tb = "t" if b == 0 else (f"t-{b}" if b > 0 else f"t+{-b}")
    cands = [(str(K), float(K), r"$t=x-a$ とおくと $e^{-x}=e^{-a}e^{-t}$。$e^{-a}$ を残し忘れていないか。"),
             (et(math.factorial(n) * (n + 1 - b)) if n + 1 - b else "0", math.factorial(n) * (n + 1 - b) * math.exp(-a),
              rf"$t=x-{a}$ のとき ${xb}=t{'+' if d >= 0 else '-'}{abs(d)}$。${tb}$ ではない。"),
             (et(math.factorial(n - 1) * (n + d)) if n + d else "0", math.factorial(n - 1) * (n + d) * math.exp(-a),
              r"$\int_0^\infty t^ne^{-t}dt=\Gamma(n+1)=n!$。$(n-1)!$ ではない。"),
             (et(-K), -K * math.exp(-a), "符号を確かめよう。")]
    sgn = "+" if d >= 0 else "-"
    bt = rf"(x{'-' if b > 0 else '+'}{abs(b)})" if b else "x"
    sol = [rf"$t=x-{a}$ とおくと $x={a}$ で $t=0$, $dx=dt$, ${xb}=t{sgn}{abs(d)}$, $e^{{-x}}=e^{{-{a}}}e^{{-t}}$。",
           rf"$$\int_{a}^\infty{bt}(x-{a})^{{{n}}}e^{{-x}}dx=e^{{-{a}}}\int_0^\infty\left(t^{{{n + 1}}}{sgn}{abs(d)}t^{{{n}}}\right)e^{{-t}}dt"
           rf"=e^{{-{a}}}\left(\Gamma({n + 2}){sgn}{abs(d)}\Gamma({n + 1})\right)$$",
           rf"$$=e^{{-{a}}}\left({math.factorial(n + 1)}{sgn}{abs(d) * math.factorial(n)}\right)={et(K)}$$"]
    return Problem("ガンマ関数(平行移動)", "標準",
                   rf"広義積分 $\displaystyle\int_{a}^\infty {bt}(x-{a})^{{{n}}}e^{{-x}}dx$ の値を求めよ。",
                   pick_choices((et(K), K * math.exp(-a)), cands),
                   hints=[DEFS, rf"$t=x-{a}$ とおいて、$\int_0^\infty t^ne^{{-t}}dt=n!$ の形にする。"],
                   solution=sol, meta=dict(f=lambda x: (x - b) * (x - a) ** n * math.exp(-x), a=a, b=math.inf))


def g_gauss(rng):
    k = rng.choice([2, 3, 4, 5, 6])
    s = F(k + 1, 2)
    ok = Num(F(1, 2)) * gamma_exact(s)
    g1 = gamma_exact(s)
    g2 = gamma_exact(F(k, 2) + 1)
    g3 = gamma_exact(k + 1)
    cands = [(g1.tex(), float(g1), r"$t=x^2$ とおくと $dx=\frac{dt}{2\sqrt t}$。$\frac12$ を忘れていないか。"),
             ((Num(F(1, 2)) * g2).tex(), float(Num(F(1, 2)) * g2), r"$dx=\frac{1}{2}t^{-\frac12}dt$ の $t^{-\frac12}$ も指数に入れる。"),
             ((Num(F(1, 2)) * g3).tex(), float(Num(F(1, 2)) * g3), r"$x^k=t^{\frac k2}$($t=x^2$)。$t^k$ ではない。")]
    sol = [r"$t=x^2$ とおくと $x=\sqrt t$, $dx=\dfrac{1}{2\sqrt t}dt$。",
           rf"$$\int_0^\infty x^{{{k}}}e^{{-x^2}}dx=\frac12\int_0^\infty t^{{{fr(F(k, 2))}}}t^{{-\frac12}}e^{{-t}}dt"
           rf"=\frac12\int_0^\infty t^{{{fr(F(k - 1, 2))}}}e^{{-t}}dt=\frac12\Gamma\left({fr(s)}\right)={ok.tex()}$$"]
    return Problem("ガウス型の積分", "標準",
                   rf"広義積分 $\displaystyle\int_0^\infty x^{{{k}}}e^{{-x^2}}dx$ の値を求めよ。",
                   pick_choices((ok.tex(), float(ok)), cands),
                   hints=[DEFS, r"$t=x^2$ とおいてガンマ関数の形にする。"],
                   solution=sol, meta=dict(f=lambda x: x ** k * math.exp(-x * x), a=0, b=math.inf))


def g_log_gamma(rng):
    n = rng.choice([2, 3, 4, 5])
    c = rng.choice([1, 1, 2, 3])
    ok = F(math.factorial(n), c ** (n + 1))
    cands = [(tex_fr(F(math.factorial(n), c ** n)), math.factorial(n) / c ** n,
              rf"$t=\log x$ とおくと $x^{{-{c + 1}}}dx=e^{{-{c}t}}dt$。$\int_0^\infty t^ne^{{-ct}}dt=\frac{{n!}}{{c^{{n+1}}}}$。"),
             (tex_fr(F(math.factorial(n - 1), c ** (n + 1))), math.factorial(n - 1) / c ** (n + 1),
              r"$\int_0^\infty t^ne^{-t}dt=\Gamma(n+1)=n!$。$(n-1)!$ ではない。"),
             (tex_fr(F(math.factorial(n + 1), c ** (n + 1))), math.factorial(n + 1) / c ** (n + 1),
              r"$\Gamma(n+1)=n!$。$(n+1)!$ ではない。"),
             (tex_fr(F(math.factorial(n), c ** (n + 2))), math.factorial(n) / c ** (n + 2),
              r"$u=ct$ とおくと $t^n\,dt=\frac{u^n}{c^{n+1}}du$。")]
    xc = "x^{-2}" if c == 1 else rf"x^{{-{c + 1}}}"
    sol = [r"$t=\log x$ とおくと $x=e^t$, $dx=e^t\,dt$。$x=1$ で $t=0$, $x\to\infty$ で $t\to\infty$。",
           rf"$$\int_1^\infty(\log x)^{{{n}}}{xc}\,dx=\int_0^\infty t^{{{n}}}e^{{-{c + 1}t}}e^t\,dt="
           rf"\int_0^\infty t^{{{n}}}e^{{-{'' if c == 1 else c}t}}dt$$"]
    if c == 1:
        sol.append(rf"$$=\Gamma({n + 1})={n}!={math.factorial(n)}$$")
    else:
        sol.append(rf"$u={c}t$ とおくと $$=\frac{{1}}{{{c}^{{{n + 1}}}}}\int_0^\infty u^{{{n}}}e^{{-u}}du=\frac{{{n}!}}{{{c ** (n + 1)}}}={tex_fr(ok)}$$")
    return Problem("対数を含む積分", "標準",
                   rf"広義積分 $\displaystyle\int_1^\infty(\log x)^{{{n}}}{xc}\,dx$ の値を求めよ。",
                   pick_choices((tex_fr(ok), float(ok)), cands),
                   hints=[DEFS, r"$t=\log x$ とおくと $x=e^t$。ガンマ関数の形になる。"],
                   solution=sol, meta=dict(f=lambda x: math.log(x) ** n * x ** (-(c + 1)), a=1, b=math.inf))


# ============================================================
# ベータ関数で表す
# ============================================================
def beta_problem(title, level, integral, c, p, q, cands, hints, sol, meta):
    ok = coef_beta(c, p, q)
    vl = value_line(c, p, q)
    sol = list(sol) + [rf"$$I={ok[0]}$$"]
    if vl:
        sol.append(rf"(値は $B(p,q)=\dfrac{{\Gamma(p)\Gamma(q)}}{{\Gamma(p+q)}}$ から $I={vl}$)")
    full = []
    for cc, pp, qq, why in cands:
        if F(pp) <= 0 or F(qq) <= 0:
            continue
        t, v = coef_beta(cc, pp, qq)
        full.append((t, v, why))
    choices = pick_choices(ok, full)
    meta = dict(meta, c=c, p=p, q=q, value=vl)
    return Problem(title, level, rf"積分 $I=\displaystyle{integral}$ をベータ関数で表せ。", choices,
                   hints=[DEFS] + hints, solution=sol, meta=meta,
                   exam_statement=rf"積分 $I=\displaystyle{integral}$ をベータ関数で表せ.")


ONE = Num(1)


def beta_poly_parts(rng):
    p, q = rng.sample(range(1, 9), 2)
    return p, q, rf"\int_0^1 {pw('x', p)}{pw('(1-x)', q)}dx"


def g_beta_poly(rng):
    p, q, integral = beta_poly_parts(rng)
    cands = [(ONE, p, q, r"$B(p,q)$ の被積分関数は $x^{p-1}(1-x)^{q-1}$。指数に $1$ を足す。"),
             (ONE, p + 1, q, r"$(1-x)$ の指数にも $1$ を足す。"),
             (ONE, p, q + 1, r"$x$ の指数にも $1$ を足す。")]
    return beta_problem("ベータ関数で表す(基本形)", "基本", integral, ONE, p + 1, q + 1, cands,
                        [r"$x^{p-1}(1-x)^{q-1}$ と見比べて $p,\ q$ を決める。"],
                        [rf"$x^{{{p}}}=x^{{{p + 1}-1}},\ (1-x)^{{{q}}}=(1-x)^{{{q + 1}-1}}$ より"],
                        dict(f=lambda x: x ** p * (1 - x) ** q, a=0, b=1))


def interval_parts(rng):
    a = rng.randint(-2, 3)
    L = rng.choice([2, 3, 4, 5])
    b = a + L
    m = rng.choice([1, 2, 3, 4])
    k = rng.choice([1, 2, 3, 5])                 # (x-a) の指数 r = k/2
    r = F(k, 2)
    if r == m:
        raise Retry
    xa = rf"(x{'-' if a > 0 else '+'}{abs(a)})" if a else "x"
    sq = rf"\sqrt{{x{'-' if a > 0 else '+'}{abs(a)}}}" if a else r"\sqrt{x}"
    rt = (sq if k == 1 else rf"{sq}^{{{k}}}") if k % 2 else (xa if k == 2 else rf"{xa}^{{{k // 2}}}")
    integral = rf"\int_{{{a}}}^{{{b}}}({b}-x)^{{{m}}}{rt}\,dx"
    c = power_num(L, m + r + 1)
    return a, b, L, m, r, integral, c


def interval_solution(a, b, L, m, r, c):
    sub = (f"{a}+{L}t" if a else f"{L}t").replace("+-", "-")
    xa = "x" if a == 0 else (f"x-{a}" if a > 0 else f"x+{-a}")
    return [rf"$x={sub}$ とおくと $dx={L}\,dt$, ${xa}={L}t$, ${b}-x={L}(1-t)$, $x:{a}\to{b}$ で $t:0\to1$。",
            rf"$$I=\int_0^1({L}(1-t))^{{{m}}}({L}t)^{{{fr(r)}}}\cdot{L}\,dt={L}^{{{fr(m + r + 1)}}}\int_0^1t^{{{fr(r)}}}(1-t)^{{{m}}}dt"
            rf"={L}^{{{fr(m + r + 1)}}}{btex(r + 1, m + 1)}$$"]


def g_beta_interval(rng):
    a, b, L, m, r, integral, c = interval_parts(rng)
    cands = [(ONE, r + 1, m + 1, rf"$x=a+(b-a)t$ と置換すると、${L}$ の累乗と $dx={L}\,dt$ の係数が出る。"),
             (power_num(L, m + r), r + 1, m + 1, rf"$dx={L}\,dt$ の ${L}$ もかける。"),
             (c, r, m, r"$t^{p-1}(1-t)^{q-1}$ と見比べる。指数に $1$ を足す。"),
             (c, m + 1, r + 1 + 1, "指数と $p,\\ q$ の対応を確かめよう。")]
    return beta_problem("ベータ関数で表す(区間 [a, b])", "標準", integral, c, r + 1, m + 1, cands,
                        [r"$x=a+(b-a)t$ と置換して $\int_0^1$ にする。"], interval_solution(a, b, L, m, r, c),
                        dict(f=lambda x: (b - x) ** m * (x - a) ** float(r), a=a, b=b))


def semi_parts(rng):
    A = rng.choice([1, 2, 3])
    n = rng.choice([1, 2, 3, 4, 5, 6])
    inner = rf"\sqrt{{{A * A}-x^2}}"
    integral = rf"\int_{{-{A}}}^{{{A}}}\left({inner}\right)^{{{n}}}dx" if n > 1 else rf"\int_{{-{A}}}^{{{A}}}{inner}\,dx"
    c = Num((2 * A) ** (n + 1))
    p = F(n, 2) + 1
    return A, n, integral, c, p


def semi_solution(A, n, c, p):
    lead = pw(rf"\left({2 * A}\sqrt{{t(1-t)}}\right)", n)
    return [rf"$x=-{A}+{2 * A}t$ とおくと $dx={2 * A}\,dt$, ${A * A}-x^2=({A}-x)({A}+x)={2 * A}(1-t)\cdot{2 * A}t={4 * A * A}t(1-t)$。",
            rf"$$I=\int_0^1{lead}\cdot{2 * A}\,dt={2 * A}^{{{n + 1}}}\int_0^1t^{{{fr(F(n, 2))}}}(1-t)^{{{fr(F(n, 2))}}}dt"
            rf"={c.tex()}{btex(p, p)}$$"]


def g_beta_semi(rng):
    A, n, integral, c, p = semi_parts(rng)
    cands = [(Num((2 * A) ** n), p, p, rf"$dx={2 * A}\,dt$ の ${2 * A}$ もかける。"),
             (Num(A ** (n + 1)), p, p, rf"$x=-{A}+{2 * A}t$ と置換すると、出てくる係数は ${2 * A}$。"),
             (c, p - 1, p - 1, r"$t^{p-1}(1-t)^{q-1}$ と見比べる。指数に $1$ を足す。"),
             (c, p, p + 1, "2つの指数は等しい($t$ と $1-t$ は対称)。")]
    return beta_problem("ベータ関数で表す(半円型)", "標準", integral, c, p, p, cands,
                        [rf"$x=-{A}+{2 * A}t$ とおき、${A * A}-x^2$ を $t(1-t)$ で表す。"], semi_solution(A, n, c, p),
                        dict(f=lambda x: max(A * A - x * x, 0.0) ** (n / 2), a=-A, b=A))


def rational_parts(rng):
    m = rng.choice([1, 2, 3, 4, 5, 6, 9])
    n = m + rng.choice([2, 3, 4])
    cc = rng.choice([2, 3, 4, 5])
    e = m + 1 - n                              # c^{m+1-n}(負)
    c = Num(F(cc) ** e)
    integral = rf"\int_0^\infty\frac{{{pw('x', m)}}}{{(x+{cc})^{{{n}}}}}dx"
    return m, n, cc, e, c, integral


def rational_solution(m, n, cc, e, c):
    return [rf"$x={cc}y$ とおくと $dx={cc}\,dy$, $x+{cc}={cc}(1+y)$。",
            rf"$$I=\int_0^\infty\frac{{{cc}^{{{m}}}y^{{{m}}}}}{{{cc}^{{{n}}}(1+y)^{{{n}}}}}{cc}\,dy={cc}^{{{e}}}\int_0^\infty\frac{{y^{{{m}}}}}{{(1+y)^{{{n}}}}}dy$$",
            rf"さらに $y=\dfrac{{u}}{{1-u}}$($u=\dfrac{{y}}{{1+y}}$)とおくと $\displaystyle\int_0^\infty\frac{{y^{{p-1}}}}{{(1+y)^{{p+q}}}}dy=B(p,q)$ なので "
            rf"$p={m + 1},\ q={n - m - 1}$。"]


def g_beta_rational(rng):
    m, n, cc, e, c, integral = rational_parts(rng)
    p, q = m + 1, n - m - 1
    cands = [(ONE, p, q, rf"$x={cc}y$ と置換したときの係数 ${cc}^{{{e}}}$ を忘れていないか。"),
             (Num(F(cc) ** (e - 1)), p, q, rf"$dx={cc}\,dy$ の ${cc}$ もかける。"),
             (Num(F(cc) ** (-e)), p, q, "係数の指数の符号を確かめよう(分母の方が次数が高い)。"),
             (c, m, n - m, r"$\frac{y^{p-1}}{(1+y)^{p+q}}$ と見比べる。$p-1=m$, $p+q=n$。")]
    return beta_problem("ベータ関数で表す(無限区間の分数式)", "標準", integral, c, p, q, cands,
                        [r"$\displaystyle\int_0^\infty\frac{y^{p-1}}{(1+y)^{p+q}}dy=B(p,q)$ を使う。まず $x=cy$ と置換する。"],
                        rational_solution(m, n, cc, e, c),
                        dict(f=lambda x: x ** m / (x + cc) ** n, a=0, b=math.inf))


def logbeta_parts(rng):
    m = rng.choice([1, 2, 3, 4])
    k = rng.choice([1, 2, 3, 5])
    r = F(k, 2)
    if r == m:
        raise Retry
    lg = r"\log x"
    rt = (rf"\sqrt{{{lg}}}" if k == 1 else rf"\sqrt{{{lg}}}^{{{k}}}") if k % 2 else (lg if k == 2 else rf"({lg})^{{{k // 2}}}")
    integral = rf"\int_1^e\frac{{(1-\log x)^{{{m}}}{rt}}}{{x}}dx"
    return m, r, integral


def g_beta_log(rng):
    m, r, integral = logbeta_parts(rng)
    cands = [(ONE, r, m, r"$t^{p-1}(1-t)^{q-1}$ と見比べる。指数に $1$ を足す。"),
             (ONE, r + 1, m, "$(1-t)$ の指数にも $1$ を足す。"),
             (ONE, r, m + 1, "$t$ の指数にも $1$ を足す。"),
             (Num(1, 1, 0), r + 2, m + 1, "$t=\\log x$ のとき $\\frac{dx}{x}=dt$。指数はそのまま。")]
    sol = [r"$t=\log x$ とおくと $\dfrac{dx}{x}=dt$, $x:1\to e$ で $t:0\to1$。",
           rf"$$I=\int_0^1{pw('t', r)}{pw('(1-t)', m)}dt$$"]
    return beta_problem("ベータ関数で表す(対数)", "標準", integral, ONE, r + 1, m + 1, cands,
                        [r"$t=\log x$ と置換する。"], sol,
                        dict(f=lambda x: (1 - math.log(x)) ** m * math.log(x) ** float(r) / x, a=1, b=math.e))


def trig_parts(rng):
    opts = [F(-1, 2), F(1, 2), F(1), F(2), F(3), F(4), F(5), F(3, 2)]
    a, b = rng.sample(opts, 2)
    if a < 0 and b < 0:
        raise Retry

    def part(fn, e):
        if e == 1:
            return rf"\{fn} x"
        if e.denominator == 1:
            return rf"\{fn}^{{{e}}}x"
        if e == F(1, 2):
            return rf"\sqrt{{\{fn} x}}"
        return rf"\sqrt{{\{fn} x}}^{{{e.numerator}}}"
    num = [part(f, e) for f, e in (("sin", a), ("cos", b)) if e > 0]
    den = [part(f, -e) for f, e in (("sin", a), ("cos", b)) if e < 0]
    body = "".join(num) if num else "1"
    expr = rf"\frac{{{body}}}{{{''.join(den)}}}" if den else body
    integral = rf"\int_0^{{\frac{{\pi}}{{2}}}}{expr}\,dx"
    return a, b, integral


def g_beta_trig(rng):
    a, b, integral = trig_parts(rng)
    p, q = (a + 1) / 2, (b + 1) / 2
    half = Num(F(1, 2))
    cands = [(ONE, p, q, r"公式は $\frac12B\left(\frac{a+1}{2},\frac{b+1}{2}\right)$。$\frac12$ を忘れていないか。"),
             (half, a + 1, b + 1, r"$B$ の中は $\frac{a+1}{2}$, $\frac{b+1}{2}$($2$ で割る)。"),
             (half, a / 2, b / 2, r"$B$ の中は $\frac{a+1}{2}$。$1$ を足してから $2$ で割る。"),
             (half, p, (b + 2) / 2, r"$B$ の中は $\frac{b+1}{2}$。")]

    def ex(e):
        return fr(e) if e.denominator != 1 else str(e)
    sol = [rf"$\sin^ax\cos^bx$ で $a={ex(a)},\ b={ex(b)}$ とすると、公式 "
           r"$\displaystyle\int_0^{\frac\pi2}\sin^ax\cos^bx\,dx=\frac12B\left(\frac{a+1}{2},\frac{b+1}{2}\right)$ より"]
    return beta_problem("ベータ関数で表す(三角関数)", "基本", integral, half, p, q, cands,
                        [r"$\displaystyle\int_0^{\frac\pi2}\sin^ax\cos^bx\,dx=\frac12B\left(\frac{a+1}{2},\frac{b+1}{2}\right)$。"
                         r"$\frac{1}{\sqrt{\sin x}}=\sin^{-\frac12}x$。"], sol,
                        dict(f=lambda x: math.sin(x) ** float(a) * math.cos(x) ** float(b), a=0, b=math.pi / 2))


# ------------------------------------------------------------
# 誘導問題(5点×2):(1) ベータ関数で表す (2) ガンマ関数を使って値を求める
# ------------------------------------------------------------
def _guided(title, integral, c, p, q, sol1, meta):
    vl = value_line(c, p, q)
    if vl is None:
        raise Retry
    t = coef_beta(c, p, q)[0]
    gp, gq, gs = (fr(p), fr(q), fr(F(p) + F(q)))
    sol2 = [rf"$$B\left({gp},{gq}\right)=\frac{{\Gamma\left({gp}\right)\Gamma\left({gq}\right)}}{{\Gamma\left({gs}\right)}}"
            rf"={beta_exact(p, q).tex()}$$", rf"$$I={vl}$$"]
    return ExamItem(title, rf"積分 $I=\displaystyle{integral}$ について, 次の問いに答えよ.",
                    [Part(r"$I$ をベータ関数で表せ.", t, sol1 + [rf"$$I={t}$$"]),
                     Part(r"$B(p,q)=\dfrac{\Gamma(p)\Gamma(q)}{\Gamma(p+q)}$ を用いて, $I$ の値を求めよ.", vl, sol2)],
                    meta=dict(meta, c=c, p=p, q=q, value=vl))


def guided_interval(rng):
    a, b, L, m, r, integral, c = interval_parts(rng)
    return _guided("ベータ関数(区間 [a, b])", integral, c, r + 1, m + 1, interval_solution(a, b, L, m, r, c),
                   dict(f=lambda x: (b - x) ** m * (x - a) ** float(r), a=a, b=b))


def guided_semi(rng):
    A, n, integral, c, p = semi_parts(rng)
    return _guided("ベータ関数(半円型)", integral, c, p, p, semi_solution(A, n, c, p),
                   dict(f=lambda x: max(A * A - x * x, 0.0) ** (n / 2), a=-A, b=A))


def guided_rational(rng):
    m, n, cc, e, c, integral = rational_parts(rng)
    return _guided("ベータ関数(無限区間の分数式)", integral, c, m + 1, n - m - 1, rational_solution(m, n, cc, e, c),
                   dict(f=lambda x: x ** m / (x + cc) ** n, a=0, b=math.inf))


REGISTRY = [
    ProblemType("sp_gamma", "ガンマ関数の値", "基本", g_gamma_half),
    ProblemType("sp_bpoly", "ベータ関数で表す(基本形)", "基本", g_beta_poly),
    ProblemType("sp_btrig", "ベータ関数で表す(三角関数)", "基本", g_beta_trig),
    ProblemType("sp_gshift", "ガンマ関数(平行移動)", "標準", g_gamma_shift),
    ProblemType("sp_gauss", "ガウス型の積分", "標準", g_gauss),
    ProblemType("sp_glog", "対数を含む積分", "標準", g_log_gamma),
    ProblemType("sp_binterval", "ベータ関数で表す(区間 [a, b])", "標準", g_beta_interval, guided=guided_interval),
    ProblemType("sp_bsemi", "ベータ関数で表す(半円型)", "標準", g_beta_semi, guided=guided_semi),
    ProblemType("sp_brational", "ベータ関数で表す(無限区間の分数式)", "標準", g_beta_rational, guided=guided_rational),
    ProblemType("sp_blog", "ベータ関数で表す(対数)", "標準", g_beta_log),
]
GUIDED_NAMES = {"sp_binterval": "ベータ関数で表す → 値(区間 [a, b])", "sp_bsemi": "ベータ関数で表す → 値(半円型)",
                "sp_brational": "ベータ関数で表す → 値(無限区間の分数式)"}


def tab():
    import streamlit as st
    with st.expander("使う定義と公式", expanded=False):
        st.markdown(DEFS)
        st.markdown(FORMULAS)
    render("special", REGISTRY, "特殊関数 ガンマ関数・ベータ関数")
