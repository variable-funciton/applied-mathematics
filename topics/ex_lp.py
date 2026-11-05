"""数学IV フーリエ解析 発展 関数空間 L² と Lᵖ:演習の問題生成

規約:区間 [−π, π] で ⟨f, g⟩ = ∫ f \\bar g dx,  c_n = (1/2π) ∫ f e^{−inx} dx。
"""
from fractions import Fraction

from topics.ex_core import Choice, ExamItem, Part, Problem, ProblemType, Retry, render, sqrt_parts, tex_fr, tex_frac_root

F = Fraction
IP = r"$\langle f,g\rangle=\int_{-\pi}^{\pi}f(x)\overline{g(x)}\,dx$, $\langle e^{imx},e^{inx}\rangle=\begin{cases}2\pi&(m=n)\\0&(m\ne n)\end{cases}$。"


def distinct(ok, cands):
    choices = [Choice(ok, True, value=ok)]
    for t, why in cands:
        if all(c.tex != t for c in choices) and len(choices) < 4:
            choices.append(Choice(t, why=why, value=t))
    if len(choices) < 4:
        raise Retry
    return choices


def cfrac(q, extra=""):
    """有理数 q に記号 extra(i, \\pi など)をつけた形:−\\frac{i}{2}, 2\\pi i など"""
    q = F(q)
    if q == 0:
        return "0"
    s = "-" if q < 0 else ""
    n, d = abs(q.numerator), q.denominator
    top = extra if (n == 1 and extra) else f"{n}{extra}"
    return s + (top if d == 1 else rf"\frac{{{top}}}{{{d}}}")


def cnum(re, im):
    """整数の複素数 re + im i"""
    if im == 0:
        return str(re)
    it = "i" if abs(im) == 1 else f"{abs(im)}i"
    if re == 0:
        return ("-" if im < 0 else "") + it
    return f"{re}{'+' if im > 0 else '-'}{it}"


def twopi(re, im, k="2"):
    """k\\pi (re + im i)"""
    if im == 0:
        return cfrac(F(int(k) * re), r"\pi")
    return rf"{k}\pi({cnum(re, im)})"


def sq(n, extra=""):
    """\\sqrt{n·extra} を簡単にした形"""
    k, m = sqrt_parts(n)
    inner = (str(m) if m > 1 else "") + extra
    root = rf"\sqrt{{{inner}}}" if inner else ""
    return (str(k) if k > 1 or not root else "") + root


def ex_(n):
    return "1" if n == 0 else ("e^{ix}" if n == 1 else f"e^{{{n}ix}}")


def lin(a, s, b, t):
    """a·s + b·t(係数 ±1 を省く)"""
    def term(c, e, first):
        if e == "1":
            body = str(abs(c))
        else:
            body = (str(abs(c)) if abs(c) != 1 else "") + e
        sign = ("-" if c < 0 else "") if first else ("-" if c < 0 else "+")
        return sign + body
    return term(a, s, True) + term(b, t, False)


# ------------------------------------------------------------
# 基本:内積の計算(共役に注意)
# ------------------------------------------------------------
def g_ip(rng):
    n = rng.choice([1, 2, 3])
    a, c = rng.choice([1, 2, 3]), rng.choice([1, 2, 3])
    b, d = rng.choice([1, 2, 3, -1, -2]), rng.choice([1, 2, 3])
    u = rng.choice(["1", "i"])
    e = ex_(n)
    f = lin(a, "1", b, e)
    dc = ("" if d == 1 else str(d)) + ("i" if u == "i" else "")
    gs = f"{c}+{dc}{e}"
    if u == "i":
        re, im = a * c, -b * d
        cands = [(twopi(a * c, b * d), r"2番目の関数は共役をとる:$\langle f,ig\rangle=\overline{i}\langle f,g\rangle=-i\langle f,g\rangle$。"),
                 (cnum(re, im), r"$\langle e^{inx},e^{inx}\rangle=2\pi$ をかけ忘れていないか。"),
                 (rf"\pi({cnum(re, im)})", r"$\int_{-\pi}^{\pi}dx=2\pi$。"),
                 (twopi(a * c + a * d, im + b * c), r"異なる $n$ の $e^{inx}$ どうしは直交する(内積 0)。")]
    else:
        re, im = a * c + b * d, 0
        cands = [(str(a * c + b * d), r"$\langle e^{inx},e^{inx}\rangle=2\pi$ をかけ忘れていないか。"),
                 (cfrac(F(a * c + b * d), r"\pi"), r"$\int_{-\pi}^{\pi}dx=2\pi$。"),
                 (cfrac(F(2 * a * c), r"\pi"), r"$\langle e^{inx},e^{inx}\rangle=2\pi\ne0$:同じ $n$ どうしの内積は残る。"),
                 (cfrac(F(2 * (a * c + b * d + a * d + b * c)), r"\pi"), r"$1$ と $e^{inx}$ は直交する($n\ne0$)。")]
    ok = twopi(re, im)
    return Problem("内積の計算", "基本",
                   rf"区間 $[-\pi,\pi]$ の内積 $\langle f,g\rangle=\displaystyle\int_{{-\pi}}^{{\pi}}f(x)\overline{{g(x)}}\,dx$ について, "
                   rf"$f(x)={f}$, $g(x)={gs}$ のとき $\langle f,g\rangle$ を求めよ。",
                   distinct(ok, cands), hints=[IP, r"$\langle f,cg\rangle=\bar c\langle f,g\rangle$(2番目の関数の係数は共役をとって外に出す)。"],
                   solution=[rf"$1$ と ${e}$ は直交し, $\langle 1,1\rangle=\langle {e},{e}\rangle=2\pi$。",
                             rf"$\langle f,g\rangle={a}\cdot\overline{{{c}}}\cdot2\pi+({b})\cdot\overline{{{dc if dc else 1}}}\cdot2\pi={ok}$。"],
                   meta=dict(n=n, a=a, b=b, c=c, d=d, u=u, re=re, im=im))


# ------------------------------------------------------------
# 基本:ノルムの計算
# ------------------------------------------------------------
def g_norm(rng):
    kind = rng.choice(["trig", "poly"])
    if kind == "trig":
        m, n = rng.sample([0, 1, 2, 3], 2)
        a, b = rng.choice([1, 2, 3]), rng.choice([1, 2, -1, -2, 3])
        f = lin(a, ex_(m), b, ex_(n))
        s = a * a + b * b
        ok = sq(2 * s, r"\pi")
        cands = [(sq(2 * (abs(a) + abs(b)) ** 2, r"\pi"), r"$\|f+g\|\ne\|f\|+\|g\|$。直交するので2乗が足し算になる。"),
                 (sq(s, r"\pi"), r"$\|e^{inx}\|_2^2=2\pi$。"), (cfrac(F(2 * s), r"\pi"), r"ノルムは2乗の積分の平方根。"),
                 (sq(s), r"$\|e^{inx}\|_2^2=\int_{-\pi}^{\pi}1\,dx=2\pi$ をかける。")]
        return Problem("ノルムの計算", "基本",
                       rf"区間 $[-\pi,\pi]$ で $f(x)={f}$ の $\|f\|_2=\left(\displaystyle\int_{{-\pi}}^{{\pi}}|f(x)|^2dx\right)^{{1/2}}$ を求めよ。",
                       distinct(ok, cands), hints=[IP, r"直交する関数では $\|f+g\|_2^2=\|f\|_2^2+\|g\|_2^2$(三平方の定理)。"],
                       solution=[rf"${ex_(m)}$ と ${ex_(n)}$ は直交するので $\|f\|_2^2={a * a}\cdot2\pi+{b * b}\cdot2\pi={2 * s}\pi$。",
                                 rf"$\|f\|_2={ok}$。"], meta=dict(kind=kind, s=s))
    k, p, c = rng.choice([1, 2, 3]), rng.choice([1, 2]), rng.choice([1, 2, 3])
    ft = (str(c) if c > 1 else "") + ("x" if k == 1 else f"x^{k}")
    if p == 1:
        ok = tex_fr(F(c, k + 1))
        cands = [(tex_frac_root(c, 2 * k + 1, 2 * k + 1), r"これは $\|f\|_2$。$p=1$ では $\int_0^1|f|\,dx$。"),
                 (tex_fr(F(c, k)), r"$\int_0^1x^kdx=\frac{1}{k+1}$。"), (tex_fr(F(c * c, k + 1)), r"$p=1$ では2乗しない。"),
                 (tex_fr(F(c, 2 * k + 1)), r"$\int_0^1x^kdx=\frac{1}{k+1}$。")]
        sol = rf"$\|f\|_1=\int_0^1 {ft}\,dx={ok}$"
    else:
        ok = tex_frac_root(c, 2 * k + 1, 2 * k + 1)
        cands = [(tex_fr(F(c * c, 2 * k + 1)), r"$\|f\|_2^2$ の平方根をとる。"), (tex_fr(F(c, k + 1)), r"これは $\|f\|_1$。"),
                 (tex_fr(F(c, 2 * k + 1)), r"$\|f\|_2=\left(\int|f|^2\right)^{1/2}$:定数 $c$ も2乗してから平方根。"),
                 (tex_frac_root(c, k + 1, k + 1), r"$\int_0^1x^{2k}dx=\frac{1}{2k+1}$。")]
        sol = rf"$\|f\|_2^2=\int_0^1 {c * c if c > 1 else ''}x^{{{2 * k}}}dx={tex_fr(F(c * c, 2 * k + 1))}$ より $\|f\|_2={ok}$"
    return Problem("ノルムの計算", "基本",
                   rf"区間 $[0,1]$ で $f(x)={ft}$ の $L^{p}$ ノルム $\|f\|_{p}=\left(\displaystyle\int_0^1|f(x)|^{p}dx\right)^{{1/{p}}}$ を求めよ。"
                   if p == 2 else rf"区間 $[0,1]$ で $f(x)={ft}$ の $L^1$ ノルム $\|f\|_1=\displaystyle\int_0^1|f(x)|\,dx$ を求めよ。",
                   distinct(ok, cands), hints=[r"$\|f\|_p=\left(\int|f|^p\right)^{1/p}$。", r"$\int_0^1x^mdx=\frac{1}{m+1}$。"],
                   solution=[sol + "。"], meta=dict(kind=kind, k=k, p=p, c=c))


# ------------------------------------------------------------
# 標準:直交するように定数を決める
# ------------------------------------------------------------
def g_orth(rng):
    m, n = rng.choice([1, 2, 3]), rng.choice([0, 1, 2])
    cv = F(-(n + 1), m + n + 1)
    xm = "x" if m == 1 else f"x^{m}"
    xn = "1" if n == 0 else ("x" if n == 1 else f"x^{n}")
    cands = [(tex_fr(F(-1, m + n + 1)), r"$\int_0^1c\,x^ndx=\frac{c}{n+1}$:定数 $c$ の項にも積分がかかる。"),
             (tex_fr(F(-(m + 1), m + n + 1)), r"$\int_0^1x^m\cdot x^ndx=\frac{1}{m+n+1}$, $\int_0^1x^ndx=\frac{1}{n+1}$。"),
             (tex_fr(-cv), "符号を確かめよう。"), (tex_fr(F(-(n + 1), m + n + 2)), r"$\int_0^1x^{m+n}dx=\frac{1}{m+n+1}$。"),
             (tex_fr(F(-1, m + 1)), r"これは $1$ と直交する場合の値。")]
    return Problem("直交するように定数を決める", "標準",
                   rf"区間 $[0,1]$ の内積 $\langle f,g\rangle=\displaystyle\int_0^1f(x)g(x)\,dx$ について, "
                   rf"$f(x)={xm}+c$ と $g(x)={xn}$ が直交する($\langle f,g\rangle=0$)ような定数 $c$ を求めよ。",
                   distinct(tex_fr(cv), cands), hints=[r"$\langle f,g\rangle=\int_0^1x^{m+n}dx+c\int_0^1x^ndx$。", r"$\int_0^1x^kdx=\frac{1}{k+1}$。"],
                   solution=[rf"$\langle f,g\rangle=\frac{{1}}{{{m + n + 1}}}+{'c' if n == 0 else rf'\frac{{c}}{{{n + 1}}}'}=0$ より $c={tex_fr(cv)}$。"],
                   meta=dict(m=m, n=n, c=cv))


# ------------------------------------------------------------
# 標準:最良近似の係数(フーリエ係数 = 正射影)
# ------------------------------------------------------------
def coef_x(kind, n):
    """f = x: c_n = i(−1)^n / n,  f = x²: c_n = 2(−1)^n / n²(n ≠ 0)"""
    s = (-1) ** n
    return (F(s, n), "i") if kind == "x" else (F(2 * s, n * n), "")


def g_proj(rng):
    kind, n = rng.choice(["x", "x2"]), rng.choice([1, 2, 3])
    q, e = coef_x(kind, n)
    ok = cfrac(q, e)
    ft = "x" if kind == "x" else "x^2"
    if kind == "x":
        cands = [(cfrac(-q, e), r"$c_n$ は $e^{-inx}$ をかけて積分する(内積は2番目の関数の共役)。"),
                 (cfrac(2 * q, r"\pi i"), r"$\langle e^{inx},e^{inx}\rangle=2\pi$ で割る。"), (cfrac(q), r"$\int xe^{-inx}dx$ は純虚数になる。"),
                 (cfrac(F(1, n), e), r"$e^{\pm in\pi}=(-1)^n$ の符号を確かめよう。")]
    else:
        cands = [(cfrac(-q), r"$e^{\pm in\pi}=(-1)^n$ の符号を確かめよう。"), (cfrac(2 * q, r"\pi"), r"$\langle e^{inx},e^{inx}\rangle=2\pi$ で割る。"),
                 (cfrac(F(2 * (-1) ** n, n)), r"部分積分を2回する:分母は $n^2$。"), (cfrac(q, "i"), r"$x^2$ は偶関数なので $c_n$ は実数。")]
    en = rf"e^{{-{n if n > 1 else ''}ix}}"
    sol1 = (rf"部分積分より $\int_{{-\pi}}^{{\pi}}x{en}dx=\frac{{2\pi i(-1)^n}}{{n}}={cfrac(2 * q, r'\pi i')}$" if kind == "x"
            else rf"部分積分を2回して $\int_{{-\pi}}^{{\pi}}x^2{en}dx=\frac{{4\pi(-1)^n}}{{n^2}}={cfrac(2 * q, r'\pi')}$")
    return Problem("最良近似の係数", "標準",
                   rf"区間 $[-\pi,\pi]$ で, $f(x)={ft}$ を $a\,{ex_(n)}$($a$ は複素数)で近似する。"
                   rf"2乗平均誤差 $\|f-a\,{ex_(n)}\|_2$ を最小にする $a$ を求めよ。",
                   distinct(ok, cands), hints=[r"最小にする $a$ は正射影の係数 $a=\dfrac{\langle f,e^{inx}\rangle}{\langle e^{inx},e^{inx}\rangle}$(フーリエ係数 $c_n$)。",
                                               r"$c_n=\frac{1}{2\pi}\int_{-\pi}^{\pi}f(x)e^{-inx}dx$, $e^{\pm in\pi}=(-1)^n$。"],
                   solution=[r"$\|f-ae^{inx}\|_2^2=\|f\|_2^2-2\pi|c_n|^2+2\pi|a-c_n|^2$ なので, $a=c_n$ のとき最小。",
                             sol1 + rf"($n={n}$)。$2\pi$ で割って $a=c_{n}={ok}$。"], meta=dict(kind=kind, n=n, q=q, e=e))


# ------------------------------------------------------------
# 標準:x^{−a} が Lᵖ に入る範囲
# ------------------------------------------------------------
def g_int(rng):
    a = rng.choice([F(1, 2), F(1, 3), F(2, 3), F(1, 4), F(3, 4), F(2, 5)])
    dom = rng.choice(["0", "inf"])
    r = tex_fr(1 / a)
    near, far = rf"1\le p<{r}", rf"p>{r}"
    at = rf"x^{{-{a.numerator}/{a.denominator}}}"
    if dom == "0":
        ok = near
        cands = [(far, r"$0$ の近くでは, $p$ が大きいほど特異性が強くなる。"), (rf"1\le p\le{r}", rf"$p={r}$ では $\int_0^1x^{{-1}}dx=\infty$。"),
                 (r"p\ge1\ (\text{すべて})", r"$\int_0^1x^{-ap}dx<\infty\iff ap<1$。")]
        where, cond = r"0<x<1", r"\int_0^1x^{-ap}dx<\infty\iff ap<1"
    else:
        ok = far
        cands = [(near, r"遠くでは, $p$ が大きいほど $x^{-ap}$ は速く減衰する。"), (rf"p\ge{r}", rf"$p={r}$ では $\int_1^\infty x^{{-1}}dx=\infty$。"),
                 (r"p\ge1\ (\text{すべて})", r"$\int_1^\infty x^{-ap}dx<\infty\iff ap>1$。")]
        where, cond = r"x>1", r"\int_1^\infty x^{-ap}dx<\infty\iff ap>1"
    return Problem("Lᵖ に入る範囲", "標準",
                   rf"$f(x)={at}$(${where}$)が $L^p$ に入る($\int|f|^pdx<\infty$)ような $p\ (\ge1)$ の範囲を求めよ。",
                   distinct(ok, cands), hints=[r"$\int|f|^pdx=\int x^{-ap}dx$。", r"$\int_0^1x^{-s}dx<\infty\iff s<1$, $\int_1^\infty x^{-s}dx<\infty\iff s>1$。"],
                   solution=[rf"${cond}$, $a={tex_fr(a)}$ より ${ok}$。"], meta=dict(a=a, dom=dom))


# ------------------------------------------------------------
# 基本:中線定理
# ------------------------------------------------------------
def g_par(rng):
    kind = rng.choice(["minus", "re"])
    while True:
        a, b = rng.choice([1, 2, 3, 4]), rng.choice([1, 2, 3, 4])
        c = rng.randint(abs(a - b) + 1, a + b - 1) if a + b - 1 >= abs(a - b) + 1 else None
        if c is not None and c * c != a * a + b * b:
            break
    s = 2 * a * a + 2 * b * b - c * c
    if kind == "minus":
        ok = sq(s)
        cands = [(sq(a * a + b * b), r"$f$ と $g$ は直交するとは限らない。中線定理を使う。"), (str(s), "平方根をとる。"),
                 (sq(s + 2 * c * c) if s > 0 else "0", r"$\|f-g\|^2=2\|f\|^2+2\|g\|^2-\|f+g\|^2$:$\|f+g\|^2$ は引く。"),
                 (sq(2 * a * a + 2 * b * b - c), r"$\|f+g\|$ も2乗する。")]
        q = r"\|f-g\|"
        sol = [r"中線定理 $\|f+g\|^2+\|f-g\|^2=2\|f\|^2+2\|g\|^2$ より", rf"$\|f-g\|^2={2 * a * a}+{2 * b * b}-{c * c}={s}$, $\|f-g\|={ok}$。"]
    else:
        val = F(c * c - a * a - b * b, 2)
        ok = tex_fr(val)
        cands = [(tex_fr(val / 2), r"$\|f+g\|^2=\|f\|^2+2\mathrm{Re}\langle f,g\rangle+\|g\|^2$:$2$ で割る。"), (tex_fr(2 * val), r"展開すると $2\mathrm{Re}\langle f,g\rangle$ が出る。"),
                 (tex_fr(-val), "符号を確かめよう。"), (tex_fr(F(c - a - b, 2)), "ノルムは2乗してから展開する。")]
        q = r"\mathrm{Re}\langle f,g\rangle"
        sol = [r"$\|f+g\|^2=\langle f+g,f+g\rangle=\|f\|^2+2\mathrm{Re}\langle f,g\rangle+\|g\|^2$ より",
               rf"$\mathrm{{Re}}\langle f,g\rangle=\frac{{{c * c}-{a * a}-{b * b}}}{{2}}={ok}$。"]
    return Problem("中線定理", "基本",
                   rf"内積空間($L^2$ など)の元 $f,g$ が $\|f\|={a}$, $\|g\|={b}$, $\|f+g\|={c}$ をみたすとき, ${q}$ を求めよ。",
                   distinct(ok, cands), hints=[r"$\|f\pm g\|^2=\|f\|^2\pm2\mathrm{Re}\langle f,g\rangle+\|g\|^2$。",
                                               r"足し合わせると中線定理 $\|f+g\|^2+\|f-g\|^2=2\|f\|^2+2\|g\|^2$。"],
                   solution=sol, meta=dict(kind=kind, a=a, b=b, c=c))


# ------------------------------------------------------------
# 誘導
# ------------------------------------------------------------
def guided_pars(rng):
    kind = rng.choice(["x", "x2"])
    if kind == "x":
        return ExamItem("パーセバルの等式と級数の和",
                        r"区間 $[-\pi,\pi]$ で $f(x)=x$ の複素フーリエ係数を $c_n=\dfrac{1}{2\pi}\displaystyle\int_{-\pi}^{\pi}f(x)e^{-inx}dx$ とする. 次の問いに答えよ.",
                        [Part(r"$c_0$ と $c_n$($n\ne0$)を求めよ.", r"c_0=0,\ c_n=\frac{(-1)^ni}{n}",
                              [r"$c_0=\frac{1}{2\pi}\int_{-\pi}^{\pi}x\,dx=0$。",
                               r"部分積分:$\int_{-\pi}^{\pi}xe^{-inx}dx=\left[\frac{xe^{-inx}}{-in}\right]_{-\pi}^{\pi}+\frac{1}{in}\int_{-\pi}^{\pi}e^{-inx}dx=\frac{2\pi(-1)^n}{-in}=\frac{2\pi(-1)^ni}{n}$ より $c_n=\frac{(-1)^ni}{n}$。"]),
                         Part(r"パーセバルの等式 $\dfrac{1}{2\pi}\displaystyle\int_{-\pi}^{\pi}|f(x)|^2dx=\sum_{n=-\infty}^{\infty}|c_n|^2$ を用いて, $\displaystyle\sum_{n=1}^{\infty}\frac{1}{n^2}$ を求めよ.",
                              r"\frac{\pi^2}{6}",
                              [r"左辺 $=\frac{1}{2\pi}\int_{-\pi}^{\pi}x^2dx=\frac{\pi^2}{3}$, 右辺 $=\sum_{n\ne0}\frac{1}{n^2}=2\sum_{n=1}^\infty\frac{1}{n^2}$。",
                               r"よって $\sum_{n=1}^\infty\frac{1}{n^2}=\frac{\pi^2}{6}$。"])], meta=dict(kind=kind))
    return ExamItem("パーセバルの等式と級数の和",
                    r"区間 $[-\pi,\pi]$ で $f(x)=x^2$ の複素フーリエ係数を $c_n=\dfrac{1}{2\pi}\displaystyle\int_{-\pi}^{\pi}f(x)e^{-inx}dx$ とする. 次の問いに答えよ.",
                    [Part(r"$c_0$ と $c_n$($n\ne0$)を求めよ.", r"c_0=\frac{\pi^2}{3},\ c_n=\frac{2(-1)^n}{n^2}",
                          [r"$c_0=\frac{1}{2\pi}\int_{-\pi}^{\pi}x^2dx=\frac{\pi^2}{3}$。",
                           r"部分積分を2回:$\int_{-\pi}^{\pi}x^2e^{-inx}dx=\frac{2}{in}\int_{-\pi}^{\pi}xe^{-inx}dx=\frac{2}{in}\cdot\frac{2\pi(-1)^ni}{n}=\frac{4\pi(-1)^n}{n^2}$ より $c_n=\frac{2(-1)^n}{n^2}$。"]),
                     Part(r"パーセバルの等式 $\dfrac{1}{2\pi}\displaystyle\int_{-\pi}^{\pi}|f(x)|^2dx=\sum_{n=-\infty}^{\infty}|c_n|^2$ を用いて, $\displaystyle\sum_{n=1}^{\infty}\frac{1}{n^4}$ を求めよ.",
                          r"\frac{\pi^4}{90}",
                          [r"左辺 $=\frac{1}{2\pi}\int_{-\pi}^{\pi}x^4dx=\frac{\pi^4}{5}$, 右辺 $=\frac{\pi^4}{9}+2\sum_{n=1}^\infty\frac{4}{n^4}$。",
                           r"$8\sum_{n=1}^\infty\frac{1}{n^4}=\frac{\pi^4}{5}-\frac{\pi^4}{9}=\frac{4\pi^4}{45}$ より $\sum_{n=1}^\infty\frac{1}{n^4}=\frac{\pi^4}{90}$。"])],
                    meta=dict(kind=kind))


def guided_orth(rng):
    m = rng.choice([1, 2, 3])
    xm = "x" if m == 1 else f"x^{m}"
    cv = F(-1, m + 1)
    nrm = tex_frac_root(m, (m + 1) * (2 * m + 1), 2 * m + 1)
    return ExamItem("直交化とノルム",
                    rf"区間 $[0,1]$ の内積を $\langle f,g\rangle=\displaystyle\int_0^1f(x)g(x)\,dx$, ノルムを $\|f\|=\sqrt{{\langle f,f\rangle}}$ とする. "
                    rf"$h(x)={xm}+c$($c$ は定数)について, 次の問いに答えよ.",
                    [Part(r"$h$ が定数関数 $1$ と直交するような $c$ を求めよ.", rf"c={tex_fr(cv)}",
                          [rf"$\langle h,1\rangle=\int_0^1({xm}+c)dx=\frac{{1}}{{{m + 1}}}+c=0$ より $c={tex_fr(cv)}$。"]),
                     Part(r"(1) の $c$ に対し, $\|h\|$ を求めよ.", rf"\|h\|={nrm}",
                          [rf"$\|h\|^2=\int_0^1\left({xm}-\frac{{1}}{{{m + 1}}}\right)^2dx=\frac{{1}}{{{2 * m + 1}}}-\frac{{2}}{{{(m + 1) ** 2}}}+\frac{{1}}{{{(m + 1) ** 2}}}"
                           rf"={tex_fr(F(m * m, (2 * m + 1) * (m + 1) ** 2))}$。",
                           rf"よって $\|h\|={nrm}$。"])], meta=dict(m=m))


def guided_par(rng):
    L = rng.choice([1, 2])
    h = tex_fr(F(L, 2))
    if L == 1:
        nf, nfg, lhs, rhs = r"2^{-1/p}", "1", "2", r"4\cdot2^{-2/p}"
    else:
        nf, nfg, lhs, rhs = "1", r"2^{1/p}", r"2\cdot2^{2/p}", "4"
    return ExamItem("Lᵖ ノルムと中線定理",
                    rf"$1\le p<\infty$ とし, 区間 $[0,{L}]$ で $\|f\|_p=\left(\displaystyle\int_0^{{{L}}}|f(x)|^pdx\right)^{{1/p}}$ とする. "
                    rf"$f=\chi_{{[0,{h}]}}$, $g=\chi_{{[{h},{L}]}}$($\chi_A$ は $A$ で $1$, それ以外で $0$ の関数)について, 次の問いに答えよ.",
                    [Part(r"$\|f\|_p$, $\|g\|_p$, $\|f+g\|_p$, $\|f-g\|_p$ を求めよ.",
                          rf"\|f\|_p=\|g\|_p={nf},\ \|f+g\|_p=\|f-g\|_p={nfg}",
                          [rf"$\|f\|_p^p=\int_0^{{{h}}}1\,dx={h}$ より $\|f\|_p={nf}$($g$ も同じ)。",
                           rf"$|f\pm g|=1$(区間全体)より $\|f\pm g\|_p^p={L}$, $\|f\pm g\|_p={nfg}$。"]),
                     Part(r"中線定理 $\|f+g\|_p^2+\|f-g\|_p^2=2\|f\|_p^2+2\|g\|_p^2$ が成り立つような $p$ を求めよ.", "p=2",
                          [rf"左辺 $={lhs}$, 右辺 $={rhs}$。",
                           r"両辺が等しいのは $2^{2/p}=2$, すなわち $p=2$ のときだけ(内積から作られるノルムは $p=2$ のときだけ)。"])],
                    meta=dict(L=L))


REGISTRY = [
    ProblemType("lp_ip", "内積の計算", "基本", g_ip),
    ProblemType("lp_norm", "ノルムの計算", "基本", g_norm),
    ProblemType("lp_par", "中線定理", "基本", g_par, guided=guided_par),
    ProblemType("lp_orth", "直交するように定数を決める", "標準", g_orth, guided=guided_orth),
    ProblemType("lp_proj", "最良近似の係数", "標準", g_proj, guided=guided_pars),
    ProblemType("lp_int", "Lᵖ に入る範囲", "標準", g_int),
]
GUIDED_NAMES = {"lp_par": "Lᵖ ノルムの計算 → 中線定理が成り立つ p",
                "lp_orth": "直交するように定数を決める → ノルム",
                "lp_proj": "フーリエ係数 → パーセバルの等式で Σ1/n² や Σ1/n⁴"}


def tab():
    render("lp", REGISTRY, "フーリエ解析 発展 関数空間 L² と Lᵖ")
