"""数学IV ベクトル解析 第2章 スカラー場とベクトル場:演習の問題生成

スカラー場・ベクトル場の成分は「単項式 c·x^a y^b z^c の和」で表す(厳密な整数計算ができる)。
"""
from fractions import Fraction
from math import gcd

from topics.ex_core import (Choice, ExamItem, Part, Problem, ProblemType, Retry, lin_tex, render, sqrt_parts,
                            tex_fr, tex_root, vtex)

V = "xyz"


# ------------------------------------------------------------
# 多項式(単項式の和)の道具
# ------------------------------------------------------------
def mono(e):
    return "".join(v + (f"^{{{k}}}" if k > 1 else "") for v, k in zip(V, e) if k > 0)


def ptex(terms):
    """[(c, (a,b,c)), ...] → 3x^2y-z+2"""
    const = sum(c for c, e in terms if not any(e))
    return lin_tex([(c, mono(e)) for c, e in terms if any(e)], const)


def pd(terms, i):
    """x_i で偏微分"""
    out = []
    for c, e in terms:
        if e[i] > 0:
            e2 = list(e)
            e2[i] -= 1
            out.append((c * e[i], tuple(e2)))
    return out


def pd_nocoef(terms, i):
    """よくある間違い:指数を前に出し忘れる"""
    return [(c, tuple(k - (j == i) for j, k in enumerate(e))) for c, e in terms if e[i] > 0]


def pd_dropvars(terms, i):
    """よくある間違い:他の変数を定数として残さず消してしまう"""
    out = []
    for c, e in terms:
        if e[i] > 0:
            e2 = [0, 0, 0]
            e2[i] = e[i] - 1
            out.append((c * e[i], tuple(e2)))
    return out


def ev(terms, P):
    return sum(c * P[0] ** e[0] * P[1] ** e[1] * P[2] ** e[2] for c, e in terms)


def rand_terms(rng, n=3, maxdeg=3, coef=(-3, 3), mixed=True):
    while True:
        terms = {}
        for _ in range(n):
            deg = rng.randint(1, maxdeg)
            e = [0, 0, 0]
            for _ in range(deg):
                e[rng.randrange(3)] += 1
            c = rng.choice([k for k in range(coef[0], coef[1] + 1) if k != 0])
            terms[tuple(e)] = terms.get(tuple(e), 0) + c
        terms = [(c, e) for e, c in terms.items() if c != 0]
        if len(terms) >= 2 and (not mixed or any(sum(k > 0 for k in e) >= 2 for _, e in terms)):
            return terms


def rand_point(rng, vals=(-2, -1, 1, 2)):
    return tuple(rng.choice(vals) for _ in range(3))


def grad_at(terms, P):
    return [ev(pd(terms, i), P) for i in range(3)]


def pt(P):
    return "(" + r",\ ".join(str(x) for x in P) + ")"


def grad_steps(terms, P, name="f"):
    g = grad_at(terms, P)
    return ([rf"$\dfrac{{\partial {name}}}{{\partial x}}={ptex(pd(terms, 0)) if pd(terms, 0) else '0'},\quad "
             rf"\dfrac{{\partial {name}}}{{\partial y}}={ptex(pd(terms, 1)) if pd(terms, 1) else '0'},\quad "
             rf"\dfrac{{\partial {name}}}{{\partial z}}={ptex(pd(terms, 2)) if pd(terms, 2) else '0'}$",
             rf"点 $\mathrm{{P}}{pt(P)}$ を代入して $\nabla {name}(\mathrm{{P}})={vtex(g)}$"], g)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def frac_root(p, N):
    """p / √N を簡単にした形(有理化)"""
    from topics.ex_core import tex_frac_root
    return tex_frac_root(p, N, N)


GRAD_HINTS = [r"$\nabla f=\left(\dfrac{\partial f}{\partial x},\ \dfrac{\partial f}{\partial y},\ \dfrac{\partial f}{\partial z}\right)$。",
              r"x で偏微分するときは、y, z を定数とみなす(消さずに残す)。"]


# ============================================================
# 基本1:勾配
# ============================================================
def g_grad(rng):
    f = rand_terms(rng)
    P = rand_point(rng)
    steps, g = grad_steps(f, P)
    if sum(x != 0 for x in g) < 2:
        raise Retry
    bad1 = [ev(pd_nocoef(f, i), P) for i in range(3)]
    bad2 = [ev(pd_dropvars(f, i), P) for i in range(3)]
    choices = [Choice(vtex(g), True, value=g),
               Choice(vtex(bad1), why="$x^n$ を偏微分すると $nx^{n-1}$。指数を前に出し忘れていないか。"),
               Choice(vtex(bad2), why="x で偏微分するとき、y, z は定数として残す。消してしまっていないか。"),
               Choice(vtex([g[1], g[2], g[0]]), why="成分の順番は $(f_x,\\ f_y,\\ f_z)$。")]
    return Problem("勾配", "基本",
                   rf"スカラー場 $f(x,y,z)={ptex(f)}$ の、点 $\mathrm{{P}}{pt(P)}$ における勾配 $\nabla f$ を求めよ。",
                   choices, hints=GRAD_HINTS, solution=steps, meta=dict(f=f, P=P))


# ============================================================
# 基本2:方向微分係数
# ============================================================
NICE = [(1, 2, 2), (2, 1, 2), (2, 2, 1), (2, 3, 6), (3, 2, 6), (6, 2, 3), (2, 6, 3), (3, 4, 0), (0, 3, 4), (4, 0, 3),
        (1, 4, 8), (4, 4, 7), (1, 0, 0), (0, 1, 0)]


def nice_dir(rng):
    v = list(rng.choice(NICE))
    rng.shuffle(v)
    v = [x * rng.choice([1, -1]) for x in v]
    n = round(sum(x * x for x in v) ** 0.5)
    return v, n


def g_directional(rng):
    f = rand_terms(rng)
    P = rand_point(rng)
    steps, g = grad_steps(f, P)
    v, n = nice_dir(rng)
    if n == 1 or dot(g, v) == 0 or sum(x != 0 for x in g) < 2:
        raise Retry
    D = Fraction(dot(g, v), n)
    N = dot(g, g)
    choices = [Choice(tex_fr(D), True, value=float(D)),
               Choice(tex_fr(dot(g, v)), why=f"方向ベクトルは単位ベクトルにしてから内積をとる($|\\vec v|={n}$ で割る)。"),
               Choice(tex_root(N), why="それは $|\\nabla f|$(方向微分係数の最大値)。指定された向きとの内積を計算しよう。"),
               Choice(tex_fr(Fraction(dot(g, v), n * n)), why="単位ベクトルにするときは $|\\vec v|$ で割る($|\\vec v|^2$ ではない)。")]
    return Problem(
        "方向微分係数", "基本",
        rf"スカラー場 $f(x,y,z)={ptex(f)}$ の、点 $\mathrm{{P}}{pt(P)}$ における "
        rf"$\vec v={vtex(v)}$ 方向の方向微分係数を求めよ。",
        choices,
        hints=[r"方向微分係数は $D_{\vec u}f=\nabla f\cdot\vec u$($\vec u$ は単位ベクトル)。",
               rf"$\vec u=\dfrac{{\vec v}}{{|\vec v|}}$、$|\vec v|={n}$。"],
        solution=steps + [rf"$\vec u=\dfrac{{1}}{{{n}}}{vtex(v)}$",
                          rf"$$D_{{\vec u}}f(\mathrm P)=\nabla f\cdot\vec u=\frac{{1}}{{{n}}}\left\{{"
                          + "+".join(f"({a})({b})" for a, b in zip(g, v)) + rf"\right\}}={tex_fr(D)}$$"],
        meta=dict(f=f, P=P, v=v))


def guided_grad_dir(rng):
    p = g_directional(rng)
    f, P, v = p.meta["f"], p.meta["P"], p.meta["v"]
    steps, g = grad_steps(f, P)
    n = round(dot(v, v) ** 0.5)
    D = Fraction(dot(g, v), n)
    return ExamItem("勾配と方向微分係数", rf"スカラー場 $f(x,y,z)={ptex(f)}$ と点 $\mathrm{{P}}{pt(P)}$ について, 次の問いに答えよ.",
                    [Part(r"点 P における勾配 $\nabla f(\mathrm P)$ を求めよ.", vtex(g), steps),
                     Part(rf"点 P における $\vec v={vtex(v)}$ 方向の方向微分係数を求めよ.", tex_fr(D),
                          p.solution[2:])], meta=p.meta)


# ============================================================
# 基本3:発散
# ============================================================
def rand_field(rng):
    return [rand_terms(rng, n=2, maxdeg=3, mixed=False) for _ in range(3)]


def ftex(F):
    return r"\vec F=\left(" + r",\ ".join(ptex(c) for c in F) + r"\right)"


def div_at(F, P):
    return sum(ev(pd(F[i], i), P) for i in range(3))


def simplify(terms):
    """同類項をまとめる"""
    acc = {}
    for c, e in terms:
        acc[e] = acc.get(e, 0) + c
    return [(c, e) for e, c in acc.items() if c != 0]


def curl_terms(F):
    def sub(a, b):
        return simplify(a + [(-c, e) for c, e in b])
    return [sub(pd(F[2], 1), pd(F[1], 2)), sub(pd(F[0], 2), pd(F[2], 0)), sub(pd(F[1], 0), pd(F[0], 1))]


def curl_at(F, P):
    return [ev(c, P) for c in curl_terms(F)]


def div_steps(F, P):
    parts = [ptex(pd(F[i], i)) if pd(F[i], i) else "0" for i in range(3)]
    vals = [ev(pd(F[i], i), P) for i in range(3)]
    return [rf"$$\nabla\cdot\vec F=\frac{{\partial F_1}}{{\partial x}}+\frac{{\partial F_2}}{{\partial y}}+\frac{{\partial F_3}}{{\partial z}}"
            rf"=({parts[0]})+({parts[1]})+({parts[2]})$$",
            rf"点 $\mathrm P{pt(P)}$ で $({vals[0]})+({vals[1]})+({vals[2]})={sum(vals)}$"]


def g_div(rng):
    F = rand_field(rng)
    P = rand_point(rng)
    d = div_at(F, P)
    vals = [ev(pd(F[i], i), P) for i in range(3)]
    if sum(v != 0 for v in vals) < 2:
        raise Retry
    wrongvar = ev(pd(F[0], 1), P) + ev(pd(F[1], 2), P) + ev(pd(F[2], 0), P)
    choices = [Choice(str(d), True, value=d),
               Choice(str(wrongvar), why="発散は $\\frac{\\partial F_1}{\\partial x}+\\frac{\\partial F_2}{\\partial y}"
                                         "+\\frac{\\partial F_3}{\\partial z}$。第 i 成分は第 i 変数で偏微分する。"),
               Choice(str(vals[0] + vals[1]), why="z 成分の偏微分 $\\frac{\\partial F_3}{\\partial z}$ を足し忘れていないか。"),
               Choice(str(vals[0] - vals[1] + vals[2]), why="発散は3つの偏微分の和(引き算はない)。")]
    for extra in (d + 1, d - 1, -d, d + 2):
        choices.append(Choice(str(extra), why="偏微分の計算を確かめよう。"))
    choices = _uniq(choices)
    return Problem("発散 div", "基本",
                   rf"ベクトル場 ${ftex(F)}$ の、点 $\mathrm P{pt(P)}$ における発散 $\nabla\cdot\vec F$ を求めよ。",
                   choices,
                   hints=[r"$\nabla\cdot\vec F=\dfrac{\partial F_1}{\partial x}+\dfrac{\partial F_2}{\partial y}+\dfrac{\partial F_3}{\partial z}$(スカラー)。"],
                   solution=div_steps(F, P), meta=dict(F=F, P=P))


def _uniq(choices):
    out, seen = [], set()
    for c in choices:
        if c.tex not in seen:
            out.append(c)
            seen.add(c.tex)
        if len(out) == 4:
            break
    return out


# ============================================================
# 基本4:回転
# ============================================================
def curl_steps(F, P):
    C = curl_terms(F)
    r = curl_at(F, P)
    return [r"$$\nabla\times\vec F=\begin{vmatrix}\vec i&\vec j&\vec k\\\dfrac{\partial}{\partial x}&\dfrac{\partial}{\partial y}&"
            r"\dfrac{\partial}{\partial z}\\F_1&F_2&F_3\end{vmatrix}=\left(\frac{\partial F_3}{\partial y}-\frac{\partial F_2}{\partial z},\ "
            r"\frac{\partial F_1}{\partial z}-\frac{\partial F_3}{\partial x},\ \frac{\partial F_2}{\partial x}-\frac{\partial F_1}{\partial y}\right)$$",
            r"$$=\left(" + r",\ ".join(ptex(c) if c else "0" for c in C) + r"\right)$$",
            rf"点 $\mathrm P{pt(P)}$ で $\nabla\times\vec F={vtex(r)}$"]


def g_curl(rng):
    F = rand_field(rng)
    P = rand_point(rng)
    r = curl_at(F, P)
    if sum(x != 0 for x in r) < 2 or r[1] == 0:
        raise Retry
    diag = [ev(pd(F[i], i), P) for i in range(3)]
    choices = [Choice(vtex(r), True, value=r),
               Choice(vtex([-x for x in r]), why="各成分の引き算の順番が逆。例えば x 成分は "
                                                "$\\frac{\\partial F_3}{\\partial y}-\\frac{\\partial F_2}{\\partial z}$。"),
               Choice(vtex([r[0], -r[1], r[2]]), why="行列式を展開するとき、j 成分(第2成分)の符号に注意。"
                                                    "$\\frac{\\partial F_1}{\\partial z}-\\frac{\\partial F_3}{\\partial x}$。"),
               Choice(vtex(diag), why="それは $\\left(\\frac{\\partial F_1}{\\partial x},\\ \\frac{\\partial F_2}{\\partial y},\\ "
                                      "\\frac{\\partial F_3}{\\partial z}\\right)$。回転は外積の形で計算する。")]
    return Problem("回転 rot", "基本",
                   rf"ベクトル場 ${ftex(F)}$ の、点 $\mathrm P{pt(P)}$ における回転 $\nabla\times\vec F$ を求めよ。",
                   choices,
                   hints=[r"$\nabla\times\vec F$ は、行列式 $\begin{vmatrix}\vec i&\vec j&\vec k\\\partial_x&\partial_y&\partial_z\\F_1&F_2&F_3\end{vmatrix}$ を展開する。",
                          r"第2成分は $\dfrac{\partial F_1}{\partial z}-\dfrac{\partial F_3}{\partial x}$(外積と同じく符号に注意)。"],
                   solution=curl_steps(F, P), meta=dict(F=F, P=P))


def guided_div_curl(rng):
    p = g_curl(rng)
    F, P = p.meta["F"], p.meta["P"]
    return ExamItem("発散と回転", rf"ベクトル場 ${ftex(F)}$ と点 $\mathrm P{pt(P)}$ について, 次の問いに答えよ.",
                    [Part(r"点 P における発散 $\nabla\cdot\vec F$ を求めよ.", str(div_at(F, P)), div_steps(F, P)),
                     Part(r"点 P における回転 $\nabla\times\vec F$ を求めよ.", vtex(curl_at(F, P)), curl_steps(F, P))], meta=p.meta)


# ============================================================
# 標準1:最も増加する向きとその割合
# ============================================================
def unit_tex(g):
    k = 0
    for x in g:
        k = gcd(k, abs(x))
    gp = [x // k for x in g]
    N = dot(gp, gp)
    if sqrt_parts(N)[1] == 1:
        r = round(N ** 0.5)
        return vtex([Fraction(x, r) for x in gp])
    return rf"\frac{{1}}{{{tex_root(N)}}}" + vtex(gp)


def g_maxdir(rng):
    f = rand_terms(rng)
    P = rand_point(rng)
    steps, g = grad_steps(f, P)
    if sum(x != 0 for x in g) < 2:
        raise Retry
    N = dot(g, g)
    u = unit_tex(g)
    um = unit_tex([-x for x in g])

    def ch(d, val):
        return rf"\vec u={d},\ \ \text{{最大値}}\ {val}"
    choices = [Choice(ch(u, tex_root(N)), True, value=N),
               Choice(ch(um, tex_root(N)), why="$-\\nabla f$ の向きは、f が最も急に減少する向き。"),
               Choice(ch(u, str(N)), why="最大値は $|\\nabla f|=\\sqrt{f_x^2+f_y^2+f_z^2}$。ルートを忘れていないか。"),
               Choice(ch(unit_tex([g[1], g[0], g[2]]) if g[0] != g[1] else unit_tex([g[0], g[2], g[1]]), tex_root(N)),
                      why="勾配の成分の順番 $(f_x,\\ f_y,\\ f_z)$ を確かめよう。")]
    return Problem(
        "最も増加する向きと最大値", "標準",
        rf"スカラー場 $f(x,y,z)={ptex(f)}$ について、点 $\mathrm P{pt(P)}$ で f が最も急に増加する向きの単位ベクトル "
        r"$\vec u$ と、その向きの方向微分係数(最大値)を求めよ。",
        choices,
        hints=[r"$D_{\vec u}f=\nabla f\cdot\vec u=|\nabla f|\cos\varphi$ は、$\vec u$ が $\nabla f$ と同じ向き($\varphi=0$)のとき最大。",
               r"最大値は $|\nabla f|$。"],
        solution=steps + [rf"$|\nabla f(\mathrm P)|=\sqrt{{{N}}}={tex_root(N)}$",
                          rf"$$\vec u=\frac{{\nabla f}}{{|\nabla f|}}={u},\qquad \text{{最大値}}\ |\nabla f(\mathrm P)|={tex_root(N)}$$"],
        meta=dict(f=f, P=P),
        exam_answer=rf"\vec u={u},\ \ \text{{最大値}}\ {tex_root(N)}")


def guided_maxdir(rng):
    p = g_maxdir(rng)
    f, P = p.meta["f"], p.meta["P"]
    steps, g = grad_steps(f, P)
    N = dot(g, g)
    return ExamItem("最も増加する向き", rf"スカラー場 $f(x,y,z)={ptex(f)}$ と点 $\mathrm{{P}}{pt(P)}$ について, 次の問いに答えよ.",
                    [Part(r"点 P における勾配 $\nabla f(\mathrm P)$ を求めよ.", vtex(g), steps),
                     Part(r"点 P で f が最も急に増加する向きの単位ベクトル $\vec u$ と, その向きの方向微分係数を求めよ.",
                          rf"\vec u={unit_tex(g)},\ \ {tex_root(N)}", p.solution[2:])], meta=p.meta)


# ============================================================
# 標準2:等位面の接平面
# ============================================================
def _sh(c):
    return "" if c == 0 else (f"-{c}" if c > 0 else f"+{-c}")


def plane_tex(n, d):
    k = 0
    for x in list(n) + [d]:
        k = gcd(k, abs(x))
    k = k or 1
    if [x for x in n if x != 0][0] < 0:
        k = -k
    return lin_tex([(x // k, v) for x, v in zip(n, V)]) + f"={d // k}"


def g_level_plane(rng):
    a, b, c = (rng.randint(1, 3) for _ in range(3))
    lin = [rng.choice([-2, -1, 0, 1, 2]) for _ in range(3)]
    Fterms = [(a, (2, 0, 0)), (b, (0, 2, 0)), (c, (0, 0, 2))] + \
             [(l, tuple(int(i == j) for j in range(3))) for i, l in enumerate(lin) if l]
    P = rand_point(rng, (-2, -1, 1, 2))
    k = ev(Fterms, P)
    g = grad_at(Fterms, P)
    if 0 in g or not any(lin):
        raise Retry
    gb = [a * P[0] + lin[0], b * P[1] + lin[1], c * P[2] + lin[2]]   # 2 をかけ忘れ
    choices = [Choice(plane_tex(g, dot(g, P)), True, value=(g, dot(g, P))),
               Choice(plane_tex(g, 0) if dot(g, P) != 0 else plane_tex(g, 1),
                      why="接平面は点 P を通る:$\\nabla F(\\mathrm P)\\cdot(\\vec x-\\vec p)=0$。定数項を確かめよう。"),
               Choice(plane_tex(list(P), dot(P, P)), why="法線ベクトルは点の位置ベクトルではなく、$\\nabla F(\\mathrm P)$。"),
               Choice(plane_tex(gb, dot(gb, P)) if any(gb) else plane_tex(g, -dot(g, P)),
                      why="$x^2$ を偏微分すると $2x$。係数 2 を忘れていないか。")]
    Ftex = ptex(Fterms)
    steps, _ = grad_steps(Fterms, P, "F")
    return Problem(
        "等位面の接平面", "標準",
        rf"曲面 ${Ftex}={k}$ 上の点 $\mathrm P{pt(P)}$ における接平面の方程式を求めよ。",
        choices,
        hints=[r"曲面 $F(x,y,z)=k$ の点 P での法線ベクトルは $\nabla F(\mathrm P)$(勾配は等位面に垂直)。",
               r"接平面:$\nabla F(\mathrm P)\cdot(x-x_0,\ y-y_0,\ z-z_0)=0$。"],
        solution=[rf"$F(x,y,z)={Ftex}$ とおく。"] + steps +
                 [rf"$${g[0]}(x{_sh(P[0])})" + "".join(("+" if c > 0 else "-") + str(abs(c)) + f"({v}{_sh(q)})"
                                                     for c, v, q in zip(g[1:], "yz", P[1:])) + "=0$$",
                  rf"$$\therefore\ {plane_tex(g, dot(g, P))}$$"],
        meta=dict(F=Fterms, P=P, k=k))


def guided_level_plane(rng):
    p = g_level_plane(rng)
    F, P = p.meta["F"], p.meta["P"]
    steps, g = grad_steps(F, P, "F")
    return ExamItem("等位面の法線と接平面", p.statement.replace("における接平面の方程式を求めよ。", "について, 次の問いに答えよ."),
                    [Part(r"$F(x,y,z)$ を左辺の式とするとき, 点 P における法線ベクトル $\nabla F(\mathrm P)$ を求めよ.",
                          vtex(g), steps),
                     Part(r"点 P における接平面の方程式を求めよ.", plane_tex(g, dot(g, P)), p.solution[-2:])], meta=p.meta)


# ============================================================
# 標準3:ベクトル解析の公式
# ============================================================
TRUE_ID = [
    (r"\nabla\times(\nabla f)=\vec 0",
     [r"$\nabla f=(f_x,\ f_y,\ f_z)$ の回転の x 成分は $\dfrac{\partial}{\partial y}f_z-\dfrac{\partial}{\partial z}f_y=f_{zy}-f_{yz}$。",
      r"f が $C^2$ 級なら偏微分の順序を交換できるので $f_{zy}=f_{yz}$、よって 0。y, z 成分も同様。"]),
    (r"\nabla\cdot(\nabla\times\vec F)=0",
     [r"$\nabla\times\vec F=\left(\partial_yF_3-\partial_zF_2,\ \partial_zF_1-\partial_xF_3,\ \partial_xF_2-\partial_yF_1\right)$。",
      r"発散をとると $\partial_x\partial_yF_3-\partial_x\partial_zF_2+\partial_y\partial_zF_1-\partial_y\partial_xF_3"
      r"+\partial_z\partial_xF_2-\partial_z\partial_yF_1=0$(偏微分の順序交換)。"]),
    (r"\nabla\cdot(\nabla f)=\dfrac{\partial^2f}{\partial x^2}+\dfrac{\partial^2f}{\partial y^2}+\dfrac{\partial^2f}{\partial z^2}",
     [r"$\nabla\cdot(f_x,\ f_y,\ f_z)=\partial_x f_x+\partial_y f_y+\partial_z f_z=f_{xx}+f_{yy}+f_{zz}$(ラプラシアン $\Delta f$)。"]),
    (r"\nabla(fg)=f\,\nabla g+g\,\nabla f",
     [r"x 成分は積の微分法より $(fg)_x=f_xg+fg_x$。y, z 成分も同様なので $\nabla(fg)=g\nabla f+f\nabla g$。"]),
    (r"\nabla\cdot(f\vec F)=f\,\nabla\cdot\vec F+\nabla f\cdot\vec F",
     [r"$\nabla\cdot(f\vec F)=\partial_x(fF_1)+\partial_y(fF_2)+\partial_z(fF_3)$。",
      r"積の微分法で $=f(\partial_xF_1+\partial_yF_2+\partial_zF_3)+(f_xF_1+f_yF_2+f_zF_3)=f\,\nabla\cdot\vec F+\nabla f\cdot\vec F$。"]),
]
FALSE_ID = [
    (r"\nabla\cdot(\nabla f)=0", "$\\nabla\\cdot(\\nabla f)$ はラプラシアン $\\Delta f$。例えば $f=x^2$ なら 2 で、0 とは限らない。"),
    (r"\nabla\times(\nabla\times\vec F)=\vec 0", "$\\nabla\\times(\\nabla\\times\\vec F)=\\nabla(\\nabla\\cdot\\vec F)-\\Delta\\vec F$ で、0 とは限らない。"),
    (r"\nabla(fg)=\nabla f\,\nabla g", "積の微分法:$\\nabla(fg)=f\\nabla g+g\\nabla f$。ベクトルどうしの積にはならない。"),
    (r"\nabla\cdot(f\vec F)=f\,\nabla\cdot\vec F", "積の微分法で $\\nabla f\\cdot\\vec F$ の項も出てくる。"),
    (r"\nabla\times(\nabla f)=\nabla f", "勾配場の回転は常に $\\vec 0$(偏微分の順序交換)。"),
    (r"\nabla\cdot(\vec F\times\vec G)=(\nabla\cdot\vec F)(\nabla\cdot\vec G)",
     "正しくは $\\nabla\\cdot(\\vec F\\times\\vec G)=\\vec G\\cdot(\\nabla\\times\\vec F)-\\vec F\\cdot(\\nabla\\times\\vec G)$。"),
]


def g_identity(rng):
    t, proof = rng.choice(TRUE_ID)
    wrong = rng.sample(FALSE_ID, 3)
    choices = [Choice(t, True)] + [Choice(w, why=why) for w, why in wrong]
    return Problem(
        "ベクトル解析の公式", "標準",
        r"f, g を滑らかなスカラー場、$\vec F,\ \vec G$ を滑らかなベクトル場とする。次のうち、"
        r"常に成り立つ式はどれか。",
        choices,
        hints=[r"成分で書いて確かめる。偏微分の順序は交換できる($f_{xy}=f_{yx}$)。",
               r"積の微分法 $(fg)'=f'g+fg'$ が、成分ごとに使われる。"],
        solution=[rf"正しいのは ${t}$。"] + proof + ["他の式は、成分で計算すると成り立たない例がある。"],
        exam_statement=rf"f, g を $C^2$ 級のスカラー場, $\vec F$ を $C^2$ 級のベクトル場とする. ${t}$ が成り立つことを示せ.",
        exam_answer="", exam_solution=proof)


REGISTRY = [
    ProblemType("grad", "勾配", "基本", g_grad),
    ProblemType("dir", "方向微分係数", "基本", g_directional, guided=guided_grad_dir),
    ProblemType("div", "発散 div", "基本", g_div),
    ProblemType("curl", "回転 rot", "基本", g_curl, guided=guided_div_curl),
    ProblemType("maxdir", "最も増加する向きと最大値", "標準", g_maxdir, guided=guided_maxdir),
    ProblemType("level", "等位面の接平面", "標準", g_level_plane, guided=guided_level_plane),
    ProblemType("identity", "ベクトル解析の公式", "標準", g_identity),
]
GUIDED_NAMES = {"dir": "勾配と方向微分係数", "curl": "発散と回転", "maxdir": "勾配と最も増加する向き",
                "level": "等位面の法線と接平面"}


def tab():
    render("ch2", REGISTRY, "第2章 スカラー場とベクトル場")
