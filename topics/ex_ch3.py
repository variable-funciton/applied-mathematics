"""数学IV ベクトル解析 第3章 線積分・面積分:演習の問題生成(線積分・ポテンシャル・グリーンの定理)

昨年度の中間試験の形に合わせている。
  スカラー場の線積分:f と |r'| がまとまって多項式になる曲線(問5の形)
  ベクトル場の線積分:らせん・線分に沿う(問6の形)
  ∇S の線積分:直接計算と S(B) − S(A) の2通り(問7の形)
  グリーンの定理:長方形・三角形・扇形の周(問8の形、方眼つきの図)
"""
import math
from fractions import Fraction

import numpy as np

from topics.ex_ch2 import V, ev, pd, ptex, rand_terms, simplify
from topics.ex_core import (Choice, ExamItem, Part, Problem, ProblemType, Retry, lin_tex, render, tex_fr, tex_root,
                            vtex)

F = Fraction
SIN_T, COS_T, SIN3_T, COS3_T = r"\sin t", r"\cos t", r"\sin^3 t", r"\cos^3 t"
NICE = [(1, 2, 2), (2, 1, 2), (2, 2, 1), (2, 3, 6), (3, 2, 6), (6, 2, 3), (3, 4, 0), (0, 3, 4), (4, 0, 3), (1, 4, 8)]
TRIPLES = [(3, 4, 5), (4, 3, 5), (6, 8, 10), (8, 6, 10), (5, 12, 13), (12, 5, 13)]


# ------------------------------------------------------------
# 1変数多項式(係数は Fraction のリスト、c[k] が t^k の係数)
# ------------------------------------------------------------
def pmul(a, b):
    out = [F(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def padd(a, b):
    n = max(len(a), len(b))
    return [(a[k] if k < len(a) else 0) + (b[k] if k < len(b) else 0) for k in range(n)]


def ppow(a, n):
    out = [F(1)]
    for _ in range(n):
        out = pmul(out, a)
    return out


def subst(terms, A, d):
    """多変数多項式に x_i = A_i + d_i t を代入した t の多項式"""
    out = [F(0)]
    for c, e in terms:
        p = [F(c)]
        for i in range(3):
            p = pmul(p, ppow([F(A[i]), F(d[i])], e[i]))
        out = padd(out, p)
    return out


def pint01(p):
    return sum(c / (k + 1) for k, c in enumerate(p))


def tpoly_tex(p, var="t"):
    terms = [(p[k], var if k == 1 else f"{var}^{{{k}}}") for k in range(len(p) - 1, 0, -1)]
    return lin_tex(terms, p[0] if p else 0)


def coef_tex(c, x):
    """係数 c と式 x の積(1 なら省略)"""
    return x if c == 1 else ("-" + x if c == -1 else f"{c}{x}")


def trig_vec(a, b2):
    """(a cos t, a sin t, z) の成分の文字列"""
    return coef_tex(a, r"\cos t"), coef_tex(a, r"\sin t")


def curve_pts(fn, t0, t1, n=200):
    t = np.linspace(t0, t1, n)
    P = np.array([fn(x) for x in t]).T
    return [list(map(float, P[0])), list(map(float, P[1])), list(map(float, P[2]))]


def pt3(P):
    return "(" + r",\ ".join(tex_fr(x) for x in P) + ")"


# ============================================================
# 基本1:スカラー場の線積分
# ============================================================
def g_scalar_line(rng):
    fam = rng.choice(["q5", "segment", "helix"])
    if fam == "q5":
        a, b, T = rng.choice([1, 2, 3]), rng.choice([1, 2, 3]), rng.choice([1, 2])
        c = 4 * b
        L = F(a * a * T) + F(4 * b * b * T ** 3, 3)
        cx, sy = trig_vec(a, b)
        curve = rf"\vec r(t)=\left({cx},\ {sy},\ {coef_tex(b, 't^2')}\right)\ (0\le t\le {T})"
        fs = rf"\sqrt{{x^2+y^2+{c}z}}"
        choices = [Choice(tex_fr(L), True, value=float(L)),
                   Choice(tex_fr(F(a * a * T) + 4 * b * b * T ** 3), why="$t^2$ を積分すると $\\frac{t^3}{3}$。$\\frac13$ を忘れていないか。"),
                   Choice(tex_fr(F(a * T) + F(4 * b * b * T ** 3, 3)), why=f"曲線上では $x^2+y^2={a * a}$($\\cos^2t+\\sin^2t=1$)。"),
                   Choice(tex_fr(F(a * a * T * T, 2) + F(4 * b * b * T ** 3, 3)),
                          why="定数の積分 $\\int_0^T k\\,dt=kT$。$\\frac{T^2}{2}$ ではない。")]
        sol = [rf"$\dfrac{{d\vec r}}{{dt}}=\left({coef_tex(-a, SIN_T)},\ {coef_tex(a, COS_T)},\ {2 * b}t\right)$ より "
               rf"$\left|\dfrac{{d\vec r}}{{dt}}\right|=\sqrt{{{a * a}+{4 * b * b}t^2}}$",
               rf"曲線上では $x^2+y^2+{c}z={a * a}+{c * b}t^2$ なので $f=\sqrt{{{a * a}+{4 * b * b}t^2}}$",
               rf"$$\int_C f\,ds=\int_0^{{{T}}}\left({a * a}+{4 * b * b}t^2\right)dt={tex_fr(L)}$$"]
        fn = lambda t: (a * math.cos(t), a * math.sin(t), b * t * t)  # noqa: E731
        meta = dict(fam=fam, a=a, b=b, T=T)
    elif fam == "segment":
        A = [rng.randint(-2, 2) for _ in range(3)]
        d = list(rng.choice(NICE))
        rng.shuffle(d)
        d = [x * rng.choice([1, -1]) for x in d]
        n = round(math.sqrt(sum(x * x for x in d)))
        co = [rng.choice([-2, -1, 1, 2, 3]) for _ in range(3)]
        k0 = rng.randint(-3, 3)
        M = [F(A[i]) + F(d[i], 2) for i in range(3)]
        fM = sum(co[i] * M[i] for i in range(3)) + k0
        if fM == 0:
            raise Retry
        L = n * fM
        B = [A[i] + d[i] for i in range(3)]
        comps = [lin_tex([(d[i], "t")], A[i]) for i in range(3)]
        curve = r"\vec r(t)=\left(" + r",\ ".join(comps) + r"\right)\ (0\le t\le 1)"
        fs = lin_tex([(co[i], V[i]) for i in range(3)], k0)
        choices = [Choice(tex_fr(L), True, value=float(L)),
                   Choice(tex_fr(fM), why=f"$ds=|\\vec r\\,'(t)|\\,dt$。$|\\vec r\\,'|={n}$ をかけ忘れていないか。"),
                   Choice(tex_fr(n * n * fM), why="$|\\vec r\\,'|=\\sqrt{x'^2+y'^2+z'^2}$。ルートを忘れていないか。"),
                   Choice(tex_fr(2 * L), why="$\\int_0^1 t\\,dt=\\frac12$。積分の計算を確かめよう。")]
        ft = [F(sum(co[i] * A[i] for i in range(3)) + k0), F(sum(co[i] * d[i] for i in range(3)))]
        sol = [rf"$\vec r\,'(t)={vtex(d)}$ より $|\vec r\,'(t)|=\sqrt{{{n * n}}}={n}$",
               rf"曲線上では $f={tpoly_tex(ft)}$",
               rf"$$\int_C f\,ds=\int_0^1\left({tpoly_tex(ft)}\right)\cdot{n}\,dt={tex_fr(L)}$$"]
        fn = lambda t: tuple(A[i] + d[i] * t for i in range(3))  # noqa: E731
        meta = dict(fam=fam, A=A, d=d, co=co, k0=k0)
    else:
        a, b, c = rng.choice(TRIPLES)
        T = rng.choice([1, 2])
        kind = rng.choice(["z", "r2"])
        cx, sy = trig_vec(a, b)
        curve = rf"\vec r(t)=\left({cx},\ {sy},\ {b}t\right)\ (0\le t\le {T})"
        if kind == "z":
            fs = "z"
            L = F(c * b * T * T, 2)
            w = [F(b * T * T, 2), F((a + b) * b * T * T, 2), F(c * b * T * T)]
            ftex_t = f"{b}t"
            integ = rf"\int_0^{{{T}}}{b}t\cdot{c}\,dt"
        else:
            fs = "x^2+y^2"
            L = F(c * a * a * T)
            w = [F(a * a * T), F((a + b) * a * a * T), F(c * a * a * T * T, 2)]
            ftex_t = f"{a * a}"
            integ = rf"\int_0^{{{T}}}{a * a}\cdot{c}\,dt"
        choices = [Choice(tex_fr(L), True, value=float(L)),
                   Choice(tex_fr(w[0]), why="$ds=|\\vec r\\,'(t)|\\,dt$。$|\\vec r\\,'|$ をかけ忘れていないか。"),
                   Choice(tex_fr(w[1]), why="$|\\vec r\\,'|=\\sqrt{x'^2+y'^2+z'^2}$。成分を足すだけでは大きさにならない。"),
                   Choice(tex_fr(w[2]), why="$t$ の積分の計算を確かめよう。")]
        sol = [rf"$\vec r\,'(t)=\left({coef_tex(-a, SIN_T)},\ {coef_tex(a, COS_T)},\ {b}\right)$ より "
               rf"$|\vec r\,'(t)|=\sqrt{{{a * a}+{b * b}}}={c}$",
               rf"曲線上では $f={ftex_t}$", rf"$$\int_C f\,ds={integ}={tex_fr(L)}$$"]
        fn = lambda t: (a * math.cos(t), a * math.sin(t), b * t)  # noqa: E731
        meta = dict(fam=fam, a=a, b=b, T=T, kind=kind)
    T1 = meta.get("T", 1)
    return Problem(
        "スカラー場の線積分", "基本",
        rf"曲線 $C:\ {curve}$ に沿って, 線積分 $\displaystyle\int_C\left({fs}\right)ds$ を求めよ。",
        choices,
        hints=[r"$\displaystyle\int_C f\,ds=\int_a^b f(\vec r(t))\,\left|\frac{d\vec r}{dt}\right|dt$。",
               r"まず $\left|\dfrac{d\vec r}{dt}\right|$ を計算し、曲線上での $f$ を $t$ で表す。"],
        solution=sol, meta=meta, figure={"kind": "curve3d", "pts": curve_pts(fn, 0, T1)})


# ============================================================
# 基本2:ベクトル場の線積分
# ============================================================
def g_vector_line(rng):
    fam = rng.choice(["helix", "segment"])
    if fam == "helix":
        a, b = rng.choice([1, 2, 3]), rng.choice([1, 2, 3])
        al, be = rng.choice([1, 2, 3]), rng.choice([1, 2, 3])
        s = rng.choice([1, -1])
        T = rng.choice([1, 2])
        I = F(s * al * a * a * T) + F(be * b * b * T * T, 2)
        Ftex = rf"\left({coef_tex(-s * al, 'y')},\ {coef_tex(s * al, 'x')},\ {coef_tex(be, 'z')}\right)"
        cx, sy = trig_vec(a, b)
        curve = rf"\vec r(t)=\left({cx},\ {sy},\ {coef_tex(b, 't')}\right)\ (0\le t\le {T})"
        choices = [Choice(tex_fr(I), True, value=float(I)),
                   Choice(tex_fr(-I), why="向きに注意。$t$ が増える向き(始点 $t=0$ → 終点)に積分する。"),
                   Choice(tex_fr(F(s * al * a * a * T) + be * b * b * T * T), why="$\\int_0^T t\\,dt=\\frac{T^2}{2}$。"),
                   Choice(tex_fr(F(s * al * a * a * T)), why="$z$ 成分 $F_3\\,\\dfrac{dz}{dt}$ の項を足し忘れていないか。")]
        g = [F(s * al * a * a), F(be * b * b)]
        sol = [rf"$\dfrac{{d\vec r}}{{dt}}=\left({coef_tex(-a, SIN_T)},\ {coef_tex(a, COS_T)},\ {b}\right)$",
               rf"曲線上では $\vec F=\left({coef_tex(-s * al * a, SIN_T)},\ {coef_tex(s * al * a, COS_T)},\ {coef_tex(be * b, 't')}\right)$",
               rf"$$\vec F\cdot\frac{{d\vec r}}{{dt}}={s * al * a * a}(\sin^2t+\cos^2t)+{be * b * b}t={tpoly_tex(g)}$$",
               rf"$$\int_C\vec F\cdot d\vec r=\int_0^{{{T}}}\left({tpoly_tex(g)}\right)dt={tex_fr(I)}$$"]
        fn = lambda t: (a * math.cos(t), a * math.sin(t), b * t)  # noqa: E731
        meta = dict(fam=fam, a=a, b=b, al=al, be=be, s=s, T=T)
        t1 = T
    else:
        Fl = [rand_terms(rng, n=2, maxdeg=2, mixed=False) for _ in range(3)]
        A = [rng.randint(-2, 2) for _ in range(3)]
        d = [rng.choice([-2, -1, 1, 2, 3]) for _ in range(3)]
        g = [F(0)]
        for i in range(3):
            g = padd(g, [x * d[i] for x in subst(Fl[i], A, d)])
        I = pint01(g)
        if I == 0:
            raise Retry
        gr = [F(0)]
        for i in range(3):
            gr = padd(gr, pmul(subst(Fl[i], A, d), [F(A[i]), F(d[i])]))
        Ir = pint01(gr)
        B = [A[i] + d[i] for i in range(3)]
        endv = sum(ev(Fl[i], B) * d[i] for i in range(3))
        Ftex = r"\left(" + r",\ ".join(ptex(c) for c in Fl) + r"\right)"
        curve = r"\vec r(t)=\left(" + r",\ ".join(lin_tex([(d[i], "t")], A[i]) for i in range(3)) + r"\right)\ (0\le t\le 1)"
        choices = [Choice(tex_fr(I), True, value=float(I)),
                   Choice(tex_fr(-I), why="向きに注意。$t$ が増える向きに積分する。"),
                   Choice(tex_fr(Ir), why="内積の相手は $\\vec r(t)$ ではなく $\\dfrac{d\\vec r}{dt}$。"),
                   Choice(tex_fr(F(endv)), why="終点での値だけでは線積分にならない。$t$ の式にしてから $0$ から $1$ まで積分する。")]
        while g and g[-1] == 0 and len(g) > 1:
            g = g[:-1]
        sol = [rf"$\dfrac{{d\vec r}}{{dt}}={vtex(d)}$",
               r"曲線上では $\vec F=\left(" + r",\ ".join(tpoly_tex(subst(Fl[i], A, d)) for i in range(3)) + r"\right)$",
               rf"$$\vec F\cdot\frac{{d\vec r}}{{dt}}={tpoly_tex(g)}$$",
               rf"$$\int_C\vec F\cdot d\vec r=\int_0^1\left({tpoly_tex(g)}\right)dt={tex_fr(I)}$$"]
        fn = lambda t: tuple(A[i] + d[i] * t for i in range(3))  # noqa: E731
        meta = dict(fam=fam, F=Fl, A=A, d=d)
        t1 = 1
    return Problem(
        "ベクトル場の線積分", "基本",
        rf"曲線 $C:\ {curve}$ に沿って, 線積分 $\displaystyle\int_C{Ftex}\cdot d\vec r$ を求めよ。",
        choices,
        hints=[r"$\displaystyle\int_C\vec F\cdot d\vec r=\int_a^b\vec F(\vec r(t))\cdot\frac{d\vec r}{dt}\,dt$。",
               r"曲線上の $\vec F$ を $t$ で表し、$\dfrac{d\vec r}{dt}$ との内積をとってから積分する。"],
        solution=sol, meta=meta, figure={"kind": "curve3d", "pts": curve_pts(fn, 0, t1)})


# ============================================================
# 基本3:∇S の線積分(問7の形)
# ============================================================
def grad_line_parts(rng):
    S = rand_terms(rng, n=rng.choice([2, 3]), maxdeg=2, coef=(-3, 3), mixed=True)
    A = [rng.randint(0, 3) for _ in range(3)]
    d = [rng.randint(1, 4) for _ in range(3)]
    B = [A[i] + d[i] for i in range(3)]
    I = ev(S, B) - ev(S, A)
    if I == 0 or ev(S, A) == 0:
        raise Retry
    grad = [pd(S, i) for i in range(3)]
    g = [F(0)]
    for i in range(3):
        g = padd(g, [x * d[i] for x in subst(grad[i], A, d)])
    while len(g) > 1 and g[-1] == 0:
        g = g[:-1]
    gtex = r"\left(" + r",\ ".join(ptex(c) if c else "0" for c in grad) + r"\right)"
    curve = r"\vec r(t)=\left(" + r",\ ".join(lin_tex([(d[i], "t")], A[i]) for i in range(3)) + r"\right)\ (0\le t\le 1)"
    return S, A, B, d, I, grad, g, gtex, curve


def segment_fig(A, B):
    return {"kind": "segment3d", "A": tuple(A), "B": tuple(B),
            "labels": [rf"$\mathrm{{A}}{pt3(A)}$", rf"$\mathrm{{B}}{pt3(B)}$"],
            "web_labels": [f"A{tuple(A)}", f"B{tuple(B)}"]}


def grad_line_solution(S, A, B, d, I, grad, g, gtex):
    return [rf"$\nabla S={gtex}$",
            rf"解法1(直接計算):$\dfrac{{d\vec r}}{{dt}}={vtex(d)}$, 曲線上で "
            r"$\nabla S=\left(" + r",\ ".join(tpoly_tex(subst(grad[i], A, d)) if grad[i] else "0" for i in range(3)) + r"\right)$",
            rf"$$\int_C\nabla S\cdot d\vec r=\int_0^1\left({tpoly_tex(g)}\right)dt={tex_fr(I)}$$",
            rf"解法2(ポテンシャル):始点 $\mathrm{{A}}{pt3(A)}$, 終点 $\mathrm{{B}}{pt3(B)}$ なので, 経路によらず",
            rf"$$\int_C\nabla S\cdot d\vec r=S(\mathrm B)-S(\mathrm A)=({ev(S, B)})-({ev(S, A)})={tex_fr(I)}$$"]


def g_grad_line(rng):
    S, A, B, d, I, grad, g, gtex, curve = grad_line_parts(rng)
    sA, sB = ev(S, A), ev(S, B)
    choices = [Choice(tex_fr(I), True, value=float(I)),
               Choice(tex_fr(sA - sB), why="$\\int_C\\nabla S\\cdot d\\vec r=S(\\text{終点})-S(\\text{始点})$。順番が逆。"),
               Choice(tex_fr(sA + sB), why="終点と始点の値の「差」をとる。"),
               Choice(tex_fr(sB), why="始点での値 $S(\\mathrm A)$ を引き忘れていないか。")]
    return Problem(
        "勾配場の線積分", "基本",
        rf"スカラー場 $S(x,y,z)={ptex(S)}$ と, 始点 $\mathrm{{A}}{pt3(A)}$ から終点 $\mathrm{{B}}{pt3(B)}$ へ向かう線分 "
        rf"$C:\ {curve}$ について, 線積分 $\displaystyle\int_C\nabla S\cdot d\vec r$ を求めよ。",
        choices,
        hints=[r"勾配場の線積分は、始点と終点だけで決まる:$\displaystyle\int_C\nabla S\cdot d\vec r=S(\text{終点})-S(\text{始点})$。",
               r"直接計算してもよい:$\nabla S$ を曲線上で $t$ の式にして $\dfrac{d\vec r}{dt}$ との内積を積分する。"],
        solution=grad_line_solution(S, A, B, d, I, grad, g, gtex),
        meta=dict(S=S, A=A, B=B), figure=segment_fig(A, B))


def guided_grad_line(rng):
    S, A, B, d, I, grad, g, gtex, curve = grad_line_parts(rng)
    sol = grad_line_solution(S, A, B, d, I, grad, g, gtex)
    return ExamItem("勾配場の線積分", rf"スカラー場 $S(x,y,z)={ptex(S)}$ と, 始点 $\mathrm{{A}}{pt3(A)}$ から終点 "
                    rf"$\mathrm{{B}}{pt3(B)}$ へ向かう線分 $C:\ {curve}$ について, 次の問いに答えよ.",
                    [Part(r"$\nabla S$ を求めよ.", gtex, [sol[0]]),
                     Part(r"線積分 $\displaystyle\int_C\nabla S\cdot d\vec r$ を求めよ.", tex_fr(I), sol[1:])],
                    meta=dict(S=S, A=A, B=B), figure=segment_fig(A, B))


# ============================================================
# 標準1:保存場のポテンシャル
# ============================================================
def potential_parts(rng):
    phi = rand_terms(rng, n=3, maxdeg=3, coef=(-3, 3), mixed=True)
    if sum(1 for _, e in phi if sum(k > 0 for k in e) >= 2) < 1 or any(not any(e) for _, e in phi):
        raise Retry
    Fl = [pd(phi, i) for i in range(3)]
    if any(not c for c in Fl):
        raise Retry
    return phi, Fl


def _integrate(terms, i):
    """x_i について積分(定数は 0)"""
    out = []
    for c, e in terms:
        e2 = list(e)
        e2[i] += 1
        out.append((F(c, e2[i]), tuple(e2)))
    return out


def g_potential(rng):
    phi, Fl = potential_parts(rng)
    naive = simplify(_integrate(Fl[0], 0) + _integrate(Fl[1], 1) + _integrate(Fl[2], 2))
    only_x = simplify(_integrate(Fl[0], 0))
    flip = [(-c, e) if k == 0 else (c, e) for k, (c, e) in enumerate(phi)]

    def ph(t):
        return r"\varphi=" + ptex(sorted(t, key=lambda x: x[1], reverse=True))
    choices = [Choice(ph(phi), True),
               Choice(ph(naive), why="各成分を積分して全部足すと、同じ項を2回以上数えてしまう。"
                                     "$\\varphi_x=F_1$ から始め、足りない項だけを $\\varphi_y=F_2,\\ \\varphi_z=F_3$ で補う。"),
               Choice(ph(only_x), why="$F_1$ を $x$ で積分しただけでは、$x$ を含まない項が抜ける。$\\varphi_y=F_2,\\ \\varphi_z=F_3$ も確かめよう。"),
               Choice(ph(flip), why="検算:$\\nabla\\varphi$ を計算して $\\vec F$ に一致するか確かめよう。")]
    Ftex = r"\vec F=\left(" + r",\ ".join(ptex(c) for c in Fl) + r"\right)"
    ix = _integrate(Fl[0], 0)
    return Problem(
        "保存場のポテンシャル", "標準",
        rf"${Ftex}$ は保存場である. $\vec F=\nabla\varphi$, $\varphi(0,0,0)=0$ を満たすポテンシャル $\varphi$ を求めよ。",
        choices,
        hints=[r"$\dfrac{\partial\varphi}{\partial x}=F_1$ を $x$ で積分すると $\varphi=\displaystyle\int F_1\,dx+g(y,z)$。",
               r"これを $y$ で偏微分して $F_2$ と比べ、$g(y,z)$ を決める。最後に $z$ についても同様に。"],
        solution=[rf"$\varphi_x=F_1$ より $\varphi={ptex(simplify(ix))}+g(y,z)$",
                  r"$\varphi_y=F_2,\ \varphi_z=F_3$ と比べて、足りない項を $g(y,z)$ で補う。",
                  rf"$\varphi(0,0,0)=0$ より定数は $0$. よって ${ph(phi)}$",
                  r"検算:$\nabla\varphi=\vec F$ が成り立つ。"],
        meta=dict(phi=phi, F=Fl),
        exam_answer=ph(phi))


# ============================================================
# 標準2:グリーンの定理(問8の形)
# ============================================================
SIN = {90: (F(1), F(0)), 60: (F(0), F(1, 2)), 30: (F(1, 2), F(0))}       # sin θ = a + b√3
OMC = {90: (F(1), F(0)), 60: (F(1, 2), F(0)), 30: (F(1), F(-1, 2))}      # 1 − cos θ = a + b√3


def combo_tex(v):
    """(定数, π の係数, √3 の係数) → TeX"""
    c, p, s = v
    return lin_tex([(p, r"\pi"), (s, r"\sqrt{3}")], c)


def combo_float(v):
    return float(v[0]) + float(v[1]) * math.pi + float(v[2]) * math.sqrt(3)


def region_integral(reg, c0, c1, c2):
    """∬_D (c0 + c1 x + c2 y) dxdy を (定数, π, √3) で"""
    if reg["kind"] == "rect":
        a, b = reg["a"], reg["b"]
        return (F(c0 * a * b) + F(c1 * a * a * b, 2) + F(c2 * a * b * b, 2), F(0), F(0))
    if reg["kind"] == "tri":
        a, b = reg["a"], reg["b"]
        return (F(c0 * a * b, 2) + F(c1 * a * a * b, 6) + F(c2 * a * b * b, 6), F(0), F(0))
    R, deg = reg["R"], reg["deg"]
    sa, sb = SIN[deg]
    oa, ob = OMC[deg]
    k = F(R ** 3, 3)
    return (k * (c1 * sa + c2 * oa), F(c0 * deg * R * R, 360), k * (c1 * sb + c2 * ob))


def green_parts(rng):
    kind = rng.choice(["rect", "tri", "sector"])
    if kind == "rect":
        reg = dict(kind=kind, a=rng.randint(2, 4), b=rng.randint(2, 5))
    elif kind == "tri":
        reg = dict(kind=kind, a=rng.randint(2, 5), b=rng.randint(2, 6))
    else:
        reg = dict(kind=kind, R=rng.choice([1, 2, 3]), deg=rng.choice([90, 60, 30]))
    # P = α xy + β y^2 + γ y + δ x^2,  Q = μ x^2 + ν xy + λ x + ρ y^2
    nz = [-3, -2, -1, 1, 2, 3, 4]
    al, be, ga, de = rng.choice(nz + [0]), rng.choice([0, 0, 1, -1]), rng.choice([0, 1, -1, 2]), rng.choice([0, 1, -2])
    mu, nu, la, rho = rng.choice(nz + [0]), rng.choice([0, 0, 1, -1, 2]), rng.choice([0, 1, -1, 3]), rng.choice([0, 2, 4, -1])
    P = [(al, (1, 1, 0)), (be, (0, 2, 0)), (ga, (0, 1, 0)), (de, (2, 0, 0))]
    Q = [(mu, (2, 0, 0)), (nu, (1, 1, 0)), (la, (1, 0, 0)), (rho, (0, 2, 0))]
    P, Q = [t for t in P if t[0]], [t for t in Q if t[0]]
    if not P or not Q:
        raise Retry
    c0, c1, c2 = la - ga, 2 * mu - al, nu - 2 * be
    if (c1, c2) == (0, 0) or (kind == "sector" and c0 == 0 and rng.random() < 0.5):
        raise Retry
    I = region_integral(reg, c0, c1, c2)
    if combo_float(I) == 0:
        raise Retry
    return reg, P, Q, (c0, c1, c2), I


def green_text(reg, P, Q):
    PQ = rf"\int_C\left({ptex(P)}\right)dx+\left({ptex(Q)}\right)dy"
    if reg["kind"] == "rect":
        a, b = reg["a"], reg["b"]
        st = (rf"4点 $\mathrm{{O}}(0,0),\ \mathrm{{A}}_0({a},0),\ \mathrm{{A}}_1({a},{b}),\ \mathrm{{A}}_2(0,{b})$ を頂点とする長方形の周を, "
              rf"$\mathrm{{O}}\to\mathrm{{A}}_0\to\mathrm{{A}}_1\to\mathrm{{A}}_2\to\mathrm{{O}}$ の向きに一周する曲線を $C$ とするとき, ")
        fig = {"kind": "polygon", "pts": [(0, 0), (a, 0), (a, b), (0, b)], "labels": ["O", "A_0", "A_1", "A_2"]}
    elif reg["kind"] == "tri":
        a, b = reg["a"], reg["b"]
        st = (rf"3点 $\mathrm{{O}}(0,0),\ \mathrm{{A}}({a},0),\ \mathrm{{B}}(0,{b})$ を頂点とする三角形の周を, "
              rf"$\mathrm{{O}}\to\mathrm{{A}}\to\mathrm{{B}}\to\mathrm{{O}}$ の向きに一周する曲線を $C$ とするとき, ")
        fig = {"kind": "polygon", "pts": [(0, 0), (a, 0), (0, b)], "labels": ["O", "A", "B"]}
    else:
        R, deg = reg["R"], reg["deg"]
        ang = {90: r"\frac{\pi}{2}", 60: r"\frac{\pi}{3}", 30: r"\frac{\pi}{6}"}[deg]
        st = (rf"原点 $\mathrm{{O}}$ を中心とする半径 ${R}$, 中心角 ${ang}$ の扇形 $\mathrm{{OAB}}$($\mathrm{{A}}({R},0)$)の周を, "
              rf"$\mathrm{{O}}\to\mathrm{{A}}\to\mathrm{{B}}\to\mathrm{{O}}$ の向きに一周する曲線を $C$ とするとき, ")
        fig = {"kind": "sector", "R": R, "deg": deg, "labels": ["O", "A", "B"]}
    return st, PQ, fig


def green_solution(reg, P, Q, cs, I):
    c0, c1, c2 = cs
    ig = lin_tex([(c1, "x"), (c2, "y")], c0)
    sol = [rf"$P={ptex(P)},\ Q={ptex(Q)}$ とおくと, "
           rf"$\dfrac{{\partial Q}}{{\partial x}}-\dfrac{{\partial P}}{{\partial y}}=\left({ptex(pd(Q, 0)) if pd(Q, 0) else '0'}\right)"
           rf"-\left({ptex(pd(P, 1)) if pd(P, 1) else '0'}\right)={ig}$",
           r"$C$ が囲む領域を $D$ とすると, グリーンの定理より"]
    if reg["kind"] == "rect":
        a, b = reg["a"], reg["b"]
        sol.append(rf"$$\iint_D\left({ig}\right)dxdy=\int_0^{{{b}}}\left(\int_0^{{{a}}}\left({ig}\right)dx\right)dy={combo_tex(I)}$$")
    elif reg["kind"] == "tri":
        a, b = reg["a"], reg["b"]
        up = lin_tex([(F(-b, a), "x")], b)
        sol.append(rf"$$\iint_D\left({ig}\right)dxdy=\int_0^{{{a}}}\left(\int_0^{{{up}}}\left({ig}\right)dy\right)dx={combo_tex(I)}$$")
    else:
        R, deg = reg["R"], reg["deg"]
        ang = {90: r"\frac{\pi}{2}", 60: r"\frac{\pi}{3}", 30: r"\frac{\pi}{6}"}[deg]
        igp = lin_tex([(c1, r"r\cos\theta"), (c2, r"r\sin\theta")], c0)
        sol.append(r"極座標 $x=r\cos\theta,\ y=r\sin\theta,\ dxdy=r\,drd\theta$ を用いて")
        sol.append(rf"$$\iint_D\left({ig}\right)dxdy=\int_0^{{{ang}}}\int_0^{{{R}}}\left({igp}\right)r\,dr\,d\theta={combo_tex(I)}$$")
    return sol


def g_green(rng):
    reg, P, Q, cs, I = green_parts(rng)
    st, PQ, fig = green_text(reg, P, Q)
    c0, c1, c2 = cs
    # 誤答:符号が逆 / Q_x + P_y / P_x + Q_y(発散の形)
    qx_c = [sum(c for c, e in pd(Q, 0) if e == (0, 0, 0)), sum(c for c, e in pd(Q, 0) if e == (1, 0, 0)),
            sum(c for c, e in pd(Q, 0) if e == (0, 1, 0))]
    py_c = [sum(c for c, e in pd(P, 1) if e == (0, 0, 0)), sum(c for c, e in pd(P, 1) if e == (1, 0, 0)),
            sum(c for c, e in pd(P, 1) if e == (0, 1, 0))]
    px_c = [sum(c for c, e in pd(P, 0) if e == (0, 0, 0)), sum(c for c, e in pd(P, 0) if e == (1, 0, 0)),
            sum(c for c, e in pd(P, 0) if e == (0, 1, 0))]
    qy_c = [sum(c for c, e in pd(Q, 1) if e == (0, 0, 0)), sum(c for c, e in pd(Q, 1) if e == (1, 0, 0)),
            sum(c for c, e in pd(Q, 1) if e == (0, 1, 0))]
    cand = [(region_integral(reg, -c0, -c1, -c2),
             "向き(反時計回りが正)と、$\\frac{\\partial Q}{\\partial x}-\\frac{\\partial P}{\\partial y}$ の引く順番を確かめよう。"),
            (region_integral(reg, *[a + b for a, b in zip(qx_c, py_c)]),
             "グリーンの定理の被積分関数は $\\frac{\\partial Q}{\\partial x}-\\frac{\\partial P}{\\partial y}$(引き算)。"),
            (region_integral(reg, *[a + b for a, b in zip(px_c, qy_c)]),
             "それは $\\frac{\\partial P}{\\partial x}+\\frac{\\partial Q}{\\partial y}$(発散の形)。線積分 $P\\,dx+Q\\,dy$ では "
             "$\\frac{\\partial Q}{\\partial x}-\\frac{\\partial P}{\\partial y}$ を使う。"),
            (tuple(2 * x for x in I), "重積分の計算(累次積分の範囲)を確かめよう。"),
            (tuple(x / 2 for x in I), "重積分の計算(累次積分の範囲)を確かめよう。")]
    choices = [Choice(combo_tex(I), True, value=combo_float(I))]
    vals = {round(combo_float(I), 9)}
    for v, why in cand:
        fv = round(combo_float(v), 9)
        if fv not in vals and len(choices) < 4:
            choices.append(Choice(combo_tex(v), why=why, value=combo_float(v)))
            vals.add(fv)
    return Problem(
        "グリーンの定理", "標準",
        st + rf"グリーンの定理を用いて, 線積分 $\displaystyle {PQ}$ を求めよ。",
        choices,
        hints=[r"グリーンの定理:$\displaystyle\oint_C P\,dx+Q\,dy=\iint_D\left(\frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y}\right)dxdy$"
               r"($C$ は $D$ の境界を反時計回りに一周する)。",
               r"$\dfrac{\partial Q}{\partial x}-\dfrac{\partial P}{\partial y}$ を計算してから、領域 $D$ で2重積分する。"
               r"扇形は極座標を使う。"],
        solution=green_solution(reg, P, Q, cs, I), meta=dict(reg=reg, P=P, Q=Q, I=I), figure=fig)


def guided_green(rng):
    reg, P, Q, cs, I = green_parts(rng)
    st, PQ, fig = green_text(reg, P, Q)
    sol = green_solution(reg, P, Q, cs, I)
    c0, c1, c2 = cs
    return ExamItem("グリーンの定理", st + r"次の問いに答えよ.",
                    [Part(rf"$P={ptex(P)},\ Q={ptex(Q)}$ とするとき, $\dfrac{{\partial Q}}{{\partial x}}-\dfrac{{\partial P}}{{\partial y}}$ を求めよ.",
                          lin_tex([(c1, "x"), (c2, "y")], c0), [sol[0]]),
                     Part(rf"グリーンの定理を用いて, 線積分 $\displaystyle {PQ}$ を求めよ.", combo_tex(I), sol[1:])],
                    meta=dict(reg=reg, P=P, Q=Q, I=I), figure=fig)


# ============================================================
# 標準3:グリーンの定理による面積
# ============================================================
def g_area_green(rng):
    kind = rng.choice(["ellipse", "astroid"])
    if kind == "ellipse":
        a, b = rng.choice([(2, 1), (3, 2), (3, 1), (4, 3), (5, 2), (4, 1)])
        if rng.random() < 0.5:
            a, b = b, a
        S = (F(0), F(a * b), F(0))
        curve = rf"\vec r(t)=\left({coef_tex(a, COS_T)},\ {coef_tex(b, SIN_T)}\right)\ (0\le t\le 2\pi)"
        wrong = [((F(0), F(2 * a * b), F(0)), "公式の $\\frac12$ を忘れていないか。"),
                 ((F(0), F(a * b, 2), F(0)), "$\\int_0^{2\\pi}(\\cos^2t+\\sin^2t)\\,dt=2\\pi$。"),
                 ((F(0), F(a * a + b * b, 2), F(0)), "$x\\,\\dfrac{dy}{dt}-y\\,\\dfrac{dx}{dt}$ を正しく計算しよう(積 $ab$ が出てくる)。")]
        sol = [rf"$x={coef_tex(a, COS_T)},\ y={coef_tex(b, SIN_T)}$ より "
               rf"$x\dfrac{{dy}}{{dt}}-y\dfrac{{dx}}{{dt}}={a * b}\cos^2t+{a * b}\sin^2t={a * b}$",
               rf"$$S=\frac12\int_0^{{2\pi}}{a * b}\,dt={combo_tex(S)}$$"]
        fn = lambda t: (a * math.cos(t), b * math.sin(t), 0)  # noqa: E731
    else:
        a = rng.choice([1, 2, 3, 4])
        S = (F(0), F(3 * a * a, 8), F(0))
        curve = rf"\vec r(t)=\left({coef_tex(a, COS3_T)},\ {coef_tex(a, SIN3_T)}\right)\ (0\le t\le 2\pi)"
        wrong = [((F(0), F(3 * a * a, 4), F(0)), "公式の $\\frac12$ を忘れていないか。"),
                 ((F(0), F(a * a), F(0)), "$\\sin^2t\\cos^2t=\\frac14\\sin^2 2t$ の積分を確かめよう。"),
                 ((F(0), F(3 * a * a, 16), F(0)), "$\\int_0^{2\\pi}\\sin^2 2t\\,dt=\\pi$。")]
        sol = [rf"$x\dfrac{{dy}}{{dt}}-y\dfrac{{dx}}{{dt}}={3 * a * a}\cos^4t\sin^2t+{3 * a * a}\sin^4t\cos^2t={3 * a * a}\sin^2t\cos^2t"
               rf"=\dfrac{{{3 * a * a}}}{{4}}\sin^2 2t$",
               rf"$$S=\frac12\int_0^{{2\pi}}\frac{{{3 * a * a}}}{{4}}\sin^22t\,dt=\frac{{{3 * a * a}}}{{8}}\cdot\pi={combo_tex(S)}$$"]
        fn = lambda t: (a * math.cos(t) ** 3, a * math.sin(t) ** 3, 0)  # noqa: E731
    choices = [Choice(combo_tex(S), True, value=combo_float(S))] + [Choice(combo_tex(v), why=w, value=combo_float(v))
                                                                    for v, w in wrong]
    return Problem(
        "グリーンの定理と面積", "標準",
        rf"曲線 $C:\ {curve}$ で囲まれた図形の面積 $S$ を, グリーンの定理から得られる公式 "
        r"$S=\dfrac12\displaystyle\oint_C(x\,dy-y\,dx)$ を用いて求めよ。",
        choices,
        hints=[r"グリーンの定理で $P=-y,\ Q=x$ とすると $\dfrac{\partial Q}{\partial x}-\dfrac{\partial P}{\partial y}=2$ なので "
               r"$\oint_C(x\,dy-y\,dx)=2S$。",
               r"$dx=\dfrac{dx}{dt}dt,\ dy=\dfrac{dy}{dt}dt$ として $t$ の積分にする。"],
        solution=sol, meta=dict(kind=kind, S=S), figure={"kind": "curve2d", "pts": curve_pts(fn, 0, 2 * math.pi)[:2]})


# ============================================================
# 面積分(平面の三角形の部分)
# ============================================================
from math import gcd as _gcd  # noqa: E402

from topics.ex_core import tex_frac_root  # noqa: E402


def plane_parts(rng):
    a, b, c = (rng.choice([1, 2, 3, 4]) for _ in range(3))
    g = _gcd(_gcd(b * c, c * a), a * b)
    eq = lin_tex([(b * c // g, "x"), (c * a // g, "y"), (a * b // g, "z")]) + f"={a * b * c // g}"
    N = a * a * b * b + b * b * c * c + c * c * a * a
    return a, b, c, eq, N


def tri_int(a, b, k0, kx, ky):
    """∬_D (k0 + kx x + ky y) dxdy、D:x/a + y/b ≤ 1, x, y ≥ 0"""
    return F(k0 * a * b, 2) + F(kx * a * a * b, 6) + F(ky * a * b * b, 6)


def plane_statement(eq):
    return rf"平面 ${eq}$ の $x\ge0,\ y\ge0,\ z\ge0$ の部分を $S$ とする. "


def g_surface_scalar(rng):
    a, b, c, eq, N = plane_parts(rng)
    al, be, ga, de = (rng.choice([-2, -1, 0, 1, 2, 3]) for _ in range(4))
    if (al, be, ga) == (0, 0, 0) or ga == 0:
        raise Retry
    k0, kx, ky = F(ga * c + de), F(al) - F(ga * c, a), F(be) - F(ga * c, b)
    K = tri_int(a, b, k0, kx, ky)
    if K == 0:
        raise Retry
    Q = K / (a * b)
    K0 = tri_int(a, b, F(de), F(al), F(be))
    ftex = lin_tex([(al, "x"), (be, "y"), (ga, "z")], de)

    def val(q):
        return tex_frac_root(q.numerator, q.denominator, N)
    choices = [Choice(val(Q), True, value=float(Q) * math.sqrt(N)),
               Choice(tex_fr(K), why="$dS=\\sqrt{1+z_x^2+z_y^2}\\,dxdy$。面積の拡大率をかけ忘れていないか。"),
               Choice(tex_fr(K * N / (a * a * b * b)), why="$dS$ の係数は $\\sqrt{1+z_x^2+z_y^2}$。ルートを忘れていないか。"),
               Choice(val(K0 / (a * b)) if K0 else val(2 * Q), why="曲面上では $z$ は $x,\\ y$ の式。$z$ に平面の式を代入しよう。")]
    zt = lin_tex([(F(-c, a), "x"), (F(-c, b), "y")], c)
    fS = lin_tex([(kx, "x"), (ky, "y")], k0)
    return Problem(
        "スカラー場の面積分", "基本",
        plane_statement(eq) + rf"面積分 $\displaystyle\iint_S\left({ftex}\right)dS$ を求めよ。",
        choices,
        hints=[r"曲面が $z=z(x,y)$ と表せるとき $dS=\sqrt{1+\left(\dfrac{\partial z}{\partial x}\right)^2+\left(\dfrac{\partial z}{\partial y}\right)^2}\,dxdy$。",
               r"$S$ を $xy$ 平面に正射影した三角形 $D$ の上で2重積分する。被積分関数の $z$ には平面の式を代入する。"],
        solution=[rf"$S:\ z={zt}$, $D:\ x\ge0,\ y\ge0,\ \dfrac{{x}}{{{a}}}+\dfrac{{y}}{{{b}}}\le1$",
                  rf"$$dS=\sqrt{{1+\left({tex_fr(F(-c, a))}\right)^2+\left({tex_fr(F(-c, b))}\right)^2}}\,dxdy=\frac{{{tex_root(N)}}}{{{a * b}}}\,dxdy$$",
                  rf"$S$ 上で ${ftex}={fS}$",
                  rf"$$\iint_S\left({ftex}\right)dS=\frac{{{tex_root(N)}}}{{{a * b}}}\iint_D\left({fS}\right)dxdy=\frac{{{tex_root(N)}}}{{{a * b}}}\cdot {tex_fr(K)}={val(Q)}$$"],
        meta=dict(abc=(a, b, c), co=(al, be, ga, de)), figure={"kind": "plane_tri3d", "abc": (a, b, c)})


def flux_parts(rng):
    a, b, c, eq, N = plane_parts(rng)
    Fc = [[rng.choice([-2, -1, 0, 0, 1, 2]) for _ in range(4)] for _ in range(3)]  # (cx, cy, cz, c0)
    if sum(1 for r in Fc if any(r)) < 2:
        raise Retry

    def onS(r):  # F_i を S 上で x, y の式に:(k0, kx, ky)
        cx, cy, cz, c0 = r
        return F(c0 + cz * c), F(cx) - F(cz * c, a), F(cy) - F(cz * c, b)
    w = [F(c, a), F(c, b), F(1)]
    k = [sum(w[i] * onS(Fc[i])[j] for i in range(3)) for j in range(3)]
    I = tri_int(a, b, *k)
    if I == 0:
        raise Retry
    I3 = tri_int(a, b, *onS(Fc[2]))
    Ftex = r"\vec F=\left(" + r",\ ".join(lin_tex([(r[0], "x"), (r[1], "y"), (r[2], "z")], r[3]) for r in Fc) + r"\right)"
    return a, b, c, eq, N, Fc, k, I, I3, Ftex


def flux_solution(a, b, c, Fc, k, I):
    zt = lin_tex([(F(-c, a), "x"), (F(-c, b), "y")], c)
    G = lin_tex([(k[1], "x"), (k[2], "y")], k[0])
    return [rf"$S:\ z={zt}$ より, 上向きの法線ベクトルは "
            rf"$\vec N=\left(-z_x,\ -z_y,\ 1\right)=\left({tex_fr(F(c, a))},\ {tex_fr(F(c, b))},\ 1\right)$ で $\vec n\,dS=\vec N\,dxdy$",
            rf"$S$ 上で $\vec F\cdot\vec N={G}$",
            rf"$$\iint_S\vec F\cdot\vec n\,dS=\iint_D\left({G}\right)dxdy={tex_fr(I)}$$"]


def g_surface_flux(rng):
    a, b, c, eq, N, Fc, k, I, I3, Ftex = flux_parts(rng)
    choices = [Choice(tex_fr(I), True, value=float(I)),
               Choice(tex_fr(-I), why="法線の向きを確かめよう。上向き($z$ 成分が正)の法線を使う。"),
               Choice(tex_fr(I3), why="法線ベクトルは $(0,0,1)$ ではない。$\\vec N=(-z_x,\\ -z_y,\\ 1)$ を使う。"),
               Choice(tex_frac_root((I * a * b).numerator, (I * a * b).denominator * N, N),
                      why="単位法線 $\\vec n$ を使うときは $dS=|\\vec N|\\,dxdy$。$\\vec n\\,dS=\\vec N\\,dxdy$ となる。")]
    return Problem(
        "ベクトル場の面積分(流束)", "標準",
        plane_statement(eq) + rf"${Ftex}$ について, 上向きの単位法線ベクトルを $\vec n$ とするとき, "
        r"$\displaystyle\iint_S\vec F\cdot\vec n\,dS$ を求めよ。",
        choices,
        hints=[r"$z=z(x,y)$ のとき $\vec n\,dS=\left(-\dfrac{\partial z}{\partial x},\ -\dfrac{\partial z}{\partial y},\ 1\right)dxdy$(上向き)。",
               r"$\vec F$ の $z$ に平面の式を代入してから内積をとり、三角形 $D$ で2重積分する。"],
        solution=flux_solution(a, b, c, Fc, k, I),
        meta=dict(abc=(a, b, c), F=Fc), figure={"kind": "plane_tri3d", "abc": (a, b, c), "normal": True})


# ============================================================
# ガウスの発散定理
# ============================================================
def gauss_parts(rng):
    kind = rng.choice(["box", "box", "sphere"])
    if kind == "box":
        a, b, c = rng.randint(1, 3), rng.randint(1, 3), rng.randint(1, 3)
        Fl = [rand_terms(rng, n=2, maxdeg=2, mixed=False) for _ in range(3)]
        dv = simplify(pd(Fl[0], 0) + pd(Fl[1], 1) + pd(Fl[2], 2))
        co = {e: c_ for c_, e in dv}
        c0, cx, cy, cz = (co.get(e, 0) for e in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
        I = (F(c0 * a * b * c) + F(cx * a * a * b * c, 2) + F(cy * a * b * b * c, 2) + F(cz * a * b * c * c, 2), F(0), F(0))
        wrongv = simplify(pd(Fl[0], 1) + pd(Fl[1], 2) + pd(Fl[2], 0))
        wco = {e: c_ for c_, e in wrongv}
        W = (F(wco.get((0, 0, 0), 0) * a * b * c) + F(wco.get((1, 0, 0), 0) * a * a * b * c, 2)
             + F(wco.get((0, 1, 0), 0) * a * b * b * c, 2) + F(wco.get((0, 0, 1), 0) * a * b * c * c, 2), F(0), F(0))
        mco = {e: c_ for c_, e in simplify(pd(Fl[0], 0) + pd(Fl[1], 1))}
        M = (F(mco.get((0, 0, 0), 0) * a * b * c) + F(mco.get((1, 0, 0), 0) * a * a * b * c, 2)
             + F(mco.get((0, 1, 0), 0) * a * b * b * c, 2) + F(mco.get((0, 0, 1), 0) * a * b * c * c, 2), F(0), F(0))
        region = rf"直方体 $V:\ 0\le x\le{a},\ 0\le y\le{b},\ 0\le z\le{c}$"
        fig = {"kind": "box3d", "abc": (a, b, c)}
        Ftex = r"\vec F=\left(" + r",\ ".join(ptex(t) for t in Fl) + r"\right)"
        dtex = ptex(dv) if dv else "0"
        sol_int = rf"$$\iiint_V\left({dtex}\right)dxdydz=\int_0^{{{c}}}\!\!\int_0^{{{b}}}\!\!\int_0^{{{a}}}\left({dtex}\right)dx\,dy\,dz={combo_tex(I)}$$"
        wrong = [(W, "発散は $\\frac{\\partial F_1}{\\partial x}+\\frac{\\partial F_2}{\\partial y}+\\frac{\\partial F_3}{\\partial z}$。偏微分する変数を確かめよう。"),
                 (M, "$\\frac{\\partial F_3}{\\partial z}$ を足し忘れていないか。")]
        meta = dict(kind=kind, abc=(a, b, c), F=Fl)
    else:
        R = rng.choice([1, 2, 3])
        if rng.random() < 0.5:
            m = rng.choice([1, 2, 3])
            Fl = [[(m, (3, 0, 0))], [(m, (0, 3, 0))], [(m, (0, 0, 3))]]
            dtex = rf"{3 * m}(x^2+y^2+z^2)"
            I = (F(0), F(12 * m * R ** 5, 5), F(0))
            sol_int = (rf"$$\iiint_V{3 * m}(x^2+y^2+z^2)\,dV=\int_0^{{2\pi}}\!\!\int_0^{{\pi}}\!\!\int_0^{{{R}}}{3 * m}r^2\cdot r^2\sin\theta\,dr\,d\theta\,d\varphi"
                       rf"={combo_tex(I)}$$")
            wrong = [((F(0), F(4 * m * R ** 3), F(0)), "$x^2+y^2+z^2=r^2$ も積分する($dV=r^2\\sin\\theta\\,dr\\,d\\theta\\,d\\varphi$)。"),
                     ((F(0), F(12 * m * R ** 4), F(0)), "それは $\\mathrm{div}\\,\\vec F$ に球面の面積をかけた値。体積で積分する。")]
        else:
            u = [rng.choice([-1, 1, 2, 3]) for _ in range(3)]
            Fl = [[(u[0], (1, 0, 0)), (rng.choice([0, 1, -1]), (0, 1, 0))],
                  [(u[1], (0, 1, 0)), (rng.choice([0, 2, -1]), (0, 0, 1))],
                  [(u[2], (0, 0, 1)), (rng.choice([0, 1, 3]), (1, 0, 0))]]
            Fl = [[t for t in comp if t[0]] for comp in Fl]
            k = sum(u)
            if k == 0:
                raise Retry
            dtex = str(k)
            I = (F(0), F(4 * k * R ** 3, 3), F(0))
            sol_int = rf"$$\iiint_V{k}\,dV={k}\cdot\frac{{4}}{{3}}\pi\cdot{R}^3={combo_tex(I)}$$"
            wrong = [((F(0), F(4 * k * R ** 2), F(0)), "それは $\\mathrm{div}\\,\\vec F$ に球面の面積をかけた値。体積で積分する。"),
                     ((F(0), F(k * R ** 3), F(0)), "球の体積は $\\frac43\\pi R^3$。")]
        region = rf"球 $V:\ x^2+y^2+z^2\le{R * R}$"
        fig = {"kind": "sphere3d", "R": R}
        Ftex = r"\vec F=\left(" + r",\ ".join(ptex(t) for t in Fl) + r"\right)"
        meta = dict(kind=kind, R=R, F=Fl)
    if combo_float(I) == 0:
        raise Retry
    return region, fig, Ftex, dtex, I, sol_int, wrong, meta


def g_gauss(rng):
    region, fig, Ftex, dtex, I, sol_int, wrong, meta = gauss_parts(rng)
    cand = [(tuple(-x for x in I), "外向きの法線で計算する(発散定理では外向きが正)。")] + wrong + \
           [(tuple(x * 2 for x in I), "積分の計算を確かめよう。"), (tuple(x / 2 for x in I), "積分の計算を確かめよう。")]
    choices = [Choice(combo_tex(I), True, value=combo_float(I))]
    seen = {round(combo_float(I), 9)}
    for v, why in cand:
        if round(combo_float(v), 9) not in seen and len(choices) < 4:
            choices.append(Choice(combo_tex(v), why=why, value=combo_float(v)))
            seen.add(round(combo_float(v), 9))
    return Problem(
        "ガウスの発散定理", "標準",
        rf"{region} の表面を $S$(外向きの単位法線ベクトル $\vec n$)とする. ${Ftex}$ について, ガウスの発散定理を用いて "
        r"$\displaystyle\iint_S\vec F\cdot\vec n\,dS$ を求めよ。",
        choices,
        hints=[r"ガウスの発散定理:$\displaystyle\iint_S\vec F\cdot\vec n\,dS=\iiint_V\nabla\cdot\vec F\,dV$。",
               r"まず $\nabla\cdot\vec F$ を計算し、立体 $V$ で3重積分する。球なら極座標を使う。"],
        solution=[rf"$\nabla\cdot\vec F={dtex}$", r"ガウスの発散定理より", sol_int],
        meta=meta, figure=fig)


def guided_gauss(rng):
    region, fig, Ftex, dtex, I, sol_int, wrong, meta = gauss_parts(rng)
    return ExamItem("ガウスの発散定理",
                    rf"{region} の表面を $S$(外向きの単位法線ベクトル $\vec n$)とし, ${Ftex}$ とする. 次の問いに答えよ.",
                    [Part(r"$\nabla\cdot\vec F$ を求めよ.", dtex, [rf"$\nabla\cdot\vec F={dtex}$"]),
                     Part(r"ガウスの発散定理を用いて, $\displaystyle\iint_S\vec F\cdot\vec n\,dS$ を求めよ.", combo_tex(I), [sol_int])],
                    meta=meta, figure=fig)


# ============================================================
# ストークスの定理
# ============================================================
def stokes_parts(rng):
    R = rng.choice([1, 2, 3])
    a1, a2 = rng.choice([-3, -2, -1, 1, 2]), rng.choice([0, 1, -1, 2])
    b1, b2 = rng.choice([-2, -1, 1, 2, 3]), rng.choice([0, 1, 2])
    c1, c2, c3 = rng.choice([0, 1, -1]), rng.choice([0, 1, 2]), rng.choice([0, 1, -2])
    if b1 == a1 or b1 == -a1:
        raise Retry
    Fl = [[(a1, (0, 1, 0)), (a2, (0, 0, 1))], [(b1, (1, 0, 0)), (b2, (0, 0, 1))],
          [(c1, (1, 0, 0)), (c2, (0, 1, 0)), (c3, (0, 0, 1))]]
    Fl = [[t for t in comp if t[0]] or [(0, (0, 0, 0))] for comp in Fl]
    rot = (c2 - b2, a2 - c1, b1 - a1)
    I = (F(0), F((b1 - a1) * R * R), F(0))
    surf = rng.choice(["hemi", "disk"])
    Ftex = r"\vec F=\left(" + r",\ ".join(ptex(t) if t[0][0] else "0" for t in Fl) + r"\right)"
    if surf == "hemi":
        stext = rf"上半球面 $S:\ x^2+y^2+z^2={R * R},\ z\ge0$(単位法線ベクトル $\vec n$ は外向き)"
        fig = {"kind": "hemisphere3d", "R": R, "orient": True}
    else:
        stext = rf"円板 $S:\ x^2+y^2\le{R * R},\ z=0$(単位法線ベクトル $\vec n=(0,0,1)$)"
        fig = {"kind": "disk3d", "R": R, "orient": True}
    ctext = rf"その境界 $C:\ x^2+y^2={R * R},\ z=0$($z$ 軸の正の側から見て反時計回り)"
    sol = [rf"$\nabla\times\vec F={vtex(list(rot))}$",
           rf"ストークスの定理より, 面積分は $C$ に沿った線積分 $\displaystyle\oint_C\vec F\cdot d\vec r$ に等しい. "
           rf"$C$ を縁とする円板 $D:\ x^2+y^2\le{R * R},\ z=0$($\vec n=(0,0,1)$)で計算してもよいので",
           rf"$$\iint_S(\nabla\times\vec F)\cdot\vec n\,dS=\iint_D{b1 - a1}\,dxdy={b1 - a1}\cdot\pi\cdot{R}^2={combo_tex(I)}$$"]
    return R, Fl, rot, I, stext, ctext, fig, Ftex, sol, (a1, b1)


def g_stokes(rng):
    R, Fl, rot, I, stext, ctext, fig, Ftex, sol, (a1, b1) = stokes_parts(rng)
    choices = [Choice(combo_tex(I), True, value=combo_float(I)),
               Choice(combo_tex(tuple(-x for x in I)), why="$C$ の向きと法線の向きの関係(右ねじ)を確かめよう。"),
               Choice(combo_tex((F(0), F((b1 + a1) * R * R), F(0))),
                      why="回転の $z$ 成分は $\\frac{\\partial F_2}{\\partial x}-\\frac{\\partial F_1}{\\partial y}$(引き算)。"),
               Choice(combo_tex((F(0), F(2 * (b1 - a1) * R * R), F(0))),
                      why="半球面の面積 $2\\pi R^2$ をかけていないか。$(\\nabla\\times\\vec F)\\cdot\\vec n$ は半球面上で一定ではない。"
                          "縁 $C$ が同じ円板で計算する。")]
    return Problem(
        "ストークスの定理", "標準",
        rf"{stext} と, {ctext} を考える. ${Ftex}$ について, ストークスの定理を用いて "
        r"$\displaystyle\iint_S(\nabla\times\vec F)\cdot\vec n\,dS$ を求めよ。",
        choices,
        hints=[r"ストークスの定理:$\displaystyle\iint_S(\nabla\times\vec F)\cdot\vec n\,dS=\oint_C\vec F\cdot d\vec r$。",
               r"縁 $C$ が同じなら、曲面を取りかえても値は同じ。$xy$ 平面上の円板で計算すると楽。"],
        solution=sol, meta=dict(R=R, F=Fl), figure=fig)


def guided_stokes(rng):
    R, Fl, rot, I, stext, ctext, fig, Ftex, sol, _ = stokes_parts(rng)
    return ExamItem("ストークスの定理", rf"{stext} と, {ctext} を考える. ${Ftex}$ について, 次の問いに答えよ.",
                    [Part(r"$\nabla\times\vec F$ を求めよ.", vtex(list(rot)), [sol[0]]),
                     Part(r"ストークスの定理を用いて, $\displaystyle\iint_S(\nabla\times\vec F)\cdot\vec n\,dS$ を求めよ.",
                          combo_tex(I), sol[1:])],
                    meta=dict(R=R, F=Fl), figure=fig)


# ============================================================
# 楕円の焦点と光の反射(法線ベクトル = 勾配)
# ============================================================
def prim(v):
    """有理数のベクトルを、同じ向きの原始的な整数ベクトルに"""
    v = [F(x) for x in v]
    L = 1
    for x in v:
        L = L * x.denominator // _gcd(L, x.denominator)
    w = [int(x * L) for x in v]
    g = 0
    for x in w:
        g = _gcd(g, abs(x))
    return [x // g for x in w]


def ellipse_parts(rng):
    from topics.vec_integrals import ELL, PYTH
    A_, B_ = ELL[rng.randrange(len(ELL))]
    C_ = int(round(math.sqrt(A_ * A_ - B_ * B_)))
    px, py = PYTH[rng.randrange(len(PYTH))]
    sx, sy = rng.choice([1, -1]), rng.choice([1, -1])
    Q = (sx * A_ * px, sy * B_ * py)
    n = (F(2) * Q[0] / A_ ** 2, F(2) * Q[1] / B_ ** 2)
    d = (Q[0] + C_, Q[1])
    k = (d[0] * n[0] + d[1] * n[1]) / (n[0] ** 2 + n[1] ** 2)
    d2 = (d[0] - 2 * k * n[0], d[1] - 2 * k * n[1])
    QF2 = (C_ - Q[0], -Q[1])
    return A_, B_, C_, Q, n, d, k, d2, QF2


def _ftex(x):
    return tex_fr(x)


def _v2(v):
    return r"\left(" + r",\ ".join(_ftex(x) for x in v) + r"\right)"


def ellipse_figs(A_, B_, C_, Q):
    Qf = (float(Q[0]), float(Q[1]))
    base = {"kind": "ellipse", "ab": (A_, B_), "Q": Qf}
    return (dict(base, rays=[((-C_, 0), Qf)]),
            dict(base, rays=[((-C_, 0), Qf), (Qf, (C_, 0))], normal=True))


def ellipse_solution(A_, B_, C_, Q, n, d, k, d2, QF2):
    ratio = d2[0] / QF2[0] if QF2[0] != 0 else d2[1] / QF2[1]
    return [rf"$g(x,y)=\dfrac{{x^2}}{{{A_}^2}}+\dfrac{{y^2}}{{{B_}^2}}$ とおくと, 法線ベクトルは "
            rf"$\vec n=\nabla g(\mathrm Q)=\left(\dfrac{{2x}}{{{A_ * A_}}},\ \dfrac{{2y}}{{{B_ * B_}}}\right)={_v2(n)}$",
            rf"入射ベクトル $\vec d=\overrightarrow{{\mathrm{{F_1Q}}}}={_v2(d)}$, "
            rf"$\dfrac{{\vec d\cdot\vec n}}{{\vec n\cdot\vec n}}={_ftex(k)}$",
            rf"$$\vec d\,'=\vec d-2\frac{{\vec d\cdot\vec n}}{{\vec n\cdot\vec n}}\vec n={_v2(d2)}$$",
            rf"$\overrightarrow{{\mathrm{{QF_2}}}}={_v2(QF2)}$ で $\vec d\,'={_ftex(ratio)}\,\overrightarrow{{\mathrm{{QF_2}}}}$"
            r"(同じ向き)なので, 反射した光は $\mathrm F_2$ を通る."]


def ellipse_statement(A_, B_, C_, Q):
    return (rf"楕円 $\dfrac{{x^2}}{{{A_}^2}}+\dfrac{{y^2}}{{{B_}^2}}=1$ の焦点 $\mathrm F_1({-C_},\ 0)$ から, 楕円上の点 "
            rf"$\mathrm Q{_v2(Q)}$ に向けて光を放つ. ")


MIRROR = (r"ただし, 法線ベクトル $\vec n$ をもつ壁で反射すると, 向き $\vec d$ は "
          r"$\vec d\,'=\vec d-2\dfrac{\vec d\cdot\vec n}{\vec n\cdot\vec n}\vec n$ に変わる.")


def g_ellipse(rng):
    A_, B_, C_, Q, n, d, k, d2, QF2 = ellipse_parts(rng)
    fig, afig = ellipse_figs(A_, B_, C_, Q)
    choices = [Choice(vtex(prim(d2)), True, value=tuple(prim(d2))),
               Choice(vtex(prim(d)), why="それは反射する前の向き $\\overrightarrow{\\mathrm{F_1Q}}$。"),
               Choice(vtex(prim(n)), why="それは法線ベクトルの向き。反射の公式で $\\vec d\\,'$ を計算しよう。"),
               Choice(vtex(prim([-x for x in d2])), why="向きが逆。$\\vec d\\,'$ は楕円の内側に向かう。")]
    return Problem(
        "楕円の焦点と光の反射", "標準",
        ellipse_statement(A_, B_, C_, Q) + "点 $\\mathrm Q$ で反射した光の進む向きを表すベクトルを選べ"
        "(向きが同じなら長さは問わない)。" + MIRROR,
        choices,
        hints=[r"楕円 $g(x,y)=\dfrac{x^2}{a^2}+\dfrac{y^2}{b^2}=1$ の点 $\mathrm Q$ での法線ベクトルは $\nabla g(\mathrm Q)$。",
               r"入射ベクトル $\vec d=\overrightarrow{\mathrm{F_1Q}}$ を反射の公式に代入する。結果はもう一方の焦点 $\mathrm F_2$ を向く。"],
        solution=ellipse_solution(A_, B_, C_, Q, n, d, k, d2, QF2),
        meta=dict(A=A_, B=B_, Q=Q), figure=fig, answer_figure=afig,
        exam_statement=ellipse_statement(A_, B_, C_, Q) + "点 $\\mathrm Q$ で反射した光が, もう一方の焦点 $\\mathrm F_2$ を"
        "通ることを示せ. " + MIRROR,
        exam_answer="")


def guided_ellipse(rng):
    A_, B_, C_, Q, n, d, k, d2, QF2 = ellipse_parts(rng)
    fig, afig = ellipse_figs(A_, B_, C_, Q)
    sol = ellipse_solution(A_, B_, C_, Q, n, d, k, d2, QF2)
    return ExamItem("楕円の焦点と光の反射", ellipse_statement(A_, B_, C_, Q) + MIRROR + " 次の問いに答えよ.",
                    [Part(rf"$g(x,y)=\dfrac{{x^2}}{{{A_}^2}}+\dfrac{{y^2}}{{{B_}^2}}$ とおくとき, 点 $\mathrm Q$ における法線ベクトル "
                          r"$\nabla g(\mathrm Q)$ を求めよ.", _v2(n), [sol[0]]),
                     Part(r"点 $\mathrm Q$ で反射した光が, もう一方の焦点 $\mathrm F_2$ を通ることを示せ.", "", sol[1:])],
                    meta=dict(A=A_, B=B_, Q=Q), figure=fig, answer_figure=afig)


REGISTRY = [
    ProblemType("sline", "スカラー場の線積分", "基本", g_scalar_line),
    ProblemType("vline", "ベクトル場の線積分", "基本", g_vector_line),
    ProblemType("gline", "勾配場の線積分", "基本", g_grad_line, guided=guided_grad_line),
    ProblemType("potential", "保存場のポテンシャル", "標準", g_potential),
    ProblemType("green", "グリーンの定理", "標準", g_green, guided=guided_green),
    ProblemType("garea", "グリーンの定理と面積", "標準", g_area_green),
    ProblemType("ssurf", "スカラー場の面積分", "基本", g_surface_scalar),
    ProblemType("flux", "ベクトル場の面積分(流束)", "標準", g_surface_flux),
    ProblemType("gauss", "ガウスの発散定理", "標準", g_gauss, guided=guided_gauss),
    ProblemType("stokes", "ストークスの定理", "標準", g_stokes, guided=guided_stokes),
    ProblemType("ellipse", "楕円の焦点と光の反射", "標準", g_ellipse, guided=guided_ellipse),
]
GUIDED_NAMES = {"gline": "勾配と勾配場の線積分", "green": "グリーンの定理(被積分関数 → 線積分)",
                "gauss": "発散 → ガウスの発散定理", "stokes": "回転 → ストークスの定理",
                "ellipse": "法線ベクトル → 反射光が焦点を通ること"}


def tab():
    render("ch3", REGISTRY, "第3章 線積分・面積分")
