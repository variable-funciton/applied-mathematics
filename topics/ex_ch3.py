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


def ellipse_parts(rng, kind):
    from topics.vec_integrals import ellipse_case, ellipse_count
    pick = rng.randrange(ellipse_count(kind))
    return ellipse_case(kind, pick, rng.choice([1, -1]))   # x 軸型の Q は右半分のみ(左半分だと分数が出る)


def _ftex(x):
    return tex_fr(x)


def _v2(v):
    return r"\left(" + r",\ ".join(_ftex(x) for x in v) + r"\right)"


def _pt(P):
    return r"\left(" + r",\ ".join(str(x) for x in P) + r"\right)"


def ellipse_figs(e):
    Qf = (float(e.Q[0]), float(e.Q[1]))
    base = {"kind": "ellipse", "ab": (e.a, e.b), "theta": e.theta, "foci": (e.F1, e.F2), "Q": Qf,
            "nvec": (float(e.n[0]), float(e.n[1]))}
    return (dict(base, rays=[(e.F1, Qf)]),
            dict(base, rays=[(e.F1, Qf), (Qf, e.F2)], normal=True))


def ellipse_solution(e):
    ratio = e.d2[0] / e.QF2[0] if e.QF2[0] != 0 else e.d2[1] / e.QF2[1]
    return [rf"$g(x,y)={e.eq}$ とおくと, $\nabla g(\mathrm Q)=\left({e.gform[0]},\ {e.gform[1]}\right)={_v2(e.grad)}$. "
            rf"反射の公式は $\vec n$ を定数倍しても変わらないので, 同じ向きの $\vec n={_v2(e.n)}$ を使う",
            rf"入射ベクトル $\vec d=\overrightarrow{{\mathrm{{F_1Q}}}}={_v2(e.d)}$, "
            rf"$\dfrac{{\vec d\cdot\vec n}}{{\vec n\cdot\vec n}}={_ftex(e.k)}$",
            rf"$$\vec d\,'=\vec d-2\frac{{\vec d\cdot\vec n}}{{\vec n\cdot\vec n}}\vec n={_v2(e.d2)}$$",
            rf"$\overrightarrow{{\mathrm{{QF_2}}}}={_v2(e.QF2)}$ で $\vec d\,'={_ftex(ratio)}\,\overrightarrow{{\mathrm{{QF_2}}}}$"
            r"(同じ向き)なので, 反射した光は $\mathrm F_2$ を通る."]


def ellipse_statement(e):
    if e.kind == "tilt":
        return (rf"楕円 ${e.eq}={e.rhs}$ の焦点は $\mathrm F_1{_pt(e.F1)}$, $\mathrm F_2{_pt(e.F2)}$ である. "
                rf"$\mathrm F_1$ から楕円上の点 $\mathrm Q{_pt(e.Q)}$ に向けて光を放つ. ")
    return (rf"楕円 ${e.eq}={e.rhs}$ の焦点 $\mathrm F_1{_pt(e.F1)}$ から, 楕円上の点 "
            rf"$\mathrm Q{_pt(e.Q)}$ に向けて光を放つ. ")


MIRROR = (r"ただし, 法線ベクトル $\vec n$ をもつ壁で反射すると, 向き $\vec d$ は "
          r"$\vec d\,'=\vec d-2\dfrac{\vec d\cdot\vec n}{\vec n\cdot\vec n}\vec n$ に変わる"
          r"(この式は $\vec n$ を定数倍しても変わらない).")

ELLIPSE_TITLES = {"x": "楕円の焦点と光の反射", "y": "楕円の焦点と光の反射(焦点が y 軸上)",
                  "tilt": "斜めの楕円の焦点と光の反射"}


def g_ellipse(rng, kind="x"):
    e = ellipse_parts(rng, kind)
    fig, afig = ellipse_figs(e)
    choices = [Choice(vtex(prim(e.d2)), True, value=tuple(prim(e.d2))),
               Choice(vtex(prim(e.d)), why="それは反射する前の向き $\\overrightarrow{\\mathrm{F_1Q}}$。"),
               Choice(vtex(prim(e.n)), why="それは法線ベクトルの向き。反射の公式で $\\vec d\\,'$ を計算しよう。"),
               Choice(vtex(prim([-x for x in e.d2])), why="向きが逆。$\\vec d\\,'$ は楕円の内側に向かう。")]
    if len({c.tex for c in choices}) < 4:
        raise Retry
    hint_g = (r"$xy$ の項も忘れずに偏微分する。" if kind == "tilt" else "")
    return Problem(
        ELLIPSE_TITLES[kind], "標準",
        ellipse_statement(e) + "点 $\\mathrm Q$ で反射した光の進む向きを表すベクトルを選べ"
        "(向きが同じなら長さは問わない)。" + MIRROR,
        choices,
        hints=[r"楕円 $g(x,y)=" + e.rhs + r"$ の点 $\mathrm Q$ での法線ベクトルは $\nabla g(\mathrm Q)$。" + hint_g
               + r"向きだけ使うので, 定数倍して整数ベクトルにしてよい。",
               r"入射ベクトル $\vec d=\overrightarrow{\mathrm{F_1Q}}$ を反射の公式に代入する。結果はもう一方の焦点 $\mathrm F_2$ を向く。"],
        solution=ellipse_solution(e),
        meta=dict(kind=kind, eq=e.eq, point=e.Q), figure=fig, answer_figure=afig,
        exam_statement=ellipse_statement(e) + "点 $\\mathrm Q$ で反射した光が, もう一方の焦点 $\\mathrm F_2$ を"
        "通ることを示せ. " + MIRROR,
        exam_answer="")


def guided_ellipse(rng, kind="x"):
    e = ellipse_parts(rng, kind)
    fig, afig = ellipse_figs(e)
    sol = ellipse_solution(e)
    return ExamItem(ELLIPSE_TITLES[kind], ellipse_statement(e) + MIRROR + " 次の問いに答えよ.",
                    [Part(rf"$g(x,y)={e.eq}$ とおくとき, 点 $\mathrm Q$ における法線ベクトル "
                          r"$\nabla g(\mathrm Q)$ を求めよ.", _v2(e.grad), [sol[0]]),
                     Part(r"点 $\mathrm Q$ で反射した光が, もう一方の焦点 $\mathrm F_2$ を通ることを示せ.", "", sol[1:])],
                    meta=dict(kind=kind, eq=e.eq, point=e.Q), figure=fig, answer_figure=afig)


def _ellipse_kind(kind):
    return (lambda rng: g_ellipse(rng, kind)), (lambda rng: guided_ellipse(rng, kind))


g_ellipse_y, guided_ellipse_y = _ellipse_kind("y")
g_ellipse_tilt, guided_ellipse_tilt = _ellipse_kind("tilt")

# ============================================================
# 標準:曲線をつないだ C1 + C2 に沿う線積分(半円 + 直径 / 線分 + 四分円)
# ============================================================
def wallis(m, n):
    """∫_0^{π/2} cos^m t sin^n t dt を (有理数, π の係数) で"""
    if m >= 2:
        a, b = wallis(m - 2, n)
        k = F(m - 1, m + n)
        return (k * a, k * b)
    if n >= 2:
        a, b = wallis(m, n - 2)
        k = F(n - 1, m + n)
        return (k * a, k * b)
    return {(0, 0): (F(0), F(1, 2)), (1, 0): (F(1), F(0)), (0, 1): (F(1), F(0)), (1, 1): (F(1, 2), F(0))}[(m, n)]


def trig_int(m, n, upper):
    """∫_0^upper cos^m sin^n(upper は "half" = π/2 か "pi" = π)"""
    a, b = wallis(m, n)
    if upper == "half":
        return (a, b)
    k = 1 + (-1) ** m
    return (k * a, k * b)


def arc_integral(P, Q, R, upper):
    """x = R cos t, y = R sin t (0 ≤ t ≤ upper) に沿う ∫ P dx + Q dy を (定数, π, 0) で"""
    c, p = F(0), F(0)
    for co, (i, j, _) in P:          # P dx = P · (−R sin t) dt
        a, b = trig_int(i, j + 1, upper)
        k = -co * R ** (i + j + 1)
        c, p = c + k * a, p + k * b
    for co, (i, j, _) in Q:          # Q dy = Q · (R cos t) dt
        a, b = trig_int(i + 1, j, upper)
        k = co * R ** (i + j + 1)
        c, p = c + k * a, p + k * b
    return (c, p, F(0))


def arc_integral_bad(P, Q, R, upper):
    """よくある間違い:dx = R sin t dt としてしまう"""
    c, p, _ = arc_integral([(-co, e) for co, e in P], Q, R, upper)
    return (c, p, F(0))


def xaxis_integral(P, a, b):
    """y = 0 の線分 (t, 0)(a ≤ t ≤ b)に沿う ∫ P dx(dy = 0)"""
    tot = F(0)
    for co, (i, j, _) in P:
        if j == 0:
            tot += F(co) * (F(b) ** (i + 1) - F(a) ** (i + 1)) / (i + 1)
    return (tot, F(0), F(0))


def cadd(*vs):
    return tuple(sum(v[k] for v in vs) for k in range(3))


def cneg(v):
    return tuple(-x for x in v)


def p_on_axis(P):
    """P(t, 0)"""
    return ptex([(c, (i, 0, 0)) for c, (i, j, _) in P if j == 0]).replace("x", "t") if any(j == 0 for _, (i, j, _) in P) else "0"


def p_on_arc(T, R, which):
    """曲線上の P, Q を t で表した文字列(x = R cos t, y = R sin t)"""
    parts = []
    for co, (i, j, _) in T:
        k = co * R ** (i + j)
        body = (r"\cos" + (f"^{i}" if i > 1 else "") + " t" if i else "") + (r"\sin" + (f"^{j}" if j > 1 else "") + " t" if j else "")
        parts.append((k, body))
    return lin_tex([(k, b) for k, b in parts if b], sum(k for k, b in parts if not b))


def piece_parts(rng):
    mons = [(1, 0, 0), (0, 1, 0), (2, 0, 0), (1, 1, 0), (0, 2, 0)]
    cf = [-3, -2, -1, 1, 2, 3]
    P = [(rng.choice(cf), e) for e in rng.sample(mons, 2)]
    Q = [(rng.choice(cf), e) for e in rng.sample(mons, 2)]
    R = rng.choice([1, 2, 3])
    fam = rng.choice(["half", "half", "quarter"])
    if fam == "half":
        I1 = arc_integral(P, Q, R, "pi")                 # C1:上半円 A(R,0) → B(-R,0)
        I2 = xaxis_integral(P, -R, R)                    # C2:線分 B → A
        bad1 = cadd(arc_integral_bad(P, Q, R, "pi"), I2)
        I2rev = cneg(I2)
    else:
        I1 = xaxis_integral(P, 0, R)                     # C1:線分 O → A(R,0)
        I2 = arc_integral(P, Q, R, "half")               # C2:四分円 A → B(0,R)
        bad1 = cadd(I1, arc_integral_bad(P, Q, R, "half"))
        I2rev = cneg(I2)
    I = cadd(I1, I2)
    if combo_float(I) == 0 or combo_float(I1) == 0 or combo_float(I2) == 0:
        raise Retry
    return fam, P, Q, R, I1, I2, I, bad1, I2rev


def piece_text(fam, P, Q, R):
    from topics.ex_fig import arc_pts, seg_pts
    PQ = rf"\int_C\left({ptex(P)}\right)dx+\left({ptex(Q)}\right)dy"
    if fam == "half":
        st = (rf"原点を中心とする半径 ${R}$ の上半円 $C_1$ を $\mathrm{{A}}({R},0)$ から $\mathrm{{B}}({-R},0)$ まで反時計回りに進み, "
              rf"続けて線分 $C_2$ を $\mathrm{{B}}$ から $\mathrm{{A}}$ まで進む曲線を $C=C_1+C_2$ とする. ")
        fig = {"kind": "path2d", "fill": True,
               "pieces": [{"pts": arc_pts(0, 0, R, 0, math.pi), "label": "C_1"}, {"pts": seg_pts((-R, 0), (R, 0)), "label": "C_2"}],
               "points": [(R, 0, "A"), (-R, 0, "B")]}
    else:
        st = (rf"原点 $\mathrm{{O}}$ から $\mathrm{{A}}({R},0)$ までの線分を $C_1$, 原点を中心とする半径 ${R}$ の円周上を "
              rf"$\mathrm{{A}}$ から $\mathrm{{B}}(0,{R})$ まで反時計回りに進む四分円を $C_2$ とし, $C=C_1+C_2$ とする. ")
        fig = {"kind": "path2d",
               "pieces": [{"pts": seg_pts((0, 0), (R, 0)), "label": "C_1"}, {"pts": arc_pts(0, 0, R, 0, math.pi / 2), "label": "C_2"}],
               "points": [(0, 0, "O"), (R, 0, "A"), (0, R, "B")]}
    return st, PQ, fig


def piece_solution(fam, P, Q, R, I1, I2, I):
    arc = (rf"$x={coef_tex(R, COS_T)},\ y={coef_tex(R, SIN_T)}$ より "
           rf"$dx={coef_tex(-R, SIN_T)}\,dt,\ dy={coef_tex(R, COS_T)}\,dt$")
    PQt = rf"曲線上で $P={p_on_arc(P, R, 'P')},\ Q={p_on_arc(Q, R, 'Q')}$"
    if fam == "half":
        s1 = [rf"$C_1:\ \vec r(t)=({coef_tex(R, COS_T)},\ {coef_tex(R, SIN_T)})\ (0\le t\le\pi)$. " + arc,
              PQt, rf"$$\int_{{C_1}}P\,dx+Q\,dy={combo_tex(I1)}$$"]
        s2 = [rf"$C_2:\ \vec r(t)=(t,\ 0)\ ({-R}\le t\le {R})$ では $y=0,\ dx=dt,\ dy=0$ なので",
              rf"$$\int_{{C_2}}P\,dx+Q\,dy=\int_{{{-R}}}^{{{R}}}\left({p_on_axis(P)}\right)dt={combo_tex(I2)}$$"]
    else:
        s1 = [rf"$C_1:\ \vec r(t)=(t,\ 0)\ (0\le t\le {R})$ では $y=0,\ dx=dt,\ dy=0$ なので",
              rf"$$\int_{{C_1}}P\,dx+Q\,dy=\int_0^{{{R}}}\left({p_on_axis(P)}\right)dt={combo_tex(I1)}$$"]
        s2 = [rf"$C_2:\ \vec r(t)=({coef_tex(R, COS_T)},\ {coef_tex(R, SIN_T)})\ \left(0\le t\le\frac{{\pi}}{{2}}\right)$. " + arc,
              PQt, rf"$$\int_{{C_2}}P\,dx+Q\,dy={combo_tex(I2)}$$"]
    tot = [rf"$$\int_C=\int_{{C_1}}+\int_{{C_2}}={combo_tex(I)}$$"]
    if fam == "half":
        qx = [(c * e[0], (e[0] - 1, e[1], 0)) for c, e in Q if e[0]]
        py = [(c * e[1], (e[0], e[1] - 1, 0)) for c, e in P if e[1]]
        g = {}
        for c, e in qx:
            g[e] = g.get(e, 0) + c
        for c, e in py:
            g[e] = g.get(e, 0) - c
        c0, c1, c2 = g.get((0, 0, 0), 0), g.get((1, 0, 0), 0), g.get((0, 1, 0), 0)
        G = (F(0) + F(2 * c2 * R ** 3, 3), F(c0 * R * R, 2), F(0))
        tot.append(rf"(検算)$C$ は半円板 $D$ の周を反時計回りに一周するので, グリーンの定理より "
                   rf"$\displaystyle\iint_D\left(\frac{{\partial Q}}{{\partial x}}-\frac{{\partial P}}{{\partial y}}\right)dxdy"
                   rf"=\iint_D\left({lin_tex([(c1, 'x'), (c2, 'y')], c0)}\right)dxdy={combo_tex(G)}$ と一致する。")
    return s1, s2, tot


def g_piece_line(rng):
    fam, P, Q, R, I1, I2, I, bad1, I2rev = piece_parts(rng)
    st, PQ, fig = piece_text(fam, P, Q, R)
    s1, s2, tot = piece_solution(fam, P, Q, R, I1, I2, I)
    cand = [(cadd(I1, I2rev), "$C_2$ の向きを確かめよう。逆向きに積分すると符号が変わる。"),
            (I1, "$C_1$ と $C_2$ の両方の線積分を足す($\\int_C=\\int_{C_1}+\\int_{C_2}$)。"),
            (I2, "$C_1$ と $C_2$ の両方の線積分を足す($\\int_C=\\int_{C_1}+\\int_{C_2}$)。"),
            (bad1, "円周上では $x=R\\cos t$ なので $dx=-R\\sin t\\,dt$。符号を確かめよう。"),
            (cneg(I), "向き(始点から終点へ)を確かめよう。")]
    choices = [Choice(combo_tex(I), True, value=combo_float(I))]
    seen = {round(combo_float(I), 9)}
    for v, why in cand:
        fv = round(combo_float(v), 9)
        if fv not in seen and len(choices) < 4:
            choices.append(Choice(combo_tex(v), why=why, value=combo_float(v)))
            seen.add(fv)
    return Problem(
        "曲線をつないだ線積分", "標準",
        st + rf"線積分 $\displaystyle {PQ}$ を求めよ。",
        choices,
        hints=[r"$\displaystyle\int_{C_1+C_2}=\int_{C_1}+\int_{C_2}$。曲線ごとにベクトル関数で表して計算し、足す。",
               r"円周 $x=R\cos t,\ y=R\sin t$ では $dx=-R\sin t\,dt,\ dy=R\cos t\,dt$。$x$ 軸上の線分では $y=0,\ dy=0$。"],
        solution=s1 + s2 + tot, meta=dict(fam=fam, P=P, Q=Q, R=R, I1=I1, I2=I2, I=I), figure=fig)


def guided_piece_line(rng):
    fam, P, Q, R, I1, I2, I, bad1, I2rev = piece_parts(rng)
    st, PQ, fig = piece_text(fam, P, Q, R)
    s1, s2, tot = piece_solution(fam, P, Q, R, I1, I2, I)
    pq = rf"\left({ptex(P)}\right)dx+\left({ptex(Q)}\right)dy"
    return ExamItem("曲線をつないだ線積分", st + r"次の問いに答えよ.",
                    [Part(rf"$\displaystyle\int_{{C_1}}{pq}$ を求めよ.", combo_tex(I1), s1),
                     Part(rf"$\displaystyle\int_{{C_2}}{pq}$ と $\displaystyle\int_C{pq}$ を求めよ.",
                          rf"\int_{{C_2}}={combo_tex(I2)},\quad \int_C={combo_tex(I)}", s2 + tot)],
                    meta=dict(fam=fam, P=P, Q=Q, R=R, I1=I1, I2=I2, I=I), figure=fig)


# ============================================================
# 基本:逆向きの曲線 −C に沿う線積分
# ============================================================
def reverse_parts(rng, fam):
    A = [rng.randint(-2, 2) for _ in range(3)]
    if fam == "vector":
        d = [rng.choice([-2, -1, 1, 2, 3]) for _ in range(3)]
    else:
        d = list(rng.choice(NICE))
        rng.shuffle(d)
        d = [x * rng.choice([1, -1]) for x in d]
    B = [A[i] + d[i] for i in range(3)]
    curve = r"\vec r(t)=\left(" + r",\ ".join(lin_tex([(d[i], "t")], A[i]) for i in range(3)) + r"\right)\ (0\le t\le 1)"
    def back(b, k):            # b − k t(係数が負なら定数を先に書く:1 − t)
        if k > 0 and b != 0:
            return tex_fr(b) + lin_tex([(-k, "t")], 0)
        return lin_tex([(-k, "t")], b)
    rcurve = r"\left(" + r",\ ".join(back(B[i], d[i]) for i in range(3)) + r"\right)\ (0\le t\le 1)"
    return A, d, B, curve, rcurve


def vector_reverse(rng):
    A, d, B, curve, rcurve = reverse_parts(rng, "vector")
    Fl = [rand_terms(rng, n=2, maxdeg=2, mixed=False) for _ in range(3)]
    g = [F(0)]
    for i in range(3):
        g = padd(g, [x * d[i] for x in subst(Fl[i], A, d)])
    I = pint01(g)
    nd = [-x for x in d]
    gr = [F(0)]                                  # −C に沿って直接計算
    for i in range(3):
        gr = padd(gr, [x * nd[i] for x in subst(Fl[i], B, nd)])
    Ir = pint01(gr)
    if I == 0 or Ir != -I:
        raise Retry
    while len(gr) > 1 and gr[-1] == 0:
        gr = gr[:-1]
    Ftex = r"\left(" + r",\ ".join(ptex(c) for c in Fl) + r"\right)"
    sol = [rf"$-C$ は $\mathrm{{B}}$ から $\mathrm{{A}}$ へ向かうので $\vec r(t)={rcurve}$ と表せる"
           r"($\vec r(1-t)$ としてもよい)。",
           rf"$\dfrac{{d\vec r}}{{dt}}={vtex(nd)}$ より $\vec F\cdot\dfrac{{d\vec r}}{{dt}}={tpoly_tex(gr)}$",
           rf"$$\int_{{-C}}\vec F\cdot d\vec r=\int_0^1\left({tpoly_tex(gr)}\right)dt={tex_fr(Ir)}$$",
           rf"($C$ に沿って計算すると ${tex_fr(I)}$ で, $\int_{{-C}}\vec F\cdot d\vec r=-\int_C\vec F\cdot d\vec r$ が成り立っている。)"]
    return A, d, B, curve, rcurve, Fl, Ftex, I, Ir, sol


def g_reverse_line(rng):
    fam = rng.choice(["vector", "scalar"])
    if fam == "vector":
        A, d, B, curve, rcurve, Fl, Ftex, I, Ir, sol = vector_reverse(rng)
        gw = [F(0)]
        for i in range(3):
            gw = padd(gw, pmul(subst(Fl[i], A, d), [F(A[i]), F(d[i])]))
        endv = sum(ev(Fl[i], A) * d[i] for i in range(3))
        cand = [(I, "向きを逆にすると $d\\vec r$ の向きが逆になり、ベクトル場の線積分は符号が変わる。"),
                (-pint01(gw), "内積の相手は $\\vec r(t)$ ではなく $\\dfrac{d\\vec r}{dt}$。"),
                (F(-endv), "1点での値だけでは線積分にならない。$t$ の式にしてから積分する。"),
                (2 * Ir, "$\\int_0^1 t^n\\,dt=\\frac{1}{n+1}$。積分の計算を確かめよう。")]
        stmt = (rf"曲線 $C:\ {curve}$ と逆向きの曲線を $-C$ とする. "
                rf"線積分 $\displaystyle\int_{{-C}}{Ftex}\cdot d\vec r$ を求めよ。")
        hints = [r"$-C$ は終点から始点へ向かう曲線。$\vec r(1-t)\ (0\le t\le1)$ で表せる。",
                 r"ベクトル場の線積分は向きで符号が変わる:$\displaystyle\int_{-C}\vec F\cdot d\vec r=-\int_C\vec F\cdot d\vec r$。"]
        meta = dict(fam=fam, A=A, d=d, F=Fl, I=I)
        ans = Ir
    else:
        A, d, B, curve, rcurve = reverse_parts(rng, "scalar")
        n = round(math.sqrt(sum(x * x for x in d)))
        co = [rng.choice([-2, -1, 1, 2, 3]) for _ in range(3)]
        k0 = rng.randint(-3, 3)
        M = [F(A[i]) + F(d[i], 2) for i in range(3)]
        fM = sum(co[i] * M[i] for i in range(3)) + k0
        if fM == 0:
            raise Retry
        J = n * fM
        fs = lin_tex([(co[i], V[i]) for i in range(3)], k0)
        ft = [F(sum(co[i] * B[i] for i in range(3)) + k0), F(-sum(co[i] * d[i] for i in range(3)))]
        cand = [(-J, "$ds=|\\vec r\\,'(t)|\\,dt$ はいつも正。スカラー場の線積分は向きを変えても値は変わらない。"),
                (fM, f"$ds=|\\vec r\\,'(t)|\\,dt$。$|\\vec r\\,'|={n}$ をかけ忘れていないか。"),
                (-fM, f"$|\\vec r\\,'|={n}$ をかけ忘れ、さらに符号を変えていないか。スカラー場の線積分は向きによらない。"),
                (n * n * fM, "$|\\vec r\\,'|=\\sqrt{x'^2+y'^2+z'^2}$。ルートを忘れていないか。")]
        stmt = (rf"曲線 $C:\ {curve}$ と逆向きの曲線を $-C$ とする. "
                rf"線積分 $\displaystyle\int_{{-C}}\left({fs}\right)ds$ を求めよ。")
        hints = [r"$-C$ は $\vec r(1-t)\ (0\le t\le1)$ で表せる。$ds=\left|\dfrac{d\vec r}{dt}\right|dt$。",
                 r"スカラー場の線積分 $\int f\,ds$ は、向きを変えても値が変わらない($ds>0$)。"]
        sol = [rf"$-C:\ \vec r(t)={rcurve}$ とすると $\dfrac{{d\vec r}}{{dt}}={vtex([-x for x in d])}$, "
               rf"$\left|\dfrac{{d\vec r}}{{dt}}\right|=\sqrt{{{n * n}}}={n}$",
               rf"曲線上では $f={tpoly_tex(ft)}$",
               rf"$$\int_{{-C}}f\,ds=\int_0^1\left({tpoly_tex(ft)}\right)\cdot{n}\,dt={tex_fr(J)}$$",
               r"($ds$ は向きによらず正なので, $\int_{-C}f\,ds=\int_C f\,ds$。ベクトル場の線積分のように符号は変わらない。)"]
        meta = dict(fam=fam, A=A, d=d, co=co, k0=k0)
        ans = J
    choices = [Choice(tex_fr(ans), True, value=float(ans))]
    seen = {ans}
    for v, why in cand:
        if v not in seen and len(choices) < 4:
            choices.append(Choice(tex_fr(v), why=why, value=float(v)))
            seen.add(v)
    return Problem("逆向きの曲線に沿う線積分", "基本", stmt, choices, hints=hints, solution=sol, meta=meta,
                   figure={"kind": "curve3d", "pts": curve_pts(lambda t: tuple(B[i] - d[i] * t for i in range(3)), 0, 1)})


def guided_reverse_line(rng):
    A, d, B, curve, rcurve, Fl, Ftex, I, Ir, sol = vector_reverse(rng)
    return ExamItem("逆向きの曲線に沿う線積分",
                    rf"曲線 $C:\ {curve}$ と逆向きの曲線を $-C$ とする. 次の問いに答えよ.",
                    [Part(r"$-C$ を表すベクトル関数を, $t$ の範囲も含めて1つ求めよ.", rf"\vec r(t)={rcurve}", sol[:1]),
                     Part(rf"線積分 $\displaystyle\int_{{-C}}{Ftex}\cdot d\vec r$ を求めよ.", tex_fr(Ir), sol[1:])],
                    meta=dict(fam="vector", A=A, d=d, F=Fl, I=I))


REGISTRY = [
    ProblemType("sline", "スカラー場の線積分", "基本", g_scalar_line),
    ProblemType("vline", "ベクトル場の線積分", "基本", g_vector_line),
    ProblemType("gline", "勾配場の線積分", "基本", g_grad_line, guided=guided_grad_line),
    ProblemType("reverse", "逆向きの曲線に沿う線積分", "基本", g_reverse_line, guided=guided_reverse_line),
    ProblemType("piece", "曲線をつないだ線積分", "標準", g_piece_line, guided=guided_piece_line),
    ProblemType("potential", "保存場のポテンシャル", "標準", g_potential),
    ProblemType("green", "グリーンの定理", "標準", g_green, guided=guided_green),
    ProblemType("garea", "グリーンの定理と面積", "標準", g_area_green),
    ProblemType("ssurf", "スカラー場の面積分", "基本", g_surface_scalar),
    ProblemType("flux", "ベクトル場の面積分(流束)", "標準", g_surface_flux),
    ProblemType("gauss", "ガウスの発散定理", "標準", g_gauss, guided=guided_gauss),
    ProblemType("stokes", "ストークスの定理", "標準", g_stokes, guided=guided_stokes),
    ProblemType("ellipse", "楕円の焦点と光の反射", "標準", g_ellipse, guided=guided_ellipse),
    ProblemType("ellipse_y", "楕円の焦点と光の反射(焦点が y 軸上)", "標準", g_ellipse_y, guided=guided_ellipse_y),
    ProblemType("ellipse_tilt", "斜めの楕円の焦点と光の反射", "標準", g_ellipse_tilt, guided=guided_ellipse_tilt),
]
GUIDED_NAMES = {"gline": "勾配と勾配場の線積分", "green": "グリーンの定理(被積分関数 → 線積分)",
                "gauss": "発散 → ガウスの発散定理", "stokes": "回転 → ストークスの定理",
                "ellipse": "法線ベクトル → 反射光が焦点を通ること",
                "ellipse_y": "法線ベクトル → 反射光が焦点を通ること(焦点が y 軸上)",
                "ellipse_tilt": "法線ベクトル → 反射光が焦点を通ること(斜めの楕円)",
                "reverse": "−C のベクトル関数 → −C に沿う線積分", "piece": "C₁, C₂ ごとの線積分 → C₁+C₂ に沿う線積分"}


def tab():
    render("ch3", REGISTRY, "第3章 線積分・面積分")
