"""数学IV 偏微分方程式 0. 変数分離法:演習の問題生成

型と記号は topics/pde_sep.py と同じ。G(y) = ∫_0^y g(s) ds。
"""
from fractions import Fraction

from topics.ex_core import Choice, ExamItem, Part, Problem, ProblemType, Retry, render, tex_fr
from topics.pde_sep import DX, DY, cterm, gs, gtex, xexp, yexp, gfun, init_tex, lin, pde_tex, series

F = Fraction
GK = ["cos", "sin", "y", "y2", "exp"]
HINT_SEP = r"$u=X(x)Y(y)$ を代入して $XY$ で割り, 「$x$ だけの式 $=$ $y$ だけの式 $=\lambda$(定数)」の形にする。"
HINT_G = r"$\dfrac{Y^{\prime}}{Y}=h(y)$ なら $Y=e^{\int_0^yh(s)\,ds}$($Y(0)=1$)。"


def distinct(ok, cands):
    choices = [Choice(ok, True, value=ok)]
    for t, why in cands:
        if all(c.tex != t for c in choices) and len(choices) < 4:
            choices.append(Choice(t, why=why, value=t))
    if len(choices) < 4:
        raise Retry
    return choices


def system(typ, A, B, C, gt, cs, as_):
    return (r"\begin{cases}" + pde_tex(typ, A, B, C, gt) + r"\\ u(x,0)=" + init_tex(cs, as_) + r"\end{cases}")


def params1(rng):
    A, B, C = rng.choice([1, 1, 2]), rng.choice([-3, -2, -1, 1, 2, 3]), rng.choice([1, 1, -1, 2])
    gk, k = rng.choice(GK), rng.choice([1, 2, 3])
    return A, B, C, gk, k


def init_terms(rng, n, lo=-3, hi=8, avoid=()):
    while True:
        as_ = rng.sample([a for a in range(lo, hi + 1) if a != 0 and a not in avoid], n)
        cs = [rng.choice([1, 2, 3, 4, 5, 8, -1, -2, -3, -5, -9]) for _ in range(n)]
        if n == 1:
            cs = [rng.choice([1, 2, 3, 5])]
        return cs, as_


# 1階の解の項:指数 [(係数, 本体)] を誤りの種類ごとに作る
def exp1(a, A, B, C, Gc, Gb, err=None):
    L = F(A * a + B)
    if err == "lam":          # λ = a としてしまう(A, B を忘れる)
        L = F(a)
    if err == "lamsign":      # λ = A a − B
        L = F(A * a - B)
    yc, gc = L / C, -Gc / C
    if err == "gsign":        # G の符号の誤り
        gc = -gc
    if err == "noC":          # C で割り忘れ
        yc, gc = L, -Gc
    return [(a, "x"), (yc, "y"), (gc, Gb)]


def sol1(cs, as_, A, B, C, Gc, Gb, err=None):
    return series(cs, [exp1(a, A, B, C, Gc, Gb, err) for a in as_])


WHY1 = {"lam": r"$e^{ax}$ の項は $X=e^{\frac{\lambda-B}{A}x}$ と比べて $\lambda=Aa+B$。",
        "lamsign": r"$AX^{\prime}+BX=\lambda X$ より $X^{\prime}=\frac{\lambda-B}{A}X$:$\lambda=Aa+B$。",
        "gsign": r"$CY^{\prime}=(\lambda-g(y))Y$ なので $G(y)$ の項の符号は $-$(右辺の $g(y)u$ を左へ移すと符号が変わる)。",
        "noC": r"$CY^{\prime}=(\lambda-g)Y$ の両辺を $C$ で割ってから積分する。"}


# ------------------------------------------------------------
# 基本:分離して得られる常微分方程式
# ------------------------------------------------------------
def coefd(c, body):
    return body if c == 1 else ("-" + body if c == -1 else str(c) + body)


def odes_tex(A, B, C, gt, sB=1, sg=1):
    xl = coefd(A, r"X^{\prime}")
    yl = coefd(C, r"Y^{\prime}")
    lb = r"\lambda" + cterm(-sB * B, "")
    lg = r"\lambda" + ("-" if sg > 0 else "+") + gt
    return xl + "=(" + lb + r")X,\quad " + yl + "=(" + lg + ")Y"


def g_sep(rng):
    A, B, C, gk, k = params1(rng)
    gt = gfun(gk, k)[0]
    ok = odes_tex(A, B, C, gt)
    cands = [(odes_tex(A, B, C, gt, sB=-1), r"$AX^{\prime}+BX=\lambda X$ より $AX^{\prime}=(\lambda-B)X$。"),
             (odes_tex(A, B, C, gt, sg=-1), r"$g(y)+C\dfrac{Y^{\prime}}{Y}=\lambda$ より $CY^{\prime}=(\lambda-g(y))Y$。"),
             (odes_tex(A, B, C, gt, sB=-1, sg=-1), "移項したときの符号を確かめよう。"),
             (odes_tex(C, B, A, gt) if A != C else odes_tex(A, -B, C, gt, sg=-1), r"$X^{\prime}$ には $A$, $Y^{\prime}$ には $C$ がかかる。")]
    return Problem("分離して得られる常微分方程式", "基本",
                   rf"偏微分方程式 ${pde_tex(0, A, B, C, gt)}$ に $u=X(x)Y(y)$ を代入し, 分離定数を $\lambda$ として変数を分離した。"
                   rf"$X$, $Y$ がみたす常微分方程式の組として正しいものを選べ。",
                   distinct(ok, cands), hints=[HINT_SEP, r"$A\dfrac{X^{\prime}}{X}+B=g(y)+C\dfrac{Y^{\prime}}{Y}=\lambda$。"],
                   solution=["代入して $XY$ で割ると $" + lin([(A, DX), (B, "")]) + "=" + lin([(1, gt), (C, DY)]) + r"=\lambda$。",
                             rf"よって ${ok}$。"], meta=dict(A=A, B=B, C=C, gk=gk, k=k))


# ------------------------------------------------------------
# 基本:初期条件が1項のときの解
# ------------------------------------------------------------
def g_term(rng):
    A, B, C, gk, k = params1(rng)
    gt, Gc, Gb, _, _ = gfun(gk, k)
    cs, as_ = init_terms(rng, 1)
    ok = sol1(cs, as_, A, B, C, Gc, Gb)
    cands = [(sol1(cs, as_, A, B, C, Gc, Gb, e), WHY1[e]) for e in ("lam", "gsign", "noC", "lamsign")]
    return Problem("初期条件が1項のときの解", "基本",
                   rf"次の問題の解を変数分離で求めよ。$${system(0, A, B, C, gt, cs, as_)}$$",
                   distinct(ok, cands), hints=[HINT_SEP, HINT_G, r"$u(x,0)=X(x)Y(0)=X(x)$ を初期条件と比べて $\lambda$ を決める。"],
                   solution=sol_lines(A, B, C, gt, Gc, Gb, cs, as_, ok), meta=dict(A=A, B=B, C=C, gk=gk, k=k, cs=cs, as_=as_))


def sol_lines(A, B, C, gt, Gc, Gb, cs, as_, ok):
    G = gtex(Gc, Gb)
    ls = ", ".join(rf"$e^{{{lin([(a, 'x')])}}}$ には $\lambda={A * a + B}$" for a in as_)
    return ["$X=e^{" + xexp(A, B) + "}$, $Y=e^{" + yexp(C) + r"}$, $G(y)=\int_0^y" + gs(gt) + r"\,ds=" + G + "$。",
            ls + "。", rf"$u(x,y)={ok}$。"]


# ------------------------------------------------------------
# 標準:初期条件が2項のときの解(重ね合わせ)
# ------------------------------------------------------------
def g_solve(rng):
    A, B, C, gk, k = params1(rng)
    gt, Gc, Gb, _, _ = gfun(gk, k)
    cs, as_ = init_terms(rng, 2)
    ok = sol1(cs, as_, A, B, C, Gc, Gb)
    cands = [(sol1(cs, as_, A, B, C, Gc, Gb, e), WHY1[e]) for e in ("lam", "gsign", "noC", "lamsign")]
    return Problem("重ね合わせで初期条件に合わせる", "標準",
                   rf"次の問題の解を変数分離で求めよ。$${system(0, A, B, C, gt, cs, as_)}$$",
                   distinct(ok, cands), hints=[HINT_SEP, HINT_G, r"初期条件の各項 $e^{ax}$ に合わせて別々の $\lambda$ を選び, 足し合わせる(重ね合わせ)。"],
                   solution=sol_lines(A, B, C, gt, Gc, Gb, cs, as_, ok), meta=dict(A=A, B=B, C=C, gk=gk, k=k, cs=cs, as_=as_))


# ------------------------------------------------------------
# 標準:2階の型
# ------------------------------------------------------------
def exp2(typ, a, B, Gc, Gb, err=None):
    if typ == 1:
        L = F(a + B)
        if err == "lam":
            L = F(a)
        if err == "inv":                      # λ で割らずに掛ける
            return [(a, "x"), (Gc * L, Gb)]
        if err == "lamsign":
            L = F(a - B)
            if L == 0:
                raise Retry
        return [(a, "x"), (Gc / L, Gb)]
    L = F(a * a + B)
    if err == "lam":
        L = F(a + B)
    if err == "lamsign":
        L = F(a * a - B)
    gc = -Gc if err != "gsign" else Gc
    return [(a, "x"), (L, "y"), (gc, Gb)]


WHY2 = {1: {"lam": r"$\dfrac{X^{\prime}}{X}+B=\lambda$ より $X=e^{(\lambda-B)x}$:$\lambda=a+B$。",
            "inv": r"$\dfrac{Y^{\prime}}{Y}=\dfrac{g(y)}{\lambda}$:$\lambda$ で割る。",
            "lamsign": r"$X^{\prime}=(\lambda-B)X$ と比べて $\lambda=a+B$。"},
        2: {"lam": r"$X^{\prime\prime}=(\lambda-B)X$ に $X=e^{ax}$ を代入すると $a^2=\lambda-B$。",
            "lamsign": r"$a^2=\lambda-B$ より $\lambda=a^2+B$。",
            "gsign": r"$Y^{\prime}=(\lambda-g(y))Y$ なので $G(y)$ の項の符号は $-$。"}}


def g_2nd(rng):
    typ = rng.choice([1, 2])
    B = rng.choice([-2, -1, 0, 1, 2, 3])
    gk = rng.choice(GK + (["zero"] if typ == 2 else []))
    k = rng.choice([1, 2])
    gt, Gc, Gb, _, _ = gfun(gk, k)
    n = rng.choice([1, 2])
    cs, as_ = init_terms(rng, n, -3, 4, avoid=(-B,) if typ == 1 else ())
    ok = series(cs, [exp2(typ, a, B, Gc, Gb) for a in as_])
    errs = ("lam", "inv", "lamsign") if typ == 1 else ("lam", "lamsign", "gsign")
    cands = []
    for e in errs:
        try:
            cands.append((series(cs, [exp2(typ, a, B, Gc, Gb, e) for a in as_]), WHY2[typ][e]))
        except Retry:
            pass
    cands.append((series(cs, [[(a, "x")] for a in as_]), r"$Y(y)$ の部分を忘れている($y=0$ でだけ初期条件と一致すればよい)。"))
    return Problem("2階の偏微分方程式", "標準",
                   rf"次の問題の変数分離形の解(初期条件の各項に $X(x)Y(y)$ の形の解を対応させて重ね合わせたもの)を求めよ。$${system(typ, 1, B, 1, gt, cs, as_)}$$",
                   distinct(ok, cands), hints=[HINT_SEP, "2階①は $\\left(\\dfrac{X^{\\prime}}{X}+B\\right)\\dfrac{Y^{\\prime}}{Y}=g(y)$ となるので, $\\dfrac{X^{\\prime}}{X}+B=\\lambda$ とおく。"
                                               if typ == 1 else r"2階②は $\dfrac{X^{\prime\prime}}{X}+B=\dfrac{Y^{\prime}}{Y}+g(y)=\lambda$。$X=e^{ax}$ なら $a^2=\lambda-B$。"],
                   solution=sol2_lines(typ, B, gt, Gc, Gb, as_, ok), meta=dict(typ=typ, B=B, gk=gk, k=k, cs=cs, as_=as_))


def sol2_lines(typ, B, gt, Gc, Gb, as_, ok):
    G = gtex(Gc, Gb)
    if typ == 1:
        first = rf"$X=e^{{(\lambda{cterm(-B, '')})x}}$, $Y=e^{{G(y)/\lambda}}$, $G(y)=\int_0^y{gs(gt)}\,ds={G}$。"
        ls = ", ".join(rf"$e^{{{lin([(a, 'x')])}}}$ には $\lambda={a + B}$" for a in as_)
    else:
        first = rf"$X=e^{{ax}}$($a^2=\lambda{cterm(-B, '')}$), $Y=e^{{\lambda y-G(y)}}$, $G(y)={G}$。"
        ls = ", ".join(rf"$e^{{{lin([(a, 'x')])}}}$ には $\lambda={a * a + B}$" for a in as_)
    return [first, ls + "。", rf"$u(x,y)={ok}$。"]


# ------------------------------------------------------------
# 標準:熱伝導方程式(橋渡し)
# ------------------------------------------------------------
def heat_sol(cs, ns, kap, err=None):
    out, first = "", True
    for c, n in zip(cs, ns):
        if err == "linear":
            rate = kap * n
            body = r"\pi t"
        elif err == "nopi":
            rate = kap * n * n
            body = "t"
        else:
            rate = kap * n * n
            body = r"\pi^2t"
        e = cterm(-rate if err != "plus" else rate, body, True)
        s = r"\sin " + ("" if n == 1 else str(n)) + r"\pi x"
        out += cterm(c, "e^{" + e + "}" + s, first)
        first = False
    return out


def g_heat(rng):
    kap = rng.choice([F(1), F(2), F(1, 2), F(3)])
    ns = sorted(rng.sample([1, 2, 3, 4], 2))
    cs = [rng.choice([1, 2, 3, 5]), rng.choice([-1, -2, 4, 6])]
    ok = heat_sol(cs, ns, kap)
    cands = [(heat_sol(cs, ns, kap, "linear"), r"$X^{\prime\prime}=-\mu X$, $X=\sin n\pi x$ より $\mu=(n\pi)^2$:2乗になる。"),
             (heat_sol(cs, ns, kap, "nopi"), r"$\mu=(n\pi)^2=n^2\pi^2$:$\pi^2$ を忘れない。"),
             (heat_sol(cs, ns, kap, "plus"), r"$T^{\prime}=-\kappa\mu T$:温度は時間とともに減衰する。")]
    kt = "" if kap == 1 else tex_fr(kap)
    f = cterm(cs[0], r"\sin " + ("" if ns[0] == 1 else str(ns[0])) + r"\pi x", True) + cterm(cs[1], r"\sin " + str(ns[1]) + r"\pi x")
    return Problem("熱伝導方程式(境界条件つき)", "標準",
                   rf"次の問題の解を変数分離で求めよ。$$\begin{{cases}}\dfrac{{\partial u}}{{\partial t}}={kt}\dfrac{{\partial^2u}}{{\partial x^2}}\ (0<x<1,\ t>0)\\ "
                   rf"u(0,t)=u(1,t)=0\\ u(x,0)={f}\end{{cases}}$$",
                   distinct(ok, cands), hints=[r"$u=X(x)T(t)$ とおくと $\dfrac{T^{\prime}}{\kappa T}=\dfrac{X^{\prime\prime}}{X}=-\mu$。",
                                               r"境界条件をみたすのは $X=\sin n\pi x$, $\mu=(n\pi)^2$, このとき $T=e^{-\kappa(n\pi)^2t}$。"],
                   solution=[r"$X_n=\sin n\pi x$, $T_n=e^{-\kappa n^2\pi^2t}$ を初期条件の各項に合わせて重ね合わせる。", rf"$u(x,t)={ok}$。"],
                   meta=dict(kap=kap, ns=ns, cs=cs))


# ------------------------------------------------------------
# 誘導
# ------------------------------------------------------------
def guided_solve(rng):
    A, B, C, gk, k = params1(rng)
    gt, Gc, Gb, _, _ = gfun(gk, k)
    cs, as_ = init_terms(rng, 2)
    ok = sol1(cs, as_, A, B, C, Gc, Gb)
    G = gtex(Gc, Gb)
    xe, ye = xexp(A, B), yexp(C)
    return ExamItem("変数分離法(1階)",
                    rf"次の問題について, 以下の問いに答えよ. $$" + system(0, A, B, C, gt, cs, as_) + "$$",
                    [Part(r"$u=X(x)Y(y)$ とおき, 分離定数を $\lambda$ として, $X(0)=Y(0)=1$ をみたす $X$, $Y$ を $\lambda$ を用いて表せ.",
                          rf"X=e^{{{xe}}},\ Y=e^{{{ye}}}\ \left(G(y)={G}\right)",
                          ["代入して $XY$ で割ると $" + lin([(A, DX), (B, "")]) + "=" + lin([(1, gt), (C, DY)]) + r"=\lambda$。",
                           rf"$X=e^{{{xe}}}$, $Y=e^{{{ye}}}$, $G(y)=\int_0^y{gs(gt)}\,ds={G}$。"]),
                     Part(r"初期条件をみたす解 $u(x,y)$ を求めよ.", rf"u(x,y)={ok}",
                          sol_lines(A, B, C, gt, Gc, Gb, cs, as_, ok)[1:])], meta=dict(A=A, B=B, C=C, gk=gk, k=k, cs=cs, as_=as_))


def guided_2nd(rng):
    B = rng.choice([-2, -1, 1, 2, 3])
    gk, k = rng.choice(GK), rng.choice([1, 2])
    gt, Gc, Gb, _, _ = gfun(gk, k)
    cs, as_ = init_terms(rng, 2, -3, 4, avoid=(-B,))
    ok = series(cs, [exp2(1, a, B, Gc, Gb) for a in as_])
    Gt = gtex(Gc, Gb)
    return ExamItem("変数分離法(2階)",
                    rf"次の問題について, 以下の問いに答えよ. $$" + system(1, 1, B, 1, gt, cs, as_) + "$$",
                    [Part(r"$u=X(x)Y(y)$ とおき, $\dfrac{X^{\prime}}{X}" + cterm(B, "") + r"=\lambda\ (\lambda\neq0)$ とする. $X(0)=Y(0)=1$ をみたす $X$, $Y$ を $\lambda$ を用いて表せ.",
                          r"X=e^{(\lambda" + cterm(-B, "") + r")x},\ Y=e^{\frac{" + Gt + r"}{\lambda}}",
                          [r"代入して $XY$ で割ると $\left(\dfrac{X^{\prime}}{X}" + cterm(B, "") + r"\right)\dfrac{Y^{\prime}}{Y}=" + gt + "$。",
                           r"$\dfrac{Y^{\prime}}{Y}=\dfrac{" + gt + r"}{\lambda}$ より $Y=e^{\frac{1}{\lambda}\int_0^y" + gs(gt) + r"\,ds}=e^{\frac{" + Gt + r"}{\lambda}}$。"]),
                     Part(r"初期条件の各項に対応する変数分離形の解を重ね合わせて, $u(x,y)$ を求めよ.", rf"u(x,y)={ok}",
                          sol2_lines(1, B, gt, Gc, Gb, as_, ok)[1:])], meta=dict(B=B, gk=gk, k=k, cs=cs, as_=as_))


def guided_heat(rng):
    kap = rng.choice([F(1), F(2), F(1, 2)])
    ns = sorted(rng.sample([1, 2, 3], 2))
    cs = [rng.choice([1, 2, 3]), rng.choice([-1, 2, 4])]
    ok = heat_sol(cs, ns, kap)
    kt = "" if kap == 1 else tex_fr(kap)
    f = cterm(cs[0], r"\sin " + ("" if ns[0] == 1 else str(ns[0])) + r"\pi x", True) + cterm(cs[1], r"\sin " + str(ns[1]) + r"\pi x")
    return ExamItem("熱伝導方程式の変数分離",
                    r"次の問題について, 以下の問いに答えよ. $$\begin{cases}\dfrac{\partial u}{\partial t}=" + kt + r"\dfrac{\partial^2u}{\partial x^2}\ (0<x<1,\ t>0)\\ "
                    r"u(0,t)=u(1,t)=0\\ u(x,0)=" + f + r"\end{cases}$$",
                    [Part(r"$u=X(x)T(t)$ とおくと $X^{\prime\prime}=-\mu X$, $X(0)=X(1)=0$ となる. $\mu>0$ のとき, これをみたす $0$ でない $X$ と $\mu$ を求めよ.",
                          r"X=\sin n\pi x,\ \mu=(n\pi)^2\ (n=1,2,3,\dots)",
                          [r"$X=p\cos\sqrt\mu x+q\sin\sqrt\mu x$, $X(0)=0$ より $p=0$。", r"$X(1)=q\sin\sqrt\mu=0$, $q\ne0$ より $\sqrt\mu=n\pi$。"]),
                     Part(r"初期条件をみたす解 $u(x,t)$ を求めよ.", rf"u(x,t)={ok}",
                          [rf"$T^{{\prime}}=-{kt}\mu T$ より $T=e^{{-{kt}n^2\pi^2t}}$。", rf"初期条件の各項に合わせて重ね合わせ, $u(x,t)={ok}$。"])],
                    meta=dict(kap=kap, ns=ns, cs=cs))


REGISTRY = [
    ProblemType("ps_sep", "分離して得られる常微分方程式", "基本", g_sep),
    ProblemType("ps_term", "初期条件が1項のときの解", "基本", g_term),
    ProblemType("ps_solve", "重ね合わせで初期条件に合わせる", "標準", g_solve, guided=guided_solve),
    ProblemType("ps_2nd", "2階の偏微分方程式", "標準", g_2nd, guided=guided_2nd),
    ProblemType("ps_heat", "熱伝導方程式(境界条件つき)", "標準", g_heat, guided=guided_heat),
]
GUIDED_NAMES = {"ps_solve": "X, Y を λ で表す → 初期条件をみたす解(1階)",
                "ps_2nd": "X, Y を λ で表す → 重ね合わせた解(2階 u_xy)",
                "ps_heat": "境界条件から X と μ → 熱伝導方程式の解"}


def tab():
    render("psep", REGISTRY, "偏微分方程式 0. 変数分離法")
