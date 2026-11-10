"""数学IV ラプラス変換 第2章 0. 線形微分方程式の解(準備):演習の問題生成

1階 y' + a y = f, 2階 y'' + a y' + b y = f(定数係数, 独立変数 t)
"""
from fractions import Fraction

from topics.ex_core import Choice, ExamItem, Part, Problem, ProblemType, Retry, render
from topics.laplace_ode import char_roots, e_t, fr, general_tex, particular, ssum, trig

F = Fraction


def distinct(ok, cands):
    choices = [Choice(ok, True, value=ok)]
    for t, why in cands:
        if all(c.tex != t for c in choices) and len(choices) < 4:
            choices.append(Choice(t, why=why, value=t))
    if len(choices) < 4:
        raise Retry
    return choices


def eq2(a, b):
    return ssum([(1, "y^{\\prime\\prime}"), (a, "y^{\\prime}"), (b, "y")])


def eq1(a):
    return ssum([(1, "y^{\\prime}"), (a, "y")])


def ex(r):
    return e_t(r) or ""


# ------------------------------------------------------------
# 基本:1階
# ------------------------------------------------------------
def g_first(rng):
    a = rng.choice([1, 2, 3, -1, -2])
    K = rng.choice([1, 2, 3, -1, -2])
    y0 = rng.choice([0, 1, 2, -1, 3])
    if y0 == K:
        raise Retry
    kind = rng.choice(["const", "exp"])
    if kind == "const":
        c, k = a * K, 0
        ftex = str(c)
    else:
        k = rng.choice([kk for kk in (-3, -2, -1, 1, 2, 3) if kk != -a])
        c = (k + a) * K
        ftex = ssum([(c, ex(k))])
    yp = ex(k)
    ok = ssum([(y0 - K, ex(-a)), (K, yp)])
    cands = [(ssum([(y0 - K, ex(a)), (K, yp)]), r"斉次方程式 $y^{\prime}+ay=0$ の解は $e^{-at}$。"),
             (ssum([(y0, ex(-a)), (K, yp)]), r"$y=Ce^{-at}+y_p$ に $t=0$ を代入して $C$ を決める:$C=y(0)-y_p(0)$。"),
             (ssum([(y0 + K, ex(-a)), (-K, yp)]), "特殊解の符号を確かめよう(代入して検算)。"),
             (ssum([(y0 - F(c, a), ex(-a)), (F(c, a), yp)]) if kind == "exp" else ssum([(y0 + K, ex(-a)), (-K, "")]),
              r"$y_p=Ae^{kt}$ を代入すると $(k+a)A=c$。" if kind == "exp" else r"$y_p=A$ を代入すると $aA=c$。")]
    return Problem("1階線形微分方程式", "基本",
                   rf"微分方程式 ${eq1(a)}={ftex}$, $y(0)={y0}$ を解け。",
                   distinct(ok, cands), hints=[r"一般解は $y=Ce^{-at}+y_p$($y_p$ は特殊解)。",
                                               r"右辺が定数なら $y_p=A$, $ce^{kt}$ なら $y_p=Ae^{kt}$ とおいて代入する。"],
                   solution=[rf"斉次方程式の解は $C{ex(-a)}$。特殊解は $y_p={ssum([(K, yp)])}$。",
                             rf"$y(0)=C{'+' if K > 0 else ''}{K}={y0}$ より $C={y0 - K}$。$y={ok}$。"],
                   meta=dict(a=a, c=c, k=k, y0=y0, K=K, kind=kind))


# ------------------------------------------------------------
# 基本:斉次方程式の一般解
# ------------------------------------------------------------
def char_case(rng, kind=None):
    kind = kind or rng.choice(["real", "double", "complex"])
    if kind == "real":
        r1, r2 = rng.sample([-4, -3, -2, -1, 1, 2, 3], 2)
        return kind, -(r1 + r2), r1 * r2, (r1, r2)
    if kind == "double":
        r = rng.choice([-3, -2, -1, 1, 2, 3])
        return kind, -2 * r, r * r, (r,)
    p, q = rng.choice([-2, -1, 0, 1, 2]), rng.choice([1, 2, 3])
    if p == q:
        raise Retry
    return kind, -2 * p, p * p + q * q, (p, q)


def cs_tex(p, q):
    qq = "" if q == 1 else str(q)
    return ex(p) + rf"\left(C_1\cos {qq}t+C_2\sin {qq}t\right)"


def g_char(rng):
    kind, a, b, r = char_case(rng)
    ok = general_tex(char_roots(a, b))
    if kind == "real":
        r1, r2 = r
        cands = [(rf"C_1{ex(-r1)}+C_2{ex(-r2)}", r"特性方程式の解 $\lambda$ をそのまま指数に使う($y=e^{\lambda t}$)。符号に注意。"),
                 (rf"(C_1+C_2t){ex(r1)}", "異なる2つの実数解なので $t$ は掛からない。"),
                 (rf"C_1{ex(r1 + r2)}+C_2{ex(r1 * r2)}", r"根と係数を取り違えていないか:$\lambda^2+a\lambda+b=0$ を解く。")]
    elif kind == "double":
        (rr,) = r
        cands = [(rf"(C_1+C_2t){ex(-rr)}", "符号に注意。"), (rf"C_1{ex(rr)}+C_2{ex(-rr)}", r"重解なので2つ目の解は $te^{\lambda t}$。"),
                 (rf"(C_1+C_2t^2){ex(rr)}", r"重解のとき2つ目の解は $te^{\lambda t}$。")]
    else:
        p, q = r
        cands = [(cs_tex(q, p) if p != 0 else cs_tex(q, 1), r"$\lambda=p\pm qi$ なら $e^{pt}(\cos qt,\ \sin qt)$:実部が指数, 虚部が角振動数。"),
                 (cs_tex(-p, q) if p != 0 else rf"C_1{ex(q)}+C_2{ex(-q)}", "符号に注意。" if p != 0 else r"虚数解 $\pm qi$ からは $\cos qt,\ \sin qt$ が出る。"),
                 (rf"C_1{ex(p + q)}+C_2{ex(p - q)}", r"虚数解なので $e^{(p\pm qi)t}=e^{pt}(\cos qt\pm i\sin qt)$ を使う。")]
    return Problem("斉次方程式の一般解", "基本",
                   rf"微分方程式 ${eq2(a, b)}=0$ の一般解を求めよ($C_1,C_2$ は任意定数)。",
                   distinct(ok, cands), hints=[r"$y=e^{\lambda t}$ を代入して特性方程式 $\lambda^2+a\lambda+b=0$ を作る。",
                                               "異なる実数解・重解・虚数解で一般解の形が変わる。"],
                   solution=[rf"特性方程式 ${ssum([(1, r'\lambda^2'), (a, r'\lambda'), (b, '')])}=0$ を解く。", rf"一般解は $y={ok}$。"],
                   meta=dict(kind=kind, a=a, b=b, r=r))


# ------------------------------------------------------------
# 基本:ロンスキアン
# ------------------------------------------------------------
def g_wron(rng):
    kind, a, b, r = char_case(rng)
    if kind == "real":
        r1, r2 = sorted(r)
        y1, y2 = ex(r1), ex(r2)
        ok = ssum([(r2 - r1, ex(r1 + r2))])
        cands = [(ssum([(r1 - r2, ex(r1 + r2))]), r"$W=y_1y_2^{\prime}-y_1^{\prime}y_2$ の順に注意。"),
                 (ssum([(r2 - r1, ex(r2 - r1))]), r"$e^{r_1t}e^{r_2t}=e^{(r_1+r_2)t}$。"), (ssum([(1, ex(r1 + r2))]) if r2 - r1 != 1 else ssum([(2, ex(r1 + r2))]), r"微分すると係数 $r_1,r_2$ が出る。"),
                 (ssum([(r1 * r2, ex(r1 + r2))]), r"$W=y_1y_2^{\prime}-y_1^{\prime}y_2=(r_2-r_1)e^{(r_1+r_2)t}$。")]
    elif kind == "double":
        (rr,) = r
        y1, y2 = ex(rr), "t" + ex(rr)
        ok = ssum([(1, ex(2 * rr))])
        cands = [(ssum([(rr, ex(2 * rr))]), r"$y_2^{\prime}=(1+\lambda t)e^{\lambda t}$ を使って計算し直そう。"),
                 (ssum([(1, "t" + ex(2 * rr))]), r"$t$ の項は打ち消し合う。"), (ssum([(-1, ex(2 * rr))]), r"$W=y_1y_2^{\prime}-y_1^{\prime}y_2$ の順に注意。"),
                 (ssum([(1, ex(rr))]), r"$e^{\lambda t}\cdot e^{\lambda t}=e^{2\lambda t}$。")]
    else:
        p, q = r
        qq = "" if q == 1 else str(q)
        y1, y2 = ex(p) + rf"\cos {qq}t", ex(p) + rf"\sin {qq}t"
        ok = ssum([(q, ex(2 * p))])
        cands = [(ssum([(1, ex(2 * p))]) if q != 1 else ssum([(2, ex(2 * p))]), r"$\sin qt$ を微分すると $q$ が出る。"), (ssum([(-q, ex(2 * p))]), r"$W=y_1y_2^{\prime}-y_1^{\prime}y_2$ の順に注意。"),
                 (ssum([(q, ex(p))]), r"$e^{pt}\cdot e^{pt}=e^{2pt}$。"), (ssum([(p if p else 2 * q, ex(2 * p))]), r"$\cos^2+\sin^2=1$ を使って整理すると $q e^{2pt}$。")]
    return Problem("ロンスキアン", "基本",
                   rf"微分方程式 ${eq2(a, b)}=0$ の基本解 $y_1={y1 or '1'}$, $y_2={y2}$ のロンスキアン $W=y_1y_2^{{\prime}}-y_1^{{\prime}}y_2$ を求めよ。",
                   distinct(ok, cands), hints=[r"$W=\begin{vmatrix}y_1&y_2\\y_1^{\prime}&y_2^{\prime}\end{vmatrix}=y_1y_2^{\prime}-y_1^{\prime}y_2$。"],
                   solution=[rf"$W={ok}$($\neq0$ なので $y_1,y_2$ は基本解になっている)。"], meta=dict(kind=kind, a=a, b=b, r=r))


# ------------------------------------------------------------
# 標準:未定係数法
# ------------------------------------------------------------
def g_undet(rng):
    kind = rng.choice(["exp", "exp_res", "cos", "sin"])
    if kind.startswith("exp"):
        r1, r2 = rng.sample([-3, -2, -1, 1, 2, 3], 2)
        a, b = -(r1 + r2), r1 * r2
        k = r1 if kind == "exp_res" else rng.choice([kk for kk in (-3, -2, -1, 1, 2, 3, 4) if kk not in (r1, r2)])
        c = rng.choice([1, 2, 3, 4, 6, -2, -3])
        _, _, ok, _ = particular(a, b, "exp", c, k)
        ftex = ssum([(c, ex(k))])
        if kind == "exp":
            P = k * k + a * k + b
            cands = [(ssum([(F(c, k * k - a * k + b) if k * k - a * k + b else F(c, 7), ex(k))]), r"$y_p=Ae^{kt}$ を代入すると $A(k^2+ak+b)=c$。$ak$ の符号に注意。"),
                     (ssum([(F(c, k * k + b) if k * k + b else F(c, 5), ex(k))]), r"$y_p^{\prime}=kAe^{kt}$ の項 $akA$ を忘れていないか。"),
                     (ssum([(F(c, P), "t" + ex(k))]), r"$k$ は特性根でないので $t$ は掛けない。"), (ssum([(F(-c, P), ex(k))]), "符号を確かめよう。")]
        else:
            dP = 2 * k + a
            cands = [(ssum([(F(c, dP), ex(k))]), r"$k$ が特性根なので $Ae^{kt}$ は斉次方程式の解になる。$t$ を掛ける。"),
                     (ssum([(F(c, 2 * k), "t" + ex(k))]), r"$y_p=Ate^{kt}$ を代入すると $A(2k+a)=c$。"),
                     (ssum([(F(-c, dP), "t" + ex(k))]), "符号を確かめよう。"), (ssum([(F(c, 2), "t^2" + ex(k))]), r"単根なので $t$ を1回だけ掛ける。")]
    else:
        a, b = rng.choice([1, 2, 3, 4]), rng.choice([1, 2, 3, 5, 6])
        w = rng.choice([1, 2])
        if b == w * w:
            raise Retry
        c = rng.choice([1, 2, 5, 10, -5])
        _, _, ok, _ = particular(a, b, kind, c, 1, w)
        _, _, swp, _ = particular(a, b, "sin" if kind == "cos" else "cos", c, 1, w)
        _, _, noa, _ = particular(0, b, kind, c, 1, w)
        _, _, nega, _ = particular(-a, b, kind, c, 1, w)
        ftex = ssum([(c, trig(kind, w))])
        cands = [(swp, r"$\cos$ と $\sin$ の係数を取り違えていないか(代入して確かめよう)。"),
                 (noa, r"$ay_p^{\prime}$ の項を忘れていないか。"), (nega, r"$y_p^{\prime}$ の符号:$(\cos\omega t)^{\prime}=-\omega\sin\omega t$。")]
    return Problem("未定係数法による特殊解", "標準",
                   rf"微分方程式 ${eq2(a, b)}={ftex}$ の特殊解 $y_p$ を1つ求めよ。",
                   distinct(ok, cands), hints=[r"右辺が $ce^{kt}$ なら $y_p=Ae^{kt}$($k$ が特性根なら $Ate^{kt}$)。",
                                               r"右辺が $c\cos\omega t$ や $c\sin\omega t$ なら $y_p=A\cos\omega t+B\sin\omega t$ とおいて係数を比べる。"],
                   solution=[particular(a, b, "exp" if kind.startswith("exp") else kind, c, k if kind.startswith("exp") else 1, 1 if kind.startswith("exp") else w)[3],
                             rf"$y_p={ok}$。"],
                   meta=dict(kind=kind, a=a, b=b, c=c, k=k if kind.startswith("exp") else None, w=w if not kind.startswith("exp") else None))


# ------------------------------------------------------------
# 標準:初期値問題(右辺が定数)
# ------------------------------------------------------------
def ivp_parts(rng):
    r1, r2 = rng.sample([-3, -2, -1, 1, 2], 2)
    a, b = -(r1 + r2), r1 * r2
    K = rng.choice([1, 2, 3, -1, -2])
    C1, C2 = rng.choice([1, 2, -1, -2, 3]), rng.choice([1, 2, -1, -3])
    y0, v0 = C1 + C2 + K, C1 * r1 + C2 * r2
    return r1, r2, a, b, K, b * K, C1, C2, y0, v0


def g_ivp(rng):
    r1, r2, a, b, K, c, C1, C2, y0, v0 = ivp_parts(rng)
    ok = ssum([(C1, ex(r1)), (C2, ex(r2)), (K, "")])
    # 特殊解を考えずに定数を決めた誤り
    D1 = F(v0 - r2 * y0, r1 - r2)
    D2 = F(y0) - D1
    cands = [(ssum([(D1, ex(r1)), (D2, ex(r2)), (K, "")]), r"初期条件は $y=C_1y_1+C_2y_2+y_p$ 全体に代入する($y_p$ も含める)。"),
             (ssum([(C1, ex(-r1)), (C2, ex(-r2)), (K, "")]), "特性根の符号に注意。"),
             (ssum([(C2, ex(r1)), (C1, ex(r2)), (K, "")]), r"$C_1,C_2$ の対応を確かめよう($y(0),\ y^{\prime}(0)$ を代入して検算)。"),
             (ssum([(C1 + 2 * K, ex(r1)), (C2 - K, ex(r2)), (-K, "")]), r"特殊解 $y_p=A$ は $bA=c$ から。")]
    return Problem("初期値問題", "標準",
                   rf"微分方程式 ${eq2(a, b)}={c}$, $y(0)={y0}$, $y^{{\prime}}(0)={v0}$ を解け。",
                   distinct(ok, cands), hints=[r"一般解 $y=C_1y_1+C_2y_2+y_p$。右辺が定数なので $y_p=A$(定数)。",
                                               r"$y(0)$, $y^{\prime}(0)$ を代入して $C_1,C_2$ の連立方程式を解く。"],
                   solution=ivp_sol(r1, r2, a, b, K, c, C1, C2, y0, v0), meta=dict(r=(r1, r2), a=a, b=b, c=c, y0=y0, v0=v0, C=(C1, C2, K)))


def ivp_sol(r1, r2, a, b, K, c, C1, C2, y0, v0):
    return [rf"特性方程式 ${ssum([(1, r'\lambda^2'), (a, r'\lambda'), (b, '')])}=0$ より $\lambda={r1},\ {r2}$。特殊解 $y_p={K}$($bA={c}$)。",
            rf"$y=C_1{ex(r1)}+C_2{ex(r2)}{'+' if K > 0 else ''}{K}$ に初期条件を代入:$C_1+C_2{'+' if K > 0 else ''}{K}={y0}$, ${ssum([(r1, 'C_1'), (r2, 'C_2')])}={v0}$。",
            rf"$C_1={C1}$, $C_2={C2}$ より $y={ssum([(C1, ex(r1)), (C2, ex(r2)), (K, '')])}$。"]


# ------------------------------------------------------------
# 誘導
# ------------------------------------------------------------
def guided_ivp(rng):
    r1, r2, a, b, K, c, C1, C2, y0, v0 = ivp_parts(rng)
    sol = ivp_sol(r1, r2, a, b, K, c, C1, C2, y0, v0)
    return ExamItem("2階線形微分方程式の初期値問題",
                    rf"微分方程式 ${eq2(a, b)}={c}$ について, 次の問いに答えよ.",
                    [Part(r"一般解を求めよ.", rf"y=C_1{ex(r1)}+C_2{ex(r2)}{'+' if K > 0 else ''}{K}", sol[:1]),
                     Part(rf"初期条件 $y(0)={y0}$, $y^{{\prime}}(0)={v0}$ をみたす解を求めよ.", rf"y={ssum([(C1, ex(r1)), (C2, ex(r2)), (K, '')])}", sol[1:])],
                    meta=dict(r=(r1, r2), a=a, b=b, c=c, y0=y0, v0=v0, C=(C1, C2, K)))


def guided_varp(rng):
    r1, r2 = sorted(rng.sample([-2, -1, 1, 2, 3], 2))
    a, b = -(r1 + r2), r1 * r2
    k = rng.choice([kk for kk in (-3, -2, -1, 1, 2, 3, 4) if kk not in (r1, r2)])
    c = (k - r1) * (k - r2)          # y_p = e^{kt}
    W = r2 - r1
    return ExamItem("定数変化法",
                    rf"微分方程式 ${eq2(a, b)}={ssum([(c, ex(k))])}$ について, 次の問いに答えよ.",
                    [Part(r"斉次方程式の基本解 $y_1,y_2$ と, ロンスキアン $W=y_1y_2^{\prime}-y_1^{\prime}y_2$ を求めよ.",
                          rf"y_1={ex(r1)},\ y_2={ex(r2)},\ W={ssum([(W, ex(r1 + r2))])}",
                          [rf"特性方程式の解は $\lambda={r1},\ {r2}$。", rf"$W={ex(r1)}\cdot\left({ssum([(r2, ex(r2))])}\right)-\left({ssum([(r1, ex(r1))])}\right)\cdot {ex(r2)}={ssum([(W, ex(r1 + r2))])}$。"]),
                     Part(r"定数変化法の公式 $y_p=-y_1\displaystyle\int\frac{y_2f}{W}dt+y_2\int\frac{y_1f}{W}dt$ を用いて, 特殊解 $y_p$ を求めよ(積分定数は $0$ とする).",
                          rf"y_p={ex(k)}",
                          [rf"$\dfrac{{y_2f}}{{W}}={ssum([(F(c, W), ex(k - r1))])}$, $\dfrac{{y_1f}}{{W}}={ssum([(F(c, W), ex(k - r2))])}$。",
                           rf"$y_p=-{ex(r1)}\cdot {ssum([(F(c, W * (k - r1)), ex(k - r1))])}+{ex(r2)}\cdot\left({ssum([(F(c, W * (k - r2)), ex(k - r2))])}\right)"
                           rf"={ssum([(F(c, W) * (F(1, k - r2) - F(1, k - r1)), ex(k))])}$。"])],
                    meta=dict(a=a, b=b, k=k, c=c, r=(r1, r2)))


REGISTRY = [
    ProblemType("od_first", "1階線形微分方程式", "基本", g_first),
    ProblemType("od_char", "斉次方程式の一般解", "基本", g_char),
    ProblemType("od_wron", "ロンスキアン", "基本", g_wron, guided=guided_varp),
    ProblemType("od_undet", "未定係数法による特殊解", "標準", g_undet),
    ProblemType("od_ivp", "初期値問題", "標準", g_ivp, guided=guided_ivp),
]
GUIDED_NAMES = {"od_wron": "基本解とロンスキアン → 定数変化法で特殊解",
                "od_ivp": "一般解 → 初期条件をみたす解"}


def tab():
    render("lode", REGISTRY, "ラプラス変換 第2章 0. 線形微分方程式の解(準備)")
