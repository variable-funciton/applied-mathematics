"""数学IV ベクトル解析 第1章 ベクトル関数:演習の問題生成

誤答は、学生がよくする間違いから作る(why に、その誤答を選んだときの指摘を書く)。
"""
from fractions import Fraction
from math import gcd

import numpy as np

from topics.ex_core import (Choice, Problem, ProblemType, Retry, lin_tex, poly_tex, render, sqrt_parts,
                            tex_fr, tex_frac_root, tex_root, vtex)


def ivec(rng, lo=-3, hi=3, nonzero=True):
    while True:
        v = [rng.randint(lo, hi) for _ in range(3)]
        if not nonzero or any(v):
            return v


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def cross_bad(a, b):
    """よくある間違い:第2成分を a1b3 − a3b1 としてしまう(符号の付け忘れ)"""
    c = cross(a, b)
    return [c[0], -c[1], c[2]]


def det3(a, b, c):
    return dot(cross(a, b), c)


def n2(v):
    return dot(v, v)


def sub(a, b):
    return [x - y for x, y in zip(a, b)]


def ptex(name, p):
    return rf"\mathrm{{{name}}}" + vtex(p).replace(r"\left(", "(").replace(r"\right)", ")")


def cross_steps(a, b):
    c = cross(a, b)
    return (rf"$$\vec a\times\vec b=\begin{{vmatrix}}\vec i&\vec j&\vec k\\{a[0]}&{a[1]}&{a[2]}\\{b[0]}&{b[1]}&{b[2]}"
            rf"\end{{vmatrix}}$$" + "\n\n"
            rf"$$=\left(({a[1]})({b[2]})-({a[2]})({b[1]}),\ ({a[2]})({b[0]})-({a[0]})({b[2]}),\ "
            rf"({a[0]})({b[1]})-({a[1]})({b[0]})\right)={vtex(c)}$$")


# ============================================================
# 基本1:なす角
# ============================================================
_ANGLE_PAIRS = None
COS2 = {Fraction(0): [90], Fraction(1, 4): [60, 120], Fraction(1, 2): [45, 135], Fraction(3, 4): [30, 150]}


def _angle_pairs():
    global _ANGLE_PAIRS
    if _ANGLE_PAIRS is None:
        vs = [[x, y, z] for x in range(-2, 3) for y in range(-2, 3) for z in range(-2, 3) if (x, y, z) != (0, 0, 0)]
        out = []
        for i, a in enumerate(vs):
            for b in vs[i + 1:]:
                d = dot(a, b)
                c2 = Fraction(d * d, n2(a) * n2(b))
                if c2 in COS2 and c2 != 1 and sum(x != 0 for x in a + b) >= 4:
                    ang = int(round(np.degrees(np.arccos(d / np.sqrt(n2(a) * n2(b))))))
                    out.append((a, b, ang))
        _ANGLE_PAIRS = out
    return _ANGLE_PAIRS


def g_angle(rng):
    a, b, ang = rng.choice(_angle_pairs())
    if rng.random() < 0.5:
        a, b = b, a
    d = dot(a, b)
    choices = [Choice(rf"{ang}^\circ", True, value=ang)]
    sup = 180 - ang
    if sup != ang:
        choices.append(Choice(rf"{sup}^\circ", why="内積の符号を確かめよう。内積が負なら $\\theta$ は鈍角($90^\\circ$ より大きい)。"))
    comp = 90 - ang if ang < 90 else ang - 90
    if comp not in (ang, sup) and comp > 0:
        choices.append(Choice(rf"{comp}^\circ", why="$\\cos\\theta$ の値と $\\sin\\theta$ の値を取り違えていないか確かめよう。"))
    pool = [30, 45, 60, 90, 120, 135, 150]
    rng.shuffle(pool)
    for x in pool:
        if len(choices) == 4:
            break
        if all(c.tex != rf"{x}^\circ" for c in choices):
            choices.append(Choice(rf"{x}^\circ", why="$|\\vec a|,\\ |\\vec b|$ と内積の計算を確かめよう。"))
    na, nb = n2(a), n2(b)
    return Problem(
        "2つのベクトルのなす角", "基本",
        rf"$\vec a={vtex(a)},\ \vec b={vtex(b)}$ のなす角 $\theta\ (0^\circ\le\theta\le180^\circ)$ を求めよ。",
        choices,
        hints=[r"$\cos\theta=\dfrac{\vec a\cdot\vec b}{|\vec a|\,|\vec b|}$ を使う。",
               r"$\vec a\cdot\vec b=a_1b_1+a_2b_2+a_3b_3$、$|\vec a|=\sqrt{a_1^2+a_2^2+a_3^2}$。"],
        solution=[rf"$\vec a\cdot\vec b=({a[0]})({b[0]})+({a[1]})({b[1]})+({a[2]})({b[2]})={d}$",
                  rf"$|\vec a|={tex_root(na)},\quad |\vec b|={tex_root(nb)}$",
                  rf"$$\cos\theta=\frac{{{d}}}{{{tex_root(na)}\cdot {tex_root(nb)}}}={tex_frac_root(d, na * nb, na * nb)}$$",
                  rf"よって $\theta={ang}^\circ$"],
        meta=dict(a=a, b=b))


# ============================================================
# 基本2:外積
# ============================================================
def g_cross(rng):
    a, b = ivec(rng), ivec(rng)
    c = cross(a, b)
    if not any(c) or c[1] == 0 or sum(x != 0 for x in a + b) < 5:
        raise Retry
    prod = [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
    choices = [Choice(vtex(c), True, value=c),
               Choice(vtex(cross_bad(a, b)), why="第2成分は $a_3b_1-a_1b_3$。行列式を展開するとき、"
                                                  "第2成分($\\vec j$ の係数)には − がつくことに注意。"),
               Choice(vtex([-x for x in c]), why="$\\vec b\\times\\vec a$ を計算していないか。"
                                                 "順番を入れかえると符号が反対になる($\\vec b\\times\\vec a=-\\vec a\\times\\vec b$)。"),
               Choice(vtex(prod), why="外積は成分どうしのかけ算ではない。行列式の形で計算しよう。")]
    return Problem(
        "外積の計算", "基本",
        rf"$\vec a={vtex(a)},\ \vec b={vtex(b)}$ のとき、外積 $\vec a\times\vec b$ を求めよ。",
        choices,
        hints=[r"$\vec a\times\vec b=\begin{vmatrix}\vec i&\vec j&\vec k\\a_1&a_2&a_3\\b_1&b_2&b_3\end{vmatrix}"
               r"=(a_2b_3-a_3b_2,\ a_3b_1-a_1b_3,\ a_1b_2-a_2b_1)$",
               r"答えが $\vec a$ とも $\vec b$ とも垂直か(内積が $0$ か)で検算できる。"],
        solution=[cross_steps(a, b),
                  rf"検算:$(\vec a\times\vec b)\cdot\vec a={dot(c, a)},\ (\vec a\times\vec b)\cdot\vec b={dot(c, b)}$"],
        meta=dict(a=a, b=b))


# ============================================================
# 基本3:面積
# ============================================================
def g_area(rng):
    tri = rng.random() < 0.6
    if tri:
        A, B, C = ivec(rng, -2, 3, False), ivec(rng, -2, 3, False), ivec(rng, -2, 3, False)
        a, b = sub(B, A), sub(C, A)
    else:
        a, b = ivec(rng), ivec(rng)
    c = cross(a, b)
    N = n2(c)
    if N == 0 or c[1] == 0:
        raise Retry
    half = Fraction(1, 2) if tri else Fraction(1)
    tex_ok = tex_frac_root(half.numerator, half.denominator, N)
    wrong_half = tex_frac_root(1, 1, N) if tri else tex_frac_root(1, 2, N)
    choices = [Choice(tex_ok, True, value=float(half) * np.sqrt(N)),
               Choice(wrong_half, why=("三角形の面積は、平行四辺形の面積(外積の大きさ)の $\\frac12$。" if tri else
                                       "平行四辺形の面積は外積の大きさそのもの。$\\frac12$ は三角形のとき。")),
               Choice(tex_frac_root(half.numerator, half.denominator, n2(a) * n2(b)),
                      why="$|\\vec a||\\vec b|$ だけでは面積にならない。$\\sin\\theta$ をかける必要がある"
                          "(外積の大きさ $|\\vec a\\times\\vec b|=|\\vec a||\\vec b|\\sin\\theta$ を使う)。"),
               Choice(tex_fr(half * sum(abs(x) for x in c)),
                      why="外積の大きさは $\\sqrt{c_1^2+c_2^2+c_3^2}$。成分の絶対値を足すだけでは大きさにならない。")]
    if tri:
        st = (rf"3点 ${ptex('A', A)},\ {ptex('B', B)},\ {ptex('C', C)}$ を頂点とする三角形 $\mathrm{{ABC}}$ の面積を求めよ。")
        sol = [rf"$\vec a=\overrightarrow{{AB}}={vtex(a)},\ \vec b=\overrightarrow{{AC}}={vtex(b)}$",
               cross_steps(a, b),
               rf"$$S=\frac12|\vec a\times\vec b|=\frac12\sqrt{{{N}}}={tex_ok}$$"]
        hints = [r"$\overrightarrow{AB}$ と $\overrightarrow{AC}$ を求め、外積を計算する。",
                 r"三角形の面積は $S=\dfrac12\left|\overrightarrow{AB}\times\overrightarrow{AC}\right|$。"]
    else:
        st = rf"$\vec a={vtex(a)},\ \vec b={vtex(b)}$ を2辺とする平行四辺形の面積を求めよ。"
        sol = [cross_steps(a, b), rf"$$S=|\vec a\times\vec b|=\sqrt{{{N}}}={tex_ok}$$"]
        hints = [r"平行四辺形の面積は外積の大きさ $S=|\vec a\times\vec b|$。"]
    return Problem("平行四辺形・三角形の面積", "基本", st, choices, hints=hints, solution=sol,
                   meta=dict(a=a, b=b, half=float(half)))


# ============================================================
# 基本4:体積(スカラー三重積)
# ============================================================
def g_volume(rng):
    tet = rng.random() < 0.5
    if tet:
        P = [ivec(rng, -2, 3, False) for _ in range(4)]
        a, b, c = sub(P[1], P[0]), sub(P[2], P[0]), sub(P[3], P[0])
    else:
        a, b, c = ivec(rng, -2, 3), ivec(rng, -2, 3), ivec(rng, -2, 3)
    D = det3(a, b, c)
    Db = dot(cross_bad(a, b), c)
    if D == 0 or abs(Db) == abs(D):
        raise Retry
    k = Fraction(1, 6) if tet else Fraction(1)
    V = k * abs(D)
    choices = [Choice(tex_fr(V), True, value=float(V)),
               Choice(tex_fr((Fraction(1, 3) if tet else Fraction(1, 6)) * abs(D)),
                      why=("四面体の体積は平行六面体の $\\frac16$(三角形の面積 $\\frac12$ × 錐の $\\frac13$)。" if tet else
                           "$\\frac16$ がつくのは四面体のとき。平行六面体はスカラー三重積の絶対値そのもの。")),
               Choice(tex_fr(k * abs(Db)), why="外積の第2成分の符号を確かめよう($a_3b_1-a_1b_3$)。"),
               Choice(tex_fr(k * abs(D) * 2) if not tet else tex_fr(abs(D)),
                      why=("四面体の体積は平行六面体の $\\frac16$。" if tet else "スカラー三重積の計算を確かめよう。"))]
    mat = (rf"\begin{{vmatrix}}{a[0]}&{a[1]}&{a[2]}\\{b[0]}&{b[1]}&{b[2]}\\{c[0]}&{c[1]}&{c[2]}\end{{vmatrix}}")
    if tet:
        st = ("4点 $" + r",\ ".join(ptex(n, p) for n, p in zip("ABCD", P)) + "$ を頂点とする四面体の体積を求めよ。")
        sol = [rf"$\vec a=\overrightarrow{{AB}}={vtex(a)},\ \vec b=\overrightarrow{{AC}}={vtex(b)},\ "
               rf"\vec c=\overrightarrow{{AD}}={vtex(c)}$",
               rf"$$(\vec a\times\vec b)\cdot\vec c={mat}={D}$$",
               rf"$$V=\frac16\left|(\vec a\times\vec b)\cdot\vec c\right|=\frac16\cdot{abs(D)}={tex_fr(V)}$$"]
        hints = [r"$\overrightarrow{AB},\ \overrightarrow{AC},\ \overrightarrow{AD}$ を求める。",
                 r"四面体の体積は $V=\dfrac16\left|(\vec a\times\vec b)\cdot\vec c\right|$。"]
    else:
        st = (rf"$\vec a={vtex(a)},\ \vec b={vtex(b)},\ \vec c={vtex(c)}$ を3辺とする平行六面体の体積を求めよ。")
        sol = [rf"$$(\vec a\times\vec b)\cdot\vec c={mat}={D}$$", rf"$$V=|{D}|={tex_fr(V)}$$"]
        hints = [r"平行六面体の体積はスカラー三重積の絶対値 $V=\left|(\vec a\times\vec b)\cdot\vec c\right|$。",
                 r"スカラー三重積は、3つのベクトルを並べた行列式に等しい。"]
    return Problem("スカラー三重積と体積", "基本", st, choices, hints=hints, solution=sol,
                   meta=dict(a=a, b=b, c=c, k=float(k)))


# ============================================================
# 基本5:速度・加速度・速さ
# ============================================================
def pder(c):
    return [k * c[k] for k in range(1, len(c))] + [0]


def pder_bad(c):
    """よくある間違い:t^n の微分で係数 n をかけ忘れる"""
    return [c[k] for k in range(1, len(c))] + [0]


def peval(c, t):
    return sum(ck * t ** k for k, ck in enumerate(c))


def g_motion(rng):
    R = []
    for _ in range(3):
        c = [rng.randint(-3, 3) for _ in range(4)]
        if rng.random() < 0.5:
            c[3] = 0
        R.append(c)
    if sum(1 for c in R if any(c[2:])) < 2:
        raise Retry
    t0 = rng.choice([-2, -1, 1, 2])
    kind = rng.choice(["速度", "加速度", "速さ"])
    V = [pder(c) for c in R]
    A = [pder(v) for v in V]
    r0 = [peval(c, t0) for c in R]
    v0 = [peval(c, t0) for c in V]
    a0 = [peval(c, t0) for c in A]
    vb = [peval(pder_bad(c), t0) for c in R]
    rt = r"\vec r(t)=\left(" + r",\ ".join(poly_tex(c) for c in R) + r"\right)"
    sol0 = [rf"$$\vec r\,'(t)=\left(" + r",\ ".join(poly_tex(v) for v in V) + r"\right)$$"]
    if kind == "速度":
        if not any(v0):
            raise Retry
        choices = [Choice(vtex(v0), True, value=v0),
                   Choice(vtex(r0), why="それは位置 $\\vec r(t_0)$。速度は $\\vec r\\,'(t)$。"),
                   Choice(vtex(a0), why="それは加速度 $\\vec r\\,''(t_0)$。速度は1回微分。"),
                   Choice(vtex(vb), why="$t^n$ の微分は $nt^{n-1}$。係数 $n$ をかけ忘れていないか。")]
        sol = sol0 + [rf"$t={t0}$ を代入して $\vec v={vtex(v0)}$"]
    elif kind == "加速度":
        if not any(a0):
            raise Retry
        ab = [peval(pder_bad(pder(c)), t0) for c in R]
        choices = [Choice(vtex(a0), True, value=a0),
                   Choice(vtex(v0), why="それは速度 $\\vec r\\,'(t_0)$。加速度は2回微分 $\\vec r\\,''(t)$。"),
                   Choice(vtex(r0), why="それは位置 $\\vec r(t_0)$。加速度は2回微分。"),
                   Choice(vtex(ab), why="2回目の微分で係数をかけ忘れていないか($t^n\\to nt^{n-1}$)。")]
        sol = sol0 + [rf"$$\vec r\,''(t)=\left(" + r",\ ".join(poly_tex(a) for a in A) + r"\right)$$",
                      rf"$t={t0}$ を代入して $\vec a={vtex(a0)}$"]
    else:
        S = n2(v0)
        if S == 0:
            raise Retry
        choices = [Choice(tex_root(S), True, value=np.sqrt(S)),
                   Choice(str(S), why="速さは $|\\vec v|=\\sqrt{v_1^2+v_2^2+v_3^2}$。ルートを忘れていないか。"),
                   Choice(tex_root(n2(a0)) if n2(a0) else "0", why="それは加速度の大きさ。速さは速度ベクトルの大きさ。"),
                   Choice(tex_root(n2(vb)) if n2(vb) else "0",
                          why="$t^n$ の微分は $nt^{n-1}$。係数 $n$ をかけ忘れていないか。")]
        sol = sol0 + [rf"$t={t0}$ で $\vec v={vtex(v0)}$", rf"$$|\vec v|=\sqrt{{{S}}}={tex_root(S)}$$"]
    return Problem(
        f"{kind}の計算", "基本",
        rf"位置ベクトルが $\displaystyle {rt}$ で表される点の、時刻 $t={t0}$ における{kind}を求めよ。",
        choices,
        hints=[r"速度 $\vec v=\vec r\,'(t)$、加速度 $\vec a=\vec r\,''(t)$、速さ $|\vec v|$。微分は成分ごとに行う。",
               r"$t^n$ の微分は $nt^{n-1}$。最後に $t$ の値を代入する。"],
        solution=sol, meta=dict(R=R, t0=t0, kind=kind))


# ============================================================
# 標準1:両方に垂直な単位ベクトル
# ============================================================
def unit_tex(c):
    g = 0
    for x in c:
        g = gcd(g, abs(x))
    cp = [x // g for x in c]
    N = n2(cp)
    if N == 1 or (N > 1 and sqrt_parts(N)[1] == 1):
        r = int(round(N ** 0.5))
        return r"\pm" + vtex([Fraction(x, r) for x in cp])
    k, m = sqrt_parts(N)
    pre = rf"\frac{{1}}{{{tex_root(N)}}}"
    return r"\pm" + pre + vtex(cp)


def g_unit_normal(rng):
    a, b = ivec(rng), ivec(rng)
    c = cross(a, b)
    if not any(c) or c[1] == 0 or sum(x != 0 for x in c) < 2:
        raise Retry
    N = n2(c)
    g = 0
    for x in c:
        g = gcd(g, abs(x))
    choices = [Choice(unit_tex(c), True, value=np.array(c) / np.sqrt(N)),
               Choice(r"\pm" + vtex(c), why="単位ベクトルにするには、大きさ $|\\vec a\\times\\vec b|$ で割る。"),
               Choice(unit_tex(cross_bad(a, b)), why="外積の第2成分の符号を確かめよう($a_3b_1-a_1b_3$)。"),
               Choice(r"\pm\frac{1}{" + str(N // (g * g)) + "}" + vtex([x // g for x in c]),
                      why="大きさは $\\sqrt{c_1^2+c_2^2+c_3^2}$。ルートを忘れていないか。")]
    return Problem(
        "両方に垂直な単位ベクトル", "標準",
        rf"$\vec a={vtex(a)},\ \vec b={vtex(b)}$ の両方に垂直な単位ベクトルを求めよ。",
        choices,
        hints=[r"$\vec a\times\vec b$ は $\vec a$ にも $\vec b$ にも垂直。",
               r"単位ベクトルは $\pm\dfrac{\vec a\times\vec b}{|\vec a\times\vec b|}$(向きは2通り)。"],
        solution=[cross_steps(a, b),
                  rf"$|\vec a\times\vec b|=\sqrt{{{N}}}" + (f"={tex_root(N)}" if tex_root(N) != rf"\sqrt{{{N}}}" else "") + "$",
                  rf"$$\pm\frac{{\vec a\times\vec b}}{{|\vec a\times\vec b|}}={unit_tex(c)}$$"],
        meta=dict(a=a, b=b))


# ============================================================
# 標準2:4点が同一平面上にある条件
# ============================================================
def g_coplanar(rng):
    A = ivec(rng, -2, 2, False)
    u, v = ivec(rng, -2, 2), ivec(rng, -2, 2)
    n = cross(u, v)
    if not any(n):
        raise Retry
    B, C = [x + y for x, y in zip(A, u)], [x + y for x, y in zip(A, v)]
    s, t = rng.choice([-2, -1, 1, 2]), rng.choice([-2, -1, 1, 2])
    D = [A[i] + s * u[i] + t * v[i] for i in range(3)]
    j = rng.choice([i for i in range(3) if n[i] != 0])  # 未知数にする成分(その係数が 0 でない)
    k_true = D[j]
    names = "xyz"
    # det(AB, AC, AD) = n · (D − A) = n_j k + (残り)
    rest = sum(n[i] * (D[i] - A[i]) for i in range(3) if i != j) - n[j] * A[j]
    nb = cross_bad(u, v)
    rest_b = sum(nb[i] * (D[i] - A[i]) for i in range(3) if i != j) - nb[j] * A[j]
    cands = [Choice(rf"k={k_true}", True, value=k_true),
             Choice(rf"k={-k_true}" if k_true != 0 else "k=1", why="移項したときの符号を確かめよう。")]
    if nb[j] != 0:
        cands.append(Choice(rf"k={tex_fr(Fraction(-rest_b, nb[j]))}",
                            why="外積の第2成分の符号を確かめよう($a_3b_1-a_1b_3$)。"))
    cands.append(Choice(rf"k={tex_fr(Fraction(rest, n[j]))}" if rest != 0 else rf"k={k_true + 2}",
                        why="行列式 $=0$ の方程式を解くとき、$k$ の係数で割る前の符号を確かめよう。"))
    for d in (1, -1, 2, -2, 3):
        cands.append(Choice(rf"k={k_true + d}", why="行列式の計算を確かめよう。"))
    choices, seen = [], set()
    for c in cands:
        if c.tex not in seen:
            choices.append(c)
            seen.add(c.tex)
        if len(choices) == 4:
            break
    Dk = list(D)
    Dtex = "(" + ",\\ ".join("k" if i == j else str(Dk[i]) for i in range(3)) + ")"
    ADk = ["k" + (f"{-A[i]:+d}" if A[i] else "") if i == j else str(D[i] - A[i]) for i in range(3)]
    AD_tex = r",\ ".join(ADk)
    return Problem(
        "4点が同一平面上にある条件", "標準",
        rf"4点 ${ptex('A', A)},\ {ptex('B', B)},\ {ptex('C', C)},\ \mathrm{{D}}{Dtex}$ が同一平面上にあるような"
        rf"定数 $k$ の値を求めよ。",
        choices,
        hints=[r"4点が同一平面上 ⇔ $\overrightarrow{AB},\ \overrightarrow{AC},\ \overrightarrow{AD}$ が作る平行六面体の体積が $0$。",
               r"$(\overrightarrow{AB}\times\overrightarrow{AC})\cdot\overrightarrow{AD}=0$ を $k$ の方程式として解く。"],
        solution=[rf"$\overrightarrow{{AB}}={vtex(u)},\ \overrightarrow{{AC}}={vtex(v)},\ "
                  rf"\overrightarrow{{AD}}=\left({AD_tex}\right)$",
                  rf"$\overrightarrow{{AB}}\times\overrightarrow{{AC}}={vtex(n)}$",
                  rf"$$(\overrightarrow{{AB}}\times\overrightarrow{{AC}})\cdot\overrightarrow{{AD}}"
                  rf"={lin_tex([(n[j], 'k')], rest)}=0$$",
                  rf"よって $k={k_true}$"],
        meta=dict(A=A, B=B, C=C, D=D, j=j))


# ============================================================
# 標準3:曲線の長さ
# ============================================================
TRIPLES = [(3, 4, 5), (4, 3, 5), (5, 12, 13), (12, 5, 13), (6, 8, 10), (8, 6, 10)]


def g_arclength(rng):
    fam = rng.choice(["らせん", "3次", "対数"])
    if fam == "らせん":
        p, q, r = rng.choice(TRIPLES)
        T = rng.choice([1, 2])  # 区間 [0, Tπ]
        Tt = r"\pi" if T == 1 else r"2\pi"
        curve = rf"\vec r(t)=({p}\cos t,\ {p}\sin t,\ {q}t)\quad(0\le t\le {Tt})"
        L = r * T
        choices = [Choice(rf"{L}\pi", True, value=L * np.pi),
                   Choice(rf"{(p + q) * T}\pi", why="$|\\vec r\\,'|=\\sqrt{x'^2+y'^2+z'^2}$。成分を足すだけでは大きさにならない。"),
                   Choice(rf"{p * T}\pi", why="$z$ 成分の微分 $z'$ を入れ忘れていないか(それは円周の長さ)。"),
                   Choice(str(r), why="$|\\vec r\\,'|$ は一定値。それを区間の長さだけ積分する(かける)。")]
        sol = [rf"$\vec r\,'(t)=(-{p}\sin t,\ {p}\cos t,\ {q})$",
               rf"$|\vec r\,'(t)|=\sqrt{{{p * p}\sin^2t+{p * p}\cos^2t+{q * q}}}=\sqrt{{{p * p + q * q}}}={r}$",
               rf"$$L=\int_0^{{{Tt}}}{r}\,dt={L}\pi$$"]
        meta = dict(fam=fam, p=p, q=q, T=T * np.pi)
    elif fam == "3次":
        m = rng.choice([1, 2])
        T = rng.choice([1, 2, 3])
        curve = rf"\vec r(t)=({3 * m}t,\ {3 * m}t^2,\ {2 * m}t^3)\quad(0\le t\le {T})".replace("(1t", "(t")
        L = 3 * m * T + 2 * m * T ** 3
        wrong1 = 3 * m * T + 3 * m * T ** 2 + 2 * m * T ** 3
        wrong2 = 3 * m * (1 + 2 * T * T)
        chord2 = (3 * m * T) ** 2 + (3 * m * T * T) ** 2 + (2 * m * T ** 3) ** 2
        choices = [Choice(str(L), True, value=L),
                   Choice(str(wrong1), why="$|\\vec r\\,'|$ は成分の和ではない。2乗の和の平方根をとる。"),
                   Choice(str(wrong2), why="それは $t=" + str(T) + "$ での速さ $|\\vec r\\,'|$。長さは速さを積分する。"),
                   Choice(tex_root(chord2), why="それは始点と終点を結ぶ線分(弦)の長さ。曲線に沿った長さは積分で求める。")]
        sol = [rf"$\vec r\,'(t)=({3 * m},\ {6 * m}t,\ {6 * m}t^2)$",
               rf"$$|\vec r\,'(t)|=\sqrt{{{9 * m * m}+{36 * m * m}t^2+{36 * m * m}t^4}}={3 * m}(1+2t^2)$$",
               rf"$$L=\int_0^{{{T}}}{3 * m}(1+2t^2)\,dt={3 * m}\left[t+\frac{{2}}{{3}}t^3\right]_0^{{{T}}}={L}$$"]
        meta = dict(fam=fam, m=m, T=T)
    else:
        T = rng.choice([2, 3, "e"])
        Tt = "e" if T == "e" else str(T)
        curve = rf"\vec r(t)=(2t,\ t^2,\ \log t)\quad(1\le t\le {Tt})"
        if T == "e":
            ok, w1, w2, w3 = "e^2", "e^2-1", "e^2-2", "e^2+1"
            val = np.e ** 2
        else:
            ok, w1, w2, w3 = (rf"{T * T - 1}+\log {T}", str(T * T - 1), rf"{T * T - 1}-\log {T}",
                              rf"{T * T}+\log {T}")
            val = T * T - 1 + np.log(T)
        choices = [Choice(ok, True, value=val),
                   Choice(w1, why="$\\frac1t$ の積分 $\\log t$ を入れ忘れていないか。"),
                   Choice(w2, why="$|\\vec r\\,'|=2t+\\frac1t$。符号を確かめよう。"),
                   Choice(w3, why="下端 $t=1$ を代入した値を引き忘れていないか。")]
        sol = [r"$\vec r\,'(t)=\left(2,\ 2t,\ \dfrac1t\right)$",
               r"$$|\vec r\,'(t)|=\sqrt{4+4t^2+\frac{1}{t^2}}=\sqrt{\left(2t+\frac1t\right)^2}=2t+\frac1t$$",
               rf"$$L=\int_1^{{{Tt}}}\left(2t+\frac1t\right)dt=\Big[t^2+\log t\Big]_1^{{{Tt}}}={ok}$$"]
        meta = dict(fam=fam, T=np.e if T == "e" else T)
    return Problem(
        "曲線の長さ", "標準", rf"曲線 $\displaystyle {curve}$ の長さ $L$ を求めよ。", choices,
        hints=[r"曲線の長さ $L=\displaystyle\int_a^b|\vec r\,'(t)|\,dt$。",
               r"$|\vec r\,'(t)|^2$ を計算すると、きれいな式の2乗になる(または一定になる)。"],
        solution=sol, meta=meta)


# ============================================================
# 標準4:接平面の方程式
# ============================================================
def _sh(c):
    """x − c の「− c」部分(c = 0 なら空)"""
    return "" if c == 0 else (f"-{c}" if c > 0 else f"+{-c}")


def g_tangent_plane(rng):
    a, b, c, d, e, g = (rng.randint(-2, 2) for _ in range(6))
    if a == 0 and c == 0 or (a, b, c).count(0) > 1:
        raise Retry
    x0, y0 = rng.choice([-1, 1, 2]), rng.choice([-2, -1, 1])
    f = lambda x, y: a * x * x + b * x * y + c * y * y + d * x + e * y + g  # noqa: E731
    z0 = f(x0, y0)
    fx = 2 * a * x0 + b * y0 + d
    fy = b * x0 + 2 * c * y0 + e
    if fx == 0 or fy == 0 or fx == fy:
        raise Retry
    k = z0 - fx * x0 - fy * y0
    ftex = lin_tex([(a, "x^2"), (b, "xy"), (c, "y^2"), (d, "x"), (e, "y")], g)

    def pl(p, q, r):
        return "z=" + lin_tex([(p, "x"), (q, "y")], r)
    choices = [Choice(pl(fx, fy, k), True, value=(fx, fy, k)),
               Choice(pl(fy, fx, z0 - fy * x0 - fx * y0), why="$f_x$ と $f_y$ を取り違えていないか。"),
               Choice(pl(fx, fy, z0 + fx * x0 + fy * y0),
                      why="$z-z_0=f_x(x-x_0)+f_y(y-y_0)$ を展開するときの符号を確かめよう。"),
               Choice(pl(fx, fy, -fx * x0 - fy * y0), why="$z_0=f(x_0,y_0)$ を足し忘れていないか。")]
    return Problem(
        "曲面の接平面", "標準",
        rf"曲面 $z={ftex}$ の、点 $({x0},\ {y0},\ {z0})$ における接平面の方程式を求めよ。",
        choices,
        hints=[r"$z=f(x,y)$ の接平面は $z-z_0=f_x(x_0,y_0)(x-x_0)+f_y(x_0,y_0)(y-y_0)$。",
               r"法線ベクトル $(f_x,\ f_y,\ -1)$ を使って $\vec n\cdot(\vec x-\vec x_0)=0$ と考えてもよい。"],
        solution=[rf"$f_x={lin_tex([(2 * a, 'x'), (b, 'y')], d)},\quad f_y={lin_tex([(b, 'x'), (2 * c, 'y')], e)}$",
                  rf"点 $({x0},{y0})$ で $f_x={fx},\ f_y={fy}$",
                  rf"$$z{_sh(z0)}={'' if fx == 1 else ('-' if fx == -1 else fx)}(x{_sh(x0)}){'+' if fy > 0 else '-'}{abs(fy) if abs(fy) != 1 else ''}(y{_sh(y0)})$$",
                  rf"$$\therefore\ {pl(fx, fy, k)}$$"],
        meta=dict(coef=(a, b, c, d, e, g), x0=x0, y0=y0))

# ============================================================
# 基本6:曲線を表すベクトル関数(円・線分・直線)
# ============================================================
INF = float("inf")


def cterm(a, terms):
    """定数 a を先頭に書いた 1 次式:a + c·v(a = 0 なら c·v だけ)"""
    body = lin_tex(terms, 0) if any(c for c, _ in terms) else ""
    if a == 0:
        return body or "0"
    if not body:
        return tex_fr(a)
    return tex_fr(a) + (body if body.startswith("-") else "+" + body)


def rvec(x, y):
    return rf"\left({x},\ {y}\right)"


def sq_tex(var, a):
    if a == 0:
        return f"{var}^2"
    return rf"({var}{'-' if a > 0 else '+'}{abs(a)})^2"


RANGE = {"seg": r"\ (0\le t\le 1)", "line": r"\ (-\infty<t<\infty)"}


def g_curve_param(rng):
    from topics.ex_fig import arc_pts, seg_pts
    kind = rng.choice(["circle", "circle", "segment", "line"])
    if kind == "circle":
        a, b = rng.randint(-3, 3), rng.randint(-3, 3)
        r = rng.choice([2, 3])
        if (a, b) == (0, 0):
            raise Retry
        R = r * r
        S = (a + r, b)

        def circ(cx, cy, rr, sw=False, cw=False):
            if sw:
                return lambda t: (cx + rr * np.sin(t), cy + rr * np.cos(t))
            return lambda t: (cx + rr * np.cos(t), cy + (-rr if cw else rr) * np.sin(t))
        C, Sn = r"\cos t", r"\sin t"
        ok = rvec(cterm(a, [(r, C)]), cterm(b, [(r, Sn)]))
        cands = [Choice(rvec(cterm(a, [(r, Sn)]), cterm(b, [(r, C)])),
                        why="$t=0$ を代入すると始点が、$t$ を少し増やすと向きがわかる。$\\cos$ と $\\sin$ を入れかえると始点も向きも変わる。",
                        value=(circ(a, b, r, sw=True), 0, 2 * np.pi)),
                 Choice(rvec(cterm(a, [(r, C)]), cterm(b, [(-r, Sn)])),
                        why="それは時計回り。$t$ が増えると $y$ が減っていないか確かめよう。",
                        value=(circ(a, b, r, cw=True), 0, 2 * np.pi)),
                 Choice(rvec(cterm(-a, [(r, C)]), cterm(-b, [(r, Sn)])),
                        why=f"円 $(x-a)^2+(y-b)^2=r^2$ の中心は $(a,\\ b)$。符号を確かめよう。",
                        value=(circ(-a, -b, r), 0, 2 * np.pi)),
                 Choice(rvec(cterm(a, [(R, C)]), cterm(b, [(R, Sn)])),
                        why=f"右辺の ${R}$ は半径の2乗。半径は $\\sqrt{{{R}}}={r}$。",
                        value=(circ(a, b, R), 0, 2 * np.pi))]
        choices = [Choice(ok, True, value=(circ(a, b, r), 0, 2 * np.pi))] + rng.sample(cands, 3)
        eq = rf"{sq_tex('x', a)}+{sq_tex('y', b)}={R}"
        stmt = (rf"円 ${eq}$ を, 点 $({S[0]},\ {S[1]})$ を始点として反時計回りに1周する曲線を表すベクトル関数 "
                r"$\vec r(t)\ (0\le t\le 2\pi)$ として正しいものを選べ。")
        ex_stmt = (rf"円 ${eq}$ を, 点 $({S[0]},\ {S[1]})$ を始点として反時計回りに1周する曲線を表すベクトル関数 "
                   r"$\vec r(t)\ (0\le t\le 2\pi)$ を1つ求めよ.")
        sol = [rf"中心は $({a},\ {b})$, 半径は $\sqrt{{{R}}}={r}$。",
               r"中心 $(a,b)$, 半径 $r$ の円を反時計回りに回るには $\vec r(t)=(a+r\cos t,\ b+r\sin t)$ とすればよい"
               r"($t=0$ で点 $(a+r,\ b)$ から出発し, $t$ が増えると反時計回りに進む)。",
               rf"$$\vec r(t)={ok}\quad(0\le t\le 2\pi)$$",
               r"(表し方は1通りではない。始点・向き・1周することが合っていればよい。)"]
        fig = {"kind": "path2d", "pieces": [{"pts": arc_pts(a, b, r, 0, 2 * np.pi), "label": "C"}],
               "points": [(S[0], S[1], "S")]}
        hints = [r"中心 $(a,b)$, 半径 $r$ の円は $x=a+r\cos t,\ y=b+r\sin t$ と表せる。",
                 r"$t=0$ を代入して始点を、$t=\frac{\pi}{2}$ を代入して進む向きを確かめよう。"]
        meta = dict(kind=kind, center=(a, b), r=r, start=S)
    else:
        A = (rng.randint(-3, 3), rng.randint(-3, 3))
        d = (rng.choice([-3, -2, -1, 1, 2, 3, 4]), rng.choice([-3, -2, -1, 0, 1, 2, 3]))
        if A == (0, 0):
            raise Retry

        def lin(P, D):
            return lambda t: (P[0] + D[0] * t, P[1] + D[1] * t)

        def lt(P, D, rg):
            return rvec(cterm(P[0], [(D[0], "t")]), cterm(P[1], [(D[1], "t")])) + RANGE[rg]
        if kind == "segment":
            B = (A[0] + d[0], A[1] + d[1])
            ok = lt(A, d, "seg")
            choices = [Choice(ok, True, value=(lin(A, d), 0, 1)),
                       Choice(lt(A, B, "seg"), why="$\\vec{\\mathrm{OA}}+t\\,\\vec{\\mathrm{OB}}$ では $t=1$ で点 $\\mathrm B$ に着かない。"
                              "向きは $\\vec{\\mathrm{AB}}=\\vec{\\mathrm{OB}}-\\vec{\\mathrm{OA}}$。", value=(lin(A, B), 0, 1)),
                       Choice(lt(B, (-d[0], -d[1]), "seg"), why="それは $\\mathrm B$ から $\\mathrm A$ へ向かう線分(向きが逆)。",
                              value=(lin(B, (-d[0], -d[1])), 0, 1)),
                       Choice(lt(A, d, "line"), why="$t$ をすべての実数にすると、線分ではなく直線全体になる。",
                              value=(lin(A, d), -INF, INF))]
            Bt = rf"\mathrm{{B}}({B[0]},\ {B[1]})"
            stmt = (rf"点 $\mathrm{{A}}({A[0]},\ {A[1]})$ から点 ${Bt}$ へ向かう線分を表すベクトル関数 $\vec r(t)$ "
                    r"として正しいものを選べ。")
            ex_stmt = (rf"点 $\mathrm{{A}}({A[0]},\ {A[1]})$ から点 ${Bt}$ へ向かう線分を表すベクトル関数 $\vec r(t)$ を, "
                       r"$t$ の範囲も含めて1つ求めよ.")
            sol = [r"$\vec r(t)=(1-t)\vec{\mathrm{OA}}+t\,\vec{\mathrm{OB}}=\vec{\mathrm{OA}}+t\,\vec{\mathrm{AB}}\ (0\le t\le 1)$ とすると, "
                   r"$t=0$ で $\mathrm A$, $t=1$ で $\mathrm B$ になる。",
                   rf"$\vec{{\mathrm{{AB}}}}=({d[0]},\ {d[1]})$ より", rf"$$\vec r(t)={ok}$$",
                   r"(表し方は1通りではない。始点・終点・向きが合っていればよい。)"]
            fig = {"kind": "path2d", "pieces": [{"pts": seg_pts(A, B), "label": "C"}],
                   "points": [(A[0], A[1], "A"), (B[0], B[1], "B")]}
            hints = [r"線分 $\mathrm{AB}$ 上の点は $\vec{\mathrm{OA}}+t\,\vec{\mathrm{AB}}$ と表せる。$t$ の範囲は?",
                     r"$t=0$ と $t=1$ を代入して、始点と終点を確かめよう。"]
            meta = dict(kind=kind, A=A, B=B)
        else:
            nrm = (d[1], -d[0])
            if d[0] * A[1] - d[1] * A[0] == 0:
                raise Retry                       # 直線が原点を通ると (d + tA) が同じ直線になる
            ok = lt(A, d, "line")
            choices = [Choice(ok, True, value=(lin(A, d), -INF, INF)),
                       Choice(lt(d, A, "line"), why="通る点と方向ベクトルの役割が入れかわっている。$\\vec r(t)=\\vec{\\mathrm{OA}}+t\\vec d$。",
                              value=(lin(d, A), -INF, INF)),
                       Choice(lt(A, nrm, "line"), why=f"$({nrm[0]},\\ {nrm[1]})$ は方向ベクトルに垂直(法線ベクトル)。",
                              value=(lin(A, nrm), -INF, INF)),
                       Choice(lt(A, d, "seg"), why="$0\\le t\\le 1$ では線分になる。直線は $t$ をすべての実数にわたらせる。",
                              value=(lin(A, d), 0, 1))]
            stmt = (rf"点 $\mathrm{{A}}({A[0]},\ {A[1]})$ を通り, 方向ベクトル $\vec d=({d[0]},\ {d[1]})$ の直線を表すベクトル関数 "
                    r"$\vec r(t)$ として正しいものを選べ。")
            ex_stmt = (rf"点 $\mathrm{{A}}({A[0]},\ {A[1]})$ を通り, 方向ベクトル $\vec d=({d[0]},\ {d[1]})$ の直線を表すベクトル関数 "
                       r"$\vec r(t)$ を, $t$ の範囲も含めて1つ求めよ.")
            sol = [r"点 $\mathrm A$ を通り方向ベクトル $\vec d$ の直線は $\vec r(t)=\vec{\mathrm{OA}}+t\vec d\ (-\infty<t<\infty)$。",
                   rf"$$\vec r(t)={ok}$$", r"(表し方は1通りではない。)"]
            P0, P1 = (A[0] - 1.2 * d[0], A[1] - 1.2 * d[1]), (A[0] + 1.2 * d[0], A[1] + 1.2 * d[1])
            fig = {"kind": "path2d", "pieces": [{"pts": seg_pts(P0, P1), "label": ""}],
                   "points": [(A[0], A[1], "A")]}
            hints = [r"直線上の点 $\mathrm{P}$ は $\vec{\mathrm{OP}}=\vec{\mathrm{OA}}+t\vec d$ と表せる。",
                     r"直線は両側に限りなくのびるので、$t$ の範囲に注意。"]
            meta = dict(kind=kind, A=A, d=d)
    return Problem("曲線を表すベクトル関数", "基本", stmt, choices, hints=hints, solution=sol, meta=meta,
                   exam_statement=ex_stmt, figure=fig)


REGISTRY = [
    ProblemType("angle", "なす角", "基本", g_angle),
    ProblemType("cross", "外積の計算", "基本", g_cross),
    ProblemType("area", "平行四辺形・三角形の面積", "基本", g_area),
    ProblemType("volume", "スカラー三重積と体積", "基本", g_volume),
    ProblemType("motion", "速度・加速度・速さ", "基本", g_motion),
    ProblemType("param", "曲線を表すベクトル関数", "基本", g_curve_param),
    ProblemType("unit", "両方に垂直な単位ベクトル", "標準", g_unit_normal),
    ProblemType("coplanar", "4点が同一平面上にある条件", "標準", g_coplanar),
    ProblemType("arclen", "曲線の長さ", "標準", g_arclength),
    ProblemType("tangent", "曲面の接平面", "標準", g_tangent_plane),
]


def tab():
    render("ch1", REGISTRY, "第1章 ベクトル関数")
