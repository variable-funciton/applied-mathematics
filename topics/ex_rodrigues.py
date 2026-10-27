"""数学IV ベクトル解析 発展:ロドリゲスの回転公式の演習

  基本:座標軸のまわりの 90°・180°・270° 回転
  標準:一般の軸のまわりの 180° 回転 / 90° 回転(軸の長さが整数になる向き) / 座標軸のまわりの回転行列
誤答は「逆向きに回す」「平行成分を忘れる」「軸を正規化しない」「1 − cos θ を忘れる」などから作る。
"""
from fractions import Fraction

from topics.ex_core import Choice, ExamItem, Part, Problem, ProblemType, Retry, render, tex_fr, vtex

F = Fraction
AXIS_NAME = {0: "x", 1: "y", 2: "z"}
NICE = [(1, 2, 2), (2, 1, 2), (2, 2, 1), (2, 3, 6), (3, 2, 6), (6, 2, 3), (2, 6, 3), (3, 6, 2), (6, 3, 2)]
SENSE = "回転の向きは、軸の向きに右ねじが進む向き(軸の正の側から見て反時計回り)とする。"
FORMULA = r"$\vec v\,'=\vec v\cos\theta+(\vec k\times\vec v)\sin\theta+\vec k\,(\vec k\cdot\vec v)(1-\cos\theta)$($|\vec k|=1$)"


def dot(a, b):
    return sum(F(x) * y for x, y in zip(a, b))


def cross(a, b):
    return [F(a[1]) * b[2] - F(a[2]) * b[1], F(a[2]) * b[0] - F(a[0]) * b[2], F(a[0]) * b[1] - F(a[1]) * b[0]]


def add(*vs):
    return [sum(F(v[i]) for v in vs) for i in range(3)]


def scal(c, v):
    return [F(c) * x for x in v]


def ivec(rng, lo=-3, hi=3):
    while True:
        v = [rng.randint(lo, hi) for _ in range(3)]
        if any(v):
            return v


def pick(correct, cands):
    choices = [Choice(vtex(correct), True, value=tuple(float(x) for x in correct))]
    seen = [tuple(correct)]
    for v, why in cands:
        if tuple(v) in seen:
            continue
        choices.append(Choice(vtex(v), why=why, value=tuple(float(x) for x in v)))
        seen.append(tuple(v))
        if len(choices) == 4:
            return choices
    raise Retry


# ============================================================
# 基本:座標軸のまわりの回転
# ============================================================
CS = {90: (0, 1), 180: (-1, 0), 270: (0, -1)}


def g_axis(rng):
    i = rng.randrange(3)
    deg = rng.choice([90, 90, 180, 270])
    c, s = CS[deg]
    k = [0, 0, 0]
    k[i] = 1
    v = ivec(rng)
    par = scal(dot(k, v), k)
    perp = add(v, scal(-1, par))
    if not any(perp) or not any(par):
        raise Retry
    kv = cross(k, v)
    ok = add(scal(c, v), scal(s, kv), scal((1 - c) * dot(k, v), k))
    j = (i + 1) % 3
    k2 = [0, 0, 0]
    k2[j] = 1
    other = add(scal(c, v), scal(s, cross(k2, v)), scal((1 - c) * dot(k2, v), k2))
    cands = [(add(scal(c, v), scal(-s, kv), scal((1 - c) * dot(k, v), k)),
              "逆向きに回していないか。軸の正の側から見て反時計回りが正の向き($\\vec k\\times\\vec v$ の向き)。"),
             (add(scal(c, v), scal(s, kv)), "第3項 $\\vec k(\\vec k\\cdot\\vec v)(1-\\cos\\theta)$ を忘れていないか。軸方向の成分は変わらないはず。"),
             (add(scal(c, perp), scal(s, kv)), "軸に平行な成分 $\\vec v_\\parallel$ も残る(回転しても変わらない)。"),
             (other, f"回転軸は ${AXIS_NAME[i]}$ 軸。"),
             (scal(-1, ok), "符号を確かめよう。")]
    sol = [rf"$\vec k={vtex(k)},\ \theta={deg}^\circ$ より $\cos\theta={c},\ \sin\theta={s}$。",
           rf"$\vec k\cdot\vec v={tex_fr(dot(k, v))},\ \vec k\times\vec v={vtex(kv)}$",
           rf"$\vec v\cos\theta={vtex(scal(c, v))}$, $(\vec k\times\vec v)\sin\theta={vtex(scal(s, kv))}$, "
           rf"$\vec k(\vec k\cdot\vec v)(1-\cos\theta)={vtex(scal((1 - c) * dot(k, v), k))}$",
           rf"$$\vec v\,'={vtex(ok)}$$",
           rf"(確かめ:${AXIS_NAME[i]}$ 成分は変わらず、長さも $\sqrt{{{dot(v, v)}}}$ のまま。)"]
    return Problem("座標軸のまわりの回転", "基本",
                   rf"ベクトル $\vec v={vtex(v)}$ を ${AXIS_NAME[i]}$ 軸のまわりに ${deg}^\circ$ 回転させたベクトルを求めよ。" + SENSE,
                   pick(ok, cands), hints=["ロドリゲスの回転公式:" + FORMULA,
                                           rf"${AXIS_NAME[i]}$ 軸の単位ベクトルは $\vec k={vtex(k)}$。"],
                   solution=sol, meta=dict(v=v, k=k, deg=deg))


# ============================================================
# 標準:一般の軸のまわりの 180° 回転
# ============================================================
def g_half(rng):
    while True:
        a = [rng.randint(-2, 2) for _ in range(3)]
        if sum(1 for x in a if x) >= 2:
            break
    v = ivec(rng)
    n = dot(a, a)
    av = dot(a, v)
    if av == 0 or not any(cross(a, v)):
        raise Retry
    ok = add(scal(2 * av / n, a), scal(-1, v))
    cands = [(add(scal(2 * av, a), scal(-1, v)), f"$\\vec k$ は単位ベクトル:$\\vec k=\\dfrac{{\\vec a}}{{|\\vec a|}}$, $|\\vec a|^2={n}$ で割る。"),
             (add(scal(av / n, a), scal(-1, v)), "$\\theta=180^\\circ$ では $1-\\cos\\theta=2$。"),
             (add(v, scal(-2 * av / n, a)), "符号を確かめよう($\\cos180^\\circ=-1$)。それは軸に垂直な平面に関する鏡映。"),
             (scal(-1, v), "$-\\vec v$ になるのは $\\vec v$ が軸に垂直なときだけ。軸方向の成分は変わらない。")]
    an = rf"\sqrt{{{n}}}"
    sol = [rf"$\vec k=\dfrac{{1}}{{{an}}}{vtex(a)}$, $\cos180^\circ=-1,\ \sin180^\circ=0$ より "
           r"$\vec v\,'=-\vec v+2(\vec k\cdot\vec v)\vec k$。",
           rf"$(\vec k\cdot\vec v)\vec k=\dfrac{{\vec a\cdot\vec v}}{{|\vec a|^2}}\vec a=\dfrac{{{tex_fr(av)}}}{{{n}}}{vtex(a)}$",
           rf"$$\vec v\,'=-{vtex(v)}+\frac{{{tex_fr(2 * av)}}}{{{n}}}{vtex(a)}={vtex(ok)}$$"]
    return Problem("一般の軸のまわりの 180° 回転", "標準",
                   rf"ベクトル $\vec v={vtex(v)}$ を、原点を通り $\vec a={vtex(a)}$ の向きの軸のまわりに $180^\circ$ 回転させたベクトルを求めよ。",
                   pick(ok, cands), hints=["ロドリゲスの回転公式:" + FORMULA,
                                           r"$\theta=180^\circ$ では $\vec v\,'=-\vec v+2(\vec k\cdot\vec v)\vec k$。"
                                           r"$\vec k=\dfrac{\vec a}{|\vec a|}$ とする。"],
                   solution=sol, meta=dict(v=v, k=a, deg=180))


# ============================================================
# 標準:一般の軸のまわりの 90° 回転(|a| が整数)
# ============================================================
def ninety_parts(rng):
    a = list(rng.choice(NICE[:3]) if rng.random() < 0.7 else rng.choice(NICE[3:]))   # |a| = 3 を多めに
    rng.shuffle(a)
    a = [x * rng.choice([1, -1]) for x in a]
    m = int(round(float(dot(a, a)) ** 0.5))
    v = ivec(rng)
    av = dot(a, v)
    axv = cross(a, v)
    if av == 0 or not any(axv):
        raise Retry
    par = scal(av / (m * m), a)
    perp = add(v, scal(-1, par))
    kv = scal(F(1, m), axv)
    ok = add(par, kv)
    return a, m, v, av, axv, par, perp, kv, ok


def ninety_solution(a, m, v, av, axv, par, perp, kv, ok):
    return [rf"$|\vec a|={m}$ より $\vec k=\dfrac{{1}}{{{m}}}{vtex(a)}$。$\cos90^\circ=0,\ \sin90^\circ=1$ なので "
            r"$\vec v\,'=\vec v_\parallel+\vec k\times\vec v$。",
            rf"$\vec v_\parallel=(\vec k\cdot\vec v)\vec k=\dfrac{{{tex_fr(av)}}}{{{m * m}}}{vtex(a)}={vtex(par)}$, "
            rf"$\vec v_\perp=\vec v-\vec v_\parallel={vtex(perp)}$",
            rf"$\vec k\times\vec v=\dfrac{{1}}{{{m}}}{vtex(a)}\times{vtex(v)}=\dfrac{{1}}{{{m}}}{vtex(axv)}={vtex(kv)}$",
            rf"$$\vec v\,'={vtex(par)}+{vtex(kv)}={vtex(ok)}$$"]


def g_ninety(rng):
    a, m, v, av, axv, par, perp, kv, ok = ninety_parts(rng)
    cands = [(add(par, scal(-1, kv)), "逆向きに回していないか($\\vec v\\times\\vec k$ ではなく $\\vec k\\times\\vec v$)。"),
             (kv, "軸に平行な成分 $\\vec v_\\parallel=(\\vec k\\cdot\\vec v)\\vec k$ も足す(回転しても変わらない)。"),
             (add(par, axv), f"$\\vec k$ は単位ベクトル。$\\vec a\\times\\vec v$ を $|\\vec a|={m}$ で割る。"),
             (add(v, kv), "$\\cos90^\\circ=0$。$\\vec v\\cos\\theta$ の項は消える。")]
    return Problem("一般の軸のまわりの 90° 回転", "標準",
                   rf"ベクトル $\vec v={vtex(v)}$ を、原点を通り $\vec a={vtex(a)}$ の向きの軸のまわりに $90^\circ$ 回転させたベクトルを求めよ。" + SENSE,
                   pick(ok, cands), hints=["ロドリゲスの回転公式:" + FORMULA,
                                           r"$\theta=90^\circ$ では $\vec v\,'=(\vec k\cdot\vec v)\vec k+\vec k\times\vec v$。$\vec k=\dfrac{\vec a}{|\vec a|}$。"],
                   solution=ninety_solution(a, m, v, av, axv, par, perp, kv, ok), meta=dict(v=v, k=a, deg=90))


def guided_ninety(rng):
    a, m, v, av, axv, par, perp, kv, ok = ninety_parts(rng)
    sol = ninety_solution(a, m, v, av, axv, par, perp, kv, ok)
    return ExamItem("ロドリゲスの回転公式",
                    rf"原点を通り $\vec a={vtex(a)}$ の向きの軸のまわりに, ベクトル $\vec v={vtex(v)}$ を $90^\circ$ 回転させる. " + SENSE.replace("。", ".")
                    + r" $\vec k=\dfrac{\vec a}{|\vec a|}$ として, 次の問いに答えよ.",
                    [Part(r"$\vec v$ を軸に平行な成分 $\vec v_\parallel$ と垂直な成分 $\vec v_\perp$ に分けよ.",
                          rf"\vec v_\parallel={vtex(par)},\ \vec v_\perp={vtex(perp)}", sol[:2]),
                     Part(r"回転させたベクトル $\vec v\,'$ を求めよ.", vtex(ok), sol[2:])],
                    meta=dict(v=v, k=a, deg=90, ok=ok))


# ============================================================
# 標準:座標軸のまわりの回転行列
# ============================================================
COS = {30: (r"\frac{\sqrt{3}}{2}", 3 ** 0.5 / 2), 45: (r"\frac{\sqrt{2}}{2}", 2 ** 0.5 / 2), 60: (r"\frac{1}{2}", 0.5),
       120: (r"-\frac{1}{2}", -0.5), 135: (r"-\frac{\sqrt{2}}{2}", -(2 ** 0.5) / 2), 150: (r"-\frac{\sqrt{3}}{2}", -(3 ** 0.5) / 2)}
SIN = {30: (r"\frac{1}{2}", 0.5), 45: (r"\frac{\sqrt{2}}{2}", 2 ** 0.5 / 2), 60: (r"\frac{\sqrt{3}}{2}", 3 ** 0.5 / 2),
       120: (r"\frac{\sqrt{3}}{2}", 3 ** 0.5 / 2), 135: (r"\frac{\sqrt{2}}{2}", 2 ** 0.5 / 2), 150: (r"\frac{1}{2}", 0.5)}


def neg(e):
    t, x = e
    return (t[1:] if t.startswith("-") else "-" + t, -x)


def axis_matrix(i, c, s):
    """座標軸 i のまわりの回転行列(要素は (TeX, 値))"""
    one, zero = ("1", 1.0), ("0", 0.0)
    if i == 0:
        return [[one, zero, zero], [zero, c, neg(s)], [zero, s, c]]
    if i == 1:
        return [[c, zero, s], [zero, one, zero], [neg(s), zero, c]]
    return [[c, neg(s), zero], [s, c, zero], [zero, zero, one]]


def mtex(M):
    return r"\begin{pmatrix}" + r"\\ ".join("&".join(e[0] for e in row) for row in M) + r"\end{pmatrix}"


def mval(M):
    return tuple(round(e[1], 12) for row in M for e in row)


def g_matrix(rng):
    i = rng.randrange(3)
    deg = rng.choice(list(COS))
    c, s = COS[deg], SIN[deg]
    ok = axis_matrix(i, c, s)
    T = [[ok[r][q] for r in range(3)] for q in range(3)]
    cands = [(T, "それは逆向き($-\\theta$)の回転行列(転置行列)。"
                 + (" $y$ 軸のまわりでは、$\\vec k\\times$ の行列 $K$ から符号が $x,\\ z$ 軸と逆の位置に来る。" if i == 1 else "")),
             (axis_matrix(i, s, c), "$\\cos\\theta$ と $\\sin\\theta$ の位置を確かめよう(対角成分は $\\cos\\theta$)。"),
             (axis_matrix((i + 1) % 3, c, s), f"回転軸は ${AXIS_NAME[i]}$ 軸。軸の成分の行と列は $(1,0,0)$ のように変わらない。"),
             (axis_matrix((i + 2) % 3, c, s), f"回転軸は ${AXIS_NAME[i]}$ 軸。")]
    choices = [Choice(mtex(ok), True, value=mval(ok))]
    seen = [mval(ok)]
    for M, why in cands:
        if mval(M) in seen:
            continue
        choices.append(Choice(mtex(M), why=why, value=mval(M)))
        seen.append(mval(M))
        if len(choices) == 4:
            break
    if len(choices) < 4:
        raise Retry
    k = [0, 0, 0]
    k[i] = 1
    Ktex = {0: r"\begin{pmatrix}0&0&0\\0&0&-1\\0&1&0\end{pmatrix}", 1: r"\begin{pmatrix}0&0&1\\0&0&0\\-1&0&0\end{pmatrix}",
            2: r"\begin{pmatrix}0&-1&0\\1&0&0\\0&0&0\end{pmatrix}"}[i]
    sol = [rf"$\vec k={vtex(k)}$ のとき $K={Ktex}$, $K^2$ は ${AXIS_NAME[i]}$ 成分以外の対角成分が $-1$ の行列。",
           r"$R=I+\sin\theta K+(1-\cos\theta)K^2$ に $\cos" + rf"{deg}^\circ={c[0]},\ \sin{deg}^\circ={s[0]}$ を入れて",
           rf"$$R={mtex(ok)}$$"]
    return Problem("座標軸のまわりの回転行列", "標準",
                   rf"${AXIS_NAME[i]}$ 軸のまわりに ${deg}^\circ$ 回転させる回転行列 $R$ を求めよ。" + SENSE,
                   choices, hints=[r"行列の形:$R=I+\sin\theta\,K+(1-\cos\theta)K^2$、$K\vec v=\vec k\times\vec v$。",
                                   r"$R$ の各列は、基本ベクトル $\vec e_1,\ \vec e_2,\ \vec e_3$ を回転させたもの。"],
                   solution=sol, meta=dict(axis=i, deg=deg))


REGISTRY = [
    ProblemType("rg_axis", "座標軸のまわりの回転", "基本", g_axis),
    ProblemType("rg_half", "一般の軸のまわりの 180° 回転", "標準", g_half),
    ProblemType("rg_ninety", "一般の軸のまわりの 90° 回転", "標準", g_ninety, guided=guided_ninety),
    ProblemType("rg_matrix", "座標軸のまわりの回転行列", "標準", g_matrix),
]
GUIDED_NAMES = {"rg_ninety": "平行・垂直成分 → 90° 回転(ロドリゲスの回転公式)"}


def tab():
    render("rodrigues", REGISTRY, "ベクトル解析 発展 ロドリゲスの回転公式")
