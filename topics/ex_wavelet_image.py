"""数学IV フーリエ解析 発展 ウェーブレットによる画像の類似度判定:演習の問題生成

2次元ハール変換(ページ本文と同じ):ブロック [[a, b], [c, d]] に対して
    A = (a+b+c+d)/4,  H = ((a+c)−(b+d))/4,  V = ((a+b)−(c+d))/4,  D = ((a+d)−(b+c))/4
    a = A+H+V+D,  b = A−H+V−D,  c = A+H−V−D,  d = A−H−V+D,  a²+b²+c²+d² = 4(A²+H²+V²+D²)
値がすべて整数になるように (A, H, V, D) から作る。
"""
import math
from fractions import Fraction

from topics.ex_core import Choice, ExamItem, Part, Problem, ProblemType, Retry, render, tex_fr

F = Fraction
CONV = (r"$A=\dfrac{a+b+c+d}{4}$, $H=\dfrac{(a+c)-(b+d)}{4}$(左右の差), "
        r"$V=\dfrac{(a+b)-(c+d)}{4}$(上下の差), $D=\dfrac{(a+d)-(b+c)}{4}$(斜め)。")
# 成分が整数でノルムも整数のベクトル(コサイン類似度が有理数になる)
NICE = [(1, 2, 2), (2, 1, 2), (2, 2, 1), (2, 3, 6), (3, 6, 2), (6, 2, 3), (1, 4, 8), (4, 1, 8), (8, 4, 1), (4, 4, 7), (4, 7, 4), (7, 4, 4)]


def distinct(ok, cands, why_default="計算を確かめよう。"):
    choices = [Choice(ok, True, value=ok)]
    for t, why in cands:
        if all(c.tex != t for c in choices) and len(choices) < 4:
            choices.append(Choice(t, why=why or why_default, value=t))
    if len(choices) < 4:
        raise Retry
    return choices


def block_from(A, H, V, D):
    return (A + H + V + D, A - H + V - D, A + H - V - D, A - H - V + D)


def coeffs(a, b, c, d):
    return (F(a + b + c + d, 4), F((a + c) - (b + d), 4), F((a + b) - (c + d), 4), F((a + d) - (b + c), 4))


def mat(a, b, c, d):
    return rf"\begin{{pmatrix}}{a}&{b}\\{c}&{d}\end{{pmatrix}}"


def ahvd(A, H, V, D):
    return rf"(A,\ H,\ V,\ D)=({tex_fr(F(A))},\ {tex_fr(F(H))},\ {tex_fr(F(V))},\ {tex_fr(F(D))})"


def pick(rng):
    while True:
        A = rng.randint(3, 8)
        H, V, D = rng.choice([-2, -1, 0, 1, 2, 3]), rng.choice([-2, -1, 1, 2]), rng.choice([-1, 0, 1, 2])
        x = block_from(A, H, V, D)
        if min(x) >= 0 and len({abs(H), abs(V), abs(D)}) == 3:
            return A, H, V, D, x


# ------------------------------------------------------------
# 基本:2×2 ブロックの2次元ハール変換
# ------------------------------------------------------------
def g_block(rng):
    A, H, V, D, x = pick(rng)
    a, b, c, d = x
    ok = ahvd(A, H, V, D)
    cands = [(ahvd(A, V, H, D), r"$H$ は左右の差 $(a+c)-(b+d)$, $V$ は上下の差 $(a+b)-(c+d)$。"),
             (ahvd(A, -H, -V, D), r"差は「左 − 右」「上 − 下」の順にとる。"),
             (ahvd(2 * A, 2 * H, 2 * V, 2 * D), r"4つの和・差を $4$ で割る。"),
             (ahvd(A, H, V, -D), r"$D=\dfrac{(a+d)-(b+c)}{4}$:左上と右下の和から右上と左下の和を引く。")]
    return Problem("2×2 ブロックのハール変換", "基本",
                   rf"画像の $2\times2$ のブロック ${mat(a, b, c, d)}$ を2次元ハール変換したときの $A,H,V,D$ を求めよ。"
                   rf"ただし $A=\dfrac{{a+b+c+d}}{{4}}$, $H=\dfrac{{(a+c)-(b+d)}}{{4}}$, $V=\dfrac{{(a+b)-(c+d)}}{{4}}$, $D=\dfrac{{(a+d)-(b+c)}}{{4}}$ とする "
                   rf"(ブロックを $\begin{{pmatrix}}a&b\\c&d\end{{pmatrix}}$ と書く)。",
                   distinct(ok, cands), hints=[CONV],
                   solution=[rf"$A=\frac{{{a}+{b}+{c}+{d}}}{{4}}={A}$, $H=\frac{{({a}+{c})-({b}+{d})}}{{4}}={H}$, "
                             rf"$V=\frac{{({a}+{b})-({c}+{d})}}{{4}}={V}$, $D=\frac{{({a}+{d})-({b}+{c})}}{{4}}={D}$。"],
                   meta=dict(x=x, c=(A, H, V, D)))


# ------------------------------------------------------------
# 基本:逆変換
# ------------------------------------------------------------
def g_inv(rng):
    A, H, V, D, x = pick(rng)
    ok = mat(*x)
    a2 = block_from(A, V, H, D)
    cands = [(mat(*a2), r"$H$ は左右の差:左の列に $+H$, 右の列に $-H$。"),
             (mat(*block_from(A, -H, -V, -D)), "符号を確かめよう。"),
             (mat(*[v for v in (A + H + V + D, A + H - V - D, A - H + V - D, A - H - V + D)]), r"$b$ は右上, $c$ は左下。"),
             (mat(*block_from(A, H, V, -D)), r"$D$ は左上・右下に $+D$, 右上・左下に $-D$。")]
    return Problem("逆変換", "基本",
                   rf"2次元ハール変換の係数が ${ahvd(A, H, V, D)}$ であるとき, もとのブロック $\begin{{pmatrix}}a&b\\c&d\end{{pmatrix}}$ を求めよ。",
                   distinct(ok, cands), hints=[CONV, r"$a=A+H+V+D$, $b=A-H+V-D$, $c=A+H-V-D$, $d=A-H-V+D$。"],
                   solution=[r"$a=A+H+V+D$, $b=A-H+V-D$, $c=A+H-V-D$, $d=A-H-V+D$ より",
                             rf"$a={x[0]}$, $b={x[1]}$, $c={x[2]}$, $d={x[3]}$。"], meta=dict(x=x, c=(A, H, V, D)))


# ------------------------------------------------------------
# 基本:どの係数が反応するか
# ------------------------------------------------------------
EDGE = {
    "縦": ("H だけが 0 でない", r"左右で明るさが違う(縦の境界):$a=c$, $b=d$ より $V=D=0$。"),
    "横": ("V だけが 0 でない", r"上下で明るさが違う(横の境界):$a=b$, $c=d$ より $H=D=0$。"),
    "市松": ("D だけが 0 でない", r"市松模様:$a=d$, $b=c$ より $H=V=0$。"),
    "一様": ("H, V, D はすべて 0", r"一様な明るさ:差はすべて $0$。"),
}


def g_edge(rng):
    kind = rng.choice(list(EDGE))
    p, q = rng.sample(range(0, 10), 2)
    blk = {"縦": (p, q, p, q), "横": (p, p, q, q), "市松": (p, q, q, p), "一様": (p, p, p, p)}[kind]
    ans, why = EDGE[kind]
    ok = rf"\text{{{ans}}}"
    cands = [(rf"\text{{{EDGE[k][0]}}}", EDGE[k][1]) for k in EDGE if k != kind]
    return Problem("どの係数が反応するか", "基本",
                   rf"ブロック ${mat(*blk)}$ を2次元ハール変換したとき, 細部の係数 $H,V,D$ について正しいものを選べ。",
                   distinct(ok, cands), hints=[CONV, "左右・上下・斜めのどの向きに明るさが変化しているかを見る。"],
                   solution=[why], meta=dict(kind=kind, blk=blk))


# ------------------------------------------------------------
# 標準:エネルギー(パーセバルの等式)
# ------------------------------------------------------------
def g_energy(rng):
    A, H, V, D, x = pick(rng)
    tot = sum(v * v for v in x)
    det = 4 * (H * H + V * V + D * D)
    kind = rng.choice(["total", "detail"])
    if kind == "total":
        ok = str(tot)
        cands = [(str(tot // 4), r"$a^2+b^2+c^2+d^2=4(A^2+H^2+V^2+D^2)$:$4$ 倍する。"), (str(2 * (A * A + H * H + V * V + D * D)), "係数は $4$ 倍。"),
                 (str(16 * (A * A + H * H + V * V + D * D)), r"$A$ などは和を $4$ で割った値。2乗の和の関係は $4$ 倍。"), (str(4 * (A + H + V + D) ** 2), "2乗の和をとる。")]
        q = r"$a^2+b^2+c^2+d^2$ を, ブロックの値を求めずに求めよ"
        sol = [r"$a^2+b^2+c^2+d^2=4\left(A^2+H^2+V^2+D^2\right)$ より", rf"$4({A * A}+{H * H}+{V * V}+{D * D})={tot}$。"]
    else:
        ok = tex_fr(F(det, tot))
        cands = [(tex_fr(F(H * H + V * V + D * D, A * A)), r"割るのは全体のエネルギー $4(A^2+H^2+V^2+D^2)$。"),
                 (tex_fr(F(4 * A * A, tot)), "これは近似 $A$ の割合。"), (tex_fr(F(abs(H) + abs(V) + abs(D), A + abs(H) + abs(V) + abs(D))), "エネルギーは2乗で測る。"),
                 (tex_fr(F(H * H + V * V + D * D, tot)), r"細部のエネルギーは $4(H^2+V^2+D^2)$。")]
        q = r"全体のエネルギー $a^2+b^2+c^2+d^2$ のうち, 細部 $H,V,D$ が受けもつ割合 $\dfrac{4(H^2+V^2+D^2)}{a^2+b^2+c^2+d^2}$ を求めよ"
        sol = [r"$a^2+b^2+c^2+d^2=4\left(A^2+H^2+V^2+D^2\right)$ より", rf"$\dfrac{{4({H * H}+{V * V}+{D * D})}}{{4({A * A}+{H * H}+{V * V}+{D * D})}}={ok}$。"]
    return Problem("エネルギー", "標準",
                   rf"2次元ハール変換の係数が ${ahvd(A, H, V, D)}$ であるブロック $\begin{{pmatrix}}a&b\\c&d\end{{pmatrix}}$ について, {q}。",
                   distinct(ok, cands), hints=[r"4つの模様 $\begin{pmatrix}1&1\\1&1\end{pmatrix}$, $\begin{pmatrix}1&-1\\1&-1\end{pmatrix}$, $\begin{pmatrix}1&1\\-1&-1\end{pmatrix}$, $\begin{pmatrix}1&-1\\-1&1\end{pmatrix}$ は互いに直交し, どれも2乗和は $4$。",
                                               r"$a^2+b^2+c^2+d^2=4\left(A^2+H^2+V^2+D^2\right)$。"],
                   solution=sol, meta=dict(kind=kind, x=x, c=(A, H, V, D)))


# ------------------------------------------------------------
# 標準:コサイン類似度と判定
# ------------------------------------------------------------
def norm_int(v):
    return math.isqrt(sum(t * t for t in v))


def g_cos(rng):
    u, v = rng.sample(NICE, 2)
    k = rng.choice([1, 1, 2])
    v2 = tuple(k * t for t in v)
    dot = sum(p * q for p, q in zip(u, v2))
    cs = F(dot, norm_int(u) * norm_int(v2))
    tau = rng.choice([F(4, 5), F(9, 10), F(19, 20)])
    same = cs >= tau
    ok = rf"\cos\theta={tex_fr(cs)},\ \text{{{'同じ' if same else '違う'}}}"
    cands = [(rf"\cos\theta={tex_fr(cs)},\ \text{{{'違う' if same else '同じ'}}}", rf"$\tau={tex_fr(tau)}$ と大小を比べる($\cos\theta\ge\tau$ なら「同じ」)。"),
             (rf"\cos\theta={dot},\ \text{{同じ}}", r"内積を $|\boldsymbol u||\boldsymbol v|$ で割る。"),
             (rf"\cos\theta={tex_fr(F(dot, norm_int(u) ** 2))},\ \text{{{'同じ' if F(dot, norm_int(u) ** 2) >= tau else '違う'}}}", r"割るのは $|\boldsymbol u|^2$ ではなく $|\boldsymbol u||\boldsymbol v|$。"),
             (rf"\cos\theta={tex_fr(1 - cs)},\ \text{{違う}}", "コサイン類似度は $1$ に近いほど似ている。")]
    vt = "(" + ",\\ ".join(map(str, v2)) + ")"
    ut = "(" + ",\\ ".join(map(str, u)) + ")"
    return Problem("コサイン類似度と判定", "標準",
                   rf"2枚の画像の特徴ベクトルが $\boldsymbol u={ut}$, $\boldsymbol v={vt}$ である。コサイン類似度 "
                   rf"$\cos\theta=\dfrac{{\boldsymbol u\cdot\boldsymbol v}}{{|\boldsymbol u||\boldsymbol v|}}$ を求め, "
                   rf"しきい値 $\tau={tex_fr(tau)}$ で $\cos\theta\ge\tau$ なら「同じ」と判定するとき, 結果を答えよ。",
                   distinct(ok, cands), hints=[r"$\boldsymbol u\cdot\boldsymbol v=u_1v_1+u_2v_2+u_3v_3$, $|\boldsymbol u|=\sqrt{u_1^2+u_2^2+u_3^2}$。"],
                   solution=[rf"$\boldsymbol u\cdot\boldsymbol v={dot}$, $|\boldsymbol u|={norm_int(u)}$, $|\boldsymbol v|={norm_int(v2)}$ より "
                             rf"$\cos\theta=\dfrac{{{dot}}}{{{norm_int(u) * norm_int(v2)}}}={tex_fr(cs)}$。",
                             rf"$\tau={tex_fr(tau)}$ と比べて「{'同じ' if same else '違う'}」。" + ("(ベクトルを定数倍しても角度は変わらない。)" if k > 1 else "")],
                   meta=dict(u=u, v=v2, cs=cs, tau=tau, same=same))


# ------------------------------------------------------------
# 誘導
# ------------------------------------------------------------
def guided_block(rng):
    A, H, V, D, x = pick(rng)
    a, b, c, d = x
    tot = sum(t * t for t in x)
    return ExamItem("2次元ハール変換とエネルギー",
                    rf"画像の $2\times2$ のブロック ${mat(a, b, c, d)}$ について, $A=\dfrac{{a+b+c+d}}{{4}}$, $H=\dfrac{{(a+c)-(b+d)}}{{4}}$, "
                    rf"$V=\dfrac{{(a+b)-(c+d)}}{{4}}$, $D=\dfrac{{(a+d)-(b+c)}}{{4}}$ とする(ブロックを $\begin{{pmatrix}}a&b\\c&d\end{{pmatrix}}$ と書く). 次の問いに答えよ.",
                    [Part(r"$A,H,V,D$ を求めよ.", ahvd(A, H, V, D),
                          [rf"$A=\frac{{{a + b + c + d}}}{{4}}={A}$, $H=\frac{{{(a + c) - (b + d)}}}{{4}}={H}$, $V=\frac{{{(a + b) - (c + d)}}}{{4}}={V}$, $D=\frac{{{(a + d) - (b + c)}}}{{4}}={D}$。"]),
                     Part(r"$a^2+b^2+c^2+d^2$ と $4\left(A^2+H^2+V^2+D^2\right)$ をそれぞれ計算し, 等しいことを確かめよ.",
                          rf"a^2+b^2+c^2+d^2=4\left(A^2+H^2+V^2+D^2\right)={tot}",
                          [rf"$a^2+b^2+c^2+d^2={a * a}+{b * b}+{c * c}+{d * d}={tot}$。",
                           rf"$4(A^2+H^2+V^2+D^2)=4({A * A}+{H * H}+{V * V}+{D * D})={tot}$。"])], meta=dict(x=x))


def guided_cos(rng):
    """2つのブロックの細部 (H, V, D) を特徴ベクトルにしてコサイン類似度を求める"""
    while True:
        u, v = rng.sample(NICE, 2)
        A1, A2 = rng.randint(9, 12), rng.randint(9, 12)
        x1, x2 = block_from(A1, *u), block_from(A2, *v)
        if min(x1 + x2) >= 0:
            break
    dot = sum(p * q for p, q in zip(u, v))
    cs = F(dot, norm_int(u) * norm_int(v))
    return ExamItem("特徴ベクトルとコサイン類似度",
                    rf"2つの画像のブロック $P={mat(*x1)}$, $Q={mat(*x2)}$ について, 2次元ハール変換の細部の係数を並べたベクトル "
                    rf"$(H,V,D)$ を特徴ベクトルとする. ただし $H=\dfrac{{(a+c)-(b+d)}}{{4}}$, $V=\dfrac{{(a+b)-(c+d)}}{{4}}$, $D=\dfrac{{(a+d)-(b+c)}}{{4}}$ "
                    rf"(ブロックを $\begin{{pmatrix}}a&b\\c&d\end{{pmatrix}}$ と書く)である. 次の問いに答えよ.",
                    [Part(r"$P$, $Q$ の特徴ベクトル $\boldsymbol u$, $\boldsymbol v$ を求めよ.",
                          rf"\boldsymbol u=({u[0]},\ {u[1]},\ {u[2]}),\ \boldsymbol v=({v[0]},\ {v[1]},\ {v[2]})",
                          [rf"$P$:$H=\frac{{{(x1[0] + x1[2]) - (x1[1] + x1[3])}}}{{4}}={u[0]}$, $V=\frac{{{(x1[0] + x1[1]) - (x1[2] + x1[3])}}}{{4}}={u[1]}$, $D=\frac{{{(x1[0] + x1[3]) - (x1[1] + x1[2])}}}{{4}}={u[2]}$。",
                           rf"$Q$:$H={v[0]}$, $V={v[1]}$, $D={v[2]}$。"]),
                     Part(r"コサイン類似度 $\dfrac{\boldsymbol u\cdot\boldsymbol v}{|\boldsymbol u||\boldsymbol v|}$ を求めよ.", tex_fr(cs),
                          [rf"$\boldsymbol u\cdot\boldsymbol v={dot}$, $|\boldsymbol u|={norm_int(u)}$, $|\boldsymbol v|={norm_int(v)}$ より "
                           rf"$\dfrac{{{dot}}}{{{norm_int(u) * norm_int(v)}}}={tex_fr(cs)}$。"])], meta=dict(u=u, v=v, x1=x1, x2=x2))


REGISTRY = [
    ProblemType("wi_block", "2×2 ブロックのハール変換", "基本", g_block, guided=guided_block),
    ProblemType("wi_inv", "逆変換", "基本", g_inv),
    ProblemType("wi_edge", "どの係数が反応するか", "基本", g_edge),
    ProblemType("wi_energy", "エネルギー", "標準", g_energy),
    ProblemType("wi_cos", "コサイン類似度と判定", "標準", g_cos, guided=guided_cos),
]
GUIDED_NAMES = {"wi_block": "2×2 ブロックのハール変換 → エネルギーの保存",
                "wi_cos": "細部の係数で特徴ベクトル → コサイン類似度"}


def tab():
    render("wvimg", REGISTRY, "フーリエ解析 発展 ウェーブレットによる画像の類似度判定")
