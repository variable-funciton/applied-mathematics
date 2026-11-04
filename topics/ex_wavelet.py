"""数学IV フーリエ解析 発展 ハール・ウェーブレット:演習の問題生成

ハール変換は「平均と差の半分」:a = (x_0 + x_1)/2, d = (x_0 − x_1)/2(ページ本文と同じ)。
4つのデータ (x_0, x_1, x_2, x_3) の完全な変換は (s | D | d_0, d_1):
    x_0 = s + D + d_0,  x_1 = s + D − d_0,  x_2 = s − D + d_1,  x_3 = s − D − d_1
関数で書くと f = s φ + D ψ(x) + d_0 ψ(2x) + d_1 ψ(2x − 1)。値はすべて整数になるように作る。
"""
from fractions import Fraction

from topics.ex_core import Choice, ExamItem, Part, Problem, ProblemType, Retry, render, tex_fr

F = Fraction
CONV = (r"ハール変換は, 隣どうしの平均 $\dfrac{x_0+x_1}{2}$ と差の半分 $\dfrac{x_0-x_1}{2}$ に分ける操作。"
        r"$\psi(x)=1\ (0\le x<\frac12),\ -1\ (\frac12\le x<1)$, それ以外 $0$。")


def vec(v):
    return "(" + ",\\ ".join(tex_fr(F(x)) for x in v) + ")"


def tr(s, D, d0, d1):
    return rf"\left({tex_fr(F(s))}\ \middle|\ {tex_fr(F(D))}\ \middle|\ {tex_fr(F(d0))},\ {tex_fr(F(d1))}\right)"


def data_from(s, D, d0, d1):
    return [s + D + d0, s + D - d0, s - D + d1, s - D - d1]


def pick_coeffs(rng, nonneg=True):
    while True:
        s, D, d0, d1 = rng.randint(2, 8), rng.choice([-3, -2, -1, 1, 2, 3]), rng.choice([-2, -1, 1, 2, 3]), rng.choice([-3, -2, -1, 1, 2])
        x = data_from(s, D, d0, d1)
        if (not nonneg or min(x) >= 0) and len({d0, d1}) == 2 and abs(D) not in (abs(d0), abs(d1)):
            return s, D, d0, d1, x


def distinct(ok, cands, why_default="計算を確かめよう。"):
    choices = [Choice(ok, True, value=ok)]
    for t, why in cands:
        if all(c.tex != t for c in choices) and len(choices) < 4:
            choices.append(Choice(t, why=why or why_default, value=t))
    if len(choices) < 4:
        raise Retry
    return choices


def full_solution(x, s, D, d0, d1):
    a0, a1 = F(x[0] + x[1], 2), F(x[2] + x[3], 2)
    return [rf"1回目:平均 $\left(\frac{{{x[0]}+{x[1]}}}{{2}},\ \frac{{{x[2]}+{x[3]}}}{{2}}\right)=({tex_fr(a0)},\ {tex_fr(a1)})$, "
            rf"差の半分 $\left(\frac{{{x[0]}-{x[1]}}}{{2}},\ \frac{{{x[2]}-{x[3]}}}{{2}}\right)=({tex_fr(F(d0))},\ {tex_fr(F(d1))})$。".replace("+-", "-").replace("--", "+"),
            rf"2回目:平均 $\frac{{{tex_fr(a0)}+{tex_fr(a1)}}}{{2}}={tex_fr(F(s))}$, 差の半分 $\frac{{{tex_fr(a0)}-{tex_fr(a1)}}}{{2}}={tex_fr(F(D))}$。"
            .replace("+-", "-").replace("--", "+"),
            rf"よって $${tr(s, D, d0, d1)}$$"]


# ------------------------------------------------------------
# 基本:1段階のハール変換
# ------------------------------------------------------------
def g_step(rng):
    s, D, d0, d1, x = pick_coeffs(rng)
    a0, a1 = s + D, s - D
    ok = rf"(a_0,\ a_1)=({a0},\ {a1}),\ \ (d_0,\ d_1)=({d0},\ {d1})"
    cands = [(rf"(a_0,\ a_1)=({a0},\ {a1}),\ \ (d_0,\ d_1)=({-d0},\ {-d1})", r"差の半分は $\dfrac{x_0-x_1}{2}$(左 − 右)。"),
             (rf"(a_0,\ a_1)=({2 * a0},\ {2 * a1}),\ \ (d_0,\ d_1)=({2 * d0},\ {2 * d1})", r"和と差を $2$ で割るのを忘れていないか。"),
             (rf"(a_0,\ a_1)=({tex_fr(F(x[0] + x[2], 2))},\ {tex_fr(F(x[1] + x[3], 2))}),\ \ (d_0,\ d_1)=({tex_fr(F(x[0] - x[2], 2))},\ {tex_fr(F(x[1] - x[3], 2))})",
              r"組にするのは隣どうし $(x_0,x_1)$, $(x_2,x_3)$。"),
             (rf"(a_0,\ a_1)=({a0},\ {a1}),\ \ (d_0,\ d_1)=({d1},\ {d0})", r"$d_0$ は $(x_0,x_1)$, $d_1$ は $(x_2,x_3)$ から。")]
    return Problem("1段階のハール変換", "基本",
                   rf"データ ${vec(x)}$ を1段階ハール変換し, 平均 $a_k=\dfrac{{x_{{2k}}+x_{{2k+1}}}}{{2}}$ と差の半分 $d_k=\dfrac{{x_{{2k}}-x_{{2k+1}}}}{{2}}$ $(k=0,1)$ を求めよ。",
                   distinct(ok, cands), hints=[CONV, r"$(x_0,x_1)$ と $(x_2,x_3)$ を組にする。"],
                   solution=[rf"$a_0=\frac{{{x[0]}+{x[1]}}}{{2}}={a0}$, $a_1=\frac{{{x[2]}+{x[3]}}}{{2}}={a1}$, "
                             rf"$d_0=\frac{{{x[0]}-{x[1]}}}{{2}}={d0}$, $d_1=\frac{{{x[2]}-{x[3]}}}{{2}}={d1}$。".replace("+-", "-").replace("--", "+")],
                   meta=dict(x=x, a=(a0, a1), d=(d0, d1)))


# ------------------------------------------------------------
# 標準:4つのデータの完全なハール変換
# ------------------------------------------------------------
def g_full(rng):
    s, D, d0, d1, x = pick_coeffs(rng)
    ok = tr(s, D, d0, d1)
    cands = [(tr(s, -D, d0, d1), r"2回目の差の半分も「左 − 右」:$\dfrac{a_0-a_1}{2}$。"),
             (tr(s, 2 * D, d0, d1), r"2回目も差を $2$ で割る。"),
             (tr(s, D, -d0, -d1), r"差の半分は $\dfrac{x_0-x_1}{2}$(左 − 右)。"),
             (tr(2 * s, D, d0, d1), r"2回目の平均も $2$ で割る(全体の平均になる)。"),
             (tr(s, D, d1, d0), r"$d_0$ は $(x_0,x_1)$, $d_1$ は $(x_2,x_3)$ から。")]
    return Problem("ハール変換(4つのデータ)", "標準",
                   rf"データ ${vec(x)}$ のハール変換 $(s\,|\,D\,|\,d_0,d_1)$ を求めよ。ただし, 1回目で平均 $(a_0,a_1)$ と差の半分 $(d_0,d_1)$ に分け, "
                   r"2回目で $(a_0,a_1)$ を平均 $s$ と差の半分 $D$ に分ける。",
                   distinct(ok, cands), hints=[CONV, "平均と差の半分を, 平均の方にもう一度行う。"],
                   solution=full_solution(x, s, D, d0, d1), meta=dict(x=x, c=(s, D, d0, d1)))


# ------------------------------------------------------------
# 基本:逆変換
# ------------------------------------------------------------
def g_inv(rng):
    s, D, d0, d1, x = pick_coeffs(rng)
    ok = vec(x)
    cands = [(vec(data_from(s, -D, d0, d1)), r"$a_0=s+D$, $a_1=s-D$(符号の順番に注意)。"),
             (vec(data_from(s, D, -d0, -d1)), r"$x_0=a_0+d_0$, $x_1=a_0-d_0$。"),
             (vec([s + D + d0, s - D + d1, s + D - d0, s - D - d1]), r"$x_0,x_1$ は $a_0$ から, $x_2,x_3$ は $a_1$ から作る。"),
             (vec(data_from(s, D, d1, d0)), r"$d_0$ は前半 $(x_0,x_1)$, $d_1$ は後半 $(x_2,x_3)$ の差。")]
    return Problem("ハール変換の逆変換", "基本",
                   rf"4つのデータのハール変換が ${tr(s, D, d0, d1)}$ であった。もとのデータ $(x_0,x_1,x_2,x_3)$ を求めよ。",
                   distinct(ok, cands), hints=[r"平均 $a$ と差の半分 $d$ から, $x_0=a+d$, $x_1=a-d$ でもとに戻る。", "粗い方から順に戻す。"],
                   solution=[rf"$a_0=s+D={s + D}$, $a_1=s-D={s - D}$。",
                             rf"$x_0,x_1=a_0\pm d_0={s + D}\pm({d0})$, $x_2,x_3=a_1\pm d_1={s - D}\pm({d1})$ より $${vec(x)}$$"],
                   meta=dict(x=x, c=(s, D, d0, d1)))


# ------------------------------------------------------------
# 標準:階段関数のハール展開
# ------------------------------------------------------------
def step_tex(x):
    return (r"f(x)=\begin{cases}" + r"\\".join(rf"{v}&\left({tex_fr(F(i, 4))}\le x<{tex_fr(F(i + 1, 4))}\right)" for i, v in enumerate(x))
            + r"\end{cases}")


def exp_tex(s, D, d0, d1):
    out = f"{s}" + "".join(f"{'+' if c >= 0 else '-'}{abs(c) if abs(c) != 1 else ''}{b}"
                           for c, b in ((D, r"\psi(x)"), (d0, r"\psi(2x)"), (d1, r"\psi(2x-1)")))
    return "f(x)=" + out


def g_expand(rng):
    s, D, d0, d1, x = pick_coeffs(rng)
    ok = exp_tex(s, D, d0, d1)
    cands = [(exp_tex(s, -D, d0, d1), r"$\psi$ は前半で $1$, 後半で $-1$:係数は(前半の平均 − 後半の平均)$\div2$。"),
             (exp_tex(s, D, -d0, -d1), r"$\psi(2x)$ は $[0,\frac14)$ で $1$, $[\frac14,\frac12)$ で $-1$。"),
             (exp_tex(s, D, d1, d0), r"$\psi(2x)$ は前半 $[0,\frac12)$, $\psi(2x-1)$ は後半 $[\frac12,1)$ を受けもつ。"),
             (exp_tex(s, 2 * D, 2 * d0, 2 * d1), r"差は $2$ で割る($\psi$ の値は $\pm1$)。")]
    return Problem("階段関数のハール展開", "標準",
                   rf"区間 $[0,1)$ の階段関数 ${step_tex(x)}$ を, $1,\ \psi(x),\ \psi(2x),\ \psi(2x-1)$ の1次結合で表せ。",
                   distinct(ok, cands), hints=[CONV, "4つの値をデータと見てハール変換すると, 係数が出る。"],
                   solution=full_solution(x, s, D, d0, d1)[:2] + [rf"よって $${ok}$$"], meta=dict(x=x, c=(s, D, d0, d1)))


# ------------------------------------------------------------
# 標準:ハール係数の積分 ∫_0^1 f(x) ψ(2^j x − k) dx
# ------------------------------------------------------------
def int_value(kind, j, k):
    return F(-1, 4 ** (j + 1)) if kind == "x" else F(-(2 * k + 1), 4 * 8 ** j)


def g_int(rng):
    kind = rng.choice(["x", "x2"])
    j = rng.choice([0, 1, 1, 2])
    k = rng.randrange(2 ** j)
    v = int_value(kind, j, k)
    ftex = "x" if kind == "x" else "x^2"
    psi = r"\psi(x)" if j == 0 else (rf"\psi({2 ** j}x)" if k == 0 else rf"\psi({2 ** j}x-{k})")
    lo, mid, hi = F(k, 2 ** j), F(2 * k + 1, 2 ** (j + 1)), F(k + 1, 2 ** j)
    p = 2 if kind == "x" else 3
    cands = [(tex_fr(-v), r"$\psi$ は前半で $+1$, 後半で $-1$。後半の方が $f$ が大きいので, 答えは負。"),
             (tex_fr(v * 2 ** j), r"区間の幅は $2^{-j}$。置換 $u=2^jx-k$ で $dx=2^{-j}du$ が出る。"),
             (tex_fr(v / 2), r"積分区間を確かめよう(前半 $[k2^{-j},\ (k+\frac12)2^{-j})$, 後半 $[(k+\frac12)2^{-j},\ (k+1)2^{-j})$)。"),
             (tex_fr(v * 2), "計算を確かめよう。")]
    sol = [rf"${psi}$ は $\left[{tex_fr(lo)},\ {tex_fr(mid)}\right)$ で $1$, $\left[{tex_fr(mid)},\ {tex_fr(hi)}\right)$ で $-1$, それ以外で $0$。",
           rf"$$\int_{{{tex_fr(lo)}}}^{{{tex_fr(mid)}}}{ftex}\,dx-\int_{{{tex_fr(mid)}}}^{{{tex_fr(hi)}}}{ftex}\,dx"
           rf"=\frac{{1}}{{{p}}}\left\{{2\left({tex_fr(mid)}\right)^{p}-\left({tex_fr(lo)}\right)^{p}-\left({tex_fr(hi)}\right)^{p}\right\}}={tex_fr(v)}$$"]
    return Problem("ハール係数の積分", "標準",
                   rf"$\psi(x)=1\ (0\le x<\frac12),\ -1\ (\frac12\le x<1)$, それ以外 $0$ とする。"
                   rf"$\displaystyle\int_0^1{ftex}\,{psi}\,dx$ を求めよ。".replace("-0)", ")"),
                   distinct(tex_fr(v), cands), hints=[r"$\psi(2^jx-k)$ が $1$ と $-1$ になる区間を求め, 2つの積分に分ける。",
                                                       r"$\displaystyle\int_a^bx^n\,dx=\frac{b^{n+1}-a^{n+1}}{n+1}$。"],
                   solution=sol, meta=dict(kind=kind, j=j, k=k, val=float(v)))


# ------------------------------------------------------------
# 誘導:ハール変換 → 圧縮(小さい係数を 0 にする)
# ------------------------------------------------------------
def guided_compress(rng):
    while True:
        s, D, d0, d1, x = pick_coeffs(rng)
        mags = sorted([(abs(D), "D"), (abs(d0), "d0"), (abs(d1), "d1")])
        if mags[0][0] != mags[1][0]:
            break
    which = mags[0][1]
    c = dict(D=D, d0=d0, d1=d1)
    orig = c[which]
    c[which] = 0
    y = data_from(s, c["D"], c["d0"], c["d1"])
    err = sum((a - b) ** 2 for a, b in zip(x, y))
    name = {"D": "D", "d0": "d_0", "d1": "d_1"}[which]
    return ExamItem("ハール変換と圧縮",
                    rf"データ ${vec(x)}$ について, 次の問いに答えよ. ハール変換は, 隣どうしの平均 $\dfrac{{x_0+x_1}}{{2}}$ と差の半分 $\dfrac{{x_0-x_1}}{{2}}$ に分ける操作を, "
                    r"平均の方にくり返して $(s\,|\,D\,|\,d_0,d_1)$ とする.",
                    [Part(r"ハール変換 $(s\,|\,D\,|\,d_0,d_1)$ を求めよ.", tr(s, D, d0, d1), full_solution(x, s, D, d0, d1)),
                     Part(r"$D,\ d_0,\ d_1$ のうち絶対値が最小のものを $0$ にして逆変換したデータと, もとのデータとの差の2乗和を求めよ.",
                          rf"{vec(y)},\ \ {err}",
                          [rf"絶対値が最小なのは ${name}={orig}$。これを $0$ にして逆変換すると $${vec(y)}$$",
                           rf"差の2乗和は ${err}$(${name}$ を消すと誤差は ${'4' if which == 'D' else '2'}{name}^2$)。"])],
                    meta=dict(x=x, y=y, err=err))


def guided_expand(rng):
    s, D, d0, d1, x = pick_coeffs(rng)
    return ExamItem("階段関数のハール展開",
                    rf"区間 $[0,1)$ の階段関数 ${step_tex(x)}$ について, 次の問いに答えよ. "
                    r"$\psi(x)=1\ (0\le x<\frac12),\ -1\ (\frac12\le x<1)$, それ以外 $0$ とする.",
                    [Part(r"$f$ を $1,\ \psi(x),\ \psi(2x),\ \psi(2x-1)$ の1次結合で表せ.", exp_tex(s, D, d0, d1),
                          full_solution(x, s, D, d0, d1)[:2]),
                     Part(r"$\displaystyle\int_0^1f(x)\psi(x)\,dx$ を求めよ.", str(D),
                          [r"$\int_0^1\psi\,dx=0$, $\psi(2x)$ と $\psi(2x-1)$ は $\psi$ と直交し, $\int_0^1\psi^2dx=1$ なので, (1) の式に $\psi$ をかけて積分すると",
                           rf"$$\int_0^1f\psi\,dx=D\int_0^1\psi^2dx={D}$$",
                           rf"(確かめ:$\displaystyle\int_0^{{1/2}}f\,dx-\int_{{1/2}}^1f\,dx=\frac{{{x[0]}+{x[1]}}}{{4}}-\frac{{{x[2]}+{x[3]}}}{{4}}={D}$)"])],
                    meta=dict(x=x, D=D))


REGISTRY = [
    ProblemType("wv_step", "1段階のハール変換", "基本", g_step),
    ProblemType("wv_inv", "ハール変換の逆変換", "基本", g_inv),
    ProblemType("wv_full", "ハール変換(4つのデータ)", "標準", g_full, guided=guided_compress),
    ProblemType("wv_expand", "階段関数のハール展開", "標準", g_expand, guided=guided_expand),
    ProblemType("wv_int", "ハール係数の積分", "標準", g_int),
]
GUIDED_NAMES = {"wv_full": "ハール変換 → 小さい係数を 0 にした圧縮と誤差",
                "wv_expand": "階段関数のハール展開 → ∫ f ψ dx"}


def tab():
    render("wavelet", REGISTRY, "フーリエ解析 発展 ハール・ウェーブレット")
