"""数学IV ラプラス変換 第1章 4+. 部分分数分解(ヘヴィサイドの方法):演習の問題生成

係数が整数になるように, 部分分数の係数から分子を作る。
"""
from fractions import Fraction

from topics.ex_core import Choice, ExamItem, Part, Problem, ProblemType, Retry, render
from topics.partial_fraction import e_t, fr, frac_term, from_roots, lin, pmul, ptex, pval, ssum

F = Fraction


def distinct(ok, cands):
    choices = [Choice(ok, True, value=ok)]
    for t, why in cands:
        if all(c.tex != t for c in choices) and len(choices) < 4:
            choices.append(Choice(t, why=why, value=t))
    if len(choices) < 4:
        raise Retry
    return choices


def padd(*ps):
    n = max(len(p) for p in ps)
    out = [F(0)] * n
    for p in ps:
        for i, c in enumerate(p):
            out[n - len(p) + i] += F(c)
    while len(out) > 1 and out[0] == 0:
        out.pop(0)
    return out


def ex(r):
    return e_t(r) or ""


def den3(rts):
    """(s−p₁)(s−p₂)… の TeX。p = 0 の因数 s は先頭に括弧なしで置く"""
    rts = sorted(rts, key=lambda p: p != 0)
    return "".join(lin(p) if p == 0 else "(" + lin(p) + ")" for p in rts)


def three(rng):
    rts = rng.sample([-3, -2, -1, 0, 1, 2, 3], 3)
    cs = [rng.choice([1, 2, 3, -1, -2]) for _ in range(3)]
    N = padd(*[[c * x for x in from_roots([q for j, q in enumerate(rts) if j != k])] for k, c in enumerate(cs)])
    if N == [0]:
        raise Retry
    return rts, cs, N


def pf3(cs, rts):
    return "".join(frac_term(c, lin(p), i == 0) for i, (c, p) in enumerate(zip(cs, rts)))


# ------------------------------------------------------------
# 基本:隠蔽法で係数を1つ
# ------------------------------------------------------------
def g_cover(rng):
    rts, cs, N = three(rng)
    k = rng.randrange(3)
    p = rts[k]
    others = den3([q for j, q in enumerate(rts) if j != k])
    Np = pval(N, F(p))
    cands = [(fr(cs[(k + 1) % 3]), "隠すのは $(s-p_k)$ の因数, 代入するのは $s=p_k$。"), (fr(cs[(k + 2) % 3]), "隠すのは $(s-p_k)$ の因数, 代入するのは $s=p_k$。"),
             (fr(Np), "分子に代入した値を, 残りの因数の値で割る。"), (fr(-cs[k]), r"残りの因数 $(p_k-p_j)$ の符号に注意。")]
    return Problem("隠蔽法", "基本",
                   rf"$\dfrac{{{ptex(N)}}}{{{den3(rts)}}}=\dfrac{{A}}{{{lin(rts[0])}}}+\dfrac{{B}}{{{lin(rts[1])}}}+\dfrac{{C}}{{{lin(rts[2])}}}$ と部分分数に分けるとき, "
                   rf"${'ABC'[k]}$ の値を求めよ。",
                   distinct(fr(cs[k]), cands), hints=[r"両辺に $(s-p)$ を掛けて $s=p$ を代入すると, その係数だけが残る(隠蔽法)。"],
                   solution=[rf"${'ABC'[k]}=\left.\dfrac{{{ptex(N)}}}{{{others}}}\right|_{{s={p}}}={fr(cs[k])}$。"], meta=dict(rts=rts, cs=cs, k=k))


# ------------------------------------------------------------
# 標準:異なる3つの根 → 逆ラプラス変換
# ------------------------------------------------------------
def g_inv3(rng):
    rts, cs, N = three(rng)
    ok = ssum([(c, ex(p)) for c, p in zip(cs, rts)])
    cands = [(ssum([(c, ex(-p)) for c, p in zip(cs, rts)]), r"$\mathcal L^{-1}\left[\frac{1}{s-p}\right]=e^{pt}$。指数の符号に注意。"),
             (ssum([(c, ex(p)) for c, p in zip(cs[1:] + cs[:1], rts)]), "係数と分母の対応を確かめよう。"),
             (ssum([(c if i else -c, ex(p)) for i, (c, p) in enumerate(zip(cs, rts))]), r"残りの因数 $(p_k-p_j)$ の符号に注意。")]
    return Problem("逆ラプラス変換(3つの因数)", "標準",
                   rf"$F(s)=\dfrac{{{ptex(N)}}}{{{den3(rts)}}}$ の逆ラプラス変換を求めよ。",
                   distinct(ok, cands), hints=[r"隠蔽法:$\frac{1}{s-p}$ の係数は $\left[(s-p)F(s)\right]_{s=p}$。",
                                               r"展開定理:$\mathcal L^{-1}\left[\frac{N(s)}{D(s)}\right]=\sum\frac{N(p_k)}{D^{\prime}(p_k)}e^{p_kt}$。"],
                   solution=[rf"$F(s)={pf3(cs, rts)}$(隠蔽法)。", rf"$f(t)={ok}$。"], meta=dict(rts=rts, cs=cs))


# ------------------------------------------------------------
# 標準:重解
# ------------------------------------------------------------
def dbl_parts(rng):
    al, be = rng.sample([-3, -2, -1, 0, 1, 2], 2)
    A, B, C = rng.choice([-2, -1, 0, 1, 2]), rng.choice([1, 2, 3, -1, -2]), rng.choice([1, 2, -1, -2, 3])
    N = padd([A * x for x in pmul([1, -al], [1, -be])], [B, -B * be], [C * x for x in pmul([1, -al], [1, -al])])
    if N == [0]:
        raise Retry
    return al, be, A, B, C, N


def g_double(rng):
    al, be, A, B, C, N = dbl_parts(rng)
    ok = ssum([(A, ex(al)), (B, "t" + ex(al)), (C, ex(be))])
    cands = [(ssum([(A + B, ex(al)), (C, ex(be))]), r"$\mathcal L^{-1}\left[\frac{1}{(s-\alpha)^2}\right]=te^{\alpha t}$。$t$ が掛かる。"),
             (ssum([(A, ex(al)), (F(B, 2), "t^2" + ex(al)), (C, ex(be))]), r"$\frac{1}{(s-\alpha)^2}$ の逆変換は $te^{\alpha t}$($\frac{2}{(s-\alpha)^3}$ なら $t^2e^{\alpha t}$)。"),
             (ssum([(-C, ex(al)), (B, "t" + ex(al)), (C, ex(be))]) if -C != A else ssum([(A + 1, ex(al)), (B, "t" + ex(al)), (C, ex(be))]),
              r"$\frac{1}{s-\alpha}$ の係数は $\frac{d}{ds}\left[(s-\alpha)^2F(s)\right]_{s=\alpha}$(微分してから代入)。"),
             (ssum([(A, ex(-al)), (B, "t" + ex(-al)), (C, ex(-be))]), "指数の符号に注意。")]
    return Problem("重解を含む部分分数分解", "標準",
                   rf"$F(s)=\dfrac{{{ptex(N)}}}{{({lin(al)})^2({lin(be)})}}$ の逆ラプラス変換を求めよ。",
                   distinct(ok, cands), hints=[r"$F(s)=\frac{A}{s-\alpha}+\frac{B}{(s-\alpha)^2}+\frac{C}{s-\beta}$ とおく。",
                                               r"$B=\left[(s-\alpha)^2F\right]_{s=\alpha}$, $A=\left[\frac{d}{ds}\{(s-\alpha)^2F\}\right]_{s=\alpha}$, $C=\left[(s-\beta)F\right]_{s=\beta}$。"],
                   solution=dbl_sol(al, be, A, B, C, N), meta=dict(al=al, be=be, A=A, B=B, C=C))


def dbl_pf(al, be, A, B, C):
    return (frac_term(A, lin(al), True) + frac_term(B, f"({lin(al)})^2", A == 0) + frac_term(C, lin(be), A == 0 and B == 0)) or "0"


def dbl_sol(al, be, A, B, C, N):
    return [rf"$B=\left.\dfrac{{{ptex(N)}}}{{{lin(be)}}}\right|_{{s={al}}}={B}$, $A=\left.\dfrac{{d}}{{ds}}\dfrac{{{ptex(N)}}}{{{lin(be)}}}\right|_{{s={al}}}={A}$, "
            rf"$C=\left.\dfrac{{{ptex(N)}}}{{({lin(al)})^2}}\right|_{{s={be}}}={C}$。",
            rf"$F(s)={dbl_pf(al, be, A, B, C)}$ より $f(t)={ssum([(A, ex(al)), (B, 't' + ex(al)), (C, ex(be))])}$。"]


# ------------------------------------------------------------
# 標準:s² + ω² を含む
# ------------------------------------------------------------
def g_complex(rng):
    a, w = rng.choice([-2, -1, 0, 1, 2]), rng.choice([1, 2, 3])
    A, B, C = rng.choice([1, 2, -1, -2]), rng.choice([-2, -1, 0, 1, 2]), rng.choice([1, 2, -1, 3])
    N = padd([A, 0, A * w * w], pmul([B, C * w], [1, -a]))
    if N == [0]:
        raise Retry
    ww = "" if w == 1 else str(w)
    cos, sin = rf"\cos {ww}t", rf"\sin {ww}t"
    ok = ssum([(A, ex(a)), (B, cos), (C, sin)])
    cands = [(ssum([(A, ex(a)), (C, cos), (B, sin)]), r"$\frac{s}{s^2+\omega^2}\to\cos\omega t$, $\frac{\omega}{s^2+\omega^2}\to\sin\omega t$。"),
             (ssum([(A, ex(a)), (B, cos), (C * w, sin)]) if w != 1 else ssum([(A, ex(a)), (B, cos), (-C, sin)]),
              r"分子を $Bs+C\omega$ の形にそろえてから逆変換する。" if w != 1 else "符号を確かめよう。"),
             (ssum([(A, ex(-a)), (B, cos), (C, sin)]) if a else ssum([(-A, ""), (B, cos), (C, sin)]), "指数の符号に注意。" if a else r"隠蔽法:$A=\left[\frac{N(s)}{s^2+\omega^2}\right]_{s=0}$。"),
             (ssum([(A, ex(a)), (-B, cos), (C, sin)]) if B else ssum([(A, ex(a)), (1, cos), (C, sin)]), r"残りの分子 $N(s)-A(s^2+\omega^2)$ を $s-a$ で割って $Bs+C\omega$ を求める。")]
    return Problem("2次式 s² + ω² を含む部分分数分解", "標準",
                   rf"$F(s)=\dfrac{{{ptex(N)}}}{{({lin(a)})(s^2+{w * w})}}$ の逆ラプラス変換を求めよ。",
                   distinct(ok, cands), hints=[r"$F(s)=\frac{A}{s-a}+\frac{Bs+C\omega}{s^2+\omega^2}$ とおき, $A$ は隠蔽法で求める。",
                                               r"$N(s)-A(s^2+\omega^2)=(s-a)(Bs+C\omega)$ から $B,C$ を求める。"],
                   solution=[rf"$A=\left.\dfrac{{{ptex(N)}}}{{s^2+{w * w}}}\right|_{{s={a}}}={A}$。",
                             rf"${ptex(N)}-({ptex([A, 0, A * w * w])})=({lin(a)})({ptex([B, C * w])})$ より $B={B}$, $C={C}$。", rf"$f(t)={ok}$。"],
                   meta=dict(a=a, w=w, A=A, B=B, C=C))


# ------------------------------------------------------------
# 誘導
# ------------------------------------------------------------
def guided_inv3(rng):
    rts, cs, N = three(rng)
    return ExamItem("部分分数分解と逆ラプラス変換",
                    rf"$F(s)=\dfrac{{{ptex(N)}}}{{{den3(rts)}}}$ について, 次の問いに答えよ.",
                    [Part(r"$F(s)$ を部分分数に分解せよ.", pf3(cs, rts),
                          [rf"隠蔽法:$\left.\dfrac{{{ptex(N)}}}{{{den3([q for j, q in enumerate(rts) if j != k])}}}\right|_{{s={p}}}={fr(cs[k])}$" for k, p in enumerate(rts)]),
                     Part(r"逆ラプラス変換 $f(t)=\mathcal L^{-1}[F(s)]$ を求めよ.", ssum([(c, ex(p)) for c, p in zip(cs, rts)]),
                          [r"$\mathcal L^{-1}\left[\dfrac{1}{s-p}\right]=e^{pt}$ より", rf"$f(t)={ssum([(c, ex(p)) for c, p in zip(cs, rts)])}$。"])],
                    meta=dict(rts=rts, cs=cs))


def guided_double(rng):
    al, be, A, B, C, N = dbl_parts(rng)
    sol = dbl_sol(al, be, A, B, C, N)
    return ExamItem("重解を含む部分分数分解",
                    rf"$F(s)=\dfrac{{{ptex(N)}}}{{({lin(al)})^2({lin(be)})}}=\dfrac{{A}}{{{lin(al)}}}+\dfrac{{B}}{{({lin(al)})^2}}+\dfrac{{C}}{{{lin(be)}}}$ について, 次の問いに答えよ.",
                    [Part(r"$A$, $B$, $C$ を求めよ.", rf"A={A},\ B={B},\ C={C}", sol[:1]),
                     Part(r"逆ラプラス変換 $f(t)=\mathcal L^{-1}[F(s)]$ を求めよ.", ssum([(A, ex(al)), (B, "t" + ex(al)), (C, ex(be))]),
                          [r"$\mathcal L^{-1}\left[\dfrac{1}{(s-\alpha)^2}\right]=te^{\alpha t}$ より", sol[1]])],
                    meta=dict(al=al, be=be, A=A, B=B, C=C))


REGISTRY = [
    ProblemType("pfr_cover", "隠蔽法", "基本", g_cover),
    ProblemType("pfr_inv3", "逆ラプラス変換(3つの因数)", "標準", g_inv3, guided=guided_inv3),
    ProblemType("pfr_double", "重解を含む部分分数分解", "標準", g_double, guided=guided_double),
    ProblemType("pfr_complex", "2次式 s² + ω² を含む部分分数分解", "標準", g_complex),
]
GUIDED_NAMES = {"pfr_inv3": "隠蔽法で部分分数分解 → 逆ラプラス変換",
                "pfr_double": "重解の係数(微分を使う)→ 逆ラプラス変換"}


def tab():
    render("pfrac", REGISTRY, "ラプラス変換 第1章 4+. 部分分数分解(ヘヴィサイドの方法)")
