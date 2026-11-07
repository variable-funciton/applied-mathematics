"""③ フーリエ解析 発展 7. 関数空間 L² と Lᵖ

内積(複素数値, 区間 [−π, π]):⟨f, g⟩ = ∫ f(x) \\overline{g(x)} dx,  ‖f‖₂ = √⟨f, f⟩。
複素フーリエ係数 c_n = (1/2π) ∫ f e^{−inx} dx = ⟨f, e^{inx}⟩ / ‖e^{inx}‖²(‖e^{inx}‖² = 2π)。
Lᵖ ノルム ‖f‖_p = (∫ |f|^p)^{1/p},  ‖f‖_∞ = sup |f|。
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import C1, C2, C3, C4, GRAY, layout, show, trapz
from topics.fourier_kernels import page_refs

PI = np.pi
XS = np.linspace(-PI, PI, 8001)

# 名前: (関数, LaTeX)
FUNCS = {
    "1": (lambda x: np.ones_like(x), "1"),
    "x": (lambda x: x, "x"),
    "x²": (lambda x: x ** 2, "x^2"),
    "|x|": (lambda x: np.abs(x), "|x|"),
    "sgn x(方形波)": (lambda x: np.sign(x), r"\mathrm{sgn}\,x"),
    "cos x": (lambda x: np.cos(x), r"\cos x"),
    "sin x": (lambda x: np.sin(x), r"\sin x"),
    "sin 2x": (lambda x: np.sin(2 * x), r"\sin 2x"),
    "e^{ix}": (lambda x: np.exp(1j * x), "e^{ix}"),
    "e^{−ix}": (lambda x: np.exp(-1j * x), "e^{-ix}"),
    "e^{2ix}": (lambda x: np.exp(2j * x), "e^{2ix}"),
    "i e^{ix}": (lambda x: 1j * np.exp(1j * x), "ie^{ix}"),
}
APPROX = {"方形波 sgn x": lambda x: np.sign(x), "のこぎり波 x": lambda x: x, "|x|": np.abs, "x²": lambda x: x ** 2}


def ip(f, g, x=XS):
    """⟨f, g⟩ = ∫ f \\bar g dx(台形公式)"""
    return complex(trapz(f * np.conj(g), x))


def nrm(f, x=XS):
    return float(np.sqrt(ip(f, f, x).real))


def coef(f, n, x=XS):
    """複素フーリエ係数 c_n = (1/2π) ∫ f e^{−inx} dx"""
    return complex(trapz(f * np.exp(-1j * n * x), x)) / (2 * PI)


def partial(f, N, x=XS):
    return sum(coef(f, n, x) * np.exp(1j * n * x) for n in range(-N, N + 1))


def ctxt(z, d=4):
    z = complex(z)
    re, im = round(z.real, d), round(z.imag, d)
    re, im = (0.0 if abs(re) < 10 ** -d else re), (0.0 if abs(im) < 10 ** -d else im)
    if im == 0:
        return f"{re:g}"
    if re == 0:
        return f"{im:g}i"
    return f"{re:g} {'+' if im > 0 else '−'} {abs(im):g}i"


def unit_ball(p, n=721):
    th = np.linspace(0, 2 * PI, n)
    c, s = np.cos(th), np.sin(th)
    if p == np.inf:
        r = 1 / np.maximum(np.abs(c), np.abs(s))
        return r * c, r * s
    r = (np.abs(c) ** p + np.abs(s) ** p) ** (-1 / p)
    return r * c, r * s


def power_norm(a, p):
    """‖x^{−a}‖_{Lᵖ(0,1)} = (1 − ap)^{−1/p}(ap < 1), それ以外は ∞"""
    return (1 - a * p) ** (-1 / p) if a * p < 1 else np.inf


def parallelogram(p):
    """[0, 1] で f = χ_[0,1/2], g = χ_[1/2,1] のとき (左辺, 右辺) = (‖f+g‖² + ‖f−g‖², 2‖f‖² + 2‖g‖²)"""
    nf = 1.0 if p == np.inf else 0.5 ** (1 / p)
    return 2.0, 4 * nf ** 2


def tab_lp():
    st.info("「ベクトルと行列」で学んだ**内積・大きさ(ノルム)・直交・正射影**を, ベクトルから関数に広げる。"
            "フーリエ級数は「関数を正規直交系で成分に分解すること」と読み直せる。"
            "さらに $L^2$ を $L^p$ に一般化すると, 発展 8. のマルチプライヤーやリースポテンシャルの話の舞台になる。")

    # ------------------------------------------------------------
    st.subheader("1. ベクトルから関数へ:内積とノルム")
    st.markdown(r"""
| | ベクトル $\boldsymbol a,\boldsymbol b\in\mathbb C^n$ | 関数 $f,g$(区間 $[-\pi,\pi]$) |
|---|---|---|
| 内積 | $(\boldsymbol a,\boldsymbol b)=\sum_k a_k\overline{b_k}$ | $\langle f,g\rangle=\displaystyle\int_{-\pi}^{\pi}f(x)\overline{g(x)}\,dx$ |
| 大きさ(ノルム) | $\lvert\boldsymbol a\rvert=\sqrt{\sum_k\lvert a_k\rvert^2}$ | $\lVert f\rVert_2=\sqrt{\langle f,f\rangle}=\left(\displaystyle\int_{-\pi}^{\pi}\lvert f(x)\rvert^2dx\right)^{1/2}$ |
| 直交 | $(\boldsymbol a,\boldsymbol b)=0$ | $\langle f,g\rangle=0$ |
| 成分 | $a_k$($k=1,\dots,n$) | フーリエ係数 $c_n$($n\in\mathbb Z$, 無限個) |
""")
    st.markdown("足し算の $\\sum_k$ が積分 $\\int dx$ に変わっただけである。"
                "$\\|f\\|_2<\\infty$ となる関数全体を $L^2$ と書き, $\\|f-g\\|_2$ を2つの関数の「距離」(2乗平均誤差)とみる。")
    st.markdown("**なぜ共役 $\\overline{g}$ をつけるのか**:複素数値の関数では, 共役をつけないと "
                "$\\int_{-\\pi}^{\\pi}e^{ix}e^{ix}dx=0$ となり, 0 でない関数の「大きさ」が 0 になってしまう。"
                "共役をつけると $\\langle f,f\\rangle=\\int|f|^2dx\\ge0$ となる。そのかわり $\\langle g,f\\rangle=\\overline{\\langle f,g\\rangle}$, "
                "$\\langle f,cg\\rangle=\\bar c\\langle f,g\\rangle$ となることに注意する。")
    st.latex(r"|\langle f,g\rangle|\le\|f\|_2\|g\|_2\quad(\text{コーシー・シュワルツ}),\qquad "
             r"\langle f,g\rangle=0\ \Rightarrow\ \|f+g\|_2^2=\|f\|_2^2+\|g\|_2^2\quad(\text{三平方の定理})")

    c = st.columns(2)
    fn = c[0].selectbox("f", list(FUNCS), index=1, key="lp_f")
    gn = c[1].selectbox("g", list(FUNCS), index=6, key="lp_g")
    f, ft = FUNCS[fn][0](XS), FUNCS[fn][1]
    g, gt = FUNCS[gn][0](XS), FUNCS[gn][1]
    v, nf, ng = ip(f, g), nrm(f), nrm(g)
    mm = st.columns(4)
    mm[0].metric("⟨f, g⟩", ctxt(v))
    mm[1].metric("‖f‖₂", f"{nf:.4f}")
    mm[2].metric("‖g‖₂", f"{ng:.4f}")
    cos = abs(v) / (nf * ng)
    mm[3].metric("|⟨f, g⟩| / (‖f‖₂‖g‖₂)", f"{cos:.4f}")
    fig = go.Figure()
    for y, nm, col, dash in ((f, f"Re f = Re {fn}", C1, None), (g, f"Re g = Re {gn}", C2, None)):
        fig.add_trace(go.Scatter(x=XS, y=np.real(y), mode="lines", line=dict(color=col, width=2, dash=dash), name=nm))
    fig.add_trace(go.Scatter(x=XS, y=np.real(f * np.conj(g)), mode="lines", line=dict(color=C3, width=1.5, dash="dot"),
                             fill="tozeroy", name="Re f ḡ(面積が Re⟨f, g⟩)"))
    layout(fig, title=f"⟨{fn}, {gn}⟩ = ∫ f ḡ dx", xlab="x", height=300)
    show(fig)
    if abs(v) < 1e-6:
        st.success(f"${ft}$ と ${gt}$ は直交する。三平方の定理:$\\|f+g\\|_2^2={nrm(f + g) ** 2:.4f}$, "
                   f"$\\|f\\|_2^2+\\|g\\|_2^2={nf ** 2 + ng ** 2:.4f}$。")
    elif cos > 1 - 1e-9:
        st.info("$|\\langle f,g\\rangle|=\\|f\\|_2\\|g\\|_2$:一方が他方の定数倍のとき(平行)だけ, コーシー・シュワルツの不等式で等号が成り立つ。")
    st.caption("実数値の関数では $\\cos\\theta=\\dfrac{\\langle f,g\\rangle}{\\|f\\|_2\\|g\\|_2}$ を2つの関数の「なす角」とみなせる。"
               "コーシー・シュワルツの不等式は, この値が −1 から 1 の間にあることを保証する。数値は台形公式による。")

    # ------------------------------------------------------------
    st.subheader("2. 正規直交系とフーリエ級数:係数は正射影")
    st.latex(r"\langle e^{imx},e^{inx}\rangle=\int_{-\pi}^{\pi}e^{i(m-n)x}dx=\begin{cases}2\pi&(m=n)\\0&(m\ne n)\end{cases}")
    st.markdown("つまり $\\{e^{inx}\\}$ は互いに直交し(大きさ $\\sqrt{2\\pi}$), $\\left\\{\\dfrac{e^{inx}}{\\sqrt{2\\pi}}\\right\\}$ は**正規直交系**になる。"
                "ベクトルの成分が「基底との内積」で求まるのと同じく, 複素フーリエ係数は")
    st.latex(r"c_n=\frac{\langle f,e^{inx}\rangle}{\langle e^{inx},e^{inx}\rangle}=\frac{1}{2\pi}\int_{-\pi}^{\pi}f(x)e^{-inx}dx")
    st.markdown("となる。部分和 $S_Nf=\\sum_{|n|\\le N}c_ne^{inx}$ は, $e^{-iNx},\\dots,e^{iNx}$ の張る空間への**正射影**であり, 次が成り立つ。")
    st.markdown(
        "- **最良近似**:どんな係数 $a_n$ に対しても $\\left\\|f-\\sum_{|n|\\le N}a_ne^{inx}\\right\\|_2\\ge\\|f-S_Nf\\|_2$。"
        "等号は $a_n=c_n$ のときだけ(フーリエ係数は2乗平均誤差を最小にする)。\n"
        "- **三平方の定理**:$f-S_Nf$ は $S_Nf$ と直交するので $\\|f\\|_2^2=\\|S_Nf\\|_2^2+\\|f-S_Nf\\|_2^2$。"
        "$\\|S_Nf\\|_2^2=2\\pi\\sum_{|n|\\le N}|c_n|^2$ より**ベッセルの不等式** $2\\pi\\sum|c_n|^2\\le\\|f\\|_2^2$。\n"
        "- **パーセバルの等式**:$f\\in L^2$ なら $\\|f-S_Nf\\|_2\\to0$($L^2$ 収束)となり, "
        "$\\dfrac{1}{2\\pi}\\displaystyle\\int_{-\\pi}^{\\pi}|f|^2dx=\\sum_{n=-\\infty}^{\\infty}|c_n|^2$。"
        "これは「無限次元の三平方の定理」である(フーリエ変換では第2章 3. のプランシュレルの等式)。")
    c = st.columns(3)
    an = c[0].selectbox("近似する関数 f", list(APPROX), key="lp_af")
    N = c[1].slider("部分和の次数 N", 0, 20, 3, key="lp_N")
    dl = c[2].slider("c₁ をずらす量 δ", -1.0, 1.0, 0.0, 0.05, key="lp_dl")
    fa = APPROX[an](XS)
    SN = partial(fa, N)
    tN = SN + (dl * np.exp(1j * XS) if N >= 1 else 0)
    fig = go.Figure(go.Scatter(x=XS, y=fa, mode="lines", line=dict(color=GRAY, width=2.5), name="f"))
    fig.add_trace(go.Scatter(x=XS, y=SN.real, mode="lines", line=dict(color=C2, width=2), name=f"S_{N} f"))
    if N >= 1 and dl != 0:
        fig.add_trace(go.Scatter(x=XS, y=tN.real, mode="lines", line=dict(color=C4, width=1.5, dash="dash"), name="c₁ を c₁ + δ にしたもの(実部)"))
    layout(fig, title="フーリエ級数の部分和 = 正射影", xlab="x", height=300)
    show(fig)
    mm = st.columns(3)
    e0 = nrm(fa - SN) ** 2
    mm[0].metric("‖f − S_N f‖₂²", f"{e0:.4f}")
    if N >= 1:
        e1 = nrm(fa - tN) ** 2
        mm[1].metric("c₁ をずらしたときの誤差²", f"{e1:.4f}", delta=f"{e1 - e0:+.4f}(= 2π|δ|² = {2 * PI * dl ** 2:.4f})", delta_color="inverse")
    else:
        mm[1].metric("c₁ をずらしたときの誤差²", "N ≥ 1 で表示")
    mm[2].metric("2π Σ|c_n|² / ‖f‖₂²", f"{nrm(SN) ** 2 / nrm(fa) ** 2:.4f}")
    st.caption("係数をずらすと, 誤差の2乗は $\\|f-S_Nf\\|_2^2+2\\pi|\\delta|^2$ に増える(三平方の定理)。"
               "下の図:$N$ を増やすと $2\\pi\\sum_{|n|\\le N}|c_n|^2$ は $\\|f\\|_2^2$ に下から近づく(ベッセル → パーセバル)。")
    Ns = np.arange(0, 31)
    cs = np.array([abs(coef(fa, n)) ** 2 for n in range(-30, 31)])
    part = [2 * PI * cs[30 - k:31 + k].sum() for k in Ns]
    fig = go.Figure(go.Bar(x=Ns, y=part, marker_color=C1, name="2π Σ_{|n|≤N} |c_n|²"))
    fig.add_trace(go.Scatter(x=[0, 30], y=[nrm(fa) ** 2] * 2, mode="lines", line=dict(color=C2, dash="dash"), name="‖f‖₂²"))
    layout(fig, title="ベッセルの不等式とパーセバルの等式", xlab="N", height=280)
    show(fig)

    # ------------------------------------------------------------
    st.subheader("3. L² から Lᵖ へ")
    st.markdown("2乗を $p$ 乗に変えると, 大きさの測り方が1つ増える($1\\le p<\\infty$):")
    st.latex(r"\|f\|_p=\left(\int|f(x)|^pdx\right)^{1/p},\qquad \|f\|_\infty=\sup_x|f(x)|")
    st.markdown("$\\|f\\|_p<\\infty$ となる関数全体を $L^p$ と書く。"
                "($\\|f\\|_\\infty$ は正確には「測度 0 の集合を除いた上限」(本質的上限)だが, ここでは最大値と思ってよい。)")
    st.markdown(r"""
| ノルム | 工学的な意味(信号 $f(t)$) |
|---|---|
| $\lVert f\rVert_1=\int\lvert f\rvert\,dt$ | 総量(絶対値の面積) |
| $\lVert f\rVert_2^2=\int\lvert f\rvert^2dt$ | エネルギー(電力の積分) |
| $\lVert f\rVert_\infty=\sup\lvert f\rvert$ | ピーク値(最大振幅) |
""")
    st.markdown("**単位球の形**:平面のベクトル $(x,y)$ で $\\|(x,y)\\|_p=(|x|^p+|y|^p)^{1/p}=1$ をみたす点を描くと, "
                "$p$ によって「円」の形が変わる。")
    c = st.columns([1, 2])
    pinf = c[0].checkbox("p = ∞", key="lp_pinf")
    p = np.inf if pinf else c[0].slider("p", 0.5, 10.0, 1.5, 0.1, key="lp_p")
    with c[1]:
        fig = go.Figure()
        for pp, nm in ((1, "p = 1"), (2, "p = 2"), (np.inf, "p = ∞")):
            x, y = unit_ball(pp)
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=GRAY, width=1, dash="dot"), name=nm))
        x, y = unit_ball(p)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=C2, width=2.5), name=f"p = {'∞' if p == np.inf else f'{p:g}'}"))
        layout(fig, title="|x|ᵖ + |y|ᵖ = 1", xlab="x", height=360, equal=True)
        show(fig)
    if p != np.inf and p < 1:
        st.warning("$p<1$ では単位球がへこむ(凸でない)。このとき三角不等式 $\\|a+b\\|\\le\\|a\\|+\\|b\\|$ が成り立たず, "
                   "ノルムにならない:たとえば $a=(1,0)$, $b=(0,1)$ で $\\|a+b\\|_p=2^{1/p}>2=\\|a\\|_p+\\|b\\|_p$。")
    st.markdown("$1\\le p\\le\\infty$ では次の不等式が成り立つ($p'$ は $\\frac1p+\\frac1{p'}=1$ をみたす数で, $p$ の**共役指数**という):")
    st.latex(r"\int|fg|\,dx\le\|f\|_p\|g\|_{p'}\quad(\text{ヘルダー}),\qquad \|f+g\|_p\le\|f\|_p+\|g\|_p\quad(\text{ミンコフスキー})")
    st.caption("ヘルダーの不等式で $p=p'=2$ とするとコーシー・シュワルツの不等式になる。ミンコフスキーの不等式は $L^p$ での三角不等式である。")

    st.markdown("**包含関係:区間の長さが有限なら「$p$ が大きいほど厳しい」**。長さ 1 の区間では "
                "$\\|f\\|_1\\le\\|f\\|_2\\le\\|f\\|_\\infty$(一般に $p<q$ なら $\\|f\\|_p\\le\\|f\\|_q$)なので $L^\\infty\\subset L^2\\subset L^1$。")
    c = st.columns([1, 2])
    a = c[0].slider("f(x) = x^{−a} の a", 0.0, 0.9, 0.4, 0.05, key="lp_a")
    ps = np.linspace(1, 20, 600)
    with c[1]:
        fig = go.Figure()
        y1 = np.array([power_norm(a, q) for q in ps])
        fig.add_trace(go.Scatter(x=ps, y=np.where(np.isfinite(y1), y1, np.nan), mode="lines", line=dict(color=C2, width=2), name=f"x^(−{a:g})"))
        xg = np.linspace(0, 1, 20001)
        y2 = [float(trapz(np.sin(PI * xg) ** q, xg) ** (1 / q)) for q in ps[::10]]
        fig.add_trace(go.Scatter(x=ps[::10], y=y2, mode="lines", line=dict(color=C1, width=2), name="sin πx"))
        fig.add_trace(go.Scatter(x=[1, 20], y=[1, 1], mode="lines", line=dict(color=C1, dash="dot", width=1), name="max sin πx = 1"))
        if a > 0:
            fig.add_vline(x=1 / a, line=dict(color=C2, dash="dash"))
        layout(fig, title="区間 (0, 1) での ‖f‖_p(p の関数として)", xlab="p", height=300, yrange=(0, 4))
        show(fig)
    st.caption(f"$\\|x^{{-a}}\\|_p=(1-ap)^{{-1/p}}$ は $p$ とともに増え, $p\\to1/a$ で無限大になる($a={a:g}$ では "
               + (f"$p<{1 / a:.3g}$ で $L^p$ に入る)。" if a > 0 else "すべての $p$ で値は 1)。")
               + "有界な $\\sin\\pi x$ では $\\|f\\|_p\\to\\max|f|=1=\\|f\\|_\\infty$($p\\to\\infty$)。")

    st.markdown("**実数全体 $\\mathbb R$ ではどちら向きの包含も成り立たない**。原点の近くの特異性と, 遠くでの減衰の2つが効く:")
    st.latex(r"x^{-a}\chi_{(0,1)}\in L^p\iff a<\frac1p,\qquad x^{-a}\chi_{(1,\infty)}\in L^p\iff a>\frac1p")
    c = st.columns([1, 2])
    pp = c[0].slider("p", 1.0, 10.0, 2.0, 0.1, key="lp_pp")
    aa = c[0].slider("a", 0.05, 1.5, 0.4, 0.05, key="lp_aa")
    loc, tail = aa * pp < 1, aa * pp > 1
    c[0].markdown(f"- $x^{{-a}}\\chi_{{(0,1)}}$:{'$L^p$ に入る' if loc else '$L^p$ に入らない'}\n"
                  f"- $x^{{-a}}\\chi_{{(1,\\infty)}}$:{'$L^p$ に入る' if tail else '$L^p$ に入らない'}")
    with c[1]:
        u = np.linspace(0, 1, 201)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.r_[u, 1, 0], y=np.r_[u, 0, 0], fill="toself", mode="none", fillcolor="rgba(31,119,180,0.25)",
                                 name="x^{−a}χ_(0,1) ∈ Lᵖ(a < 1/p)"))
        fig.add_trace(go.Scatter(x=np.r_[u, 1, 0], y=np.r_[u, 1.5, 1.5], fill="toself", mode="none", fillcolor="rgba(214,39,40,0.2)",
                                 name="x^{−a}χ_(1,∞) ∈ Lᵖ(a > 1/p)"))
        fig.add_trace(go.Scatter(x=u, y=u, mode="lines", line=dict(color=GRAY), name="a = 1/p"))
        fig.add_trace(go.Scatter(x=[1 / pp], y=[aa], mode="markers", marker=dict(color="black", size=11), name="選んだ (1/p, a)"))
        layout(fig, title="横軸 1/p, 縦軸 a の図", xlab="1/p", height=320, xrange=(0, 1), yrange=(0, 1.5))
        show(fig)
    st.caption("たとえば $x^{-0.4}\\chi_{(0,1)}$ は $L^2$ に入るが $L^3$ に入らず, $x^{-0.4}\\chi_{(1,\\infty)}$ は $L^3$ に入るが $L^2$ に入らない。"
               "調和解析では, このように横軸に $1/p$ をとる図がよく使われる(ハーディ・リトルウッド・ソボレフの条件 $\\frac1q=\\frac1p-\\alpha$ もこの図で直線になる)。")

    # ------------------------------------------------------------
    st.subheader("4. 内積をもつのは p = 2 だけ:中線定理")
    st.markdown("内積から決まるノルムでは, 平行四辺形の2本の対角線について**中線定理**(平行四辺形の法則)が成り立つ:")
    st.latex(r"\|f+g\|^2+\|f-g\|^2=2\|f\|^2+2\|g\|^2")
    st.markdown("逆に, この等式をみたすノルムは必ず内積から作られる(ジョルダン・フォン・ノイマンの定理, 1935)。"
                "区間 $[0,1]$ で $f=\\chi_{[0,1/2]}$, $g=\\chi_{[1/2,1]}$ とすると, $|f\\pm g|=1$ より $\\|f\\pm g\\|_p=1$, "
                "$\\|f\\|_p=\\|g\\|_p=2^{-1/p}$ なので")
    st.latex(r"(\text{左辺})=2,\qquad(\text{右辺})=4\cdot2^{-2/p}=2^{2-2/p}")
    pq = np.linspace(1, 10, 400)
    fig = go.Figure(go.Scatter(x=pq, y=[parallelogram(q)[0] for q in pq], mode="lines", line=dict(color=C1, width=2), name="左辺 ‖f+g‖² + ‖f−g‖²"))
    fig.add_trace(go.Scatter(x=pq, y=[parallelogram(q)[1] for q in pq], mode="lines", line=dict(color=C2, width=2), name="右辺 2‖f‖² + 2‖g‖²"))
    fig.add_trace(go.Scatter(x=[2], y=[2], mode="markers", marker=dict(color="black", size=10), name="p = 2 で一致"))
    layout(fig, title="中線定理が成り立つのは p = 2 のときだけ", xlab="p", height=300)
    show(fig)
    st.markdown("両辺が等しいのは $2-\\frac2p=1$, すなわち $p=2$ のときだけである($p=\\infty$ でも左辺 2, 右辺 4)。"
                "したがって $p\\ne2$ の $L^p$ には, ノルムに合う内積がない。角度・直交・正射影が使えるのは $L^2$ だけで, "
                "フーリエ解析で $L^2$ が特別に扱いやすい(プランシュレルの等式)理由はここにある。"
                "$p\\ne2$ の $L^p$ での議論は難しくなるが, その分だけ豊かな理論がある(発展 8. のヒルベルト変換・マルチプライヤー定理・ハーディ・リトルウッド・ソボレフの不等式)。")

    # ------------------------------------------------------------
    with st.expander("コラム:完備性とルベーグ積分(少しだけ)"):
        st.markdown("有理数の列が有理数に収束するとは限らない($\\sqrt2$ に近づく列)ように, リーマン積分できる関数の列が "
                    "$L^2$ ノルムで「収束しそうなのに, 極限がリーマン積分できない」ことがある。"
                    "ルベーグ積分を使うと $L^p$ は**完備**(収束しそうな列は必ず極限をもつ)になる(F. リース, 1910 ほか)。"
                    "そのおかげで, $\\sum|c_n|^2<\\infty$ をみたす数列 $(c_n)$ には必ずそれをフーリエ係数とする $L^2$ の関数がある"
                    "(リース・フィッシャーの定理)。完備な内積空間を**ヒルベルト空間**という。")
    with st.expander("コラム:Lᵖ のさらに先(オーリッチ空間・モレー空間)"):
        st.markdown("- **オーリッチ空間**:$|f|^p$ の $t^p$ を, もっと一般の増加関数 $\\Phi(t)$(たとえば $t\\log(e+t)$, $e^t-1$)に置き換えて"
                    "大きさを測る空間(バーンバウム・オーリッチ, 1931)。$p=1$ のように $L^p$ だけでは足りない境目で使われる"
                    "(例:$f\\in L\\log L$ ならヒルベルト変換 $Hf$ は区間上で可積分になる(ジグムント))。\n"
                    "- **モレー空間**:すべての区間 $I$ で $\\dfrac{1}{|I|^\\lambda}\\displaystyle\\int_I|f|^pdx$ が一様に有界な関数の空間"
                    "($0\\le\\lambda\\le1$。$\\lambda=0$ で $L^p$, $\\lambda=1$ で $L^\\infty$)。局所的な大きさを細かく測る空間で, "
                    "偏微分方程式の解のなめらかさの研究から生まれた(モレー, 1938)。リースポテンシャルのモレー空間での有界性(アダムスの定理)は発展 8. を参照。\n\n"
                    "これらの空間での作用素(極大作用素・特異積分・分数冪積分など)の有界性は, 現在も活発に研究されている。")

    with st.expander("練習問題"):
        st.markdown("**問1** 区間 $[-\\pi,\\pi]$ で $\\langle e^{2ix},e^{2ix}\\rangle$, $\\langle e^{ix},e^{2ix}\\rangle$, $\\langle 1+e^{ix},\\,i e^{ix}\\rangle$ を求めよ。")
        st.markdown("**問2** 区間 $[0,1]$ で $f(x)=x$ の $\\|f\\|_1$, $\\|f\\|_2$, $\\|f\\|_\\infty$ を求め, $\\|f\\|_1\\le\\|f\\|_2\\le\\|f\\|_\\infty$ を確かめよ。")
        st.markdown("**問3** $f(x)=x^{-1/3}$($0<x<1$)が $L^p(0,1)$ に入る $p\\ (\\ge1)$ の範囲を求めよ。")
        if st.toggle("解答・解説を表示", key="pr_lp"):
            st.markdown("**答1** $2\\pi$, $0$。$\\langle 1+e^{ix},ie^{ix}\\rangle=\\overline{i}\\,\\langle 1+e^{ix},e^{ix}\\rangle=-i(0+2\\pi)=-2\\pi i$"
                        "(2番目の関数の定数は共役をとって外に出る)。")
            st.markdown("**答2** $\\|f\\|_1=\\int_0^1x\\,dx=\\frac12$, $\\|f\\|_2=\\left(\\int_0^1x^2dx\\right)^{1/2}=\\frac{1}{\\sqrt3}\\approx0.577$, "
                        "$\\|f\\|_\\infty=1$。$\\frac12<\\frac1{\\sqrt3}<1$。")
            st.markdown("**答3** $\\int_0^1x^{-p/3}dx<\\infty\\iff\\frac p3<1$ より $1\\le p<3$。")

    page_refs([("stein2007", "第3章:内積空間とフーリエ級数の2乗平均収束(パーセバルの等式)。"),
               ("steinreal2005", "ルベーグ積分・Lᵖ 空間・ヒルベルト空間(完備性, リース・フィッシャーの定理)。"),
               ("riesz1910", "Lᵖ 空間を導入した論文。"),
               ("jordanvonneumann1935", "中線定理による内積空間の特徴づけ。"),
               ("birnbaumorlicz1931", "オーリッチ空間の始まり。"),
               ("zygmund2002", "L log L とヒルベルト変換(共役関数)の可積分性。"),
               ("morrey1938", "モレー空間の始まり(楕円型偏微分方程式の解の正則性)。"),
               ("adams1975", "モレー空間でのリースポテンシャルの有界性。")])
