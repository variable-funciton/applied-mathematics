"""③ フーリエ解析 第1章:良い核と総和法
  5. 良い核と近似単位元(ディリクレ核とフェイェール核)
  6. チェザロ平均と総和法
  7. ポアソン核とアーベル総和法

合成積は第1章4節と同じ (f*g)(x) = (1/2π)∫_{−π}^{π} f(x−t) g(t) dt。
"""
import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import C1, C2, C3, C4, GRAY, layout, show
from topics.fourier import wrap
from topics.fourier3 import XG, cn_grid, per_conv
import importlib

PI = np.pi


def ref_line(key):
    """参考文献の1行。呼ぶたびに今の topics.references を見る
    (Streamlit は変更されたファイルだけを読み直すので, 読み込み時に作った表は古いまま残ることがある)"""
    R = importlib.import_module("topics.references")
    for rs in R.REFS.values():
        for r in rs:
            if r["key"] == key:
                return R.fmt(r)
    return f"(文献 {key} が見つかりません)"


def page_refs(items):
    """[(文献のキー, 該当箇所の説明), ...]"""
    with st.expander("参考文献"):
        st.markdown("\n".join(f"- {ref_line(k)}  \n  {where}" for k, where in items))
        st.caption("書誌情報は「📚 参考文献」のページにまとめてある(BibTeX もダウンロードできる)。")


# ------------------------------------------------------------
# 核
# ------------------------------------------------------------
def dirichlet(N, x=XG):
    s = np.sin(x / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.sin((N + 0.5) * x) / s
    return np.where(np.abs(s) < 1e-12, 2 * N + 1.0, v)


def fejer(N, x=XG):
    s = np.sin(x / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = (np.sin((N + 1) * x / 2) / s) ** 2 / (N + 1)
    return np.where(np.abs(s) < 1e-12, N + 1.0, v)


def poisson(r, x=XG):
    return (1 - r * r) / (1 - 2 * r * np.cos(x) + r * r)


def heat(t, x=XG, nmax=600):
    n = np.arange(1, nmax + 1)
    return 1 + 2 * (np.exp(-n ** 2 * t)[:, None] * np.cos(np.outer(n, x))).sum(axis=0)


def box(d, x=XG):
    """高さ π/δ, 幅 2δ(格子の上で平均がちょうど 1 になるように高さをそろえる)"""
    v = (np.abs(x) < d).astype(float)
    return v / np.mean(v)


KERNELS = {
    "ディリクレ核 D_N": (dirichlet, lambda k: k, "N", r"D_N(x)=\sum_{n=-N}^{N}e^{inx}=\frac{\sin\left(N+\frac12\right)x}{\sin\frac x2}"),
    "フェイェール核 F_N": (fejer, lambda k: k, "N",
                      r"F_N(x)=\sum_{n=-N}^{N}\left(1-\frac{|n|}{N+1}\right)e^{inx}=\frac{1}{N+1}\left(\frac{\sin\frac{(N+1)x}{2}}{\sin\frac x2}\right)^2"),
    "ポアソン核 P_r": (poisson, lambda k: 1 - 1 / (k + 1), "r", r"P_r(x)=\sum_{n=-\infty}^{\infty}r^{|n|}e^{inx}=\frac{1-r^2}{1-2r\cos x+r^2}"),
    "熱核 H_t": (heat, lambda k: 1 / k ** 2, "t", r"H_t(x)=\sum_{n=-\infty}^{\infty}e^{-n^2t}e^{inx}"),
    "箱型の核 K_δ": (box, lambda k: PI / (k + 1), "δ", r"K_\delta(x)=\begin{cases}\pi/\delta&(|x|<\delta)\\0&(\delta\le|x|\le\pi)\end{cases}"),
}


def mean(v):
    """(1/2π)∫_{−π}^{π} v dx(格子の平均)"""
    return float(np.mean(v))


def kernel_stats(v, d):
    return mean(v), mean(np.abs(v)), mean(np.where(np.abs(XG) >= d, np.abs(v), 0.0))


@st.cache_data
def stats_curve(name, d, kmax=60):
    fn, par, _, _ = KERNELS[name]
    return np.array([kernel_stats(fn(par(k)), d) for k in range(1, kmax + 1)])


TESTS = {"三角波 |x|(連続)": lambda x: np.abs(wrap(x)), "矩形波(x = 0, ±π で不連続)": lambda x: np.sign(wrap(x)),
         "cos x + 0.5 sin 3x(なめらか)": lambda x: np.cos(x) + 0.5 * np.sin(3 * x)}


def tab_good_kernels():
    st.info("部分和 $S_N=f*D_N$, フェイェール平均 $\\sigma_N=f*F_N$ のように, フーリエ級数の近似は「ある核との合成積」で表せる。"
            "核が次の3つの性質をもてば, 合成積は $f$ に収束する。このような核の列を**良い核**(近似単位元)という。")
    st.subheader("良い核の定義")
    st.markdown("周期 $2\\pi$ の関数の列 $\\{K_n\\}$ が次の (i)〜(iii) をみたすとき, **良い核**(good kernels)という。")
    st.latex(r"\text{(i)}\ \ \frac{1}{2\pi}\int_{-\pi}^{\pi}K_n(x)\,dx=1\qquad"
             r"\text{(ii)}\ \ \frac{1}{2\pi}\int_{-\pi}^{\pi}|K_n(x)|\,dx\le M\ \ (n\text{ によらない定数 }M)")
    st.latex(r"\text{(iii)}\ \ \text{どんな }\delta>0\text{ に対しても}\quad \int_{\delta\le|x|\le\pi}|K_n(x)|\,dx\ \xrightarrow{n\to\infty}\ 0")
    st.markdown("(i) は「平均すると 1」, (ii) は「大きさが一様におさえられる」, (iii) は「重みが $x=0$ のまわりに集中していく」という意味。"
                "$K_n\\ge0$ なら (ii) は (i) から自動的に成り立つ。")
    st.markdown("**定理**:$\\{K_n\\}$ が良い核で $f$ が(有界で)積分可能ならば, $f$ が連続な点 $x$ で")
    st.latex(r"(f*K_n)(x)=\frac{1}{2\pi}\int_{-\pi}^{\pi}f(x-t)K_n(t)\,dt\ \xrightarrow{n\to\infty}\ f(x)")
    st.markdown("が成り立つ。$f$ が全体で連続なら, 収束は一様。")
    with st.expander("定理の証明"):
        st.markdown("(i) より $f(x)=\\dfrac{1}{2\\pi}\\int f(x)K_n(t)\\,dt$ と書けるので")
        st.latex(r"(f*K_n)(x)-f(x)=\frac{1}{2\pi}\int_{-\pi}^{\pi}\{f(x-t)-f(x)\}K_n(t)\,dt")
        st.markdown("$\\varepsilon>0$ に対し, 連続性から $|t|<\\delta$ で $|f(x-t)-f(x)|<\\varepsilon$ となる $\\delta$ をとる。$|f|\\le B$ とすると")
        st.latex(r"|(f*K_n)(x)-f(x)|\le\frac{\varepsilon}{2\pi}\int_{|t|<\delta}|K_n|\,dt+\frac{2B}{2\pi}\int_{\delta\le|t|\le\pi}|K_n|\,dt"
                 r"\le\varepsilon M+\frac{B}{\pi}\int_{\delta\le|t|\le\pi}|K_n|\,dt")
        st.markdown("(iii) より第2項は $n\\to\\infty$ で $0$ なので, $\\limsup|(f*K_n)(x)-f(x)|\\le\\varepsilon M$。$\\varepsilon$ は任意なので収束する。"
                    "$f$ が連続なら $\\delta$ を $x$ によらずにとれる(一様連続)ので, 一様収束になる。")
        st.caption("「近似単位元」という名前は, 合成積 $f*K_n\\to f$ が「$f*\\delta=f$ となるデルタ関数 $\\delta$(合成積の単位元)」を近似していることからきている。")

    st.subheader("核を見くらべる")
    c = st.columns(3)
    name = c[0].selectbox("核", list(KERNELS), index=1, key="gk_k")
    k = c[1].slider("番号 n(大きいほど集中)", 1, 60, 8, key="gk_n")
    d = c[2].slider("(iii) を調べる δ", 0.1, 1.5, 0.5, 0.05, key="gk_d")
    fn, par, pname, tex = KERNELS[name]
    p = par(k)
    st.latex(tex)
    v = fn(p)
    i1, i2, i3 = kernel_stats(v, d)
    cA, cB = st.columns(2)
    with cA:
        fig = go.Figure(go.Scatter(x=XG, y=v, mode="lines", line=dict(color=C1, width=2), name=name))
        fig.add_vrect(x0=-d, x1=d, fillcolor="rgba(44,160,44,0.10)", line_width=0)
        fig.add_hline(y=0, line=dict(color=GRAY, width=1))
        layout(fig, title=f"{name}({pname} = {p:.4g})", xlab="x", height=340, legend=False)
        show(fig)
        m = st.columns(3)
        m[0].metric("(i) 平均", f"{i1:.4f}")
        m[1].metric("(ii) |K| の平均", f"{i2:.4f}")
        m[2].metric(f"(iii) |x| ≥ {d:g} の部分", f"{i3:.4f}")
    with cB:
        S = stats_curve(name, d)
        ks = np.arange(1, 61)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ks, y=S[:, 1], mode="lines+markers", marker=dict(size=4), line=dict(color=C2), name="(ii) |K| の平均"))
        fig.add_trace(go.Scatter(x=ks, y=S[:, 2], mode="lines+markers", marker=dict(size=4), line=dict(color=C3), name="(iii) 外側の部分"))
        if name.startswith("ディリクレ"):
            fig.add_trace(go.Scatter(x=ks, y=4 / PI ** 2 * np.log(ks) + 1.27, mode="lines", line=dict(color=GRAY, dash="dot"),
                                     name="(4/π²) log N + 1.27"))
        fig.add_vline(x=k, line=dict(color=GRAY, dash="dot"))
        layout(fig, title="n を大きくしたとき", xlab="n", height=340)
        show(fig)
    if name.startswith("ディリクレ"):
        st.warning("ディリクレ核は良い核ではない。(i) はみたすが, (ii) の値 $L_N=\\frac{1}{2\\pi}\\int|D_N|$(ルベーグ定数)は "
                   "$\\frac{4}{\\pi^2}\\log N$ のように限りなく大きくなり, (iii) も 0 に近づかない(外側で正負に振動し続ける)。"
                   "これが部分和 $S_N$ の収束のむずかしさ(ギブス現象, 連続関数でも発散しうること)の原因である。")
    else:
        st.success("この核は良い核(n → ∞ で (ii) は有界, (iii) は 0 に近づく)。")

    st.subheader("良い核との合成積は f に近づく")
    test = st.selectbox("関数 f", list(TESTS), key="gk_f")
    f = TESTS[test](XG)
    g = per_conv(f, v)
    fig = go.Figure(go.Scatter(x=XG, y=f, mode="lines", line=dict(color=GRAY, width=2.5), name="f"))
    fig.add_trace(go.Scatter(x=XG, y=g, mode="lines", line=dict(color=C2, width=2), name="f * K"))
    layout(fig, title=f"f * K({name}, {pname} = {p:.4g})", xlab="x", height=340)
    show(fig)
    away = (np.abs(XG) > 0.3) & (np.abs(np.abs(XG) - PI) > 0.3)
    err = float(np.max(np.abs(g - f)[away if test.startswith("矩形") else slice(None)]))
    st.write(f"最大誤差 {'(不連続点から 0.3 以上はなれた所)' if test.startswith('矩形') else ''} = {err:.4f}")
    st.caption("矩形波の不連続点では, 対称な核なら左右の極限の平均 0 に近づく。ディリクレ核では不連続点のそばの飛び出し(約 9%)が残り, "
               "フェイェール核・ポアソン核・熱核では飛び出さない($K\\ge0$ なので, f*K は f の最大値と最小値の間に入る)。")

    with st.expander("練習問題"):
        st.markdown("**問1** フェイェール核が (i) をみたすことを, $F_N(x)=\\sum_{|n|\\le N}\\left(1-\\frac{|n|}{N+1}\\right)e^{inx}$ から示せ。")
        st.markdown("**問2** 箱型の核 $K_\\delta$($\\delta\\to0$)が良い核であることを示せ。")
        st.markdown("**問3** $F_N\\ge0$ を用いて, $|f|\\le B$ ならば $|\\sigma_Nf(x)|\\le B$ であることを示せ。")
        if st.toggle("解答・解説を表示", key="pr_gk"):
            st.markdown("**答1** $\\frac{1}{2\\pi}\\int_{-\\pi}^{\\pi}e^{inx}dx$ は $n=0$ のとき 1, それ以外は 0 なので, $n=0$ の項の係数 $1$ だけが残る。")
            st.markdown("**答2** (i) $\\frac{1}{2\\pi}\\cdot\\frac{\\pi}{\\delta}\\cdot2\\delta=1$。(ii) $K_\\delta\\ge0$ なので (i) から 1。"
                        "(iii) 任意の $\\eta>0$ に対し, $\\delta<\\eta$ なら $|x|\\ge\\eta$ で $K_\\delta=0$。")
            st.markdown("**答3** $|\\sigma_Nf(x)|\\le\\frac{1}{2\\pi}\\int|f(x-t)|F_N(t)\\,dt\\le B\\cdot\\frac{1}{2\\pi}\\int F_N=B$。"
                        "部分和ではこれが成り立たない($D_N$ は負の値をとる)。")
    page_refs([("stein2007", "第2章(合成積, 良い核)。この節の定義・定理はこの本の形にそろえた。"),
               ("katznelson2004", "第I章(総和核 summability kernel)。"),
               ("grafakos2014", "第1章(近似単位元 approximate identity)。")])


# ============================================================
# 6. チェザロ平均と総和法
# ============================================================
SERIES = {
    "1 − 1 + 1 − 1 + ⋯": (lambda n: (-1.0) ** n, "1/2", "1/2"),
    "1 − 2 + 3 − 4 + ⋯": (lambda n: (-1.0) ** n * (n + 1), "存在しない", "1/4"),
    "1 − 1/2 + 1/3 − 1/4 + ⋯(収束する)": (lambda n: (-1.0) ** n / (n + 1), "log 2", "log 2"),
    "Σ_{n≥1} cos nθ(θ = π/3)": (lambda n: np.where(n == 0, 0.0, np.cos(n * PI / 3)), "−1/2", "−1/2"),
}


def cesaro(a):
    s = np.cumsum(a)
    c1 = np.cumsum(s) / np.arange(1, len(s) + 1)
    c2 = np.cumsum(c1) / np.arange(1, len(s) + 1)
    return s, c1, c2


def tab_cesaro():
    st.info("部分和 $s_n$ が収束しない級数でも, 部分和の**平均**をとると収束することがある。"
            "このように, 級数に「和」を割り当てる方法を**総和法**という。フーリエ級数では, チェザロ平均が良い核(フェイェール核)との合成積になる。")
    st.subheader("チェザロ平均")
    st.markdown("級数 $\\sum_{n=0}^{\\infty}a_n$ の部分和を $s_n=a_0+a_1+\\cdots+a_n$ とし, その平均")
    st.latex(r"\sigma_N=\frac{s_0+s_1+\cdots+s_N}{N+1}")
    st.markdown("が $N\\to\\infty$ で $\\sigma$ に収束するとき, 級数は**チェザロ総和可能**(または $(C,1)$ 総和可能)で, 和は $\\sigma$ であるという。")
    st.markdown("- **正則性**:$s_n\\to s$ なら $\\sigma_N\\to s$(収束する級数には, ふつうの和を割り当てる)。\n"
                "- 例:$1-1+1-1+\\cdots$ は $s_n=1,0,1,0,\\ldots$ で収束しないが, $\\sigma_N\\to\\dfrac12$。")
    c = st.columns(2)
    sname = c[0].selectbox("級数", list(SERIES), key="ce_s")
    Nmax = c[1].slider("項の数", 20, 400, 100, 10, key="ce_N")
    fa, c1v, _ = SERIES[sname]
    n = np.arange(Nmax + 1)
    s, c1, c2 = cesaro(fa(n))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=n, y=s, mode="lines+markers", marker=dict(size=3), line=dict(color=GRAY, width=1), name="部分和 s_n"))
    fig.add_trace(go.Scatter(x=n, y=c1, mode="lines", line=dict(color=C1, width=2.5), name="チェザロ平均 σ_n"))
    fig.add_trace(go.Scatter(x=n, y=c2, mode="lines", line=dict(color=C2, width=2, dash="dash"), name="σ_n をもう一度平均"))
    lim = float(np.max(np.abs(c1[len(c1) // 2:]))) * 2.5 + 0.5 if sname.startswith("1 − 2") else None
    layout(fig, title=sname, xlab="n", height=380, yrange=(-lim, lim) if lim else None)
    show(fig)
    st.write(f"n = {Nmax}:部分和 {s[-1]:.4f}, チェザロ平均 {c1[-1]:.4f}, もう一度平均 {c2[-1]:.4f}"
             f"(チェザロ和は {c1v})")
    st.caption("$1-2+3-4+\\cdots$ では $\\sigma_n$ も収束しない($0$ と $\\frac12$ の付近を行き来する)が, もう一度平均すると $\\frac14$ に近づく"
               "(2回平均する方法は $(C,2)$ 総和法と同じ和を与える)。$\\sum\\cos n\\theta$ は $\\theta\\ne0$ でチェザロ和 $-\\frac12$:"
               "これはフェイェール核 $F_N(\\theta)\\to0$ と同じこと($\\frac12+\\sum\\cos n\\theta$ の部分和が $\\frac12D_N$)。")

    st.subheader("フーリエ級数のチェザロ平均:フェイェールの定理")
    st.latex(r"\sigma_Nf(x)=\frac{S_0f(x)+\cdots+S_Nf(x)}{N+1}=\sum_{n=-N}^{N}\left(1-\frac{|n|}{N+1}\right)c_ne^{inx}=(f*F_N)(x)")
    st.markdown("**フェイェールの定理**(1904):$f$ が連続な点で $\\sigma_Nf(x)\\to f(x)$。$f$ が連続な周期関数なら, 一様に収束する。"
                "フェイェール核は良い核なので, 前の節の定理からすぐにわかる。")
    st.caption("部分和 $S_Nf$ では, 連続関数でも収束しない点がありうる(デュ・ボワ=レーモン, 1873)。"
               "チェザロ平均をとることで, 連続関数ならいつでも収束する近似が得られる。")
    c = st.columns(2)
    test = c[0].selectbox("関数 f", list(TESTS), key="ce_f")
    Nn = c[1].slider("N", 1, 60, 10, key="ce_NN")
    f = TESTS[test](XG)
    SN = per_conv(f, dirichlet(Nn))
    sN = per_conv(f, fejer(Nn))
    cA, cB = st.columns(2)
    with cA:
        fig = go.Figure(go.Scatter(x=XG, y=f, mode="lines", line=dict(color=GRAY, width=2.5), name="f"))
        fig.add_trace(go.Scatter(x=XG, y=SN, mode="lines", line=dict(color=C1), name=f"S_{Nn} f"))
        fig.add_trace(go.Scatter(x=XG, y=sN, mode="lines", line=dict(color=C2), name=f"σ_{Nn} f"))
        layout(fig, title="部分和とチェザロ平均", xlab="x", height=340)
        show(fig)
    with cB:
        Ns = np.arange(1, 61)
        away = (np.abs(XG) > 0.3) & (np.abs(np.abs(XG) - PI) > 0.3) if test.startswith("矩形") else np.ones_like(XG, bool)
        eS = [float(np.max(np.abs(per_conv(f, dirichlet(k)) - f)[away])) for k in Ns]
        eF = [float(np.max(np.abs(per_conv(f, fejer(k)) - f)[away])) for k in Ns]
        fig = go.Figure(go.Scatter(x=Ns, y=eS, mode="lines", line=dict(color=C1), name="S_N の最大誤差"))
        fig.add_trace(go.Scatter(x=Ns, y=eF, mode="lines", line=dict(color=C2), name="σ_N の最大誤差"))
        fig.update_yaxes(type="log")
        layout(fig, title="最大誤差(対数目盛)", xlab="N", height=340)
        show(fig)
    st.caption("なめらかな関数では部分和の方が速く収束する(チェザロ平均は誤差が 1/N 程度より小さくならない)。"
               "チェザロ平均の長所は, 連続でさえあれば必ず一様収束し, 飛び出さないこと。目的に応じて総和法を使い分ける。")

    st.subheader("総和法をまとめると")
    st.markdown(r"""
| 総和法 | 級数 $\sum a_n$ に割り当てる値 | フーリエ級数では |
|---|---|---|
| ふつうの和 | $\lim s_n$ | $S_Nf=f*D_N$(ディリクレ核, 良い核ではない) |
| チェザロ(C,1) | $\lim\frac{s_0+\cdots+s_N}{N+1}$ | $\sigma_Nf=f*F_N$(フェイェール核) |
| アーベル(次の節) | $\lim_{r\to1-0}\sum a_nr^n$ | $A_rf=f*P_r$(ポアソン核) |
""")
    st.caption("収束する級数はチェザロ総和可能で, チェザロ総和可能な級数はアーベル総和可能(和はすべて同じ)。逆は成り立たない。")
    with st.expander("練習問題"):
        st.markdown("**問1** $1-1+1-1+\\cdots$ の $\\sigma_N$ を $N$ が偶数・奇数の場合に分けて求め, チェザロ和が $\\frac12$ であることを示せ。")
        st.markdown("**問2** $s_n\\to s$ ならば $\\sigma_N\\to s$ であることを示せ(正則性)。")
        st.markdown("**問3** $\\sigma_Nf=\\sum_{|n|\\le N}\\left(1-\\frac{|n|}{N+1}\\right)c_ne^{inx}$ を確かめよ。")
        if st.toggle("解答・解説を表示", key="pr_ce"):
            st.markdown("**答1** $N=2m$ のとき $s_0+\\cdots+s_N=m+1$ なので $\\sigma_N=\\frac{m+1}{2m+1}$, $N=2m+1$ のとき $\\sigma_N=\\frac{m+1}{2m+2}=\\frac12$。"
                        "どちらも $\\frac12$ に収束する。")
            st.markdown("**答2** $\\varepsilon>0$ に対し, $n\\ge n_0$ で $|s_n-s|<\\varepsilon$ とする。")
            st.latex(r"|\sigma_N-s|\le\frac{1}{N+1}\sum_{n<n_0}|s_n-s|+\frac{1}{N+1}\sum_{n=n_0}^{N}|s_n-s|"
                     r"\le\frac{C}{N+1}+\varepsilon\ \xrightarrow{N\to\infty}\ \varepsilon")
            st.markdown("**答3** $S_kf=\\sum_{|n|\\le k}c_ne^{inx}$ を $k=0,\\ldots,N$ について足すと, $c_ne^{inx}$ は $k\\ge|n|$ の $N+1-|n|$ 回現れる。"
                        "$N+1$ で割ると係数は $1-\\frac{|n|}{N+1}$。")
    page_refs([("stein2007", "第2章(チェザロ総和法, フェイェールの定理)。"),
               ("hardy1949", "総和法の一般論(正則性, (C,k) 総和法, 総和法どうしの関係)。"),
               ("zygmund2002", "三角級数の総和法の詳しい理論。"),
               ("fejer1904", "フェイェールの定理の原論文。"),
               ("korner1988", "フェイェールの定理とその周辺の読み物。")])


# ============================================================
# 7. ポアソン核とアーベル総和法
# ============================================================
ABEL_CLOSED = {
    "1 − 1 + 1 − 1 + ⋯": lambda r: 1 / (1 + r),
    "1 − 2 + 3 − 4 + ⋯": lambda r: 1 / (1 + r) ** 2,
    "1 − 1/2 + 1/3 − 1/4 + ⋯(収束する)": lambda r: np.log1p(r) / r,
    "Σ_{n≥1} cos nθ(θ = π/3)": lambda r: (r * np.cos(PI / 3) - r * r) / (1 - 2 * r * np.cos(PI / 3) + r * r),
}
ABEL_TEX = {
    "1 − 1 + 1 − 1 + ⋯": r"\sum(-1)^nr^n=\frac{1}{1+r}\to\frac12",
    "1 − 2 + 3 − 4 + ⋯": r"\sum(-1)^n(n+1)r^n=\frac{1}{(1+r)^2}\to\frac14",
    "1 − 1/2 + 1/3 − 1/4 + ⋯(収束する)": r"\sum\frac{(-1)^n}{n+1}r^n=\frac{\log(1+r)}{r}\to\log2",
    "Σ_{n≥1} cos nθ(θ = π/3)": r"\sum_{n\ge1}r^n\cos n\theta=\frac{r\cos\theta-r^2}{1-2r\cos\theta+r^2}\to-\frac12",
}

BOUND = {"矩形波": lambda x: np.sign(wrap(x)), "三角波 |θ|": lambda x: np.abs(wrap(x)), "cos 3θ": lambda x: np.cos(3 * x),
         "パルス(|θ| < 0.6 で 1)": lambda x: (np.abs(wrap(x)) < 0.6).astype(float)}


@st.cache_data
def disk_values(bname, nmax=200, ng=141):
    """単位円板のディリクレ問題の解 u(ρ, φ) = Σ ρ^{|n|} c_n e^{inφ}(境界の値 f の複素フーリエ係数 c_n)"""
    n, c = cn_grid(BOUND[bname](XG), nmax)
    g = np.linspace(-1, 1, ng)
    X, Y = np.meshgrid(g, g)
    rho, phi = np.hypot(X, Y), np.arctan2(Y, X)
    inside = rho <= 1
    u = np.full(X.shape, np.nan)
    rr, pp = rho[inside], phi[inside]
    u[inside] = np.real((c[None, :] * rr[:, None] ** np.abs(n)[None, :] * np.exp(1j * np.outer(pp, n))).sum(axis=1))
    return g, u


def tab_poisson_abel():
    st.info("$r\\ (0<r<1)$ をかけて $\\sum a_nr^n$ とし, $r\\to1$ の極限をとるのが**アーベル総和法**。"
            "フーリエ級数では, これが**ポアソン核**との合成積になり, 単位円板の**ディリクレ問題**(ラプラス方程式)の解を与える。")
    st.subheader("アーベル総和法")
    st.markdown("級数 $\\sum_{n=0}^{\\infty}a_n$ に対して, $0<r<1$ で $A(r)=\\sum_{n=0}^{\\infty}a_nr^n$ が収束し, "
                "$\\displaystyle\\lim_{r\\to1-0}A(r)=s$ となるとき, 級数は**アーベル総和可能**で, 和は $s$ であるという。")
    c = st.columns(2)
    sname = c[0].selectbox("級数", list(ABEL_CLOSED), key="ab_s")
    r1 = c[1].slider("r", 0.5, 0.999, 0.9, 0.001, key="ab_r")
    st.latex(ABEL_TEX[sname])
    rs = np.linspace(0.3, 0.999, 400)
    nn = np.arange(20000)
    fa = SERIES[sname][0]
    num = float(np.sum(fa(nn) * r1 ** nn))
    fig = go.Figure(go.Scatter(x=rs, y=ABEL_CLOSED[sname](rs), mode="lines", line=dict(color=C1, width=2.5), name="A(r)"))
    fig.add_trace(go.Scatter(x=[r1], y=[num], mode="markers", marker=dict(color=C2, size=11), name="級数の数値計算"))
    layout(fig, title="A(r) = Σ aₙ rⁿ と r → 1", xlab="r", height=320)
    show(fig)
    st.write(f"r = {r1}:Σ aₙ rⁿ(20000 項まで)= {num:.5f}、式の値 = {float(ABEL_CLOSED[sname](r1)):.5f}")
    st.markdown(r"""
| 級数 | ふつうの和 | チェザロ和 | アーベル和 |
|---|---|---|---|
| $1-1+1-\cdots$ | なし | $\frac12$ | $\frac12$ |
| $1-2+3-4+\cdots$ | なし | なし | $\frac14$ |
| $1-\frac12+\frac13-\cdots$ | $\log2$ | $\log2$ | $\log2$ |
""")
    st.caption("アーベル総和法は, チェザロ総和法より「強い」(より多くの級数に和を割り当てる)。"
               "逆に, アーベル和があり $na_n\\to0$ ならふつうに収束する(タウバーの定理)。このような「逆向きの定理」をタウバー型定理という。")

    st.subheader("ポアソン核")
    st.markdown("フーリエ級数 $\\sum c_ne^{inx}$ のアーベル平均は, ポアソン核との合成積になる。")
    st.latex(r"A_rf(x)=\sum_{n=-\infty}^{\infty}r^{|n|}c_ne^{inx}=(f*P_r)(x),\qquad "
             r"P_r(x)=\sum_{n=-\infty}^{\infty}r^{|n|}e^{inx}=\frac{1-r^2}{1-2r\cos x+r^2}")
    with st.expander("ポアソン核の式の導き方"):
        st.markdown("$\\omega=re^{ix}$ とおくと, 等比級数の和から")
        st.latex(r"\sum_{n=0}^{\infty}\omega^n+\sum_{n=1}^{\infty}\bar\omega^n=\frac{1}{1-\omega}+\frac{\bar\omega}{1-\bar\omega}"
                 r"=\frac{1-|\omega|^2}{|1-\omega|^2}=\frac{1-r^2}{1-2r\cos x+r^2}")
        st.markdown("$P_r>0$, $\\frac{1}{2\\pi}\\int P_r=1$($n=0$ の項だけが残る), $\\delta\\le|x|\\le\\pi$ で "
                    "$P_r(x)\\le\\dfrac{1-r^2}{1-2r\\cos\\delta+r^2}\\to0$ なので, $r\\to1$ で**良い核**。")
    r = st.slider("r(ポアソン核)", 0.1, 0.99, 0.8, 0.01, key="ab_pr")
    bname = st.selectbox("境界の関数 f(θ)", list(BOUND), key="ab_b")
    f = BOUND[bname](XG)
    cA, cB = st.columns(2)
    with cA:
        fig = go.Figure(go.Scatter(x=XG, y=poisson(r), mode="lines", line=dict(color=C1, width=2), name="P_r"))
        fig.add_trace(go.Scatter(x=XG, y=fejer(max(1, int(round(1 / (1 - r)))) ), mode="lines",
                                 line=dict(color=C3, dash="dot"), name="F_N(N ≈ 1/(1−r))"))
        layout(fig, title=f"ポアソン核 P_r(r = {r})", xlab="x", height=340)
        show(fig)
    with cB:
        fig = go.Figure(go.Scatter(x=XG, y=f, mode="lines", line=dict(color=GRAY, width=2.5), name="f"))
        for rr, col in ((0.5, C4), (0.8, C3), (r, C2)):
            fig.add_trace(go.Scatter(x=XG, y=per_conv(f, poisson(rr)), mode="lines", line=dict(color=col), name=f"A_r f(r = {rr})"))
        layout(fig, title="アーベル平均 A_r f = f * P_r", xlab="x", height=340)
        show(fig)

    st.subheader("単位円板のディリクレ問題")
    st.markdown("極座標 $(\\rho,\\ \\theta)$ で $u(\\rho,\\theta)=(f*P_\\rho)(\\theta)=\\sum\\rho^{|n|}c_ne^{in\\theta}$ とおくと, "
                "$\\rho^{|n|}e^{in\\theta}$ はどれも調和関数($z^n,\\ \\bar z^n$)なので, $u$ は円板の中でラプラス方程式 $\\Delta u=0$ をみたし, "
                "$\\rho\\to1$ で境界の値 $f(\\theta)$ に近づく(ポアソン核が良い核だから)。")
    st.latex(r"\Delta u=\frac{\partial^2u}{\partial x^2}+\frac{\partial^2u}{\partial y^2}=0\ (\rho<1),\qquad u(1,\theta)=f(\theta)")
    g, u = disk_values(bname)
    fig = go.Figure(go.Heatmap(x=g, y=g, z=u, colorscale="RdBu_r", colorbar=dict(title="u")))
    th = np.linspace(0, 2 * PI, 300)
    fig.add_trace(go.Scatter(x=np.cos(th), y=np.sin(th), mode="lines", line=dict(color="black", width=1.5), showlegend=False))
    fig.add_trace(go.Scatter(x=r * np.cos(th), y=r * np.sin(th), mode="lines", line=dict(color=GRAY, dash="dot"), name=f"ρ = {r}"))
    layout(fig, title=f"境界の値 f = {bname} の調和関数 u(ρ, θ)", xlab="x", ylab="y", height=440, equal=True, legend=False)
    show(fig)
    st.caption("点線の円(ρ = r)の上の u の値が, 上の図のアーベル平均 A_r f。中心の値は境界の値の平均 c₀(平均値の性質)。"
               "熱が十分に時間がたって落ち着いた温度分布(定常状態)とも考えられる(⑤ 偏微分方程式のラプラス方程式の節)。")
    with st.expander("練習問題"):
        st.markdown("**問1** $1-2+3-4+\\cdots$ のアーベル和を求めよ。($\\frac{1}{(1+r)^2}=\\sum(-1)^n(n+1)r^n$ を使う)")
        st.markdown("**問2** 境界の値が $f(\\theta)=\\cos3\\theta$ のとき, 単位円板のディリクレ問題の解 $u$ を $x,\\ y$ の式で表せ。")
        st.markdown("**問3** チェザロ和が $s$ の級数は, アーベル和も $s$ であることを, $\\sum a_nr^n=(1-r)^2\\sum(n+1)\\sigma_nr^n$ を用いて説明せよ。")
        if st.toggle("解答・解説を表示", key="pr_ab"):
            st.markdown("**答1** $\\frac{1}{1+r}$ を微分して符号を変えると $\\frac{1}{(1+r)^2}=\\sum(-1)^n(n+1)r^n$。$r\\to1$ で $\\frac14$。")
            st.markdown("**答2** $c_{\\pm3}=\\frac12$ だけなので $u=\\rho^3\\cos3\\theta=\\mathrm{Re}\\,z^3=x^3-3xy^2$。")
            st.markdown("**答3** $\\sum s_nr^n=\\frac{1}{1-r}\\sum a_nr^n$, $\\sum(n+1)\\sigma_nr^n=\\frac{1}{1-r}\\sum s_nr^n$ から式が出る。"
                        "$(1-r)^2\\sum(n+1)r^n=1$ なので, 右辺は $\\sigma_n$ の重みつき平均で, $r\\to1$ で重みは大きい $n$ に移る。"
                        "$\\sigma_n\\to s$ なら, その重みつき平均も $s$ に近づく。")
    page_refs([("stein2007", "第2章(アーベル総和法, ポアソン核, 単位円板のディリクレ問題)。"),
               ("katznelson2004", "第I章(ポアソン核)。"),
               ("hardy1949", "アーベル総和法とタウバー型定理。"),
               ("jimbo2024", "調和関数とディリクレ問題(複素関数の立場から)。")])
