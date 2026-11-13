"""⑦ 特殊関数 第2章 直交関数系:3〜6. ルジャンドル・チェビシェフ・エルミート・ラゲール多項式

各ページの構成:定義 → 直交性 → 漸化式と母関数 → 微分方程式(級数解法で導く)→ 関数の展開 → 応用(4学科)→ 歴史 → 練習問題。
史実・書誌は 2026 年 11 月に確認(MacTutor, Numdam, EuDML, arXiv など)。
"""
from math import factorial

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.interpolate import BarycentricInterpolator
from scipy.special import eval_genlaguerre

from common import C1, C2, C3, C4, GRAY, layout, show, trapz
from topics.fourier_apps import refs
from topics.fourier_kernels import page_refs
from topics.orth_core import POLY, WEIGHTS, hermite_functions, inner, laguerre_functions, norm2, p_eval, p_tex

COLS = [C1, C2, C3, C4, GRAY, "#8e44ad", "#16a085", "#d35400"]
SYM = {"legendre": "P", "chebyshev": "T", "hermite": "H", "laguerre": "L"}
BOUNDS = {"legendre": ("-1", "1"), "chebyshev": ("-1", "1"), "hermite": (r"-\infty", r"\infty"), "laguerre": ("0", r"\infty")}
RANGE = {"legendre": (-1, 1), "chebyshev": (-1, 1), "hermite": (-2.5, 2.5), "laguerre": (0, 15)}
GEN = {  # 母関数:(TeX, 数値, n 番目の係数に掛けるもの)
    "legendre": (r"\frac{1}{\sqrt{1-2xt+t^2}}=\sum_{n=0}^{\infty}P_n(x)\,t^n", lambda x, t: 1 / np.sqrt(1 - 2 * x * t + t * t), lambda n: 1.0),
    "chebyshev": (r"\frac{1-xt}{1-2xt+t^2}=\sum_{n=0}^{\infty}T_n(x)\,t^n", lambda x, t: (1 - x * t) / (1 - 2 * x * t + t * t), lambda n: 1.0),
    "hermite": (r"e^{2xt-t^2}=\sum_{n=0}^{\infty}H_n(x)\frac{t^n}{n!}", lambda x, t: np.exp(2 * x * t - t * t), lambda n: 1 / factorial(n)),
    "laguerre": (r"\frac{1}{1-t}\,e^{-\frac{xt}{1-t}}=\sum_{n=0}^{\infty}L_n(x)\,t^n", lambda x, t: np.exp(-x * t / (1 - t)) / (1 - t), lambda n: 1.0),
}
REC = {
    "legendre": r"(n+1)P_{n+1}(x)=(2n+1)xP_n(x)-nP_{n-1}(x)",
    "chebyshev": r"T_{n+1}(x)=2xT_n(x)-T_{n-1}(x)",
    "hermite": r"H_{n+1}(x)=2xH_n(x)-2nH_{n-1}(x)",
    "laguerre": r"(n+1)L_{n+1}(x)=(2n+1-x)L_n(x)-nL_{n-1}(x)",
}


# ------------------------------------------------------------
# 共通の部品
# ------------------------------------------------------------
def poly_list(fam, nmax=5):
    s = SYM[fam]
    st.latex(r"\begin{aligned}" + r"\\".join(f"{s}_{n}(x)&={p_tex(POLY[fam](n))}" for n in range(nmax + 1)) + r"\end{aligned}")


def poly_plot(fam, pre):
    c = st.columns([1, 2])
    nmax = c[0].select_slider("何次まで表示するか", [2, 3, 4, 5, 6], 4, key=f"{pre}_n")
    with c[0]:
        poly_list(fam, nmax)
    x = np.linspace(*RANGE[fam], 1201)
    fig = go.Figure()
    for n in range(nmax + 1):
        fig.add_trace(go.Scatter(x=x, y=p_eval(fam, n, x), name=f"{SYM[fam]}_{n}(x)", line=dict(color=COLS[n % 8])))
    yr = {"legendre": [-1.1, 1.1], "chebyshev": [-1.1, 1.1], "hermite": [-60, 60], "laguerre": [-15, 25]}[fam]
    layout(fig, None, xlab="x", height=380, yrange=yr)
    with c[1]:
        show(fig)


def weight(fam, x):
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"legendre": np.ones_like(x), "chebyshev": 1 / np.sqrt(np.clip(1 - x * x, 1e-15, None)),
                "hermite": np.exp(-x * x), "laguerre": np.exp(-x)}[fam]


def ortho_check(fam, pre):
    """p_m p_n w のグラフと積分(厳密値)"""
    name, itex, wtex, _, const = WEIGHTS[fam]
    s = SYM[fam]
    c = st.columns([1, 2])
    m = c[0].select_slider("m", [0, 1, 2, 3, 4, 5], 2, key=f"{pre}_m")
    n = c[0].select_slider("n", [0, 1, 2, 3, 4, 5], 3, key=f"{pre}_k")
    val = inner(POLY[fam](m), POLY[fam](n), fam)
    a, b = BOUNDS[fam]
    c[0].latex(rf"\int_{{{a}}}^{{{b}}}{s}_{m}(x){s}_{n}(x)\,{wtex}\,dx="
               + (p_tex([val]) if not const or val == 0 else (p_tex([val]) if val != 1 else "") + const))
    if m == n:
        c[0].caption("$m=n$ のときはノルムの2乗(正)。")
    else:
        c[0].caption("$m\\neq n$ のとき, 正の部分と負の部分の面積が打ち消し合って $0$ になる。")
    lo, hi = RANGE[fam] if fam != "laguerre" else (0, 25)
    if fam == "chebyshev":  # 端で重みが発散するので θ で描く
        th = np.linspace(0, np.pi, 1201)
        y = np.cos(m * th) * np.cos(n * th)
        xx, xl = th, "θ(x = cos θ)"
        ttl = "x = cos θ と置換した被積分関数 cos mθ cos nθ"
    else:
        xx = np.linspace(lo, hi, 2001)
        y = p_eval(fam, m, xx) * p_eval(fam, n, xx) * weight(fam, xx)
        xl, ttl = "x", f"被積分関数 {s}_m(x) {s}_n(x) w(x)"
    fig = go.Figure(go.Scatter(x=xx, y=np.where(y > 0, y, 0), fill="tozeroy", line=dict(color=C1, width=0), name="正の部分"))
    fig.add_trace(go.Scatter(x=xx, y=np.where(y < 0, y, 0), fill="tozeroy", line=dict(color=C2, width=0), name="負の部分"))
    fig.add_trace(go.Scatter(x=xx, y=y, line=dict(color="black", width=1.5), name="被積分関数"))
    layout(fig, ttl, xlab=xl, height=340)
    with c[1]:
        show(fig)


def genfun_check(fam, pre):
    tex, G, fac = GEN[fam]
    st.latex(tex)
    c = st.columns([1, 2])
    t = c[0].select_slider("t", [-0.6, -0.3, 0.2, 0.4, 0.6], 0.4, key=f"{pre}_t")
    N = c[0].select_slider("足す項の数", [1, 2, 3, 5, 8, 12, 20], 3, key=f"{pre}_gN")
    x = np.linspace(*RANGE[fam], 801)
    S = sum(p_eval(fam, n, x) * fac(n) * t ** n for n in range(N))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=G(x, t), name="母関数(左辺)", line=dict(color=GRAY, width=4)))
    fig.add_trace(go.Scatter(x=x, y=S, name=f"右辺の部分和({N} 項)", line=dict(color=C1, dash="dash")))
    layout(fig, f"t = {t} での母関数と部分和", xlab="x", height=320)
    with c[1]:
        show(fig)


# 展開する関数:fam -> {名前: (関数, TeX)}
EXP = {
    "legendre": {"|x|": (np.abs, r"|x|"), "階段関数 H(x)": (lambda x: (x > 0).astype(float), r"H(x)"), "e^x": (np.exp, r"e^x")},
    "chebyshev": {"|x|": (np.abs, r"|x|"), "階段関数 H(x)": (lambda x: (x > 0).astype(float), r"H(x)"), "1/(1 + 25x²)": (lambda x: 1 / (1 + 25 * x * x), r"\frac{1}{1+25x^2}")},
    "hermite": {"ずらしたガウス関数 e^{−(x−1)²/2}": (lambda x: np.exp(-(x - 1) ** 2 / 2), r"e^{-(x-1)^2/2}"),
                "矩形パルス(|x| < 1 で 1)": (lambda x: (np.abs(x) < 1).astype(float), r"\chi_{(-1,1)}(x)"),
                "e^{−|x|}": (lambda x: np.exp(-np.abs(x)), r"e^{-|x|}")},
    "laguerre": {"減衰振動 e^{−t/2} sin 2t": (lambda x: np.exp(-x / 2) * np.sin(2 * x), r"e^{-t/2}\sin 2t"),
                 "1次遅れ e^{−2t}": (lambda x: np.exp(-2 * x), r"e^{-2t}"),
                 "むだ時間つき(1 < t < 3 で 1)": (lambda x: ((x > 1) & (x < 3)).astype(float), r"\chi_{(1,3)}(t)")},
}


@st.cache_data(show_spinner=False)
def expansion_coefs(fam, fname, nmax):
    f = EXP[fam][fname][0]
    if fam == "legendre":
        g, w = np.polynomial.legendre.leggauss(300)
        xs = np.concatenate([(g - 1) / 2, (g + 1) / 2])
        ws = np.concatenate([w / 2, w / 2])
        return np.array([(2 * n + 1) / 2 * np.sum(ws * f(xs) * p_eval(fam, n, xs)) for n in range(nmax + 1)])
    if fam == "chebyshev":
        th = (np.arange(40000) + 0.5) * np.pi / 40000
        fv = f(np.cos(th))
        return np.array([(1 if n == 0 else 2) / np.pi * np.sum(fv * np.cos(n * th)) * np.pi / 40000 for n in range(nmax + 1)])
    if fam == "hermite":
        x = np.linspace(-14, 14, 56001)
        ph = hermite_functions(nmax, x)
        return np.array([trapz(f(x) * ph[n], x) for n in range(nmax + 1)])
    x = np.linspace(0, 120, 120001)
    lf = laguerre_functions(nmax, x)
    return np.array([trapz(f(x) * lf[n], x) for n in range(nmax + 1)])


def partial_sum(fam, coefs, x):
    N = len(coefs) - 1
    if fam == "hermite":
        return coefs @ hermite_functions(N, x)
    if fam == "laguerre":
        return coefs @ laguerre_functions(N, x)
    return sum(cn * p_eval(fam, n, x) for n, cn in enumerate(coefs))


def expansion_demo(fam, pre, title_fn="関数"):
    c = st.columns([1, 2])
    fname = c[0].selectbox(title_fn, list(EXP[fam]), key=f"{pre}_f")
    N = c[0].select_slider("何次まで使うか N", [2, 4, 6, 10, 16, 24, 40], 6, key=f"{pre}_N")
    co = expansion_coefs(fam, fname, 40)[: N + 1]
    f = EXP[fam][fname][0]
    lo, hi = {"legendre": (-1, 1), "chebyshev": (-1, 1), "hermite": (-6, 6), "laguerre": (0, 15)}[fam]
    x = np.linspace(lo, hi, 1201)
    err = float(np.sqrt(trapz((f(x) - partial_sum(fam, co, x)) ** 2 * (weight(fam, x) if fam in ("legendre",) else 1), x)))
    c[0].latex(", ".join(f"c_{n}\\approx{v:.3f}" for n, v in enumerate(co[:5])) + r",\ \dots")
    c[0].metric("区間での2乗平均の誤差", f"{err:.4f}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=f(x), name=f"f = {fname}", line=dict(color=GRAY, width=4)))
    fig.add_trace(go.Scatter(x=x, y=partial_sum(fam, co, x), name=f"N = {N} までの和", line=dict(color=C1)))
    layout(fig, None, xlab="t" if fam == "laguerre" else "x", height=360)
    with c[1]:
        show(fig)


# ------------------------------------------------------------
# 3. ルジャンドル多項式
# ------------------------------------------------------------
def tab_legendre():
    st.info("ルジャンドル多項式 $P_n(x)$ は, 区間 $[-1,1]$ で重み $1$ の内積について直交する多項式である。"
            "球のまわりの重力や電気のポテンシャルを $1/r$ の展開で表すときに現れ, $x=\\cos\\theta$ として角度の関数として使われる。")

    st.subheader("1. 定義:ロドリゲスの公式")
    st.latex(r"P_n(x)=\frac{1}{2^nn!}\frac{d^n}{dx^n}(x^2-1)^n\qquad(P_n(1)=1)")
    st.markdown("$P_n$ は $n$ 次の多項式で, $n$ が偶数なら偶関数, 奇数なら奇関数。区間 $(-1,1)$ に $n$ 個の零点をもつ。")
    poly_plot("legendre", "olg")

    st.subheader("2. 直交性")
    st.latex(r"\int_{-1}^{1}P_m(x)P_n(x)\,dx=\begin{cases}0&(m\neq n)\\[2pt]\dfrac{2}{2n+1}&(m=n)\end{cases}")
    st.markdown("**理由**($m<n$):ロドリゲスの公式を使い, $n$ 回部分積分すると $x^m$ の $n$ 回微分 $=0$ が出てくる。"
                "$(x^2-1)^n$ の $k$ 回微分($k<n$)は $x=\\pm1$ で $0$ なので, 境界の項はすべて消える。")
    ortho_check("legendre", "olg")

    st.subheader("3. 漸化式と母関数")
    st.latex(REC["legendre"])
    st.markdown("母関数は, 点 $x$ と距離 $1$ の点との距離の逆数 $\\dfrac1{\\sqrt{1-2xt+t^2}}$ の展開である($t$ が距離の比, $x$ が角度の余弦)。")
    genfun_check("legendre", "olg")

    st.subheader("4. 微分方程式:級数解法で導く")
    st.markdown("ルジャンドルの微分方程式 $(1-x^2)y^{\\prime\\prime}-2xy^{\\prime}+\\nu(\\nu+1)y=0$ に $y=\\sum a_kx^k$ を代入すると, $x^k$ の係数は")
    st.latex(r"(k+2)(k+1)a_{k+2}-k(k-1)a_k-2ka_k+\nu(\nu+1)a_k=0")
    st.markdown("$k(k-1)+2k=k(k+1)$ なので")
    st.latex(r"a_{k+2}=\frac{k(k+1)-\nu(\nu+1)}{(k+2)(k+1)}a_k=\frac{(k-\nu)(k+\nu+1)}{(k+2)(k+1)}a_k")
    st.markdown("$\\nu=n$(整数)のとき $a_{n+2}=0$ で級数が止まり, $n$ 次の多項式解が得られる。$P_n(1)=1$ となるように定数倍したものが $P_n$ である。"
                "例えば $n=2$, $a_0=1$, $a_1=0$ とすると $a_2=\\dfrac{(0-2)(0+3)}{2\\cdot1}=-3$ で $y=1-3x^2=-2P_2(x)$。")
    st.caption("この方程式は, ラプラス方程式 $\\Delta u=0$ を球座標で変数分離したとき, 角度 $\\theta$ の方程式として $x=\\cos\\theta$ で現れる。"
               "$\\nu$ が整数でない解は $x=\\pm1$(北極・南極)で発散する(「1. 級数解法」のページの図を参照)。")

    st.subheader("5. ルジャンドル展開(フーリエ・ルジャンドル級数)")
    st.latex(r"f(x)\approx\sum_{n=0}^{N}c_nP_n(x),\qquad c_n=\frac{2n+1}{2}\int_{-1}^{1}f(x)P_n(x)\,dx")
    expansion_demo("legendre", "olg")

    st.subheader("6. 応用")
    st.markdown("**電気:多重極展開**  $z$ 軸上の点 $a$ にある点電荷が, 距離 $r$($>a$), 角度 $\\theta$ の点につくる電位は, 母関数を使って")
    st.latex(r"\frac{1}{\sqrt{r^2-2ar\cos\theta+a^2}}=\frac1r\cdot\frac{1}{\sqrt{1-2\cos\theta\,(a/r)+(a/r)^2}}=\sum_{n=0}^{\infty}\frac{a^n}{r^{n+1}}P_n(\cos\theta)")
    st.markdown("と展開できる。$n=0$ が点電荷, $n=1$ が双極子, $n=2$ が四重極の項で, 遠くでは前の項ほど効く。")
    c = st.columns([1, 2])
    ra = c[0].select_slider("距離の比 r / a", [1.2, 1.5, 2.0, 3.0, 5.0], 1.5, key="olg_r")
    N = c[0].select_slider("何項まで(n ≤ N)", [0, 1, 2, 4, 8, 16], 2, key="olg_mp")
    th = np.linspace(0, np.pi, 721)
    exact = 1 / np.sqrt(ra ** 2 - 2 * ra * np.cos(th) + 1)
    approx = sum(p_eval("legendre", n, np.cos(th)) / ra ** (n + 1) for n in range(N + 1))
    c[0].metric("最大誤差", f"{float(np.max(np.abs(exact - approx))):.4f}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.degrees(th), y=exact, name="厳密な電位(a = 1)", line=dict(color=GRAY, width=4)))
    fig.add_trace(go.Scatter(x=np.degrees(th), y=approx, name=f"多重極展開(n ≤ {N})", line=dict(color=C1, dash="dash")))
    layout(fig, "点電荷の電位と多重極展開", xlab="角度 θ(度)", ylab="電位(比例定数を除く)", height=340)
    with c[1]:
        show(fig)
    st.markdown("**機械・都市:ガウス・ルジャンドル求積**  $P_n$ の $n$ 個の零点 $x_i$ と重み $w_i$ を使う $\\int_{-1}^1f\\,dx\\approx\\sum w_if(x_i)$ は, "
                "$2n-1$ 次までの多項式で**厳密**になる。有限要素法の要素ごとの積分(剛性行列の計算)に使われる。")
    c = st.columns([1, 2])
    nq = c[0].select_slider("分点の数 n", [1, 2, 3, 4, 5, 8], 3, key="olg_q")
    xi, wi = np.polynomial.legendre.leggauss(nq)
    ex = np.e - 1 / np.e
    gauss = float(np.sum(wi * np.exp(xi)))
    xt = np.linspace(-1, 1, max(nq, 2))
    trap = float(trapz(np.exp(xt), xt))
    c[0].latex(r"\int_{-1}^{1}e^x\,dx=e-\frac1e")
    c[0].metric("ガウス求積の誤差", f"{abs(gauss - ex):.2e}")
    c[0].metric(f"台形公式({max(nq, 2)} 点)の誤差", f"{abs(trap - ex):.2e}")
    x = np.linspace(-1, 1, 801)
    fig = go.Figure(go.Scatter(x=x, y=p_eval("legendre", nq, x), name=f"P_{nq}(x)", line=dict(color=C1)))
    fig.add_trace(go.Scatter(x=xi, y=0 * xi, mode="markers", marker=dict(size=11, color=C2), name="分点(P_n の零点)"))
    layout(fig, "ガウス・ルジャンドル求積の分点", xlab="x", height=300, yrange=[-1.1, 1.1])
    with c[1]:
        show(fig)
    st.markdown(r"""
| 学科 | 使われ方 |
|---|---|
| 機械 | ガウス・ルジャンドル求積(有限要素法の要素の積分) |
| 電気電子 | 電位・電界の多重極展開, 球の外の電位(ラプラス方程式の解) |
| 化学 | 原子軌道の角度部分(s, p, d 軌道)は $P_n(\cos\theta)$ を一般化した球面調和関数で表される |
| 都市 | 地球の重力場(ジオイド)を球面調和関数で表す。地球の扁平さを表す主な項は $P_2(\cos\theta)$ の項 |
""")

    st.subheader("7. 歴史")
    st.markdown(r"""
| 年 | できごと |
|---|---|
| 1780年代 | ルジャンドル(A.-M. Legendre)が回転楕円体の引力や惑星の形の研究で, 距離の逆数の展開の係数として導入 |
| 1816 | ロドリゲス(O. Rodrigues)が学位論文で, $n$ 回微分による表示(ロドリゲスの公式)を与える |
| 19世紀 | ラプラス方程式の球座標での解法(球面調和関数)の一部として, 物理・天文学で広く使われる |
""")

    with st.expander("練習問題"):
        st.markdown("**問1** ロドリゲスの公式から $P_2(x)$ を求めよ。")
        st.markdown("**問2** 漸化式を使って $P_3(x)$ を求め, $P_3\\left(\\dfrac12\\right)$ を計算せよ。")
        st.markdown("**問3** $\\displaystyle\\int_{-1}^1x^2P_2(x)\\,dx$ を求めよ。")
        st.markdown("**問4** $x^3$ を $P_1$, $P_3$ で表せ。")
        if st.toggle("解答・解説を表示", key="pr_olg"):
            st.markdown("**答1** $\\dfrac{1}{8}\\dfrac{d^2}{dx^2}(x^4-2x^2+1)=\\dfrac18(12x^2-4)=\\dfrac{3x^2-1}{2}$。")
            st.markdown("**答2** $3P_3=5xP_2-2P_1$ より $P_3=\\dfrac{5x^3-3x}{2}$, $P_3\\left(\\dfrac12\\right)=\\dfrac{5/8-3/2}{2}=-\\dfrac{7}{16}$。")
            st.markdown("**答3** $x^2=\\dfrac13P_0+\\dfrac23P_2$ と直交性より $\\dfrac23\\cdot\\dfrac25=\\dfrac4{15}$。")
            st.markdown("**答4** $x^3=\\dfrac{2P_3+3x}{5}=\\dfrac35P_1+\\dfrac25P_3$。")

    page_refs([("inui1962", "ルジャンドル関数と球関数。"), ("rodrigues1816", "ロドリゲスの公式。"), ("szego1939", "直交多項式の一般論。"),
               ("nist2010", "第18章:ルジャンドル多項式の公式。")])
    refs([("MacTutor:Adrien-Marie Legendre", "https://mathshistory.st-andrews.ac.uk/Biographies/Legendre.html", "ルジャンドルの生涯と仕事(英語)"),
          ("MacTutor:Benjamin Olinde Rodrigues", "https://mathshistory.st-andrews.ac.uk/Biographies/Rodrigues.html", "ロドリゲスの公式を含む学位論文(英語)"),
          ("On the Work of Benjamin Olinde Rodrigues (1795-1851)(arXiv:2006.00196)", "https://arxiv.org/abs/2006.00196", "ロドリゲスの仕事の解説(英語)"),
          ("DLMF §18.3 Definitions", "https://dlmf.nist.gov/18.3", "古典直交多項式の区間・重み・ノルム(英語)")])


# ------------------------------------------------------------
# 4. チェビシェフ多項式
# ------------------------------------------------------------
def tab_chebyshev():
    st.info("チェビシェフ多項式 $T_n(x)$ は $T_n(\\cos\\theta)=\\cos n\\theta$ で定まる多項式で, フーリエ級数の $\\cos n\\theta$ を $x=\\cos\\theta$ で書き直したものである。"
            "区間 $[-1,1]$ 全体で誤差が均等になる近似(最良一様近似)と深く関係し, 数値計算やフィルタの設計に使われる。")

    st.subheader("1. 定義:$T_n(\\cos\\theta)=\\cos n\\theta$")
    st.markdown("$\\cos2\\theta=2\\cos^2\\theta-1$, $\\cos3\\theta=4\\cos^3\\theta-3\\cos\\theta$ より $T_2(x)=2x^2-1$, $T_3(x)=4x^3-3x$。"
                "一般に, 加法定理 $\\cos(n+1)\\theta+\\cos(n-1)\\theta=2\\cos\\theta\\cos n\\theta$ から漸化式が出る。")
    poly_plot("chebyshev", "och")
    st.caption("$\\lvert T_n(x)\\rvert\\le1$ で, 区間 $[-1,1]$ の中で $n+1$ 回 $\\pm1$ の値をとる(等しく振動する)。")

    st.subheader("2. 直交性")
    st.latex(r"\int_{-1}^{1}T_m(x)T_n(x)\frac{dx}{\sqrt{1-x^2}}=\int_0^{\pi}\cos m\theta\cos n\theta\,d\theta=\begin{cases}0&(m\neq n)\\ \pi&(m=n=0)\\ \dfrac\pi2&(m=n\ge1)\end{cases}")
    ortho_check("chebyshev", "och")

    st.subheader("3. 漸化式と母関数")
    st.latex(REC["chebyshev"])
    genfun_check("chebyshev", "och")

    st.subheader("4. 微分方程式:級数解法で導く")
    st.markdown("$y=\\cos n\\theta$ は $\\dfrac{d^2y}{d\\theta^2}+n^2y=0$ をみたす。$x=\\cos\\theta$ に変数変換すると")
    st.latex(r"(1-x^2)y^{\prime\prime}-xy^{\prime}+n^2y=0\qquad(\text{チェビシェフの微分方程式})")
    st.markdown("$y=\\sum a_kx^k$ を代入すると, $x^k$ の係数から")
    st.latex(r"(k+2)(k+1)a_{k+2}-k(k-1)a_k-ka_k+n^2a_k=0\ \Longrightarrow\ a_{k+2}=\frac{k^2-n^2}{(k+2)(k+1)}a_k=\frac{(k-n)(k+n)}{(k+2)(k+1)}a_k")
    st.markdown("$k=n$ で係数が $0$ になり多項式解が得られる。例えば $n=3$, $a_0=0$, $a_1=1$ では $a_3=\\dfrac{(1-3)(1+3)}{3\\cdot2}=-\\dfrac43$ で, "
                "$y=x-\\dfrac43x^3=-\\dfrac13T_3(x)$。")

    st.subheader("5. チェビシェフ展開")
    st.latex(r"f(x)\approx\sum_{n=0}^{N}c_nT_n(x),\qquad c_n=\frac{2}{\pi}\int_0^{\pi}f(\cos\theta)\cos n\theta\,d\theta\quad\left(c_0\ \text{は}\ \frac1\pi\int_0^\pi\right)")
    st.markdown("$f(\\cos\\theta)$ のフーリエ余弦級数そのものなので, 高速フーリエ変換で速く計算できる。")
    expansion_demo("chebyshev", "och")

    st.subheader("6. 応用")
    st.markdown("**最良一様近似とルンゲ現象**  最高次の係数が $1$ の $n$ 次多項式の中で, 区間 $[-1,1]$ での最大値 $\\max\\lvert p(x)\\rvert$ が最小になるのは "
                "$2^{1-n}T_n(x)$ である(最大値 $2^{1-n}$)。このため, 補間の点を $T_n$ の零点 $x_k=\\cos\\dfrac{(2k+1)\\pi}{2n}$ にとると補間の誤差が小さくなる。"
                "等間隔の点で補間すると, 端の近くで大きく振動する(**ルンゲ現象**)。")
    c = st.columns([1, 2])
    n = c[0].select_slider("補間の点の数", [5, 7, 9, 11, 15, 21], 11, key="och_rn")
    f = lambda x: 1 / (1 + 25 * x * x)  # noqa: E731
    xe = np.linspace(-1, 1, n)
    xc = np.cos((2 * np.arange(n) + 1) * np.pi / (2 * n))
    x = np.linspace(-1, 1, 1201)
    pe, pc = BarycentricInterpolator(xe, f(xe))(x), BarycentricInterpolator(xc, f(xc))(x)
    c[0].latex(r"f(x)=\frac{1}{1+25x^2}")
    c[0].metric("等間隔の点:最大誤差", f"{float(np.max(np.abs(pe - f(x)))):.3f}")
    c[0].metric("チェビシェフ点:最大誤差", f"{float(np.max(np.abs(pc - f(x)))):.3f}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=f(x), name="f(x)", line=dict(color=GRAY, width=4)))
    fig.add_trace(go.Scatter(x=x, y=pe, name="等間隔の点で補間", line=dict(color=C2)))
    fig.add_trace(go.Scatter(x=x, y=pc, name="チェビシェフ点で補間", line=dict(color=C1)))
    fig.add_trace(go.Scatter(x=xc, y=f(xc), mode="markers", marker=dict(color=C1, size=7), name="チェビシェフ点"))
    layout(fig, "補間の点の選び方と誤差", xlab="x", height=360, yrange=[-0.5, 1.5])
    with c[1]:
        show(fig)
    st.markdown("**電気:チェビシェフ・フィルタ**  通過域で利得が少し波打つ(リプル)かわりに, 遮断周波数 $\\omega_c$ の外で急に減衰する低域通過フィルタ:")
    st.latex(r"\lvert H(j\omega)\rvert^2=\frac{1}{1+\varepsilon^2T_n^2(\omega/\omega_c)}\qquad\left(\text{バターワース}:\ \frac{1}{1+(\omega/\omega_c)^{2n}}\right)")
    c = st.columns([1, 2])
    order = c[0].select_slider("次数 n", [2, 3, 4, 5, 6], 4, key="och_ord")
    rip = c[0].select_slider("リプル(dB)", [0.1, 0.5, 1.0, 3.0], 1.0, key="och_rip")
    eps = np.sqrt(10 ** (rip / 10) - 1)
    w = np.linspace(0, 3, 1201)
    Tn = np.cos(order * np.arccos(np.clip(w, -1, 1))) * (w <= 1) + np.cosh(order * np.arccosh(np.maximum(w, 1))) * (w > 1)
    gc = 10 * np.log10(1 / (1 + eps ** 2 * Tn ** 2))
    gb = 10 * np.log10(1 / (1 + w ** (2 * order)))
    c[0].metric("ω = 2ω_c での減衰(チェビシェフ)", f"{float(np.interp(2, w, gc)):.1f} dB")
    c[0].metric("ω = 2ω_c での減衰(バターワース)", f"{float(np.interp(2, w, gb)):.1f} dB")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w, y=gc, name="チェビシェフ", line=dict(color=C1)))
    fig.add_trace(go.Scatter(x=w, y=gb, name="バターワース", line=dict(color=C2, dash="dash")))
    layout(fig, "低域通過フィルタの利得", xlab="ω / ω_c", ylab="利得(dB)", height=340, yrange=[-60, 3])
    with c[1]:
        show(fig)
    st.caption("$\\lvert\\omega/\\omega_c\\rvert>1$ では $T_n(x)=\\cosh(n\\,\\mathrm{arccosh}\\,x)$ と急に大きくなるので, 減衰が急になる。")
    st.markdown(r"""
| 学科 | 使われ方 |
|---|---|
| 機械 | リンク機構の設計(チェビシェフの出発点), 制御系の関数近似 |
| 電気電子 | チェビシェフ・フィルタ, アンテナアレイの設計(ドルフ・チェビシェフ配列) |
| 化学 | 物性値・測定データを区間全体で誤差が均等な多項式で近似する |
| 都市 | 観測データの近似, 数値計算(関数の値を計算するプログラムでもチェビシェフ展開が使われる) |
""")

    st.subheader("7. 歴史")
    st.markdown(r"""
| 年 | できごと |
|---|---|
| 1854 | チェビシェフ(P. L. Chebyshev)が論文「平行四辺形と呼ばれる機構の理論」で, $0$ からのずれが最小の多項式として導入(回転運動を直線運動に変えるリンク機構の研究から) |
| 1850年代後半 | 連分数・最小2乗法の研究の中で, 直交多項式の一般論を展開 |
| 20世紀 | 数値解析(関数近似・補間・スペクトル法)やフィルタ設計の基本的な道具になる |
""")

    with st.expander("練習問題"):
        st.markdown("**問1** $\\cos4\\theta$ を $\\cos\\theta$ で表して $T_4(x)$ を求めよ。")
        st.markdown("**問2** $T_5\\left(\\dfrac12\\right)$ を求めよ。")
        st.markdown("**問3** $x^2$ を $T_0$, $T_2$ で表せ。")
        if st.toggle("解答・解説を表示", key="pr_och"):
            st.markdown("**答1** 漸化式 $T_4=2xT_3-T_2=8x^4-8x^2+1$。")
            st.markdown("**答2** $\\dfrac12=\\cos\\dfrac\\pi3$ より $T_5\\left(\\dfrac12\\right)=\\cos\\dfrac{5\\pi}{3}=\\dfrac12$。")
            st.markdown("**答3** $T_2=2x^2-1$ より $x^2=\\dfrac12T_0+\\dfrac12T_2$。")

    page_refs([("szego1939", "チェビシェフ多項式と最良近似。"), ("nist2010", "第18章:チェビシェフ多項式の公式。")])
    refs([("MacTutor:Pafnuty Lvovich Chebyshev", "https://mathshistory.st-andrews.ac.uk/Biographies/Chebyshev.html", "チェビシェフの生涯と機構の研究(英語)"),
          ("The Life, Work and Legacy of P. L. Chebyshev(arXiv:2111.12551)", "https://arxiv.org/abs/2111.12551", "チェビシェフの仕事の解説(英語)"),
          ("DLMF §18.3 Definitions", "https://dlmf.nist.gov/18.3", "古典直交多項式の区間・重み・ノルム(英語)")])


# ------------------------------------------------------------
# 5. エルミート多項式
# ------------------------------------------------------------
def tab_hermite():
    st.info("エルミート多項式 $H_n(x)$ は, 実数全体 $(-\\infty,\\infty)$ で重み $e^{-x^2}$ について直交する多項式である。"
            "ガウス関数と組み合わせたエルミート関数 $H_n(x)e^{-x^2/2}$ は, 量子力学の調和振動子の波動関数であり, フーリエ変換で形を変えない関数でもある。")
    st.caption("ここでは物理学の規約 $H_n$ を使う。確率・統計では $He_n(x)=2^{-n/2}H_n(x/\\sqrt2)$(重み $e^{-x^2/2}$)を使うことが多い。")

    st.subheader("1. 定義:ロドリゲスの公式")
    st.latex(r"H_n(x)=(-1)^ne^{x^2}\frac{d^n}{dx^n}e^{-x^2}")
    st.markdown("例えば $\\dfrac{d}{dx}e^{-x^2}=-2xe^{-x^2}$ より $H_1=2x$, $\\dfrac{d^2}{dx^2}e^{-x^2}=(4x^2-2)e^{-x^2}$ より $H_2=4x^2-2$。")
    poly_plot("hermite", "ohm")

    st.subheader("2. 直交性")
    st.latex(r"\int_{-\infty}^{\infty}H_m(x)H_n(x)e^{-x^2}dx=\begin{cases}0&(m\neq n)\\ \sqrt\pi\,2^nn!&(m=n)\end{cases}")
    st.markdown("ノルムの計算にはガウス積分 $\\int_{-\\infty}^{\\infty}e^{-x^2}dx=\\sqrt\\pi$ を使う(第1章「ガンマ関数とベータ関数の関係」)。")
    ortho_check("hermite", "ohm")

    st.subheader("3. 漸化式と母関数")
    st.latex(REC["hermite"] + r",\qquad H_n^{\prime}(x)=2nH_{n-1}(x)")
    genfun_check("hermite", "ohm")

    st.subheader("4. 微分方程式:級数解法で導く")
    st.latex(r"y^{\prime\prime}-2xy^{\prime}+2\lambda y=0\qquad(\text{エルミートの微分方程式})")
    st.markdown("$y=\\sum a_kx^k$ を代入すると $-2xy^{\\prime}$ から $-2ka_k$ が出て, $x^k$ の係数から")
    st.latex(r"(k+2)(k+1)a_{k+2}-2ka_k+2\lambda a_k=0\ \Longrightarrow\ a_{k+2}=\frac{2(k-\lambda)}{(k+2)(k+1)}a_k")
    st.markdown("$\\lambda=n$ のとき $a_{n+2}=0$ で, $n$ 次の多項式解になる。最高次の係数を $2^n$ にしたものが $H_n$。"
                "$\\lambda$ が整数でないと $\\dfrac{a_{k+2}}{a_k}\\approx\\dfrac2k$ となり, 解は $e^{x^2}$ のように増える。")

    st.subheader("5. エルミート関数による展開")
    st.markdown("$\\varphi_n(x)=\\dfrac{H_n(x)e^{-x^2/2}}{\\sqrt{\\sqrt\\pi\\,2^nn!}}$ とすると, 重みなしの内積 $\\int\\varphi_m\\varphi_n\\,dx$ で正規直交になる。"
                "実数全体で2乗が積分できる関数 $f$ は $f\\approx\\sum c_n\\varphi_n$, $c_n=\\int f\\varphi_n\\,dx$ と展開できる。")
    expansion_demo("hermite", "ohm")
    st.markdown("**フーリエ変換との関係**  $F(\\omega)=\\int f(x)e^{-i\\omega x}dx$ の規約で")
    st.latex(r"\mathcal F[\varphi_n](\omega)=\sqrt{2\pi}\,(-i)^n\varphi_n(\omega)")
    st.markdown("すなわちエルミート関数は**フーリエ変換の固有関数**である($n=0$ はガウス関数が自分自身に移るという③の結果)。")

    st.subheader("6. 応用")
    st.markdown("**化学:量子力学の調和振動子(分子の振動)**  単位を選んで $-\\dfrac12\\psi^{\\prime\\prime}+\\dfrac12x^2\\psi=E\\psi$ とすると, "
                "$\\psi=y\\,e^{-x^2/2}$ とおけば $y$ はエルミートの方程式をみたす($E=\\lambda+\\frac12$)。$\\psi$ が遠くで $0$ になるには $\\lambda=n$(整数)が必要で,")
    st.latex(r"E_n=n+\frac12,\qquad\psi_n=\varphi_n(x)")
    st.markdown("エネルギーはとびとびの値をとる。")
    c = st.columns([1, 2])
    nmax = c[0].select_slider("何番目の状態まで", [1, 2, 3, 4, 6, 8], 3, key="ohm_q")
    sq = c[0].toggle("確率密度 |ψ|² を表示", key="ohm_sq")
    x = np.linspace(-4.5, 4.5, 1201)
    ph = hermite_functions(nmax, x)
    fig = go.Figure(go.Scatter(x=x, y=x * x / 2, name="ポテンシャル x²/2", line=dict(color=GRAY, width=3)))
    for n in range(nmax + 1):
        E = n + 0.5
        y = ph[n] ** 2 if sq else ph[n]
        fig.add_trace(go.Scatter(x=x, y=E + 0 * x, showlegend=False, line=dict(color="#ccc", width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=x, y=E + 0.6 * y, name=f"n = {n}(E = {E})", line=dict(color=COLS[n % 8])))
        xt = np.sqrt(2 * n + 1)
        fig.add_trace(go.Scatter(x=[-xt, xt], y=[E, E], mode="markers", marker=dict(color=COLS[n % 8], symbol="line-ns-open", size=12), showlegend=False))
    layout(fig, "調和振動子のエネルギー準位と波動関数(縦線:古典的な折り返し点)", xlab="x", ylab="E", height=440, yrange=[0, nmax + 1.5])
    with c[1]:
        show(fig)
    st.markdown("**選択則**:漸化式から $x\\varphi_n=\\sqrt{\\dfrac{n+1}{2}}\\varphi_{n+1}+\\sqrt{\\dfrac n2}\\varphi_{n-1}$ なので, "
                "$\\int\\varphi_mx\\varphi_n\\,dx$ は $m=n\\pm1$ のときだけ $0$ でない。赤外線の吸収で, 振動の状態が1つずつしか変わらない理由である。")
    st.markdown(r"""
| 学科 | 使われ方 |
|---|---|
| 機械 | 入力がガウス分布でばらつくときの応答の期待値の計算(ガウス・エルミート求積) |
| 電気電子 | レーザー光の横モード(エルミート・ガウスモード), 光通信 |
| 化学 | 分子振動のエネルギー準位と赤外吸収の選択則 |
| 都市 | 材料定数・荷重のばらつき(正規分布)を考えた構造の信頼性の評価 |
""")

    st.subheader("7. 歴史")
    st.markdown(r"""
| 年 | できごと |
|---|---|
| 1810 | ラプラス(P.-S. Laplace)が, 今とは見分けにくい形で扱う |
| 1859 | チェビシェフが詳しく研究する |
| 1864 | エルミート(C. Hermite)が関数の新しい級数展開として論じ, のちにエルミート多項式と呼ばれる |
| 1926 | シュレディンガー方程式の調和振動子の解として量子力学に現れる |
""")

    with st.expander("練習問題"):
        st.markdown("**問1** 漸化式から $H_3(x)$, $H_4(x)$ を求めよ。")
        st.markdown("**問2** $\\displaystyle\\int_{-\\infty}^{\\infty}H_1(x)^2e^{-x^2}dx$ を, $\\int x^2e^{-x^2}dx=\\dfrac{\\sqrt\\pi}{2}$ を使って求めよ。")
        st.markdown("**問3** $H_n^{\\prime}=2nH_{n-1}$ を $n=3$ で確かめよ。")
        if st.toggle("解答・解説を表示", key="pr_ohm"):
            st.markdown("**答1** $H_3=2xH_2-4H_1=8x^3-12x$, $H_4=2xH_3-6H_2=16x^4-48x^2+12$。")
            st.markdown("**答2** $4\\int x^2e^{-x^2}dx=2\\sqrt\\pi$(公式 $\\sqrt\\pi\\,2^nn!$ で $n=1$)。")
            st.markdown("**答3** $H_3^{\\prime}=24x^2-12=6(4x^2-2)=6H_2$。")

    page_refs([("hermite1864", "エルミート多項式による展開。"), ("schrodinger1926", "調和振動子(量子力学)。"), ("szego1939", "エルミート多項式。"),
               ("nist2010", "第18章:エルミート多項式の公式。")])
    refs([("DLMF §18.3 Definitions", "https://dlmf.nist.gov/18.3", "古典直交多項式の区間・重み・ノルム(英語)")])


# ------------------------------------------------------------
# 6. ラゲール多項式
# ------------------------------------------------------------
def hydrogen_radial(n, l, r):
    """水素原子の動径関数 R_{nl}(r)(ボーア半径を単位, 数値で規格化)"""
    rho = 2 * r / n
    R = rho ** l * np.exp(-rho / 2) * eval_genlaguerre(n - l - 1, 2 * l + 1, rho)
    return R / np.sqrt(trapz(R * R * r * r, r))


def tab_laguerre():
    st.info("ラゲール多項式 $L_n(x)$ は, 区間 $[0,\\infty)$ で重み $e^{-x}$ について直交する多項式である。"
            "時間 $t\\ge0$ で減衰する信号を表すのに向いており, ラプラス変換と相性がよい。水素原子の電子の分布(動径関数)にも現れる。")

    st.subheader("1. 定義:ロドリゲスの公式")
    st.latex(r"L_n(x)=\frac{e^x}{n!}\frac{d^n}{dx^n}\left(x^ne^{-x}\right)=\sum_{k=0}^{n}\binom nk\frac{(-1)^k}{k!}x^k\qquad(L_n(0)=1)")
    poly_plot("laguerre", "olq")

    st.subheader("2. 直交性")
    st.latex(r"\int_0^{\infty}L_m(x)L_n(x)e^{-x}dx=\begin{cases}0&(m\neq n)\\1&(m=n)\end{cases}")
    st.markdown("計算にはガンマ関数 $\\int_0^\\infty x^ke^{-x}dx=\\Gamma(k+1)=k!$ を使う(第1章「ガンマ関数」)。")
    ortho_check("laguerre", "olq")

    st.subheader("3. 漸化式と母関数")
    st.latex(REC["laguerre"])
    genfun_check("laguerre", "olq")

    st.subheader("4. 微分方程式:級数解法で導く")
    st.latex(r"xy^{\prime\prime}+(1-x)y^{\prime}+\lambda y=0\qquad(\text{ラゲールの微分方程式})")
    st.markdown("$y=\\sum a_kx^k$ を代入する。$xy^{\\prime\\prime}=\\sum(k+1)ka_{k+1}x^k$, $y^{\\prime}=\\sum(k+1)a_{k+1}x^k$, $xy^{\\prime}=\\sum ka_kx^k$ より, $x^k$ の係数から")
    st.latex(r"(k+1)ka_{k+1}+(k+1)a_{k+1}-ka_k+\lambda a_k=0\ \Longrightarrow\ a_{k+1}=\frac{k-\lambda}{(k+1)^2}a_k")
    st.markdown("$\\lambda=n$ で $a_{n+1}=0$ となり, $a_0=1$ とすると $a_k=\\dbinom nk\\dfrac{(-1)^k}{k!}$, つまり $L_n(x)$ が得られる。")
    st.caption("$x=0$ は $y^{\\prime\\prime}$ の係数 $x$ が $0$ になる確定特異点だが, 決定方程式が $r^2=0$ なので $r=0$ のべき級数解が1つ得られる"
               "(「1. 級数解法」の4.)。")

    st.subheader("5. ラゲール関数による展開とラプラス変換")
    st.markdown("$\\ell_n(t)=e^{-t/2}L_n(t)$ は $[0,\\infty)$ で重みなしの内積について正規直交になる(**ラゲール関数**)。"
                "インパルス応答のような $t\\ge0$ で減衰する関数を $g(t)\\approx\\sum c_n\\ell_n(t)$, $c_n=\\int_0^\\infty g\\,\\ell_n\\,dt$ と近似できる。")
    expansion_demo("laguerre", "olq", "インパルス応答 g(t)")
    st.markdown("ラプラス変換は, $\\mathcal L[t^k]=\\dfrac{k!}{s^{k+1}}$ と2項定理から")
    st.latex(r"\mathcal L[L_n(t)]=\sum_{k=0}^n\binom nk\frac{(-1)^k}{s^{k+1}}=\frac{(s-1)^n}{s^{n+1}},\qquad\mathcal L[\ell_n(t)]=\frac{(s-\frac12)^n}{(s+\frac12)^{n+1}}")
    st.markdown("どれも同じ極 $s=-\\frac12$ をもつので, 伝達関数を少ない数の係数 $c_n$ で表せる(制御のモデル化に使われるラゲール・ネットワーク)。")

    st.subheader("6. 応用")
    st.markdown("**化学:水素原子の電子の分布**  水素原子のシュレディンガー方程式を球座標で変数分離すると, 距離 $r$ の部分(動径関数)は")
    st.latex(r"R_{n\ell}(r)\propto\rho^{\ell}e^{-\rho/2}L^{(2\ell+1)}_{n-\ell-1}(\rho),\qquad\rho=\frac{2r}{na_0}")
    st.markdown("と表される。$L^{(\\alpha)}_m$ はラゲール多項式を一般化したもの(**ラゲールの陪多項式**, 重み $x^\\alpha e^{-x}$ で直交)。"
                "距離 $r$ に電子を見つける確率密度は $r^2R_{n\\ell}^2$ で, 節(確率が $0$ の球面)の数は $n-\\ell-1$ 個。")
    c = st.columns([1, 2])
    n = c[0].select_slider("主量子数 n", [1, 2, 3, 4], 3, key="olq_hn")
    ls = list(range(n))
    l_ = c[0].radio("方位量子数 ℓ", ls, format_func=lambda v: f"ℓ = {v}({'spdf'[v]} 軌道)", horizontal=True, key="olq_hl")
    l_ = min(l_, n - 1)
    r = np.linspace(1e-6, 40, 4001)
    P = r * r * hydrogen_radial(n, l_, r) ** 2
    c[0].metric("節の数 n − ℓ − 1", f"{n - l_ - 1}")
    c[0].metric("最も見つかりやすい距離", f"{float(r[np.argmax(P)]):.2f} a₀")
    fig = go.Figure(go.Scatter(x=r, y=P, fill="tozeroy", name=f"{n}{'spdf'[l_]} 軌道", line=dict(color=C1)))
    layout(fig, f"水素原子の {n}{'spdf'[l_]} 軌道の動径分布 r²R²", xlab="r(ボーア半径 a₀ を単位)", height=340)
    with c[1]:
        show(fig)
    st.markdown(r"""
| 学科 | 使われ方 |
|---|---|
| 機械 | インパルス応答・伝達関数のモデル化(ラゲール関数による近似) |
| 電気電子 | ラゲール・ネットワーク(全域通過要素を並べたフィルタ), 信号の近似 |
| 化学 | 水素原子・水素類似原子の動径関数と電子の分布 |
| 都市 | 区間 $[0,\infty)$ の積分の数値計算(ガウス・ラゲール求積) |
""")

    st.subheader("7. 歴史")
    st.markdown(r"""
| 年 | できごと |
|---|---|
| 1879 | ラゲール(E. Laguerre)が積分 $\int_x^\infty\frac{e^{-x}}{x}dx$ を連分数で近似する研究で導入し, 直交性を示す |
| 1926 | シュレディンガー(E. Schrödinger)の水素原子の解に, ラゲールの陪多項式として現れる |
""")

    with st.expander("練習問題"):
        st.markdown("**問1** ロドリゲスの公式から $L_2(x)$ を求めよ。")
        st.markdown("**問2** $\\displaystyle\\int_0^\\infty L_1(x)^2e^{-x}dx$ を, $\\int_0^\\infty x^ke^{-x}dx=k!$ を使って求めよ。")
        st.markdown("**問3** $\\mathcal L[L_2(t)]$ を求めよ。")
        if st.toggle("解答・解説を表示", key="pr_olq"):
            st.markdown("**答1** $\\dfrac{e^x}{2}(x^2e^{-x})^{\\prime\\prime}=\\dfrac{e^x}{2}(2-4x+x^2)e^{-x}=1-2x+\\dfrac{x^2}{2}$。")
            st.markdown("**答2** $(1-x)^2=1-2x+x^2$ より $0!-2\\cdot1!+2!=1$。")
            st.markdown("**答3** $\\dfrac1s-\\dfrac2{s^2}+\\dfrac1{s^3}=\\dfrac{(s-1)^2}{s^3}$。")

    page_refs([("laguerre1879", "ラゲール多項式と直交性。"), ("schrodinger1926", "水素原子の動径関数。"), ("szego1939", "ラゲール多項式。"),
               ("nist2010", "第18章:ラゲール多項式の公式。")])
    refs([("Numdam:E. Laguerre, Sur l'intégrale ∫ e^{−x} dx/x(1879)", "https://www.numdam.org/item/BSMF_1879__7__72_1", "ラゲールの原論文(フランス語)"),
          ("MacTutor:Edmond Nicolas Laguerre", "https://mathshistory.st-andrews.ac.uk/Biographies/Laguerre/", "ラゲールの生涯と仕事(英語)"),
          ("DLMF §18.3 Definitions", "https://dlmf.nist.gov/18.3", "古典直交多項式の区間・重み・ノルム(英語)")])
