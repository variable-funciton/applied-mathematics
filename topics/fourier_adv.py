"""③ フーリエ解析 発展:ギブス現象・ピンスキー現象・倉坪現象

  1. ギブス現象(1次元のフーリエ級数, 不連続点のそばの飛び出し)
  2. ピンスキー現象(R^d の球の特性関数のフーリエ積分, 球の中心で収束しない)
  3. 倉坪現象(トーラス T^d の球の特性関数のフーリエ級数, d ≥ 5 で有理点で収束しない)

トーラス上の球面部分和 S_λf(x) = Σ_{|m|<λ} f̂(m) e^{2πi m·x} は, f̂ が |m| だけで決まるので
  S_λf(x) = Σ_{k<λ²} f̂(√k) · w_x(k),   w_x(k) = Σ_{|m|²=k} cos(2π m·x)
と書ける。w_x は1次元のテータ級数 Σ_n cos(2πn x_j) q^{n²} の積の係数なので, 畳み込みで速く計算できる。
"""
import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.special import gamma, jv, sici

from common import C1, C2, C3, C4, GRAY, animate, layout, show

PI = np.pi
SI_PI = sici(PI)[0]                      # Si(π) = 1.8519...
GIBBS = SI_PI / PI - 0.5                  # 跳びに対する飛び出しの割合 0.0894898...


# ============================================================
# 1. ギブス現象
# ============================================================
def sq_partial(x, N, factor=None):
    """矩形波(−1, 1)のフーリエ級数の部分和 (4/π) Σ_{k 奇数 ≤ N} w_k sin kx / k"""
    s = np.zeros_like(x, dtype=float)
    for k in range(1, N + 1, 2):
        w = 1.0
        if factor == "fejer":
            w = 1 - k / (N + 1)
        elif factor == "lanczos":
            w = np.sinc(k / (N + 1))
        s += w * np.sin(k * x) / k
    return 4 / PI * s


def tab_gibbs():
    st.info("不連続点をもつ関数のフーリエ級数の部分和は, 不連続点のそばで**跳びの約 9% だけ飛び出す**。"
            "項を増やしても飛び出しの高さは小さくならず, 幅が狭くなるだけである。これを**ギブス現象**という。")
    st.markdown("矩形波 $f(x)=\\begin{cases}-1&(-\\pi<x<0)\\\\1&(0<x<\\pi)\\end{cases}$ の部分和")
    st.latex(r"S_N(x)=\frac{4}{\pi}\sum_{\substack{k=1\\k:\text{奇数}}}^{N}\frac{\sin kx}{k}")
    st.markdown("の最大値は, $N\\to\\infty$ で $1$ ではなく")
    st.latex(r"\lim_{N\to\infty}\max_x S_N(x)=\frac{2}{\pi}\,\mathrm{Si}(\pi)=1.17897\cdots,\qquad "
             r"\mathrm{Si}(t)=\int_0^t\frac{\sin u}{u}\,du")
    st.markdown(f"に近づく。跳びの大きさ $2$ に対する飛び出しの割合は $\\dfrac{{\\mathrm{{Si}}(\\pi)}}{{\\pi}}-\\dfrac12={GIBBS:.4f}$(約 9%)で, "
                "どんな不連続点でも, 跳びの大きさに対して同じ割合になる。")

    N = st.slider("N(項の最大の次数)", 1, 99, 15, 2, key="gb_N")
    x = np.linspace(-PI, PI, 4001)
    s = sq_partial(x, N)
    cA, cB = st.columns(2)
    with cA:
        fig = go.Figure(go.Scatter(x=x, y=np.sign(x), mode="lines", line=dict(color=GRAY, width=2), name="f(x)"))
        fig.add_trace(go.Scatter(x=x, y=s, mode="lines", line=dict(color=C1), name=f"S_{N}(x)"))
        fig.add_hline(y=2 / PI * SI_PI, line=dict(color=C2, dash="dash"), annotation_text="1.179")
        layout(fig, title="矩形波と部分和", xlab="x", height=360, yrange=(-1.4, 1.4))
        show(fig)
    with cB:
        xz = np.linspace(-0.6, 0.6, 1201)
        fig = go.Figure(go.Scatter(x=xz, y=np.sign(xz), mode="lines", line=dict(color=GRAY, width=2), name="f(x)"))
        fig.add_trace(go.Scatter(x=xz, y=sq_partial(xz, N), mode="lines", line=dict(color=C1), name=f"S_{N}(x)"))
        fig.add_hline(y=2 / PI * SI_PI, line=dict(color=C2, dash="dash"))
        layout(fig, title="不連続点 x = 0 の付近を拡大", xlab="x", height=360, yrange=(-1.4, 1.4))
        show(fig)
    m = float(s.max())
    st.write(f"max S_{N}(x) = {m:.5f}(飛び出し {m - 1:.5f}, 跳び 2 に対して {100 * (m - 1) / 2:.2f}%)。"
             f"N → ∞ の極限は 1.17898(8.95%)。")

    st.subheader("飛び出しは小さくならず, 幅が縮むだけ")
    st.markdown("横軸を $N$ 倍して $t=Nx$ で見ると, $N$ によらず同じ形 $\\dfrac{2}{\\pi}\\mathrm{Si}(t)$ に近づく。")
    t = np.linspace(-12, 12, 1201)
    fig = go.Figure(go.Scatter(x=t, y=2 / PI * sici(t)[0], mode="lines", line=dict(color="black", width=3),
                               name="(2/π) Si(t)"))
    for Nn, col in ((5, C3), (15, C4), (45, C2), (135, C1)):
        fig.add_trace(go.Scatter(x=t, y=sq_partial(t / Nn, Nn), mode="lines", line=dict(color=col, width=1.6),
                                 name=f"S_{Nn}(t/N)"))
    layout(fig, title="S_N(t/N) と (2/π)Si(t)", xlab="t = Nx", height=360, yrange=(-1.4, 1.4))
    show(fig)
    st.caption("最大値は $t=\\pi$(つまり $x=\\pi/N$ の付近)でとる。$N$ が大きいほど飛び出しは不連続点に近づくが, 高さは変わらない。")
    Ns = list(range(1, 81, 2))
    xs = np.linspace(-0.9, 0.9, 721)
    fig = animate(xs, [[sq_partial(xs, n), np.sign(xs)] for n in Ns], ["S_N(x)", "f(x)"], [str(n) for n in Ns],
                  colors=[C1, GRAY], title="N を増やしていく(▶ 再生)", xlab="x", yrange=(-1.4, 1.4), height=400,
                  label_prefix="N = ", duration=120)
    show(fig)

    with st.expander("なぜ 1.179 になるのか(導出)"):
        st.markdown("$N=2M-1$ とする。項別に微分して, 三角関数の和の公式を使うと")
        st.latex(r"S_N'(x)=\frac4\pi\sum_{j=1}^{M}\cos(2j-1)x=\frac{2}{\pi}\cdot\frac{\sin 2Mx}{\sin x}")
        st.markdown("なので, $x>0$ で最初に $S_N'=0$ となるのは $x=\\dfrac{\\pi}{2M}$。そこでの値は")
        st.latex(r"S_N\!\left(\frac{\pi}{2M}\right)=\frac{2}{\pi}\sum_{j=1}^{M}\frac{\sin\frac{(2j-1)\pi}{2M}}{\frac{(2j-1)\pi}{2M}}\cdot\frac{\pi}{M}"
                 r"\ \xrightarrow{M\to\infty}\ \frac{2}{\pi}\int_0^{\pi}\frac{\sin u}{u}\,du=\frac{2}{\pi}\mathrm{Si}(\pi)")
        st.caption("和は, 区間 $[0,\\ \\pi]$ を幅 $\\pi/M$ に分けた $\\dfrac{\\sin u}{u}$ の中点公式(リーマン和)。")

    st.subheader("飛び出しをおさえる:総和法")
    st.markdown("係数に重みをかけると飛び出しは小さくなる(なめらかさとのかね合い)。")
    st.latex(r"\text{フェイェール:}\ 1-\frac{k}{N+1},\qquad \text{ランチョス:}\ \mathrm{sinc}\frac{k}{N+1}=\frac{\sin\frac{k\pi}{N+1}}{\frac{k\pi}{N+1}}")
    N2 = st.slider("N(総和法の比較)", 3, 99, 31, 2, key="gb_N2")
    fig = go.Figure(go.Scatter(x=xz, y=np.sign(xz), mode="lines", line=dict(color=GRAY, width=2), name="f"))
    rows = []
    for fac, nm, col in ((None, "部分和 S_N", C1), ("lanczos", "ランチョスの σ 因子", C3), ("fejer", "フェイェール平均 σ_N", C2)):
        y = sq_partial(xz, N2, fac)
        fig.add_trace(go.Scatter(x=xz, y=y, mode="lines", line=dict(color=col), name=nm))
        rows.append(f"{nm}:最大値 {float(sq_partial(x, N2, fac).max()):.4f}")
    layout(fig, title=f"x = 0 の付近(N = {N2})", xlab="x", height=360, yrange=(-1.4, 1.4))
    show(fig)
    st.write("、".join(rows))
    st.caption("フェイェール平均は飛び出さないかわりに, 不連続点のそばの立ち上がりがなだらかになる。"
               "ランチョスの因子はその中間(飛び出しは跳びの約 1%)。")
    with st.expander("歴史"):
        st.markdown("- 1848 年に H. Wilbraham が最初に指摘したが, 注目されなかった。\n"
                    "- 1898 年, A. A. Michelson が調和解析機(フーリエ級数を機械で足し合わせる装置)で矩形波の飛び出しに気づき, "
                    "J. W. Gibbs が Nature 誌への手紙(1898, 1899)で数学的に説明した。\n"
                    "- 1906 年に M. Bôcher が「ギブス現象」と名づけた。ギブス–ウィルブラハム現象ともいう。")
    with st.expander("練習問題"):
        st.markdown("**問1** のこぎり波 $f(x)=x\\ (-\\pi<x<\\pi)$ の部分和は, $x=\\pi$ の付近でおよそどこまで飛び出すか。")
        st.markdown("**問2** 上の導出にならって, $S_N\\!\\left(\\dfrac{\\pi}{2M}\\right)$ を $M=1,\\ 2$ について計算し, 1.179 に近づくことを確かめよ。")
        if st.toggle("解答・解説を表示", key="pr_fadv_gibbs"):
            st.markdown("**答1**")
            st.write(f"跳びの大きさは 2π なので, 飛び出しは約 {GIBBS:.4f} × 2π ≈ {GIBBS * 2 * PI:.3f}。"
                     f"π ≈ 3.142 を超えて, およそ {PI + GIBBS * 2 * PI:.3f} まで上がる。")
            st.markdown("**答2**")
            st.latex(r"M=1:\ S_1\!\left(\frac\pi2\right)=\frac4\pi=1.273,\qquad "
                     r"M=2:\ S_3\!\left(\frac\pi4\right)=\frac4\pi\left(\frac{1}{\sqrt2}+\frac{1}{3\sqrt2}\right)=\frac{16}{3\sqrt2\,\pi}=1.200")


# ============================================================
# 2. ピンスキー現象(R^d のフーリエ積分)
# ============================================================
U = np.linspace(0, 80, 32001)


@st.cache_data
def center_curve(d):
    """S_R f(0) を u = aR の関数として:(2^{1−d/2}/Γ(d/2)) ∫_0^u s^{d/2−1} J_{d/2}(s) ds"""
    g = np.where(U > 0, np.power(np.maximum(U, 1e-300), d / 2 - 1) * jv(d / 2, U), 0.0)
    if d == 1:
        g = np.where(U > 0, g, np.sqrt(2 / PI))       # s^{−1/2} J_{1/2}(s) → √(2/π)
    cum = np.r_[0, np.cumsum((g[1:] + g[:-1]) / 2 * np.diff(U))]
    return 2 ** (1 - d / 2) / gamma(d / 2) * cum


@st.cache_data
def radial_profile(d, R, a=1.0, nr=6001):
    """S_R f(ρ) = a^{d/2} ρ^{1−d/2} ∫_0^R J_{d/2}(ar) J_{d/2−1}(rρ) dr(ρ > 0)。ρ = 0 は center_curve"""
    rho = np.linspace(1e-3, 2.0, 400)
    r = np.linspace(0, R, nr)
    Ja = jv(d / 2, a * r)
    M = jv(d / 2 - 1, np.outer(rho, r)) * Ja[None, :]
    integ = (M[:, 1:] + M[:, :-1]).sum(axis=1) / 2 * (r[1] - r[0])
    prof = a ** (d / 2) * rho ** (1 - d / 2) * integ
    c0 = float(np.interp(a * R, U, center_curve(d)))
    return np.r_[0.0, rho], np.r_[c0, prof]


def tab_pinsky():
    st.info("3次元で, 球の内側で 1, 外側で 0 となる関数(球の**特性関数**)をフーリエ積分で表すと, "
            "不連続点のない**球の中心**で収束しない。これを**ピンスキー現象**という。球面上の不連続から出た波が, "
            "中心に集まる(レンズの焦点のように)ために起こる。")
    st.markdown("$d$ 次元の関数 $f$ のフーリエ変換 $F(\\omega)=\\displaystyle\\int_{\\mathbb R^d}f(x)e^{-i\\omega\\cdot x}dx$ に対して, "
                "半径 $R$ の球の中の周波数だけを使った**球面部分和**(部分積分)を考える。")
    st.latex(r"S_Rf(x)=\frac{1}{(2\pi)^d}\int_{|\omega|<R}F(\omega)e^{i\omega\cdot x}\,d\omega,\qquad "
             r"f(x)=\chi_{\{|x|<a\}}=\begin{cases}1&(|x|<a)\\0&(|x|>a)\end{cases}")
    st.markdown("球の中心 $x=0$ での値は, 次のように計算できる($\\mathrm{Si}$ は正弦積分, $J_0$ はベッセル関数)。")
    st.latex(r"d=1:\ S_Rf(0)=\frac{2}{\pi}\mathrm{Si}(aR)\to1,\qquad d=2:\ S_Rf(0)=1-J_0(aR)\to1")
    st.latex(r"d=3:\ S_Rf(0)=\frac{2}{\pi}\left\{\mathrm{Si}(aR)-\sin aR\right\}\approx 1-\frac{2}{\pi}\sin aR\quad(\text{収束しない})")
    with st.expander("d = 3 の計算"):
        st.markdown("球対称な関数のフーリエ変換は $|\\omega|=r$ だけで決まり, 半径 $a$ の球の特性関数では")
        st.latex(r"F(\omega)=\int_{|x|<a}e^{-i\omega\cdot x}dx=\frac{4\pi(\sin ar-ar\cos ar)}{r^3}")
        st.markdown("極座標で($d\\omega=r^2\\sin\\theta\\,dr\\,d\\theta\\,d\\phi$, 球面の面積 $4\\pi r^2$)")
        st.latex(r"S_Rf(0)=\frac{1}{8\pi^3}\int_0^R F(r)\,4\pi r^2\,dr=\frac{2}{\pi}\int_0^R\frac{\sin ar-ar\cos ar}{r}dr"
                 r"=\frac{2}{\pi}\left\{\mathrm{Si}(aR)-\sin aR\right\}")
        st.caption("$\\mathrm{Si}(aR)\\to\\dfrac{\\pi}{2}$ なので, $S_Rf(0)$ は $1$ のまわりを振幅 $\\dfrac{2}{\\pi}\\approx0.64$ で振動し続ける。")

    ds = st.multiselect("次元 d", [1, 2, 3, 4, 5], default=[1, 2, 3], key="pk_d")
    umax = st.slider("aR の範囲", 10, 80, 40, 5, key="pk_u")
    m = U <= umax
    fig = go.Figure()
    cols = {1: C3, 2: C4, 3: C2, 4: C1, 5: "#9467bd"}
    for d in ds or [3]:
        fig.add_trace(go.Scatter(x=U[m], y=center_curve(d)[m], mode="lines", line=dict(color=cols[d], width=2.2),
                                 name=f"d = {d}"))
    fig.add_hline(y=1, line=dict(color=GRAY, dash="dash"))
    layout(fig, title="球の中心での値 S_R f(0)(f(0) = 1)", xlab="aR", height=380,
           yrange=None if (set(ds) & {4, 5}) else (-0.2, 2.0))
    show(fig)
    st.caption("$d=1,\\ 2$ では $1$ に収束する($d=2$ はゆっくり)。$d=3$ では振幅 $\\frac{2}{\\pi}$ で振動し続け, "
               "$d\\ge4$ では振幅が $R^{(d-3)/2}$ のように大きくなる。")

    st.subheader("中心以外では収束する(d = 3)")
    R = st.slider("R(使う周波数の上限, a = 1)", 5.0, 60.0, 20.0, 2.5, key="pk_R")
    rho, prof = radial_profile(3, R)
    cA, cB = st.columns(2)
    with cA:
        fig = go.Figure(go.Scatter(x=rho, y=(rho < 1).astype(float), mode="lines", line=dict(color=GRAY, width=2),
                                   name="f(ρ)"))
        fig.add_trace(go.Scatter(x=rho, y=prof, mode="lines", line=dict(color=C1), name="S_R f(ρ)"))
        fig.add_trace(go.Scatter(x=[0], y=[prof[0]], mode="markers", marker=dict(color=C2, size=11), name="中心"))
        layout(fig, title="中心からの距離 ρ = |x| の関数として", xlab="ρ", height=360, yrange=(-0.4, 1.9))
        show(fig)
    with cB:
        g = np.linspace(-1.6, 1.6, 161)
        X, Y = np.meshgrid(g, g)
        Z = np.interp(np.hypot(X, Y), rho, prof)
        fig = go.Figure(go.Heatmap(x=g, y=g, z=Z, colorscale="RdBu_r", zmin=-0.6, zmax=1.6,
                                   colorbar=dict(title="S_R f")))
        th = np.linspace(0, 2 * PI, 200)
        fig.add_trace(go.Scatter(x=np.cos(th), y=np.sin(th), mode="lines", line=dict(color="black", dash="dot"),
                                 name="球面 |x| = 1"))
        layout(fig, title="z = 0 の断面", xlab="x", ylab="y", height=360, equal=True, legend=False)
        show(fig)
    st.write(f"中心の値 S_R f(0) = {prof[0]:.4f}(R を変えると 1 のまわりを ±0.64 で上下する)。")
    st.caption("$\\rho\\ne0$ では, $R$ を大きくすると $f$ に近づく(球面 $\\rho=1$ のそばにはギブス現象の飛び出しが残る)。"
               "中心だけが特別なのは, 球面上のすべての点から等しい距離にあるため, 球面から来る振動が同じ位相でそろって重なるからである。")
    with st.expander("まとめと文献"):
        st.markdown(r"""
| 次元 | 不連続な球面のそば | 球の中心 |
|---|---|---|
| $d=1$ | ギブス現象 | 収束 |
| $d=2$ | ギブス現象 | 収束 |
| $d=3$ | ギブス現象 | **振動して収束しない**(振幅 $\frac{2}{\pi}$) |
| $d\ge4$ | ギブス現象($d=4$ まで) | 振幅が $R^{(d-3)/2}$ で大きくなる |
""")
        st.markdown("- M. A. Pinsky, N. K. Stanton, P. E. Trapa, Fourier series of radial functions in several variables, "
                    "J. Funct. Anal. 116 (1993).\n"
                    "- M. A. Pinsky, Introduction to Fourier Analysis and Wavelets, Brooks/Cole (2002).\n"
                    "- M. E. Taylor, The Gibbs phenomenon, the Pinsky phenomenon, and variants for eigenfunction expansions, "
                    "Comm. PDE 27 (2002).\n"
                    "- 「ピンスキー現象」の名前は J.-P. Kahane による(1995)。")


# ============================================================
# 3. 倉坪現象(トーラス上のフーリエ級数)
# ============================================================
def theta(K, x):
    """Σ_n cos(2πnx) q^{n²} の係数(q^0 .. q^K)"""
    n = np.arange(-math.isqrt(K), math.isqrt(K) + 1)
    c = np.zeros(K + 1)
    np.add.at(c, n * n, np.cos(2 * PI * n * x))
    return c


def lattice_weights(K, xs):
    """w_x(k) = Σ_{|m|²=k} cos(2π m·x)(k = 0..K)"""
    out = np.zeros(K + 1)
    out[0] = 1.0
    for x in xs:
        out = mul_theta(out, x)
    return out


def mul_theta(w, x):
    """w に Σ_n cos(2πnx) q^{n²} をかける(q^K まで)。テータ級数は n² の項しかないので, ずらして足す"""
    K = len(w) - 1
    out = w.copy()                                   # n = 0
    for n in range(1, math.isqrt(K) + 1):
        out[n * n:] += 2 * math.cos(2 * PI * n * x) * w[:K + 1 - n * n]
    return out


def ball_fhat(K, a, d):
    """f̂(m) = a^{d/2} J_{d/2}(2πa|m|) / |m|^{d/2}(|m|² = k), f̂(0) = 球の体積"""
    r = np.sqrt(np.arange(K + 1, dtype=float))
    out = np.empty(K + 1)
    out[0] = PI ** (d / 2) * a ** d / gamma(d / 2 + 1)
    out[1:] = a ** (d / 2) * jv(d / 2, 2 * PI * a * r[1:]) / r[1:] ** (d / 2)
    return out


@st.cache_data
def torus_partial(xs, a, lam):
    """λ = √k ごとの S_λ f(x)(|m|² ≤ k の和)"""
    K = int(lam * lam)
    return np.cumsum(ball_fhat(K, a, len(xs)) * lattice_weights(K, list(xs)))


def torus_f(xs, a):
    """周期的にならべた球の特性関数(境界では 1/2)"""
    x = np.array([v - round(v) for v in xs])
    r = float(np.linalg.norm(x))
    return 1.0 if r < a - 1e-12 else (0.5 if abs(r - a) < 1e-12 else 0.0)


def pinsky_amp(d, a):
    return a ** ((d - 3) / 2) * PI ** ((d - 4) / 2) / gamma(d / 2)


IRR = [math.sqrt(2) - 1, PI / 10, math.sqrt(5) - 2, math.e / 10 + 0.05, math.sqrt(3) - 1.4, math.log(2) - 0.25,
       math.sqrt(7) - 2.3]


def points_for(d):
    return {"中心 0(ピンスキー)": tuple([0.0] * d),
            "有理点 (1/2, 0, …, 0)": tuple([0.5] + [0.0] * (d - 1)),
            "有理点 (1/3, 1/3, 0, …)": tuple(([1 / 3, 1 / 3] + [0.0] * d)[:d]),
            "無理点 (√2−1, π/10, …)": tuple(IRR[:d])}


def tab_kuratsubo():
    st.info("ピンスキー現象はフーリエ**積分**の話だった。周期的な場合(トーラス上のフーリエ**級数**)では, "
            "さらに高い次元 $d\\ge5$ で, 不連続点から離れた**有理点**(座標がすべて有理数の点)でも収束しないことが, "
            "倉坪茂彦によって示された(2010)。ここではこれを**倉坪現象**と呼ぶ。無理数の点では収束する(ほとんどすべての点で収束する)。")
    st.markdown("$d$ 次元トーラス $\\mathbb T^d=(-\\tfrac12,\\tfrac12]^d$(各座標が周期 1)の上で, 半径 $a\\ (0<a<\\tfrac12)$ の球の特性関数 $f$ を考える。"
                "フーリエ係数と球面部分和は")
    st.latex(r"\hat f(m)=\int_{\mathbb T^d}f(x)e^{-2\pi i m\cdot x}dx=\frac{a^{d/2}J_{d/2}(2\pi a|m|)}{|m|^{d/2}},\qquad "
             r"S_\lambda f(x)=\sum_{m\in\mathbb Z^d,\ |m|<\lambda}\hat f(m)\,e^{2\pi i m\cdot x}")
    st.caption("$m=(m_1,\\ldots,m_d)$ は整数の組(格子点)。$\\hat f(m)$ は $|m|$ だけで決まるので, 和は「半径 $\\lambda$ の球の中の格子点」についての和になる。"
               "$d=1$ なら $\\hat f(m)=\\dfrac{\\sin 2\\pi am}{\\pi m}$ で, ふつうのフーリエ級数。")

    st.subheader("3つの現象")
    st.markdown(r"""
| 現象 | 次元 | どこで | 何が起こるか |
|---|---|---|---|
| ギブス–ウィルブラハム現象 | $d\le4$ | 球面 $\lvert x\rvert=a$ のそば | 跳びの約 9% の飛び出し |
| ピンスキー現象 | $d\ge3$ | 球の中心 | $S_\lambda f(0)-1$ が振幅 $P_d\lambda^{(d-3)/2}$ で振動 |
| **倉坪現象** | $d\ge5$ | すべての有理点 | $S_\lambda f(x)-f(x)$ が $0$ に近づかない($d\ge6$ では $\lambda^{(d-5)/2}$ で大きくなる) |
""")
    st.caption("$P_d=\\dfrac{a^{(d-3)/2}\\pi^{(d-4)/2}}{\\Gamma(d/2)}$($d=3$ では $\\frac{2}{\\pi}$)。一方 $d\\ge4$ では, "
               "ほとんどすべての点(無理点の大部分)で $S_\\lambda f(x)\\to f(x)$ となる。有理点は稠密なので, "
               "「収束しない点がいたるところにある」のに「ほとんどすべての点で収束する」。")

    c = st.columns(3)
    d = c[0].selectbox("次元 d", [1, 2, 3, 4, 5, 6, 7], index=4, key="ku_d")
    a = c[1].select_slider("球の半径 a", [0.15, 0.2, 0.25, 0.3], value=0.25, key="ku_a")
    lam = c[2].slider("λ の上限", 10, 80, 50, 5, key="ku_lam")
    pts = points_for(d)
    sel = st.multiselect("調べる点", list(pts), default=[k for k in pts if "有理点 (1/2" in k or "無理点" in k], key="ku_p")
    norm = st.radio("縦軸", ["誤差 S_λ f(x) − f(x)", "誤差 ÷ λ^{(d−5)/2}"], horizontal=True, key="ku_n")
    fig = go.Figure()
    colors = [C2, C1, C4, C3]
    for i, name in enumerate(sel or list(pts)[1:2]):
        xs = pts[name]
        s = torus_partial(xs, a, float(lam))
        L = np.sqrt(np.arange(len(s), dtype=float))
        e = s - torus_f(xs, a)
        if norm.startswith("誤差 ÷"):
            e = e / np.maximum(L, 1) ** ((d - 5) / 2)
        fig.add_trace(go.Scatter(x=L[1:], y=e[1:], mode="lines", line=dict(color=colors[i % 4], width=1.4, shape="hv"),
                                 name=name))
        if name.startswith("中心") and d >= 3 and not norm.startswith("誤差 ÷"):
            env = pinsky_amp(d, a) * L ** ((d - 3) / 2)
            for sg in (1, -1):
                fig.add_trace(go.Scatter(x=L[1:], y=sg * env[1:], mode="lines", line=dict(color=GRAY, dash="dot"),
                                         name="±P_d λ^{(d−3)/2}", showlegend=sg > 0))
    fig.add_hline(y=0, line=dict(color="black", width=1))
    layout(fig, title=f"d = {d}, a = {a}:球面部分和の誤差", xlab="λ", height=420)
    show(fig)
    st.caption("横軸 λ を増やすと, 半径 λ の球に入る格子点が増える(格子点が入るたびに値がとぶので階段状になる)。"
               "d = 5 で有理点と無理点を比べると, 無理点の誤差は 0 に近づくが, 有理点の誤差は小さくならない。d = 4 では有理点でも 0 に近づく。")

    st.subheader("3つの現象を1本の線で見る")
    st.markdown("点 $x=(t,0,\\ldots,0)$ を $0\\le t\\le\\tfrac12$ で動かして, $S_\\lambda f$ をグラフにする。")
    c2 = st.columns(2)
    d2 = c2[0].selectbox("次元 d", [1, 2, 3, 4, 5], index=2, key="ku_d2")
    lam2 = c2[1].slider("λ", 10, 60, 30, 1, key="ku_lam2")
    ts, vals = axis_profile(d2, 0.25, float(lam2))
    fig = go.Figure(go.Scatter(x=ts, y=(ts < 0.25).astype(float), mode="lines", line=dict(color=GRAY, width=2), name="f"))
    fig.add_trace(go.Scatter(x=ts, y=vals, mode="lines", line=dict(color=C1), name=f"S_λ f(t, 0, …, 0)"))
    fig.add_vline(x=0.25, line=dict(color=GRAY, dash="dot"), annotation_text="球面 t = a")
    layout(fig, title=f"d = {d2}, a = 1/4, λ = {lam2}", xlab="t", height=360)
    show(fig)
    st.caption("$t=a$ のそばの飛び出しがギブス現象, $t=0$ で $1$ から大きくずれるのがピンスキー現象($d\\ge3$)。"
               "$t=\\frac12$ は有理点で, $d=5$ では λ を変えてもずれが残る。")

    with st.expander("なぜ有理点で収束しないのか(格子点の問題)"):
        st.markdown("$\\hat f(m)$ は $|m|$ だけで決まるので, $|m|^2=k$ となる格子点の数 $r_d(k)$ をまとめて")
        st.latex(r"S_\lambda f(0)=\sum_{k<\lambda^2}r_d(k)\,\hat f(\sqrt k)")
        st.markdown("と書ける。$x$ が有理点なら $e^{2\\pi i m\\cdot x}$ は周期的にしか変わらないので, 同じような和になる。"
                    "$d\\ge5$ では $r_d(k)$ が $k^{d/2-1}$ に比例して**規則正しく**増える(下の図)ので, $\\hat f$ の振動が打ち消し合わず, 誤差が残る。"
                    "$d=2$ の $r_2(k)$ は $0$ になる $k$ も多く, 不規則である(ガウスの円の問題)。")
        dd = st.selectbox("格子点の数 r_d(k) を見る次元", [2, 3, 4, 5, 6], index=3, key="ku_rd")
        K = 400
        rk = lattice_weights(K, [0.0] * dd)
        k = np.arange(1, K + 1)
        fig = go.Figure(go.Bar(x=k, y=rk[1:] / k ** (dd / 2 - 1), marker_color=C1, name="r_d(k)/k^{d/2−1}"))
        layout(fig, title=f"d = {dd}:r_d(k) ÷ k^(d/2−1)", xlab="k = |m|²", height=300, legend=False)
        show(fig)
        st.caption("$d=5,\\ 6$ ではほぼ一定の幅に収まる。$d=2,\\ 3$ では大きくばらつく。"
                   "このように, 多次元フーリエ級数の収束は, 格子点の数え上げ(格子点問題)と深く結びついている。")

    with st.expander("文献"):
        st.markdown("- S. Kuratsubo, J. Fourier Anal. Appl. 16 (2010), 52–59.\n"
                    "- S. Kuratsubo, E. Nakai, K. Ootsubo, Generalized Hardy identity and relations to Gibbs–Wilbraham "
                    "and Pinsky phenomena, J. Funct. Anal. 259 (2010), 315–342.\n"
                    "- S. Kuratsubo, E. Nakai, Multiple Fourier series and lattice point problems, J. Funct. Anal. 282 (2022), 109272.")
        st.caption("図の数値は, 格子点の和をテータ級数 $\\sum_n\\cos(2\\pi nx_j)q^{n^2}$ の積として計算している。")


@st.cache_data
def axis_profile(d, a, lam, nt=241):
    """S_λ f(t, 0, …, 0) = Σ_n cos(2πnt) A_n, A_n = Σ_j f̂(√(j+n²)) w_0^{(d−1)}(j)"""
    K = int(lam * lam)
    base = lattice_weights(K, [0.0] * (d - 1))
    fh = ball_fhat(K, a, d)
    ns = np.arange(0, math.isqrt(K) + 1)
    A = np.array([float(np.dot(fh[n * n:], base[:K + 1 - n * n])) for n in ns])
    ts = np.linspace(0, 0.5, nt)
    mult = np.where(ns == 0, 1.0, 2.0)
    vals = np.cos(2 * PI * np.outer(ts, ns)) @ (mult * A)
    return ts, vals
