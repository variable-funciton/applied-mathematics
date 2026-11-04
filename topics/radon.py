"""③ フーリエ解析 発展:ラドン変換(R² と R³)

R²:Rf(θ, s) = ∫_{x·ω = s} f dℓ,  ω = (cos θ, sin θ)
    投影切断定理  ∫ Rf(θ, s) e^{−iσs} ds = F(σω)(F は2次元フーリエ変換 F(ξ) = ∫ f(x) e^{−ix·ξ} dx)
    逆変換(フィルタ補正逆投影)  f(x) = (1/2π) ∫_0^π (Λ Rf)(θ, x·ω) dθ,  Λ はフーリエ乗数 |σ|
R³:Rf(ω, s) = ∫_{x·ω = s} f dS,  ω ∈ S²
    逆変換(局所的)  f(x) = −(1/8π²) ∫_{S²} ∂²_s Rf(ω, x·ω) dω
図形は楕円(R²)と球(R³)の組合せなので, ラドン変換は式で正確に計算できる。
"""
import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.special import j1

from common import C1, C2, C3, C4, GRAY, layout, show
from topics.fourier_kernels import page_refs

PI = np.pi

# (中心 x0, y0, 半軸 a, b, 回転角[度], 密度)
PHANTOMS = {
    "円板(半径 0.6)": [(0, 0, 0.6, 0.6, 0, 1.0)],
    "2つの円板": [(-0.4, 0.1, 0.3, 0.3, 0, 1.0), (0.45, -0.2, 0.2, 0.2, 0, 0.6)],
    "頭部を模した図形(楕円の組合せ)": [(0, 0, 0.69, 0.92, 0, 1.0), (0, -0.0184, 0.6624, 0.874, 0, -0.8),
                             (0.22, 0, 0.11, 0.31, -18, -0.2), (-0.22, 0, 0.16, 0.41, 18, -0.2),
                             (0, 0.35, 0.21, 0.25, 0, 0.1), (0, -0.605, 0.046, 0.023, 0, 0.3)],
    "細長い楕円(斜め)": [(0.1, 0.05, 0.75, 0.15, 30, 1.0)],
}
NG = 161
GRID = np.linspace(-1, 1, NG)


def phantom_image(ells, X=None, Y=None):
    if X is None:
        X, Y = np.meshgrid(GRID, GRID)
    img = np.zeros_like(X, dtype=float)
    for x0, y0, a, b, ph, rho in ells:
        c, s_ = math.cos(math.radians(ph)), math.sin(math.radians(ph))
        u = (X - x0) * c + (Y - y0) * s_
        v = -(X - x0) * s_ + (Y - y0) * c
        img += rho * ((u / a) ** 2 + (v / b) ** 2 <= 1)
    return img


def radon_ellipses(ells, th, s):
    """楕円の組合せのラドン変換(th: 角度の配列, s: 位置の配列)→ 形 (len(th), len(s))"""
    T, S = np.meshgrid(th, s, indexing="ij")
    out = np.zeros_like(T)
    for x0, y0, a, b, ph, rho in ells:
        phr = math.radians(ph)
        r2 = (a * np.cos(T - phr)) ** 2 + (b * np.sin(T - phr)) ** 2
        sp = S - (x0 * np.cos(T) + y0 * np.sin(T))
        out += np.where(sp ** 2 < r2, 2 * rho * a * b * np.sqrt(np.maximum(r2 - sp ** 2, 0)) / r2, 0.0)
    return out


def ft_ellipses(ells, xi1, xi2):
    """楕円の組合せの2次元フーリエ変換 F(ξ) = ∫ f e^{−ix·ξ} dx"""
    out = np.zeros(np.broadcast(xi1, xi2).shape, dtype=complex)
    for x0, y0, a, b, ph, rho in ells:
        c, s_ = math.cos(math.radians(ph)), math.sin(math.radians(ph))
        q = np.hypot(a * (xi1 * c + xi2 * s_), b * (-xi1 * s_ + xi2 * c))
        core = np.where(q > 1e-9, 2 * j1(np.maximum(q, 1e-12)) / np.maximum(q, 1e-12), 1.0)
        out += rho * PI * a * b * core * np.exp(-1j * (x0 * xi1 + y0 * xi2))
    return out


def ramlak(n, ds):
    """ランプフィルタ(ラマチャンドラン–ラクシュミナラヤナン)の離散核 h[k](カク–スレイニー)"""
    k = np.arange(-n, n + 1)
    h = np.zeros(len(k))
    h[k == 0] = 1 / (4 * ds ** 2)
    odd = k % 2 == 1
    h[odd] = -1 / (PI ** 2 * k[odd] ** 2 * ds ** 2)
    return h


@st.cache_data
def reconstruct(name, n_theta, filtered=True, ns=321, smax=1.5):
    ells = PHANTOMS[name]
    th = np.arange(n_theta) * PI / n_theta
    s = np.linspace(-smax, smax, ns)
    ds = s[1] - s[0]
    P = radon_ellipses(ells, th, s)
    if filtered:
        h = ramlak(ns - 1, ds)
        P = np.array([np.convolve(p, h)[ns - 1:2 * ns - 1] * ds for p in P])   # h の中心(k = 0)が添字 ns − 1
    X, Y = np.meshgrid(GRID, GRID)
    img = np.zeros_like(X)
    for i, t in enumerate(th):
        img += np.interp(X * math.cos(t) + Y * math.sin(t), s, P[i], left=0, right=0)
    img *= PI / n_theta                         # dθ
    return img if filtered else img / (2 * PI)


# ---------- R³ ----------
BALLS = {"球 1つ(半径 0.6)": [((0, 0, 0), 0.6, 1.0)],
         "球 2つ": [((-0.35, 0.1, 0.0), 0.35, 1.0), ((0.4, -0.2, 0.2), 0.25, 0.7)]}


def radon3_balls(balls, w, s):
    """球の組合せの3次元ラドン変換:平面 x·ω = s での断面(円板)の面積 × 密度"""
    out = np.zeros_like(np.asarray(s, float))
    for c, a, rho in balls:
        sp = s - np.dot(c, w)
        out = out + rho * PI * np.maximum(a * a - sp ** 2, 0.0)
    return out


def fib_sphere(n):
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(1 - z * z)
    ph = PI * (1 + 5 ** 0.5) * i
    return np.stack([r * np.cos(ph), r * np.sin(ph), z], axis=1)


def inverse3_at(balls, x, n=60000, h=0.01):
    """f(x) = −(1/8π²) ∫_{S²} ∂²_s Rf(ω, x·ω) dω を数値で(2階差分 + 球面上の一様な点)"""
    W = fib_sphere(n)
    s0 = W @ np.asarray(x, float)
    tot = np.zeros(n)
    for c, a, rho in balls:
        sp = s0 - W @ np.asarray(c, float)
        g = lambda u: rho * PI * np.maximum(a * a - u ** 2, 0.0)  # noqa: E731
        tot += (g(sp + h) - 2 * g(sp) + g(sp - h)) / h ** 2
    return float(-(4 * PI / n) * tot.sum() / (8 * PI ** 2))


def sphere_mesh(c, a, n=24):
    u, v = np.meshgrid(np.linspace(0, 2 * PI, n), np.linspace(0, PI, n))
    return c[0] + a * np.cos(u) * np.sin(v), c[1] + a * np.sin(u) * np.sin(v), c[2] + a * np.cos(v)


def tab_radon():
    st.info("ラドン変換は, 関数を**直線(平面)に沿って積分した値**の集まり。X 線 CT は, 体のいろいろな方向の X 線の弱まり方(直線に沿う積分)から, "
            "体の中の様子 $f$ を復元する。その数学的な基礎がラドン(1917)の逆変換公式で, フーリエ変換と深く結びついている。")
    st.subheader("R² のラドン変換")
    st.markdown("方向 $\\omega=(\\cos\\theta,\\ \\sin\\theta)$ に垂直で, 原点からの符号つき距離が $s$ の直線 $\\{x\\cdot\\omega=s\\}$ に沿って $f$ を積分する:")
    st.latex(r"Rf(\theta,s)=\int_{x\cdot\omega=s}f\,d\ell=\int_{-\infty}^{\infty}f(s\,\omega+t\,\omega^{\perp})\,dt,\qquad "
             r"\omega^\perp=(-\sin\theta,\ \cos\theta)")
    st.markdown("例:半径 $a$ の円板の特性関数なら, 直線が円板を切る長さなので $Rf(\\theta,s)=2\\sqrt{a^2-s^2}\\ (|s|<a)$。"
                "$(\\theta,s)$ を横軸・縦軸にして $Rf$ を表した図を**サイノグラム**という(点はサインカーブを描く)。")
    name = st.selectbox("図形 f", list(PHANTOMS), index=2, key="rd_f")
    ells = PHANTOMS[name]
    img = phantom_image(ells)
    th = np.linspace(0, PI, 181)
    sgrid = np.linspace(-1.5, 1.5, 241)
    sino = radon_ellipses(ells, th, sgrid)
    cA, cB = st.columns(2)
    with cA:
        fig = go.Figure(go.Heatmap(x=GRID, y=GRID, z=img, colorscale="Gray", showscale=False))
        layout(fig, title="f(x, y)", xlab="x", ylab="y", height=360, equal=True, legend=False)
        show(fig)
    with cB:
        fig = go.Figure(go.Heatmap(x=np.degrees(th), y=sgrid, z=sino.T, colorscale="Gray", colorbar=dict(title="Rf")))
        layout(fig, title="サイノグラム Rf(θ, s)", xlab="θ [度]", ylab="s", height=360, legend=False)
        show(fig)

    st.subheader("投影切断定理(フーリエ変換との関係)")
    st.markdown("投影 $Rf(\\theta,\\cdot)$ を $s$ について1次元フーリエ変換すると, $f$ の2次元フーリエ変換を, 原点を通る方向 $\\omega$ の直線で切ったものになる:")
    st.latex(r"\int_{-\infty}^{\infty}Rf(\theta,s)\,e^{-i\sigma s}\,ds=\hat f(\sigma\omega)\qquad\left(\hat f(\xi)=\int_{\mathbb R^2}f(x)e^{-ix\cdot\xi}dx\right)")
    with st.expander("証明"):
        st.markdown("$x=s\\omega+t\\omega^\\perp$ と変数変換すると($dx=ds\\,dt$), $x\\cdot\\omega=s$ なので")
        st.latex(r"\hat f(\sigma\omega)=\int f(x)e^{-i\sigma x\cdot\omega}dx=\int_{-\infty}^{\infty}\left(\int_{-\infty}^{\infty}f(s\omega+t\omega^\perp)dt\right)e^{-i\sigma s}ds"
                 r"=\int Rf(\theta,s)e^{-i\sigma s}ds")
    tdeg = st.slider("投影の角度 θ [度]", 0, 179, 30, key="rd_th")
    t = math.radians(tdeg)
    s1 = np.linspace(-2, 2, 4001)
    p = radon_ellipses(ells, np.array([t]), s1)[0]
    sig = np.linspace(-40, 40, 401)
    Fp = (p[None, :] * np.exp(-1j * np.outer(sig, s1))).sum(axis=1) * (s1[1] - s1[0])
    Fs = ft_ellipses(ells, sig * math.cos(t), sig * math.sin(t))
    cA, cB = st.columns(2)
    with cA:
        fig = go.Figure(go.Scatter(x=s1, y=p, mode="lines", line=dict(color=C1, width=2), name="Rf(θ, s)"))
        layout(fig, title=f"投影 Rf(θ, s)(θ = {tdeg}°)", xlab="s", height=320, legend=False)
        show(fig)
    with cB:
        fig = go.Figure(go.Scatter(x=sig, y=np.abs(Fp), mode="lines", line=dict(color=C2, width=3), name="投影の1次元フーリエ変換"))
        fig.add_trace(go.Scatter(x=sig, y=np.abs(Fs), mode="lines", line=dict(color="black", width=1.5, dash="dot"),
                                 name="f̂(σω)(2次元フーリエ変換の切り口)"))
        layout(fig, title="大きさの比較", xlab="σ", height=320)
        show(fig)
    st.caption(f"最大のずれ:{float(np.max(np.abs(Fp - Fs))):.1e}。すべての方向の投影がわかれば, $\\hat f$ が平面全体でわかり, 逆フーリエ変換で $f$ が求まる。")

    st.subheader("逆変換:フィルタ補正逆投影法")
    st.markdown("投影切断定理と, 2次元の逆フーリエ変換を極座標 $\\xi=\\sigma\\omega$($d\\xi=|\\sigma|\\,d\\sigma\\,d\\theta$)で書くことから")
    st.latex(r"f(x)=\frac{1}{2\pi}\int_0^{\pi}(\Lambda Rf)(\theta,\ x\cdot\omega)\,d\theta,\qquad "
             r"\Lambda g=\frac{1}{2\pi}\int_{-\infty}^{\infty}|\sigma|\,\hat g(\sigma)e^{i\sigma s}d\sigma")
    st.markdown("$\\Lambda$ は周波数 $\\sigma$ に $|\\sigma|$ をかける**ランプフィルタ**。フィルタをかけずに各直線へ値を戻す(**逆投影**)だけでは, 像がぼける。")
    with st.expander("式の導き方"):
        st.latex(r"f(x)=\frac{1}{4\pi^2}\int_{\mathbb R^2}\hat f(\xi)e^{ix\cdot\xi}d\xi=\frac{1}{4\pi^2}\int_0^{\pi}\int_{-\infty}^{\infty}\hat f(\sigma\omega)e^{i\sigma x\cdot\omega}|\sigma|\,d\sigma\,d\theta")
        st.markdown("($\\theta$ を $0\\le\\theta<\\pi$, $\\sigma$ を $-\\infty<\\sigma<\\infty$ とすると平面全体を1回ずつ覆う)。"
                    "$\\hat f(\\sigma\\omega)$ を投影のフーリエ変換におきかえると上の式になる。")
    c = st.columns(2)
    nth = c[0].select_slider("投影の数(角度の数)", [4, 8, 16, 32, 64, 128, 180], value=32, key="rd_n")
    filt = c[1].radio("方法", ["フィルタ補正逆投影", "逆投影のみ(フィルタなし)"], horizontal=True, key="rd_m")
    rec = reconstruct(name, nth, filt.startswith("フィルタ"))
    cA, cB = st.columns(2)
    with cA:
        fig = go.Figure(go.Heatmap(x=GRID, y=GRID, z=rec, colorscale="Gray", colorbar=dict(title="値")))
        layout(fig, title=f"復元した像({nth} 方向, {filt})", xlab="x", ylab="y", height=380, equal=True, legend=False)
        show(fig)
    with cB:
        mid = NG // 2
        fig = go.Figure(go.Scatter(x=GRID, y=img[mid], mode="lines", line=dict(color=GRAY, width=2.5, shape="hv"), name="f(x, 0)"))
        fig.add_trace(go.Scatter(x=GRID, y=rec[mid], mode="lines", line=dict(color=C2), name="復元(y = 0)"))
        layout(fig, title="y = 0 の断面", xlab="x", height=380)
        show(fig)
    if filt.startswith("フィルタ"):
        st.write(f"2乗平均誤差 {math.sqrt(float(np.mean((rec - img) ** 2))):.3f}(投影の数を増やすと, 筋状の誤差が減る)")
    st.caption("ランプフィルタ $|\\sigma|$ は, 掛け算で書くと「すべての $s$ のデータ」を使う(非局所的)。"
               "2次元では, 1点の値を求めるのに, その点を通らない直線のデータも必要になる。")

    st.subheader("R³ のラドン変換")
    st.markdown("3次元では, 単位ベクトル $\\omega\\in S^2$ に垂直な**平面** $\\{x\\cdot\\omega=s\\}$ で積分する:")
    st.latex(r"Rf(\omega,s)=\int_{x\cdot\omega=s}f\,dS,\qquad \int_{-\infty}^{\infty}Rf(\omega,s)e^{-i\sigma s}ds=\hat f(\sigma\omega)\quad(\text{投影切断定理})")
    st.markdown("例:半径 $a$ の球の特性関数なら, 平面で切った断面(円板)の面積なので $Rf(\\omega,s)=\\pi(a^2-s^2)\\ (|s|<a)$。"
                "逆変換は**局所的**な式になる:")
    st.latex(r"f(x)=-\frac{1}{8\pi^2}\int_{S^2}\frac{\partial^2Rf}{\partial s^2}(\omega,\ x\cdot\omega)\,d\omega")
    st.markdown("$x$ を通る平面のデータ($s=x\\cdot\\omega$ のごく近く)だけで $f(x)$ が決まる。"
                "奇数次元では局所的, 偶数次元では非局所的になる(波動方程式のホイヘンスの原理と同じ理由)。")
    with st.expander("球の例で逆変換を確かめる"):
        st.markdown("$|x|<a$ なら, すべての $\\omega$ で $|x\\cdot\\omega|<a$ なので $\\dfrac{\\partial^2Rf}{\\partial s^2}=-2\\pi$。")
        st.latex(r"f(x)=-\frac{1}{8\pi^2}\int_{S^2}(-2\pi)\,d\omega=\frac{2\pi\cdot4\pi}{8\pi^2}=1")
    bname = st.selectbox("球の配置", list(BALLS), key="rd_b")
    balls = BALLS[bname]
    c = st.columns(3)
    th3 = math.radians(c[0].slider("ω の天頂角 [度]", 0, 180, 60, key="rd_t3"))
    ph3 = math.radians(c[1].slider("ω の方位角 [度]", 0, 360, 30, key="rd_p3"))
    s3 = c[2].slider("平面の位置 s", -1.0, 1.0, 0.1, 0.05, key="rd_s3")
    w = np.array([math.sin(th3) * math.cos(ph3), math.sin(th3) * math.sin(ph3), math.cos(th3)])
    cA, cB = st.columns(2)
    with cA:
        fig = go.Figure()
        for (cc, a, rho), col in zip(balls, ("Blues", "Reds")):
            X, Y, Z = sphere_mesh(cc, a)
            fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale=col, showscale=False, opacity=0.6))
        e1 = np.cross(w, [0, 0, 1.0]) if abs(w[2]) < 0.9 else np.cross(w, [1.0, 0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(w, e1)
        uu, vv = np.meshgrid(np.linspace(-1, 1, 2), np.linspace(-1, 1, 2))
        P0 = s3 * w
        fig.add_trace(go.Surface(x=P0[0] + uu * e1[0] + vv * e2[0], y=P0[1] + uu * e1[1] + vv * e2[1],
                                 z=P0[2] + uu * e1[2] + vv * e2[2], colorscale=[[0, "gold"], [1, "gold"]], showscale=False, opacity=0.45))
        fig.add_trace(go.Scatter3d(x=[0, w[0]], y=[0, w[1]], z=[0, w[2]], mode="lines", line=dict(color="black", width=6), name="ω"))
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=30, b=0), showlegend=False,
                          scene=dict(aspectmode="cube", xaxis=dict(range=[-1, 1]), yaxis=dict(range=[-1, 1]), zaxis=dict(range=[-1, 1])))
        show(fig)
    with cB:
        ss = np.linspace(-1.2, 1.2, 481)
        R = radon3_balls(balls, w, ss)
        fig = go.Figure(go.Scatter(x=ss, y=R, mode="lines", line=dict(color=C1, width=2.5), name="Rf(ω, s)"))
        fig.add_trace(go.Scatter(x=[s3], y=[float(radon3_balls(balls, w, np.array([s3]))[0])], mode="markers",
                                 marker=dict(color=C2, size=11), name="いまの平面"))
        layout(fig, title="Rf(ω, s)(断面の面積 × 密度)", xlab="s", height=420, legend=False)
        show(fig)
    xq = st.select_slider("逆変換で f を求める点 x = (t, 0, 0) の t", [-0.9, -0.5, -0.2, 0.0, 0.3, 0.5, 0.7, 0.9], value=0.0, key="rd_x")
    val = inverse3_at(balls, (xq, 0, 0))
    true = sum(rho for c_, a, rho in balls if np.linalg.norm(np.subtract((xq, 0, 0), c_)) < a)
    st.write(f"逆変換の式で数値計算した f({xq}, 0, 0) ≈ {val:.3f}(本当の値 {true:g})")
    st.caption("球面上の 6 万方向について, s 方向の2階差分で ∂²Rf/∂s² を計算して足し合わせた。")

    st.subheader("ラドン変換が使われている分野・研究")
    st.markdown(
        "- **医用画像**:X 線 CT(コーマックとハウンスフィールド, 1979 年ノーベル生理学・医学賞)。"
        "PET・SPECT では, 体内での減衰を含む「減衰つきラドン変換」の逆問題になる。3次元の円錐ビーム CT は, 3次元ラドン変換の微分と結びつく(グランジャ, 1991)。\n"
        "- **工業用 CT・非破壊検査**:部品の内部の欠陥や鋳物の空洞を, 壊さずに調べる。\n"
        "- **地球物理学**:地震波の伝わる時間から地球内部の速度の分布を求める地震波トモグラフィ。\n"
        "- **電子顕微鏡**:クライオ電子顕微鏡によるタンパク質の立体構造の決定は, 3次元の投影切断定理を使う。\n"
        "- **数学**:\n"
        "  - 積分幾何(直線・平面などの族の上の積分から関数を決める, ヘルガソン)\n"
        "  - 偏微分方程式(平面波への分解, 奇数次元でのホイヘンスの原理と局所的な逆変換)\n"
        "  - 調和解析(ラドン変換や X 線変換の $L^p$–$L^q$ 評価, フーリエ制限問題・掛谷問題との関係)\n"
        "  - 逆問題(限られた角度・少ない投影からの安定な再構成, 雑音への強さ)\n"
        "- **数値計算・画像処理**:フィルタ補正逆投影法・代数的再構成法, 少ないデータからの再構成(圧縮センシング), 深層学習による再構成。")
    with st.expander("練習問題"):
        st.markdown("**問1** $f(x)=e^{-|x|^2}$($x\\in\\mathbb R^2$)のラドン変換を求めよ。また, 投影切断定理を確かめよ。")
        st.markdown("**問2** 半径 $a$ の円板の特性関数で, $\\displaystyle\\int_{-\\infty}^{\\infty}Rf(\\theta,s)\\,ds$ を求めよ。これは何を表すか。")
        st.markdown("**問3** $\\mathbb R^3$ の半径 $a$ の球の特性関数について, $Rf(\\omega,s)=\\pi(a^2-s^2)\\ (|s|<a)$ を確かめよ。")
        if st.toggle("解答・解説を表示", key="pr_radon"):
            st.markdown("**答1** $f(s\\omega+t\\omega^\\perp)=e^{-s^2-t^2}$ なので $Rf(\\theta,s)=e^{-s^2}\\int e^{-t^2}dt=\\sqrt\\pi\\,e^{-s^2}$。"
                        "$s$ について1次元フーリエ変換すると $\\pi e^{-\\sigma^2/4}$。一方 $\\hat f(\\xi)=\\pi e^{-|\\xi|^2/4}$ なので, $\\xi=\\sigma\\omega$ で一致する。")
            st.markdown("**答2** $\\int_{-a}^{a}2\\sqrt{a^2-s^2}\\,ds=\\pi a^2$。どの方向でも $f$ の積分(円板の面積)になる($\\hat f(0)$:投影切断定理で $\\sigma=0$)。")
            st.markdown("**答3** 平面 $x\\cdot\\omega=s$ は中心から距離 $|s|$ なので, 断面は半径 $\\sqrt{a^2-s^2}$ の円板。面積は $\\pi(a^2-s^2)$。")
    page_refs([("radon1917", "ラドン変換と逆変換の原論文。"), ("cormack1963", "X 線 CT の数学的な基礎。"),
               ("nobel1979", "CT の開発に対するノーベル賞。"), ("natterer2001", "CT の数学(ラドン変換の理論と数値解法)。"),
               ("kakslaney2001", "フィルタ補正逆投影法などの再構成アルゴリズム(Web で読める)。"),
               ("epstein2007", "医用画像の数学(フーリエ解析・ラドン変換・CT・MRI)。"),
               ("helgason1999", "積分幾何としてのラドン変換。"),
               ("grangeat1991", "円錐ビーム CT と3次元ラドン変換。"),
               ("stein2007", "第6章(ℝ^d のフーリエ変換)の中で, ラドン変換と X 線 CT を扱っている。")])
