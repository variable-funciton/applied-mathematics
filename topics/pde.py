"""⑤ 偏微分方程式:熱方程式・差分法の安定性・波動方程式・ラプラス方程式"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import C1, C2, C3, C4, GRAY, animate, layout, show, trapz

PI = np.pi
XS = np.linspace(0, 1, 20001)  # 係数計算用

HEAT_INIT = {
    "三角形(中央が高い)": lambda x: 1 - np.abs(2 * x - 1),
    "中央だけ高温": lambda x: (np.abs(x - 0.5) < 0.1).astype(float),
    "一様 u = 1": lambda x: np.ones_like(x),
    "sin πx": lambda x: np.sin(PI * x),
    "sin πx + 0.5 sin 5πx": lambda x: np.sin(PI * x) + 0.5 * np.sin(5 * PI * x),
}


def sine_coeffs(f, N):
    """[0,1] 上のフーリエ正弦係数 b_n = 2∫ f sin nπx dx"""
    fx = f(XS)
    n = np.arange(1, N + 1)[:, None]
    b = 2 * trapz(fx * np.sin(n * PI * XS), XS, axis=1)
    b[np.abs(b) < 1e-10] = 0
    return b




# ---------- 熱方程式 ----------
def tab_heat():
    st.info("両端を 0℃ に保った棒の温度分布。変数分離で解くと、"
            "初期温度を**フーリエ正弦級数**に展開し、各成分が $e^{-\\kappa(n\\pi)^2t}$ で減衰する形になる。")
    st.latex(r"\frac{\partial u}{\partial t}=\kappa\frac{\partial^2u}{\partial x^2}\ (0<x<1),\quad u(0,t)=u(1,t)=0,"
             r"\quad u(x,0)=f(x)")
    st.latex(r"u(x,t)=\sum_{n=1}^{\infty}b_n\,e^{-\kappa(n\pi)^2t}\sin n\pi x,\qquad b_n=2\int_0^1f(x)\sin n\pi x\,dx")
    c1, c2, c3 = st.columns(3)
    name = c1.selectbox("初期温度 f(x)", list(HEAT_INIT), key="ph_f")
    kappa = c2.slider("熱拡散率 κ", 0.1, 2.0, 1.0, 0.1, key="ph_k")
    N = c3.slider("級数の項数 N", 1, 80, 40, key="ph_N")
    f = HEAT_INIT[name]
    b = sine_coeffs(f, N)
    n = np.arange(1, N + 1)

    x = np.linspace(0, 1, 401)
    T_end = 3 / (kappa * PI ** 2)
    times = T_end * (np.linspace(0, 1, 61) ** 2)
    S = np.sin(np.outer(n, PI * x))

    def u(t):
        return (b * np.exp(-kappa * (n * PI) ** 2 * t)) @ S

    frames = [[u(t)] for t in times]
    fig = animate(x, frames, ["u(x, t)"], [f"{t:.4f}" for t in times], colors=[C2],
                  static=[go.Scatter(x=x, y=f(x), mode="lines", line=dict(color=GRAY, dash="dot"),
                                     name="初期温度 f(x)")],
                  xlab="x", ylab="温度 u", yrange=(-0.2, 1.6), title="温度分布の時間変化(▶ で再生)")
    show(fig)

    cA, cB = st.columns(2)
    with cA:
        rows = [{"n": int(k), "b_n": f"{bk:.4f}", "減衰率 κ(nπ)²": f"{kappa * (k * PI) ** 2:.1f}",
                 "半減時間": f"{np.log(2) / (kappa * (k * PI) ** 2):.5f}"}
                for k, bk in zip(n[:8], b[:8])]
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption("n が大きい(細かい凹凸の)成分ほど n² に比例して速く消える。"
                   "そのため温度分布はすぐになめらかになり、最後は sin πx の形で 0 に近づく。")
    with cB:
        tt = np.linspace(0, T_end, 120)
        H = np.array([u(t) for t in tt])
        fig = go.Figure(go.Heatmap(x=x, y=tt, z=H, colorscale="Inferno", zmin=0,
                                   colorbar=dict(title="u", thickness=12)))
        layout(fig, title="x–t 平面で見た温度", xlab="x", ylab="t", height=360)
        show(fig)

    with st.expander("変数分離の流れ"):
        st.latex(r"u=X(x)T(t)\ \Rightarrow\ \frac{T'}{\kappa T}=\frac{X''}{X}=-\lambda")
        st.latex(r"X''+\lambda X=0,\ X(0)=X(1)=0\ \Rightarrow\ \lambda=(n\pi)^2,\ X=\sin n\pi x")
        st.latex(r"T'=-\kappa(n\pi)^2T\ \Rightarrow\ T=e^{-\kappa(n\pi)^2t}")
        st.write("これらを重ね合わせ、$t=0$ で $f(x)$ になるように係数を決める(③フーリエ級数)。")
    with st.expander("練習問題"):
        st.markdown("**問1** $f(x)=\\sin\\pi x+0.5\\sin 5\\pi x$ のときの解を書け。")
        st.markdown("**問2** f(x) = 1(一様)のとき $b_n$ を求めよ。")
        if st.toggle("解答・解説を表示", key="pr_pde_82"):
            st.markdown("**答1**")
            st.latex(r"u=e^{-\kappa\pi^2t}\sin\pi x+0.5\,e^{-25\kappa\pi^2t}\sin 5\pi x")
            st.write("第2項は第1項の 25 倍の速さで減衰する。")
            st.markdown("**答2**")
            st.latex(r"b_n=2\int_0^1\sin n\pi x\,dx=\frac{2(1-(-1)^n)}{n\pi}=\begin{cases}\frac{4}{n\pi}&(n\ \text{奇数})\\0&(n\ \text{偶数})\end{cases}")


# ---------- 差分法 ----------
def tab_ftcs():
    st.info("熱方程式をコンピュータで解く最も簡単な方法(陽解法)。"
            "時間の刻みを大きくしすぎると、解が振動して爆発する(**数値的不安定**)。")
    st.latex(r"\frac{u_j^{k+1}-u_j^k}{\Delta t}=\kappa\frac{u_{j+1}^k-2u_j^k+u_{j-1}^k}{\Delta x^2}"
             r"\ \Longrightarrow\ u_j^{k+1}=u_j^k+r\,(u_{j+1}^k-2u_j^k+u_{j-1}^k),\quad r=\frac{\kappa\Delta t}{\Delta x^2}")
    c1, c2 = st.columns(2)
    nx = c1.slider("分割数(Δx = 1/分割数)", 10, 40, 20, key="pf_nx")
    r = c2.slider("r = κΔt/Δx²", 0.10, 0.70, 0.40, 0.01, key="pf_r")
    kappa, t_end = 1.0, 0.06
    dx = 1 / nx
    dt = r * dx ** 2 / kappa
    steps = int(round(t_end / dt))
    x = np.linspace(0, 1, nx + 1)
    f = HEAT_INIT["三角形(中央が高い)"]
    U = f(x)
    snaps, snap_t = [U.copy()], [0.0]
    every = max(1, steps // 40)
    for k in range(1, steps + 1):
        U[1:-1] = U[1:-1] + r * (U[2:] - 2 * U[1:-1] + U[:-2])
        if k % every == 0 or k == steps:
            snaps.append(U.copy())
            snap_t.append(k * dt)

    nn = np.arange(1, 200)
    b = sine_coeffs(f, 199)
    # 厳密解(級数)を格子点の上で計算して比べる
    Sg = np.sin(np.outer(nn, PI * x))
    frames = [[s, (b * np.exp(-kappa * (nn * PI) ** 2 * t)) @ Sg] for s, t in zip(snaps, snap_t)]
    fig = animate(x, frames, ["差分法", "厳密解(級数)"], [f"{t:.4f}" for t in snap_t],
                  colors=[C2, C1], dashes=["solid", "dot"], xlab="x", ylab="u", yrange=(-0.6, 1.3),
                  title=f"Δt = {dt:.2e}、ステップ数 {steps}(▶ で再生)",
                  modes=["lines+markers", "lines"])
    show(fig)

    err = np.max(np.abs(snaps[-1] - frames[-1][1]))
    m1, m2, m3 = st.columns(3)
    m1.metric("r", f"{r:.2f}", "安定" if r <= 0.5 else "不安定", delta_color="normal" if r <= 0.5 else "inverse")
    m2.metric(f"t = {snap_t[-1]:.3f} での最大誤差", f"{err:.3e}")
    m3.metric("max |u|", f"{np.max(np.abs(snaps[-1])):.3e}")
    if r > 0.5:
        st.error("r > 1/2 なので不安定です。最も細かいギザギザ(格子点ごとに符号が変わる成分)が"
                 "1ステップごとに |1 − 4r| 倍に増幅され、やがて解を壊します。")
    else:
        st.success("r ≤ 1/2 なので安定。Δx を細かくすると、厳密解との誤差が小さくなる。")

    th = np.linspace(0, PI, 300)
    fig = go.Figure(go.Scatter(x=th, y=1 - 4 * r * np.sin(th / 2) ** 2, mode="lines",
                               line=dict(color=C2), name="増幅率 G"))
    fig.add_hrect(y0=-1, y1=1, fillcolor="rgba(44,160,44,0.12)", line_width=0)
    fig.update_xaxes(tickvals=[0, PI / 2, PI], ticktext=["0(なめらか)", "π/2", "π(ギザギザ)"])
    layout(fig, title="フォン・ノイマンの安定性解析:G(θ) = 1 − 4r sin²(θ/2)", xlab="波数 θ = kΔx",
           ylab="G", height=300, yrange=(-2, 1.2), legend=False)
    show(fig)
    st.caption("緑の帯 |G| ≤ 1 の中に収まれば安定。θ = π で G = 1 − 4r なので、条件は r ≤ 1/2。")

    with st.expander("練習問題"):
        st.markdown("**問3** κ = 1、Δx = 0.05 のとき、陽解法が安定になる Δt の最大値は?")
        st.markdown("**問4** Δx を半分にすると、同じ時刻まで計算するのに必要なステップ数は何倍になるか。")
        if st.toggle("解答・解説を表示", key="pr_pde_146"):
            st.markdown("**答3**")
            st.latex(r"\Delta t\le\frac{\Delta x^2}{2\kappa}=\frac{0.0025}{2}=0.00125")
            st.markdown("**答4**")
            st.write("Δt ∝ Δx² なので Δt は 1/4、ステップ数は **4 倍**(格子点も 2 倍なので計算量は 8 倍)。")


# ---------- 波動方程式 ----------
def tab_wave():
    st.info("両端を固定した弦の振動。各成分 $\\sin n\\pi x$ は減衰せずに振動し続ける(定在波)。"
            "ダランベールの解では、**右へ進む波と左へ進む波の和**として見える。")
    st.latex(r"\frac{\partial^2u}{\partial t^2}=\frac{\partial^2u}{\partial x^2},\quad u(0,t)=u(1,t)=0,\quad "
             r"u(x,0)=f(x),\ u_t(x,0)=0")
    st.latex(r"u(x,t)=\sum_{n=1}^{\infty}b_n\cos n\pi t\,\sin n\pi x=\frac12\left\{F(x-t)+F(x+t)\right\}")
    c1, c2 = st.columns(2)
    kind = c1.selectbox("初期の形 f(x)", ["弦をはじく(三角形)", "固有振動 sin nπx", "ガウス波束"], key="pw_f")
    if kind.startswith("弦"):
        p = c2.slider("はじく位置 p", 0.1, 0.9, 0.3, 0.05, key="pw_p")
        f = lambda x: np.where(x < p, x / p, (1 - x) / (1 - p))  # noqa: E731
    elif kind.startswith("固有"):
        m = c2.slider("n", 1, 6, 2, key="pw_n")
        f = lambda x: np.sin(m * PI * x)  # noqa: E731
    else:
        x0 = c2.slider("波束の中心", 0.15, 0.85, 0.3, 0.05, key="pw_x0")
        f = lambda x: np.exp(-((x - x0) / 0.05) ** 2)  # noqa: E731
    dal = st.checkbox("右へ進む波・左へ進む波(ダランベール)も表示", value=True, key="pw_dal")

    N = 100
    b = sine_coeffs(f, N)
    n = np.arange(1, N + 1)
    x = np.linspace(0, 1, 401)
    S = np.sin(np.outer(n, PI * x))

    def F(s):  # 奇関数・周期 2 に拡張
        y = np.mod(s, 2)
        return np.where(y <= 1, f(np.minimum(y, 1)), -f(np.clip(2 - y, 0, 1)))

    times = np.linspace(0, 2, 81)
    frames = []
    for t in times:
        u = (b * np.cos(n * PI * t)) @ S
        frames.append([u, 0.5 * F(x - t), 0.5 * F(x + t)] if dal else [u])
    names = ["u(x, t)(級数解)", "½F(x − t):右へ進む", "½F(x + t):左へ進む"] if dal else ["u(x, t)"]
    fig = animate(x, frames, names, [f"{t:.3f}" for t in times], colors=[C2, C1, C3],
                  dashes=["solid", "dash", "dash"], xlab="x", ylab="u", yrange=(-1.2, 1.2),
                  static=[go.Scatter(x=x, y=f(x), mode="lines", line=dict(color=GRAY, dash="dot"),
                                     name="初期の形")],
                  title="弦の振動(t = 0〜2 で1周期、▶ で再生)", duration=80)
    show(fig)

    fig = go.Figure(go.Bar(x=n[:20], y=np.abs(b[:20]), marker_color=C1))
    layout(fig, title="各固有振動の大きさ |b_n|(倍音の強さ)", xlab="n", height=280, legend=False)
    show(fig)
    if kind.startswith("弦"):
        st.latex(r"b_n=\frac{2\sin n\pi p}{(n\pi)^2p(1-p)}")
        st.caption("はじく位置 p = 1/2 のとき、偶数番目の倍音(sin nπp = 0)が消える。"
                   "ギターで弾く位置を変えると音色が変わるのはこのため。")

    with st.expander("練習問題"):
        st.markdown("**問5** 長さ 1、波の速さ 1 の弦の、第 n 固有振動の周期を求めよ。")
        st.markdown("**問6** $u=\\frac12\\{F(x-t)+F(x+t)\\}$ が波動方程式を満たすことを確かめよ。")
        if st.toggle("解答・解説を表示", key="pr_pde_204"):
            st.markdown("**答5**")
            st.write("$\\cos n\\pi t$ の周期なので $\\dfrac{2}{n}$。基本振動(n = 1)の周期は 2。")
            st.markdown("**答6**")
            st.latex(r"u_{tt}=\frac12\{F''(x-t)+F''(x+t)\}=u_{xx}")


# ---------- ラプラス方程式 ----------
LAP_TOP = {
    "一定 g(x) = 1": lambda x: np.ones_like(x),
    "g(x) = sin πx": lambda x: np.sin(PI * x),
    "g(x) = 4x(1 − x)": lambda x: 4 * x * (1 - x),
    "左半分だけ g = 1": lambda x: (x < 0.5).astype(float),
}


def tab_laplace():
    st.info("十分時間がたった後の温度分布(定常状態)。上の辺だけ温度 g(x)、他の3辺は 0℃ の板。")
    st.latex(r"\frac{\partial^2u}{\partial x^2}+\frac{\partial^2u}{\partial y^2}=0,\qquad "
             r"u(x,b)=g(x),\ \ u=0\ (\text{他の3辺})")
    st.latex(r"u(x,y)=\sum_{n=1}^{\infty}c_n\frac{\sinh n\pi y}{\sinh n\pi b}\sin n\pi x,\qquad c_n=2\int_0^1g(x)\sin n\pi x\,dx")
    c1, c2 = st.columns(2)
    name = c1.selectbox("上の辺の温度 g(x)", list(LAP_TOP), key="pl_g")
    bH = c2.slider("板の高さ b", 0.5, 2.0, 1.0, 0.1, key="pl_b")
    N = 120
    cn = sine_coeffs(LAP_TOP[name], N)
    n = np.arange(1, N + 1)

    def U(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        k = (n * PI)[:, None]
        ratio = np.exp(k * (y.ravel()[None, :] - bH)) * (1 - np.exp(-2 * k * y.ravel()[None, :])) \
            / (1 - np.exp(-2 * k * bH))
        val = np.sum(cn[:, None] * ratio * np.sin(k * x.ravel()[None, :]), axis=0)
        return val.reshape(x.shape)

    xs = np.linspace(0, 1, 101)
    ys = np.linspace(0, bH, int(100 * bH) + 1)
    X, Y = np.meshgrid(xs, ys)
    Z = U(X, Y)

    d1, d2, d3 = st.columns(3)
    x0 = d1.slider("円の中心 x", 0.1, 0.9, 0.5, 0.05, key="pl_x0")
    y0 = d2.slider("円の中心 y", 0.1, float(bH) - 0.1, float(min(0.5, bH - 0.1)), 0.05, key=f"pl_y0_{bH:.1f}")
    rmax = float(min(x0, 1 - x0, y0, bH - y0)) - 0.01
    r = d3.slider("円の半径", 0.02, max(rmax, 0.03), min(0.2, max(rmax, 0.03)), 0.01,
                  key=f"pl_r_{rmax:.2f}")
    th = np.linspace(0, 2 * PI, 721)[:-1]
    mean = float(np.mean(U(x0 + r * np.cos(th), y0 + r * np.sin(th))))
    center = float(U(np.array([x0]), np.array([y0]))[0])

    fig = go.Figure(go.Contour(x=xs, y=ys, z=Z, colorscale="Inferno", ncontours=20,
                               contours=dict(coloring="heatmap", showlabels=True,
                                             labelfont=dict(color="white", size=10)),
                               colorbar=dict(title="u", thickness=12), zmin=0, zmax=1))
    tt = np.linspace(0, 2 * PI, 200)
    fig.add_trace(go.Scatter(x=x0 + r * np.cos(tt), y=y0 + r * np.sin(tt), mode="lines",
                             line=dict(color=C3, width=3), name="円"))
    fig.add_trace(go.Scatter(x=[x0], y=[y0], mode="markers", marker=dict(color=C3, size=9), name="中心"))
    layout(fig, height=520, equal=True, xlab="x", ylab="y", legend=False)
    cA, cB = st.columns([3, 2])
    with cA:
        show(fig)
    with cB:
        st.markdown("**平均値の性質**")
        st.latex(r"u(x_0,y_0)=\frac{1}{2\pi}\int_0^{2\pi}u(x_0+r\cos\theta,\ y_0+r\sin\theta)\,d\theta")
        st.metric("中心の値 u(x₀, y₀)", f"{center:.5f}")
        st.metric("円周上の平均", f"{mean:.5f}")
        st.caption("調和関数(ラプラス方程式の解)は、どの円でも「中心の値 = 円周上の平均」。"
                   "このため内部に最大・最小をもたない(最大値原理):一番熱い所・冷たい所は必ず境界にある。")

    with st.expander("練習問題"):
        st.markdown("**問7** g(x) = sin πx、b = 1 のときの解を書け。")
        st.markdown("**問8** $u=x^2-y^2$ がラプラス方程式を満たすことを確かめ、"
                    "原点中心・半径 r の円周上での平均を求めよ。")
        if st.toggle("解答・解説を表示", key="pr_pde_275"):
            st.markdown("**答7**")
            st.latex(r"u(x,y)=\frac{\sinh\pi y}{\sinh\pi}\sin\pi x")
            st.markdown("**答8**")
            st.latex(r"u_{xx}+u_{yy}=2-2=0,\qquad \frac{1}{2\pi}\int_0^{2\pi}r^2\cos 2\theta\,d\theta=0=u(0,0)")
