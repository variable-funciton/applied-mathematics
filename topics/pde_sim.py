"""⑤ 偏微分方程式:時間発展シミュレーション(熱伝導・波動)

境界条件と初期条件を選び、「シミュレーション開始」で時間とともに変化するようすを再生する。
- 熱伝導方程式:クランク・ニコルソン法(無条件安定。はじめの数ステップは後退オイラー法で、とがった初期値の振動を抑える)
- 波動方程式  :中心差分(リープフロッグ法)。クーラン数 ν = cΔt/Δx = 0.9(CFL 条件 ν ≤ 1 を満たす)
外力・周期的な境界がない場合は、フーリエ級数による厳密解と重ねて比べられる。
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import C1, C2, C3, GRAY, layout
from topics.player import player

PI = np.pi
NF = 121                         # 保存するコマ数
XF = np.linspace(0, 1, 4001)     # 係数計算用の細かい格子


def _trapz(y, x, axis=-1):
    f = getattr(np, "trapezoid", None) or np.trapz
    return f(y, x=x, axis=axis)


# ------------------------------------------------------------
# 初期条件
# ------------------------------------------------------------
def init_shape(name, x, x0=0.5, k=1):
    if name.startswith("パルス"):
        w = 0.08
        return np.where(np.abs(x - x0) < w, np.cos(PI * (x - x0) / (2 * w)) ** 2, 0.0)
    if name.startswith("矩形"):
        return (np.abs(x - x0) < 0.08).astype(float)
    if name.startswith("三角形"):
        return np.where(x < x0, x / max(x0, 1e-9), (1 - x) / max(1 - x0, 1e-9)).clip(0)
    if name.startswith("ガウス"):
        return np.exp(-((x - x0) / 0.06) ** 2)
    if name.startswith("正弦波"):
        return np.sin(k * PI * x)
    if name.startswith("一様"):
        return np.ones_like(x)
    return np.zeros_like(x)          # 静止


# ------------------------------------------------------------
# 固有関数(厳密解の級数用)
# ------------------------------------------------------------
def eigen(bc, K):
    """bc: "DD"(両端固定)/ "NN"(両端自由・断熱)/ "DN"(左固定・右自由)
    戻り値: (μ_k の配列, φ_k(x) を作る関数, 正規化 1/∫φ²)"""
    k = np.arange(K)
    if bc == "DD":
        mu = (k + 1) * PI
        return mu, (lambda m, x: np.sin(m[:, None] * x)), np.full(K, 2.0)
    if bc == "NN":
        mu = k * PI
        norm = np.full(K, 2.0)
        norm[0] = 1.0
        return mu, (lambda m, x: np.cos(m[:, None] * x)), norm
    mu = (k + 0.5) * PI
    return mu, (lambda m, x: np.sin(m[:, None] * x)), np.full(K, 2.0)


def series_coeffs(f_fine, bc, K):
    mu, phi, norm = eigen(bc, K)
    return mu, phi, norm * _trapz(f_fine * phi(mu, XF), XF, axis=1)


# ------------------------------------------------------------
# 熱伝導方程式
# ------------------------------------------------------------
HEAT_BC = {
    "両端の温度を一定に保つ": "DD",
    "両端を断熱する": "NN",
    "左端の温度を一定・右端を断熱": "DN",
    "地表面の温度が周期的に変わる(地盤の温度)": "PER",
}
HEAT_IC = ["パルス(中央だけ温める)", "矩形(一部分だけ高温)", "三角形", "ガウス型", "正弦波 sin kπx", "一様(全体が同じ温度)", "静止(はじめは 0)"]


def second_diff(N, h, left, right):
    """2階差分の行列(left/right は "D" か "N")。D の行は 0 にしておく"""
    D = (np.diag(-2.0 * np.ones(N)) + np.diag(np.ones(N - 1), 1) + np.diag(np.ones(N - 1), -1)) / h ** 2
    for i, side, j in ((0, left, 1), (N - 1, right, N - 2)):
        D[i] = 0.0
        if side == "N":                 # 鏡像点 u_{-1} = u_1 で ∂u/∂x = 0
            D[i, i] = -2.0 / h ** 2
            D[i, j] = 2.0 / h ** 2
    return D


@st.cache_data(show_spinner=False, max_entries=32)
def heat_run(bc, ic, x0, k, A, uL, uR, kappa, T, period, N=101, steps=2400):
    x = np.linspace(0, 1, N)
    h = x[1] - x[0]
    left = "D" if bc in ("DD", "DN", "PER") else "N"
    right = "D" if bc == "DD" else "N"
    D = second_diff(N, h, left, right)
    I = np.eye(N)

    def gL(t):
        return A * np.sin(2 * PI * t / period) if bc == "PER" else uL

    u = A * init_shape(ic, x, x0, k) if bc != "PER" else np.zeros(N)
    if left == "D":
        u[0] = gL(0.0)
    if right == "D":
        u[-1] = uR
    dt = T / steps

    def stepper(theta, d):
        Aimp = I - theta * kappa * d * D
        Bexp = I + (1 - theta) * kappa * d * D
        if left == "D":
            Aimp[0] = 0.0
            Aimp[0, 0] = 1.0
        if right == "D":
            Aimp[-1] = 0.0
            Aimp[-1, -1] = 1.0
        return np.linalg.inv(Aimp), Bexp

    Ainv_be, B_be = stepper(1.0, dt / 2)     # 立ち上がり(後退オイラー、半ステップ ×4)
    Ainv_cn, B_cn = stepper(0.5, dt)
    save = set(np.linspace(0, steps, NF).round().astype(int).tolist())
    snaps, t = [u.copy()], 0.0

    def advance(u, t, Ainv, B, d):
        rhs = B @ u
        t += d
        if left == "D":
            rhs[0] = gL(t)
        if right == "D":
            rhs[-1] = uR
        return Ainv @ rhs, t

    for n in range(1, steps + 1):
        if n <= 2:                        # 最初の 2 ステップは後退オイラー(半ステップ ×2)
            u, t = advance(u, t, Ainv_be, B_be, dt / 2)
            u, t = advance(u, t, Ainv_be, B_be, dt / 2)
        else:
            u, t = advance(u, t, Ainv_cn, B_cn, dt)
        if n in save:
            snaps.append(u.copy())
    while len(snaps) < NF:
        snaps.append(u.copy())
    return x, np.array(snaps), np.linspace(0, T, NF)


@st.cache_data(show_spinner=False, max_entries=32)
def heat_exact(bc, ic, x0, k, A, uL, uR, kappa, T, N=101, K=200):
    """フーリエ級数による厳密解(定常解 + 減衰する成分の和)"""
    x = np.linspace(0, 1, N)
    if bc == "DD":
        steady_f, steady = uL + (uR - uL) * XF, uL + (uR - uL) * x
    elif bc == "DN":
        steady_f, steady = uL + 0 * XF, uL + 0 * x
    else:
        steady_f, steady = 0 * XF, 0 * x
    mu, phi, c = series_coeffs(A * init_shape(ic, XF, x0, k) - steady_f, bc, K)
    P = phi(mu, x)
    ts = np.linspace(0, T, NF)
    return np.array([steady + (c * np.exp(-kappa * mu ** 2 * t)) @ P for t in ts])


def _strip_and_line(x, u, label, yr, extra=(), colorscale="RdBu_r", zr=None, line_name="数値解"):
    """上:グラフ、下:色の帯(温度なら棒・地盤の色、波なら変位の色)"""
    zr = zr or yr
    fig = go.Figure()
    for tr in extra:
        fig.add_trace(tr)
    fig.add_trace(go.Scatter(x=x, y=u, mode="lines", name=line_name, line=dict(color=C1, width=3)))
    fig.add_trace(go.Heatmap(x=x, y=[0], z=[u], zmin=zr[0], zmax=zr[1], colorscale=colorscale,
                             showscale=False, yaxis="y2", hoverinfo="skip"))
    layout(fig, title=label, height=440, xlab="位置 x", ylab="u", xrange=(0, 1), yrange=yr)
    fig.update_layout(yaxis=dict(domain=[0.24, 1.0]),
                      yaxis2=dict(domain=[0.0, 0.11], showticklabels=False, fixedrange=True),
                      uirevision="keep")
    return fig


def _yr(*arrays, pad=0.08):
    lo = min(float(np.min(a)) for a in arrays)
    hi = max(float(np.max(a)) for a in arrays)
    if hi - lo < 1e-9:
        lo, hi = lo - 1, hi + 1
    m = (hi - lo) * pad
    return (lo - m, hi + m)


def xt_diagram(x, ts, snaps, title, colorscale="RdBu_r", zmid=None):
    fig = go.Figure(go.Heatmap(x=x, y=ts, z=snaps, colorscale=colorscale, zmid=zmid,
                               colorbar=dict(title="u", thickness=12)))
    layout(fig, title=title, height=380, xlab="位置 x", ylab="時刻 t", legend=False)
    return fig


def _heat():
    st.markdown(
        "棒(または地盤)の温度 $u(x,t)$ の変化。熱は温度の高いところから低いところへ広がり、"
        "時間がたつと**定常状態**(境界条件だけで決まる温度分布)に近づく。")
    st.latex(r"\frac{\partial u}{\partial t}=\kappa\frac{\partial^2u}{\partial x^2}\quad(0<x<1)")
    c1, c2 = st.columns(2)
    bcname = c1.selectbox("境界条件", list(HEAT_BC), key="hs_bc")
    bc = HEAT_BC[bcname]
    uL = uR = 0.0
    period = 1.0
    if bc == "PER":
        ic = "静止(はじめは 0)"
        c2.caption("地表面 $x=0$ の温度を $u(0,t)=A\\sin(2\\pi t/P)$ で変え、深い所 $x=1$ は断熱とする。"
                   "はじめは全体が 0(平均の温度)。")
        A = c2.slider("地表面の温度の振れ幅 A", 0.5, 2.0, 1.0, 0.1, key="hs_A")
        period = c2.slider("周期 P(1日・1年などに相当)", 0.1, 2.0, 0.5, 0.1, key="hs_P")
        x0, k = 0.5, 1
    else:
        ic = c2.selectbox("初期条件 u(x, 0)", HEAT_IC[:-1], key="hs_ic")
        A = c2.slider("初期温度の大きさ", 0.5, 2.0, 1.0, 0.1, key="hs_A")
        x0, k = 0.5, 1
        if ic.startswith("正弦波"):
            k = c2.slider("k", 1, 6, 1, key="hs_k")
        elif not ic.startswith("一様"):
            x0 = c2.slider("山の位置 x₀", 0.15, 0.85, 0.5, 0.05, key="hs_x0")
        if bc in ("DD", "DN"):
            uL = c1.slider("左端の温度 u(0, t)", -1.0, 1.0, 0.0, 0.1, key="hs_uL")
        if bc == "DD":
            uR = c1.slider("右端の温度 u(1, t)", -1.0, 1.0, 0.0, 0.1, key="hs_uR")
    kappa = c1.select_slider("熱拡散率 κ", [0.01, 0.02, 0.05, 0.1, 0.2, 0.5], value=0.05, key="hs_kappa")
    T = c1.select_slider("計算する時間 T", [0.2, 0.5, 1.0, 2.0, 3.0, 5.0], value=1.0, key="hs_T")

    x, snaps, ts = heat_run(bc, ic, x0, k, A, uL, uR, kappa, T, period)
    exact = None
    if bc != "PER":
        exact = heat_exact(bc, ic, x0, k, A, uL, uR, kappa, T)
        show_ex = st.checkbox("フーリエ級数による厳密解を重ねる(破線)", value=True, key="hs_ex")
    yr = _yr(snaps, *( [exact] if exact is not None else []))
    labels = [f"t = {t:.3f}" for t in ts]
    static = [go.Scatter(x=x, y=snaps[0], mode="lines", name="初期温度", line=dict(color=GRAY, dash="dot"))]
    if bc == "PER":
        q = np.sqrt(PI / (kappa * period))           # √(ω/2κ)
        env = A * np.exp(-q * x)
        static += [go.Scatter(x=x, y=env, mode="lines", name="振れ幅の包絡線 ±A e^{-x√(ω/2κ)}",
                              line=dict(color=C3, dash="dash", width=1.5)),
                   go.Scatter(x=x, y=-env, mode="lines", showlegend=False, line=dict(color=C3, dash="dash", width=1.5))]
        yr = (-A * 1.1, A * 1.1)

    def draw(i):
        extra = list(static)
        if exact is not None and show_ex:
            extra.append(go.Scatter(x=x, y=exact[i], mode="lines", name="厳密解(級数)",
                                    line=dict(color=C2, dash="dash", width=2)))
        return _strip_and_line(x, snaps[i], f"温度分布  {labels[i]}", yr, extra, zr=yr)

    player("heatsim", labels, draw)
    if exact is not None:
        err = float(np.max(np.abs(snaps - exact)[1:]))
        st.caption(f"数値解と厳密解の差の最大値(t > 0): {err:.2e}  "
                   "(格子 101 点、クランク・ニコルソン法。とがった初期値ほど、はじめの差が大きい)")
    with st.expander("時空間図(横:位置 x、縦:時刻 t、色:温度)", expanded=False):
        st.plotly_chart(xt_diagram(x, ts, snaps, "温度の時空間図"), width="stretch", key="hs_xt")
    with st.expander("土木・建築での見かた"):
        st.markdown(
            "- **断熱**の境界は「熱が出入りしない面」。外壁の断熱材や、地盤の深い所のモデルになる。\n"
            "- **地表面の温度が周期的に変わる**場合、地中の温度の振れ幅は深さとともに $e^{-x\\sqrt{\\omega/2\\kappa}}$ で小さくなり、"
            "ピークの時刻も遅れる($\\omega=2\\pi/P$)。周期が長い(年変化)ほど深くまで伝わる。凍結深度や、地中に埋める管の深さを考える基礎になる。\n"
            "- 地盤中の水分の移動(圧密・浸透)も同じ形の拡散方程式で表される(テルツァギの圧密方程式)。")


# ------------------------------------------------------------
# 波動方程式
# ------------------------------------------------------------
WAVE_BC = {
    "両端固定(弦・橋のケーブル)": "DD",
    "両端自由": "NN",
    "左端固定・右端自由(地盤・建物のせん断振動)": "DN",
    "左端を正弦波で揺らす・右端自由(地震の揺れの入力)": "FORCE",
}
WAVE_IC = ["パルス(短い山)", "三角形(弦をつまんではなす)", "ガウス型", "正弦波 sin kπx", "静止(変位なし)"]


@st.cache_data(show_spinner=False, max_entries=32)
def wave_run(bc, ic, x0, k, A, vel, c, gamma, T, freq, N=401, nu=0.9):
    x = np.linspace(0, 1, N)
    h = x[1] - x[0]
    left = "N" if bc == "NN" else "D"
    right = "D" if bc == "DD" else "N"

    def lap(u):
        """2階差分(行列を作らずにずらして計算する)"""
        d = np.empty_like(u)
        d[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / h ** 2
        d[0] = 2 * (u[1] - u[0]) / h ** 2 if left == "N" else 0.0
        d[-1] = 2 * (u[-2] - u[-1]) / h ** 2 if right == "N" else 0.0
        return d
    dt = nu * h / c
    steps = int(np.ceil(T / dt))
    dt = T / steps

    def g(t):
        return A * 0.3 * np.sin(2 * PI * freq * t) if bc == "FORCE" else 0.0

    u0 = A * init_shape(ic, x, x0, k)
    v0 = vel * A * 3.0 * init_shape("ガウス", x, x0) if vel else np.zeros(N)
    if left == "D":
        u0[0] = g(0.0)
        v0[0] = 0.0
    if right == "D":
        u0[-1] = 0.0
        v0[-1] = 0.0
    c2 = c * c
    acc0 = c2 * lap(u0) - gamma * v0
    u1 = u0 + dt * v0 + 0.5 * dt * dt * acc0
    if left == "D":
        u1[0] = g(dt)
    if right == "D":
        u1[-1] = 0.0
    save = set(np.linspace(0, steps, NF).round().astype(int).tolist())
    snaps = [u0.copy()]
    if 1 in save:
        snaps.append(u1.copy())
    a, b = 1 + gamma * dt / 2, 1 - gamma * dt / 2
    up, uc = u0, u1
    for n in range(1, steps):
        un = (2 * uc - b * up + dt * dt * c2 * lap(uc)) / a
        if left == "D":
            un[0] = g((n + 1) * dt)
        if right == "D":
            un[-1] = 0.0
        up, uc = uc, un
        if n + 1 in save:
            snaps.append(uc.copy())
    snaps = np.array(snaps[:NF])
    return x, snaps, np.linspace(0, T, len(snaps))


@st.cache_data(show_spinner=False, max_entries=32)
def wave_exact(bc, ic, x0, k, A, vel, c, T, N=401, K=400):
    x = np.linspace(0, 1, N)
    mu, phi, a = series_coeffs(A * init_shape(ic, XF, x0, k), bc, K)
    vf = vel * A * 3.0 * init_shape("ガウス", XF, x0) if vel else 0 * XF
    _, _, b = series_coeffs(vf, bc, K)
    P = phi(mu, x)
    w = c * mu
    ts = np.linspace(0, T, NF)
    out = []
    for t in ts:
        sinpart = np.where(w > 0, np.sin(w * t) / np.where(w > 0, w, 1), t)
        out.append((a * np.cos(w * t) + b * sinpart) @ P)
    return np.array(out)


def _wave():
    st.markdown(
        "弦(または地盤・建物)の変位 $u(x,t)$ の変化。山は左右に分かれて速さ $c$ で進み、"
        "**固定端**では山が谷になって(位相が反転して)、**自由端**では山のまま反射する。")
    st.latex(r"\frac{\partial^2 u}{\partial t^2}=c^2\frac{\partial^2u}{\partial x^2}-\gamma\frac{\partial u}{\partial t}\quad(0<x<1)")
    c1, c2 = st.columns(2)
    bcname = c1.selectbox("境界条件", list(WAVE_BC), key="ws_bc")
    bc = WAVE_BC[bcname]
    c = c1.select_slider("波の速さ c", [0.5, 1.0, 1.5, 2.0], value=1.0, key="ws_c")
    gamma = c1.slider("減衰 γ", 0.0, 3.0, 0.0, 0.1, key="ws_gamma")
    T = c1.select_slider("計算する時間 T", [1.0, 2.0, 4.0, 6.0, 8.0], value=4.0, key="ws_T")
    freq = 0.0
    f1 = c / 4.0
    if bc == "FORCE":
        ic, vel, x0, k = "静止(変位なし)", False, 0.5, 1
        A = 1.0
        freq = c2.slider("揺らす振動数 f", 0.05, 1.5, round(f1, 2), 0.01, key="ws_f")
        c2.caption(f"この条件の1次固有振動数は $f_1=c/4={f1:.3f}$(周期 $4/c$)。"
                   f"$f$ を $f_1$ や $3f_1={3 * f1:.3f}$ に近づけると**共振**して揺れが大きくなる。")
    else:
        ic = c2.selectbox("初期変位 u(x, 0)", WAVE_IC[:-1] + ["静止(変位なし)"], key="ws_ic")
        vel = c2.checkbox("初速度を与える(x₀ の付近をたたく)", value=False, key="ws_vel")
        if ic.startswith("静止") and not vel:
            st.warning("変位も初速度も 0 だと何も動かない。初期変位を選ぶか、初速度を与えよう。")
        A = c2.slider("大きさ", 0.5, 2.0, 1.0, 0.1, key="ws_A")
        x0, k = 0.3, 1
        if ic.startswith("正弦波"):
            k = c2.slider("k", 1, 6, 1, key="ws_k")
        if not ic.startswith("正弦波") or vel:
            x0 = c2.slider("山(たたく位置)x₀", 0.15, 0.85, 0.3, 0.05, key="ws_x0")

    x, snaps, ts = wave_run(bc, ic, x0, k, A, vel, c, gamma, T, freq)
    exact = None
    if bc != "FORCE" and gamma == 0:
        exact = wave_exact(bc, ic, x0, k, A, vel, c, T)
        show_ex = st.checkbox("フーリエ級数による厳密解を重ねる(破線)", value=False, key="ws_ex")
    yr = _yr(snaps, *([exact] if exact is not None else []))
    m = max(abs(yr[0]), abs(yr[1]))
    yr = (-m, m)
    labels = [f"t = {t:.3f}" for t in ts]
    static = [go.Scatter(x=x, y=snaps[0], mode="lines", name="初期変位", line=dict(color=GRAY, dash="dot"))]

    def draw(i):
        extra = list(static)
        if exact is not None and show_ex:
            extra.append(go.Scatter(x=x, y=exact[i], mode="lines", name="厳密解(級数)",
                                    line=dict(color=C2, dash="dash", width=2)))
        return _strip_and_line(x, snaps[i], f"変位  {labels[i]}", yr, extra, colorscale="PuOr", zr=yr,
                               line_name="数値解(変位)")

    player("wavesim", labels, draw)
    if exact is not None:
        st.caption(f"数値解と厳密解の差の最大値: {float(np.max(np.abs(snaps - exact))):.2e}"
                   "(格子 401 点、クーラン数 0.9。とがった形ほど、波の形が少しずつくずれる=数値分散)")
    with st.expander("時空間図(横:位置 x、縦:時刻 t、色:変位)", expanded=True):
        st.caption("山の進む道すじが斜めの線になる。傾きが波の速さ、端で折り返すのが反射。")
        st.plotly_chart(xt_diagram(x, ts, snaps, "変位の時空間図", colorscale="PuOr", zmid=0),
                        width="stretch", key="ws_xt")
    with st.expander("土木・建築での見かた"):
        st.markdown(
            "- **左端固定・右端自由**は、硬い基盤の上にのった表層地盤(上面が地表で自由)や、根元を固定した建物の"
            "せん断振動のモデル。1次固有周期は $T_1=4H/V_s$($H$:層の厚さ、$V_s$:せん断波の速さ)で、ここでは $H=1$, $V_s=c$。\n"
            "- **地震の揺れの入力**では、基盤の揺れの振動数が固有振動数に近いと**共振**して地表の揺れが大きくなる。"
            "減衰 $\\gamma$ を大きくすると共振のピークが抑えられる(制振・免震の考え方)。\n"
            "- **両端固定**は弦や吊り橋のケーブル。固有振動数は $f_k=kc/2$ で、三角形の初期変位は多くのモードの重ね合わせになる。")


def tab_sim():
    st.info("境界条件と初期条件を選んで「▶ シミュレーション開始」を押すと、時間とともに変わるようすを再生する。"
            "再生のあとは「時刻」のスライダーで好きな瞬間を確かめられる。下の色の帯は棒(地盤・弦)そのものの色分け。")
    kind = st.radio("方程式", ["熱伝導方程式(熱の拡散)", "波動方程式(振動と反射)"], horizontal=True, key="sim_kind")
    if kind.startswith("熱"):
        _heat()
    else:
        _wave()
