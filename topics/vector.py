"""① ベクトル解析:勾配・発散・回転・グリーンの定理"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import C1, C2, C3, C4, GRAY, arrow, layout, quiver, show

# ------------------------------------------------------------
# スカラー場(勾配タブ用)
# ------------------------------------------------------------
E = lambda x, y: np.exp(-(x ** 2 + y ** 2))  # noqa: E731
SCALARS = {
    "x² + y²(お椀)": dict(
        f=lambda x, y: x ** 2 + y ** 2, fx=lambda x, y: 2 * x, fy=lambda x, y: 2 * y,
        tex=r"f(x,y)=x^2+y^2", grad=r"\nabla f=(2x,\ 2y)", R=2.0),
    "x² − y²(鞍点)": dict(
        f=lambda x, y: x ** 2 - y ** 2, fx=lambda x, y: 2 * x, fy=lambda x, y: -2 * y,
        tex=r"f(x,y)=x^2-y^2", grad=r"\nabla f=(2x,\ -2y)", R=2.0),
    "e^{−(x²+y²)}(山)": dict(
        f=E, fx=lambda x, y: -2 * x * E(x, y), fy=lambda x, y: -2 * y * E(x, y),
        tex=r"f(x,y)=e^{-(x^2+y^2)}", grad=r"\nabla f=-2e^{-(x^2+y^2)}(x,\ y)", R=2.0),
    "x e^{−(x²+y²)}(山と谷)": dict(
        f=lambda x, y: x * E(x, y), fx=lambda x, y: (1 - 2 * x ** 2) * E(x, y),
        fy=lambda x, y: -2 * x * y * E(x, y),
        tex=r"f(x,y)=x\,e^{-(x^2+y^2)}", grad=r"\nabla f=e^{-(x^2+y^2)}(1-2x^2,\ -2xy)", R=2.0),
    "sin x cos y(卵パック)": dict(
        f=lambda x, y: np.sin(x) * np.cos(y), fx=lambda x, y: np.cos(x) * np.cos(y),
        fy=lambda x, y: -np.sin(x) * np.sin(y),
        tex=r"f(x,y)=\sin x\cos y", grad=r"\nabla f=(\cos x\cos y,\ -\sin x\sin y)", R=3.0),
}


# ------------------------------------------------------------
# ベクトル場(発散・回転・グリーンの定理用)
# ------------------------------------------------------------
def _r2(x, y):
    return x ** 2 + y ** 2


def fields(a=1.0, b=0.0, c=0.0, d=1.0):
    z = lambda x, y: 0 * x  # noqa: E731
    return {
        "湧き出し F = (x, y)": dict(
            P=lambda x, y: x, Q=lambda x, y: y, div=lambda x, y: 2 + z(x, y), rot=z,
            tex=r"\boldsymbol F=(x,\ y)", dtex=r"2", rtex=r"0"),
        "渦 F = (−y, x)": dict(
            P=lambda x, y: -y, Q=lambda x, y: x, div=z, rot=lambda x, y: 2 + z(x, y),
            tex=r"\boldsymbol F=(-y,\ x)", dtex=r"0", rtex=r"2"),
        "せん断流 F = (y, 0)": dict(
            P=lambda x, y: y, Q=z, div=z, rot=lambda x, y: -1 + z(x, y),
            tex=r"\boldsymbol F=(y,\ 0)", dtex=r"0", rtex=r"-1"),
        "鞍形の流れ F = (x, −y)": dict(
            P=lambda x, y: x, Q=lambda x, y: -y, div=z, rot=z,
            tex=r"\boldsymbol F=(x,\ -y)", dtex=r"0", rtex=r"0"),
        "渦巻き F = (x − y, x + y)": dict(
            P=lambda x, y: x - y, Q=lambda x, y: x + y, div=lambda x, y: 2 + z(x, y),
            rot=lambda x, y: 2 + z(x, y),
            tex=r"\boldsymbol F=(x-y,\ x+y)", dtex=r"2", rtex=r"2"),
        "F = (xy, y²)": dict(
            P=lambda x, y: x * y, Q=lambda x, y: y ** 2, div=lambda x, y: 3 * y,
            rot=lambda x, y: -x,
            tex=r"\boldsymbol F=(xy,\ y^2)", dtex=r"y+2y=3y", rtex=r"0-x=-x"),
        "F = (−y³, x³)": dict(
            P=lambda x, y: -y ** 3, Q=lambda x, y: x ** 3, div=z,
            rot=lambda x, y: 3 * x ** 2 + 3 * y ** 2,
            tex=r"\boldsymbol F=(-y^3,\ x^3)", dtex=r"0", rtex=r"3x^2+3y^2"),
        "点渦 F = (−y, x)/(x²+y²)": dict(
            P=lambda x, y: -y / _r2(x, y), Q=lambda x, y: x / _r2(x, y), div=z, rot=z,
            tex=r"\boldsymbol F=\dfrac{(-y,\ x)}{x^2+y^2}", dtex=r"0\ (\text{原点以外})",
            rtex=r"0\ (\text{原点以外})", singular=True),
        "点源 F = (x, y)/(x²+y²)": dict(
            P=lambda x, y: x / _r2(x, y), Q=lambda x, y: y / _r2(x, y), div=z, rot=z,
            tex=r"\boldsymbol F=\dfrac{(x,\ y)}{x^2+y^2}", dtex=r"0\ (\text{原点以外})",
            rtex=r"0\ (\text{原点以外})", singular=True),
        "線形 F = (ax + by, cx + dy)(係数を自分で決める)": dict(
            P=lambda x, y: a * x + b * y, Q=lambda x, y: c * x + d * y,
            div=lambda x, y: a + d + z(x, y), rot=lambda x, y: c - b + z(x, y),
            tex=rf"\boldsymbol F=({a:g}x {b:+g}y,\ {c:g}x {d:+g}y)",
            dtex=rf"a+d={a + d:g}", rtex=rf"c-b={c - b:g}", custom=True),
    }


def pick_field(key):
    names = list(fields())
    name = st.selectbox("ベクトル場", names, key=f"{key}_field")
    a = b = c = 0.0
    d = 1.0
    if name.startswith("線形"):
        k1, k2, k3, k4 = st.columns(4)
        a = k1.slider("a", -2.0, 2.0, 1.0, 0.5, key=f"{key}_a")
        b = k2.slider("b", -2.0, 2.0, -1.0, 0.5, key=f"{key}_b")
        c = k3.slider("c", -2.0, 2.0, 1.0, 0.5, key=f"{key}_c")
        d = k4.slider("d", -2.0, 2.0, 0.0, 0.5, key=f"{key}_d")
    return name, fields(a, b, c, d)[name]


def field_quiver(F, R, n=17, length=None, normalize=True):
    g = np.linspace(-R, R, n)
    X, Y = np.meshgrid(g, g)
    with np.errstate(divide="ignore", invalid="ignore"):
        U, V = F["P"](X, Y), F["Q"](X, Y)
    if F.get("singular"):
        bad = _r2(X, Y) < (0.6 * (g[1] - g[0])) ** 2
        U, V = np.where(bad, np.nan, U), np.where(bad, np.nan, V)
    return quiver(X, Y, U, V, length=length or 0.75 * (g[1] - g[0]), normalize=normalize,
                  color=GRAY, name="F の向き")


# ------------------------------------------------------------


# ---------- 勾配 ----------
def tab_gradient():
    st.info("勾配 $\\nabla f$ は「$f$ が最も急に増える向き」を指し、等高線に直交する。")
    name = st.selectbox("スカラー場 f", list(SCALARS), key="g_f")
    S = SCALARS[name]
    R = S["R"]
    st.latex(S["tex"] + r",\qquad " + S["grad"])

    c1, c2, c3 = st.columns(3)
    x0 = c1.slider("点の x 座標", -R, R, 0.8, 0.05, key="g_x")
    y0 = c2.slider("点の y 座標", -R, R, 0.5, 0.05, key="g_y")
    th = c3.slider("方向 u の角度 θ(度)", 0, 359, 30, key="g_th")

    gx, gy = float(S["fx"](x0, y0)), float(S["fy"](x0, y0))
    gnorm = np.hypot(gx, gy)
    u = np.array([np.cos(np.radians(th)), np.sin(np.radians(th))])
    Du = gx * u[0] + gy * u[1]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("f(x₀, y₀)", f"{float(S['f'](x0, y0)):.4f}")
    m2.metric("∇f(x₀, y₀)", f"({gx:.3f}, {gy:.3f})")
    m3.metric("|∇f|(最大の傾き)", f"{gnorm:.4f}")
    m4.metric("方向微分 D_u f", f"{Du:.4f}")

    g = np.linspace(-R, R, 160)
    X, Y = np.meshgrid(g, g)
    fig = go.Figure()
    fig.add_trace(go.Contour(x=g, y=g, z=S["f"](X, Y), ncontours=24, colorscale="RdBu_r",
                             opacity=0.55, contours=dict(coloring="heatmap", showlines=True),
                             showscale=False, name="f", hoverinfo="skip"))
    gq = np.linspace(-R, R, 15)
    XQ, YQ = np.meshgrid(gq, gq)
    fig.add_trace(quiver(XQ, YQ, S["fx"](XQ, YQ), S["fy"](XQ, YQ),
                         length=0.7 * (gq[1] - gq[0]), color="#333333", name="∇f の向き"))
    sc = 0.35 * R / gnorm if gnorm > 1e-12 else 0.0  # 表示用に長さをそろえる
    fig.add_trace(arrow(x0, y0, gx * sc, gy * sc, color=C2, name="∇f(点)"))
    fig.add_trace(arrow(x0, y0, 0.3 * R * u[0], 0.3 * R * u[1], color=C3, name="方向 u"))
    fig.add_trace(go.Scatter(x=[x0], y=[y0], mode="markers", marker=dict(size=10, color="black"),
                             name="点 (x₀, y₀)"))
    layout(fig, height=520, equal=True, xlab="x", ylab="y", xrange=(-R, R), yrange=(-R, R))

    cA, cB = st.columns([3, 2])
    with cA:
        show(fig)
    with cB:
        ths = np.linspace(0, 360, 361)
        Dus = gx * np.cos(np.radians(ths)) + gy * np.sin(np.radians(ths))
        f2 = go.Figure(go.Scatter(x=ths, y=Dus, mode="lines", line=dict(color=C3), name="D_u f"))
        f2.add_trace(go.Scatter(x=[th], y=[Du], mode="markers", marker=dict(size=11, color=C3),
                                name="いまの θ"))
        if gnorm > 0:
            th_max = np.degrees(np.arctan2(gy, gx)) % 360
            f2.add_trace(go.Scatter(x=[th_max], y=[gnorm], mode="markers",
                                    marker=dict(size=11, color=C2, symbol="star"),
                                    name="最大(∇f の向き)"))
        layout(f2, title="方向微分 D_u f と角度 θ", height=330, xlab="θ(度)", ylab="D_u f")
        show(f2)
        st.latex(r"D_{\boldsymbol u}f=\nabla f\cdot\boldsymbol u=|\nabla f|\cos\varphi")
        st.caption("φ は ∇f と u のなす角。φ = 0 で最大 |∇f|、φ = 90°(等高線の向き)で 0。")

    if st.checkbox("曲面 z = f(x, y) を3Dで見る", key="g_3d"):
        f3 = go.Figure(go.Surface(x=g, y=g, z=S["f"](X, Y), colorscale="RdBu_r", opacity=0.9,
                                  showscale=False, contours=dict(z=dict(show=True, usecolormap=True,
                                                                        project_z=True))))
        f3.add_trace(go.Scatter3d(x=[x0], y=[y0], z=[float(S["f"](x0, y0))], mode="markers",
                                  marker=dict(size=6, color="black"), name="点"))
        f3.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0),
                         scene=dict(aspectmode="cube"))
        show(f3)

    with st.expander("練習問題"):
        st.markdown("**問1** $f(x,y)=x^2y+y^3$ の点 $(1,1)$ における勾配と、"
                    "$\\boldsymbol u=\\frac{1}{\\sqrt2}(1,1)$ 方向の方向微分を求めよ。")
        st.latex(r"\nabla f=(2xy,\ x^2+3y^2)=(2,\ 4),\qquad D_{\boldsymbol u}f=\frac{2+4}{\sqrt2}=3\sqrt2")
        st.markdown("**問2** 方向微分が 0 になる向きは、等高線とどんな関係にあるか。")
        st.write("$\\nabla f\\cdot\\boldsymbol u=0$ より $\\boldsymbol u\\perp\\nabla f$。"
                 "その向きに進んでも $f$ は変わらないので、**等高線の接線方向**。"
                 "よって $\\nabla f$ は等高線に直交する。")


# ---------- 発散・回転 ----------
def tab_div_rot():
    st.info("発散 div は「湧き出しの強さ」、回転 rot は「渦の強さ」。"
            "どちらも **小さな円での流出量・循環を面積で割った極限** として定義できる。")
    name, F = pick_field("dr")
    st.latex(F["tex"] + r",\qquad \mathrm{div}\,\boldsymbol F=\frac{\partial P}{\partial x}"
             r"+\frac{\partial Q}{\partial y}=" + F["dtex"]
             + r",\qquad \mathrm{rot}\,\boldsymbol F=\frac{\partial Q}{\partial x}"
             r"-\frac{\partial P}{\partial y}=" + F["rtex"])

    R = 2.0
    c1, c2, c3, c4 = st.columns(4)
    bg = c1.radio("背景の色", ["div", "rot", "なし"], horizontal=True, key="dr_bg")
    px = c2.slider("円の中心 x", -1.5, 1.5, 0.8, 0.1, key="dr_px")
    py = c3.slider("円の中心 y", -1.5, 1.5, 0.6, 0.1, key="dr_py")
    eps = c4.slider("円の半径 ε", 0.05, 1.0, 0.4, 0.05, key="dr_eps")

    # 小さな円での流出量・循環(数値積分)
    n = 2000
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xc, yc = px + eps * np.cos(t), py + eps * np.sin(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        Pc, Qc = F["P"](xc, yc), F["Q"](xc, yc)
    dt = 2 * np.pi / n
    flux = float(np.sum(Pc * np.cos(t) + Qc * np.sin(t)) * eps * dt)
    circ = float(np.sum(-Pc * np.sin(t) + Qc * np.cos(t)) * eps * dt)
    area = np.pi * eps ** 2
    contains_origin = F.get("singular") and np.hypot(px, py) < eps

    fig = go.Figure()
    if bg != "なし":
        g = np.linspace(-R, R, 161)
        X, Y = np.meshgrid(g, g)
        Z = F[bg](X, Y).astype(float)
        if F.get("singular"):
            Z = np.where(_r2(X, Y) < 0.02, np.nan, Z)
        zmax = max(float(np.nanmax(np.abs(Z))), 1e-9)
        fig.add_trace(go.Heatmap(x=g, y=g, z=Z, colorscale="RdBu_r", zmin=-zmax, zmax=zmax,
                                 opacity=0.6, colorbar=dict(title=bg, thickness=12),
                                 hovertemplate=f"x=%{{x:.2f}}<br>y=%{{y:.2f}}<br>{bg}=%{{z:.3f}}<extra></extra>"))
    fig.add_trace(field_quiver(F, R))
    tt = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(go.Scatter(x=px + eps * np.cos(tt), y=py + eps * np.sin(tt), mode="lines",
                             line=dict(color=C2, width=3), name="半径 ε の円"))
    k = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        fig.add_trace(quiver(px + eps * np.cos(k), py + eps * np.sin(k),
                             F["P"](px + eps * np.cos(k), py + eps * np.sin(k)),
                             F["Q"](px + eps * np.cos(k), py + eps * np.sin(k)),
                             length=0.25, normalize=True, color=C2, name="円周上の F", width=2.2))
    if F.get("singular"):
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(symbol="x", size=12,
                                 color="black"), name="特異点(原点)"))
    layout(fig, height=540, equal=True, xlab="x", ylab="y", xrange=(-R, R), yrange=(-R, R))

    cA, cB = st.columns([3, 2])
    with cA:
        show(fig)
    with cB:
        with np.errstate(divide="ignore", invalid="ignore"):
            dv, rt = float(F["div"](px, py)), float(F["rot"](px, py))
        st.markdown("**円の中心での値(公式)**")
        st.write(f"div F = **{dv:.4f}**、 rot F = **{rt:.4f}**")
        st.markdown("**半径 ε の円で数値計算**")
        st.latex(r"\frac{1}{\pi\varepsilon^2}\oint_C\boldsymbol F\cdot\boldsymbol n\,ds"
                 rf"={flux / area:.4f}")
        st.latex(r"\frac{1}{\pi\varepsilon^2}\oint_C\boldsymbol F\cdot d\boldsymbol r"
                 rf"={circ / area:.4f}")
        st.caption("ε を小さくすると、上の値が div F・rot F に近づく(定数の場なら最初から一致)。")
        if contains_origin:
            st.warning(f"円が特異点(原点)を含んでいます。流出量 = {flux:.4f}、循環 = {circ:.4f}"
                       "(どちらかが 2π ≈ 6.2832)。ε → 0 でも面積で割った値は発散します。"
                       "原点に「点状の湧き出し・渦」が集中しているためです。")

    with st.expander("見どころ"):
        st.markdown(
            "- **せん断流** $(y,0)$:流線はまっすぐなのに rot = −1。水車を置くと回る(上と下で流速が違う)。\n"
            "- **点渦**:流れはぐるぐる回っているのに、原点以外では rot = 0。"
            "原点を含まない円では循環 0、含む円では 2π。\n"
            "- **線形の場**:div = a + d(行列のトレース)、rot = c − b。"
            "行列 $\\begin{pmatrix}a&b\\\\c&d\\end{pmatrix}$ の固有値と流れの形の関係も調べてみよう。"
        )
    with st.expander("練習問題"):
        st.markdown("**問3** $\\boldsymbol F=(x^2y,\\ xy^2)$ の div と rot を求めよ。")
        st.latex(r"\mathrm{div}\,\boldsymbol F=2xy+2xy=4xy,\qquad"
                 r"\mathrm{rot}\,\boldsymbol F=y^2-x^2")
        st.markdown("**問4** $\\boldsymbol F=\\nabla f$(勾配場)ならば rot F = 0 となることを示せ。")
        st.latex(r"\mathrm{rot}\,\nabla f=\frac{\partial}{\partial x}\frac{\partial f}{\partial y}"
                 r"-\frac{\partial}{\partial y}\frac{\partial f}{\partial x}=0\quad"
                 r"(\text{偏微分の順序交換})")


# ---------- グリーンの定理 ----------
def tab_green():
    st.info("グリーンの定理:境界を一周する線積分 = 内部の rot(または div)の面積分。")
    st.latex(r"\oint_C P\,dx+Q\,dy=\iint_D\left(\frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y}"
             r"\right)dxdy,\qquad \oint_C P\,dy-Q\,dx=\iint_D\left(\frac{\partial P}{\partial x}"
             r"+\frac{\partial Q}{\partial y}\right)dxdy")
    name, F = pick_field("gr")
    st.latex(F["tex"] + r",\quad \mathrm{rot}\,\boldsymbol F=" + F["rtex"]
             + r",\quad \mathrm{div}\,\boldsymbol F=" + F["dtex"])

    c1, c2, c3, c4 = st.columns(4)
    shape = c1.radio("領域 D", ["円", "長方形"], horizontal=True, key="gr_shape")
    cx = c2.slider("中心 x", -1.5, 1.5, 0.5, 0.1, key="gr_cx")
    cy = c3.slider("中心 y", -1.5, 1.5, 0.0, 0.1, key="gr_cy")
    if shape == "円":
        Rr = c4.slider("半径", 0.2, 1.5, 1.0, 0.05, key="gr_R")
        w = h = 2 * Rr
    else:
        w = c4.slider("横の長さ", 0.2, 3.0, 2.0, 0.1, key="gr_w")
        h = st.slider("縦の長さ", 0.2, 3.0, 1.2, 0.1, key="gr_h")

    # 境界の点列(反時計回り)と dx, dy
    n = 4000
    if shape == "円":
        t = (np.arange(n) + 0.5) * 2 * np.pi / n
        bx, by = cx + Rr * np.cos(t), cy + Rr * np.sin(t)
        dx, dy = -Rr * np.sin(t) * 2 * np.pi / n, Rr * np.cos(t) * 2 * np.pi / n
    else:
        m = n // 4
        s = (np.arange(m) + 0.5) / m
        x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
        bx = np.concatenate([x0 + w * s, np.full(m, x1), x1 - w * s, np.full(m, x0)])
        by = np.concatenate([np.full(m, y0), y0 + h * s, np.full(m, y1), y1 - h * s])
        dx = np.concatenate([np.full(m, w / m), np.zeros(m), np.full(m, -w / m), np.zeros(m)])
        dy = np.concatenate([np.zeros(m), np.full(m, h / m), np.zeros(m), np.full(m, -h / m)])
    with np.errstate(divide="ignore", invalid="ignore"):
        P, Q = F["P"](bx, by), F["Q"](bx, by)
    circ = float(np.sum(P * dx + Q * dy))
    flux = float(np.sum(P * dy - Q * dx))

    # 面積分(円は極座標、長方形は中点則)
    k = 400
    if shape == "円":
        r = (np.arange(k) + 0.5) * Rr / k
        th = (np.arange(k) + 0.5) * 2 * np.pi / k
        RR, TT = np.meshgrid(r, th)
        X, Y = cx + RR * np.cos(TT), cy + RR * np.sin(TT)
        dA = RR * (Rr / k) * (2 * np.pi / k)
    else:
        xs = cx - w / 2 + (np.arange(k) + 0.5) * w / k
        ys = cy - h / 2 + (np.arange(k) + 0.5) * h / k
        X, Y = np.meshgrid(xs, ys)
        dA = (w / k) * (h / k)
    rot_int = float(np.sum(F["rot"](X, Y) * dA))
    div_int = float(np.sum(F["div"](X, Y) * dA))

    if shape == "円":
        inside_origin = np.hypot(cx, cy) < Rr
    else:
        inside_origin = abs(cx) < w / 2 and abs(cy) < h / 2

    R = 2.5
    fig = go.Figure()
    fig.add_trace(field_quiver(F, R, n=19))
    fig.add_trace(go.Scatter(x=np.append(bx, bx[0]), y=np.append(by, by[0]), mode="lines",
                             fill="toself", fillcolor="rgba(214,39,40,0.12)",
                             line=dict(color=C2, width=3), name="境界 C(反時計回り)"))
    idx = np.linspace(0, len(bx) - 1, 8).astype(int)
    fig.add_trace(quiver(bx[idx], by[idx], dx[idx], dy[idx], length=0.25, color=C2,
                         name="向き", width=2.5))
    if F.get("singular"):
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(symbol="x", size=12,
                                 color="black"), name="特異点(原点)"))
    layout(fig, height=520, equal=True, xlab="x", ylab="y", xrange=(-R, R), yrange=(-R, R))

    cA, cB = st.columns([3, 2])
    with cA:
        show(fig)
    with cB:
        st.markdown("**循環(接線方向)**")
        st.latex(rf"\oint_C P\,dx+Q\,dy={circ:.5f}")
        st.latex(rf"\iint_D \mathrm{{rot}}\,\boldsymbol F\,dxdy={rot_int:.5f}")
        st.markdown("**流出量(法線方向)**")
        st.latex(rf"\oint_C P\,dy-Q\,dx={flux:.5f}")
        st.latex(rf"\iint_D \mathrm{{div}}\,\boldsymbol F\,dxdy={div_int:.5f}")
        ok = abs(circ - rot_int) < 1e-3 * max(1, abs(circ)) and \
            abs(flux - div_int) < 1e-3 * max(1, abs(flux))
        if F.get("singular") and inside_origin:
            st.error("両辺が一致しません!領域の中に F が定義されない点(原点)があるので、"
                     "グリーンの定理の仮定「D で F が C¹ 級」が成り立たないためです。"
                     "線積分の 2π は、原点に集中した渦・湧き出しの強さを表します。")
        elif ok:
            st.success("線積分と面積分が一致しました(数値誤差の範囲)。")
        else:
            st.warning("数値誤差がやや大きいようです。")

    with st.expander("練習問題"):
        st.markdown("**問5** $C$:単位円(反時計回り)のとき、$\\displaystyle\\oint_C -y\\,dx+x\\,dy$ を"
                    "グリーンの定理で求めよ。")
        st.latex(r"\iint_D (1-(-1))\,dxdy=2\times(\text{円の面積})=2\pi")
        st.write("一般に $\\frac12\\oint_C -y\\,dx+x\\,dy$ は D の面積になる(面積公式)。")
        st.markdown("**問6** 点渦 $\\boldsymbol F=\\frac{(-y,x)}{x^2+y^2}$ について、"
                    "原点を囲む任意の単純閉曲線での循環が $2\\pi$ になる理由を説明せよ。")
        st.write("原点を囲む小さな円 $C_\\varepsilon$ をとり、$C$ と $C_\\varepsilon$ で挟まれた領域に"
                 "グリーンの定理を使うと rot F = 0 より $\\oint_C=\\oint_{C_\\varepsilon}$。"
                 "円周上では $\\boldsymbol F\\cdot d\\boldsymbol r=d\\theta$ なので $2\\pi$。")
