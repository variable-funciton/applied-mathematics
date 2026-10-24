"""① ベクトル解析 第3章 線積分・面積分(グリーンの定理は vector.py)"""
from fractions import Fraction as Fr

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import (C1, C2, C3, C4, GRAY, arrow, arrow3d, cones, layout, layout3d, quiver, show,
                    trapz, vec_tex)
from topics.vector import SCALARS

PI = np.pi


# ------------------------------------------------------------
# 平面曲線(線積分用):r(t), r'(t), t の範囲
# ------------------------------------------------------------
PATHS2 = {
    "線分 (0,0)→(1,1)": dict(r=lambda t: (t, t), d=lambda t: (1 + 0 * t, 1 + 0 * t), t=(0, 1),
                           tex=r"\vec r(t)=(t,\ t),\ 0\le t\le1,\ |\vec r'|=\sqrt2"),
    "放物線 y = x²(0≤x≤1)": dict(r=lambda t: (t, t ** 2), d=lambda t: (1 + 0 * t, 2 * t), t=(0, 1),
                               tex=r"\vec r(t)=(t,\ t^2),\ 0\le t\le1,\ |\vec r'|=\sqrt{1+4t^2}"),
    "単位円(一周)": dict(r=lambda t: (np.cos(t), np.sin(t)), d=lambda t: (-np.sin(t), np.cos(t)),
                     t=(0, 2 * PI), tex=r"\vec r(t)=(\cos t,\ \sin t),\ 0\le t\le2\pi,\ |\vec r'|=1"),
    "四分円 (1,0)→(0,1)": dict(r=lambda t: (np.cos(t), np.sin(t)), d=lambda t: (-np.sin(t), np.cos(t)),
                            t=(0, PI / 2), tex=r"\vec r(t)=(\cos t,\ \sin t),\ 0\le t\le\frac\pi2"),
}
SFUN = {
    "f = 1(曲線の長さ)": (lambda x, y: 1 + 0 * x, r"f=1"),
    "f = x + y": (lambda x, y: x + y, r"f=x+y"),
    "f = xy": (lambda x, y: x * y, r"f=xy"),
    "f = x² + y²": (lambda x, y: x ** 2 + y ** 2, r"f=x^2+y^2"),
    "f = 1 + x²": (lambda x, y: 1 + x ** 2, r"f=1+x^2"),
}


# ============================================================
# 1. スカラー場の線積分
# ============================================================
def tab_scalar_line():
    st.info("曲線 C に沿って f を積分する。f ≥ 0 なら、C の上に立てた高さ f の「カーテン」の面積。")
    st.latex(r"\int_C f\,ds=\int_a^b f(x(t),y(t))\,|\vec r'(t)|\,dt,\qquad |\vec r'(t)|=\sqrt{x'(t)^2+y'(t)^2}")
    c1, c2 = st.columns(2)
    pname = c1.selectbox("曲線 C", list(PATHS2), key="sl_c")
    fname = c2.selectbox("スカラー場 f", list(SFUN), index=1, key="sl_f")
    P = PATHS2[pname]
    f, ftex = SFUN[fname]
    st.latex(P["tex"] + r",\qquad " + ftex)

    t = np.linspace(*P["t"], 20001)
    x, y = P["r"](t)
    dx, dy = P["d"](t)
    val = float(trapz(f(x, y) * np.hypot(dx, dy), t))

    ts = np.linspace(*P["t"], 200)
    xs, ys = P["r"](ts)
    zs = f(xs, ys)
    fig = go.Figure()
    fig.add_trace(go.Surface(x=np.vstack([xs, xs]), y=np.vstack([ys, ys]), z=np.vstack([0 * zs, zs]),
                             colorscale=[[0, C4], [1, C4]], opacity=0.55, showscale=False, name="カーテン"))
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=0 * xs, mode="lines", line=dict(color=C1, width=6), name="C"))
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=C2, width=5), name="高さ f"))
    g = np.linspace(-1.2, 1.2, 40)
    X, Y = np.meshgrid(g, g)
    fig.add_trace(go.Surface(x=X, y=Y, z=f(X, Y), opacity=0.18, colorscale="Greys", showscale=False,
                             name="z = f(x,y)", hoverinfo="skip"))
    layout3d(fig, height=500)
    cA, cB = st.columns([3, 2])
    with cA:
        show(fig)
    with cB:
        st.metric("∫_C f ds(数値積分)", f"{val:.6f}")
        st.caption("同じ曲線を逆向きにたどっても、ds > 0 なので値は変わらない(ベクトル場の線積分との違い)。")

    with st.expander("練習問題"):
        st.markdown("**問1** C:線分 $(0,0)\\to(1,1)$ のとき $\\int_C (x+y)\\,ds$ を求めよ。")
        st.latex(r"\int_0^1 2t\cdot\sqrt2\,dt=\sqrt2\approx1.414214")
        st.markdown("**問2** C:単位円のとき $\\int_C (x^2+y^2)\\,ds$ を求めよ。")
        st.latex(r"\int_0^{2\pi}1\cdot1\,dt=2\pi")


# ============================================================
# 2. ベクトル場の線積分
# ============================================================
VF2 = {
    "F = (y, x) = ∇(xy)": dict(P=lambda x, y: y, Q=lambda x, y: x, phi=lambda x, y: x * y,
                              tex=r"\vec F=(y,\ x)=\nabla(xy)", ptex=r"\varphi=xy"),
    "F = (2xy, x²) = ∇(x²y)": dict(P=lambda x, y: 2 * x * y, Q=lambda x, y: x ** 2,
                                  phi=lambda x, y: x ** 2 * y, tex=r"\vec F=(2xy,\ x^2)=\nabla(x^2y)",
                                  ptex=r"\varphi=x^2y"),
    "F = (−y, x)": dict(P=lambda x, y: -y, Q=lambda x, y: x, phi=None, tex=r"\vec F=(-y,\ x)"),
    "F = (y², x)": dict(P=lambda x, y: y ** 2, Q=lambda x, y: x, phi=None, tex=r"\vec F=(y^2,\ x)"),
}


def _paths_AB(bx, by):
    """原点 O から B=(bx,by) への3つの経路:(名前, x(t), y(t), x'(t), y'(t))"""
    t = (np.arange(20000) + 0.5) / 20000  # 中点則
    one = np.ones_like(t)
    out = [("直線", bx * t, by * t, bx * one, by * one),
           ("放物線 y = (b_y/b_x²) x²", bx * t, by * t ** 2, bx * one, 2 * by * t)]
    # 折れ線 O→(bx,0)→B
    h = t < 0.5
    s = np.where(h, 2 * t, 2 * t - 1)
    out.append(("折れ線 O→(b_x,0)→B", np.where(h, bx * s, bx), np.where(h, 0, by * s),
                np.where(h, 2 * bx, 0), np.where(h, 0, 2 * by)))
    return t, out


def tab_vector_line():
    st.info("ベクトル場の線積分は、力 F が経路に沿ってする**仕事**。"
            "F が勾配場(保存場)$\\vec F=\\nabla\\varphi$ なら、値は始点と終点だけで決まる。")
    st.latex(r"\int_C\vec F\cdot d\vec r=\int_C P\,dx+Q\,dy=\int_a^b\left(P\,x'(t)+Q\,y'(t)\right)dt")
    name = st.selectbox("ベクトル場", list(VF2), key="vl_f")
    F = VF2[name]
    st.latex(F["tex"])
    c1, c2 = st.columns(2)
    bx = c1.slider("終点 B の x", 0.2, 1.5, 1.0, 0.1, key="vl_bx")
    by = c2.slider("終点 B の y", 0.2, 1.5, 1.0, 0.1, key="vl_by")
    t, paths = _paths_AB(bx, by)
    rows = []
    fig = go.Figure()
    g = np.linspace(-0.2, 1.7, 14)
    X, Y = np.meshgrid(g, g)
    fig.add_trace(quiver(X, Y, F["P"](X, Y), F["Q"](X, Y), length=0.11, color=GRAY, name="F"))
    for (nm, x, y, dx, dy), col in zip(paths, (C1, C2, C3)):
        W = float(np.mean(F["P"](x, y) * dx + F["Q"](x, y) * dy))
        rows.append({"経路": nm, "∫_C F·dr": f"{W:.6f}"})
        fig.add_trace(go.Scatter(x=x[::50], y=y[::50], mode="lines", line=dict(color=col, width=3), name=nm))
    fig.add_trace(go.Scatter(x=[0, bx], y=[0, by], mode="markers+text", text=["O", "B"],
                             textposition="top left", marker=dict(size=9, color="black"), showlegend=False))
    layout(fig, height=480, equal=True, xlab="x", ylab="y", xrange=(-0.2, 1.7), yrange=(-0.2, 1.7))
    cA, cB = st.columns([3, 2])
    with cA:
        show(fig)
    with cB:
        st.dataframe(rows, hide_index=True, width="stretch")
        if F["phi"] is not None:
            st.success("経路によらず同じ値(保存場)。")
            st.latex(F["ptex"] + rf",\quad \varphi(B)-\varphi(O)={F['phi'](bx, by):.6f}")
        else:
            st.warning("経路によって値が違う(保存場ではない)。")
        st.latex(r"\vec F\ \text{が保存場}\iff \frac{\partial Q}{\partial x}=\frac{\partial P}{\partial y}"
                 r"\ (\text{単連結な領域で})")

    with st.expander("練習問題"):
        st.markdown("**問3** $\\vec F=(-y,\\ x)$ を、O から B(1,1) まで直線と折れ線(O→(1,0)→B)で積分せよ。")
        st.latex(r"\text{直線:}\int_0^1(-t+t)\,dt=0,\qquad \text{折れ線:}\ 0+\int_0^1 1\,dy=1")
        st.markdown("**問4** $\\vec F=(2xy,\\ x^2)$ の線積分を O から $(2,3)$ まで求めよ。")
        st.latex(r"\varphi=x^2y\ \Rightarrow\ \varphi(2,3)-\varphi(0,0)=12")


# ============================================================
# 面積分の共通部品
# ============================================================
SURF_I = {
    "平面 x + y + z = 1(第1象限、上向き)": dict(
        r=lambda s, t: np.array([s * (1 - t), s * t, 1 - s]), u=(0, 1), v=(0, 1),
        tex=r"\vec r(s,t)=(s(1-t),\ st,\ 1-s)"),
    "上半球面 x²+y²+z²=1, z≥0(外向き)": dict(
        r=lambda u, v: np.array([np.sin(u) * np.cos(v), np.sin(u) * np.sin(v), np.cos(u)]),
        u=(0, PI / 2), v=(0, 2 * PI), tex=r"\vec r(u,v)=(\sin u\cos v,\ \sin u\sin v,\ \cos u)"),
    "円柱の側面 x²+y²=1, 0≤z≤1(外向き)": dict(
        r=lambda v, u: np.array([np.cos(v), np.sin(v), u + 0 * v]), u=(0, 2 * PI), v=(0, 1),
        tex=r"\vec r(\theta,z)=(\cos\theta,\ \sin\theta,\ z)"),
    "放物面 z = 1 − x² − y², z≥0(上向き)": dict(
        r=lambda u, v: np.array([u * np.cos(v), u * np.sin(v), 1 - u ** 2]), u=(0, 1), v=(0, 2 * PI),
        tex=r"\vec r(u,v)=(u\cos v,\ u\sin v,\ 1-u^2)"),
}
F3 = {
    "F = (0, 0, 1)": (lambda x, y, z: (0 * x, 0 * x, 1 + 0 * x), r"\vec F=(0,\ 0,\ 1)"),
    "F = (x, y, z)": (lambda x, y, z: (x, y, z), r"\vec F=(x,\ y,\ z)"),
    "F = (y, −x, z)": (lambda x, y, z: (y, -x, z), r"\vec F=(y,\ -x,\ z)"),
    "F = (0, 0, z)": (lambda x, y, z: (0 * x, 0 * x, z), r"\vec F=(0,\ 0,\ z)"),
    "F = (x², 0, 0)": (lambda x, y, z: (x ** 2, 0 * x, 0 * x), r"\vec F=(x^2,\ 0,\ 0)"),
}
SF3 = {
    "f = 1(曲面積)": (lambda x, y, z: 1 + 0 * x, r"f=1"),
    "f = z": (lambda x, y, z: z, r"f=z"),
    "f = x² + y²": (lambda x, y, z: x ** 2 + y ** 2, r"f=x^2+y^2"),
}


def _surf_grid(S, k=300):
    (u0, u1), (v0, v1) = S["u"], S["v"]
    uu = u0 + (np.arange(k) + 0.5) * (u1 - u0) / k
    vv = v0 + (np.arange(k) + 0.5) * (v1 - v0) / k
    U, V = np.meshgrid(uu, vv)
    h = 1e-6
    R = S["r"](U, V)
    Ru = (S["r"](U + h, V) - S["r"](U - h, V)) / (2 * h)
    Rv = (S["r"](U, V + h) - S["r"](U, V - h)) / (2 * h)
    N = np.cross(Ru, Rv, axis=0)
    dA = (u1 - u0) * (v1 - v0) / k ** 2
    return R, N, dA


def tab_surface_integral():
    st.info("曲面上での積分。スカラー場は $dS=|\\vec r_u\\times\\vec r_v|\\,du\\,dv$、"
            "ベクトル場は法線方向の成分を積分して**流束**(曲面を通り抜ける量)を求める。")
    st.latex(r"\iint_S f\,dS=\iint_D f(\vec r)\,|\vec r_u\times\vec r_v|\,du\,dv,\qquad "
             r"\iint_S\vec F\cdot\vec n\,dS=\iint_D\vec F(\vec r)\cdot(\vec r_u\times\vec r_v)\,du\,dv")
    sname = st.selectbox("曲面 S", list(SURF_I), key="si_s")
    S = SURF_I[sname]
    st.latex(S["tex"])
    c1, c2 = st.columns(2)
    fname = c1.selectbox("スカラー場 f", list(SF3), key="si_f")
    Fname = c2.selectbox("ベクトル場 F", list(F3), index=1, key="si_F")
    f, ftex = SF3[fname]
    F, Ftex = F3[Fname]

    R, N, dA = _surf_grid(S)
    nN = np.linalg.norm(N, axis=0)
    I_f = float(np.sum(f(*R) * nN) * dA)
    Fv = np.array(F(*R))
    I_F = float(np.sum(np.sum(Fv * N, axis=0)) * dA)

    Rg, Ng, _ = _surf_grid(S, 50)
    Fg = np.array(F(*Rg))
    Fn = np.sum(Fg * Ng, axis=0) / np.maximum(np.linalg.norm(Ng, axis=0), 1e-12)
    fig = go.Figure(go.Surface(x=Rg[0], y=Rg[1], z=Rg[2], surfacecolor=Fn, colorscale="RdBu_r",
                               cmid=0, opacity=0.9, colorbar=dict(title="F·n", thickness=12)))
    Rc, Nc, _ = _surf_grid(S, 7)
    Nc = Nc / np.maximum(np.linalg.norm(Nc, axis=0), 1e-12)
    fig.add_trace(cones(*Rc, *Nc, size=0.35, colorscale=[[0, C3], [1, C3]], name="法線 n"))
    Fc = np.array(F(*Rc))
    fig.add_trace(cones(*Rc, *Fc, size=0.5, colorscale=[[0, "#333"], [1, "#333"]], name="F"))
    layout3d(fig, height=520)
    cA, cB = st.columns([3, 2])
    with cA:
        show(fig)
        st.caption("色:F·n(赤=外向き/上向きに通り抜ける、青=逆向き)。緑の円錐=法線、黒=F。")
    with cB:
        st.latex(ftex)
        st.metric("∬_S f dS", f"{I_f:.6f}")
        st.latex(Ftex)
        st.metric("∬_S F·n dS(流束)", f"{I_F:.6f}")

    with st.expander("練習問題"):
        st.markdown("**問5** 平面 $x+y+z=1$ の第1象限の部分 S について、$\\iint_S dS$ を求めよ。")
        st.write("1辺 $\\sqrt2$ の正三角形なので $\\frac{\\sqrt3}{4}\\cdot2=\\frac{\\sqrt3}{2}\\approx0.866025$。")
        st.markdown("**問6** 上半球面 S(外向き)で $\\vec F=(x,y,z)$ の流束を求めよ。")
        st.write("S 上で $\\vec F\\cdot\\vec n=|\\vec r|=1$ なので、流束 = 面積 = $2\\pi$。")


# ============================================================
# 5. ガウスの発散定理
# ============================================================
GF = {
    "F = (x, y, z)": dict(F=lambda x, y, z: (x, y, z), div=lambda x, y, z: 3 + 0 * x,
                          tex=r"\vec F=(x,\ y,\ z),\quad\nabla\cdot\vec F=3"),
    "F = (x², y², z²)": dict(F=lambda x, y, z: (x ** 2, y ** 2, z ** 2), div=lambda x, y, z: 2 * (x + y + z),
                             tex=r"\vec F=(x^2,\ y^2,\ z^2),\quad\nabla\cdot\vec F=2(x+y+z)"),
    "F = (x³, y³, z³)": dict(F=lambda x, y, z: (x ** 3, y ** 3, z ** 3),
                             div=lambda x, y, z: 3 * (x ** 2 + y ** 2 + z ** 2),
                             tex=r"\vec F=(x^3,\ y^3,\ z^3),\quad\nabla\cdot\vec F=3(x^2+y^2+z^2)"),
    "F = (y, z, x)": dict(F=lambda x, y, z: (y, z, x), div=lambda x, y, z: 0 * x,
                          tex=r"\vec F=(y,\ z,\ x),\quad\nabla\cdot\vec F=0"),
    "点源 F = r/|r|³": dict(F=lambda x, y, z: tuple(c / (x * x + y * y + z * z) ** 1.5 for c in (x, y, z)),
                          div=lambda x, y, z: 0 * x, singular=True,
                          tex=r"\vec F=\frac{\vec r}{|\vec r|^3},\quad\nabla\cdot\vec F=0\ (\text{原点以外})"),
}


def tab_gauss():
    st.info("閉曲面から出ていく流束 = 内部の湧き出し(div)の合計。"
            "小さな球で割ると、**div は「単位体積あたりの湧き出し量」**であることがわかる。")
    st.latex(r"\iint_S\vec F\cdot\vec n\,dS=\iiint_V\nabla\cdot\vec F\,dV,\qquad "
             r"\nabla\cdot\vec F(P)=\lim_{V\to P}\frac{1}{|V|}\iint_S\vec F\cdot\vec n\,dS")
    name = st.selectbox("ベクトル場", list(GF), key="ga_f")
    G = GF[name]
    st.latex(G["tex"])
    c = st.columns(5)
    shape = c[0].radio("領域 V", ["球", "立方体"], key="ga_shape")
    cx = c[1].slider("中心 x", -1.0, 1.0, 0.5, 0.1, key="ga_cx")
    cy = c[2].slider("中心 y", -1.0, 1.0, 0.0, 0.1, key="ga_cy")
    cz = c[3].slider("中心 z", -1.0, 1.0, 0.0, 0.1, key="ga_cz")
    R = c[4].slider("半径(立方体は半辺)", 0.05, 1.5, 1.0, 0.05, key="ga_R")
    ctr = np.array([cx, cy, cz])

    if shape == "球":
        k = 300
        th = (np.arange(k) + 0.5) * PI / k
        ph = (np.arange(2 * k) + 0.5) * PI / k
        TH, PH = np.meshgrid(th, ph)
        nvec = np.array([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)])
        X = ctr[:, None, None] + R * nvec
        with np.errstate(all="ignore"):
            Fv = np.array(G["F"](*X))
        flux = float(np.nansum(np.sum(Fv * nvec, axis=0) * R ** 2 * np.sin(TH)) * (PI / k) ** 2)
        m = 80
        rr = (np.arange(m) + 0.5) * R / m
        tt = (np.arange(m) + 0.5) * PI / m
        pp = (np.arange(2 * m) + 0.5) * PI / m
        RR, TT, PP = np.meshgrid(rr, tt, pp, indexing="ij")
        Xv = ctr[0] + RR * np.sin(TT) * np.cos(PP)
        Yv = ctr[1] + RR * np.sin(TT) * np.sin(PP)
        Zv = ctr[2] + RR * np.cos(TT)
        vol_int = float(np.sum(G["div"](Xv, Yv, Zv) * RR ** 2 * np.sin(TT)) * (R / m) * (PI / m) ** 2)
        vol = 4 / 3 * PI * R ** 3
        inside0 = np.linalg.norm(ctr) < R
    else:
        k = 200
        s = -R + (np.arange(k) + 0.5) * 2 * R / k
        A, B = np.meshgrid(s, s)
        flux = 0.0
        for ax in range(3):
            for sg in (1, -1):
                pts = [None, None, None]
                o = [i for i in range(3) if i != ax]
                pts[ax] = ctr[ax] + sg * R + 0 * A
                pts[o[0]] = ctr[o[0]] + A
                pts[o[1]] = ctr[o[1]] + B
                with np.errstate(all="ignore"):
                    Fv = G["F"](*pts)
                flux += float(np.nansum(sg * Fv[ax]) * (2 * R / k) ** 2)
        m = 80
        g = -R + (np.arange(m) + 0.5) * 2 * R / m
        Xv, Yv, Zv = np.meshgrid(ctr[0] + g, ctr[1] + g, ctr[2] + g, indexing="ij")
        vol_int = float(np.sum(G["div"](Xv, Yv, Zv)) * (2 * R / m) ** 3)
        vol = (2 * R) ** 3
        inside0 = np.all(np.abs(ctr) < R)

    # 図
    fig = go.Figure()
    if shape == "球":
        th, ph = np.meshgrid(np.linspace(0, PI, 40), np.linspace(0, 2 * PI, 60))
        nv = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
        Xs = ctr[:, None, None] + R * nv
        with np.errstate(all="ignore"):
            Fn = np.sum(np.array(G["F"](*Xs)) * nv, axis=0)
        fig.add_trace(go.Surface(x=Xs[0], y=Xs[1], z=Xs[2], surfacecolor=Fn, colorscale="RdBu_r", cmid=0,
                                 opacity=0.55, colorbar=dict(title="F·n", thickness=12)))
    else:
        for ax in range(3):
            for sg in (1, -1):
                a, b = np.meshgrid(np.linspace(-R, R, 20), np.linspace(-R, R, 20))
                pts = [None, None, None]
                o = [i for i in range(3) if i != ax]
                pts[ax] = ctr[ax] + sg * R + 0 * a
                pts[o[0]] = ctr[o[0]] + a
                pts[o[1]] = ctr[o[1]] + b
                with np.errstate(all="ignore"):
                    Fn = sg * np.array(G["F"](*pts))[ax]
                fig.add_trace(go.Surface(x=pts[0], y=pts[1], z=pts[2], surfacecolor=Fn, colorscale="RdBu_r",
                                         cmid=0, opacity=0.55, showscale=(ax == 0 and sg == 1),
                                         colorbar=dict(title="F·n", thickness=12)))
    g = np.linspace(-2, 2, 6)
    Xc, Yc, Zc = np.meshgrid(g, g, g)
    with np.errstate(all="ignore"):
        U = np.array(G["F"](Xc, Yc, Zc))
    if G.get("singular"):
        U = U / np.maximum(np.linalg.norm(U, axis=0), 1e-9)
    fig.add_trace(cones(Xc, Yc, Zc, *U, size=0.6, colorscale="Greys", name="F"))
    layout3d(fig, height=520, ranges=[(-2.2, 2.2)] * 3)

    cA, cB = st.columns([3, 2])
    with cA:
        show(fig)
    with cB:
        st.metric("流束 ∬_S F·n dS", f"{flux:.6f}")
        st.metric("∭_V div F dV", f"{vol_int:.6f}")
        if G.get("singular") and inside0:
            st.error("V が原点(F の特異点)を含むので両辺は一致しない。流束は 4π ≈ 12.566 "
                     "(原点に点状の湧き出しが集中している:ガウスの法則)。")
        st.markdown("**発散の意味**")
        st.latex(rf"\frac{{\text{{流束}}}}{{|V|}}={flux / vol:.5f}")
        with np.errstate(all="ignore"):
            dv = float(G["div"](*ctr))
        st.latex(rf"\nabla\cdot\vec F(\text{{中心}})={dv:.5f}")
        st.caption("半径を小さくすると、流束 ÷ 体積 が中心での div に近づく。")

    with st.expander("練習問題"):
        st.markdown("**問7** 半径 a の球面 S(外向き)で $\\vec F=(x,y,z)$ の流束を求めよ。")
        st.latex(r"\iiint_V 3\,dV=3\cdot\frac43\pi a^3=4\pi a^3")
        st.markdown("**問8** 立方体 $0\\le x,y,z\\le1$ の表面での $\\vec F=(x^2,y^2,z^2)$ の流束を求めよ。")
        st.latex(r"\iiint_V 2(x+y+z)\,dV=2\cdot3\cdot\frac12=3")


# ============================================================
# 6. ストークスの定理
# ============================================================
SF = {
    "F = (−y, x, 0)": lambda x, y, z: (-y, x, 0 * x),
    "F = (−y, x, z)": lambda x, y, z: (-y, x, z),
    "F = (y, z, x)": lambda x, y, z: (y, z, x),
    "F = (z, x, y)": lambda x, y, z: (z, x, y),
    "F = (−y³, x³, 0)": lambda x, y, z: (-y ** 3, x ** 3, 0 * x),
    "F = (yz, xz, xy) = ∇(xyz)": lambda x, y, z: (y * z, x * z, x * y),
}
STEX = {
    "F = (−y, x, 0)": r"\nabla\times\vec F=(0,\ 0,\ 2)",
    "F = (−y, x, z)": r"\nabla\times\vec F=(0,\ 0,\ 2)",
    "F = (y, z, x)": r"\nabla\times\vec F=(-1,\ -1,\ -1)",
    "F = (z, x, y)": r"\nabla\times\vec F=(1,\ 1,\ 1)",
    "F = (−y³, x³, 0)": r"\nabla\times\vec F=(0,\ 0,\ 3x^2+3y^2)",
    "F = (yz, xz, xy) = ∇(xyz)": r"\nabla\times\vec F=\vec 0",
}


def curl(F, x, y, z, h=1e-5):
    def d(i, ax):
        e = [0, 0, 0]
        e[ax] = h
        return (F(x + e[0], y + e[1], z + e[2])[i] - F(x - e[0], y - e[1], z - e[2])[i]) / (2 * h)
    return np.array([d(2, 1) - d(1, 2), d(0, 2) - d(2, 0), d(1, 0) - d(0, 1)])


def tab_stokes():
    st.info("閉曲線 C に沿った循環 = C を縁とする曲面 S 上の rot の流束。"
            "**縁が同じなら、どんな曲面でも値は同じ**。rot は「その点での単位面積あたりの回転の強さと軸」を表す。")
    st.latex(r"\oint_C\vec F\cdot d\vec r=\iint_S(\nabla\times\vec F)\cdot\vec n\,dS")
    name = st.selectbox("ベクトル場", list(SF), key="sk_f")
    F = SF[name]
    st.latex(STEX[name])
    c1, c2, c3 = st.columns(3)
    surf = c1.radio("曲面 S", ["円板", "上半球面", "放物面(高さ h)"], key="sk_s")
    R = c2.slider("縁の円の半径 R", 0.3, 1.5, 1.0, 0.1, key="sk_R")
    hh = c3.slider("放物面の高さ h", -1.5, 1.5, 0.8, 0.1, key="sk_h", disabled=not surf.startswith("放物"))

    if surf == "円板":
        r = lambda u, v: np.array([u * np.cos(v), u * np.sin(v), 0 * u])  # noqa: E731
        ur = (0, R)
    elif surf == "上半球面":
        r = lambda u, v: np.array([R * np.sin(u) * np.cos(v), R * np.sin(u) * np.sin(v), R * np.cos(u)])  # noqa
        ur = (0, PI / 2)
    else:
        r = lambda u, v: np.array([u * np.cos(v), u * np.sin(v), hh * (1 - u ** 2 / R ** 2)])  # noqa
        ur = (0, R)
    S = dict(r=r, u=ur, v=(0, 2 * PI))
    Rg, N, dA = _surf_grid(S, 300)
    surf_int = float(np.sum(np.sum(curl(F, *Rg) * N, axis=0)) * dA)
    t = np.linspace(0, 2 * PI, 8001)
    x, y, z = R * np.cos(t), R * np.sin(t), 0 * t
    Fv = F(x, y, z)
    line_int = float(trapz(Fv[0] * (-R * np.sin(t)) + Fv[1] * (R * np.cos(t)), t))

    Rs, _, _ = _surf_grid(S, 40)
    fig = go.Figure(go.Surface(x=Rs[0], y=Rs[1], z=Rs[2], opacity=0.5, colorscale="Blues", showscale=False))
    fig.add_trace(go.Scatter3d(x=x[::40], y=y[::40], z=z[::40], mode="lines", line=dict(color=C2, width=7),
                               name="縁 C"))
    for k in range(0, 8000, 1000):
        for tr in arrow3d([x[k], y[k], 0], 0.3 * np.array([-np.sin(t[k]), np.cos(t[k]), 0]), C2, "C の向き",
                          showlegend=(k == 0)):
            fig.add_trace(tr)
    Rc, _, _ = _surf_grid(S, 6)
    Cc = curl(F, *Rc)
    if np.abs(Cc).max() > 1e-8:
        fig.add_trace(cones(*Rc, *Cc, size=0.5, colorscale=[[0, C3], [1, C3]], name="rot F"))
    layout3d(fig, height=500)
    cA, cB = st.columns([3, 2])
    with cA:
        show(fig)
    with cB:
        st.metric("循環 ∮_C F·dr", f"{line_int:.6f}")
        st.metric("∬_S (rot F)·n dS", f"{surf_int:.6f}")
        st.caption("曲面を円板・半球・放物面と取りかえても、右辺は変わらない(縁 C が同じだから)。緑の円錐は rot F。")

    st.divider()
    st.subheader("回転の意味:小さな円での循環")
    st.write("点 P のまわりに、単位法線 n に垂直な半径 ε の小円をとる。循環 ÷ 面積 は (rot F)·n に近づき、"
             "**n が rot F の向きのとき最大**になる(水車の軸を回す向き)。")
    d = st.columns(5)
    px_ = d[0].slider("P の x", -1.0, 1.0, 0.5, 0.1, key="sk_px")
    py_ = d[1].slider("P の y", -1.0, 1.0, 0.3, 0.1, key="sk_py")
    pz_ = d[2].slider("P の z", -1.0, 1.0, 0.0, 0.1, key="sk_pz")
    th = np.radians(d[3].slider("軸 n の天頂角 θ(度)", 0, 180, 0, 5, key="sk_th"))
    ph = np.radians(d[4].slider("軸 n の方位角 φ(度)", 0, 355, 0, 5, key="sk_ph"))
    P = np.array([px_, py_, pz_])
    n = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    e1 = np.cross(n, [0, 0, 1]) if abs(n[2]) < 0.9 else np.cross(n, [1, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    eps = 0.05
    tt = np.linspace(0, 2 * PI, 4001)
    pts = P[:, None] + eps * (np.cos(tt)[None] * e1[:, None] + np.sin(tt)[None] * e2[:, None])
    dr = eps * (-np.sin(tt)[None] * e1[:, None] + np.cos(tt)[None] * e2[:, None])
    Fp = np.array(F(*pts))
    circ = float(trapz(np.sum(Fp * dr, axis=0), tt))
    rotP = curl(F, *P)
    m1, m2, m3 = st.columns(3)
    m1.metric("循環 ÷ 面積(ε = 0.05)", f"{circ / (PI * eps ** 2):.4f}")
    m2.metric("(rot F)(P)·n", f"{rotP @ n:.4f}")
    m3.metric("|rot F(P)|(最大値)", f"{np.linalg.norm(rotP):.4f}")
    st.caption(f"rot F(P) = ({rotP[0]:.3f}, {rotP[1]:.3f}, {rotP[2]:.3f})。θ, φ を動かして n を rot F に合わせると、"
               "左の値が最大になる。")

    with st.expander("練習問題"):
        st.markdown("**問9** $\\vec F=(-y,\\ x,\\ 0)$、C:$x^2+y^2=a^2,\\ z=0$(反時計回り)のとき、両辺を計算せよ。")
        st.latex(r"\oint_C\vec F\cdot d\vec r=\int_0^{2\pi}a^2\,dt=2\pi a^2,\qquad "
                 r"\iint_S(0,0,2)\cdot\vec n\,dS=2\cdot\pi a^2")
        st.markdown("**問10** $\\vec F=\\nabla\\varphi$ のとき、任意の閉曲線で循環が 0 になることを、"
                    "ストークスの定理から説明せよ。")
        st.write("$\\nabla\\times\\nabla\\varphi=\\vec0$ なので右辺が 0。")


# ============================================================
# 7. 勾配の意味
# ============================================================
def tab_gradient_meaning():
    st.info("山の地図(等高線)で、いつも一番急な方向に登ると、その道は**等高線と直交**しながら頂上へ向かう。"
            "3次元では、∇F は等位面 F = c の**法線ベクトル**になる。")
    st.latex(r"\frac{d\vec r}{ds}=\frac{\nabla f}{|\nabla f|}\quad(\text{最急上昇曲線}),\qquad "
             r"\nabla F\perp\{F=c\}")
    name = st.selectbox("スカラー場 f(山の高さ)", list(SCALARS), index=3, key="gm_f")
    Sf = SCALARS[name]
    R = Sf["R"]
    c1, c2, c3 = st.columns(3)
    x0 = c1.slider("出発点 x", -R, R, -0.4 * R, 0.05, key="gm_x")
    y0 = c2.slider("出発点 y", -R, R, 0.3 * R, 0.05, key="gm_y")
    mode = c3.radio("進む向き", ["登る(+∇f)", "下る(−∇f)"], key="gm_dir")
    sgn = 1 if mode.startswith("登") else -1

    pts = [np.array([x0, y0])]
    h = 0.01 * R
    for _ in range(3000):
        p = pts[-1]
        g = np.array([Sf["fx"](*p), Sf["fy"](*p)], float)
        ng = np.linalg.norm(g)
        if ng < 1e-4 or np.abs(p).max() > R:
            break
        k1 = sgn * g / ng
        q = p + h / 2 * k1
        g2 = np.array([Sf["fx"](*q), Sf["fy"](*q)], float)
        if np.linalg.norm(g2) < 1e-9:
            break
        pts.append(p + h * sgn * g2 / np.linalg.norm(g2))
    pts = np.array(pts)

    g = np.linspace(-R, R, 160)
    X, Y = np.meshgrid(g, g)
    fig = go.Figure(go.Contour(x=g, y=g, z=Sf["f"](X, Y), ncontours=24, colorscale="Earth", opacity=0.75,
                               contours=dict(coloring="heatmap", showlines=True), showscale=False,
                               hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines", line=dict(color=C2, width=4),
                             name="最急上昇/下降曲線"))
    fig.add_trace(go.Scatter(x=[x0], y=[y0], mode="markers", marker=dict(size=11, color="black"), name="出発点"))
    layout(fig, height=500, equal=True, xlab="x", ylab="y", xrange=(-R, R), yrange=(-R, R))

    cA, cB = st.columns(2)
    with cA:
        show(fig)
    with cB:
        f3 = go.Figure(go.Surface(x=g, y=g, z=Sf["f"](X, Y), colorscale="Earth", opacity=0.8, showscale=False))
        f3.add_trace(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=Sf["f"](pts[:, 0], pts[:, 1]) + 0.01,
                                  mode="lines", line=dict(color=C2, width=7), name="道"))
        layout3d(f3, height=500, aspect="cube")
        show(f3)
    st.write(f"到着点:({pts[-1, 0]:.3f}, {pts[-1, 1]:.3f})、高さ f = {Sf['f'](*pts[-1]):.4f}"
             "(∇f = 0 の点:山頂・谷底、または鞍点の近くで止まる)")

    st.divider()
    st.subheader("3次元:等位面と法線")
    st.latex(r"F(x,y,z)=\frac{x^2}{a^2}+\frac{y^2}{b^2}+\frac{z^2}{c^2}=1,\qquad "
             r"\nabla F=\left(\frac{2x}{a^2},\ \frac{2y}{b^2},\ \frac{2z}{c^2}\right)")
    d = st.columns(5)
    a = d[0].slider("a", 0.5, 2.0, 1.5, 0.1, key="gm_a")
    b = d[1].slider("b", 0.5, 2.0, 1.0, 0.1, key="gm_b")
    cc = d[2].slider("c", 0.5, 2.0, 0.8, 0.1, key="gm_c")
    u = np.radians(d[3].slider("点の天頂角(度)", 5, 175, 60, 5, key="gm_u"))
    v = np.radians(d[4].slider("点の方位角(度)", 0, 355, 30, 5, key="gm_v"))
    P = np.array([a * np.sin(u) * np.cos(v), b * np.sin(u) * np.sin(v), cc * np.cos(u)])
    gF = np.array([2 * P[0] / a ** 2, 2 * P[1] / b ** 2, 2 * P[2] / cc ** 2])
    U, V = np.meshgrid(np.linspace(0, PI, 40), np.linspace(0, 2 * PI, 60))
    f4 = go.Figure(go.Surface(x=a * np.sin(U) * np.cos(V), y=b * np.sin(U) * np.sin(V), z=cc * np.cos(U),
                              opacity=0.55, colorscale="Blues", showscale=False))
    for tr in arrow3d(P, gF / np.linalg.norm(gF) * 0.9, C2, "∇F(法線)"):
        f4.add_trace(tr)
    e1 = np.cross(gF, [0, 0, 1])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.array([1.0, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(gF / np.linalg.norm(gF), e1)
    A_, B_ = np.meshgrid([-0.7, 0.7], [-0.7, 0.7])
    Pl = P[:, None, None] + A_ * e1[:, None, None] + B_ * e2[:, None, None]
    f4.add_trace(go.Surface(x=Pl[0], y=Pl[1], z=Pl[2], opacity=0.5, colorscale=[[0, C4], [1, C4]],
                            showscale=False, name="接平面"))
    layout3d(f4, height=460)
    cA, cB = st.columns([3, 2])
    with cA:
        show(f4)
    with cB:
        st.latex(rf"P={vec_tex(P)}")
        st.latex(rf"\nabla F(P)={vec_tex(gF)}")
        st.markdown("**接平面**")
        st.latex(rf"\frac{{x_0x}}{{a^2}}+\frac{{y_0y}}{{b^2}}+\frac{{z_0z}}{{c^2}}=1")
        st.caption("∇F·(X − P) = 0 を整理すると、この形になる(第1章「曲面」の接平面と同じもの)。")

    with st.expander("練習問題"):
        st.markdown("**問11** 曲面 $x^2+2y^2+3z^2=6$ の点 $(1,1,1)$ における法線ベクトルと接平面を求めよ。")
        st.latex(r"\nabla F=(2x,\ 4y,\ 6z)=(2,\ 4,\ 6),\qquad 2(x-1)+4(y-1)+6(z-1)=0\ \Leftrightarrow\ x+2y+3z=6")


# ============================================================
# 8. 楕円の内部での光の反射
# ============================================================
def reflect_path(a, b, p, d, nref):
    """楕円 x²/a²+y²/b²=1 の内部で、点 p から方向 d に出た光の反射点の列"""
    pts, dirs, normals = [np.array(p, float)], [np.array(d, float) / np.linalg.norm(d)], []
    for _ in range(nref):
        p, d = pts[-1], dirs[-1]
        A = d[0] ** 2 / a ** 2 + d[1] ** 2 / b ** 2
        B = 2 * (p[0] * d[0] / a ** 2 + p[1] * d[1] / b ** 2)
        C = p[0] ** 2 / a ** 2 + p[1] ** 2 / b ** 2 - 1
        t = (-B + np.sqrt(max(B * B - 4 * A * C, 0))) / (2 * A)
        q = p + t * d
        n = np.array([2 * q[0] / a ** 2, 2 * q[1] / b ** 2])
        d2 = d - 2 * (d @ n) / (n @ n) * n
        pts.append(q)
        dirs.append(d2 / np.linalg.norm(d2))
        normals.append(n / np.linalg.norm(n))
    return np.array(pts), np.array(dirs), np.array(normals)


PYTH = [(Fr(3, 5), Fr(4, 5)), (Fr(4, 5), Fr(3, 5)), (Fr(5, 13), Fr(12, 13)), (Fr(12, 13), Fr(5, 13)),
        (Fr(8, 17), Fr(15, 17)), (Fr(15, 17), Fr(8, 17))]
ELL = [(5, 4), (5, 3), (10, 8), (10, 6), (13, 12), (13, 5), (17, 15), (17, 8), (25, 24), (25, 7)]


def _f(x):
    x = Fr(x)
    return f"{x.numerator}" if x.denominator == 1 else rf"\frac{{{x.numerator}}}{{{x.denominator}}}" \
        if x >= 0 else rf"-\frac{{{-x.numerator}}}{{{x.denominator}}}"


def _vf(v):
    return r"\left(" + r",\ ".join(_f(c) for c in v) + r"\right)"


def tab_ellipse():
    st.info("楕円の内部で光を放つと、壁で反射を繰り返す。反射の法則は**法線ベクトル = 勾配**を使って書ける。"
            "焦点から出た光は、1回反射するともう一方の焦点を通る。")
    st.latex(r"g(x,y)=\frac{x^2}{a^2}+\frac{y^2}{b^2},\quad \vec n=\nabla g=\left(\frac{2x}{a^2},\ \frac{2y}{b^2}\right),"
             r"\quad \vec d\,'=\vec d-2\frac{\vec d\cdot\vec n}{\vec n\cdot\vec n}\,\vec n\ (\text{鏡映})")
    c = st.columns(4)
    a = c[0].slider("長半径 a", 1.0, 3.0, 2.5, 0.1, key="el_a")
    b = c[1].slider("短半径 b", 0.5, 3.0, 1.5, 0.1, key="el_b")
    nref = c[2].slider("反射の回数", 1, 40, 3, key="el_n")
    th = c[3].slider("放つ向き(度)", 0, 359, 50, key="el_th")
    a, b = max(a, b), min(a, b)
    cfoc = np.sqrt(a * a - b * b)
    start = st.radio("出発点", ["焦点 F₁(−c, 0)", "自由に決める"], horizontal=True, key="el_start")
    if start.startswith("焦点"):
        p0 = np.array([-cfoc, 0.0])
    else:
        e1, e2 = st.columns(2)
        p0 = np.array([e1.slider("出発点 x", -0.9, 0.9, 0.3, 0.05, key="el_px") * a,
                       e2.slider("出発点 y", -0.9, 0.9, 0.2, 0.05, key="el_py") * b])
        if p0[0] ** 2 / a ** 2 + p0[1] ** 2 / b ** 2 >= 1:
            st.warning("出発点が楕円の外です。内側に戻してください。")
            return
    d0 = np.array([np.cos(np.radians(th)), np.sin(np.radians(th))])
    pts, dirs, normals = reflect_path(a, b, p0, d0, nref)

    tt = np.linspace(0, 2 * PI, 400)
    fig = go.Figure(go.Scatter(x=a * np.cos(tt), y=b * np.sin(tt), mode="lines", line=dict(color=C1, width=3),
                               name="楕円"))
    fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines+markers", line=dict(color=C2, width=2),
                             marker=dict(size=5), name="光の道筋"))
    for k, (q, n) in enumerate(zip(pts[1:4], normals[:3])):
        fig.add_trace(arrow(q[0], q[1], -0.5 * n[0], -0.5 * n[1], color=C3, name="法線 ∇g" if k == 0 else "",
                            width=2))
    fig.add_trace(go.Scatter(x=[-cfoc, cfoc], y=[0, 0], mode="markers+text", text=["F₁", "F₂"],
                             textposition="bottom center", marker=dict(size=10, color="black", symbol="star"),
                             name="焦点"))
    layout(fig, height=480, equal=True, xlab="x", ylab="y", xrange=(-a - 0.3, a + 0.3), yrange=(-b - 0.3, b + 0.3))
    cA, cB = st.columns([3, 2])
    with cA:
        show(fig)
    with cB:
        rows = [{"回": k, "反射点 x": f"{q[0]:.4f}", "反射点 y": f"{q[1]:.4f}",
                 "反射後の向き": f"({d[0]:.3f}, {d[1]:.3f})"} for k, (q, d) in enumerate(zip(pts[1:], dirs[1:]), 1)]
        st.dataframe(rows[:12], hide_index=True, width="stretch")
        if start.startswith("焦点"):
            q, d = pts[1], dirs[1]
            dist = abs((cfoc - q[0]) * d[1] - (0 - q[1]) * d[0])
            st.success(f"1回目の反射後の光線と F₂ の距離:{dist:.2e}(= 0、F₂ を通る)")
        st.caption("反射を繰り返すと、焦点から出た光は長軸に近づいていく。焦点以外から出すと、"
                   "道筋は共焦点の楕円や双曲線に接するように並ぶ。")

    st.divider()
    st.subheader("サンプル問題(乱数で数値を変えて出題)")
    if "el_seed" not in st.session_state:
        st.session_state["el_seed"] = 1
    if st.button("新しい問題をつくる", key="el_new"):
        st.session_state["el_seed"] += 1
    rng = np.random.default_rng(st.session_state["el_seed"])
    A_, B_ = ELL[rng.integers(len(ELL))]
    C_ = int(round(np.sqrt(A_ * A_ - B_ * B_)))
    px, py = PYTH[rng.integers(len(PYTH))]
    sx, sy = int(rng.choice([1, -1])), int(rng.choice([1, -1]))
    A_, B_ = int(A_), int(B_)
    Q = (sx * A_ * px, sy * B_ * py)
    st.markdown(f"楕円 $\\dfrac{{x^2}}{{{A_}^2}}+\\dfrac{{y^2}}{{{B_}^2}}=1$ の焦点 $F_1({-C_},\\ 0)$ から、"
                f"楕円上の点 $Q{_vf(Q)}$ に向けて光を放つ。")
    st.markdown("1. $g(x,y)=\\dfrac{x^2}{a^2}+\\dfrac{y^2}{b^2}$ とおき、Q における法線ベクトル $\\nabla g$ を求めよ。\n"
                "2. 入射ベクトル $\\vec d=\\overrightarrow{F_1Q}$ と、反射後の方向ベクトル $\\vec d\\,'$ を求めよ。\n"
                "3. 反射した光がもう一方の焦点 $F_2$ を通ることを示せ。")
    st.caption("ヒント(鏡映):法線ベクトル $\\vec n$ の直線で反射すると "
               "$\\vec d\\,'=\\vec d-2\\dfrac{\\vec d\\cdot\\vec n}{\\vec n\\cdot\\vec n}\\vec n$ となる。")
    with st.expander("解答"):
        n = (Fr(2) * Q[0] / A_ ** 2, Fr(2) * Q[1] / B_ ** 2)
        d = (Q[0] + C_, Q[1])
        k = (d[0] * n[0] + d[1] * n[1]) / (n[0] ** 2 + n[1] ** 2)
        d2 = (d[0] - 2 * k * n[0], d[1] - 2 * k * n[1])
        QF2 = (C_ - Q[0], -Q[1])
        ratio = d2[0] / QF2[0] if QF2[0] != 0 else d2[1] / QF2[1]
        st.latex(r"1.\ \nabla g(Q)=\left(\frac{2x}{a^2},\ \frac{2y}{b^2}\right)=" + _vf(n))
        st.latex(r"2.\ \vec d=\overrightarrow{F_1Q}=" + _vf(d) + r",\quad \frac{\vec d\cdot\vec n}{\vec n\cdot\vec n}="
                 + _f(k) + r",\quad \vec d\,'=" + _vf(d2))
        st.latex(r"3.\ \overrightarrow{QF_2}=" + _vf(QF2) + r",\quad \vec d\,'=" + _f(ratio)
                 + r"\,\overrightarrow{QF_2}\ (\text{平行で同じ向き})")
        st.caption("Q から $\\vec d\\,'$ の向きに進むと F₂ に着く。")
