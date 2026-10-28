"""① ベクトル解析 第3章 線積分・面積分(グリーンの定理は vector.py)"""
import math
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
        st.markdown("**問2** C:単位円のとき $\\int_C (x^2+y^2)\\,ds$ を求めよ。")
        if st.toggle("解答・解説を表示", key="pr_vec_integrals_75"):
            st.markdown("**答1**")
            st.latex(r"\int_0^1 2t\cdot\sqrt2\,dt=\sqrt2\approx1.414214")
            st.markdown("**答2**")
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
    with st.expander("保存場とは", expanded=True):
        st.markdown(
            "ベクトル場 $\\vec F$ が、あるスカラー場 $\\varphi$ の勾配として $\\vec F=\\nabla\\varphi$ と書けるとき、"
            "$\\vec F$ を**保存場**(保存力の場)、$\\varphi$ を**ポテンシャル**という"
            "(物理では $\\vec F=-\\nabla U$ と書き、$U$ を位置エネルギーとよぶ)。"
        )
        st.latex(r"\int_C\nabla\varphi\cdot d\vec r=\int_a^b\frac{d}{dt}\varphi(\vec r(t))\,dt"
                 r"=\varphi(\mathrm B)-\varphi(\mathrm A)")
        st.markdown(
            "なので、保存場では次が成り立つ(単連結な領域では、これらはすべて同値)。\n"
            "1. 線積分(仕事)は**始点と終点だけで決まり、途中の経路によらない**\n"
            "2. どんな閉曲線を一周しても $\\oint_C\\vec F\\cdot d\\vec r=0$(一周すると、した仕事が 0 に戻る)\n"
            "3. $\\mathrm{rot}\\,\\vec F=\\vec 0$(平面なら $\\dfrac{\\partial Q}{\\partial x}=\\dfrac{\\partial P}{\\partial y}$)\n\n"
            "「保存」の名前は、力学的エネルギー(運動エネルギー + 位置エネルギー)が保存されることから。"
            "重力や静電気力は保存力、摩擦力は保存力ではない(同じ点に戻っても、経路が長いほど多くの仕事が失われる)。"
        )
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
        st.markdown("**問4** $\\vec F=(2xy,\\ x^2)$ の線積分を O から $(2,3)$ まで求めよ。")
        if st.toggle("解答・解説を表示", key="pr_vec_integrals_162"):
            st.markdown("**答3**")
            st.latex(r"\text{直線:}\int_0^1(-t+t)\,dt=0,\qquad \text{折れ線:}\ 0+\int_0^1 1\,dy=1")
            st.markdown("**答4**")
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
    with st.expander("流束とは", expanded=True):
        st.markdown(
            "$\\vec F$ を水の流れの速度と考える。曲面 $S$ を網のように置いたとき、"
            "**単位時間に網を通り抜ける水の量**が流束(フラックス)である。\n"
            "- 小さな面 $dS$ を通り抜ける量は、流れの**法線方向の成分** $\\vec F\\cdot\\vec n$ と面積の積 "
            "$\\vec F\\cdot\\vec n\\,dS$(面に沿う成分は網を通り抜けない)\n"
            "- 流れが面に垂直なら最大、面に平行なら 0、$\\vec n$ と逆向きに通り抜けるなら負\n"
            "- これを曲面全体で足し合わせたものが流束 $\\displaystyle\\iint_S\\vec F\\cdot\\vec n\\,dS$\n\n"
            "流束の符号は、どちら側を表とするか(単位法線 $\\vec n$ の向き)で決まる。"
            "閉曲面では外向きを正にとり、「中から出ていく量」を表す。電場なら電気力線の本数、熱なら熱の流れの量にあたる。"
        )
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
        st.markdown("**問6** 上半球面 S(外向き)で $\\vec F=(x,y,z)$ の流束を求めよ。")
        if st.toggle("解答・解説を表示", key="pr_vec_integrals_266"):
            st.markdown("**答5**")
            st.write("1辺 $\\sqrt2$ の正三角形なので $\\frac{\\sqrt3}{4}\\cdot2=\\frac{\\sqrt3}{2}\\approx0.866025$。")
            st.markdown("**答6**")
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
        st.markdown("**問8** 立方体 $0\\le x,y,z\\le1$ の表面での $\\vec F=(x^2,y^2,z^2)$ の流束を求めよ。")
        if st.toggle("解答・解説を表示", key="pr_vec_integrals_400"):
            st.markdown("**答7**")
            st.latex(r"\iiint_V 3\,dV=3\cdot\frac43\pi a^3=4\pi a^3")
            st.markdown("**答8**")
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
        st.markdown("**問10** $\\vec F=\\nabla\\varphi$ のとき、任意の閉曲線で循環が 0 になることを、"
                    "ストークスの定理から説明せよ。")
        if st.toggle("解答・解説を表示", key="pr_vec_integrals_515"):
            st.markdown("**答9**")
            st.latex(r"\oint_C\vec F\cdot d\vec r=\int_0^{2\pi}a^2\,dt=2\pi a^2,\qquad "
                     r"\iint_S(0,0,2)\cdot\vec n\,dS=2\cdot\pi a^2")
            st.markdown("**答10**")
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
        if st.toggle("解答・解説を表示", key="pr_vec_integrals_616"):
            st.markdown("**答11**")
            st.latex(r"\nabla F=(2x,\ 4y,\ 6z)=(2,\ 4,\ 6),\qquad 2(x-1)+4(y-1)+6(z-1)=0\ \Leftrightarrow\ x+2y+3z=6")


# ============================================================
# 8. 楕円の内部での光の反射
# ============================================================
def reflect_path(a, b, p, d, nref, foci=None):
    """楕円 x²/a²+y²/b²=1 の内部で、点 p から方向 d に出た光が壁に当たる点の列(nref 回分)。
    foci=(F_自分, F_相手) を与えると、焦点を出た光として「反射後は必ずもう一方の焦点へ向かう」性質を使って
    向きを決める(丸め誤差が積み重なって長軸からずれるのを防ぐ)。"""
    pts, dirs, normals = [np.array(p, float)], [np.array(d, float) / np.linalg.norm(d)], []
    for k in range(nref):
        p, d = pts[-1], dirs[-1]
        A = d[0] ** 2 / a ** 2 + d[1] ** 2 / b ** 2
        B = 2 * (p[0] * d[0] / a ** 2 + p[1] * d[1] / b ** 2)
        C = p[0] ** 2 / a ** 2 + p[1] ** 2 / b ** 2 - 1
        t = (-B + np.sqrt(max(B * B - 4 * A * C, 0))) / (2 * A)
        q = p + t * d
        n = np.array([2 * q[0] / a ** 2, 2 * q[1] / b ** 2])
        if foci is not None:
            target = np.array(foci[(k + 1) % 2], float)  # 1回目の反射後は相手の焦点、2回目は元の焦点…
            d2 = target - q
        else:
            d2 = d - 2 * (d @ n) / (n @ n) * n
        pts.append(q)
        dirs.append(d2 / np.linalg.norm(d2))
        normals.append(n / np.linalg.norm(n))
    return np.array(pts), np.array(dirs), np.array(normals)


def _light_animation(path, static, a, b, frames=120, title=None, height=480):
    """折れ線 path に沿って光の点が進むアニメーション(通った跡を残す)"""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.r_[0, np.cumsum(seg)]
    total = cum[-1]
    ss = np.linspace(0, total, frames)

    def upto(sv):
        k = int(np.searchsorted(cum, sv, side="right")) - 1
        k = min(max(k, 0), len(seg) - 1)
        r = (sv - cum[k]) / seg[k] if seg[k] > 0 else 0
        head = path[k] + r * (path[k + 1] - path[k])
        trail = np.vstack([path[:k + 1], head])
        return trail, head

    def tr(sv):
        trail, head = upto(sv)
        return [go.Scatter(x=trail[:, 0], y=trail[:, 1], mode="lines", line=dict(color=C2, width=2.5), name="光の道筋"),
                go.Scatter(x=[head[0]], y=[head[1]], mode="markers",
                           marker=dict(color="#ffcc00", size=14, line=dict(color=C2, width=2)), name="光")]

    fig = go.Figure(data=tr(0.0) + list(static),
                    frames=[go.Frame(data=tr(sv), name=str(i), traces=[0, 1]) for i, sv in enumerate(ss)])
    fig.update_layout(
        updatemenus=[dict(type="buttons", showactive=False, x=0, y=-0.1, xanchor="left", yanchor="top",
                          direction="left",
                          buttons=[dict(label="▶ 光を放つ", method="animate",
                                        args=[None, dict(frame=dict(duration=40, redraw=False), fromcurrent=True,
                                                         transition=dict(duration=0))]),
                                   dict(label="■ 停止", method="animate",
                                        args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])])],
        sliders=[dict(active=0, x=0.25, y=-0.06, len=0.75, currentvalue=dict(prefix="進んだ距離の割合 "),
                      steps=[dict(method="animate", label=f"{i / (frames - 1):.0%}" if i % 10 == 0 else "",
                                  args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=False),
                                                       transition=dict(duration=0))])
                             for i in range(frames)])],
    )
    layout(fig, title=title, height=height, equal=True, xlab="x", ylab="y",
           xrange=(-a * 1.12, a * 1.12), yrange=(-b * 1.15, b * 1.15))
    fig.update_layout(margin=dict(b=100))
    return fig


# 楕円 x^2/p + y^2/q = 1 (p - q = c^2) と, その上の格子点 Q(x0, y0) (第1象限).
# 法線を整数ベクトルにすると, 反射の計算に出てくる数はすべて 30 以下の整数になる組だけを選んである.
# Q を左半分 (x < 0) に移すと d・n / n・n が 6/5 などになるので, 出題は右半分 (符号は y のみ) に限る.
# 焦点が y 軸上の楕円は, この表の x と y を入れかえて作る (計算に出てくる数は同じ).
ELL_CASES = [(45, 36, 3, (5, 4)), (40, 24, 4, (5, 3)), (40, 15, 5, (4, 3)), (16, 12, 2, (2, 3)),
             (18, 2, 4, (3, 1)), (45, 20, 5, (3, 4)), (18, 9, 3, (4, 1)), (45, 9, 6, (5, 2)), (72, 36, 6, (8, 2))]

# 斜めの楕円 A x^2 + B xy + C y^2 = D. 焦点 F1 = (-u, -v), F2 = (u, v), 長半径^2 = a2.
# 焦点 ±f (f は整数ベクトル) の楕円は a2 |P|^2 - (P・f)^2 = a2 (a2 - |f|^2) と書けるので, 係数はすべて整数になる.
# Q は対称軸の上にない格子点で, ∇g は整数ベクトル, d・n / n・n は整数になる組だけを選んである.
# (x^2 + xy + y^2 = D の形は, 焦点と Q がともに格子点になる D がないので入れていない)
TILT_CASES = [(5, 8, 5, 9, 9, (2, -2), (1, -2)), (5, 8, 5, 9, 9, (2, -2), (2, -1)),
              (2, 1, 2, 30, 20, (2, -2), (4, -1)), (2, 1, 2, 30, 20, (2, -2), (-1, 4)),
              (17, 8, 2, 18, 18, (1, -4), (0, -3)), (2, 8, 17, 18, 18, (4, -1), (3, 0)),
              (7, 4, 4, 24, 8, (1, -2), (2, -1)), (4, 4, 7, 24, 8, (2, -1), (1, -2)),
              (8, 4, 5, 36, 9, (1, -2), (2, -2)), (5, 4, 8, 36, 9, (2, -1), (2, -2))]
ELL_KINDS = {"x": "焦点が x 軸上", "y": "焦点が y 軸上", "tilt": "斜めの楕円"}


def prim_int(v):
    """有理数のベクトルを, 同じ向きの原始的な整数ベクトルに"""
    v = [Fr(x) for x in v]
    L = 1
    for x in v:
        L = L * x.denominator // math.gcd(L, x.denominator)
    w = [int(x * L) for x in v]
    g = math.gcd(*[abs(x) for x in w])
    return [x // g for x in w]


def _mono(c, var, first):
    """係数 c の項 c·var を TeX に(先頭でなければ符号つき)"""
    if c == 0:
        return ""
    sign = "-" if c < 0 else ("" if first else "+")
    m = abs(c)
    return sign + ("" if m == 1 and var else str(m)) + var


def _poly(terms):
    out, first = "", True
    for c, var in terms:
        t = _mono(c, var, first)
        if t:
            out, first = out + t, False
    return out or "0"


def ellipse_count(kind):
    return len(TILT_CASES) if kind == "tilt" else len(ELL_CASES)


def ellipse_case(kind, pick, s=1):
    """反射の問題 1 題分のデータ. kind は "x", "y", "tilt". s = ±1 で y 方向(y 軸型は x 方向)に反転する.
    F1 から Q に向けて光を放ち, Q で反射した光が F2 を通る."""
    from types import SimpleNamespace
    if kind in ("x", "y"):
        p, q, c, (x0, y0) = ELL_CASES[pick]
        if kind == "x":
            P_, Qd, F2, Q, theta = p, q, (c, 0), (x0, s * y0), 0.0
        else:
            P_, Qd, F2, Q, theta = q, p, (0, c), (s * y0, x0), math.pi / 2
        eq = rf"\dfrac{{x^2}}{{{P_}}}+\dfrac{{y^2}}{{{Qd}}}"
        rhs = "1"
        gform = (rf"\dfrac{{2x}}{{{P_}}}", rf"\dfrac{{2y}}{{{Qd}}}")
        grad = (Fr(2 * Q[0], P_), Fr(2 * Q[1], Qd))
        a2, b2 = max(P_, Qd), min(P_, Qd)
    else:
        A, B, C, D, a2, (u, v), Q = TILT_CASES[pick]
        if s == -1:
            B, v, Q = -B, -v, (Q[0], -Q[1])
        F2, theta = (u, v), math.atan2(v, u)
        eq = _poly([(A, "x^2"), (B, "xy"), (C, "y^2")])
        rhs = str(D)
        gform = (_poly([(2 * A, "x"), (B, "y")]), _poly([(B, "x"), (2 * C, "y")]))
        grad = (Fr(2 * A * Q[0] + B * Q[1]), Fr(B * Q[0] + 2 * C * Q[1]))
        b2 = a2 - u * u - v * v
    F1 = (-F2[0], -F2[1])
    n = tuple(prim_int(grad))
    d = (Q[0] - F1[0], Q[1] - F1[1])
    k = Fr(d[0] * n[0] + d[1] * n[1], n[0] ** 2 + n[1] ** 2)
    d2 = (d[0] - 2 * k * n[0], d[1] - 2 * k * n[1])
    QF2 = (F2[0] - Q[0], F2[1] - Q[1])
    a_, b_ = math.sqrt(a2), math.sqrt(b2)
    ext = (math.hypot(a_ * math.cos(theta), b_ * math.sin(theta)), math.hypot(a_ * math.sin(theta), b_ * math.cos(theta)))
    return SimpleNamespace(kind=kind, eq=eq, rhs=rhs, gform=gform, F1=F1, F2=F2, Q=Q, grad=grad, n=n, d=d, k=k,
                           d2=d2, QF2=QF2, a=a_, b=b_, theta=theta, ext=ext)


def ellipse_points(e, num=400):
    """楕円 e の周上の点 (x 座標の列, y 座標の列)"""
    t = np.linspace(0, 2 * np.pi, num)
    X, Y = e.a * np.cos(t), e.b * np.sin(t)
    c, s = math.cos(e.theta), math.sin(e.theta)
    return c * X - s * Y, s * X + c * Y


def _ptex(P):
    return "(" + r",\ ".join(str(x) for x in P) + ")"


def ellipse_statement_md(e):
    """問題文の最初の文(焦点の位置と, 光を放つ点)"""
    if e.kind == "tilt":
        return (rf"楕円 ${e.eq}={e.rhs}$ の焦点は $F_1{_ptex(e.F1)}$、$F_2{_ptex(e.F2)}$ である。"
                rf"$F_1$ から楕円上の点 $Q{_ptex(e.Q)}$ に向けて光を放つ。")
    return rf"楕円 ${e.eq}={e.rhs}$ の焦点 $F_1{_ptex(e.F1)}$ から、楕円上の点 $Q{_ptex(e.Q)}$ に向けて光を放つ。"


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
    d0[np.abs(d0) < 1e-12] = 0.0  # 0°, 90°, 180°, 270° をちょうどの向きにする
    focus = start.startswith("焦点")
    F1, F2 = np.array([-cfoc, 0.0]), np.array([cfoc, 0.0])
    # 最後の反射のあと、次に壁に当たるまでの道筋も描く(焦点を通ることが見えるように)
    pts, dirs, normals = reflect_path(a, b, p0, d0, nref + 1, foci=(F1, F2) if focus else None)
    path = pts[:nref + 2]

    tt = np.linspace(0, 2 * PI, 400)
    static = [go.Scatter(x=a * np.cos(tt), y=b * np.sin(tt), mode="lines", line=dict(color=C1, width=3), name="楕円"),
              go.Scatter(x=pts[1:nref + 1, 0], y=pts[1:nref + 1, 1], mode="markers", marker=dict(size=6, color=C2),
                         name="反射点")]
    for k, (q, n) in enumerate(zip(pts[1:min(nref, 3) + 1], normals[:3])):
        static.append(arrow(q[0], q[1], -0.5 * n[0], -0.5 * n[1], color=C3, name="法線 ∇g" if k == 0 else "", width=2))
    static.append(go.Scatter(x=[-cfoc, cfoc], y=[0, 0], mode="markers+text", text=["F₁", "F₂"],
                             textposition="bottom center", marker=dict(size=11, color="black", symbol="star"),
                             name="焦点"))
    fig = _light_animation(path, static, a, b, title="▶ を押すと光が進む(スライダーで途中の位置も見られる)")
    cA, cB = st.columns([3, 2])
    with cA:
        show(fig)
    with cB:
        rows = [{"回": k, "反射点 x": f"{q[0]:.4f}", "反射点 y": f"{q[1]:.4f}",
                 "反射後の向き": f"({d[0]:.3f}, {d[1]:.3f})"}
                for k, (q, d) in enumerate(zip(pts[1:nref + 1], dirs[1:nref + 1]), 1)]
        st.dataframe(rows[:12], hide_index=True, width="stretch")
        if focus:
            q = pts[1]
            dg = d0 - 2 * (d0 @ normals[0]) * normals[0]  # 鏡映の公式でそのまま計算した向き
            dist = abs((cfoc - q[0]) * dg[1] - (0 - q[1]) * dg[0])
            st.success(f"鏡映の公式で計算した1回目の反射光と F₂ の距離:{dist:.1e}(= 0、F₂ を通る)")
            st.caption("最後の反射のあとも、次に壁に当たるまでの道筋を描いている。"
                       "反射のたびに、光は F₂ → F₁ → F₂ … と焦点を交互に通る。")
        st.caption("焦点から出た光は、反射を繰り返すと長軸に近づいていく。焦点以外から出すと、"
                   "道筋は共焦点の楕円や双曲線に接するように並ぶ。")
    with st.expander("数値計算についての注意(焦点から 180° に放った場合など)"):
        st.write("鏡映の公式をそのまま何十回も繰り返すと、1回ごとに 10⁻¹⁶ 程度の丸め誤差が入る。"
                 "長軸に沿った往復の道筋は不安定(わずかなずれが反射のたびに大きくなる)なので、"
                 "誤差が積み重なって長軸から外れてしまう。焦点から出す場合は、"
                 "「反射後は必ずもう一方の焦点を通る」という定理を使って向きを決めているので、ずれない。")

    st.divider()
    st.subheader("サンプル問題(乱数で数値を変えて出題)")
    if "el_seed" not in st.session_state:
        st.session_state["el_seed"] = 1
    if st.button("新しい問題をつくる", key="el_new"):
        st.session_state["el_seed"] += 1
    kind = st.radio("楕円の種類", list(ELL_KINDS), format_func=ELL_KINDS.get, horizontal=True, key="el_kind")
    rng = np.random.default_rng(st.session_state["el_seed"])
    pick = int(rng.integers(ellipse_count(kind)))
    e = ellipse_case(kind, pick, int(rng.choice([1, -1])))
    g_tex = rf"g(x,y)={e.eq}"
    st.markdown(ellipse_statement_md(e))
    st.markdown(f"1. ${g_tex}$ とおき、Q における法線ベクトル $\\nabla g$ を求めよ。\n"
                "2. 入射ベクトル $\\vec d=\\overrightarrow{F_1Q}$ と、反射後の方向ベクトル $\\vec d\\,'$ を求めよ。\n"
                "3. 反射した光がもう一方の焦点 $F_2$ を通ることを示せ。")
    st.caption("ヒント(鏡映):法線ベクトル $\\vec n$ の直線で反射すると "
               "$\\vec d\\,'=\\vec d-2\\dfrac{\\vec d\\cdot\\vec n}{\\vec n\\cdot\\vec n}\\vec n$ となる。"
               "この式は $\\vec n$ を定数倍しても変わらないので、$\\vec n$ は向きが同じなら簡単なものに取りかえてよい。")
    with st.expander("解答"):
        d2, QF2, k = e.d2, e.QF2, e.k
        ratio = d2[0] / QF2[0] if QF2[0] != 0 else d2[1] / QF2[1]
        st.latex(rf"1.\ \nabla g(Q)=\left({e.gform[0]},\ {e.gform[1]}\right)=" + _vf(e.grad)
                 + r"\quad\Longrightarrow\quad \vec n=" + _vf(e.n) + r"\ (\text{同じ向きの整数ベクトル})")
        st.latex(r"2.\ \vec d=\overrightarrow{F_1Q}=" + _vf(e.d) + r",\quad \frac{\vec d\cdot\vec n}{\vec n\cdot\vec n}="
                 + _f(k) + r",\quad \vec d\,'=" + _vf(d2))
        st.latex(r"3.\ \overrightarrow{QF_2}=" + _vf(QF2) + r",\quad \vec d\,'=" + _f(ratio)
                 + r"\,\overrightarrow{QF_2}\ (\text{平行で同じ向き})")
        st.caption("Q から $\\vec d\\,'$ の向きに進むと F₂ に着く。")
    st.markdown("**図で確かめる**(▶ で光を放つ)")
    Qf = np.array([float(e.Q[0]), float(e.Q[1])])
    nQ = np.array([float(e.n[0]), float(e.n[1])])
    nQ /= np.linalg.norm(nQ)
    path = np.array([e.F1, Qf, e.F2], dtype=float)
    ex, ey = ellipse_points(e)
    sc = e.a * 0.25
    static = [go.Scatter(x=ex, y=ey, mode="lines", line=dict(color=C1, width=3), name="楕円"),
              arrow(Qf[0], Qf[1], -sc * nQ[0], -sc * nQ[1], color=C3, name="法線 ∇g(Q)", width=2),
              go.Scatter(x=[e.F1[0], e.F2[0], Qf[0]], y=[e.F1[1], e.F2[1], Qf[1]], mode="markers+text",
                         text=["F₁", "F₂", "Q"], textposition="bottom center",
                         marker=dict(size=11, color="black", symbol="star"), name="焦点と Q")]
    show(_light_animation(path, static, e.ext[0], e.ext[1], frames=80, height=440))
    st.caption("F₁ を出た光は Q で反射し、もう一方の焦点 F₂ にちょうど届く。")
