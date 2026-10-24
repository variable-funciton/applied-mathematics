"""① ベクトル解析 第1章 ベクトル関数:内積・外積/ベクトル関数/曲線/曲面"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import C1, C2, C3, C4, GRAY, arrow3d, layout, layout3d, show, trapz, vec_tex

PI = np.pi


def _row(v):
    """1行3列の表示 (a1, a2, a3)"""
    return vec_tex(v, 4)


def _ijk(v):
    """a1 i + a2 j + a3 k の表示"""
    out = ""
    for c, e in zip(v, (r"\vec{i}", r"\vec{j}", r"\vec{k}")):
        c = float(c)
        if abs(c) < 1e-12:
            continue
        s = "-" if c < 0 else ("+" if out else "")
        m = abs(c)
        out += s + ("" if abs(m - 1) < 1e-12 else f"{m:.4g}") + e
    return out or r"\vec{0}"


# ============================================================
# 1. 内積・外積
# ============================================================
def tab_products():
    st.info("ベクトルは1行3列 $(a_1,\\ a_2,\\ a_3)$ で表す。基本ベクトルは "
            "$\\vec{i}=(1,0,0),\\ \\vec{j}=(0,1,0),\\ \\vec{k}=(0,0,1)$。")
    st.markdown("**ベクトル a, b の成分**")
    c = st.columns(6)
    a = np.array([c[i].slider(f"a{i + 1}", -3, 3, v, key=f"vp_a{i}") for i, v in enumerate((2, 1, 0))], float)
    b = np.array([c[i + 3].slider(f"b{i + 1}", -3, 3, v, key=f"vp_b{i}") for i, v in enumerate((1, 2, 1))], float)
    use_c = st.checkbox("3つ目のベクトル c を加えて、スカラー三重積(平行六面体の体積)も見る", key="vp_usec")
    if use_c:
        cc = st.columns(3)
        cv = np.array([cc[i].slider(f"c{i + 1}", -3, 3, v, key=f"vp_c{i}") for i, v in enumerate((0, 0, 2))], float)

    dot = float(a @ b)
    cr = np.cross(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos = dot / (na * nb) if na * nb > 0 else np.nan

    st.latex(rf"\vec a={_row(a)}={_ijk(a)},\qquad \vec b={_row(b)}={_ijk(b)}")
    cA, cB = st.columns([2, 3])
    with cA:
        st.markdown("**内積**")
        st.latex(rf"\vec a\cdot\vec b=a_1b_1+a_2b_2+a_3b_3="
                 rf"({a[0]:g})({b[0]:g})+({a[1]:g})({b[1]:g})+({a[2]:g})({b[2]:g})={dot:g}")
        if np.isfinite(cos):
            st.latex(rf"\cos\theta=\frac{{\vec a\cdot\vec b}}{{|\vec a||\vec b|}}=\frac{{{dot:g}}}"
                     rf"{{{na:.4g}\times{nb:.4g}}}={cos:.4f},\quad \theta={np.degrees(np.arccos(np.clip(cos, -1, 1))):.2f}^\circ")
        st.markdown("**外積**")
        st.latex(r"\vec a\times\vec b=\begin{vmatrix}\vec i&\vec j&\vec k\\"
                 rf"{a[0]:g}&{a[1]:g}&{a[2]:g}\\{b[0]:g}&{b[1]:g}&{b[2]:g}\end{{vmatrix}}"
                 rf"={_row(cr)}")
        st.latex(rf"|\vec a\times\vec b|={np.linalg.norm(cr):.4f}\ (\text{{平行四辺形の面積}})")
        st.caption(f"確認:(a×b)·a = {cr @ a:g}、(a×b)·b = {cr @ b:g}(a×b は a にも b にも垂直)")
        if use_c:
            vol = float(cr @ cv)
            st.markdown("**スカラー三重積**")
            st.latex(rf"(\vec a\times\vec b)\cdot\vec c=\begin{{vmatrix}}{a[0]:g}&{a[1]:g}&{a[2]:g}\\"
                     rf"{b[0]:g}&{b[1]:g}&{b[2]:g}\\{cv[0]:g}&{cv[1]:g}&{cv[2]:g}\end{{vmatrix}}={vol:g}")
            st.caption(f"平行六面体の体積 = |{vol:g}| = {abs(vol):g}(0 なら3つのベクトルは同一平面上)")
    with cB:
        fig = go.Figure()
        P = np.array([[0, 0, 0], a, a + b, b])
        fig.add_trace(go.Mesh3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], i=[0, 0], j=[1, 2], k=[2, 3],
                                color=C4, opacity=0.35, name="平行四辺形", hoverinfo="skip"))
        for tr in arrow3d([0, 0, 0], a, C1, "a") + arrow3d([0, 0, 0], b, C2, "b") + \
                arrow3d([0, 0, 0], cr, C3, "a × b"):
            fig.add_trace(tr)
        if na > 0:
            proj = dot / na ** 2 * a
            fig.add_trace(go.Scatter3d(x=[b[0], proj[0]], y=[b[1], proj[1]], z=[b[2], proj[2]],
                                       mode="lines", line=dict(color=GRAY, dash="dash", width=4),
                                       name="b の a への正射影"))
        if use_c:
            for tr in arrow3d([0, 0, 0], cv, "#9467bd", "c"):
                fig.add_trace(tr)
            V = np.array([[0, 0, 0], a, b, cv, a + b, a + cv, b + cv, a + b + cv])
            for i, j in [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4), (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)]:
                fig.add_trace(go.Scatter3d(x=V[[i, j], 0], y=V[[i, j], 1], z=V[[i, j], 2], mode="lines",
                                           line=dict(color="#9467bd", width=2), showlegend=False))
        m = max(3, np.abs(np.r_[a, b, cr]).max()) * 1.1
        layout3d(fig, height=520, ranges=[(-m, m)] * 3)
        show(fig)

    with st.expander("公式のまとめ"):
        st.latex(r"\vec i\cdot\vec i=\vec j\cdot\vec j=\vec k\cdot\vec k=1,\quad \vec i\cdot\vec j=\vec j\cdot\vec k=\vec k\cdot\vec i=0")
        st.latex(r"\vec i\times\vec j=\vec k,\quad \vec j\times\vec k=\vec i,\quad \vec k\times\vec i=\vec j,\quad"
                 r"\vec b\times\vec a=-\vec a\times\vec b")
        st.latex(r"|\vec a\times\vec b|=|\vec a||\vec b|\sin\theta,\qquad \vec a\perp\vec b\iff\vec a\cdot\vec b=0,"
                 r"\qquad \vec a\parallel\vec b\iff\vec a\times\vec b=\vec 0")
    with st.expander("練習問題"):
        st.markdown("**問1** $\\vec a=(1,2,2),\\ \\vec b=(2,-1,2)$ のなす角と、"
                    "$\\vec a,\\vec b$ の両方に垂直な単位ベクトルを求めよ。")
        st.latex(r"\vec a\cdot\vec b=2-2+4=4,\ |\vec a|=|\vec b|=3,\ \cos\theta=\frac49")
        st.latex(r"\vec a\times\vec b=(4+2,\ 4-2,\ -1-4)=(6,\ 2,\ -5),\quad \pm\frac{1}{\sqrt{65}}(6,\ 2,\ -5)")
        st.markdown("**問2** 3点 $A(1,0,0),\\ B(0,2,0),\\ C(0,0,3)$ を頂点とする三角形の面積を求めよ。")
        st.latex(r"\overrightarrow{AB}\times\overrightarrow{AC}=(-1,2,0)\times(-1,0,3)=(6,\ 3,\ 2),\quad"
                 r"S=\frac12\sqrt{36+9+4}=\frac72")


# ============================================================
# 曲線の定義(ベクトル関数・曲線のタブで共通)
# ============================================================
CURVES = {
    "円 (cos t, sin t, 0)": dict(
        r=lambda t: np.array([np.cos(t), np.sin(t), 0 * t]),
        d1=lambda t: np.array([-np.sin(t), np.cos(t), 0 * t]),
        d2=lambda t: np.array([-np.cos(t), -np.sin(t), 0 * t]),
        tex=r"\vec r(t)=(\cos t,\ \sin t,\ 0)", t=(0.0, 2 * PI),
        L=lambda a, b: b - a, Ltex=r"\int|\vec r'|dt=\int 1\,dt"),
    "らせん (cos t, sin t, 0.2t)": dict(
        r=lambda t: np.array([np.cos(t), np.sin(t), 0.2 * t]),
        d1=lambda t: np.array([-np.sin(t), np.cos(t), 0.2 + 0 * t]),
        d2=lambda t: np.array([-np.cos(t), -np.sin(t), 0 * t]),
        tex=r"\vec r(t)=(\cos t,\ \sin t,\ 0.2t)", t=(0.0, 4 * PI),
        L=lambda a, b: np.sqrt(1.04) * (b - a), Ltex=r"\int\sqrt{1+0.2^2}\,dt"),
    "放物線 (t, t², 0)": dict(
        r=lambda t: np.array([t, t ** 2, 0 * t]),
        d1=lambda t: np.array([1 + 0 * t, 2 * t, 0 * t]),
        d2=lambda t: np.array([0 * t, 2 + 0 * t, 0 * t]),
        tex=r"\vec r(t)=(t,\ t^2,\ 0)", t=(-1.5, 1.5),
        L=lambda a, b: (lambda F: F(b) - F(a))(lambda t: (t * np.sqrt(1 + 4 * t * t)) / 2
                                                  + np.arcsinh(2 * t) / 4),
        Ltex=r"\int\sqrt{1+4t^2}\,dt"),
    "サイクロイド (t − sin t, 1 − cos t, 0)": dict(
        r=lambda t: np.array([t - np.sin(t), 1 - np.cos(t), 0 * t]),
        d1=lambda t: np.array([1 - np.cos(t), np.sin(t), 0 * t]),
        d2=lambda t: np.array([np.sin(t), np.cos(t), 0 * t]),
        tex=r"\vec r(t)=(t-\sin t,\ 1-\cos t,\ 0)", t=(0.0, 2 * PI),
        L=None, Ltex=r"\int 2\left|\sin\tfrac t2\right|dt\quad(1\text{ アーチで }8)"),
    "ねじれ3次曲線 (t, t², t³)": dict(
        r=lambda t: np.array([t, t ** 2, t ** 3]),
        d1=lambda t: np.array([1 + 0 * t, 2 * t, 3 * t ** 2]),
        d2=lambda t: np.array([0 * t, 2 + 0 * t, 6 * t]),
        tex=r"\vec r(t)=(t,\ t^2,\ t^3)", t=(-1.2, 1.2),
        L=None, Ltex=r"\int\sqrt{1+4t^2+9t^4}\,dt\ (\text{数値計算})"),
}


def _curve_pick(key):
    name = st.selectbox("曲線", list(CURVES), index=1, key=f"{key}_c")
    C = CURVES[name]
    st.latex(C["tex"])
    return name, C


def _curve_fig(C, t0, t1):
    tt = np.linspace(t0, t1, 600)
    R = C["r"](tt)
    fig = go.Figure(go.Scatter3d(x=R[0], y=R[1], z=R[2], mode="lines", line=dict(color=GRAY, width=4),
                                 name="曲線"))
    return fig, R


# ============================================================
# 2. ベクトル関数
# ============================================================
def tab_vector_function():
    st.info("t を時刻と考えると、$\\vec r(t)$ は点の位置、$\\vec r'(t)$ は速度(接線方向)、"
            "$\\vec r''(t)$ は加速度。微分は成分ごとに行う。")
    st.latex(r"\vec r'(t)=\lim_{h\to0}\frac{\vec r(t+h)-\vec r(t)}{h}=\left(x'(t),\ y'(t),\ z'(t)\right)")
    name, C = _curve_pick("vv")
    t0, t1 = C["t"]
    c1, c2 = st.columns(2)
    t = c1.slider("t", float(t0), float(t1), float(t0 + 0.3 * (t1 - t0)), 0.01, key=f"vv_t_{name}")
    h = c2.slider("h(差分の幅)", 0.01, 2.0, 1.0, 0.01, key="vv_h")
    show_acc = st.checkbox("加速度 r''(t) も表示", key="vv_acc")

    p, v, acc = C["r"](t), C["d1"](t), C["d2"](t)
    q = C["r"](t + h)
    sec = (q - p) / h
    fig, R = _curve_fig(C, t0, t1 + h)
    fig.add_trace(go.Scatter3d(x=[p[0], q[0]], y=[p[1], q[1]], z=[p[2], q[2]], mode="markers",
                               marker=dict(size=5, color=["black", GRAY]), name="r(t), r(t+h)"))
    for tr in arrow3d([0, 0, 0], p, GRAY, "位置 r(t)", width=3):
        fig.add_trace(tr)
    for tr in arrow3d(p, sec, C4, "(r(t+h) − r(t))/h"):
        fig.add_trace(tr)
    for tr in arrow3d(p, v, C2, "速度 r'(t)"):
        fig.add_trace(tr)
    if show_acc:
        for tr in arrow3d(p, acc, C3, "加速度 r''(t)"):
            fig.add_trace(tr)
    layout3d(fig, height=540)
    cA, cB = st.columns([3, 2])
    with cA:
        show(fig)
    with cB:
        st.latex(rf"\vec r(t)={vec_tex(p)}")
        st.latex(rf"\vec r'(t)={vec_tex(v)}")
        st.latex(rf"\frac{{\vec r(t+h)-\vec r(t)}}{{h}}={vec_tex(sec)}")
        st.latex(rf"|\vec r'(t)|={np.linalg.norm(v):.4f}\ (\text{{速さ}})")
        if show_acc:
            st.latex(rf"\vec r''(t)={vec_tex(acc)}")
        st.caption("h を小さくすると、オレンジ(差分商)が赤(r′(t)、接線方向)に近づく。")

    with st.expander("微分の公式"):
        st.latex(r"(\vec a\cdot\vec b)'=\vec a'\cdot\vec b+\vec a\cdot\vec b',\qquad "
                 r"(\vec a\times\vec b)'=\vec a'\times\vec b+\vec a\times\vec b'")
        st.latex(r"|\vec a(t)|=\text{一定}\ \Longrightarrow\ \vec a\cdot\vec a'=0\ (\vec a\perp\vec a')")
        st.caption("円運動では |r| が一定なので、位置と速度は直交する(円タブで確かめよう)。")
    with st.expander("練習問題"):
        st.markdown("**問3** $\\vec r(t)=(\\cos t,\\ \\sin t,\\ t)$ の速度・加速度・速さを求めよ。")
        st.latex(r"\vec r'=(-\sin t,\ \cos t,\ 1),\quad \vec r''=(-\cos t,\ -\sin t,\ 0),\quad |\vec r'|=\sqrt2")


# ============================================================
# 3. 曲線
# ============================================================
def tab_curve():
    st.info("曲線の長さは速さの積分。接線・法線と曲率 κ で、曲線の「曲がり具合」を測る。")
    st.latex(r"s=\int_a^b|\vec r'(t)|\,dt,\quad \vec t=\frac{\vec r'}{|\vec r'|},\quad "
             r"\kappa=\frac{|\vec r'\times\vec r''|}{|\vec r'|^3},\quad \rho=\frac1\kappa\ (\text{曲率半径})")
    name, C = _curve_pick("cv")
    t0, t1 = C["t"]
    c1, c2 = st.columns(2)
    ab = c1.slider("長さを測る区間 [a, b]", float(t0), float(t1), (float(t0), float(t1)), 0.01,
                   key=f"cv_ab_{name}")
    t = c2.slider("接線・曲率を見る t", float(t0), float(t1), float(t0 + 0.4 * (t1 - t0)), 0.01,
                  key=f"cv_t_{name}")

    tt = np.linspace(ab[0], ab[1], 20001)
    sp = np.linalg.norm(C["d1"](tt), axis=0)
    L_num = float(trapz(sp, tt))

    p, v, acc = C["r"](t), C["d1"](t), C["d2"](t)
    nv = np.linalg.norm(v)
    cr = np.cross(v, acc)
    kappa = np.linalg.norm(cr) / nv ** 3 if nv > 1e-12 else np.nan
    T = v / nv if nv > 1e-12 else np.zeros(3)
    fig, _ = _curve_fig(C, t0, t1)
    tt2 = np.linspace(ab[0], ab[1], 400)
    R2 = C["r"](tt2)
    fig.add_trace(go.Scatter3d(x=R2[0], y=R2[1], z=R2[2], mode="lines", line=dict(color=C1, width=8),
                               name="長さを測る部分"))
    fig.add_trace(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers", marker=dict(size=5, color="black"),
                               name="点"))
    for tr in arrow3d(p, T, C2, "単位接線 t"):
        fig.add_trace(tr)
    if np.isfinite(kappa) and kappa > 1e-9:
        # 主法線 n = (r'×r'')×r' の向き
        nvec = np.cross(cr, v)
        nvec /= np.linalg.norm(nvec)
        for tr in arrow3d(p, nvec, C3, "主法線 n"):
            fig.add_trace(tr)
        rho = 1 / kappa
        if rho < 20:
            ctr = p + rho * nvec
            ph = np.linspace(0, 2 * PI, 200)
            circ = ctr[:, None] + rho * (-np.cos(ph)[None, :] * nvec[:, None] + np.sin(ph)[None, :] * T[:, None])
            fig.add_trace(go.Scatter3d(x=circ[0], y=circ[1], z=circ[2], mode="lines",
                                       line=dict(color=C3, width=3, dash="dash"), name="曲率円"))
    layout3d(fig, height=540)

    cA, cB = st.columns([3, 2])
    with cA:
        show(fig)
    with cB:
        st.markdown("**曲線の長さ**")
        st.latex(r"s=" + C["Ltex"])
        st.metric("数値積分", f"{L_num:.6f}")
        if C["L"] is not None:
            st.metric("公式", f"{float(C['L'](ab[0], ab[1])):.6f}")
        st.markdown("**点 t での量**")
        st.latex(rf"|\vec r'|={nv:.4f},\quad \kappa={kappa:.4f}")
        if np.isfinite(kappa) and kappa > 1e-9:
            st.latex(rf"\rho=\frac1\kappa={1 / kappa:.4f}")
        else:
            st.write("曲率 0(この点では直線的)")
        st.caption("曲率円:その点で曲線に最もよく接する円。曲がりが急なほど小さい。")

    with st.expander("練習問題"):
        st.markdown("**問4** らせん $\\vec r(t)=(a\\cos t,\\ a\\sin t,\\ bt)\\ (0\\le t\\le 2\\pi)$ の長さと曲率を求めよ。")
        st.latex(r"|\vec r'|=\sqrt{a^2+b^2},\quad s=2\pi\sqrt{a^2+b^2},\quad \kappa=\frac{a}{a^2+b^2}")
        st.markdown("**問5** サイクロイド1アーチ $(0\\le t\\le 2\\pi)$ の長さを求めよ。")
        st.latex(r"|\vec r'|=\sqrt{2-2\cos t}=2\sin\frac t2,\quad s=\int_0^{2\pi}2\sin\frac t2\,dt=8")


# ============================================================
# 4. 曲面
# ============================================================
SURFS = {
    "球面(半径1)": dict(r=lambda u, v: np.array([np.sin(u) * np.cos(v), np.sin(u) * np.sin(v), np.cos(u) + 0 * v]),
                     u=(0, PI), v=(0, 2 * PI), tex=r"\vec r(u,v)=(\sin u\cos v,\ \sin u\sin v,\ \cos u)",
                     area=4 * PI, atex=r"4\pi", nt=r"|\vec r_u\times\vec r_v|=\sin u"),
    "円柱面(半径1、高さ2)": dict(r=lambda u, v: np.array([np.cos(v), np.sin(v), u + 0 * v]),
                          u=(0, 2), v=(0, 2 * PI), tex=r"\vec r(u,v)=(\cos v,\ \sin v,\ u)",
                          area=4 * PI, atex=r"2\pi\times2=4\pi", nt=r"|\vec r_u\times\vec r_v|=1"),
    "放物面 z = x² + y²(x²+y²≤1)": dict(
        r=lambda u, v: np.array([u * np.cos(v), u * np.sin(v), u ** 2 + 0 * v]), u=(0, 1), v=(0, 2 * PI),
        tex=r"\vec r(u,v)=(u\cos v,\ u\sin v,\ u^2)", area=PI / 6 * (5 * np.sqrt(5) - 1),
        atex=r"\frac{\pi}{6}(5\sqrt5-1)", nt=r"|\vec r_u\times\vec r_v|=u\sqrt{1+4u^2}"),
    "平面 z = 1 − x − y(0≤x,y≤1)": dict(
        r=lambda u, v: np.array([u + 0 * v, v + 0 * u, 1 - u - v]), u=(0, 1), v=(0, 1),
        tex=r"\vec r(u,v)=(u,\ v,\ 1-u-v)", area=np.sqrt(3), atex=r"\sqrt3",
        nt=r"\vec r_u\times\vec r_v=(1,\ 1,\ 1)"),
    "鞍面 z = xy(|x|,|y|≤1)": dict(
        r=lambda u, v: np.array([u + 0 * v, v + 0 * u, u * v]), u=(-1, 1), v=(-1, 1),
        tex=r"\vec r(u,v)=(u,\ v,\ uv)", area=None, atex=None,
        nt=r"\vec r_u\times\vec r_v=(-v,\ -u,\ 1)"),
    "トーラス(R=2, a=0.6)": dict(
        r=lambda u, v: np.array([(2 + 0.6 * np.cos(u)) * np.cos(v), (2 + 0.6 * np.cos(u)) * np.sin(v),
                                 0.6 * np.sin(u) + 0 * v]), u=(0, 2 * PI), v=(0, 2 * PI),
        tex=r"\vec r(u,v)=((R+a\cos u)\cos v,\ (R+a\sin u)\ldots)", area=4 * PI ** 2 * 2 * 0.6,
        atex=r"4\pi^2Ra", nt=r"|\vec r_u\times\vec r_v|=a(R+a\cos u)"),
}
SURFS["トーラス(R=2, a=0.6)"]["tex"] = r"\vec r(u,v)=\left((R+a\cos u)\cos v,\ (R+a\cos u)\sin v,\ a\sin u\right)"


def partials(r, u, v, h=1e-6):
    ru = (r(u + h, v) - r(u - h, v)) / (2 * h)
    rv = (r(u, v + h) - r(u, v - h)) / (2 * h)
    return ru, rv


def tab_surface():
    st.info("曲面は2つのパラメータ (u, v) のベクトル関数。$\\vec r_u,\\ \\vec r_v$ は接平面を張り、"
            "外積 $\\vec r_u\\times\\vec r_v$ が法線ベクトル、その大きさが面積の拡大率になる。")
    st.latex(r"\vec n=\vec r_u\times\vec r_v,\qquad \text{接平面:}\ \vec n\cdot(\vec x-\vec r_0)=0,"
             r"\qquad S=\iint_D|\vec r_u\times\vec r_v|\,du\,dv")
    name = st.selectbox("曲面", list(SURFS), key="vs_s")
    S = SURFS[name]
    st.latex(S["tex"])
    (u0, u1), (v0, v1) = S["u"], S["v"]
    c1, c2 = st.columns(2)
    su = c1.slider("点の位置 u(範囲内の割合)", 0.0, 1.0, 0.35, 0.01, key="vs_u")
    sv = c2.slider("点の位置 v(範囲内の割合)", 0.0, 1.0, 0.15, 0.01, key="vs_v")
    u = u0 + su * (u1 - u0)
    v = v0 + sv * (v1 - v0)

    p = S["r"](u, v)
    ru, rv = partials(S["r"], u, v)
    n = np.cross(ru, rv)
    nn = np.linalg.norm(n)

    U, V = np.meshgrid(np.linspace(u0, u1, 60), np.linspace(v0, v1, 80))
    R = S["r"](U, V)
    fig = go.Figure(go.Surface(x=R[0], y=R[1], z=R[2], opacity=0.75, colorscale="Blues", showscale=False,
                               name="曲面", hoverinfo="skip"))
    sc = 0.6
    for tr in arrow3d(p, ru / max(np.linalg.norm(ru), 1e-9) * sc, C2, "r_u") + \
            arrow3d(p, rv / max(np.linalg.norm(rv), 1e-9) * sc, C4, "r_v") + \
            arrow3d(p, n / max(nn, 1e-9) * 0.8, C3, "法線 r_u × r_v"):
        fig.add_trace(tr)
    if nn > 1e-9:
        e1 = ru / np.linalg.norm(ru)
        e2 = np.cross(n / nn, e1)
        a, b = np.meshgrid(np.linspace(-0.6, 0.6, 2), np.linspace(-0.6, 0.6, 2))
        Pl = p[:, None, None] + a * e1[:, None, None] + b * e2[:, None, None]
        fig.add_trace(go.Surface(x=Pl[0], y=Pl[1], z=Pl[2], opacity=0.45, showscale=False,
                                 colorscale=[[0, C4], [1, C4]], name="接平面", hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers", marker=dict(size=5, color="black"),
                               name="点"))
    layout3d(fig, height=560)

    k = 400
    uu = u0 + (np.arange(k) + 0.5) * (u1 - u0) / k
    vv = v0 + (np.arange(k) + 0.5) * (v1 - v0) / k
    UU, VV = np.meshgrid(uu, vv)
    Ru, Rv = partials(S["r"], UU, VV, 1e-5)
    area = float(np.sum(np.linalg.norm(np.cross(Ru, Rv, axis=0), axis=0)) * (u1 - u0) * (v1 - v0) / k ** 2)

    cA, cB = st.columns([3, 2])
    with cA:
        show(fig)
    with cB:
        st.latex(rf"\vec r_0={vec_tex(p)}")
        st.latex(rf"\vec r_u={vec_tex(ru)},\ \ \vec r_v={vec_tex(rv)}")
        st.latex(rf"\vec n=\vec r_u\times\vec r_v={vec_tex(n)}")
        if nn > 1e-9:
            d = float(n @ p)
            st.markdown("**接平面の方程式**")
            st.latex(rf"{n[0]:.3g}(x-{p[0]:.3g})+{n[1]:.3g}(y-{p[1]:.3g})+{n[2]:.3g}(z-{p[2]:.3g})=0"
                     .replace("+-", "-").replace("--", "+"))
            st.latex(rf"\Longleftrightarrow\ {n[0]:.3g}x+{n[1]:.3g}y+{n[2]:.3g}z={d:.3g}".replace("+-", "-"))
        else:
            st.warning("この点では r_u × r_v = 0(パラメータの特異点:球面の極など)。法線が決まりません。")
        st.markdown("**曲面積**")
        st.latex(S["nt"])
        st.metric("数値積分", f"{area:.6f}")
        if S["area"] is not None:
            st.metric("公式", f"{S['area']:.6f}")
            st.latex(r"S=" + S["atex"])

    with st.expander("z = f(x, y) の形の曲面"):
        st.latex(r"\vec r(x,y)=(x,\ y,\ f(x,y)),\quad \vec r_x\times\vec r_y=(-f_x,\ -f_y,\ 1)")
        st.latex(r"\text{接平面:}\ z-z_0=f_x(x_0,y_0)(x-x_0)+f_y(x_0,y_0)(y-y_0),\qquad "
                 r"S=\iint_D\sqrt{1+f_x^2+f_y^2}\,dxdy")
    with st.expander("練習問題"):
        st.markdown("**問6** 曲面 $z=x^2+y^2$ の点 $(1,1,2)$ における接平面の方程式を求めよ。")
        st.latex(r"f_x=2,\ f_y=2\ \Rightarrow\ z-2=2(x-1)+2(y-1),\ \ z=2x+2y-2")
        st.markdown("**問7** 半径 $a$ の球面の面積が $4\\pi a^2$ であることを示せ。")
        st.latex(r"|\vec r_u\times\vec r_v|=a^2\sin u,\quad S=\int_0^{2\pi}\!\!\int_0^\pi a^2\sin u\,du\,dv=4\pi a^2")
