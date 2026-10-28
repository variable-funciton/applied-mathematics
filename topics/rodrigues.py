"""① ベクトル解析 発展:ロドリゲスの回転公式

単位ベクトル k を軸に、ベクトル v を角 θ だけ(右ねじの向きに)回転させると
    v' = v cosθ + (k × v) sinθ + k (k·v)(1 − cosθ)
導出 → 動かして確かめる(3D) → 使う場面 → 例題 の順。
"""
import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import C1, C2, C3, C4, GRAY, arrow3d, layout3d
from topics.player import player
from topics.vec_apps import examples


def rodrigues(v, k, th):
    """v を単位ベクトル k のまわりに th [rad] 回転(numpy)"""
    v, k = np.asarray(v, float), np.asarray(k, float)
    k = k / np.linalg.norm(k)
    return v * math.cos(th) + np.cross(k, v) * math.sin(th) + k * np.dot(k, v) * (1 - math.cos(th))


def rot_matrix(k, th):
    """R = I + sinθ K + (1 − cosθ) K²(K は k× を表す交代行列)"""
    k = np.asarray(k, float)
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


# ------------------------------------------------------------
# 導出の図(例:k = (0, 0, 1)、v = (1.6, 0.7, 1.3)、θ = 60°)
# ------------------------------------------------------------
DK = np.array([0.0, 0.0, 1.0])
DV = np.array([1.6, 0.7, 1.3])
DTH = math.radians(60)
CAM = dict(eye=dict(x=1.5, y=-1.6, z=0.9))
C5 = "#9467bd"


def _label(fig, p, text, color="black", size=15):
    fig.add_trace(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="text", text=[text], showlegend=False,
                               textfont=dict(size=size, color=color), hoverinfo="skip"))


def _arrow(fig, o, v, color, name, width=6):
    for tr in arrow3d(o, v, color=color, name=name, width=width, showlegend=bool(name)):
        fig.add_trace(tr)


def _seg(fig, p, q, color=GRAY, dash="dot", width=3, name=None):
    fig.add_trace(go.Scatter3d(x=[p[0], q[0]], y=[p[1], q[1]], z=[p[2], q[2]], mode="lines",
                               line=dict(color=color, width=width, dash=dash), showlegend=name is not None,
                               name=name or "", hoverinfo="skip"))


def _axis(fig):
    _seg(fig, (0, 0, -0.4), (0, 0, 2.1), color=GRAY, dash="dash", width=3, name="回転軸")
    o = np.array([-0.55, -0.55, 0.0])               # 軸と重ならないよう、軸の横に平行に描く
    _arrow(fig, o, DK, "#222", "k(軸の向きの単位ベクトル、|k| = 1)", width=8)
    _label(fig, o + DK * 1.15, "k")


def _circle(fig, center, e1, e2, r, color="#bbbbbb", t0=0, t1=2 * np.pi, width=3, name=None):
    t = np.linspace(t0, t1, 120)
    P = center[:, None] + r * (np.outer(e1, np.cos(t)) + np.outer(e2, np.sin(t)))
    fig.add_trace(go.Scatter3d(x=P[0], y=P[1], z=P[2], mode="lines", line=dict(color=color, width=width),
                               showlegend=name is not None, name=name or "", hoverinfo="skip"))


def _right_angle(fig, corner, d1, d2, s=0.15):
    d1, d2 = d1 / np.linalg.norm(d1) * s, d2 / np.linalg.norm(d2) * s
    P = [corner + d1, corner + d1 + d2, corner + d2]
    fig.add_trace(go.Scatter3d(x=[p[0] for p in P], y=[p[1] for p in P], z=[p[2] for p in P], mode="lines",
                               line=dict(color="#555", width=3), showlegend=False, hoverinfo="skip"))


def _fig3d(fig, title):
    layout3d(fig, height=430, title=title, ranges=[(-1.2, 2.0), (-1.2, 2.0), (-0.4, 2.2)])
    fig.update_layout(scene_camera=CAM, uirevision="keep", legend=dict(orientation="h", y=-0.02))
    return fig


def _parts():
    par = DK * np.dot(DK, DV)
    perp = DV - par
    kv = np.cross(DK, DV)
    return par, perp, kv, np.linalg.norm(perp)


def fig_step1():
    par, perp, kv, r = _parts()
    fig = go.Figure()
    _axis(fig)
    _arrow(fig, (0, 0, 0), DV, C1, "v", width=8)
    _arrow(fig, (0, 0, 0), par, GRAY, "v∥ = (k·v)k(軸方向の成分)", width=7)
    _arrow(fig, par, perp, C4, "v⊥ = v − v∥(軸に垂直な成分)", width=7)
    _right_angle(fig, par, -par, perp)
    _label(fig, DV * 1.08, "v", C1)
    _label(fig, par + np.array([-0.35, 0, 0]), "v∥", "#555")
    _label(fig, par + perp * 0.5 + np.array([0, 0, 0.18]), "v⊥", C4)
    return _fig3d(fig, "(i) v を軸に平行な部分と垂直な部分に分ける")


def fig_step2():
    par, perp, kv, r = _parts()
    fig = go.Figure()
    _axis(fig)
    _seg(fig, (0, 0, 0), par, color=GRAY, dash="solid", width=4)
    _circle(fig, par, perp / r, kv / r, r, name="v⊥ の先端が回る円(半径 |v⊥|)")
    _arrow(fig, par, perp, C4, "v⊥", width=7)
    _arrow(fig, par, kv, C3, "k × v(= k × v⊥)", width=7)
    _right_angle(fig, par, perp, kv)
    _label(fig, par + perp * 1.15, "v⊥", C4)
    _label(fig, par + kv * 1.2, "k×v", C3)
    return _fig3d(fig, "(ii) 回転の平面にある、直交する同じ長さの2本")


def fig_step3():
    """回転の平面を、軸の正の側(k の先)から見下ろした図"""
    fig = go.Figure()
    t = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(go.Scatter(x=np.cos(t), y=np.sin(t), mode="lines", line=dict(color="#bbbbbb", width=2), showlegend=False,
                             hoverinfo="skip"))
    c, s = math.cos(DTH), math.sin(DTH)

    def arr(x0, y0, x1, y1, color, name, width=3):
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                           arrowhead=3, arrowsize=1.4, arrowwidth=width, arrowcolor=color, text="")
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=color, width=width), name=name))
    arr(0, 0, 1, 0, C4, "v⊥(長さ r)")
    arr(0, 0, 0, 1, C3, "k × v(長さ r)")
    arr(0, 0, c, s, C2, "v⊥′(回転後)", width=4)
    fig.add_trace(go.Scatter(x=[c, c, 0], y=[0, s, s], mode="lines", line=dict(color=GRAY, dash="dot", width=2),
                             showlegend=False, hoverinfo="skip"))
    ta = np.linspace(0, DTH, 30)
    fig.add_trace(go.Scatter(x=0.28 * np.cos(ta), y=0.28 * np.sin(ta), mode="lines", line=dict(color=C2, width=2),
                             showlegend=False, hoverinfo="skip"))
    for x, y, txt, col in ((0.38, 0.17, "θ", C2), (c / 2, -0.1, "r cos θ", "#555"), (-0.2, s / 2, "r sin θ", "#555"),
                           (1.1, 0.0, "v⊥", C4), (0.0, 1.1, "k×v", C3), (c + 0.12, s + 0.08, "v⊥′", C2),
                           (0.0, -0.08, "軸(手前向き)", "#222")):
        fig.add_annotation(x=x, y=y, text=txt, showarrow=False, font=dict(size=15, color=col))
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(size=9, color="#222", symbol="circle-dot"),
                             showlegend=False))
    fig.update_layout(height=430, title="(iii) 軸の正の側から見た回転の平面", margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.05))
    fig.update_xaxes(range=[-1.3, 1.4], visible=False)
    fig.update_yaxes(range=[-1.3, 1.4], visible=False, scaleanchor="x", scaleratio=1)
    return fig


def fig_step4():
    par, perp, kv, r = _parts()
    pperp = perp * math.cos(DTH) + kv * math.sin(DTH)
    vp = par + pperp
    fig = go.Figure()
    _axis(fig)
    _circle(fig, par, perp / r, kv / r, r)
    _circle(fig, par, perp / r, kv / r, r, color=C2, t1=DTH, width=6, name="角 θ")
    _arrow(fig, (0, 0, 0), DV, "#9ecae1", "v(回転前)", width=5)
    _arrow(fig, (0, 0, 0), par, GRAY, "v∥(変わらない)", width=7)
    _arrow(fig, par, pperp, C4, "v⊥′(回転した垂直成分)", width=7)
    _arrow(fig, (0, 0, 0), vp, C2, "v′ = v∥ + v⊥′", width=8)
    _label(fig, vp * 1.1, "v′", C2)
    _label(fig, DV * 1.1, "v", "#6baed6")
    return _fig3d(fig, "(iv) 変わらない v∥ と、回った v⊥′ を足す")


def fig_step5():
    par, perp, kv, r = _parts()
    kkv = np.cross(DK, kv)
    fig = go.Figure()
    _axis(fig)
    _circle(fig, par, perp / r, kv / r, r)
    _arrow(fig, par, perp, C4, "v⊥", width=6)
    _arrow(fig, par, kv, C3, "k × v", width=7)
    _arrow(fig, par, kkv, C5, "k × (k × v) = −v⊥", width=7)
    _arrow(fig, (0, 0, 0), DV, C1, "v", width=6)
    _label(fig, par + kv * 1.2, "k×v", C3)
    _label(fig, par + kkv * 1.2, "k×(k×v)", C5)
    _label(fig, par + perp * 1.2, "v⊥", C4)
    return _fig3d(fig, "(v) k × をもう一度かけると、さらに 90° 回って −v⊥ になる")


def _step(text_fn, fig, key):
    a, b = st.columns([1.15, 1])
    with a:
        text_fn()
    with b:
        st.plotly_chart(fig, width="stretch", key=key)


def _derivation():
    st.subheader("1. 公式")
    st.markdown("単位ベクトル $\\vec k$($|\\vec k|=1$)を回転軸とし、ベクトル $\\vec v$ を、$\\vec k$ の向きに右ねじが進む向き"
                "($\\vec k$ の先から見て反時計回り)に角 $\\theta$ だけ回転させたベクトルを $\\vec v\\,'$ とすると")
    st.latex(r"\vec v\,'=\vec v\cos\theta+(\vec k\times\vec v)\sin\theta+\vec k\,(\vec k\cdot\vec v)(1-\cos\theta)")
    st.subheader("2. 導出")
    st.caption("図の例:$\\vec k=(0,0,1)$(軸は $z$ 軸)、$\\vec v=(1.6,\\ 0.7,\\ 1.3)$、$\\theta=60^\\circ$。図はドラッグで回せる。")

    def t1():
        st.markdown("**(i) 軸に平行な部分と垂直な部分に分ける**")
        st.markdown(
            "回転させても、軸に平行な部分は動かない。動くのは軸に垂直な部分だけである。そこで $\\vec v$ を2つに分ける。\n\n"
            "- 軸方向の成分は、$\\vec v$ の $\\vec k$ への正射影:"
            "$\\vec v_\\parallel=\\dfrac{\\vec k\\cdot\\vec v}{|\\vec k|^2}\\vec k=(\\vec k\\cdot\\vec v)\\,\\vec k$"
            "(**$|\\vec k|=1$ を使った**)。\n"
            "- 残りが軸に垂直な成分:$\\vec v_\\perp=\\vec v-(\\vec k\\cdot\\vec v)\\vec k$。\n\n"
            "確かめ:$\\vec k\\cdot\\vec v_\\perp=\\vec k\\cdot\\vec v-(\\vec k\\cdot\\vec v)|\\vec k|^2=0$ なので、本当に垂直になっている。")
    _step(t1, fig_step1(), "rg_d1")

    def t2():
        st.markdown("**(ii) 回転の平面に、直交する同じ長さの2本をとる**")
        st.markdown(
            "$\\vec v_\\perp$ の先端は、軸に垂直な平面(灰色の円)の上を回る。この平面に「座標軸」を2本用意する。\n\n"
            "- 1本目:$\\vec v_\\perp$ そのもの。\n"
            "- 2本目:$\\vec k\\times\\vec v$。外積なので $\\vec k$ にも $\\vec v$ にも垂直 → 回転の平面上にあり、"
            "$\\vec v_\\perp$ を軸のまわりに $90^\\circ$ 回した向きになる(右ねじの向き)。\n\n"
            "$\\vec k\\times\\vec v=\\vec k\\times(\\vec v_\\parallel+\\vec v_\\perp)=\\vec k\\times\\vec v_\\perp$"
            "($\\vec v_\\parallel$ は $\\vec k$ と平行なので $\\vec k\\times\\vec v_\\parallel=\\vec 0$)。\n\n"
            "長さは $|\\vec k\\times\\vec v_\\perp|=|\\vec k||\\vec v_\\perp|\\sin90^\\circ=|\\vec v_\\perp|$"
            "(**$|\\vec k|=1$ を使った**)。2本は直交し、長さが等しい。")
    _step(t2, fig_step2(), "rg_d2")

    def t3():
        st.markdown("**(iii) 平面の回転と同じに考える**")
        st.markdown(
            "軸の正の側から見下ろすと、右の図のように、ふつうの平面の回転になる。$r=|\\vec v_\\perp|$ とおき、"
            "$\\vec v_\\perp$ を $x$ 軸、$\\vec k\\times\\vec v$ を $y$ 軸の向きとみる。\n\n"
            "半径 $r$ の円周上で角 $\\theta$ だけ回った点は、$x$ 方向に $r\\cos\\theta$、$y$ 方向に $r\\sin\\theta$ 進んだ所にある。"
            "$x$ 方向の長さ $r$ のベクトルが $\\vec v_\\perp$、$y$ 方向が $\\vec k\\times\\vec v$ なので")
        st.latex(r"\vec v_{\perp}\,'=\vec v_\perp\cos\theta+(\vec k\times\vec v)\sin\theta")
        st.caption("$\\theta=90^\\circ$ なら $\\vec v_\\perp\\,'=\\vec k\\times\\vec v$、$\\theta=180^\\circ$ なら $-\\vec v_\\perp$ になる。")
    _step(t3, fig_step3(), "rg_d3")

    def t4():
        st.markdown("**(iv) 足し合わせる**")
        st.markdown("回転後のベクトルは、変わらない $\\vec v_\\parallel$ と、回った $\\vec v_\\perp\\,'$ の和:"
                    "$\\vec v\\,'=\\vec v_\\parallel+\\vec v_\\perp\\,'$。(i) と (iii) を代入すると")
        st.latex(r"\vec v\,'=(\vec k\cdot\vec v)\vec k+\left\{\vec v-(\vec k\cdot\vec v)\vec k\right\}\cos\theta+(\vec k\times\vec v)\sin\theta")
        st.markdown("$(\\vec k\\cdot\\vec v)\\vec k$ の項をまとめると、公式が得られる:")
        st.latex(r"\vec v\,'=\vec v\cos\theta+(\vec k\times\vec v)\sin\theta+\vec k\,(\vec k\cdot\vec v)(1-\cos\theta)")
    _step(t4, fig_step4(), "rg_d4")

    def t5():
        st.markdown("**(v) 行列の形にする**")
        st.markdown(
            "ベクトル三重積の公式 $\\vec a\\times(\\vec b\\times\\vec c)=(\\vec a\\cdot\\vec c)\\vec b-(\\vec a\\cdot\\vec b)\\vec c$ で "
            "$\\vec a=\\vec b=\\vec k,\\ \\vec c=\\vec v$ とすると")
        st.latex(r"\vec k\times(\vec k\times\vec v)=(\vec k\cdot\vec v)\vec k-(\vec k\cdot\vec k)\vec v=(\vec k\cdot\vec v)\vec k-\vec v")
        st.markdown(
            "ここで **$\\vec k\\cdot\\vec k=|\\vec k|^2=1$ を使った**。つまり $\\vec k\\times(\\vec k\\times\\vec v)=-\\vec v_\\perp$:"
            "$\\vec k\\times$ を2回かけると $\\vec v_\\perp$ が $180^\\circ$ 回る(右の図の紫)。これを (iv) に入れると "
            "$\\vec k(\\vec k\\cdot\\vec v)=\\vec v+\\vec k\\times(\\vec k\\times\\vec v)$ より")
        st.latex(r"\vec v\,'=\vec v+\sin\theta\,(\vec k\times\vec v)+(1-\cos\theta)\,\vec k\times(\vec k\times\vec v)")
        st.markdown(
            "行列で書くときは、**ベクトルを縦ベクトル(3行1列の行列)** $\\vec v=\\begin{pmatrix}v_1\\\\v_2\\\\v_3\\end{pmatrix}$ として扱う。"
            "外積 $\\vec k\\times\\vec v$ は、次の交代行列 $K$ との積で書ける:")
        st.latex(r"K\vec v=\begin{pmatrix}0&-k_3&k_2\\ k_3&0&-k_1\\ -k_2&k_1&0\end{pmatrix}\begin{pmatrix}v_1\\v_2\\v_3\end{pmatrix}"
                 r"=\begin{pmatrix}k_2v_3-k_3v_2\\ k_3v_1-k_1v_3\\ k_1v_2-k_2v_1\end{pmatrix}=\vec k\times\vec v")
        st.markdown("$\\vec k\\times(\\vec k\\times\\vec v)=K(K\\vec v)=K^2\\vec v$ なので、$\\vec v\\,'=R\\vec v$($3\\times3$ 行列と縦ベクトルの積)と書ける:")
        st.latex(r"R=I+\sin\theta\,K+(1-\cos\theta)K^2")
        st.caption("$\\vec k=(0,0,1)$ とすると、$xy$ 平面の回転行列 $\\begin{pmatrix}\\cos\\theta&-\\sin\\theta\\\\ \\sin\\theta&\\cos\\theta\\end{pmatrix}$ に戻る(例題1)。"
                   "$R$ の第 $j$ 列は、基本ベクトル $\\vec e_j$(縦ベクトル)を回転させたもの。")
    _step(t5, fig_step5(), "rg_d5")


AXES = {"z 軸 (0, 0, 1)": (0, 0, 1), "x 軸 (1, 0, 0)": (1, 0, 0), "(1, 1, 1)/√3": (1, 1, 1),
        "(1, 2, 2)/3": (1, 2, 2), "自由に決める(方位角・仰角)": None}


def _figure(v, k, th, deg):
    v, k = np.asarray(v, float), np.asarray(k, float) / np.linalg.norm(k)
    vp = rodrigues(v, k, th)
    par = k * np.dot(k, v)
    perp = v - par
    kv = np.cross(k, v)
    r = np.linalg.norm(perp)
    fig = go.Figure()
    L = max(2.5, np.linalg.norm(v) * 1.2)
    fig.add_trace(go.Scatter3d(x=[-L * k[0], L * k[0]], y=[-L * k[1], L * k[1]], z=[-L * k[2], L * k[2]], mode="lines",
                               line=dict(color=GRAY, width=3, dash="dash"), name="回転軸"))
    for tr in arrow3d((0, 0, 0), k, color="#333", name="k(単位ベクトル)", width=7):
        fig.add_trace(tr)
    if r > 1e-9:
        t = np.linspace(0, 2 * np.pi, 120)
        e1, e2 = perp / r, kv / r
        circ = par[:, None] + r * (np.outer(e1, np.cos(t)) + np.outer(e2, np.sin(t)))
        fig.add_trace(go.Scatter3d(x=circ[0], y=circ[1], z=circ[2], mode="lines", line=dict(color="#bbbbbb", width=3),
                                   name="先端の通る円"))
        ta = np.linspace(0, th, max(2, int(abs(deg) / 3) + 2))
        arc = par[:, None] + r * (np.outer(e1, np.cos(ta)) + np.outer(e2, np.sin(ta)))
        fig.add_trace(go.Scatter3d(x=arc[0], y=arc[1], z=arc[2], mode="lines", line=dict(color=C2, width=7),
                                   name=f"回転した角 θ = {deg:g}°"))
        for tr in arrow3d(par, perp, color=C4, name="v⊥", width=4):
            fig.add_trace(tr)
        for tr in arrow3d(par, kv, color=C3, name="k × v", width=4):
            fig.add_trace(tr)
    for tr in arrow3d((0, 0, 0), par, color=GRAY, name="v∥ = (k·v)k", width=4):
        fig.add_trace(tr)
    for tr in arrow3d((0, 0, 0), v, color=C1, name="v", width=7):
        fig.add_trace(tr)
    for tr in arrow3d((0, 0, 0), vp, color=C2, name="v′(回転後)", width=7):
        fig.add_trace(tr)
    title = f"θ = {deg:g}°   v′ = (" + ", ".join(f"{x:.3f}" for x in vp) + ")"
    layout3d(fig, height=560, title=title, ranges=[(-L, L)] * 3)
    fig.update_layout(uirevision="keep")
    return fig, vp


def _interactive():
    st.subheader("3. 動かして確かめる")
    c1, c2 = st.columns([1, 1])
    with c1:
        name = st.selectbox("回転軸の向き", list(AXES), key="rg_axis")
        if AXES[name] is None:
            az = st.slider("方位角 φ [°]", 0, 360, 30, 5, key="rg_az")
            el = st.slider("仰角 [°]", -90, 90, 50, 5, key="rg_el")
            k = (math.cos(math.radians(el)) * math.cos(math.radians(az)),
                 math.cos(math.radians(el)) * math.sin(math.radians(az)), math.sin(math.radians(el)))
        else:
            k = AXES[name]
    with c2:
        vx = st.slider("v の x 成分", -2.0, 2.0, 1.0, 0.5, key="rg_vx")
        vy = st.slider("v の y 成分", -2.0, 2.0, 0.0, 0.5, key="rg_vy")
        vz = st.slider("v の z 成分", -2.0, 2.0, 1.0, 0.5, key="rg_vz")
    v = (vx, vy, vz)
    kk = np.asarray(k, float) / np.linalg.norm(k)
    st.caption(f"k = ({kk[0]:.3f}, {kk[1]:.3f}, {kk[2]:.3f}),  k·v = {np.dot(kk, v):.3f}。"
               "灰色が軸に平行な部分 v∥(回転しても変わらない)、橙が v⊥、緑が k × v。v⊥ と k × v の張る平面の中で先端が円を描く。")
    degs = list(range(0, 361, 5))
    labels = [f"θ = {d}°" for d in degs]

    def draw(i):
        return _figure(v, k, math.radians(degs[i]), degs[i])[0]

    player("rodrigues", labels, draw)
    st.caption("回転しても長さは変わらない(|v′| = |v|)。回転行列 R は直交行列(RᵀR = I, det R = 1)。")


SCENES = [
    ("機械(ロボット)", "ロボットアームの関節は、それぞれの軸のまわりの回転。手先の位置は回転を順にかけて求める(順運動学)。"
     "軸が原点を通らないときは、軸上の点 $\\vec c$ を使って $\\vec p\\,'=\\vec c+R(\\vec p-\\vec c)$(例題3)。"),
    ("電気(磁気モーメント)", "磁場 $\\vec B$ の中の磁気モーメントや荷電粒子の速度は $\\dfrac{d\\vec v}{dt}=\\vec\\omega\\times\\vec v$ の形の方程式に従い、"
     "解は $\\vec\\omega$ のまわりの回転(歳差運動):$\\vec v(t)$ はロドリゲスの公式で $\\theta=|\\vec\\omega|t$ としたもの(例題4)。MRI の原理につながる。"),
    ("都市(測量・GNSS)", "地球の自転や座標系の回転(地心座標と局所座標の変換、ドローン写真測量のカメラの向き)は、軸と角度で表した回転として計算する。"),
    ("化学(分子・結晶)", "分子の立体配置の回転、結晶の回転対称(3回軸 = $(1,1,1)$ のまわりの $120^\\circ$ 回転など)の確認に使う(例題2)。"),
    ("情報・CG", "3D CG やゲーム、カメラの姿勢推定(OpenCV の `Rodrigues` 関数)は、回転を「軸 × 角度」の回転ベクトルで持ち、この公式で行列に直す。"
     "四元数 $q=\\left(\\cos\\frac\\theta2,\\ \\vec k\\sin\\frac\\theta2\\right)$ による回転も、展開するとこの公式になる。"),
]


EXAMPLES = [
    (r"$\vec k=(0,\ 0,\ 1)$ のとき、回転行列 $R=I+\sin\theta K+(1-\cos\theta)K^2$ を求めよ。また $\vec v=(1,\ 2,\ 3)$ を $z$ 軸のまわりに "
     r"$90^\circ$ 回転させたベクトルを求めよ。",
     [r"$K=\begin{pmatrix}0&-1&0\\1&0&0\\0&0&0\end{pmatrix}$, $K^2=\begin{pmatrix}-1&0&0\\0&-1&0\\0&0&0\end{pmatrix}$ より",
      r"$$R=\begin{pmatrix}\cos\theta&-\sin\theta&0\\ \sin\theta&\cos\theta&0\\ 0&0&1\end{pmatrix}$$($xy$ 平面の回転と同じ)",
      r"$\theta=90^\circ$ では $\vec v\,'=(-2,\ 1,\ 3)$。公式でも $\vec v\cos90^\circ+(\vec k\times\vec v)\sin90^\circ+\vec k(\vec k\cdot\vec v)(1-0)"
      r"=(-2,\ 1,\ 0)+(0,\ 0,\ 3)=(-2,\ 1,\ 3)$。"]),
    (r"軸 $\vec k=\dfrac{1}{\sqrt3}(1,\ 1,\ 1)$ のまわりに $\vec v=(1,\ 0,\ 0)$ を $120^\circ$ 回転させたベクトルを求めよ。",
     [r"$\cos120^\circ=-\dfrac12,\ \sin120^\circ=\dfrac{\sqrt3}{2}$。$\vec k\cdot\vec v=\dfrac{1}{\sqrt3}$, "
      r"$\vec k\times\vec v=\dfrac{1}{\sqrt3}(0,\ 1,\ -1)$。",
      r"$$\vec v\,'=-\frac12(1,0,0)+\frac{\sqrt3}{2}\cdot\frac{1}{\sqrt3}(0,1,-1)+\frac{1}{\sqrt3}\cdot\frac{1}{\sqrt3}(1,1,1)\cdot\frac32=(0,\ 1,\ 0)$$",
      r"$x$ 軸 → $y$ 軸 → $z$ 軸 と入れかわる回転(立方体や結晶の3回対称軸)。"]),
    (r"ロボットの関節の軸が点 $\mathrm C(1,\ 1,\ 0)$ を通り $z$ 軸に平行である。先端の点 $\mathrm P(2,\ 1,\ 5)$ をこの軸のまわりに "
     r"$90^\circ$ 回転させた点 $\mathrm P'$ を求めよ。",
     [r"軸上の点 $\mathrm C$ から見たベクトル $\vec{\mathrm{CP}}=(1,\ 0,\ 5)$ を回転させてから $\mathrm C$ を足す:$\vec p\,'=\vec c+R(\vec p-\vec c)$。",
      r"$z$ 軸方向の $90^\circ$ 回転で $(1,0,5)\to(0,\ 1,\ 5)$。よって $\mathrm P'=(1,1,0)+(0,1,5)=(1,\ 2,\ 5)$。"]),
    (r"$\vec\omega=(0,\ 0,\ 2)$ とする。$\dfrac{d\vec v}{dt}=\vec\omega\times\vec v,\ \vec v(0)=(1,\ 0,\ 1)$ の解が、"
     r"ロドリゲスの公式で $\vec k=\dfrac{\vec\omega}{|\vec\omega|},\ \theta=|\vec\omega|t$ としたものになることを確かめよ。",
     [r"$\vec k=(0,0,1),\ \theta=2t$ で公式を使うと $\vec v(t)=(\cos2t,\ \sin2t,\ 1)$。",
      r"微分すると $\vec v\,'(t)=(-2\sin2t,\ 2\cos2t,\ 0)$。一方 $\vec\omega\times\vec v=(0,0,2)\times(\cos2t,\sin2t,1)=(-2\sin2t,\ 2\cos2t,\ 0)$ で一致する。",
      r"$\vec v(0)=(1,0,1)$ も満たす。$\vec\omega$ のまわりを角速度 $|\vec\omega|$ で回り続ける(歳差運動)。"]),
]


def tab_rodrigues():
    st.info("3次元の回転を「回転軸の単位ベクトル $\\vec k$」と「回転角 $\\theta$」で表す公式。"
            "内積(軸方向の成分)と外積(軸に垂直な $90^\\circ$ 回した向き)だけで、どんな軸のまわりの回転も書ける。")
    _derivation()
    _interactive()
    st.subheader("4. どんな場面で使うか")
    for nm, txt in SCENES:
        st.markdown(f"- **{nm}**:{txt}")
    examples("rodrigues", EXAMPLES)
