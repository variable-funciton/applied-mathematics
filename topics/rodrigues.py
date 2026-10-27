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


def _derivation():
    st.subheader("1. 公式")
    st.markdown("単位ベクトル $\\vec k$($|\\vec k|=1$)を回転軸とし、ベクトル $\\vec v$ を $\\vec k$ の向きに右ねじが進む向きに角 $\\theta$ だけ回転させたベクトルを $\\vec v\\,'$ とすると")
    st.latex(r"\vec v\,'=\vec v\cos\theta+(\vec k\times\vec v)\sin\theta+\vec k\,(\vec k\cdot\vec v)(1-\cos\theta)")
    st.subheader("2. 導出")
    st.markdown(
        "**(i) 軸に平行な部分と垂直な部分に分ける。** 回転しても、軸に平行な部分は変わらず、垂直な部分だけが軸のまわりを回る。")
    st.latex(r"\vec v_{\parallel}=(\vec k\cdot\vec v)\,\vec k,\qquad \vec v_{\perp}=\vec v-(\vec k\cdot\vec v)\,\vec k")
    st.markdown(
        "**(ii) 回転の平面に直交する2本のベクトルをとる。** $\\vec v_\\perp$ と、それを軸のまわりに $90^\\circ$ 回したベクトル "
        "$\\vec k\\times\\vec v_\\perp$ は、どちらも軸に垂直で互いに直交し、長さが等しい($|\\vec k\\times\\vec v_\\perp|=|\\vec k||\\vec v_\\perp|\\sin90^\\circ=|\\vec v_\\perp|$)。"
        "さらに $\\vec k\\times\\vec v_\\parallel=\\vec 0$ なので $\\vec k\\times\\vec v_\\perp=\\vec k\\times\\vec v$。")
    st.markdown(
        "**(iii) 平面の回転と同じに考える。** $\\vec v_\\perp$ を $x$ 軸、$\\vec k\\times\\vec v$ を $y$ 軸とみれば、"
        "角 $\\theta$ の回転は $(\\cos\\theta,\\ \\sin\\theta)$ 方向への移動なので")
    st.latex(r"\vec v_{\perp}\,'=\vec v_\perp\cos\theta+(\vec k\times\vec v)\sin\theta")
    st.markdown("**(iv) 足し合わせる。** $\\vec v\\,'=\\vec v_\\parallel+\\vec v_\\perp\\,'$ に (i) を代入すると")
    st.latex(r"\vec v\,'=(\vec k\cdot\vec v)\vec k+\left\{\vec v-(\vec k\cdot\vec v)\vec k\right\}\cos\theta+(\vec k\times\vec v)\sin\theta"
             r"=\vec v\cos\theta+(\vec k\times\vec v)\sin\theta+\vec k\,(\vec k\cdot\vec v)(1-\cos\theta)")
    st.markdown(
        "**(v) 行列の形。** ベクトル三重積 $\\vec k\\times(\\vec k\\times\\vec v)=(\\vec k\\cdot\\vec v)\\vec k-\\vec v$ を使うと "
        "$\\vec k(\\vec k\\cdot\\vec v)=\\vec v+\\vec k\\times(\\vec k\\times\\vec v)$ なので、$\\vec k\\times$ を表す行列を $K$ として")
    st.latex(r"\vec v\,'=R\vec v,\qquad R=I+\sin\theta\,K+(1-\cos\theta)K^2,\qquad "
             r"K=\begin{pmatrix}0&-k_3&k_2\\ k_3&0&-k_1\\ -k_2&k_1&0\end{pmatrix}")
    st.caption("$\\vec k=(0,0,1)$ とすると、$xy$ 平面の回転行列 $\\begin{pmatrix}\\cos\\theta&-\\sin\\theta\\\\ \\sin\\theta&\\cos\\theta\\end{pmatrix}$ に戻る(例題1)。")


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
