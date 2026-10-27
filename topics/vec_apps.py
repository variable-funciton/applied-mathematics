"""① ベクトル解析 応用:学科別の応用例とサンプル例題(機械・電気・化学・都市)

各節は「応用例の説明 → 動かして確かめる図 → 例題(問題を先に並べ、解答はトグルで表示)」の順。
例題の数値は tests で独立に検算している(EXAMPLES の各 check 関数)。
"""
import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import C1, C2, C3, C4, GRAY, layout, quiver


def examples(key, items):
    """items: [(問題文, [解答の段落, ...]), ...]"""
    st.subheader("例題")
    for i, (q, _) in enumerate(items, 1):
        st.markdown(f"**例題{i}** {q}")
    if st.toggle("解答・解説を表示", key=f"app_ans_{key}"):
        for i, (_, ans) in enumerate(items, 1):
            st.markdown(f"**答{i}**")
            for a in ans:
                st.markdown(a)


def _trap(y, x):
    return float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(x)))


def line_integral(Fx, Fy, path, n=4001):
    """∫_C F·dr(path(t) = (x, y), 0 ≤ t ≤ 1)"""
    t = np.linspace(0, 1, n)
    x, y = path(t)
    return _trap(Fx(x, y) * np.gradient(x, t) + Fy(x, y) * np.gradient(y, t), t)


def circle_flux(Ex, Ey, cx, cy, R, n=4000):
    """円(反時計回り)を外向きに通る流束 ∮ E·n ds"""
    th = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    x, y = cx + R * np.cos(th), cy + R * np.sin(th)
    return float(np.sum(Ex(x, y) * np.cos(th) + Ey(x, y) * np.sin(th)) * R * 2 * np.pi / n)


def circle_circulation(Ex, Ey, cx, cy, R, n=4000):
    th = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    x, y = cx + R * np.cos(th), cy + R * np.sin(th)
    return float(np.sum(-Ex(x, y) * np.sin(th) + Ey(x, y) * np.cos(th)) * R * 2 * np.pi / n)


# ============================================================
# 機械:仕事と線積分、回転する物体、流れ
# ============================================================
MECH_F = {
    "F = (2xy, x²)(保存力:ポテンシャル x²y)": (lambda x, y: 2 * x * y, lambda x, y: x ** 2 + 0 * y),
    "F = (−y, x)(回転させる力:保存力でない)": (lambda x, y: -y, lambda x, y: x + 0 * y),
    "F = (0, −mg)(重力、m g = 1)": (lambda x, y: 0 * x, lambda x, y: -1 + 0 * y),
}
MECH_PATH = {
    "直線 O → A": lambda t: (t, 2 * t),
    "折れ線 O → (1, 0) → A": lambda t: (np.where(t < 0.5, 2 * t, 1.0), np.where(t < 0.5, 0.0, 2 * (2 * t - 1))),
    "放物線 y = 2x²": lambda t: (t, 2 * t ** 2),
    "折れ線 O → (0, 2) → A": lambda t: (np.where(t < 0.5, 0.0, 2 * t - 1), np.where(t < 0.5, 4 * t, 2.0)),
}


def tab_mech():
    st.info("機械工学では、**力のする仕事**(線積分)、**回転する物体の速度場**(回転 rot)、"
            "**流体の流量**(面積分・発散 div)としてベクトル解析が使われる。")
    st.markdown(
        "- **仕事** $W=\\displaystyle\\int_C\\vec F\\cdot d\\vec r$:物体を曲線 $C$ に沿って動かすときに力 $\\vec F$ がする仕事。"
        "**保存力**(重力・ばねの力)なら $\\vec F=\\nabla\\varphi$ と書け、仕事は道筋によらず始点と終点だけで決まる。\n"
        "- **剛体の回転**:角速度ベクトル $\\vec\\omega$ で回る物体の各点の速度は $\\vec v=\\vec\\omega\\times\\vec r$ で、"
        "$\\mathrm{rot}\\,\\vec v=2\\vec\\omega$(回転の強さの2倍)。\n"
        "- **流体**:流量は速度場の流束 $\\iint_S\\vec v\\cdot\\vec n\\,dS$。水のような非圧縮性の流れでは $\\mathrm{div}\\,\\vec v=0$(連続の式)。")
    st.subheader("動かして確かめる:道筋を変えて仕事を比べる")
    c1, c2 = st.columns(2)
    fname = c1.radio("力", list(MECH_F), key="mech_F")
    pname = c2.radio("道筋(O(0, 0) → A(1, 2))", list(MECH_PATH), key="mech_P")
    Fx, Fy = MECH_F[fname]
    g = np.linspace(-0.25, 2.25, 11)
    X, Y = np.meshgrid(g, g)
    fig = go.Figure(quiver(X, Y, Fx(X, Y), Fy(X, Y), length=0.18, normalize=True, color=GRAY, name="力 F"))
    rows = []
    for i, (nm, path) in enumerate(MECH_PATH.items()):
        t = np.linspace(0, 1, 201)
        x, y = path(t)
        W = line_integral(Fx, Fy, path)
        rows.append((nm, W))
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=nm, line=dict(width=5 if nm == pname else 1.8,
                                 color=[C1, C2, C3, C4][i], dash="solid" if nm == pname else "dot")))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 2], mode="markers+text", text=["O", "A"], textposition="top left",
                             marker=dict(size=10, color="black"), showlegend=False))
    layout(fig, height=480, equal=True, xlab="x", ylab="y", xrange=(-0.3, 2.3), yrange=(-0.3, 2.3))
    a, b = st.columns([3, 2])
    with a:
        st.plotly_chart(fig, width="stretch", key="mech_fig")
    with b:
        st.markdown("**仕事 $W$(数値計算)**")
        st.markdown("\n".join(f"- {nm}: **{W:.4f}**" for nm, W in rows))
        same = max(W for _, W in rows) - min(W for _, W in rows) < 1e-6
        st.success("道筋によらず仕事は同じ → 保存力") if same else st.warning("道筋によって仕事が違う → 保存力ではない")
        st.caption("保存力かどうかは $\\mathrm{rot}\\,\\vec F=\\vec 0$(平面なら $\\frac{\\partial F_2}{\\partial x}-\\frac{\\partial F_1}{\\partial y}=0$)で判定できる。")
    examples("mech", MECH_EXAMPLES)


MECH_EXAMPLES = [
    (r"力 $\vec F=(2xy,\ x^2)$ の下で、物体を原点 $\mathrm O$ から点 $\mathrm A(1,\ 2)$ まで動かす。"
     r"(1) 線分 $\mathrm{OA}$、(2) 折れ線 $\mathrm O\to(1,0)\to\mathrm A$ に沿って、力のする仕事をそれぞれ求めよ。"
     r"(3) $\vec F=\nabla\varphi$ となる $\varphi$ を求め、結果を説明せよ。",
     [r"(1) $\vec r(t)=(t,\ 2t)\ (0\le t\le1)$ より $\vec F\cdot\dfrac{d\vec r}{dt}=2t\cdot2t\cdot1+t^2\cdot2=6t^2$, "
      r"$W=\displaystyle\int_0^16t^2dt=2$。",
      r"(2) $(t,0)$ では $\vec F=(0,\ t^2)$, $d\vec r=(1,0)dt$ なので $0$。$(1,t)\ (0\le t\le2)$ では $\vec F\cdot(0,1)=1$ なので "
      r"$\displaystyle\int_0^2dt=2$。合計 $W=2$。",
      r"(3) $\varphi=x^2y$ とすると $\nabla\varphi=(2xy,\ x^2)=\vec F$。仕事は $\varphi(\mathrm A)-\varphi(\mathrm O)=2$ で、"
      r"道筋によらない(保存力)。"]),
    (r"$z$ 軸のまわりに角速度 $\omega=3\ \mathrm{rad/s}$ で回る円板の速度場 $\vec v=\vec\omega\times\vec r$($\vec\omega=(0,0,3)$)を求め、"
     r"$\mathrm{rot}\,\vec v$ を計算せよ。また、半径 $2\ \mathrm m$ の円周に沿う循環 $\displaystyle\oint_C\vec v\cdot d\vec r$ を求め、"
     r"ストークスの定理で確かめよ。",
     [r"$\vec v=(0,0,3)\times(x,y,z)=(-3y,\ 3x,\ 0)$。",
      r"$\mathrm{rot}\,\vec v=\left(0,\ 0,\ \dfrac{\partial(3x)}{\partial x}-\dfrac{\partial(-3y)}{\partial y}\right)=(0,\ 0,\ 6)=2\vec\omega$。",
      r"円周 $\vec r=(2\cos t,\ 2\sin t,\ 0)$ では $\vec v\cdot\dfrac{d\vec r}{dt}=12\sin^2t+12\cos^2t=12$, "
      r"$\displaystyle\oint_C\vec v\cdot d\vec r=\int_0^{2\pi}12\,dt=24\pi\ \mathrm{m^2/s}$。",
      r"ストークスの定理:$\displaystyle\iint_S(\mathrm{rot}\,\vec v)\cdot\vec n\,dS=6\times\pi\cdot2^2=24\pi$ で一致する。"]),
    (r"平面の水の流れ $\vec v=(x,\ -y)\ \mathrm{[m/s]}$ について、$\mathrm{div}\,\vec v$ を求めよ。"
     r"また、正方形 $0\le x\le1,\ 0\le y\le1$ の周から外へ流れ出る流量 $\displaystyle\oint_C\vec v\cdot\vec n\,ds$ を求めよ。",
     [r"$\mathrm{div}\,\vec v=\dfrac{\partial x}{\partial x}+\dfrac{\partial(-y)}{\partial y}=1-1=0$(非圧縮性の流れ)。",
      r"右の辺 $x=1$:$\vec n=(1,0)$, $\vec v\cdot\vec n=1$ で $1$。左の辺 $x=0$:$\vec v\cdot(-1,0)=-x=0$ で $0$。",
      r"上の辺 $y=1$:$\vec v\cdot(0,1)=-1$ で $-1$。下の辺 $y=0$:$\vec v\cdot(0,-1)=y=0$ で $0$。",
      r"合計 $0$。右から入った分だけ上へ出ていく。ガウスの発散定理(平面版)$\displaystyle\iint_D\mathrm{div}\,\vec v\,dxdy=0$ と一致する。"]),
]


# ============================================================
# 電気:電場と電位、ガウスの法則、アンペールの法則
# ============================================================
def line_charge_field(charges):
    """平面の線電荷(単位 λ/(2πε₀) = 1):E = Σ q (r - rᵢ)/|r - rᵢ|²、φ = -Σ q log|r - rᵢ|"""
    def Ex(x, y):
        return sum(q * (x - a) / ((x - a) ** 2 + (y - b) ** 2) for q, a, b in charges)

    def Ey(x, y):
        return sum(q * (y - b) / ((x - a) ** 2 + (y - b) ** 2) for q, a, b in charges)

    def phi(x, y):
        return sum(-q * 0.5 * np.log((x - a) ** 2 + (y - b) ** 2) for q, a, b in charges)
    return Ex, Ey, phi


def tab_elec():
    st.info("電気工学では、**電場と電位**(勾配)、**ガウスの法則**(発散・面積分)、"
            "**アンペールの法則・電磁誘導**(回転・線積分)としてベクトル解析が使われる。マクスウェル方程式はすべて div と rot で書かれている。")
    st.markdown(
        "- **電位と電場**:$\\vec E=-\\nabla\\varphi$。電場は電位の下がる向きで、等電位線に垂直。\n"
        "- **ガウスの法則**:閉曲面を貫く電場の流束は、内側の電荷だけで決まる:$\\displaystyle\\oiint_S\\vec E\\cdot\\vec n\\,dS=\\frac{Q}{\\varepsilon_0}$、"
        "微分形は $\\mathrm{div}\\,\\vec E=\\dfrac{\\rho}{\\varepsilon_0}$。\n"
        "- **アンペールの法則**:電流のまわりの磁場の循環 $\\displaystyle\\oint_C\\vec B\\cdot d\\vec r=\\mu_0I$、微分形は $\\mathrm{rot}\\,\\vec B=\\mu_0\\vec j$。")
    st.subheader("動かして確かめる:閉曲線を貫く電場の流束(ガウスの法則)")
    st.caption("紙面に垂直な線電荷が2本ある平面の電場(単位は λ/(2πε₀) = 1)。円を動かすと、流束は円の内側の電荷だけで決まる。")
    c1, c2, c3, c4, c5 = st.columns(5)
    q1 = c1.slider("電荷 q₁(点 (−1, 0))", -2.0, 2.0, 1.0, 0.5, key="el_q1")
    q2 = c2.slider("電荷 q₂(点 (1, 0))", -2.0, 2.0, -1.0, 0.5, key="el_q2")
    cx = c3.slider("円の中心 x", -2.0, 2.0, -1.0, 0.1, key="el_cx")
    cy = c4.slider("円の中心 y", -2.0, 2.0, 0.0, 0.1, key="el_cy")
    R = c5.slider("円の半径", 0.2, 2.5, 0.8, 0.1, key="el_R")
    charges = [(q1, -1.0, 0.0), (q2, 1.0, 0.0)]
    Ex, Ey, phi = line_charge_field(charges)
    gb = np.linspace(-3, 3, 151)
    XB, YB = np.meshgrid(gb, gb)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = phi(XB, YB)
    P = np.clip(P, -3, 3)
    fig = go.Figure(go.Contour(x=gb, y=gb, z=P, colorscale="RdBu_r", contours=dict(coloring="heatmap", showlines=True),
                               line=dict(width=0.6), opacity=0.55, colorbar=dict(title="電位 φ", thickness=12), hoverinfo="skip"))
    g = np.linspace(-2.8, 2.8, 15)
    X, Y = np.meshgrid(g, g)
    with np.errstate(divide="ignore", invalid="ignore"):
        U, V = Ex(X, Y), Ey(X, Y)
    fig.add_trace(quiver(X, Y, U, V, length=0.3, normalize=True, color="#222", name="電場 E"))
    th = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(go.Scatter(x=cx + R * np.cos(th), y=cy + R * np.sin(th), mode="lines", line=dict(color=C3, width=3),
                             name="閉曲線 C"))
    for q, a, b in charges:
        fig.add_trace(go.Scatter(x=[a], y=[b], mode="markers+text", text=[f"{q:+g}"], textposition="top center",
                                 marker=dict(size=14, color=C2 if q > 0 else C1, line=dict(color="black", width=1)),
                                 showlegend=False))
    layout(fig, height=520, equal=True, xlab="x", ylab="y", xrange=(-3, 3), yrange=(-3, 3))
    near = any(abs(math.hypot(a - cx, b - cy) - R) < 0.05 for _, a, b in charges)
    flux = None if near else circle_flux(Ex, Ey, cx, cy, R)
    inside = sum(q for q, a, b in charges if math.hypot(a - cx, b - cy) < R)
    A, B = st.columns([3, 2])
    with A:
        st.plotly_chart(fig, width="stretch", key="el_fig")
    with B:
        if flux is None:
            st.warning("円が電荷の上を通っている。少し動かそう。")
        else:
            st.metric("流束 ∮E·n ds(数値)", f"{flux:.4f}")
            st.metric("2π × (内側の電荷の和)", f"{2 * math.pi * inside:.4f}")
        st.caption("円の外の電荷は、入る分と出る分が打ち消し合うので流束に効かない。"
                   "これが $\\mathrm{div}\\,\\vec E=0$(電荷のない所)とガウスの発散定理の意味。")
    examples("elec", ELEC_EXAMPLES)


ELEC_EXAMPLES = [
    (r"電位が $\varphi=x^2-y^2+2z\ \mathrm{[V]}$($x,y,z$ の単位は $\mathrm{m}$)で与えられるとき、電場 $\vec E=-\nabla\varphi$ を求め、"
     r"点 $(1,\ 2,\ 0)$ での電場とその大きさを求めよ。",
     [r"$\nabla\varphi=(2x,\ -2y,\ 2)$ より $\vec E=(-2x,\ 2y,\ -2)$。",
      r"点 $(1,2,0)$ では $\vec E=(-2,\ 4,\ -2)\ \mathrm{V/m}$, $|\vec E|=\sqrt{4+16+4}=2\sqrt6\ \mathrm{V/m}$。"]),
    (r"原点にある点電荷 $q$ がつくる電場 $\vec E=\dfrac{q}{4\pi\varepsilon_0}\dfrac{\vec r}{|\vec r|^3}$ について、"
     r"原点を中心とする半径 $a$ の球面 $S$ を外向きに貫く流束 $\displaystyle\iint_S\vec E\cdot\vec n\,dS$ を求めよ。"
     r"また、$\vec r\ne\vec 0$ で $\mathrm{div}\,\vec E=0$ であることを示せ。",
     [r"球面上では $\vec n=\dfrac{\vec r}{a}$, $|\vec r|=a$ なので $\vec E\cdot\vec n=\dfrac{q}{4\pi\varepsilon_0a^2}$(一定)。",
      r"$$\iint_S\vec E\cdot\vec n\,dS=\frac{q}{4\pi\varepsilon_0a^2}\times4\pi a^2=\frac{q}{\varepsilon_0}$$($a$ によらない:ガウスの法則)",
      r"$r=|\vec r|$ として $\dfrac{\partial}{\partial x}\dfrac{x}{r^3}=\dfrac{1}{r^3}-\dfrac{3x^2}{r^5}$。"
      r"$y,\ z$ も同様に足すと $\dfrac{3}{r^3}-\dfrac{3(x^2+y^2+z^2)}{r^5}=0$。"]),
    (r"$z$ 軸上を流れる直線電流 $I$ のまわりの磁場は $\vec B=\dfrac{\mu_0I}{2\pi}\dfrac{(-y,\ x,\ 0)}{x^2+y^2}$ である。"
     r"$xy$ 平面上の原点中心・半径 $a$ の円 $C$(反時計回り)に沿う循環 $\displaystyle\oint_C\vec B\cdot d\vec r$ を求めよ。"
     r"また、$z$ 軸の外で $\mathrm{rot}\,\vec B=\vec 0$ であることを示せ。",
     [r"$\vec r=(a\cos t,\ a\sin t,\ 0)$ とすると $\vec B\cdot\dfrac{d\vec r}{dt}=\dfrac{\mu_0I}{2\pi a^2}(a^2\sin^2t+a^2\cos^2t)=\dfrac{\mu_0I}{2\pi}$。",
      r"$$\oint_C\vec B\cdot d\vec r=\int_0^{2\pi}\frac{\mu_0I}{2\pi}dt=\mu_0I$$($a$ によらない:アンペールの法則)",
      r"$\mathrm{rot}\,\vec B$ の $z$ 成分は $\dfrac{\mu_0I}{2\pi}\left\{\dfrac{\partial}{\partial x}\dfrac{x}{x^2+y^2}+\dfrac{\partial}{\partial y}\dfrac{y}{x^2+y^2}\right\}"
      r"=\dfrac{\mu_0I}{2\pi}\cdot\dfrac{(y^2-x^2)+(x^2-y^2)}{(x^2+y^2)^2}=0$。$x,\ y$ 成分は $0$。"
      r"電流の外では渦なし(第2章「天気予報の風」のランキン渦の外側と同じ形)。"]),
]


# ============================================================
# 化学:拡散(フィックの法則)と連続の式
# ============================================================
def gauss_conc(x, y, M, D, t):
    return M / (4 * math.pi * D * t) * np.exp(-(x ** 2 + y ** 2) / (4 * D * t))


def tab_chem():
    st.info("化学では、**濃度の勾配による拡散**(フィックの法則)、**物質量の保存**(連続の式・発散)、"
            "**反応器や膜を通る物質の流束**(面積分)としてベクトル解析が使われる。")
    st.markdown(
        "- **フィックの第1法則**:拡散流束 $\\vec J=-D\\nabla c$($D$:拡散係数、$c$:濃度)。物質は濃度の高い所から低い所へ、"
        "勾配と逆向きに移動する。熱の場合は $\\vec q=-k\\nabla T$(フーリエの法則)。\n"
        "- **連続の式**:$\\dfrac{\\partial c}{\\partial t}=-\\mathrm{div}\\,\\vec J$(出ていく分だけ減る)。"
        "合わせると拡散方程式 $\\dfrac{\\partial c}{\\partial t}=D\\nabla^2c$ になる(⑤ 偏微分方程式の熱方程式と同じ形)。\n"
        "- **流束**:膜や粒子の表面を通る物質の量は $\\iint_S\\vec J\\cdot\\vec n\\,dS$。")
    st.subheader("動かして確かめる:点からの拡散と、円を通って出ていく量")
    st.caption("時刻 0 に原点に置いた物質量 M が平面に広がるときの濃度 c(x, y, t) と拡散流束 J = −D∇c。")
    c1, c2, c3 = st.columns(3)
    D = c1.select_slider("拡散係数 D", [0.05, 0.1, 0.2, 0.5], value=0.1, key="ch_D")
    t = c2.slider("時刻 t", 0.1, 5.0, 1.0, 0.1, key="ch_t")
    R = c3.slider("円の半径 R", 0.2, 2.0, 0.8, 0.1, key="ch_R")
    M = 1.0
    L = 2.5
    gb = np.linspace(-L, L, 121)
    XB, YB = np.meshgrid(gb, gb)
    C = gauss_conc(XB, YB, M, D, t)
    fig = go.Figure(go.Heatmap(x=gb, y=gb, z=C, colorscale="YlOrRd", colorbar=dict(title="濃度 c", thickness=12),
                               hoverinfo="skip"))
    g = np.linspace(-L * 0.9, L * 0.9, 13)
    X, Y = np.meshgrid(g, g)
    cc = gauss_conc(X, Y, M, D, t)
    Jx, Jy = cc * X / (2 * t), cc * Y / (2 * t)        # J = −D∇c = c r/(2t)
    fig.add_trace(quiver(X, Y, Jx, Jy, length=0.35, normalize=False, color="#222", name="拡散流束 J"))
    th = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(go.Scatter(x=R * np.cos(th), y=R * np.sin(th), mode="lines", line=dict(color=C1, width=3), name="円 C"))
    layout(fig, height=500, equal=True, xlab="x", ylab="y", xrange=(-L, L), yrange=(-L, L))
    flux = circle_flux(lambda x, y: gauss_conc(x, y, M, D, t) * x / (2 * t),
                       lambda x, y: gauss_conc(x, y, M, D, t) * y / (2 * t), 0, 0, R)
    inside = M * (1 - math.exp(-R * R / (4 * D * t)))
    dmdt = -M * R * R / (4 * D * t * t) * math.exp(-R * R / (4 * D * t))
    A, B = st.columns([3, 2])
    with A:
        st.plotly_chart(fig, width="stretch", key="ch_fig")
    with B:
        st.metric("円の内側の物質量", f"{inside:.4f}")
        st.metric("外へ出る流束 ∮J·n ds(数値)", f"{flux:.4f}")
        st.metric("内側の量の減る速さ −dm/dt", f"{-dmdt:.4f}")
        st.caption("流束と「内側の量の減る速さ」が一致する:連続の式 ∂c/∂t = −div J を円の内側で積分した形(発散定理)。")
    examples("chem", CHEM_EXAMPLES)


CHEM_EXAMPLES = [
    (r"濃度分布が $c=3x^2+2y^2+z\ \mathrm{[mol/m^3]}$ のとき、点 $(1,\ 1,\ 1)$ での拡散流束 $\vec J=-D\nabla c$ を求めよ。"
     r"また、この点で濃度が最も速く増加する向きの単位ベクトルを求めよ。",
     [r"$\nabla c=(6x,\ 4y,\ 1)$ より、点 $(1,1,1)$ で $\nabla c=(6,\ 4,\ 1)$, $\vec J=-D(6,\ 4,\ 1)$。",
      r"最も速く増加する向きは $\nabla c$ の向き:$\dfrac{1}{\sqrt{53}}(6,\ 4,\ 1)$。物質はその逆向きに拡散する。"]),
    (r"濃度分布 $c=x^2+y^2+z^2$ について、$\mathrm{div}\,\vec J$($\vec J=-D\nabla c$)を求め、"
     r"連続の式 $\dfrac{\partial c}{\partial t}=-\mathrm{div}\,\vec J$ から、この瞬間に濃度が増えるか減るかを答えよ。",
     [r"$\vec J=-D(2x,\ 2y,\ 2z)$ より $\mathrm{div}\,\vec J=-D(2+2+2)=-6D$。",
      r"$\dfrac{\partial c}{\partial t}=6D>0$ なので、どの点でも濃度は増える(周りから物質が集まってくる。$\nabla^2c>0$ の点は「くぼみ」)。"]),
    (r"半径 $a$ の球形の粒子が濃度 $c_\infty$ の溶液中にあり、表面の濃度が $c_s$ に保たれている。"
     r"定常状態の濃度は $c=c_\infty-(c_\infty-c_s)\dfrac{a}{r}$($r=|\vec r|\ge a$)である。"
     r"(1) $\nabla^2\left(\dfrac1r\right)=0$ を確かめ、この $c$ が定常な拡散方程式 $\nabla^2c=0$ を満たすことを示せ。"
     r"(2) 半径 $R\ (\ge a)$ の球面を通って粒子へ向かう物質の流量(内向きの流束)を求めよ。",
     [r"(1) $\nabla\dfrac1r=-\dfrac{\vec r}{r^3}$。電気の例題2と同じ計算で $\mathrm{div}\dfrac{\vec r}{r^3}=0$ なので $\nabla^2\dfrac1r=0$。"
      r"$c$ は定数と $\dfrac1r$ の定数倍の和なので $\nabla^2c=0$。",
      r"(2) $\nabla c=(c_\infty-c_s)a\dfrac{\vec r}{r^3}$, $\vec J=-D\nabla c$。球面上では $\vec J\cdot\left(-\dfrac{\vec r}{R}\right)=\dfrac{D(c_\infty-c_s)a}{R^2}$(内向き)。",
      r"$$\text{流量}=\frac{D(c_\infty-c_s)a}{R^2}\times4\pi R^2=4\pi Da(c_\infty-c_s)$$"
      r"$R$ によらない(定常なので途中でたまらない)。溶解・結晶成長・触媒反応の速さの基本式になる。"]),
]


# ============================================================
# 都市:地形の勾配と雨水、地下水、建物の熱
# ============================================================
HILLS = [(60.0, 1.0, 1.2, 1.0), (35.0, -1.4, -0.6, 0.8), (25.0, 1.6, -1.5, 0.7)]   # 高さ, 中心 x, y, 広がり


def terrain(x, y):
    h = 5.0 + 0 * x
    for H, a, b, s in HILLS:
        h = h + H * np.exp(-((x - a) ** 2 + (y - b) ** 2) / (2 * s * s))
    return h


def terrain_grad(x, y):
    gx, gy = 0 * x, 0 * y
    for H, a, b, s in HILLS:
        e = H * np.exp(-((x - a) ** 2 + (y - b) ** 2) / (2 * s * s))
        gx = gx - e * (x - a) / (s * s)
        gy = gy - e * (y - b) / (s * s)
    return gx, gy


def rain_path(x0, y0, h=0.01, n=1500, L=3.0):
    """最大傾斜の向き −∇h にそって流れる雨水の道(勾配の大きさで正規化して進む)"""
    p = [(x0, y0)]
    x, y = x0, y0
    for _ in range(n):
        gx, gy = terrain_grad(x, y)
        m = math.hypot(gx, gy)
        if m < 1e-3 or abs(x) > L or abs(y) > L:
            break
        x, y = x - h * gx / m, y - h * gy / m
        p.append((x, y))
    return np.array(p)


def tab_city():
    st.info("都市・建設系では、**地形の勾配と雨水の流れ**(勾配)、**地下水や汚染物質の移動**(ダルシーの法則・発散)、"
            "**建物の壁を通る熱**(流束)としてベクトル解析が使われる。")
    st.markdown(
        "- **地形**:標高 $h(x,y)$ の勾配 $\\nabla h$ は最も急に登る向き。雨水は $-\\nabla h$ の向き(最大傾斜線)に流れ、"
        "傾斜の大きさは $|\\nabla h|$(傾斜角 $\\theta$ は $\\tan\\theta=|\\nabla h|$)。流域や土砂災害の危険箇所の判定に使う。\n"
        "- **地下水**:ダルシーの法則 $\\vec v=-K\\nabla H$($K$:透水係数、$H$:水頭)。井戸のくみ上げや地盤の浸透の計算になる。\n"
        "- **建物の熱**:壁を通る熱流束 $\\vec q=-k\\nabla T$。壁全体の熱損失は $\\iint_S\\vec q\\cdot\\vec n\\,dS$。")
    st.subheader("動かして確かめる:雨水はどこへ流れるか")
    c1, c2 = st.columns(2)
    x0 = c1.slider("雨の落ちる点 x [km]", -2.5, 2.5, 0.3, 0.1, key="ct_x")
    y0 = c2.slider("雨の落ちる点 y [km]", -2.5, 2.5, 2.0, 0.1, key="ct_y")
    gb = np.linspace(-3, 3, 121)
    XB, YB = np.meshgrid(gb, gb)
    fig = go.Figure(go.Contour(x=gb, y=gb, z=terrain(XB, YB), colorscale="Earth", ncontours=18,
                               contours=dict(showlabels=True, labelfont=dict(size=10)),
                               colorbar=dict(title="標高 [m]", thickness=12), hoverinfo="skip"))
    g = np.linspace(-2.8, 2.8, 13)
    X, Y = np.meshgrid(g, g)
    gx, gy = terrain_grad(X, Y)
    fig.add_trace(quiver(X, Y, -gx, -gy, length=0.3, normalize=True, color="#333", name="−∇h(水の流れる向き)"))
    path = rain_path(x0, y0)
    fig.add_trace(go.Scatter(x=path[:, 0], y=path[:, 1], mode="lines", line=dict(color="#1f77b4", width=4), name="雨水の道"))
    fig.add_trace(go.Scatter(x=[x0], y=[y0], mode="markers", marker=dict(size=11, color=C2), name="雨の落ちる点"))
    layout(fig, height=520, equal=True, xlab="x [km]", ylab="y [km]", xrange=(-3, 3), yrange=(-3, 3))
    hx, hy = terrain_grad(np.array(x0), np.array(y0))
    hx, hy = float(hx) / 1000, float(hy) / 1000                       # m/km → m/m
    slope = math.hypot(hx, hy)
    A, B = st.columns([3, 2])
    with A:
        st.plotly_chart(fig, width="stretch", key="ct_fig")
    with B:
        st.metric("標高 h", f"{float(terrain(np.array(x0), np.array(y0))):.1f} m")
        st.metric("勾配の大きさ |∇h|", f"{slope:.4f}")
        st.metric("傾斜角 θ", f"{math.degrees(math.atan(slope)):.2f}°")
        st.caption("矢印は等高線に垂直。雨水の道は等高線を直角に横切りながら、くぼ地や谷へ集まる。")
    examples("city", CITY_EXAMPLES)


CITY_EXAMPLES = [
    (r"ある斜面の標高が $h=50-0.02x^2-0.01y^2\ \mathrm{[m]}$($x,y$ は $\mathrm{m}$)で表されるとき、点 $(10,\ 20)$ での勾配 $\nabla h$、"
     r"雨水の流れる向きの単位ベクトル、斜面の傾斜角 $\theta$ を求めよ。",
     [r"$\nabla h=(-0.04x,\ -0.02y)$ より、点 $(10,20)$ で $\nabla h=(-0.4,\ -0.4)$。",
      r"雨水は $-\nabla h=(0.4,\ 0.4)$ の向きに流れる。単位ベクトルは $\dfrac{1}{\sqrt2}(1,\ 1)$。",
      r"$|\nabla h|=0.4\sqrt2\approx0.566$ なので $\tan\theta=0.566$, $\theta\approx29.5^\circ$。"]),
    (r"厚さ $b$ の帯水層の中心にある井戸から、毎秒 $Q\ \mathrm{[m^3/s]}$ の地下水をくみ上げる。"
     r"井戸に向かう地下水の流速を $\vec v=-\dfrac{Q}{2\pi b}\dfrac{(x,\ y)}{x^2+y^2}$ とするとき、井戸を中心とする半径 $r$ の円を"
     r"外向きに通る流量 $b\displaystyle\oint_C\vec v\cdot\vec n\,ds$ を求めよ。また、井戸の外で $\mathrm{div}\,\vec v=0$ を示せ。",
     [r"円周上では $\vec n=\dfrac{(x,y)}{r}$ なので $\vec v\cdot\vec n=-\dfrac{Q}{2\pi br}$(一定)。",
      r"$$b\oint_C\vec v\cdot\vec n\,ds=b\times\left(-\frac{Q}{2\pi br}\right)\times2\pi r=-Q$$"
      r"どの半径でも $-Q$(内向きに $Q$):くみ上げた量だけ周囲から集まる。",
      r"$\dfrac{\partial}{\partial x}\dfrac{x}{x^2+y^2}+\dfrac{\partial}{\partial y}\dfrac{y}{x^2+y^2}=\dfrac{(y^2-x^2)+(x^2-y^2)}{(x^2+y^2)^2}=0$ "
      r"より $\mathrm{div}\,\vec v=0$。井戸の外では水は湧き出しも消えもしない。"]),
    (r"厚さ $0.2\ \mathrm m$ のコンクリート壁(熱伝導率 $k=1.5\ \mathrm{W/(m\cdot K)}$)の室内側が $20^\circ\mathrm C$、屋外側が $5^\circ\mathrm C$ である。"
     r"壁の中の温度を $T=20-75x$($x$ は室内側からの距離 $[\mathrm{m}]$)とするとき、熱流束 $\vec q=-k\nabla T$ を求め、"
     r"面積 $10\ \mathrm{m^2}$ の壁から逃げる熱量(毎秒)を求めよ。",
     [r"$\nabla T=(-75,\ 0,\ 0)\ \mathrm{K/m}$ より $\vec q=-1.5\times(-75,\ 0,\ 0)=(112.5,\ 0,\ 0)\ \mathrm{W/m^2}$(屋外向き)。",
      r"壁の面の法線 $\vec n=(1,0,0)$ について $\displaystyle\iint_S\vec q\cdot\vec n\,dS=112.5\times10=1125\ \mathrm W$。",
      r"断熱材で $k$ を小さくすると、同じ温度差でも熱流束が小さくなる。"]),
]
