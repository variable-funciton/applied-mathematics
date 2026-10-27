"""④ 複素関数:極形式/絶対値と偏角/正則関数/コーシー・リーマン/逆関数/複素積分/コーシーの積分定理・積分表示/関数の展開"""
from math import factorial

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import C1, C2, C3, C4, GRAY, arrow, fmt_c, layout, show, trapz

PI = np.pi


def _zpick(cols, key, default):
    x = cols[0].slider(f"{key} の実部", -2.0, 2.0, default[0], 0.1, key=f"zp_{key}_x")
    y = cols[1].slider(f"{key} の虚部", -2.0, 2.0, default[1], 0.1, key=f"zp_{key}_y")
    return complex(x, y)


def _polar_tex(z):
    r, th = abs(z), np.angle(z)
    return rf"{r:.3f}\,e^{{{th:.3f}i}}={r:.3f}(\cos{np.degrees(th):.1f}^\circ+i\sin{np.degrees(th):.1f}^\circ)"


def _plane(fig, m=3.0, height=480, title=None):
    fig.add_hline(y=0, line=dict(color=GRAY, width=1))
    fig.add_vline(x=0, line=dict(color=GRAY, width=1))
    return layout(fig, height=height, equal=True, xlab="実軸", ylab="虚軸", xrange=(-m, m), yrange=(-m, m), title=title)


# ============================================================
# 1-1. 複素数と極形式
# ============================================================
def tab_polar():
    st.info("複素数 $z=x+iy$ は平面上の点。極形式 $z=re^{i\\theta}$ で書くと、"
            "**かけ算 = 長さのかけ算 + 角度の足し算**(回転と拡大)になる。")
    st.latex(r"z=x+iy=r(\cos\theta+i\sin\theta)=re^{i\theta},\qquad z_1z_2=r_1r_2e^{i(\theta_1+\theta_2)},\quad "
             r"\frac{z_1}{z_2}=\frac{r_1}{r_2}e^{i(\theta_1-\theta_2)}")
    op = st.radio("演算", ["和 z₁ + z₂", "積 z₁ z₂", "商 z₁ / z₂", "共役 z̄₁", "累乗 z₁ⁿ(ド・モアブル)"],
                  horizontal=True, key="cp_op")
    c = st.columns(4)
    z1 = _zpick(c[:2], "z₁", (1.2, 0.6))
    z2 = _zpick(c[2:], "z₂", (0.4, 1.0))
    fig = go.Figure()
    fig.add_trace(arrow(0, 0, z1.real, z1.imag, C1, "z₁"))
    if op.startswith("累乗"):
        n = st.slider("n", 1, 12, 5, key="cp_n")
        pw = np.array([z1 ** k for k in range(n + 1)])
        fig.add_trace(go.Scatter(x=pw.real, y=pw.imag, mode="lines+markers+text", text=[f"z^{k}" for k in range(n + 1)],
                                 textposition="top right", line=dict(color=C3, dash="dot"), name="z⁰, z¹, …, zⁿ"))
        res = z1 ** n
        st.latex(rf"z_1^{{{n}}}=r^{{{n}}}e^{{i{n}\theta}}={_polar_tex(res)}")
        st.caption("|z₁| > 1 なら外へ広がる渦巻き、|z₁| < 1 なら内へ巻き込む渦巻き、|z₁| = 1 なら円周上を回る。")
    else:
        if not op.startswith("共役"):
            fig.add_trace(arrow(0, 0, z2.real, z2.imag, C2, "z₂"))
        if op.startswith("和"):
            res = z1 + z2
            fig.add_trace(go.Scatter(x=[z1.real, res.real, z2.real], y=[z1.imag, res.imag, z2.imag], mode="lines",
                                     line=dict(color=GRAY, dash="dot"), name="平行四辺形"))
        elif op.startswith("積"):
            res = z1 * z2
        elif op.startswith("商"):
            res = z1 / z2 if z2 != 0 else np.nan
        else:
            res = np.conj(z1)
        if np.isfinite(res):
            fig.add_trace(arrow(0, 0, res.real, res.imag, C3, "結果"))
            st.latex(rf"\text{{結果}}={fmt_c(res, 3)}={_polar_tex(res)}".replace("−", "-"))
    st.latex(rf"z_1={fmt_c(z1, 2)}={_polar_tex(z1)}".replace("−", "-"))
    st.latex(rf"z_2={fmt_c(z2, 2)}={_polar_tex(z2)}".replace("−", "-"))
    tt = np.linspace(0, 2 * PI, 200)
    fig.add_trace(go.Scatter(x=np.cos(tt), y=np.sin(tt), mode="lines", line=dict(color=GRAY, width=1, dash="dot"),
                             name="単位円"))
    _plane(fig, m=max(3.0, 1.1 * max(abs(z1), abs(z2), abs(res) if np.isfinite(res) else 0)))
    show(fig)
    with st.expander("練習問題"):
        st.markdown("**問1** $(1+i)^{10}$ を求めよ。")
        st.markdown("**問2** $z=1+\\sqrt3 i$ をかけると、平面上の点はどう動くか。")
        if st.toggle("解答・解説を表示", key="pr_complex2_76"):
            st.markdown("**答1**")
            st.latex(r"1+i=\sqrt2e^{i\pi/4}\ \Rightarrow\ (1+i)^{10}=32e^{i5\pi/2}=32i")
            st.markdown("**答2**")
            st.write("$z=2e^{i\\pi/3}$ なので、**2倍に拡大して 60° 回転**。")


# ============================================================
# 1-2. 絶対値と偏角
# ============================================================
def tab_abs_arg():
    st.info("絶対値はかけ算で積に、偏角はかけ算で和になる。偏角は 2π の整数倍だけ不定で、"
            "−π < Arg z ≤ π にとったものを主値という。")
    st.latex(r"|z_1z_2|=|z_1||z_2|,\quad \arg(z_1z_2)=\arg z_1+\arg z_2,\quad |z_1+z_2|\le|z_1|+|z_2|,\quad "
             r"\arg z=\mathrm{Arg}\,z+2n\pi")
    st.subheader("n 乗根")
    c = st.columns(3)
    n = c[0].slider("n", 2, 12, 5, key="ca_n")
    zr = c[1].slider("z の絶対値", 0.2, 3.0, 1.0, 0.1, key="ca_r")
    zt = c[2].slider("z の偏角(度)", -180, 180, 60, 5, key="ca_t")
    z = zr * np.exp(1j * np.radians(zt))
    roots = zr ** (1 / n) * np.exp(1j * (np.radians(zt) + 2 * PI * np.arange(n)) / n)
    st.latex(rf"w^{n}=z\ \Longrightarrow\ w_k=\sqrt[{n}]{{r}}\,\exp\!\left(i\,\frac{{\theta+2k\pi}}{{{n}}}\right)\quad(k=0,1,\ldots,{n - 1})")
    fig = go.Figure()
    tt = np.linspace(0, 2 * PI, 300)
    fig.add_trace(go.Scatter(x=zr ** (1 / n) * np.cos(tt), y=zr ** (1 / n) * np.sin(tt), mode="lines",
                             line=dict(color=GRAY, dash="dot"), name=f"半径 r^(1/{n}) の円"))
    fig.add_trace(go.Scatter(x=np.r_[roots.real, roots.real[0]], y=np.r_[roots.imag, roots.imag[0]],
                             mode="lines+markers", line=dict(color=C2), marker=dict(size=9), name="n 乗根(正 n 角形)"))
    fig.add_trace(go.Scatter(x=[z.real], y=[z.imag], mode="markers", marker=dict(color=C1, size=12, symbol="star"),
                             name="z"))
    _plane(fig, m=max(1.6, zr * 1.2))
    cA, cB = st.columns([3, 2])
    with cA:
        show(fig)
    with cB:
        st.dataframe([{"k": k, "w_k": fmt_c(w, 4), "偏角(度)": f"{np.degrees(np.angle(w)):.1f}"} for k, w in enumerate(roots)],
                     hide_index=True, width="stretch")
        st.caption(f"n 乗根は円周上に等間隔(360°/{n} = {360 / n:.1f}°ごと)に並ぶ。")

    st.subheader("三角不等式")
    c = st.columns(4)
    z1 = _zpick(c[:2], "a", (1.5, 0.5))
    z2 = _zpick(c[2:], "b", (-0.5, 1.2))
    fig = go.Figure()
    fig.add_trace(arrow(0, 0, z1.real, z1.imag, C1, "z₁"))
    fig.add_trace(arrow(z1.real, z1.imag, z2.real, z2.imag, C2, "z₂"))
    s = z1 + z2
    fig.add_trace(arrow(0, 0, s.real, s.imag, C3, "z₁ + z₂"))
    _plane(fig, m=3, height=380)
    show(fig)
    st.write(f"|z₁ + z₂| = **{abs(s):.4f}** ≤ |z₁| + |z₂| = **{abs(z1) + abs(z2):.4f}**"
             "(等号は z₁, z₂ が同じ向きのとき)。")
    with st.expander("練習問題"):
        st.markdown("**問3** $z^3=8i$ を解け。")
        if st.toggle("解答・解説を表示", key="pr_complex2_129"):
            st.markdown("**答3**")
            st.latex(r"8i=8e^{i\pi/2}\ \Rightarrow\ z=2e^{i(\pi/6+2k\pi/3)}=\sqrt3+i,\ -\sqrt3+i,\ -2i")


# ============================================================
# 1-4. 正則関数
# ============================================================
HF = {
    "f = z²(正則)": (lambda z: z ** 2, lambda z: 2 * z),
    "f = e^z(正則)": (np.exp, np.exp),
    "f = 1/z(z ≠ 0 で正則)": (lambda z: 1 / z, lambda z: -1 / z ** 2),
    "f = z̄(正則でない)": (np.conj, None),
    "f = |z|²(正則でない)": (lambda z: np.abs(z) ** 2, None),
    "f = Re z(正則でない)": (lambda z: np.real(z) + 0j, None),
}


def tab_holomorphic():
    st.info("複素微分 $f'(z_0)=\\lim_{h\\to0}\\frac{f(z_0+h)-f(z_0)}{h}$ では、h が**どの方向から 0 に近づいても**"
            "同じ値になる必要がある。これが成り立つ(正則な)関数は特別。")
    c = st.columns(4)
    name = c[0].selectbox("関数", list(HF), key="ch_f")
    x0 = c[1].slider("z₀ の実部", -1.5, 1.5, 0.8, 0.1, key="ch_x")
    y0 = c[2].slider("z₀ の虚部", -1.5, 1.5, 0.5, 0.1, key="ch_y")
    hr = c[3].slider("|h|", 0.001, 0.5, 0.2, 0.001, key="ch_h")
    f, fp = HF[name]
    z0 = complex(x0, y0)
    ph = np.linspace(0, 2 * PI, 72, endpoint=False)
    h = hr * np.exp(1j * ph)
    with np.errstate(all="ignore"):
        Q = (f(z0 + h) - f(z0)) / h
    fig = go.Figure(go.Scatter(x=Q.real, y=Q.imag, mode="markers",
                               marker=dict(color=np.degrees(ph), colorscale="HSV", size=9,
                                           colorbar=dict(title="h の方向(度)", thickness=12)),
                               name="差分商"))
    if fp is not None:
        d = fp(z0)
        fig.add_trace(go.Scatter(x=[d.real], y=[d.imag], mode="markers", marker=dict(symbol="x", size=14, color="black"),
                                 name="f′(z₀)"))
    m = max(1.0, float(np.nanmax(np.abs(Q))) * 1.2)
    _plane(fig, m=m, height=480, title="h の方向ごとの差分商 (f(z₀+h) − f(z₀))/h")
    cA, cB = st.columns([3, 2])
    with cA:
        show(fig)
    with cB:
        spread = float(np.nanmax(np.abs(Q - np.nanmean(Q))))
        st.metric("差分商のばらつき", f"{spread:.4f}")
        if fp is not None:
            st.success(f"|h| を小さくすると、すべての方向で f′(z₀) = {fmt_c(fp(z0), 4)} に集まる → 正則。")
        else:
            st.error("|h| を小さくしても、方向によって値が違う(円や線分に広がる) → 複素微分できない。")
        st.caption("例:f = z̄ では (z̄₀ + h̄ − z̄₀)/h = h̄/h = e^{−2iφ} で、方向 φ により単位円を一周する。")
    with st.expander("練習問題"):
        st.markdown("**問4** $f(z)=\\bar z$ が、どの点でも微分可能でないことを示せ。")
        if st.toggle("解答・解説を表示", key="pr_complex2_182"):
            st.markdown("**答4**")
            st.latex(r"\frac{\bar h}{h}=\begin{cases}1&(h\ \text{実数})\\-1&(h\ \text{純虚数})\end{cases}\ \Rightarrow\ \text{極限なし}")


# ============================================================
# 1-5. コーシー・リーマンの関係式
# ============================================================
CRF = {
    "f = z²": (lambda z: z ** 2, r"u=x^2-y^2,\ \ v=2xy"),
    "f = z³": (lambda z: z ** 3, r"u=x^3-3xy^2,\ \ v=3x^2y-y^3"),
    "f = e^z": (np.exp, r"u=e^x\cos y,\ \ v=e^x\sin y"),
    "f = 1/z": (lambda z: 1 / z, r"u=\frac{x}{x^2+y^2},\ \ v=\frac{-y}{x^2+y^2}"),
    "f = sin z": (np.sin, r"u=\sin x\cosh y,\ \ v=\cos x\sinh y"),
    "f = z̄(正則でない)": (np.conj, r"u=x,\ \ v=-y"),
    "f = |z|²(正則でない)": (lambda z: np.abs(z) ** 2 + 0j, r"u=x^2+y^2,\ \ v=0"),
}


def tab_cauchy_riemann():
    st.info("$f=u+iv$ が正則 ⇔ u, v がコーシー・リーマンの関係式を満たす。"
            "このとき **u = 一定 と v = 一定 の曲線は直交**し、u, v はどちらも調和関数になる。")
    st.latex(r"\frac{\partial u}{\partial x}=\frac{\partial v}{\partial y},\qquad \frac{\partial u}{\partial y}=-\frac{\partial v}{\partial x},"
             r"\qquad f'(z)=u_x+iv_x,\qquad \Delta u=\Delta v=0")
    c = st.columns(3)
    name = c[0].selectbox("関数", list(CRF), key="cr2_f")
    x0 = c[1].slider("点の x", -1.5, 1.5, 0.7, 0.1, key="cr2_x")
    y0 = c[2].slider("点の y", -1.5, 1.5, 0.4, 0.1, key="cr2_y")
    f, tex = CRF[name]
    st.latex(tex)
    h = 1e-5
    with np.errstate(all="ignore"):
        fx = (f(complex(x0 + h, y0)) - f(complex(x0 - h, y0))) / (2 * h)
        fy = (f(complex(x0, y0 + h)) - f(complex(x0, y0 - h))) / (2 * h)
    ux, vx, uy, vy = fx.real, fx.imag, fy.real, fy.imag
    st.dataframe([{"": "1つ目の式", "左辺": f"u_x = {ux:.5f}", "右辺": f"v_y = {vy:.5f}",
                   "成立?": "○" if abs(ux - vy) < 1e-6 else "×"},
                  {"": "2つ目の式", "左辺": f"u_y = {uy:.5f}", "右辺": f"−v_x = {-vx:.5f}",
                   "成立?": "○" if abs(uy + vx) < 1e-6 else "×"}], hide_index=True, width="stretch")
    g = np.linspace(-2, 2, 241)
    X, Y = np.meshgrid(g, g)
    with np.errstate(all="ignore"):
        W = f(X + 1j * Y)
    U, V = np.clip(W.real, -6, 6), np.clip(W.imag, -6, 6)
    fig = go.Figure()
    fig.add_trace(go.Contour(x=g, y=g, z=U, ncontours=26, contours=dict(coloring="none"),
                             line=dict(color=C1, width=1.6), showscale=False, name="u = 一定", showlegend=True))
    fig.add_trace(go.Contour(x=g, y=g, z=V, ncontours=26, contours=dict(coloring="none"),
                             line=dict(color=C2, width=1.6), showscale=False, name="v = 一定", showlegend=True))
    fig.add_trace(go.Scatter(x=[x0], y=[y0], mode="markers", marker=dict(color="black", size=9), name="点"))
    layout(fig, height=520, equal=True, xlab="x", ylab="y", xrange=(-2, 2), yrange=(-2, 2))
    show(fig)
    st.caption("正則なら、青(u の等高線)と赤(v の等高線)は直角に交わる(f′ ≠ 0 の点で)。")
    with st.expander("練習問題"):
        st.markdown("**問5** $u=x^2-y^2+2x$ を実部とする正則関数を求めよ。")
        if st.toggle("解答・解説を表示", key="pr_complex2_235"):
            st.markdown("**答5**")
            st.latex(r"v_y=u_x=2x+2\Rightarrow v=2xy+2y+\varphi(x),\ v_x=-u_y=2y\Rightarrow\varphi'=0")
            st.latex(r"f=(x^2-y^2+2x)+i(2xy+2y)+iC=z^2+2z+iC")


# ============================================================
# 1-6. 逆関数(多価性)
# ============================================================
def tab_inverse_fn():
    st.info("$e^w=z$ や $w^2=z$ の解 $w=\\log z,\\ \\sqrt z$ は、z が原点のまわりを一周すると値が変わる(多価関数)。"
            "主値は偏角を $-\\pi<\\theta\\le\\pi$ に制限して1つに決めたもの。")
    st.latex(r"\log z=\log|z|+i\arg z=\mathrm{Log}\,z+2n\pi i,\qquad \sqrt z=\sqrt r\,e^{i\theta/2}\ (\text{2価}),\qquad "
             r"z^\alpha=e^{\alpha\log z}")
    c = st.columns(3)
    fn = c[0].radio("関数", ["√z", "log z", "∛z"], key="ci_f")
    r = c[1].slider("円の半径 r", 0.3, 2.5, 1.5, 0.1, key="ci_r")
    th = c[2].slider("z が回った角度 θ(度)", 0, 1080, 360, 10, key="ci_th")
    ts = np.radians(np.linspace(0, th, 400))
    z = r * np.exp(1j * ts)
    if fn == "√z":
        w = np.sqrt(r) * np.exp(1j * ts / 2)
        wp = np.sqrt(z[-1])
        period = 720
    elif fn == "∛z":
        w = r ** (1 / 3) * np.exp(1j * ts / 3)
        wp = z[-1] ** (1 / 3)
        period = 1080
    else:
        w = np.log(r) + 1j * ts
        wp = np.log(z[-1])
        period = None
    cA, cB = st.columns(2)
    with cA:
        fig = go.Figure(go.Scatter(x=z.real, y=z.imag, mode="lines", line=dict(color=C1, width=3), name="z の道"))
        fig.add_trace(go.Scatter(x=[z[0].real, z[-1].real], y=[z[0].imag, z[-1].imag], mode="markers",
                                 marker=dict(color=["black", C2], size=11), name="始点・終点"))
        fig.add_trace(go.Scatter(x=[-3, 0], y=[0, 0], mode="lines", line=dict(color=C4, width=4, dash="dot"),
                                 name="主値の切断線"))
        _plane(fig, m=3, height=420, title="z 平面")
        show(fig)
    with cB:
        fig = go.Figure(go.Scatter(x=w.real, y=w.imag, mode="lines", line=dict(color=C2, width=3), name="連続に追った w"))
        fig.add_trace(go.Scatter(x=[w[-1].real], y=[w[-1].imag], mode="markers", marker=dict(color=C2, size=11),
                                 name="連続に追った値"))
        fig.add_trace(go.Scatter(x=[wp.real], y=[wp.imag], mode="markers", marker=dict(color="black", symbol="x", size=13),
                                 name="主値"))
        m = max(2.0, float(np.abs(w).max()) * 1.15)
        if fn == "log z":
            layout(fig, height=420, xlab="Re w", ylab="Im w", xrange=(-2, 2), yrange=(-1, max(7, np.radians(th) + 1)),
                   title="w 平面")
        else:
            _plane(fig, m=m, height=420, title="w 平面")
        show(fig)
    if period:
        st.write(f"z が1周(360°)しても w は元に戻らず、{period}° 回って初めて戻る。")
    else:
        st.write(f"z が1周するごとに、log z の虚部は 2π ずつ増える(いま {np.radians(th) / (2 * PI):.2f} 周)。")
    with st.expander("練習問題"):
        st.markdown("**問6** $\\log(-1)$、$\\mathrm{Log}\\,i$、$i^i$ の主値を求めよ。")
        if st.toggle("解答・解説を表示", key="pr_complex2_293"):
            st.markdown("**答6**")
            st.latex(r"\log(-1)=(2n+1)\pi i,\quad \mathrm{Log}\,i=\frac\pi2i,\quad i^i=e^{i\,\mathrm{Log}\,i}=e^{-\pi/2}\approx0.2079")


# ============================================================
# 2-1. 複素積分
# ============================================================
CIF = {
    "f = z": (lambda z: z, lambda z: z ** 2 / 2),
    "f = z²": (lambda z: z ** 2, lambda z: z ** 3 / 3),
    "f = e^z": (np.exp, np.exp),
    "f = 1/z": (lambda z: 1 / z, None),
    "f = z̄": (np.conj, None),
}


def path_int(f, z, dz):
    """中点則(t は区間の中点で与える):∫_0^1 f(z(t)) z'(t) dt"""
    return np.mean(f(z) * dz)


def tab_complex_integral():
    st.info("複素積分は、曲線をパラメータ表示して $\\int f(z(t))z'(t)\\,dt$ を計算する。"
            "正則関数なら、途中の道を変形しても(特異点をまたがなければ)値は変わらない。")
    st.latex(r"\int_Cf(z)\,dz=\int_a^bf(z(t))\,z'(t)\,dt")
    c = st.columns(5)
    name = c[0].selectbox("関数", list(CIF), key="cint_f")
    A = complex(c[1].slider("始点 A の実部", -2.0, 2.0, 1.0, 0.1, key="cint_ax"),
                c[2].slider("始点 A の虚部", -2.0, 2.0, 0.0, 0.1, key="cint_ay"))
    B = complex(c[3].slider("終点 B の実部", -2.0, 2.0, -1.0, 0.1, key="cint_bx"),
                c[4].slider("終点 B の虚部", -2.0, 2.0, 0.5, 0.1, key="cint_by"))
    k = st.slider("曲げた道の「ふくらみ」k", -2.0, 2.0, 1.0, 0.05, key="cint_k")
    f, Fp = CIF[name]
    t = (np.arange(20000) + 0.5) / 20000
    paths = {
        "直線": (A + (B - A) * t, (B - A) * np.ones_like(t)),
        "曲げた道": (A + (B - A) * t + 1j * k * (B - A) * t * (1 - t), (B - A) + 1j * k * (B - A) * (1 - 2 * t)),
    }
    h = t < 0.5
    s = np.where(h, 2 * t, 2 * t - 1)
    corner = complex(A.real, B.imag)
    paths["折れ線(縦→横)"] = (np.where(h, A + (corner - A) * s, corner + (B - corner) * s),
                          np.where(h, 2 * (corner - A), 2 * (B - corner)))
    fig = go.Figure()
    rows = []
    for (nm, (z, dz)), col in zip(paths.items(), (C1, C2, C3)):
        with np.errstate(all="ignore"):
            val = path_int(f, z, dz)
        rows.append({"道": nm, "∫ f dz": fmt_c(val, 5) if np.isfinite(val) else "定義されない(0 を通る)"})
        fig.add_trace(go.Scatter(x=z.real[::40], y=z.imag[::40], mode="lines", line=dict(color=col, width=3), name=nm))
    fig.add_trace(go.Scatter(x=[A.real, B.real], y=[A.imag, B.imag], mode="markers+text", text=["A", "B"],
                             textposition="top center", marker=dict(size=10, color="black"), showlegend=False))
    if name == "f = 1/z":
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(symbol="x", size=13, color=C2), name="特異点"))
    _plane(fig, m=3, height=460)
    cA, cB = st.columns([3, 2])
    with cA:
        show(fig)
    with cB:
        st.dataframe(rows, hide_index=True, width="stretch")
        if Fp is not None:
            st.success(f"原始関数があるので、どの道でも F(B) − F(A) = {fmt_c(Fp(B) - Fp(A), 5)}")
        elif name == "f = 1/z":
            st.warning("原点をはさんで道を変えると、値が 2πi ≈ 6.2832i だけ変わる(log z の多価性)。")
        else:
            st.error("z̄ は正則でないので、道によって値が変わる。")
    with st.expander("練習問題"):
        st.markdown("**問7** C:$z=t+it^2\\ (0\\le t\\le1)$ のとき $\\int_C z\\,dz$ と $\\int_C\\bar z\\,dz$ を求めよ。")
        if st.toggle("解答・解説を表示", key="pr_complex2_360"):
            st.markdown("**答7**")
            st.latex(r"\int_C z\,dz=\frac{(1+i)^2}{2}=i,\qquad \int_C\bar z\,dz=\int_0^1(t-it^2)(1+2it)\,dt=1+\frac{i}{3}")


# ============================================================
# 2-2. コーシーの積分定理
# ============================================================
SHAPES = {
    "円": lambda th, R: R * np.exp(1j * th),
    "楕円": lambda th, R: R * (1.5 * np.cos(th) + 0.7j * np.sin(th)),
    "正方形": None,
    "星形": lambda th, R: R * (1 + 0.35 * np.cos(5 * th)) * np.exp(1j * th),
}
CTF = {
    "f = z²": lambda z: z ** 2,
    "f = e^z sin z": lambda z: np.exp(z) * np.sin(z),
    "f = 1/z": lambda z: 1 / z,
    "f = 1/z²": lambda z: 1 / z ** 2,
    "f = 1/(z − 1)": lambda z: 1 / (z - 1),
    "f = z̄(正則でない)": np.conj,
}


def closed_curve(shape, R, c, n=20000):
    th = np.linspace(0, 2 * PI, n + 1)
    if shape == "正方形":
        s = th / (2 * PI) * 4
        side = np.minimum(np.floor(s), 3)
        u = s - side
        pts = np.select([side == 0, side == 1, side == 2, side == 3],
                        [R * (1 - 1j) + 2j * R * u, R * (1 + 1j) - 2 * R * u, R * (-1 + 1j) - 2j * R * u,
                         R * (-1 - 1j) + 2 * R * u])
    else:
        pts = SHAPES[shape](th, R)
    return c + pts


def tab_cauchy_theorem():
    st.info("f が閉曲線 C とその内部で正則なら、$\\oint_C f\\,dz=0$。曲線の形はどうでもよい。"
            "内部に特異点があると、値は特異点だけで決まる(形を変えても同じ)。")
    st.latex(r"\oint_Cf(z)\,dz=0\quad(f\ \text{が}\ C\ \text{の内部で正則}),\qquad \oint_C\frac{dz}{z-a}=2\pi i\quad(a\ \text{が内部})")
    c = st.columns(5)
    name = c[0].selectbox("関数", list(CTF), index=2, key="ct_f")
    shape = c[1].selectbox("曲線の形", list(SHAPES), key="ct_s")
    R = c[2].slider("大きさ", 0.3, 2.0, 1.0, 0.05, key="ct_R")
    cx = c[3].slider("中心の実部", -2.0, 2.0, 0.0, 0.1, key="ct_cx")
    cy = c[4].slider("中心の虚部", -2.0, 2.0, 0.0, 0.1, key="ct_cy")
    z = closed_curve(shape, R, complex(cx, cy))
    dz = np.diff(z)
    zm = (z[1:] + z[:-1]) / 2
    with np.errstate(all="ignore"):
        val = np.sum(CTF[name](zm) * dz)
    area = 0.5 * np.sum((z[:-1].real * z[1:].imag - z[1:].real * z[:-1].imag))
    fig = go.Figure(go.Scatter(x=z.real, y=z.imag, mode="lines", fill="toself", fillcolor="rgba(31,119,180,0.12)",
                               line=dict(color=C1, width=3), name="C"))
    sing = {"f = 1/z": [0], "f = 1/z²": [0], "f = 1/(z − 1)": [1]}.get(name, [])
    if sing:
        fig.add_trace(go.Scatter(x=sing, y=[0] * len(sing), mode="markers", marker=dict(symbol="x", size=13, color=C2),
                                 name="特異点"))
    _plane(fig, m=3.5, height=460)
    cA, cB = st.columns([3, 2])
    with cA:
        show(fig)
    with cB:
        st.latex(r"\oint_Cf\,dz\approx " + fmt_c(val, 5).replace("−", "-"))
        if name.startswith("f = z̄"):
            st.latex(rf"2i\times(\text{{面積}})={fmt_c(2j * area, 5)}".replace("−", "-"))
            st.caption("z̄ は正則でないので 0 にならない。グリーンの定理から $\\oint\\bar z\\,dz=2i\\times$面積。")
        elif name == "f = 1/z²":
            st.caption("原点を囲んでも 0(原始関数 −1/z が1価なので)。留数が 0 の例。")
        else:
            st.caption("特異点を囲めば 2πi ≈ 6.28319i、囲まなければ 0。形を変えても変わらない。")
    with st.expander("グリーンの定理からの証明"):
        st.latex(r"\oint_C f\,dz=\oint_C(u\,dx-v\,dy)+i\oint_C(v\,dx+u\,dy)"
                 r"=\iint_D(-v_x-u_y)\,dxdy+i\iint_D(u_x-v_y)\,dxdy=0")
        st.caption("最後はコーシー・リーマンの関係式による。(① ベクトル解析のグリーンの定理とつながる)")


# ============================================================
# 2-3. コーシーの積分表示
# ============================================================
CFF = {
    "e^z": (np.exp, [np.exp] * 4),
    "sin z": (np.sin, [np.sin, np.cos, lambda z: -np.sin(z), lambda z: -np.cos(z)]),
    "z³ − z": (lambda z: z ** 3 - z, [lambda z: z ** 3 - z, lambda z: 3 * z ** 2 - 1, lambda z: 6 * z,
                                    lambda z: 6 + 0 * z]),
    "1/(z − 3)(z = 3 で特異)": (lambda z: 1 / (z - 3), [lambda z: 1 / (z - 3), lambda z: -1 / (z - 3) ** 2,
                                                     lambda z: 2 / (z - 3) ** 3, lambda z: -6 / (z - 3) ** 4]),
}


def tab_cauchy_formula():
    st.info("正則関数の値は、まわりの閉曲線上の値だけで決まる。導関数も同様に積分で表せる"
            "(正則関数は何回でも微分できる)。")
    st.latex(r"f(a)=\frac{1}{2\pi i}\oint_C\frac{f(z)}{z-a}dz,\qquad f^{(n)}(a)=\frac{n!}{2\pi i}\oint_C\frac{f(z)}{(z-a)^{n+1}}dz")
    c = st.columns(5)
    name = c[0].selectbox("f(z)", list(CFF), key="cf_f")
    n = c[1].slider("微分の回数 n", 0, 3, 0, key="cf_n")
    a = complex(c[2].slider("a の実部", -2.0, 2.0, 0.5, 0.1, key="cf_ax"),
                c[3].slider("a の虚部", -2.0, 2.0, 0.3, 0.1, key="cf_ay"))
    R = c[4].slider("円 C の半径(中心は原点)", 0.2, 4.0, 1.5, 0.1, key="cf_R")
    f, ders = CFF[name]
    th = np.linspace(0, 2 * PI, 8192, endpoint=False)
    z = R * np.exp(1j * th)
    with np.errstate(all="ignore"):
        val = factorial(n) / (2j * PI) * np.sum(f(z) / (z - a) ** (n + 1) * 1j * z) * (2 * PI / 8192)
    exact = ders[n](a)
    inside = abs(a) < R
    fig = go.Figure(go.Scatter(x=z.real, y=z.imag, mode="lines", line=dict(color=C1, width=3), name="C"))
    fig.add_trace(go.Scatter(x=[a.real], y=[a.imag], mode="markers+text", text=["a"], textposition="top right",
                             marker=dict(size=11, color=C2), name="a"))
    if "z − 3" in name:
        fig.add_trace(go.Scatter(x=[3], y=[0], mode="markers", marker=dict(symbol="x", size=13, color="black"),
                                 name="f の特異点"))
    _plane(fig, m=4.5, height=440)
    cA, cB = st.columns([3, 2])
    with cA:
        show(fig)
    with cB:
        st.latex(rf"\frac{{{factorial(n)}}}{{2\pi i}}\oint_C\frac{{f(z)}}{{(z-a)^{{{n + 1}}}}}dz\approx " + fmt_c(val, 6).replace("−", "-"))
        st.latex(rf"f^{{({n})}}(a)=" + fmt_c(exact, 6).replace("−", "-"))
        if not inside:
            st.warning("a が C の外にあるので、積分は 0(被積分関数が C の内部で正則)。")
        elif "z − 3" in name and R > 3:
            st.error("C の内部に f の特異点 z = 3 があるので、積分表示は成り立たない。")
        else:
            st.success("一致。")
    with st.expander("練習問題"):
        st.markdown("**問8** $\\displaystyle\\oint_{|z|=2}\\frac{e^z}{(z-1)^3}dz$ を求めよ。")
        if st.toggle("解答・解説を表示", key="pr_complex2_488"):
            st.markdown("**答8**")
            st.latex(r"\frac{2\pi i}{2!}f''(1)=\pi i\,e")


# ============================================================
# 2-4. 関数の展開
# ============================================================
EXF = {
    "1/(1 − z)": dict(f=lambda z: 1 / (1 - z), sing=[1], tex=r"\frac{1}{1-z}"),
    "e^z": dict(f=np.exp, sing=[], tex=r"e^z"),
    "1/(1 + z²)": dict(f=lambda z: 1 / (1 + z ** 2), sing=[1j, -1j], tex=r"\frac{1}{1+z^2}"),
    "Log(1 + z)": dict(f=lambda z: np.log(1 + z), sing=[-1], tex=r"\mathrm{Log}(1+z)"),
    "tan z": dict(f=np.tan, sing=[PI / 2, -PI / 2], tex=r"\tan z"),
}


def taylor_coeffs(f, c, rho, N, M=1024):
    th = 2 * PI * np.arange(M) / M
    vals = f(c + rho * np.exp(1j * th))
    a = np.fft.fft(vals) / M
    return np.array([a[n] / rho ** n for n in range(N + 1)])


def tab_expansion():
    st.info("正則関数は、点 c のまわりでテイラー級数に展開できる。収束半径は **c から一番近い特異点までの距離**。"
            "実軸上だけ見ていてもわからない(1/(1+z²) は実数全体でなめらかなのに、収束半径は 1)。")
    st.latex(r"f(z)=\sum_{n=0}^{\infty}a_n(z-c)^n,\qquad a_n=\frac{f^{(n)}(c)}{n!}=\frac{1}{2\pi i}\oint\frac{f(z)}{(z-c)^{n+1}}dz")
    c = st.columns(4)
    name = c[0].selectbox("関数", list(EXF), index=2, key="ex_f")
    N = c[1].slider("次数 N", 1, 40, 8, key="ex_N")
    cx = c[2].slider("中心 c の実部", -1.0, 1.0, 0.0, 0.1, key="ex_cx")
    cy = c[3].slider("中心 c の虚部", -1.0, 1.0, 0.0, 0.1, key="ex_cy")
    E = EXF[name]
    cc = complex(cx, cy)
    Rc = min((abs(s - cc) for s in E["sing"]), default=np.inf)
    if name.startswith("Log"):
        st.caption("Log(1+z) の主値は z ≤ −1 の実軸で不連続なので、中心は z = −1 から離しておく。")
    if Rc < 0.05:
        st.warning("中心が特異点に近すぎます。")
        return
    a = taylor_coeffs(E["f"], cc, 0.5 * min(Rc, 2.0), N)
    Rtex = r"\infty" if not np.isfinite(Rc) else f"{Rc:.4f}"
    st.latex(rf"f(z)={E['tex']},\qquad \text{{収束半径}}\ R={Rtex}")
    st.dataframe([{"n": k, "a_n": fmt_c(a[k], 5)} for k in range(min(N, 8) + 1)], hide_index=True, width="stretch")

    g = np.linspace(-2.5, 2.5, 301)
    X, Y = np.meshgrid(g, g)
    Z = X + 1j * Y
    S = sum(a[k] * (Z - cc) ** k for k in range(N + 1))
    with np.errstate(all="ignore"):
        err = np.log10(np.abs(E["f"](Z) - S) + 1e-16)
    fig = go.Figure(go.Heatmap(x=g, y=g, z=np.clip(err, -12, 3), colorscale="Viridis",
                               colorbar=dict(title="log₁₀|f − S_N|", thickness=12)))
    if np.isfinite(Rc):
        tt = np.linspace(0, 2 * PI, 300)
        fig.add_trace(go.Scatter(x=cx + Rc * np.cos(tt), y=cy + Rc * np.sin(tt), mode="lines",
                                 line=dict(color="white", width=3, dash="dash"), name="収束円"))
    for s in E["sing"]:
        fig.add_trace(go.Scatter(x=[np.real(s)], y=[np.imag(s)], mode="markers",
                                 marker=dict(symbol="x", size=13, color=C2), showlegend=False))
    fig.add_trace(go.Scatter(x=[cx], y=[cy], mode="markers", marker=dict(color="white", size=9), name="中心 c"))
    layout(fig, title="誤差 |f − S_N| の大きさ(暗いほど小さい)", height=500, equal=True, xlab="Re z", ylab="Im z",
           xrange=(-2.5, 2.5), yrange=(-2.5, 2.5))
    cA, cB = st.columns([3, 2])
    with cA:
        show(fig)
    with cB:
        xr = np.linspace(-2.5, 2.5, 801)
        with np.errstate(all="ignore"):
            fr = E["f"](xr + 0j).real
        Sr = sum(a[k] * (xr - cc) ** k for k in range(N + 1)).real
        f2 = go.Figure(go.Scatter(x=xr, y=fr, mode="lines", line=dict(color=GRAY, width=2), name="f(x)"))
        f2.add_trace(go.Scatter(x=xr, y=Sr, mode="lines", line=dict(color=C1, width=2), name=f"S_{N}(x)"))
        layout(f2, title="実軸上で比べる", xlab="x", height=500, yrange=(-3, 4))
        show(f2)
        st.caption("収束円の内側では N を増やすと誤差が 0 に、外側では発散する。")
    with st.expander("ローラン展開"):
        st.write("特異点のまわりでは、負のべきも含むローラン級数に展開する。展開の形は領域で変わる。例:")
        st.latex(r"\frac{1}{z(1-z)}=\frac1z+1+z+z^2+\cdots\ (0<|z|<1),\qquad "
                 r"\frac{1}{z(1-z)}=-\frac{1}{z^2}-\frac{1}{z^3}-\cdots\ (|z|>1)")
        st.caption("$\\frac1z$ の係数が留数(0 < |z| < 1 の展開で 1)。")
    with st.expander("練習問題"):
        st.markdown("**問9** $\\dfrac{1}{1+z^2}$ を $z=0$ のまわりで展開し、収束半径を求めよ。")
        if st.toggle("解答・解説を表示", key="pr_complex2_570"):
            st.markdown("**答9**")
            st.latex(r"1-z^2+z^4-\cdots,\quad R=1\ (\text{特異点}\ \pm i\ \text{までの距離})")
