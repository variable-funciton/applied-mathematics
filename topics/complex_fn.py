"""④ 複素関数:等角写像・定義域の色付け・留数定理・実積分への応用"""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import C1, C2, C3, C4, GRAY, fmt_c, hsv_to_rgb, layout, show, trapz

PI = np.pi

MAPS = {
    "w = z²": dict(f=lambda z: z ** 2, tex=r"w=z^2"),
    "w = 1/z": dict(f=lambda z: 1 / z, tex=r"w=\frac1z"),
    "w = e^z": dict(f=np.exp, tex=r"w=e^z=e^x(\cos y+i\sin y)"),
    "w = z + 1/z(ジューコフスキー変換)": dict(f=lambda z: z + 1 / z, tex=r"w=z+\frac1z"),
    "w = (z − 1)/(z + 1)(メビウス変換)": dict(f=lambda z: (z - 1) / (z + 1), tex=r"w=\frac{z-1}{z+1}"),
    "w = sin z": dict(f=np.sin, tex=r"w=\sin z"),
}




# ---------- 写像 ----------
def _gradient(i, n, base):
    """系列ごとの色(濃→淡)"""
    t = i / max(n - 1, 1)
    if base == "blue":
        return f"rgb({int(30 + 120 * t)},{int(80 + 100 * t)},{int(200 - 40 * t)})"
    return f"rgb({int(214 - 20 * t)},{int(60 + 120 * t)},{int(40 + 20 * t)})"


def tab_mapping():
    st.info("z 平面の格子が、w = f(z) でどう曲がるかを見る。"
            "f′(z) ≠ 0 の点では**格子の直交(角度)が保たれる**(等角写像)。")
    c1, c2 = st.columns(2)
    name = c1.selectbox("写像", list(MAPS), key="cm_map")
    grid = c2.radio("z 平面の図形", ["直交格子", "極座標格子"], horizontal=True, key="cm_grid")
    f = MAPS[name]["f"]
    st.latex(MAPS[name]["tex"])

    lines = []  # (z の配列, 系列, 番号, 本数)
    nl = 9
    s = np.linspace(0, 1, 500)
    if grid == "直交格子":
        d1, d2 = st.columns(2)
        xr = d1.slider("x の範囲", -3.0, 3.0, (0.2, 1.8) if "1/z" in name else (-1.5, 1.5), 0.1,
                       key=f"cm_xr_{name}")
        yr = d2.slider("y の範囲", -3.0, 3.0, (-1.0, 1.0) if "e^z" not in name else (-1.5, 1.5), 0.1,
                       key=f"cm_yr_{name}")
        for i, x in enumerate(np.linspace(*xr, nl)):
            lines.append((x + 1j * (yr[0] + (yr[1] - yr[0]) * s), "A", i, nl))
        for i, y in enumerate(np.linspace(*yr, nl)):
            lines.append((xr[0] + (xr[1] - xr[0]) * s + 1j * y, "B", i, nl))
        labA, labB = "x = 一定", "y = 一定"
    else:
        d1, d2 = st.columns(2)
        rr = d1.slider("r の範囲", 0.1, 3.0, (0.5, 1.5), 0.1, key="cm_rr")
        tr = d2.slider("θ の範囲(度)", 0, 360, (0, 360), 15, key="cm_tr")
        th = np.radians(np.linspace(*tr, 500))
        for i, r in enumerate(np.linspace(*rr, nl)):
            lines.append((r * np.exp(1j * th), "A", i, nl))
        ang = np.radians(np.linspace(tr[0], tr[1], nl if tr[1] - tr[0] < 360 else nl + 1))[:nl]
        for i, a in enumerate(ang):
            lines.append(((rr[0] + (rr[1] - rr[0]) * s) * np.exp(1j * a), "B", i, nl))
        labA, labB = "r = 一定(円)", "θ = 一定(半直線)"

    e1, e2 = st.columns(2)
    x0 = e1.slider("注目する点 z₀ の実部", -2.0, 2.0, 0.6, 0.05, key="cm_x0")
    y0 = e2.slider("注目する点 z₀ の虚部", -2.0, 2.0, 0.6, 0.05, key="cm_y0")
    z0 = np.complex128(x0 + 1j * y0)

    WCLIP = 8.0
    figz, figw = go.Figure(), go.Figure()
    for z, fam, i, n in lines:
        col = _gradient(i, n, "blue" if fam == "A" else "red")
        show_leg = i == 0
        lab = labA if fam == "A" else labB
        figz.add_trace(go.Scatter(x=z.real, y=z.imag, mode="lines", line=dict(color=col, width=2),
                                  name=lab, legendgroup=fam, showlegend=show_leg, hoverinfo="skip"))
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            w = f(z)
        w = np.where(np.abs(w) > WCLIP, np.nan, w)
        figw.add_trace(go.Scatter(x=w.real, y=w.imag, mode="lines", line=dict(color=col, width=2),
                                  name=lab, legendgroup=fam, showlegend=show_leg, hoverinfo="skip"))

    # 注目点の小さな十字:角度が保たれるか
    h = 0.12
    cross = [(z0 - h + 2 * h * s, C3), (z0 - 1j * h + 2j * h * s, C4)]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for zc, col in cross:
            figz.add_trace(go.Scatter(x=zc.real, y=zc.imag, mode="lines", line=dict(color=col, width=4),
                                      showlegend=False))
            wc = f(zc)
            wc = np.where(np.abs(wc) > WCLIP, np.nan, wc)
            figw.add_trace(go.Scatter(x=wc.real, y=wc.imag, mode="lines", line=dict(color=col, width=4),
                                      showlegend=False))
        w0 = f(z0)
        dh = 1e-6
        fp = (f(z0 + dh) - f(z0 - dh)) / (2 * dh)
    figz.add_trace(go.Scatter(x=[x0], y=[y0], mode="markers", marker=dict(color="black", size=8),
                              name="z₀"))
    if np.isfinite(w0) and abs(w0) <= WCLIP:
        figw.add_trace(go.Scatter(x=[np.real(w0)], y=[np.imag(w0)], mode="markers",
                                  marker=dict(color="black", size=8), name="f(z₀)"))

    if "ジューコフスキー" in name and st.checkbox("円を写して翼型をつくる", value=True, key="cm_wing"):
        g1, g2 = st.columns(2)
        a = g1.slider("円の中心のずれ(左へ)", 0.0, 0.3, 0.1, 0.01, key="cm_wa")
        b = g2.slider("円の中心のずれ(上へ)", 0.0, 0.3, 0.1, 0.01, key="cm_wb")
        c = -a + 1j * b
        R = abs(1 - c)
        zc = c + R * np.exp(1j * np.linspace(0, 2 * PI, 600))
        wc = f(zc)
        figz.add_trace(go.Scatter(x=zc.real, y=zc.imag, mode="lines", fill="toself",
                                  fillcolor="rgba(255,127,14,0.25)", line=dict(color=C4, width=3),
                                  name="z = 1 を通る円"))
        figw.add_trace(go.Scatter(x=wc.real, y=wc.imag, mode="lines", fill="toself",
                                  fillcolor="rgba(255,127,14,0.25)", line=dict(color=C4, width=3),
                                  name="翼型"))
        st.caption("z = 1 を通り、z = −1 を内側に含む円は、尖った後縁をもつ翼の形に写る"
                   "(z = 1 では f′(z) = 1 − 1/z² = 0 なので角度が保たれず、尖る)。")

    layout(figz, title="z 平面", height=480, equal=True, xlab="x = Re z", ylab="y = Im z")
    layout(figw, title="w 平面", height=480, equal=True, xlab="u = Re w", ylab="v = Im w")
    cA, cB = st.columns(2)
    with cA:
        show(figz)
    with cB:
        show(figw)

    if np.isfinite(fp) and np.isfinite(w0):
        st.write(f"z₀ = {fmt_c(z0, 2)} で f′(z₀) ≈ **{fmt_c(fp, 3)}** → 近くの図形は "
                 f"**{abs(fp):.3f} 倍**に拡大され、**{np.degrees(np.angle(fp)):.1f}°** 回転する"
                 "(緑とオレンジの十字が、直交したまま写っていることを確かめよう)。")
    if np.isfinite(fp) and abs(fp) < 0.05:
        st.warning("f′(z₀) ≈ 0 の点(臨界点)の近くでは角度が保たれません。"
                   "例えば w = z² では原点で角度が2倍になります。")

    with st.expander("練習問題"):
        st.markdown("**問1** $w=z^2$ で、直線 $x=1$ はどんな曲線に写るか。")
        st.latex(r"z=1+iy\ \Rightarrow\ w=(1-y^2)+2iy,\quad u=1-\frac{v^2}{4}\ (\text{左に開いた放物線})")
        st.markdown("**問2** $w=\\frac1z$ で、円 $|z-1|=1$(原点を通る)はどんな図形に写るか。")
        st.write("原点を通る円は直線に写る。$z=2$ → $w=\\frac12$、$z=1\\pm i$ → $w=\\frac{1\\mp i}{2}$ より、"
                 "**直線 $u=\\frac12$**。(極座標格子で試してみよう)")


# ---------- 定義域の色付け ----------
DC = {
    "z": dict(f=lambda z: z, tex=r"f(z)=z", zp=[(0, 1)]),
    "z²": dict(f=lambda z: z ** 2, tex=r"f(z)=z^2", zp=[(0, 2)]),
    "z³ − 1": dict(f=lambda z: z ** 3 - 1, tex=r"f(z)=z^3-1",
                   zp=[(np.exp(2j * PI * k / 3), 1) for k in range(3)]),
    "1/z": dict(f=lambda z: 1 / z, tex=r"f(z)=\frac1z", zp=[(0, -1)]),
    "(z² − 1)/(z² + 1)": dict(f=lambda z: (z ** 2 - 1) / (z ** 2 + 1), tex=r"f(z)=\frac{z^2-1}{z^2+1}",
                              zp=[(1, 1), (-1, 1), (1j, -1), (-1j, -1)]),
    "(z − 1)²/(z + 1)": dict(f=lambda z: (z - 1) ** 2 / (z + 1), tex=r"f(z)=\frac{(z-1)^2}{z+1}",
                            zp=[(1, 2), (-1, -1)]),
    "z + 1/z": dict(f=lambda z: z + 1 / z, tex=r"f(z)=z+\frac1z", zp=[(1j, 1), (-1j, 1), (0, -1)]),
    "e^z": dict(f=np.exp, tex=r"f(z)=e^z", zp=[]),
    "sin z": dict(f=np.sin, tex=r"f(z)=\sin z", zp=[(k * PI, 1) for k in range(-3, 4)]),
}


def color_image(f, R, n=480):
    xs = np.linspace(-R, R, n)
    X, Y = np.meshgrid(xs, xs)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        W = f(X + 1j * Y)
    hue = (np.angle(W) / (2 * PI)) % 1.0
    lm = np.log2(np.abs(W) + 1e-300)
    val = 0.62 + 0.38 * (lm - np.floor(lm))
    rgb = hsv_to_rgb(np.nan_to_num(hue), 0.9, np.nan_to_num(val, nan=1.0))
    bad = ~np.isfinite(W)
    rgb[bad] = 1.0
    return xs, (rgb * 255).astype(np.uint8)


def tab_domain_coloring():
    st.info("各点 z を、**f(z) の偏角を色**、**|f(z)| を明るさの縞**で塗る。"
            "零点のまわりでは色が 赤→黄→緑→青→紫 と反時計回りに回り、極のまわりでは逆回りになる。")
    c1, c2 = st.columns(2)
    name = c1.selectbox("関数", list(DC), index=4, key="dc_f")
    R = c2.slider("表示範囲 |Re z|, |Im z| ≤", 1.0, 5.0, 2.0, 0.5, key="dc_R")
    D = DC[name]
    st.latex(D["tex"])

    xs, img = color_image(D["f"], R)
    fig = px.imshow(img, x=xs, y=xs, origin="lower")

    d1, d2, d3 = st.columns(3)
    cx = d1.slider("円の中心 Re", -2.0, 2.0, 0.0, 0.1, key="dc_cx")
    cy = d2.slider("円の中心 Im", -2.0, 2.0, 0.0, 0.1, key="dc_cy")
    r = d3.slider("円の半径", 0.1, 3.0, 1.3, 0.05, key="dc_r")
    c = cx + 1j * cy
    th = np.linspace(0, 2 * PI, 6001)
    zc = c + r * np.exp(1j * th)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        wc = D["f"](zc)
    wind = (np.unwrap(np.angle(wc))[-1] - np.unwrap(np.angle(wc))[0]) / (2 * PI)
    fig.add_trace(go.Scatter(x=zc.real, y=zc.imag, mode="lines", line=dict(color="white", width=4),
                             name="円 C", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=zc.real, y=zc.imag, mode="lines", line=dict(color="black", width=1.5),
                             showlegend=False, hoverinfo="skip"))
    for p, m in D["zp"]:
        if abs(p.real) <= R and abs(p.imag) <= R:
            fig.add_trace(go.Scatter(x=[p.real], y=[p.imag], mode="markers+text",
                                     marker=dict(symbol="circle" if m > 0 else "x", size=9, color="black"),
                                     text=[f"零点({m}位)" if m > 0 else f"極({-m}位)"],
                                     textposition="top right", textfont=dict(color="black"),
                                     showlegend=False))
    layout(fig, height=560, xlab="Re z", ylab="Im z", legend=False)
    show(fig)

    inside = [(p, m) for p, m in D["zp"] if abs(p - c) < r]
    Z = sum(m for _, m in inside if m > 0)
    P = -sum(m for _, m in inside if m < 0)
    on = any(abs(abs(p - c) - r) < 0.03 for p, _ in D["zp"])
    m1, m2, m3 = st.columns(3)
    m1.metric("C の内側の零点の個数 Z", Z)
    m2.metric("C の内側の極の個数 P", P)
    m3.metric("C を一周したときの色の回転数", "—" if on else f"{round(wind, 3) + 0.0:.3f}")
    st.latex(r"\text{偏角の原理:}\quad \frac{1}{2\pi i}\oint_C\frac{f'(z)}{f(z)}dz"
             r"=\frac{1}{2\pi}\Delta_C\arg f(z)=Z-P")
    if on:
        st.warning("円が零点か極の上を通っています。半径を少し変えてください。")

    with st.expander("練習問題"):
        st.markdown("**問3** $f(z)=\\dfrac{(z-1)^2}{z+1}$ で、$|z|=2$ を一周すると色は何回回るか。")
        st.write("内側に 2 位の零点 $z=1$ と 1 位の極 $z=-1$ があるので、$Z-P=2-1=$ **1 回**。")
        st.markdown("**問4** $e^z$ の画像に、色が一周する点がないのはなぜか。")
        st.write("$e^z\\ne 0$ で極もないので、どんな閉曲線でも $Z-P=0$。"
                 "色(偏角 = Im z)は虚軸方向に周期 $2\\pi$ で縞になる。")


# ---------- 留数定理 ----------
FR = {
    "1/z": dict(f=lambda z: 1 / z, tex=r"f(z)=\frac1z", poles=[(0, 1, "1")]),
    "1/(z² + 1)": dict(f=lambda z: 1 / (z ** 2 + 1), tex=r"f(z)=\frac{1}{z^2+1}",
                       poles=[(1j, -0.5j, "1"), (-1j, 0.5j, "1")]),
    "1/(z(z − 2))": dict(f=lambda z: 1 / (z * (z - 2)), tex=r"f(z)=\frac{1}{z(z-2)}",
                         poles=[(0, -0.5, "1"), (2, 0.5, "1")]),
    "z/(z² − 1)": dict(f=lambda z: z / (z ** 2 - 1), tex=r"f(z)=\frac{z}{z^2-1}",
                       poles=[(1, 0.5, "1"), (-1, 0.5, "1")]),
    "e^z / z²": dict(f=lambda z: np.exp(z) / z ** 2, tex=r"f(z)=\frac{e^z}{z^2}", poles=[(0, 1, "2")]),
    "sin z / z⁴": dict(f=lambda z: np.sin(z) / z ** 4, tex=r"f(z)=\frac{\sin z}{z^4}",
                       poles=[(0, -1 / 6, "3")]),
    "1/(z − 1)²": dict(f=lambda z: 1 / (z - 1) ** 2, tex=r"f(z)=\frac{1}{(z-1)^2}", poles=[(1, 0, "2")]),
    "e^{1/z}(真性特異点)": dict(f=lambda z: np.exp(1 / z), tex=r"f(z)=e^{1/z}=1+\frac1z+\frac{1}{2!z^2}+\cdots",
                            poles=[(0, 1, "真性")]),
    "z²(正則)": dict(f=lambda z: z ** 2, tex=r"f(z)=z^2", poles=[]),
}


def tab_residue():
    st.info("閉曲線 C に沿った積分は、C の内側にある特異点の**留数の和だけ**で決まる。")
    st.latex(r"\oint_C f(z)\,dz=2\pi i\sum_{C\ \text{の内側}}\mathrm{Res}(f,\,a_k)")
    name = st.selectbox("関数", list(FR), index=1, key="cr_f")
    F = FR[name]
    st.latex(F["tex"])
    c1, c2, c3 = st.columns(3)
    cx = c1.slider("円 C の中心 Re", -3.0, 3.0, 0.0, 0.1, key="cr_cx")
    cy = c2.slider("円 C の中心 Im", -3.0, 3.0, 0.5, 0.1, key="cr_cy")
    R = c3.slider("円 C の半径", 0.1, 4.0, 1.0, 0.05, key="cr_R")
    c = cx + 1j * cy

    n = 8192
    th = np.arange(n) * 2 * PI / n
    z = c + R * np.exp(1j * th)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        integral = np.sum(F["f"](z) * 1j * R * np.exp(1j * th)) * 2 * PI / n
    rows, total = [], 0
    on = False
    for p, res, order in F["poles"]:
        d = abs(p - c)
        ins = d < R
        on |= abs(d - R) < 0.05
        if ins:
            total += res
        rows.append({"特異点 a": fmt_c(p, 3), "位数": order, "留数 Res": fmt_c(res, 4),
                     "C の内側?": "○" if ins else "×"})

    fig = go.Figure()
    tt = np.linspace(0, 2 * PI, 400)
    fig.add_trace(go.Scatter(x=(c + R * np.exp(1j * tt)).real, y=(c + R * np.exp(1j * tt)).imag,
                             mode="lines", fill="toself", fillcolor="rgba(31,119,180,0.12)",
                             line=dict(color=C1, width=3), name="C(反時計回り)"))
    for p, res, order in F["poles"]:
        ins = abs(p - c) < R
        fig.add_trace(go.Scatter(x=[np.real(p)], y=[np.imag(p)], mode="markers+text",
                                 marker=dict(symbol="x", size=14, color=C2 if ins else GRAY,
                                             line=dict(width=3)),
                                 text=[f"Res = {fmt_c(res, 3)}"], textposition="top right",
                                 showlegend=False))
    layout(fig, height=480, equal=True, xlab="Re z", ylab="Im z", xrange=(-4, 4), yrange=(-4, 4),
           legend=False)
    cA, cB = st.columns([3, 2])
    with cA:
        show(fig)
    with cB:
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch")
        else:
            st.write("特異点はありません(z 平面全体で正則)。")
        st.markdown("**数値積分**")
        st.latex(r"\oint_C f(z)\,dz\approx " + fmt_c(integral, 5).replace("−", "-"))
        st.markdown("**留数定理**")
        st.latex(r"2\pi i\sum\mathrm{Res}=" + fmt_c(2j * PI * total, 5).replace("−", "-"))
        if on:
            st.warning("C が特異点のすぐ近くを通っています。積分が定義されない/数値誤差が大きくなります。")
        elif abs(integral - 2j * PI * total) < 1e-6 * max(1, abs(integral)):
            st.success("一致しました。")

    with st.expander("留数の求め方"):
        st.latex(r"\text{1位の極:}\ \mathrm{Res}(f,a)=\lim_{z\to a}(z-a)f(z),\qquad "
                 r"f=\frac{g}{h}\ \text{なら}\ \frac{g(a)}{h'(a)}")
        st.latex(r"m\text{位の極:}\ \mathrm{Res}(f,a)=\frac{1}{(m-1)!}\lim_{z\to a}\frac{d^{m-1}}{dz^{m-1}}"
                 r"\left\{(z-a)^mf(z)\right\}")
        st.write("一般には、ローラン展開 $f(z)=\\sum c_n(z-a)^n$ の $c_{-1}$ が留数。"
                 "例:$\\frac{\\sin z}{z^4}=\\frac{1}{z^3}-\\frac{1}{6z}+\\cdots$ より $-\\frac16$、"
                 "$e^{1/z}$ は $\\frac1z$ の係数 1。")
    with st.expander("練習問題"):
        st.markdown("**問5** $\\displaystyle\\oint_{|z|=2}\\frac{z}{z^2-1}dz$ を求めよ。")
        st.latex(r"\mathrm{Res}(1)=\frac{1}{2},\ \mathrm{Res}(-1)=\frac{-1}{-2}=\frac12\ \Rightarrow\ "
                 r"2\pi i\left(\frac12+\frac12\right)=2\pi i")
        st.markdown("**問6** $\\displaystyle\\oint_{|z|=1}\\frac{e^z}{z^2}dz$ を求めよ。")
        st.latex(r"\frac{e^z}{z^2}=\frac{1}{z^2}+\frac1z+\frac12+\cdots\ \Rightarrow\ \mathrm{Res}=1,"
                 r"\ \ \oint=2\pi i")


# ---------- 実積分 ----------
RI = {
    "1/(x² + 1)": dict(f=lambda z: 1 / (z ** 2 + 1), poles=[(1j, 1 / 2j)], exact=PI,
                       tex=r"\int_{-\infty}^{\infty}\frac{dx}{x^2+1}", ans=r"\pi"),
    "1/(x² + 1)²": dict(f=lambda z: 1 / (z ** 2 + 1) ** 2, poles=[(1j, 1 / 4j)], exact=PI / 2,
                        tex=r"\int_{-\infty}^{\infty}\frac{dx}{(x^2+1)^2}", ans=r"\frac{\pi}{2}"),
    "1/(x⁴ + 1)": dict(f=lambda z: 1 / (z ** 4 + 1),
                       poles=[(np.exp(1j * PI / 4), -np.exp(1j * PI / 4) / 4),
                              (np.exp(3j * PI / 4), -np.exp(3j * PI / 4) / 4)],
                       exact=PI / np.sqrt(2), tex=r"\int_{-\infty}^{\infty}\frac{dx}{x^4+1}",
                       ans=r"\frac{\pi}{\sqrt2}"),
    "cos x/(x² + 1)": dict(f=lambda z: np.exp(1j * z) / (z ** 2 + 1), poles=[(1j, np.exp(-1) / 2j)],
                           exact=PI / np.e, tex=r"\int_{-\infty}^{\infty}\frac{\cos x}{x^2+1}dx"
                           r"=\mathrm{Re}\int_{-\infty}^{\infty}\frac{e^{ix}}{x^2+1}dx",
                           ans=r"\frac{\pi}{e}"),
}


def _pieces(F, R):
    x = np.linspace(-R, R, 40001)
    line = trapz(F["f"](x + 0j), x)
    th = np.linspace(0, PI, 20001)
    z = R * np.exp(1j * th)
    arc = trapz(F["f"](z) * 1j * z, th)
    return line, arc


def tab_real_integral():
    st.info("実軸上の積分を、上半平面の半円をつけた閉曲線の積分に変え、留数定理で求める。"
            "半径 R → ∞ で半円部分の積分が 0 になるのがポイント。")
    name = st.selectbox("求める積分", list(RI), key="ri_f")
    F = RI[name]
    R = st.slider("半円の半径 R", 1.1, 30.0, 3.0, 0.1, key="ri_R")
    total = 2j * PI * sum(r for _, r in F["poles"])
    st.latex(F["tex"] + "=" + F["ans"] + rf"\approx {F['exact']:.6f}")

    line, arc = _pieces(F, R)
    fig = go.Figure()
    tt = np.linspace(0, PI, 300)
    fig.add_trace(go.Scatter(x=np.r_[-R, R, R * np.cos(tt)], y=np.r_[0, 0, R * np.sin(tt)],
                             mode="lines", fill="toself", fillcolor="rgba(31,119,180,0.10)",
                             line=dict(color=C1, width=3), name="閉曲線"))
    for p, r in F["poles"]:
        fig.add_trace(go.Scatter(x=[p.real], y=[p.imag], mode="markers+text",
                                 marker=dict(symbol="x", size=13, color=C2, line=dict(width=3)),
                                 text=[f"Res = {fmt_c(r, 3)}"], textposition="top right", showlegend=False))
    lim = max(R * 1.1, 2)
    layout(fig, height=360, equal=True, xlab="Re z", ylab="Im z", xrange=(-lim, lim),
           yrange=(-0.2 * lim, lim), legend=False)

    cA, cB = st.columns([3, 2])
    with cA:
        show(fig)
    with cB:
        st.latex(rf"\int_{{-R}}^{{R}}f(x)\,dx={np.real(line):.6f}")
        st.latex(r"\int_{C_R}f(z)\,dz=" + fmt_c(arc, 6).replace("−", "-"))
        st.latex(r"\text{合計}=" + fmt_c(line + arc, 6).replace("−", "-"))
        st.latex(r"2\pi i\sum\mathrm{Res}=" + fmt_c(total, 6).replace("−", "-"))
        st.caption("どの R でも「合計」は留数定理の値に一致。R を大きくすると半円の寄与が消え、"
                   "実軸上の積分だけが残る。")

    Rs = np.linspace(1.2, 30, 60)
    arcs = [abs(_pieces(F, r)[1]) for r in Rs]
    fig = go.Figure(go.Scatter(x=Rs, y=arcs, mode="lines", line=dict(color=C2), name="|半円の積分|"))
    fig.add_trace(go.Scatter(x=[R], y=[abs(arc)], mode="markers", marker=dict(size=10, color=C2),
                             name="いまの R"))
    layout(fig, title="半円部分の積分の大きさ → 0", xlab="R", height=300, yrange=(0, max(arcs) * 1.1))
    show(fig)

    with st.expander("練習問題"):
        st.markdown("**問7** $\\displaystyle\\int_{-\\infty}^{\\infty}\\frac{dx}{x^2+1}=\\pi$ を留数定理で示せ。")
        st.latex(r"\mathrm{Res}\left(\frac{1}{z^2+1},\,i\right)=\frac{1}{2i},\quad 2\pi i\cdot\frac{1}{2i}=\pi")
        st.write("半円上では $\\left|\\int_{C_R}\\right|\\le \\pi R\\cdot\\frac{1}{R^2-1}\\to 0$。")
