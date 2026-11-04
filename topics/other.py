"""⑥ その他:デルタ関数のモデル関数/フーリエ変換とフーリエ級数の関係
(サンプリング定理は ③ フーリエ解析 第3章 5節(topics/dft.py)へ移した)"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import C1, C2, C3, C4, GRAY, layout, show, trapz

PI = np.pi

DELTA = {
    "矩形": dict(d=lambda t, e: (np.abs(t) < e / 2) / e,
                F=lambda w, e: np.where(np.abs(w) < 1e-12, 1.0, np.sin(w * e / 2) / np.where(w == 0, 1, w * e / 2)),
                tex=r"\delta_\varepsilon(t)=\begin{cases}\frac1\varepsilon&(|t|<\frac\varepsilon2)\\0&(\text{その他})\end{cases}",
                Ftex=r"\frac{\sin(\omega\varepsilon/2)}{\omega\varepsilon/2}"),
    "ガウス": dict(d=lambda t, e: np.exp(-t ** 2 / (2 * e * e)) / (e * np.sqrt(2 * PI)),
                F=lambda w, e: np.exp(-(e * w) ** 2 / 2),
                tex=r"\delta_\varepsilon(t)=\frac{1}{\varepsilon\sqrt{2\pi}}e^{-t^2/(2\varepsilon^2)}",
                Ftex=r"e^{-\varepsilon^2\omega^2/2}"),
    "ローレンツ(コーシー)": dict(d=lambda t, e: e / (PI * (t * t + e * e)),
                          F=lambda w, e: np.exp(-e * np.abs(w)),
                          tex=r"\delta_\varepsilon(t)=\frac{1}{\pi}\frac{\varepsilon}{t^2+\varepsilon^2}",
                          Ftex=r"e^{-\varepsilon|\omega|}"),
    "sinc(ディリクレ核型)": dict(d=lambda t, e: np.where(np.abs(t) < 1e-12, 1 / (PI * e),
                                                    np.sin(t / e) / (PI * np.where(t == 0, 1, t))),
                             F=lambda w, e: (np.abs(w) < 1 / e).astype(float),
                             tex=r"\delta_\varepsilon(t)=\frac{\sin(t/\varepsilon)}{\pi t}",
                             Ftex=r"\begin{cases}1&(|\omega|<1/\varepsilon)\\0&(|\omega|>1/\varepsilon)\end{cases}"),
}
TESTF = {
    "g(t) = cos t": (np.cos, 1.0),
    "g(t) = e^{−(t−0.5)²}": (lambda t: np.exp(-(t - 0.5) ** 2), np.exp(-0.25)),
    "g(t) = 1/(1 + t²) + t": (lambda t: 1 / (1 + t ** 2) + t, 1.0),
}


def tab_delta():
    st.info("デルタ関数 δ(t) は「t = 0 にだけ集中した、面積 1 のパルス」。普通の関数ではないが、"
            "**面積 1 のまま幅を細くしていく関数の列**(モデル関数)の極限として扱える。")
    st.latex(r"\int_{-\infty}^{\infty}\delta_\varepsilon(t)\,dt=1,\qquad "
             r"\lim_{\varepsilon\to0}\int_{-\infty}^{\infty}\delta_\varepsilon(t)\,g(t)\,dt=g(0),\qquad "
             r"\mathcal F[\delta_\varepsilon](\omega)\to1")
    c1, c2, c3 = st.columns(3)
    name = c1.selectbox("モデル関数", list(DELTA), key="dl_m")
    gname = c2.selectbox("試験関数 g", list(TESTF), key="dl_g")
    eps = c3.slider("ε", 0.02, 1.0, 0.3, 0.01, key="dl_e")
    D = DELTA[name]
    g, g0 = TESTF[gname]
    st.latex(D["tex"] + r",\qquad \mathcal F[\delta_\varepsilon]=" + D["Ftex"])

    t = np.linspace(-60, 60, 600001)
    dv = D["d"](t, eps)
    area = float(trapz(dv, t))
    pair = float(trapz(dv * g(t), t))
    m1, m2, m3 = st.columns(3)
    m1.metric("面積 ∫δ_ε dt", f"{area:.5f}")
    m2.metric("∫δ_ε g dt", f"{pair:.5f}")
    m3.metric("g(0)", f"{g0:.5f}")

    cA, cB = st.columns(2)
    with cA:
        tp = np.linspace(-3, 3, 3001)
        fig = go.Figure()
        for e, op in ((1.0, 0.25), (0.5, 0.45)):
            fig.add_trace(go.Scatter(x=tp, y=D["d"](tp, e), mode="lines", line=dict(color=GRAY, width=1),
                                     opacity=op + 0.3, name=f"ε = {e:g}"))
        fig.add_trace(go.Scatter(x=tp, y=D["d"](tp, eps), mode="lines", line=dict(color=C1, width=2.5),
                                 name=f"ε = {eps:g}"))
        layout(fig, title="δ_ε(t)", xlab="t", height=380, yrange=(-0.4 / eps, 1.2 / eps))
        show(fig)
    with cB:
        w = np.linspace(-40, 40, 2001)
        fig = go.Figure(go.Scatter(x=w, y=D["F"](w, eps), mode="lines", line=dict(color=C2, width=2.5),
                                   name="F[δ_ε](ω)"))
        fig.add_hline(y=1, line=dict(color=GRAY, dash="dash"), annotation_text="F[δ] = 1")
        layout(fig, title="フーリエ変換(ε → 0 ですべての ω で 1 に)", xlab="ω", height=380, yrange=(-0.3, 1.2))
        show(fig)
    st.caption("ε を小さくすると、δ_ε は高く細くなり、∫δ_ε g dt は g(0) に近づく。"
               "δ はすべての周波数を同じ強さで含む(白色)。ガウス型は「⑤ 偏微分方程式」の熱核、"
               "ローレンツ型は上半平面のポアソン核と同じ形。")
    with st.expander("練習問題"):
        st.markdown("**問1** $\\displaystyle\\int_{-\\infty}^{\\infty}\\delta(t-2)(t^2+1)\\,dt$、"
                    "$\\mathcal L[\\delta(t-3)]$ を求めよ。")
        st.markdown("**問2** 単位ステップ関数 U(t) の導関数が δ(t) と考えられる理由を、モデル関数で説明せよ。")
        if st.toggle("解答・解説を表示", key="pr_other_80"):
            st.markdown("**答1**")
            st.latex(r"2^2+1=5,\qquad e^{-3s}")
            st.markdown("**答2**")
            st.write("矩形モデル δ_ε の積分 $\\int_{-\\infty}^t\\delta_\\varepsilon$ は、幅 ε で 0 から 1 へ直線的に上がる関数。"
                     "ε → 0 で U(t) になる。")


# ============================================================
def tab_series_vs_transform():
    st.info("周期 T のパルス列のフーリエ係数は、1個のパルスのフーリエ変換の値を飛び飛びに拾ったもの。"
            "T → ∞(周期がなくなる)で、線スペクトルの間隔が詰まり、級数が積分(フーリエ変換)になる。")
    st.latex(r"c_n=\frac1T\int_{-T/2}^{T/2}f(t)e^{-in\omega_0t}dt=\frac1TF(n\omega_0),\quad \omega_0=\frac{2\pi}{T}")
    st.latex(r"f(t)=\sum_n c_ne^{in\omega_0t}=\frac{1}{2\pi}\sum_nF(n\omega_0)e^{in\omega_0t}\,\omega_0\ "
             r"\xrightarrow{T\to\infty}\ \frac{1}{2\pi}\int_{-\infty}^{\infty}F(\omega)e^{i\omega t}d\omega")
    T = st.slider("周期 T", 1.5, 30.0, 4.0, 0.5, key="sv_T")
    w0 = 2 * PI / T
    F = lambda w: np.where(np.abs(w) < 1e-12, 1.0, 2 * np.sin(w / 2) / np.where(w == 0, 1, w))  # noqa: E731
    t = np.linspace(-15, 15, 3001)
    train = (np.abs(((t + T / 2) % T) - T / 2) < 0.5).astype(float)
    fig = go.Figure(go.Scatter(x=t, y=train, mode="lines", line=dict(color=C1), name="パルス列(幅 1)"))
    layout(fig, title=f"周期 T = {T:g} のパルス列", xlab="t", height=240, legend=False, yrange=(-0.2, 1.3))
    show(fig)
    nmax = int(30 / w0)
    n = np.arange(-nmax, nmax + 1)
    wn = n * w0
    w = np.linspace(-30, 30, 2001)
    fig = go.Figure(go.Scatter(x=w, y=F(w), mode="lines", line=dict(color=GRAY, dash="dot"), name="F(ω)(1個のパルス)"))
    xs, ys = [], []
    for x_, y_ in zip(wn, T * (F(wn) / T)):
        xs += [x_, x_, None]
        ys += [0, y_, None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=C2, width=2), name="T·c_n(線スペクトル)"))
    fig.add_trace(go.Scatter(x=wn, y=F(wn), mode="markers", marker=dict(color=C2, size=5), showlegend=False))
    layout(fig, title=f"線スペクトルの間隔 ω₀ = 2π/T = {w0:.3f}", xlab="ω", height=360)
    show(fig)
    st.caption("T を大きくすると、線(赤)が F(ω)(灰色)の上にびっしり並んでいく。")
