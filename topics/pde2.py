"""⑤ 偏微分方程式:フーリエ変換と偏微分方程式(無限に長い棒・弦)"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import C1, C2, C3, GRAY, animate, layout, show, trapz

PI = np.pi
INIT = {
    "矩形(|x| < 1 で 1)": lambda x: (np.abs(x) < 1).astype(float),
    "ガウス関数 e^{−x²}": lambda x: np.exp(-x ** 2),
    "2つの山": lambda x: np.exp(-4 * (x + 2) ** 2) + 0.7 * np.exp(-4 * (x - 2) ** 2),
}


def tab_fourier_pde():
    st.info("区間が無限に長いときは、フーリエ級数の代わりにフーリエ変換を使う。"
            "x についてフーリエ変換すると、偏微分方程式が ω ごとの常微分方程式になる。")
    eq = st.radio("方程式", ["熱伝導方程式 u_t = κ u_xx", "波動方程式 u_tt = c² u_xx"], horizontal=True, key="fp2_eq")
    c1, c2 = st.columns(2)
    name = c1.selectbox("初期条件 f(x)", list(INIT), key="fp2_f")
    par = c2.slider("κ" if eq.startswith("熱") else "c", 0.1, 2.0, 1.0, 0.1, key="fp2_p")
    f = INIT[name]
    L, N = 40.0, 4096
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    w = 2 * PI * np.fft.fftfreq(N, d=L / N)
    fh = np.fft.fft(f(x))
    view = np.abs(x) <= 8

    if eq.startswith("熱"):
        st.latex(r"\hat u(\omega,t)=\hat f(\omega)\,e^{-\kappa\omega^2t}\ \Longrightarrow\ "
                 r"u(x,t)=\frac{1}{2\pi}\int\hat f(\omega)e^{-\kappa\omega^2t}e^{i\omega x}d\omega"
                 r"=\int_{-\infty}^{\infty}f(y)\,\frac{1}{\sqrt{4\pi\kappa t}}e^{-\frac{(x-y)^2}{4\kappa t}}dy")
        st.caption("後ろの式は、初期温度と熱核(ガウス関数=デルタ関数のモデル関数)との合成積。")
        times = np.linspace(0, 3, 61) ** 2 / 3
        uh = lambda t: fh * np.exp(-par * w ** 2 * t)  # noqa: E731
    else:
        st.latex(r"\hat u(\omega,t)=\hat f(\omega)\cos(c\omega t)\ \Longrightarrow\ "
                 r"u(x,t)=\frac12\left\{f(x-ct)+f(x+ct)\right\}\quad(\text{初速度 0})")
        st.caption("cos(cωt) = (e^{icωt} + e^{−icωt})/2 と時間推移の性質から、右と左へ進む波に分かれる。")
        times = np.linspace(0, 6 / par, 61)
        uh = lambda t: fh * np.cos(par * w * t)  # noqa: E731

    frames = [[np.real(np.fft.ifft(uh(t)))[view]] for t in times]
    fig = animate(x[view], frames, ["u(x, t)"], [f"{t:.3f}" for t in times], colors=[C2],
                  static=[go.Scatter(x=x[view], y=f(x[view]), mode="lines", line=dict(color=GRAY, dash="dot"),
                                     name="f(x)")],
                  xlab="x", ylab="u", yrange=(-0.2, 1.2), title="▶ で再生")
    show(fig)

    T = times[-1]
    ws = np.fft.fftshift(w)
    sel = np.abs(ws) < 12
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ws[sel], y=np.abs(np.fft.fftshift(fh))[sel] * L / N, mode="lines",
                             line=dict(color=GRAY, dash="dot"), name="|f̂(ω)|"))
    fig.add_trace(go.Scatter(x=ws[sel], y=np.abs(np.fft.fftshift(uh(T * 0.1)))[sel] * L / N, mode="lines",
                             line=dict(color=C1), name=f"|û(ω, t)|(t = {T * 0.1:.3f})"))
    layout(fig, title="周波数ごとの振る舞い", xlab="ω", height=320)
    show(fig)
    if eq.startswith("熱"):
        st.caption("高い周波数ほど e^{−κω²t} で速く消える:熱方程式はなめらかにする(ぼかす)働きをもつ。")
        if name.startswith("ガウス"):
            t1 = times[-1]
            num = np.real(np.fft.ifft(uh(t1)))
            ex = np.exp(-x ** 2 / (1 + 4 * par * t1)) / np.sqrt(1 + 4 * par * t1)
            st.write(f"確認:ガウス型の初期値の厳密解 $u=\\frac{{1}}{{\\sqrt{{1+4\\kappa t}}}}e^{{-x^2/(1+4\\kappa t)}}$ と、"
                     f"t = {t1:.2f} での最大誤差 {np.max(np.abs(num - ex)):.1e}")
    else:
        st.caption("すべての周波数で振幅 |f̂| が保たれる:波は形を変えずに伝わる。")
    with st.expander("練習問題"):
        st.markdown("**問1** 熱方程式 $u_t=\\kappa u_{xx}$ を x についてフーリエ変換すると、$\\hat u$ の満たす方程式は?")
        if st.toggle("解答・解説を表示", key="pr_pde2_71"):
            st.markdown("**答1**")
            st.latex(r"\frac{\partial\hat u}{\partial t}=\kappa(i\omega)^2\hat u=-\kappa\omega^2\hat u\ "
                     r"\Rightarrow\ \hat u=\hat f(\omega)e^{-\kappa\omega^2t}")
