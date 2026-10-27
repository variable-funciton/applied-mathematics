"""③ フーリエ解析:複素フーリエ級数/積分定理/性質と公式/スペクトル/DFT と複素フーリエ係数"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import C1, C2, C3, C4, GRAY, fmt_c, layout, show, trapz
from topics.fourier import WAVES, coefficients, wrap

PI = np.pi


def complex_coeffs(name, d=1.0):
    a, b, _ = coefficients(name, d)
    c = (a - 1j * b) / 2
    c[0] = a[0] / 2
    return c  # c[n] (n ≥ 0)、c[−n] = conj(c[n])(f が実数値)


def c_of(c, n):
    return c[n] if n >= 0 else np.conj(c[-n])


# ============================================================
# 1-2. 複素フーリエ級数
# ============================================================
def tab_complex_series():
    st.info("$\\cos nx,\\ \\sin nx$ をまとめて $e^{inx}$ で表す。係数 $c_n$ は1つの複素数で、"
            "**大きさ = 振幅、偏角 = 位相**。負の n も使う(両側スペクトル)。")
    st.latex(r"f(x)\sim\sum_{n=-\infty}^{\infty}c_n e^{in\pi x/L},\quad c_n=\frac{1}{2L}\int_{-L}^{L}f(x)e^{-in\pi x/L}dx,"
             r"\quad c_{\pm n}=\frac{a_n\mp ib_n}{2},\ c_0=\frac{a_0}{2}")
    c1, c2, c3 = st.columns(3)
    name = c1.selectbox("波形(周期 2π)", list(WAVES), key="fc_w")
    N = c2.slider("N(−N 〜 N の和)", 1, 30, 4, key="fc_N")
    x0 = c3.slider("フェーザーを見る点 x", -3.14, 3.14, 1.0, 0.01, key="fc_x")
    c = complex_coeffs(name)
    W = WAVES[name]
    ns = np.arange(-N, N + 1)
    cn = np.array([c_of(c, n) for n in ns])

    cA, cB = st.columns(2)
    with cA:
        fig = go.Figure(go.Bar(x=ns, y=np.abs(cn), marker_color=C1, name="|c_n|"))
        layout(fig, title="両側振幅スペクトル |c_n|", xlab="n", height=300, legend=False)
        show(fig)
        ph = np.where(np.abs(cn) > 1e-8, np.angle(cn), np.nan)
        fig = go.Figure(go.Scatter(x=ns, y=ph, mode="markers", marker=dict(color=C2, size=9), name="arg c_n"))
        fig.update_yaxes(tickvals=[-PI, -PI / 2, 0, PI / 2, PI], ticktext=["−π", "−π/2", "0", "π/2", "π"])
        layout(fig, title="位相スペクトル arg c_n", xlab="n", height=280, legend=False, yrange=(-3.5, 3.5))
        show(fig)
        st.caption("f が実数値なので $c_{-n}=\\overline{c_n}$:振幅は左右対称、位相は反対称。")
    with cB:
        order = [0] + [k for m in range(1, N + 1) for k in (m, -m)]
        z = 0j
        xs, ys = [0.0], [0.0]
        for n in order:
            z = z + c_of(c, n) * np.exp(1j * n * x0)
            xs.append(z.real)
            ys.append(z.imag)
        fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=C1, width=2),
                                   marker=dict(size=4), name="c_n e^{inx} をつなぐ"))
        fig.add_trace(go.Scatter(x=[z.real], y=[z.imag], mode="markers", marker=dict(color=C2, size=12),
                                 name=f"和 S_N(x) = {z.real:.4f}"))
        fig.add_trace(go.Scatter(x=[float(W['f'](np.array([x0]), 1.0)[0])], y=[0], mode="markers",
                                 marker=dict(color="black", symbol="x", size=11), name="f(x)"))
        m = max(1.2, np.abs(np.array(xs + ys)).max() * 1.2)
        layout(fig, title=f"フェーザー図(x = {x0:.2f})", xlab="実部", ylab="虚部", height=600, equal=True,
               xrange=(-m, m), yrange=(-m, m))
        show(fig)
        st.caption("回転する矢印 $c_n e^{inx}$ を順につなぐと、$c_n$ と $c_{-n}$ の虚部が打ち消し合い、"
                   "先端は**実軸上**の $S_N(x)$ に着く。")

    x = np.linspace(-3 * PI, 3 * PI, 2001)
    S = np.real(sum(c_of(c, n) * np.exp(1j * n * x) for n in ns))
    fig = go.Figure(go.Scatter(x=x, y=W["f"](x, 1.0), mode="lines", line=dict(color=GRAY), name="f(x)"))
    fig.add_trace(go.Scatter(x=x, y=S, mode="lines", line=dict(color=C1, width=2.5), name="Σ_{−N}^{N} c_n e^{inx}"))
    fig.add_vline(x=x0, line=dict(color=C2, dash="dot"))
    layout(fig, xlab="x", height=320)
    show(fig)
    rows = [{"n": int(n), "c_n": fmt_c(c_of(c, n), 5), "|c_n|": f"{abs(c_of(c, n)):.5f}"} for n in range(0, min(N, 8) + 1)]
    with st.expander("係数の表(n ≥ 0)"):
        st.dataframe(rows, hide_index=True, width="stretch")

    with st.expander("練習問題"):
        st.markdown("**問1** $f(x)=x\\ (-\\pi<x<\\pi)$ の複素フーリエ係数を求めよ。")
        st.markdown("**問2** 周期 2($L=1$)、$f(x)=x\\ (-1<x<1)$ の通常のフーリエ係数を求めよ。")
        if st.toggle("解答・解説を表示", key="pr_fourier2_83"):
            st.markdown("**答1**")
            st.latex(r"c_n=\frac{1}{2\pi}\int_{-\pi}^{\pi}xe^{-inx}dx=\frac{(-1)^n i}{n}\ (n\ne0),\quad c_0=0")
            st.write("確認:$b_n=2(-1)^{n+1}/n$ より $c_n=-ib_n/2=(-1)^n i/n$。")
            st.markdown("**答2**")
            st.latex(r"b_n=\int_{-1}^{1}x\sin n\pi x\,dx=\frac{2(-1)^{n+1}}{n\pi}")


# ============================================================
# 2-1. フーリエ変換と積分定理
# ============================================================
FTP = {
    "矩形パルス(|t| < 1)": dict(f=lambda t: (np.abs(t) < 1) + 0.5 * (np.abs(t) == 1),
                              F=lambda w: np.where(np.abs(w) < 1e-12, 2.0, 2 * np.sin(w) / np.where(w == 0, 1, w)),
                              Ftex=r"F(\omega)=\frac{2\sin\omega}{\omega}", jump=1.0),
    "片側指数 e^{−t}U(t)": dict(f=lambda t: np.where(t > 0, np.exp(-np.abs(t)), np.where(t == 0, 0.5, 0.0)),
                              F=lambda w: 1 / (1 + 1j * w), Ftex=r"F(\omega)=\frac{1}{1+i\omega}", jump=0.0),
    "両側指数 e^{−|t|}": dict(f=lambda t: np.exp(-np.abs(t)), F=lambda w: 2 / (1 + w ** 2),
                           Ftex=r"F(\omega)=\frac{2}{1+\omega^2}", jump=None),
    "三角パルス (1 − |t|)₊": dict(f=lambda t: np.maximum(1 - np.abs(t), 0),
                              F=lambda w: np.where(np.abs(w) < 1e-6, 1.0, 2 * (1 - np.cos(w)) / np.where(w == 0, 1, w) ** 2),
                              Ftex=r"F(\omega)=\frac{2(1-\cos\omega)}{\omega^2}", jump=None),
}


def tab_integral_theorem():
    st.info("フーリエ変換して、逆変換で戻す。積分範囲を $|\\omega|\\le\\Omega$ に制限した $f_\\Omega$ は、"
            "Ω → ∞ で f に近づく(不連続点では左右の平均に)。級数の部分和に当たるもの。")
    st.latex(r"F(\omega)=\int_{-\infty}^{\infty}f(t)e^{-i\omega t}dt,\qquad "
             r"f_\Omega(t)=\frac{1}{2\pi}\int_{-\Omega}^{\Omega}F(\omega)e^{i\omega t}d\omega\ \xrightarrow{\Omega\to\infty}\ "
             r"\frac{f(t+0)+f(t-0)}{2}")
    c1, c2 = st.columns(2)
    name = c1.selectbox("関数", list(FTP), key="fi_f")
    Om = c2.slider("Ω(使う周波数の上限)", 1.0, 80.0, 8.0, 0.5, key="fi_O")
    P = FTP[name]
    st.latex(P["Ftex"])
    t = np.linspace(-4, 4, 481)
    w = np.linspace(-Om, Om, max(2001, int(Om * 60)))
    Fw = P["F"](w).astype(complex)
    fO = np.real(trapz(Fw[None, :] * np.exp(1j * np.outer(t, w)), w, axis=1)) / (2 * PI)
    cA, cB = st.columns([3, 2])
    with cA:
        fig = go.Figure(go.Scatter(x=t, y=P["f"](t), mode="lines", line=dict(color=GRAY, width=2), name="f(t)"))
        fig.add_trace(go.Scatter(x=t, y=fO, mode="lines", line=dict(color=C1, width=2.5), name="f_Ω(t)"))
        layout(fig, xlab="t", height=380, yrange=(-0.3, 1.35))
        show(fig)
    with cB:
        wp = np.linspace(-max(Om, 20), max(Om, 20), 1601)
        fig = go.Figure(go.Scatter(x=wp, y=np.abs(P["F"](wp)), mode="lines", line=dict(color=C2), name="|F(ω)|"))
        fig.add_vrect(x0=-Om, x1=Om, fillcolor="rgba(31,119,180,0.12)", line_width=0)
        layout(fig, title="使う周波数の範囲(青)", xlab="ω", height=380, legend=False)
        show(fig)
    if P["jump"] is not None:
        tj = P["jump"]
        val = np.real(trapz(Fw * np.exp(1j * w * tj), w)) / (2 * PI)
        st.write(f"不連続点 t = {tj:g} での値:f_Ω({tj:g}) = **{val:.4f}**(→ 0.5 = 左右の平均)")

    with st.expander("フーリエ級数との対応"):
        st.markdown(r"""
| | フーリエ級数 | フーリエ変換 |
|---|---|---|
| 対象 | 周期関数 | 周期のない関数 |
| 係数/変換 | $c_n=\frac{1}{2L}\int f e^{-in\pi x/L}dx$ | $F(\omega)=\int f e^{-i\omega t}dt$ |
| 復元 | $\sum c_n e^{in\pi x/L}$ | $\frac{1}{2\pi}\int F e^{i\omega t}d\omega$ |
| 近似 | 部分和 $S_N$ | $f_\Omega$ |
""")
    with st.expander("練習問題"):
        st.markdown("**問3** フーリエ積分定理を使って、$\\displaystyle\\int_0^\\infty\\frac{\\sin\\omega}{\\omega}d\\omega$ を求めよ。")
        if st.toggle("解答・解説を表示", key="pr_fourier2_149"):
            st.markdown("**答3**")
            st.latex(r"f(0)=1=\frac{1}{2\pi}\int_{-\infty}^{\infty}\frac{2\sin\omega}{\omega}d\omega"
                     r"\ \Rightarrow\ \int_0^\infty\frac{\sin\omega}{\omega}d\omega=\frac\pi2")


# ============================================================
# 2-2. フーリエ変換の性質と公式
# ============================================================
def ft_num(f, w, T=30, n=6001):
    t = np.linspace(-T, T, n)
    return trapz(f(t)[None, :] * np.exp(-1j * np.outer(w, t)), t, axis=1)


def tab_properties():
    st.info("時間をずらすと位相だけが変わる(振幅スペクトルは同じ)。cos をかける(変調)と、スペクトルが ±ω₀ に移る。")
    st.markdown(r"""
| 性質 | $f(t)$ | $F(\omega)$ |
|---|---|---|
| 線形性 | $af+bg$ | $aF+bG$ |
| 時間推移 | $f(t-\tau)$ | $e^{-i\omega\tau}F(\omega)$ |
| 周波数推移 | $e^{i\omega_0t}f(t)$ | $F(\omega-\omega_0)$ |
| 変調 | $f(t)\cos\omega_0t$ | $\frac12\{F(\omega-\omega_0)+F(\omega+\omega_0)\}$ |
| 相似 | $f(at)$ | $\frac{1}{|a|}F\!\left(\frac\omega a\right)$ |
| 微分 | $f'(t)$ | $i\omega F(\omega)$ |
| 合成積 | $(f*g)(t)=\int f(\tau)g(t-\tau)d\tau$ | $F(\omega)G(\omega)$ |
| 対称性 | $F(t)$ | $2\pi f(-\omega)$ |
""")
    c1, c2 = st.columns(2)
    base = c1.radio("もとの関数", ["ガウス関数 e^{−t²/2}", "矩形パルス(|t|<1)"], key="fp_b")
    prop = c2.radio("性質", ["時間推移", "変調", "相似", "微分"], key="fp_p")
    if base.startswith("ガウス"):
        f = lambda t: np.exp(-t ** 2 / 2)  # noqa: E731
        F = lambda w: np.sqrt(2 * PI) * np.exp(-w ** 2 / 2)  # noqa: E731
    else:
        f = lambda t: (np.abs(t) < 1).astype(float)  # noqa: E731
        F = lambda w: np.where(np.abs(w) < 1e-12, 2.0, 2 * np.sin(w) / np.where(w == 0, 1, w))  # noqa: E731
        if prop == "微分":
            st.warning("矩形パルスの微分はデルタ関数になるので、微分はガウス関数で試してください。")
            return
    if prop == "時間推移":
        tau = st.slider("τ", -3.0, 3.0, 1.5, 0.1, key="fp_tau")
        g = lambda t: f(t - tau)  # noqa: E731
        G = lambda w: np.exp(-1j * w * tau) * F(w)  # noqa: E731
    elif prop == "変調":
        w0 = st.slider("ω₀", 0.0, 10.0, 5.0, 0.5, key="fp_w0")
        g = lambda t: f(t) * np.cos(w0 * t)  # noqa: E731
        G = lambda w: 0.5 * (F(w - w0) + F(w + w0))  # noqa: E731
    elif prop == "相似":
        a = st.slider("a", 0.25, 4.0, 2.0, 0.25, key="fp_a")
        g = lambda t: f(a * t)  # noqa: E731
        G = lambda w: F(w / a) / a  # noqa: E731
    else:
        g = lambda t: -t * np.exp(-t ** 2 / 2)  # noqa: E731
        G = lambda w: 1j * w * F(w)  # noqa: E731

    t = np.linspace(-6, 6, 1201)
    w = np.linspace(-15, 15, 601)
    Gn = ft_num(g, np.linspace(-15, 15, 31))
    cA, cB = st.columns(2)
    with cA:
        fig = go.Figure(go.Scatter(x=t, y=f(t), mode="lines", line=dict(color=GRAY, dash="dot"), name="f(t)"))
        fig.add_trace(go.Scatter(x=t, y=g(t), mode="lines", line=dict(color=C1, width=2.5), name="g(t)"))
        layout(fig, title="時間領域", xlab="t", height=380)
        show(fig)
    with cB:
        Gw = G(w).astype(complex)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=w, y=np.abs(F(w)), mode="lines", line=dict(color=GRAY, dash="dot"), name="|F(ω)|"))
        fig.add_trace(go.Scatter(x=w, y=np.abs(Gw), mode="lines", line=dict(color=C2, width=2.5), name="|G(ω)|"))
        fig.add_trace(go.Scatter(x=w, y=Gw.real, mode="lines", line=dict(color=C3, width=1), name="Re G"))
        fig.add_trace(go.Scatter(x=w, y=Gw.imag, mode="lines", line=dict(color=C4, width=1), name="Im G"))
        fig.add_trace(go.Scatter(x=np.linspace(-15, 15, 31), y=np.abs(Gn), mode="markers",
                                 marker=dict(color="black", size=6), name="|G| 数値積分"))
        layout(fig, title="周波数領域", xlab="ω", height=380)
        show(fig)
    if prop == "時間推移":
        st.caption("|G| は |F| と同じ。ずらした量は Re G, Im G の振動(位相 −ωτ)に現れる。")

    with st.expander("練習問題"):
        st.markdown("**問4** $f(t)=e^{-|t|}$ のとき、$e^{-|t-2|}$ と $e^{-|t|}\\cos 3t$ のフーリエ変換を求めよ。")
        if st.toggle("解答・解説を表示", key="pr_fourier2_228"):
            st.markdown("**答4**")
            st.latex(r"F=\frac{2}{1+\omega^2},\quad e^{-2i\omega}\frac{2}{1+\omega^2},\quad "
                     r"\frac{1}{1+(\omega-3)^2}+\frac{1}{1+(\omega+3)^2}")


# ============================================================
# 2-3. スペクトル
# ============================================================
def tab_spectrum():
    st.info("$|F(\\omega)|$ を振幅スペクトル、$\\arg F(\\omega)$ を位相スペクトル、$|F(\\omega)|^2$ をエネルギースペクトルという。"
            "時間領域と周波数領域のエネルギーは等しい(パーセバルの等式)。")
    st.latex(r"\int_{-\infty}^{\infty}|f(t)|^2dt=\frac{1}{2\pi}\int_{-\infty}^{\infty}|F(\omega)|^2d\omega")
    sig = st.selectbox("信号", ["減衰振動 e^{−at} sin ω₀t (t ≥ 0)", "波束 e^{−t²/(2σ²)} cos ω₀t", "矩形パルス(幅 w)"],
                       key="sp_s")
    c = st.columns(2)
    if sig.startswith("減衰"):
        a = c[0].slider("a(減衰の速さ)", 0.1, 3.0, 0.5, 0.1, key="sp_a")
        w0 = c[1].slider("ω₀", 1.0, 10.0, 5.0, 0.5, key="sp_w0")
        f = lambda t: np.where(t >= 0, np.exp(-a * np.abs(t)) * np.sin(w0 * t), 0.0)  # noqa: E731
        F = lambda w: w0 / ((a + 1j * w) ** 2 + w0 ** 2)  # noqa: E731
        st.latex(r"F(\omega)=\frac{\omega_0}{(a+i\omega)^2+\omega_0^2}")
        note = f"ピークは ω ≈ ±ω₀ = ±{w0:g}、幅はおよそ 2a = {2 * a:g}。減衰が速い(a 大)ほどスペクトルは広がる。"
    elif sig.startswith("波束"):
        s_ = c[0].slider("σ(継続時間)", 0.2, 3.0, 1.0, 0.1, key="sp_sig")
        w0 = c[1].slider("ω₀", 0.0, 10.0, 5.0, 0.5, key="sp_w0b")
        f = lambda t: np.exp(-t ** 2 / (2 * s_ ** 2)) * np.cos(w0 * t)  # noqa: E731
        F = lambda w: s_ * np.sqrt(2 * PI) / 2 * (np.exp(-s_ ** 2 * (w - w0) ** 2 / 2)  # noqa: E731
                                                  + np.exp(-s_ ** 2 * (w + w0) ** 2 / 2))
        st.latex(r"F(\omega)=\frac{\sigma\sqrt{2\pi}}{2}\left\{e^{-\sigma^2(\omega-\omega_0)^2/2}+e^{-\sigma^2(\omega+\omega_0)^2/2}\right\}")
        note = "長く続く波ほど(σ 大)、周波数がはっきり決まる(ピークが細い)。"
    else:
        wd = c[0].slider("幅 w", 0.2, 4.0, 1.0, 0.1, key="sp_w")
        f = lambda t: (np.abs(t) < wd / 2).astype(float)  # noqa: E731
        F = lambda w: np.where(np.abs(w) < 1e-12, wd, 2 * np.sin(w * wd / 2) / np.where(w == 0, 1, w))  # noqa: E731
        st.latex(r"F(\omega)=\frac{2\sin(\omega w/2)}{\omega}")
        note = "急に立ち上がる信号は、高い周波数まで成分をもつ(|F| がゆっくり減衰する)。"

    t = np.linspace(-8, 12, 4001)
    w = np.linspace(-15, 15, 1501)
    Fw = F(w).astype(complex)
    cA, cB = st.columns(2)
    with cA:
        fig = go.Figure(go.Scatter(x=t, y=f(t), mode="lines", line=dict(color=C1), name="f(t)"))
        layout(fig, title="信号 f(t)", xlab="t", height=320, legend=False)
        show(fig)
        fig = go.Figure(go.Scatter(x=w, y=np.where(np.abs(Fw) > 1e-6, np.angle(Fw), np.nan), mode="lines",
                                   line=dict(color=C3), name="arg F"))
        layout(fig, title="位相スペクトル arg F(ω)", xlab="ω", height=280, legend=False, yrange=(-3.5, 3.5))
        show(fig)
    with cB:
        fig = go.Figure(go.Scatter(x=w, y=np.abs(Fw), mode="lines", line=dict(color=C2), name="|F|"))
        layout(fig, title="振幅スペクトル |F(ω)|", xlab="ω", height=320, legend=False)
        show(fig)
        fig = go.Figure(go.Scatter(x=w, y=np.abs(Fw) ** 2, mode="lines", fill="tozeroy", line=dict(color=C4),
                                   name="|F|²"))
        layout(fig, title="エネルギースペクトル |F(ω)|²", xlab="ω", height=280, legend=False)
        show(fig)
    tt = np.linspace(-40, 60, 400001)
    ww = np.linspace(-400, 400, 800001)
    E_t = float(trapz(f(tt) ** 2, tt))
    E_w = float(trapz(np.abs(F(ww)) ** 2, ww)) / (2 * PI)
    m1, m2 = st.columns(2)
    m1.metric("∫|f|² dt", f"{E_t:.5f}")
    m2.metric("(1/2π)∫|F|² dω", f"{E_w:.5f}")
    st.caption(note + "(右の数値積分は |ω| ≤ 400 で打ち切っているので、矩形パルスでは少し小さく出る)")


# ============================================================
# 2-4. 離散フーリエ変換と複素フーリエ級数
# ============================================================
def tab_dft_series():
    st.info("周期関数を1周期で N 点サンプリングして DFT すると、$X_k/N$ は複素フーリエ係数 $c_k$ の近似になる。"
            "ずれの原因は、高い周波数が低い周波数に折り返す**エイリアシング**。")
    st.latex(r"\frac{X_k}{N}=\frac1N\sum_{j=0}^{N-1}f(x_j)e^{-ikx_j}\ \approx\ \frac{1}{2\pi}\int_0^{2\pi}f(x)e^{-ikx}dx=c_k,"
             r"\qquad x_j=\frac{2\pi j}{N},\qquad \frac{X_k}{N}=\sum_{m=-\infty}^{\infty}c_{k+mN}")
    c1, c2 = st.columns(2)
    name = c1.selectbox("波形", list(WAVES), index=2, key="fdc_w")
    N = c2.select_slider("サンプル数 N", [4, 8, 16, 32, 64, 128, 256], value=16, key="fdc_N")
    W = WAVES[name]
    c = complex_coeffs(name)
    xj = 2 * PI * np.arange(N) / N
    X = np.fft.fft(W["f"](xj, 1.0)) / N
    K = min(N // 2 - 1, 16)
    ks = np.arange(-K, K + 1)
    dft = np.array([X[k % N] for k in ks])
    ex = np.array([c_of(c, k) for k in ks])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ks - 0.18, y=np.abs(ex), width=0.35, marker_color=C1, name="|c_k|(積分)"))
    fig.add_trace(go.Bar(x=ks + 0.18, y=np.abs(dft), width=0.35, marker_color=C2, name="|X_k / N|(DFT)"))
    layout(fig, xlab="k", height=360)
    show(fig)
    rows = [{"k": int(k), "c_k": fmt_c(c_of(c, k), 5), "X_k/N": fmt_c(X[k % N], 5),
             "差": f"{abs(c_of(c, k) - X[k % N]):.2e}"} for k in range(0, min(K, 6) + 1)]
    st.dataframe(rows, hide_index=True, width="stretch")
    k = 1
    alias = sum(c_of(c, k + m * N) for m in range(-(120 // N) - 1, 120 // N + 2) if abs(k + m * N) <= 120)
    st.write(f"エイリアシングの式の確認(k = 1):Σ c_(1+mN) = {fmt_c(alias, 5)}、X₁/N = {fmt_c(X[1], 5)}"
             "(和は |n| ≤ 120 で打ち切り)")
    st.caption("N を増やすとずれは小さくなる。なめらかな波形(三角波・全波整流)ほど c_k が速く減衰するので、"
               "少ない N でもよく一致する。")
    with st.expander("フィルタへの応用"):
        st.write("DFT で周波数成分に分け、不要な成分を 0 にして逆 DFT すると、雑音を取り除ける。"
                 "「離散フーリエ変換とフィルタ」のタブで試せる。")


# ============================================================
# 2-4'. 既存:FFT とフィルタは fourier.tab_dft
# ============================================================


# ============================================================
# 1-3. フーリエ級数の収束定理とパーセバルの等式
# ============================================================
def tab_convergence_parseval():
    from topics.fourier import NMAX, partial_sum

    st.info("フーリエ級数はどんな意味で f に収束するのか。各点での収束(収束定理)と、"
            "2乗平均の意味での収束(パーセバルの等式)の2つを見る。ここでは周期 2π(L = π)で考える。")

    st.subheader("フーリエ級数の収束定理")
    st.markdown("f が周期 2π で**区分的に滑らか**(有限個の点を除いて f, f′ が連続で、その点で左右の極限が存在する)ならば、"
                "すべての x で")
    st.latex(r"\lim_{N\to\infty}S_N(x)=\frac{f(x+0)+f(x-0)}{2},\qquad "
             r"S_N(x)=\frac{a_0}{2}+\sum_{n=1}^{N}(a_n\cos nx+b_n\sin nx)=\sum_{n=-N}^{N}c_ne^{inx}")
    st.caption("f が連続な点では f(x) に、不連続点では左右の極限の平均(跳びの真ん中)に収束する。")

    c1, c2 = st.columns(2)
    name = c1.selectbox("波形", list(WAVES), key="cp_w")
    k0 = c2.slider("調べる点 x₀ = kπ/24 の k", -24, 24, 0, key="cp_k")
    x0 = k0 * PI / 24
    c2.caption(f"x₀ = {k0}π/24 ≈ {x0:.4f}")
    W = WAVES[name]
    a, b, energy = coefficients(name, 1.0)
    fl = float(W["f"](np.array([x0 - 1e-9]), 1.0)[0])
    fr = float(W["f"](np.array([x0 + 1e-9]), 1.0)[0])
    target = (fl + fr) / 2
    Ns = np.arange(1, NMAX + 1)
    SN = np.array([partial_sum(np.array([x0]), a, b, N)[0] for N in Ns])
    cA, cB = st.columns(2)
    with cA:
        x = np.linspace(-PI, PI, 1201)
        fig = go.Figure(go.Scatter(x=x, y=W["f"](x, 1.0), mode="lines", line=dict(color=GRAY, width=2), name="f(x)"))
        fig.add_trace(go.Scatter(x=x, y=partial_sum(x, a, b, 15), mode="lines", line=dict(color=C1), name="S₁₅(x)"))
        fig.add_trace(go.Scatter(x=[x0], y=[target], mode="markers", marker=dict(color=C2, size=11),
                                 name="(f(x₀+0)+f(x₀−0))/2"))
        layout(fig, title="波形と調べる点", xlab="x", height=360)
        show(fig)
    with cB:
        fig = go.Figure(go.Scatter(x=Ns, y=SN, mode="lines+markers", marker=dict(size=4), line=dict(color=C1),
                                   name="S_N(x₀)"))
        fig.add_hline(y=target, line=dict(color=C2, dash="dash"), annotation_text=f"{target:.4f}")
        layout(fig, title="S_N(x₀) を N の関数として見る", xlab="N", height=360, legend=False)
        show(fig)
    if abs(fl - fr) > 1e-6:
        st.warning(f"x₀ は不連続点:f(x₀−0) = {fl:g}、f(x₀+0) = {fr:g}。S_N(x₀) はその平均 {target:g} に収束する。")
    else:
        st.write(f"x₀ では f は連続で、S_N(x₀) → f(x₀) = {target:.5f}(N = {NMAX} で {SN[-1]:.5f})。")
        st.caption("不連続点のすぐ近くでは、収束はとても遅い(ギブス現象の山が近くにあるため)。")

    st.divider()
    st.subheader("パーセバルの等式")
    st.markdown("関数と数列それぞれの「大きさ」を、次のノルムで測る。")
    st.latex(r"\|f\|_{L^2}=\left(\frac{1}{2\pi}\int_{-\pi}^{\pi}|f(x)|^2\,dx\right)^{1/2},\qquad "
             r"\|\{c_n\}\|_{\ell^2}=\left(\sum_{n=-\infty}^{\infty}|c_n|^2\right)^{1/2}")
    st.caption("$L^2$ ノルムは関数の2乗平均の平方根、$\\ell^2$ ノルムは数列の各項の2乗の和の平方根"
               "(ベクトルの長さ $\\sqrt{x_1^2+x_2^2+\\cdots}$ を無限次元にしたもの)。")
    st.markdown("**パーセバルの等式**:f が $L^2$ の関数(2乗が積分可能)ならば、フーリエ係数 $c_n$ について")
    st.latex(r"\|f\|_{L^2}=\|\{c_n\}\|_{\ell^2}\quad\text{すなわち}\quad "
             r"\frac{1}{2\pi}\int_{-\pi}^{\pi}|f(x)|^2dx=\sum_{n=-\infty}^{\infty}|c_n|^2")
    st.latex(r"\left(\text{実数の形:}\ \frac1\pi\int_{-\pi}^{\pi}|f(x)|^2dx=\frac{a_0^2}{2}+\sum_{n=1}^{\infty}(a_n^2+b_n^2)\right)")
    st.markdown("さらに、部分和との誤差(2乗平均誤差)は、使わなかった係数の $\\ell^2$ ノルムに等しい:")
    st.latex(r"\|f-S_N\|_{L^2}^2=\sum_{|n|>N}|c_n|^2\ \xrightarrow{N\to\infty}\ 0")
    st.caption("各点では収束しない場所(ギブス現象)があっても、2乗平均の意味ではいつも 0 に収束する。")

    N = st.slider("N", 1, NMAX, 10, key="cp_N")
    fnorm2 = energy / 2
    cn2 = np.r_[a[0] ** 2 / 4, (a[1:] ** 2 + b[1:] ** 2) / 4]  # |c_n|² (n ≥ 0)、|c_{−n}| = |c_n|
    partial = cn2[0] + 2 * np.cumsum(cn2[1:])
    m1, m2, m3 = st.columns(3)
    m1.metric("‖f‖²_L²", f"{fnorm2:.6f}")
    m2.metric(f"Σ_{{|n|≤{N}}} |c_n|²", f"{partial[N - 1]:.6f}")
    m3.metric(f"‖f − S_{N}‖²_L²", f"{fnorm2 - partial[N - 1]:.2e}")
    cA, cB = st.columns(2)
    with cA:
        fig = go.Figure(go.Scatter(x=Ns, y=partial, mode="lines", line=dict(color=C1, width=2.5),
                                   name="Σ_{|n|≤N} |c_n|²"))
        fig.add_hline(y=fnorm2, line=dict(color=C2, dash="dash"), annotation_text="‖f‖²")
        fig.add_vline(x=N, line=dict(color=GRAY, dash="dot"))
        layout(fig, title="係数の2乗和は ‖f‖² に近づく", xlab="N", height=340, legend=False)
        show(fig)
    with cB:
        err = np.maximum(fnorm2 - partial, 1e-16)
        fig = go.Figure(go.Scatter(x=Ns, y=err, mode="lines", line=dict(color=C3, width=2.5), name="‖f − S_N‖²"))
        fig.add_vline(x=N, line=dict(color=GRAY, dash="dot"))
        fig.update_yaxes(type="log")
        layout(fig, title="2乗平均誤差 ‖f − S_N‖²(対数目盛)", xlab="N", height=340, legend=False)
        show(fig)
    st.caption("誤差の減り方は、なめらかな波形ほど速い(矩形波・のこぎり波は 1/N、三角波は 1/N³ 程度)。")

    with st.expander("応用:級数の和を求める(バーゼル問題)"):
        st.markdown("$f(x)=x\\ (-\\pi<x<\\pi)$ では $|c_n|=\\dfrac1{|n|}\\ (n\\ne0),\\ c_0=0$。パーセバルの等式より")
        st.latex(r"\frac{1}{2\pi}\int_{-\pi}^{\pi}x^2dx=\frac{\pi^2}{3}=\sum_{n\ne0}\frac{1}{n^2}=2\sum_{n=1}^{\infty}\frac1{n^2}"
                 r"\quad\Longrightarrow\quad \sum_{n=1}^{\infty}\frac{1}{n^2}=\frac{\pi^2}{6}")
        n = np.arange(1, 10001)
        st.write(f"確認:Σ_{{n≤10000}} 1/n² = {np.sum(1 / n ** 2):.6f}、π²/6 = {PI ** 2 / 6:.6f}")
    with st.expander("練習問題"):
        st.markdown("**問1** 矩形波($-1\\ (-\\pi<x<0),\\ 1\\ (0<x<\\pi)$)にパーセバルの等式を使い、"
                    "$\\displaystyle\\sum_{k=1}^{\\infty}\\frac{1}{(2k-1)^2}$ を求めよ。")
        st.markdown("**問2** 矩形波のフーリエ級数は $x=0$ でどんな値に収束するか。収束定理から答えよ。")
        if st.toggle("解答・解説を表示", key="pr_fourier2_434"):
            st.markdown("**答1**")
            st.latex(r"b_{2k-1}=\frac{4}{(2k-1)\pi},\quad \frac1\pi\int_{-\pi}^{\pi}1\,dx=2=\sum_k\frac{16}{(2k-1)^2\pi^2}"
                     r"\ \Rightarrow\ \sum_{k=1}^{\infty}\frac{1}{(2k-1)^2}=\frac{\pi^2}{8}")
            st.markdown("**答2**")
            st.write("左右の極限 −1 と 1 の平均で **0**。")
