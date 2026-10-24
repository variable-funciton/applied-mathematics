"""③ フーリエ解析:フーリエ級数・離散フーリエ変換・フーリエ変換"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import C1, C2, C3, C4, GRAY, layout, show, trapz

PI = np.pi


def wrap(x):
    """周期 2π の周期関数にするため [−π, π) に戻す"""
    return (x + PI) % (2 * PI) - PI


WAVES = {
    "矩形波": dict(
        f=lambda x, d: np.sign(wrap(x)),
        tex=r"f(x)=\begin{cases}-1&(-\pi<x<0)\\ \ \ 1&(0<x<\pi)\end{cases}",
        ser=r"f(x)=\frac{4}{\pi}\left(\sin x+\frac{\sin 3x}{3}+\frac{\sin 5x}{5}+\cdots\right)", jump=True),
    "のこぎり波": dict(
        f=lambda x, d: wrap(x), tex=r"f(x)=x\quad(-\pi<x<\pi)",
        ser=r"f(x)=2\left(\sin x-\frac{\sin 2x}{2}+\frac{\sin 3x}{3}-\cdots\right)", jump=True),
    "三角波": dict(
        f=lambda x, d: np.abs(wrap(x)), tex=r"f(x)=|x|\quad(-\pi<x<\pi)",
        ser=r"f(x)=\frac{\pi}{2}-\frac{4}{\pi}\left(\cos x+\frac{\cos 3x}{3^2}+\frac{\cos 5x}{5^2}+\cdots\right)",
        jump=False),
    "全波整流 |sin x|": dict(
        f=lambda x, d: np.abs(np.sin(x)), tex=r"f(x)=|\sin x|",
        ser=r"f(x)=\frac{2}{\pi}-\frac{4}{\pi}\left(\frac{\cos 2x}{1\cdot 3}+\frac{\cos 4x}{3\cdot 5}"
            r"+\frac{\cos 6x}{5\cdot 7}+\cdots\right)", jump=False),
    "半波整流": dict(
        f=lambda x, d: np.maximum(np.sin(x), 0), tex=r"f(x)=\max(\sin x,\ 0)",
        ser=r"f(x)=\frac1\pi+\frac12\sin x-\frac{2}{\pi}\left(\frac{\cos 2x}{1\cdot 3}"
            r"+\frac{\cos 4x}{3\cdot 5}+\cdots\right)", jump=False),
    "パルス列(幅 2d)": dict(
        f=lambda x, d: (np.abs(wrap(x)) < d).astype(float),
        tex=r"f(x)=\begin{cases}1&(|x|<d)\\0&(d<|x|<\pi)\end{cases}",
        ser=r"f(x)=\frac{d}{\pi}+\sum_{n=1}^{\infty}\frac{2\sin nd}{n\pi}\cos nx", jump=True),
}

NMAX = 120


@st.cache_data
def coefficients(name, d):
    x = np.linspace(-PI, PI, 40001)
    f = WAVES[name]["f"](x, d)
    n = np.arange(NMAX + 1)[:, None]
    a = trapz(f * np.cos(n * x), x, axis=1) / PI
    b = trapz(f * np.sin(n * x), x, axis=1) / PI
    a[np.abs(a) < 1e-9] = 0
    b[np.abs(b) < 1e-9] = 0
    energy = trapz(f ** 2, x) / PI
    return a, b, energy


def partial_sum(x, a, b, N, fejer=False):
    s = np.full_like(x, a[0] / 2)
    for n in range(1, N + 1):
        wgt = 1 - n / (N + 1) if fejer else 1.0
        s += wgt * (a[n] * np.cos(n * x) + b[n] * np.sin(n * x))
    return s




# ---------- フーリエ級数 ----------
def tab_series():
    st.info("周期 2L の関数を、いろいろな周波数の cos・sin の和で表す。項を増やすと元の波形に近づいていく。")
    st.markdown("**一般周期 2L のフーリエ級数**")
    st.latex(r"f(x)\sim\frac{a_0}{2}+\sum_{n=1}^{\infty}\left(a_n\cos\frac{n\pi x}{L}+b_n\sin\frac{n\pi x}{L}\right)")
    st.latex(r"a_n=\frac1L\int_{-L}^{L}f(x)\cos\frac{n\pi x}{L}\,dx\quad(n=0,1,2,\ldots),\qquad "
             r"b_n=\frac1L\int_{-L}^{L}f(x)\sin\frac{n\pi x}{L}\,dx\quad(n=1,2,\ldots)")
    st.caption("L = π とすると、周期 2π のよく使う形 $a_n=\\frac1\\pi\\int_{-\\pi}^{\\pi}f(x)\\cos nx\\,dx$ などになる。")

    c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
    name = c1.selectbox("波形", list(WAVES), key="fs_wave")
    Lc = c2.radio("周期 2L", ["2π(L = π)", "2(L = 1)", "4(L = 2)", "10(L = 5)"], key="fs_L")
    N = c3.slider("項数 N(部分和 S_N)", 1, NMAX, 5, key="fs_N")
    d = 1.0
    if "パルス" in name:
        d = c4.slider("d", 0.1, 3.0, 1.0, 0.1, key="fs_d")
    L = {"2π(L = π)": PI, "2(L = 1)": 1.0, "4(L = 2)": 2.0, "10(L = 5)": 5.0}[Lc]
    W = WAVES[name]
    st.markdown("**選んだ波形(L = π の形で表示)**")
    st.latex(W["tex"])
    st.latex(W["ser"])
    if L != PI:
        st.caption(f"周期 {2 * L:g} の場合は、上の式の x を πx/{L:g} に置きかえたもの。"
                   "係数 a_n, b_n は L によらず同じ値になる(置換積分 x = Lu/π で確かめよう)。")

    a, b, energy = coefficients(name, d)
    fejer = st.checkbox("フェイェール平均(チェザロ平均)σ_N も表示する", key="fs_fej")

    xx = np.linspace(-3 * L, 3 * L, 3001)
    x = PI * xx / L
    f = W["f"](x, d)
    S = partial_sum(x, a, b, N)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xx, y=f, mode="lines", line=dict(color=GRAY, width=2), name="f(x)"))
    fig.add_trace(go.Scatter(x=xx, y=S, mode="lines", line=dict(color=C1, width=2.5),
                             name=f"部分和 S_{N}(x)"))
    if fejer:
        fig.add_trace(go.Scatter(x=xx, y=partial_sum(x, a, b, N, True), mode="lines",
                                 line=dict(color=C3, width=2), name=f"フェイェール平均 σ_{N}(x)"))
    lab = "π" if L == PI else f"{L:g}"
    fig.update_xaxes(tickvals=[k * L for k in range(-3, 4)],
                     ticktext=[("0" if k == 0 else f"{'−' if k < 0 else ''}{abs(k) if abs(k) > 1 else ''}{lab}")
                               if L == PI else f"{k * L:g}" for k in range(-3, 4)])
    layout(fig, height=420, xlab="x")
    show(fig)

    if W["jump"]:
        over = S.max() - f.max()
        st.metric("最大の行き過ぎ max S_N − max f", f"{over:.4f}")
        st.caption("不連続点の近くでは、N を増やしても行き過ぎ(跳びの約 9%)が消えない:**ギブス現象**。"
                   "ただし行き過ぎる場所は不連続点に近づき、平均二乗誤差は 0 に向かう"
                   "(「3. 収束定理とパーセバルの等式」の節)。フェイェール平均を使うとギブス現象は起こらない。")

    st.subheader("スペクトル")
    st.markdown(
        "フーリエ級数は、f を周波数 n の波 $\\cos\\frac{n\\pi x}{L},\\ \\sin\\frac{n\\pi x}{L}$ に分解したもの。"
        "**各周波数の波がどれだけの大きさで含まれているか**を、n ごとに並べたものを**スペクトル**という。\n"
        "- 振幅スペクトル:$A_n=\\sqrt{a_n^2+b_n^2}$($a_n\\cos+b_n\\sin=A_n\\cos(\\cdots-\\varphi_n)$ とまとめたときの振幅)\n"
        "- 棒が高い n ほど、その周波数の成分を多く含む。音なら、どの高さの音がどれだけ混ざっているか(音色)にあたる。\n"
        "- なめらかな波形ほど、n が大きくなると速く 0 に近づく(三角波は 1/n²、矩形波は 1/n)。"
    )
    cA, cB = st.columns(2)
    with cA:
        n = np.arange(0, N + 1)
        amp = np.hypot(a[:N + 1], b[:N + 1])
        amp[0] = abs(a[0]) / 2
        fig = go.Figure(go.Bar(x=n, y=amp, marker_color=C1, name="振幅"))
        layout(fig, title="振幅スペクトル A_n = √(a_n² + b_n²)(A_0 = |a_0|/2)", xlab="周波数の番号 n",
               ylab="振幅", height=330, legend=False)
        show(fig)
    with cB:
        rows = [{"n": k, "a_n": f"{a[k]:.5f}", "b_n": f"{b[k]:.5f}"} for k in range(0, min(N, 12) + 1)]
        st.dataframe(rows, hide_index=True, width="stretch", height=330)

    with st.expander("収束の理論(ディリクレ核とフェイェール核)"):
        st.write("部分和と、その平均は畳み込みで書ける:")
        st.latex(r"S_N f=f*D_N,\ \ D_N(x)=\frac{\sin\left(N+\frac12\right)x}{2\pi\sin\frac x2},\qquad"
                 r"\sigma_N f=f*F_N,\ \ F_N(x)=\frac{1}{2\pi(N+1)}\left(\frac{\sin\frac{(N+1)x}{2}}{\sin\frac x2}\right)^2")
        xk = np.linspace(-PI, PI, 2001)
        xk[np.abs(xk) < 1e-9] = 1e-9
        Dn = np.sin((N + 0.5) * xk) / (2 * PI * np.sin(xk / 2))
        Fn = (np.sin((N + 1) * xk / 2) / np.sin(xk / 2)) ** 2 / (2 * PI * (N + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xk, y=Dn, mode="lines", line=dict(color=C1), name="D_N(ディリクレ核)"))
        fig.add_trace(go.Scatter(x=xk, y=Fn, mode="lines", line=dict(color=C3), name="F_N(フェイェール核)"))
        layout(fig, height=320, xlab="x")
        show(fig)
        st.write("$F_N\\ge 0$ で積分が 1 の「よい核」なので、連続関数なら $\\sigma_N f\\to f$(一様収束)。"
                 "$D_N$ は負の値をとり、$\\int|D_N|\\to\\infty$ となるのがギブス現象や発散の原因。")

    with st.expander("練習問題"):
        st.markdown("**問1** $f(x)=x\\ (-\\pi<x<\\pi)$ のフーリエ係数を求めよ。")
        st.latex(r"a_n=0\ (\text{奇関数}),\qquad b_n=\frac1\pi\int_{-\pi}^{\pi}x\sin nx\,dx=\frac{2(-1)^{n+1}}{n}")
        st.markdown("**問2** 矩形波の級数で $x=\\frac\\pi2$ とおいて、$1-\\frac13+\\frac15-\\cdots$ の値を求めよ。")
        st.latex(r"1=\frac4\pi\left(1-\frac13+\frac15-\cdots\right)\ \Longrightarrow\ 1-\frac13+\frac15-\cdots=\frac\pi4")
        st.markdown("**問3** 不連続点 $x=0$ で、矩形波のフーリエ級数はどんな値に収束するか。")
        st.write("すべての項 $\\sin nx$ が 0 なので **0**。左右の極限の平均 $\\frac{(-1)+1}{2}=0$ に一致する。")


# ---------- 離散フーリエ変換 ----------
def tab_dft():
    st.info("計測した信号(サンプル列)を FFT で周波数成分に分け、不要な成分(ノイズ)を取り除く。")
    st.latex(r"x(t)=A_1\sin 2\pi f_1t+A_2\sin 2\pi f_2t+(\text{ノイズ}),\qquad "
             r"X_k=\sum_{n=0}^{N-1}x_n\,e^{-2\pi i kn/N}")
    c1, c2, c3 = st.columns(3)
    f1 = c1.slider("f₁ [Hz]", 1, 120, 5, key="fd_f1")
    A1 = c1.slider("A₁", 0.0, 2.0, 1.0, 0.1, key="fd_A1")
    f2 = c2.slider("f₂ [Hz]", 1, 120, 40, key="fd_f2")
    A2 = c2.slider("A₂", 0.0, 2.0, 0.5, 0.1, key="fd_A2")
    fs = c3.slider("サンプリング周波数 fs [Hz]", 20, 500, 200, 10, key="fd_fs")
    sig = c3.slider("ノイズの大きさ σ", 0.0, 1.0, 0.3, 0.05, key="fd_sig")

    Ny = fs / 2
    for k, fk in ((1, f1), (2, f2)):
        if fk > Ny:
            alias = abs(fk - fs * round(fk / fs))
            st.warning(f"f{k} = {fk} Hz はナイキスト周波数 fs/2 = {Ny:g} Hz を超えています。"
                       f"スペクトルでは **{alias:g} Hz** に現れます(エイリアシング)。")

    rng = np.random.default_rng(0)
    T = 1.0
    t = np.arange(0, T, 1 / fs)
    clean = A1 * np.sin(2 * PI * f1 * t) + A2 * np.sin(2 * PI * f2 * t)
    x = clean + sig * rng.standard_normal(len(t))
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(t), 1 / fs)
    amp = np.abs(X) / len(t) * 2
    amp[0] /= 2

    fc = st.slider("ローパスフィルタ:この周波数より上を消す [Hz]", 1.0, float(Ny), float(min(Ny, 20)),
                   1.0, key="fd_fc")
    Xf = np.where(freqs <= fc, X, 0)
    xf = np.fft.irfft(Xf, n=len(t))

    tf = np.linspace(0, T, 4000)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=x, mode="lines+markers" if fs <= 120 else "lines",
                             line=dict(color=GRAY, width=1), marker=dict(size=4), name="観測信号(ノイズ入り)"))
    fig.add_trace(go.Scatter(x=t, y=xf, mode="lines", line=dict(color=C2, width=2.5),
                             name=f"フィルタ後(≤ {fc:g} Hz)"))
    fig.add_trace(go.Scatter(x=tf, y=A1 * np.sin(2 * PI * f1 * tf), mode="lines",
                             line=dict(color=C1, dash="dot"), name="f₁ 成分(真の値)", visible="legendonly"))
    layout(fig, title="時間領域", xlab="t [s]", height=380, xrange=(0, 0.5))
    show(fig)

    fig = go.Figure(go.Bar(x=freqs, y=amp, marker_color=C1, name="振幅"))
    fig.add_vrect(x0=fc, x1=Ny, fillcolor="rgba(214,39,40,0.12)", line_width=0,
                  annotation_text="消す範囲", annotation_position="top left")
    layout(fig, title="振幅スペクトル(片側)", xlab="周波数 [Hz]", height=330, legend=False)
    show(fig)
    st.caption("ノイズはすべての周波数に薄く広がり、正弦波は1本の鋭いピークになる。"
               "ピークを残して他を消すとノイズが減る。")

    with st.expander("練習問題"):
        st.markdown("**問4** fs = 100 Hz で 70 Hz の正弦波をサンプリングすると、何 Hz に見えるか。")
        st.write("ナイキスト周波数は 50 Hz。|70 − 100| = **30 Hz** に見える(上のスライダーで確かめよう)。")
        st.markdown("**問5** 1 秒間のデータを FFT したとき、周波数の刻み(分解能)は何 Hz か。")
        st.write("**1 Hz**(= 1 / 観測時間)。2 秒間なら 0.5 Hz。")


# ---------- フーリエ変換 ----------
PULSES = {
    "矩形パルス": dict(
        f=lambda t, p: (np.abs(t) < p / 2).astype(float),
        F=lambda w, p: np.where(np.abs(w) < 1e-12, p, 2 * np.sin(w * p / 2) / np.where(w == 0, 1, w)),
        tex=r"f(t)=\begin{cases}1&(|t|<\frac w2)\\0&(|t|>\frac w2)\end{cases}",
        Ftex=r"F(\omega)=\frac{2\sin(\omega w/2)}{\omega}", par="幅 w"),
    "ガウス関数": dict(
        f=lambda t, p: np.exp(-t ** 2 / (2 * p ** 2)),
        F=lambda w, p: p * np.sqrt(2 * PI) * np.exp(-(p * w) ** 2 / 2),
        tex=r"f(t)=e^{-t^2/(2\sigma^2)}", Ftex=r"F(\omega)=\sigma\sqrt{2\pi}\,e^{-\sigma^2\omega^2/2}",
        par="幅 σ"),
    "両側指数関数": dict(
        f=lambda t, p: np.exp(-np.abs(t) / p),
        F=lambda w, p: 2 * (1 / p) / ((1 / p) ** 2 + w ** 2),
        tex=r"f(t)=e^{-a|t|}\quad(a=1/\tau)", Ftex=r"F(\omega)=\frac{2a}{a^2+\omega^2}", par="幅 τ"),
}


def tab_transform():
    st.info("周期をもたない関数のスペクトル。**時間で狭い関数ほど、周波数では広がる**(不確定性)。")
    st.latex(r"F(\omega)=\int_{-\infty}^{\infty}f(t)\,e^{-i\omega t}\,dt,\qquad "
             r"f(t)=\frac{1}{2\pi}\int_{-\infty}^{\infty}F(\omega)\,e^{i\omega t}\,d\omega")
    c1, c2 = st.columns(2)
    name = c1.selectbox("関数", list(PULSES), key="ft_f")
    P = PULSES[name]
    p = c2.slider(P["par"], 0.2, 4.0, 1.0, 0.1, key="ft_p")
    st.latex(P["tex"] + r"\quad\Longrightarrow\quad " + P["Ftex"])

    t = np.linspace(-8, 8, 2001)
    w = np.linspace(-20, 20, 2001)
    cA, cB = st.columns(2)
    with cA:
        fig = go.Figure(go.Scatter(x=t, y=P["f"](t, p), mode="lines", fill="tozeroy",
                                   line=dict(color=C1), name="f(t)"))
        layout(fig, title="時間領域 f(t)", xlab="t", height=360, yrange=(-0.1, 1.2), legend=False)
        show(fig)
    with cB:
        fig = go.Figure(go.Scatter(x=w, y=P["F"](w, p), mode="lines", line=dict(color=C2),
                                   name="F(ω)(公式)"))
        tn = np.linspace(-60, 60, 240001)
        fn = P["f"](tn, p)
        wn = np.linspace(-18, 18, 25)
        Fn = [trapz(fn * np.cos(wk * tn), tn) for wk in wn]  # 偶関数なので cos だけ
        fig.add_trace(go.Scatter(x=wn, y=Fn, mode="markers", marker=dict(color="black", size=6),
                                 name="数値積分"))
        layout(fig, title="周波数領域 F(ω)", xlab="ω", height=360)
        show(fig)
    st.caption("f が偶関数なので F(ω) は実数。黒い点は定義の積分を数値計算した値で、公式と一致する。")
    if name == "ガウス関数":
        st.success(f"時間の幅 σ = {p:g}、周波数の幅 1/σ = {1 / p:.3g}、積 = 1。"
                   "ガウス関数はフーリエ変換してもガウス関数で、幅の積が最小になる。")
    elif name == "矩形パルス":
        st.write(f"F(ω) が最初に 0 になる ω = 2π/w = **{2 * PI / p:.3f}**。幅 w を半分にすると、2倍に広がる。")

    with st.expander("練習問題"):
        st.markdown("**問6** 矩形パルス(幅 w)のフーリエ変換を求めよ。")
        st.latex(r"F(\omega)=\int_{-w/2}^{w/2}e^{-i\omega t}dt=\left[\frac{e^{-i\omega t}}{-i\omega}\right]_{-w/2}^{w/2}"
                 r"=\frac{2\sin(\omega w/2)}{\omega}")
        st.markdown("**問7** $f(t)$ のフーリエ変換を $F(\\omega)$ とするとき、$f(at)\\ (a>0)$ のフーリエ変換を求めよ。")
        st.latex(r"\int f(at)e^{-i\omega t}dt=\frac1a F\!\left(\frac{\omega}{a}\right)")
        st.write("時間で $1/a$ 倍に縮めると、周波数では $a$ 倍に広がる。")
