"""② ラプラス変換:相似性と移動法則/微分法則と積分法則/合成積/伝達関数とデルタ関数/演算子法"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import C1, C2, C3, C4, GRAY, layout, show, trapz

PI = np.pi


def lap(g, s, sig, n=120001):
    """数値ラプラス変換(sig:g の収束座標)"""
    T = min(50 / max(s - sig, 1e-3) + 5, 3000)
    t = np.linspace(0, T, n)
    return trapz(g(t) * np.exp(-s * t), t)


def cumint(y, t):
    """累積台形積分 ∫_0^t"""
    return np.concatenate([[0], np.cumsum((y[1:] + y[:-1]) / 2 * np.diff(t))])


# ============================================================
# 1-2. 相似性と移動法則
# ============================================================
BASE = {
    "sin t": dict(f=np.sin, F=lambda s: 1 / (s ** 2 + 1), s0=0.0, tex=r"\sin t", Ftex=r"\frac{1}{s^2+1}"),
    "1": dict(f=lambda t: np.ones_like(t), F=lambda s: 1 / s, s0=0.0, tex=r"1", Ftex=r"\frac1s"),
    "t": dict(f=lambda t: t, F=lambda s: 1 / s ** 2, s0=0.0, tex=r"t", Ftex=r"\frac{1}{s^2}"),
    "e^{−t}": dict(f=lambda t: np.exp(-t), F=lambda s: 1 / (s + 1), s0=-1.0, tex=r"e^{-t}",
                   Ftex=r"\frac{1}{s+1}"),
    "cos 2t": dict(f=lambda t: np.cos(2 * t), F=lambda s: s / (s ** 2 + 4), s0=0.0, tex=r"\cos 2t",
                   Ftex=r"\frac{s}{s^2+4}"),
}


def tab_shift():
    st.info("t 側で「縮める・指数をかける・ずらす」と、s 側でそれぞれ決まった変化が起こる。")
    st.markdown("$\\mathcal L[f(t)]=F(s)$ とする。")
    st.markdown("**相似法則**(t 軸方向に $\\frac1a$ 倍に縮める)")
    st.latex(r"\mathcal L[f(at)]=\frac1a\,F\!\left(\frac sa\right)\qquad(a>0)")
    st.markdown("**第1移動法則**(指数関数 $e^{\\alpha t}$ をかけると、**s 軸方向**に α だけずれる)")
    st.latex(r"\mathcal L[e^{\alpha t}f(t)]=F(s-\alpha)")
    st.markdown("**第2移動法則**(**t 軸方向**に τ だけ遅らせると、$e^{-\\tau s}$ がかかる)")
    st.latex(r"\mathcal L[f(t-\tau)\,U(t-\tau)]=e^{-\tau s}F(s)\qquad(\tau\ge0,\ U\text{ は単位ステップ関数})")
    st.divider()
    c1, c2 = st.columns(2)
    name = c1.selectbox("もとの関数 f(t)", list(BASE), key="ls_f")
    law = c2.radio("確かめる法則", ["相似法則", "第1移動法則(s 軸の移動)", "第2移動法則(t 軸の移動)"], key="ls_law")
    st.markdown(f"**{law.split('(')[0]}** を確かめる")
    B = BASE[name]
    f, F, s0 = B["f"], B["F"], B["s0"]

    if law == "相似法則":
        a = st.slider("a", 0.25, 4.0, 2.0, 0.25, key="ls_a")
        g = lambda t: f(a * t)  # noqa: E731
        G = lambda s: F(s / a) / a  # noqa: E731
        sg = a * s0
        st.latex(rf"\mathcal L[f({a:g}t)]=\frac{{1}}{{{a:g}}}F\!\left(\frac{{s}}{{{a:g}}}\right)")
    elif law.startswith("第1"):
        al = st.slider("α", -2.0, 2.0, -0.5, 0.1, key="ls_al")
        g = lambda t: np.exp(al * t) * f(t)  # noqa: E731
        G = lambda s: F(s - al)  # noqa: E731
        sg = s0 + al
        st.latex(rf"\mathcal L[e^{{{al:g}t}}{B['tex']}]=F(s-({al:g}))")
    else:
        tau = st.slider("τ", 0.0, 4.0, 1.5, 0.1, key="ls_tau")
        g = lambda t: np.where(t >= tau, f(np.maximum(t - tau, 0)), 0.0)  # noqa: E731
        G = lambda s: np.exp(-tau * s) * F(s)  # noqa: E731
        sg = s0
        st.latex(rf"\mathcal L[f(t-{tau:g})U(t-{tau:g})]=e^{{-{tau:g}s}}F(s)")
    st.latex(rf"f(t)={B['tex']},\quad F(s)={B['Ftex']}")

    t = np.linspace(0, 10, 2001)
    cA, cB = st.columns(2)
    with cA:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=f(t), mode="lines", line=dict(color=GRAY, dash="dot"), name="f(t)"))
        fig.add_trace(go.Scatter(x=t, y=g(t), mode="lines", line=dict(color=C1, width=2.5), name="変換後 g(t)"))
        layout(fig, title="t 領域", xlab="t", height=380, yrange=(-2, 4))
        show(fig)
    with cB:
        lo = max(sg, s0) + 0.15
        ss = np.linspace(lo, lo + 5, 300)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ss, y=F(ss), mode="lines", line=dict(color=GRAY, dash="dot"), name="F(s)"))
        fig.add_trace(go.Scatter(x=ss, y=G(ss), mode="lines", line=dict(color=C2, width=2.5), name="G(s)(公式)"))
        sn = np.linspace(lo + 0.3, lo + 4.5, 8)
        fig.add_trace(go.Scatter(x=sn, y=[lap(g, s, sg) for s in sn], mode="markers",
                                 marker=dict(color="black", size=7), name="g(t) を数値積分"))
        layout(fig, title="s 領域", xlab="s", height=380)
        show(fig)
    st.caption("黒い点:g(t) を定義どおり数値積分した値。公式の曲線と一致する。")

    with st.expander("練習問題"):
        st.markdown("**問1** $\\mathcal L[e^{-2t}\\cos 3t]$ を求めよ。")
        st.latex(r"\mathcal L[\cos3t]=\frac{s}{s^2+9}\ \Rightarrow\ \frac{s+2}{(s+2)^2+9}")
        st.markdown("**問2** $\\mathcal L^{-1}\\left[\\dfrac{e^{-2s}}{s^2}\\right]$ を求めよ。")
        st.latex(r"(t-2)\,U(t-2)")
        st.markdown("**問3** $\\mathcal L[\\sin\\omega t]$ を相似法則で求めよ。")
        st.latex(r"\frac1\omega\cdot\frac{1}{(s/\omega)^2+1}=\frac{\omega}{s^2+\omega^2}")


# ============================================================
# 1-3. 微分法則と積分法則
# ============================================================
FD = {
    "sin 2t": dict(f=lambda t: np.sin(2 * t), d=lambda t: 2 * np.cos(2 * t), dd=lambda t: -4 * np.sin(2 * t),
                   F=lambda s: 2 / (s ** 2 + 4), tex=r"\sin 2t", f0=0.0, d0=2.0),
    "t²": dict(f=lambda t: t ** 2, d=lambda t: 2 * t, dd=lambda t: 2 + 0 * t, F=lambda s: 2 / s ** 3,
               tex=r"t^2", f0=0.0, d0=0.0),
    "e^{−2t}": dict(f=lambda t: np.exp(-2 * t), d=lambda t: -2 * np.exp(-2 * t), dd=lambda t: 4 * np.exp(-2 * t),
                    F=lambda s: 1 / (s + 2), tex=r"e^{-2t}", f0=1.0, d0=-2.0),
    "cos t + t": dict(f=lambda t: np.cos(t) + t, d=lambda t: 1 - np.sin(t), dd=lambda t: -np.cos(t),
                      F=lambda s: s / (s ** 2 + 1) + 1 / s ** 2, tex=r"\cos t+t", f0=1.0, d0=1.0),
}


def tab_diff_int():
    st.info("t で微分すると s 倍(初期値の補正つき)、t で積分すると 1/s 倍。"
            "これで微分方程式が s の代数方程式に変わる。")
    st.markdown("$\\mathcal L[f(t)]=F(s)$ とする。")
    st.markdown("**微分法則**(t で微分すると、s 倍して初期値を引く)")
    st.latex(r"\mathcal L[f'(t)]=sF(s)-f(0)")
    st.latex(r"\mathcal L[f''(t)]=s^2F(s)-sf(0)-f'(0)")
    st.markdown("**積分法則**(0 から t まで積分すると、s で割る)")
    st.latex(r"\mathcal L\!\left[\int_0^tf(u)\,du\right]=\frac{F(s)}{s}")
    st.markdown("**像関数の微分法則**(t をかけると、F を s で微分して −1 倍)")
    st.latex(r"\mathcal L[t\,f(t)]=-F'(s)")
    st.divider()
    st.markdown("**4つの法則を、数値積分(定義どおりの計算)と比べて確かめる**")
    c1, c2 = st.columns(2)
    name = c1.selectbox("f(t)", list(FD), key="ldi_f")
    s = c2.slider("s", 0.5, 4.0, 1.5, 0.1, key="ldi_s")
    D = FD[name]
    t = np.linspace(0, 60 / s + 5, 200001)
    ft = D["f"](t)
    I = cumint(ft, t)
    e = np.exp(-s * t)
    Fs = D["F"](s)
    dF = (D["F"](s + 1e-5) - D["F"](s - 1e-5)) / 2e-5
    rows = [
        {"法則": "微分法則 L[f']", "数値積分": trapz(D["d"](t) * e, t), "公式": s * Fs - D["f0"], "公式の形": "sF − f(0)"},
        {"法則": "微分法則 L[f'']", "数値積分": trapz(D["dd"](t) * e, t), "公式": s * s * Fs - s * D["f0"] - D["d0"],
         "公式の形": "s²F − s f(0) − f'(0)"},
        {"法則": "積分法則 L[∫₀ᵗ f]", "数値積分": trapz(I * e, t), "公式": Fs / s, "公式の形": "F / s"},
        {"法則": "像関数の微分 L[t f]", "数値積分": trapz(t * ft * e, t), "公式": -dF, "公式の形": "−F'(s)"},
    ]
    for r in rows:
        r["数値積分"] = f"{r['数値積分']:.6f}"
        r["公式"] = f"{r['公式']:.6f}"
    st.latex(rf"f(t)={D['tex']},\quad f(0)={D['f0']:g},\quad f'(0)={D['d0']:g},\quad F({s:g})={Fs:.6f}")
    st.dataframe(rows, hide_index=True, width="stretch")

    tp = np.linspace(0, 6, 1201)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tp, y=D["f"](tp), mode="lines", line=dict(color=C1, width=2.5), name="f(t)"))
    fig.add_trace(go.Scatter(x=tp, y=D["d"](tp), mode="lines", line=dict(color=C2), name="f'(t)"))
    fig.add_trace(go.Scatter(x=tp, y=cumint(D["f"](tp), tp), mode="lines", line=dict(color=C3),
                             name="∫₀ᵗ f(u) du"))
    layout(fig, xlab="t", height=360, yrange=(-4, 6))
    show(fig)

    with st.expander("微分法則の導き方(部分積分)"):
        st.latex(r"\int_0^\infty f'(t)e^{-st}dt=\Big[f(t)e^{-st}\Big]_0^\infty+s\int_0^\infty f(t)e^{-st}dt=-f(0)+sF(s)")
    with st.expander("練習問題"):
        st.markdown("**問4** $\\mathcal L[t\\sin t]$ を求めよ。")
        st.latex(r"-\frac{d}{ds}\frac{1}{s^2+1}=\frac{2s}{(s^2+1)^2}")
        st.markdown("**問5** $\\mathcal L^{-1}\\left[\\dfrac{1}{s(s^2+1)}\\right]$ を積分法則で求めよ。")
        st.latex(r"\int_0^t\sin u\,du=1-\cos t")


# ============================================================
# 2-2. 合成積
# ============================================================
CF = {
    "1": (lambda t: np.ones_like(t), lambda s: 1 / s, r"1"),
    "t": (lambda t: t, lambda s: 1 / s ** 2, r"t"),
    "e^{−t}": (lambda t: np.exp(-t), lambda s: 1 / (s + 1), r"e^{-t}"),
    "e^{−2t}": (lambda t: np.exp(-2 * t), lambda s: 1 / (s + 2), r"e^{-2t}"),
    "sin t": (np.sin, lambda s: 1 / (s ** 2 + 1), r"\sin t"),
    "パルス U(t) − U(t−1)": (lambda t: ((t >= 0) & (t < 1)).astype(float), lambda s: (1 - np.exp(-s)) / s,
                            r"U(t)-U(t-1)"),
}


def tab_convolution():
    st.info("合成積は「g を左右反転して t だけずらし、f とかけて面積をとる」操作。"
            "ラプラス変換すると、ただのかけ算になる。")
    st.latex(r"(f*g)(t)=\int_0^t f(\tau)\,g(t-\tau)\,d\tau,\qquad \mathcal L[f*g]=F(s)\,G(s)")
    c1, c2, c3 = st.columns(3)
    fn = c1.selectbox("f", list(CF), index=5, key="lc_f")
    gn = c2.selectbox("g", list(CF), index=2, key="lc_g")
    t0 = c3.slider("t", 0.0, 8.0, 1.5, 0.05, key="lc_t")
    f, F, ftex = CF[fn]
    g, G, gtex = CF[gn]
    st.latex(rf"f(t)={ftex},\qquad g(t)={gtex}")

    tau = np.linspace(-1, 8, 1801)
    fv = np.where(tau >= 0, f(np.maximum(tau, 0)), 0)
    gv = np.where((t0 - tau >= 0) & (tau >= 0), g(np.maximum(t0 - tau, 0)), 0)
    prod = fv * gv
    tt = np.linspace(0, 8, 1601)
    dt = tt[1] - tt[0]
    conv = np.convolve(f(tt), g(tt))[:len(tt)] * dt - 0.5 * dt * (f(tt) * g(0 * tt) + f(0 * tt) * g(tt))
    val = float(trapz(prod, tau))

    cA, cB = st.columns(2)
    with cA:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tau, y=prod, mode="lines", fill="tozeroy", line=dict(color=C4, width=0.5),
                                 fillcolor="rgba(255,127,14,0.35)", name="f(τ)g(t−τ)"))
        fig.add_trace(go.Scatter(x=tau, y=fv, mode="lines", line=dict(color=C1, width=2.5), name="f(τ)"))
        fig.add_trace(go.Scatter(x=tau, y=np.where(tau <= t0, g(np.abs(t0 - tau)), np.nan), mode="lines",
                                 line=dict(color=C2, width=2.5), name="g(t − τ)(反転してずらす)"))
        fig.add_vline(x=t0, line=dict(color=GRAY, dash="dot"), annotation_text="τ = t")
        layout(fig, title="τ の関数として見る", xlab="τ", height=380, yrange=(-1.2, 2.2))
        show(fig)
    with cB:
        fig = go.Figure(go.Scatter(x=tt, y=conv, mode="lines", line=dict(color=C3, width=2.5), name="(f*g)(t)"))
        fig.add_trace(go.Scatter(x=[t0], y=[val], mode="markers", marker=dict(color=C4, size=11),
                                 name="オレンジの面積"))
        layout(fig, title="合成積 (f*g)(t)", xlab="t", height=380)
        show(fig)
    sc = 1.5
    Lc = trapz(np.interp(np.linspace(0, 8, 1601), tt, conv) * np.exp(-sc * tt), tt)
    tail = "(t ≤ 8 で打ち切り)"
    st.write(f"確認:s = {sc} で L[f*g] ≈ {Lc:.4f}{tail}、F(s)G(s) = {F(sc) * G(sc):.4f}")

    with st.expander("練習問題"):
        st.markdown("**問6** $1*e^{-t}$ を定義から計算し、$\\mathcal L$ の積の形とも比べよ。")
        st.latex(r"\int_0^t e^{-(t-\tau)}d\tau=1-e^{-t},\qquad \frac1s\cdot\frac{1}{s+1}=\frac1s-\frac{1}{s+1}")
        st.markdown("**問7** $\\mathcal L^{-1}\\left[\\dfrac{1}{(s^2+1)^2}\\right]$ を合成積で表せ。")
        st.latex(r"\sin t*\sin t=\int_0^t\sin\tau\sin(t-\tau)\,d\tau=\frac12(\sin t-t\cos t)")


# ============================================================
# 2-3. 伝達関数とデルタ関数
# ============================================================
def impulse_response(kind, T, w0, z, t):
    if kind == "1次遅れ系":
        return np.exp(-t / T) / T
    if z < 1 - 1e-9:
        wd = w0 * np.sqrt(1 - z * z)
        return w0 / np.sqrt(1 - z * z) * np.exp(-z * w0 * t) * np.sin(wd * t)
    if z <= 1 + 1e-9:
        return w0 ** 2 * t * np.exp(-w0 * t)
    q = w0 * np.sqrt(z * z - 1)
    p1, p2 = -z * w0 + q, -z * w0 - q
    return w0 ** 2 / (p1 - p2) * (np.exp(p1 * t) - np.exp(p2 * t))


def tab_transfer():
    st.info("入力 r(t) と出力 y(t) の関係を $Y(s)=G(s)R(s)$ で表すとき、G(s) を**伝達関数**という。"
            "デルタ関数(瞬間的な衝撃)を入れたときの出力 g(t) = $\\mathcal L^{-1}[G]$ が**インパルス応答**で、"
            "任意の入力に対する出力は $y=g*r$ で求まる。")
    st.latex(r"\mathcal L[\delta(t)]=1,\qquad Y(s)=G(s)\cdot1\ \Rightarrow\ y(t)=g(t),\qquad y=g*r")
    c1, c2 = st.columns(2)
    kind = c1.radio("システム", ["1次遅れ系", "2次系"], key="lt_sys")
    inp = c2.radio("入力 r(t)", ["幅 ε の矩形パルス(デルタ関数の近似)", "ステップ U(t)", "正弦波 sin ωt"],
                   key="lt_in")
    d = st.columns(4)
    T = d[0].slider("時定数 T", 0.2, 3.0, 1.0, 0.1, key="lt_T", disabled=kind != "1次遅れ系")
    w0 = d[1].slider("ω₀", 0.5, 4.0, 2.0, 0.1, key="lt_w0", disabled=kind == "1次遅れ系")
    z = d[2].slider("ζ", 0.05, 2.0, 0.3, 0.05, key="lt_z", disabled=kind == "1次遅れ系")
    if inp.startswith("幅"):
        eps = d[3].slider("ε", 0.02, 3.0, 1.0, 0.02, key="lt_eps")
    elif inp.startswith("正弦"):
        om = d[3].slider("ω", 0.2, 5.0, 1.0, 0.1, key="lt_om")
    if kind == "1次遅れ系":
        st.latex(rf"G(s)=\frac{{1}}{{Ts+1}}=\frac{{1}}{{{T:g}s+1}},\qquad g(t)=\frac1Te^{{-t/T}}")
    else:
        st.latex(rf"G(s)=\frac{{\omega_0^2}}{{s^2+2\zeta\omega_0s+\omega_0^2}}=\frac{{{w0 * w0:.3g}}}"
                 rf"{{s^2+{2 * z * w0:.3g}s+{w0 * w0:.3g}}}")

    t = np.linspace(0, 15, 7501)
    dt = t[1] - t[0]
    g = impulse_response(kind, T, w0, z, t)
    if inp.startswith("幅"):
        r = np.where(t < eps, 1 / eps, 0.0)
    elif inp.startswith("ステップ"):
        r = np.ones_like(t)
    else:
        r = np.sin(om * t)
    y = np.convolve(g, r)[:len(t)] * dt
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, mode="lines", line=dict(color=C2, width=2.5), name="出力 y = g * r"))
    fig.add_trace(go.Scatter(x=t, y=g, mode="lines", line=dict(color=C1, dash="dash"), name="インパルス応答 g(t)"))
    fig.add_trace(go.Scatter(x=t, y=np.minimum(r, 3), mode="lines", line=dict(color=GRAY, dash="dot"),
                             name="入力 r(t)(3 で切って表示)"))
    layout(fig, xlab="t", height=400)
    show(fig)
    if inp.startswith("幅"):
        st.write(f"ε = {eps:g}:ε を小さくすると、出力(赤)がインパルス応答(青の破線)に近づく。")
        ss = np.linspace(0.01, 10, 300)
        f2 = go.Figure(go.Scatter(x=ss, y=(1 - np.exp(-eps * ss)) / (eps * ss), mode="lines",
                                  line=dict(color=C1), name="L[パルス]"))
        f2.add_hline(y=1, line=dict(color=C2, dash="dash"), annotation_text="L[δ] = 1")
        layout(f2, title="矩形パルスのラプラス変換 (1 − e^{−εs})/(εs) → 1", xlab="s", height=280, yrange=(0, 1.1))
        show(f2)
    elif inp.startswith("ステップ"):
        st.write("ステップ応答はインパルス応答の積分。最終値は G(0) = 1(最終値の定理 $\\lim_{s\\to0}sY(s)$)。")

    with st.expander("デルタ関数の性質"):
        st.latex(r"\delta(t)=0\ (t\ne0),\quad \int_{-\infty}^{\infty}\delta(t)\,dt=1,\quad "
                 r"\int f(t)\delta(t-a)\,dt=f(a),\quad \mathcal L[\delta(t-a)]=e^{-as}")
        st.caption("いろいろな近似(モデル関数)は「⑥ その他」で比べられる。")
    with st.expander("練習問題"):
        st.markdown("**問8** $y'+2y=r(t),\\ y(0)=0$ の伝達関数とインパルス応答を求めよ。")
        st.latex(r"(s+2)Y=R\ \Rightarrow\ G(s)=\frac{1}{s+2},\quad g(t)=e^{-2t}")
        st.markdown("**問9** 問8で $r(t)=U(t)$ のときの出力を合成積で求めよ。")
        st.latex(r"y=\int_0^te^{-2\tau}d\tau=\frac12(1-e^{-2t})")


# ============================================================
# 2-4. ラプラス変換と演算子法
# ============================================================
OPR = {
    "1": (lambda t: np.ones_like(t), r"1"),
    "t": (lambda t: t, r"t"),
    "e^{t}": (np.exp, r"e^{t}"),
    "sin 2t": (lambda t: np.sin(2 * t), r"\sin 2t"),
}


def inv_op(al, r, t):
    """1/(D − α) r = e^{αt} ∫_0^t e^{−ατ} r(τ) dτ"""
    return np.exp(al * t) * cumint(np.exp(-al * t) * r, t)


def tab_operator():
    st.info("微分を $D=\\dfrac{d}{dt}$ と書き、D を文字のように扱って方程式を解く(ヘヴィサイドの演算子法)。"
            "初期値 0 のとき、D は s、1/D は $\\int_0^t$ に対応し、ラプラス変換で正当化される。")
    st.markdown(r"""
| 演算子法 | ラプラス変換 | 意味 |
|---|---|---|
| $D$ | $s$ | 微分(初期値 0) |
| $\dfrac1D$ | $\dfrac1s$ | $\displaystyle\int_0^t$ |
| $\dfrac{1}{D-\alpha}r$ | $\dfrac{R(s)}{s-\alpha}$ | $\displaystyle e^{\alpha t}\int_0^te^{-\alpha\tau}r(\tau)\,d\tau=e^{\alpha t}*r$ |
""")
    c = st.columns(3)
    al = c[0].slider("α", -3, 2, -1, key="lo_al")
    be = c[1].slider("β", -3, 2, -2, key="lo_be")
    rn = c[2].selectbox("右辺 r(t)", list(OPR), key="lo_r")
    r, rtex = OPR[rn]
    p, q = -(al + be), al * be
    st.latex(rf"(D-({al}))(D-({be}))\,y=\left(D^2{p:+d}D{q:+d}\right)y={rtex},\qquad y(0)=y'(0)=0".replace("+0D", "")
             .replace("+0\\right", "\\right"))

    t = np.linspace(0, 3, 30001)
    rv = r(t)
    seq = inv_op(al, inv_op(be, rv, t), t)
    st.markdown("**方法1:1/(D − α) を順に作用させる**")
    st.latex(rf"y=\frac{{1}}{{D-({al})}}\left[\frac{{1}}{{D-({be})}}{rtex}\right]")
    if al != be:
        pf = (inv_op(al, rv, t) - inv_op(be, rv, t)) / (al - be)
        st.markdown("**方法2:部分分数に分ける**")
        st.latex(rf"\frac{{1}}{{(D-({al}))(D-({be}))}}=\frac{{1}}{{{al - be}}}\left(\frac{{1}}{{D-({al})}}"
                 rf"-\frac{{1}}{{D-({be})}}\right)")
    else:
        pf = None
        st.caption("α = β(重根)のときは部分分数に分けられないが、方法1はそのまま使える。")

    # 比較用:RK4
    def rhs(tt, Y):
        return np.array([Y[1], np.interp(tt, t, rv) - p * Y[1] - q * Y[0]])
    tk = np.linspace(0, 3, 3001)
    Y = np.zeros((len(tk), 2))
    for i in range(len(tk) - 1):
        h = tk[1] - tk[0]
        k1 = rhs(tk[i], Y[i])
        k2 = rhs(tk[i] + h / 2, Y[i] + h / 2 * k1)
        k3 = rhs(tk[i] + h / 2, Y[i] + h / 2 * k2)
        k4 = rhs(tk[i] + h, Y[i] + h * k3)
        Y[i + 1] = Y[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t[::30], y=seq[::30], mode="lines", line=dict(color=C1, width=4), name="方法1"))
    if pf is not None:
        fig.add_trace(go.Scatter(x=t[::30], y=pf[::30], mode="lines", line=dict(color=C2, dash="dash", width=2.5),
                                 name="方法2"))
    fig.add_trace(go.Scatter(x=tk[::30], y=Y[::30, 0], mode="markers", marker=dict(color="black", size=4),
                             name="数値解(ルンゲ・クッタ)"))
    layout(fig, xlab="t", ylab="y", height=380)
    show(fig)
    st.caption(f"t = 3 での値:方法1 = {seq[-1]:.6f}" + (f"、方法2 = {pf[-1]:.6f}" if pf is not None else "")
               + f"、数値解 = {Y[-1, 0]:.6f}")

    with st.expander("練習問題"):
        st.markdown("**問10** $(D^2+3D+2)y=1,\\ y(0)=y'(0)=0$ を演算子法で解け。")
        st.latex(r"\frac{1}{(D+1)(D+2)}1=\left(\frac{1}{D+1}-\frac{1}{D+2}\right)1=(1-e^{-t})-\frac12(1-e^{-2t})")
        st.latex(r"y=\frac12-e^{-t}+\frac12e^{-2t}")
        st.write("(上で α = −1、β = −2、r = 1 にして確かめよう)")
