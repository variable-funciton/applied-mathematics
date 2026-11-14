"""③ フーリエ解析 発展 8. ヒルベルト変換

規約:F(ω) = ∫ f(t) e^{−iωt} dt。Hf(t) = (1/π) p.v. ∫ f(s)/(t − s) ds = (1/(πt)) * f,  (Hf)^(ω) = −i sgn(ω) F(ω)。
(符号が逆の定義も使われる。)9. フーリエマルチプライヤーの導入を兼ねる。
史実・書誌は 2026 年 11 月に確認(King の本の序論, EuDML, AMS, Acta Math. など)。
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.integrate import quad
from scipy.special import dawsn

from common import C1, C2, C3, C4, GRAY, layout, show, trapz
from topics.fourier_apps import refs
from topics.fourier_kernels import page_refs

PI = np.pi
LT, NT = 200.0, 2 ** 16
TT = -LT + 2 * LT * np.arange(NT) / NT
DT = TT[1] - TT[0]
WW = 2 * PI * np.fft.fftfreq(NT, DT)


def hilbert_fft(v):
    """等間隔の格子上で (Hf)^ = −i sgn(ω) F を使って計算する"""
    return np.real(np.fft.ifft(-1j * np.sign(np.fft.fftfreq(len(v))) * np.fft.fft(v)))


def analytic(v):
    """解析信号 z = f + i Hf"""
    return v + 1j * hilbert_fft(v)


def log_ratio(t, a=-1.0, b=1.0):
    """H χ_[a,b](t) = (1/π) log |(t − a)/(t − b)|"""
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.log(np.abs((t - a) / (t - b))) / PI
    return np.where(np.isfinite(v), v, np.nan)


# 名前: (f, Hf の厳密な式 or None, f の TeX, Hf の TeX)
PAIRS = {
    "ローレンツ型 1/(1 + t²)": (lambda t: 1 / (1 + t * t), lambda t: t / (1 + t * t), r"\frac{1}{1+t^2}", r"\frac{t}{1+t^2}"),
    "ガウス関数 e^{−t²}": (lambda t: np.exp(-t * t), lambda t: 2 / np.sqrt(PI) * dawsn(t), r"e^{-t^2}", r"\frac{2}{\sqrt\pi}e^{-t^2}\int_0^te^{s^2}ds"),
    "矩形 χ_[−1,1]": (lambda t: (np.abs(t) < 1).astype(float), log_ratio, r"\chi_{[-1,1]}(t)", r"\frac1\pi\log\left|\frac{t+1}{t-1}\right|"),
    "波束 e^{−t²/8} cos 5t": (lambda t: np.exp(-t * t / 8) * np.cos(5 * t), lambda t: np.exp(-t * t / 8) * np.sin(5 * t), r"e^{-t^2/8}\cos5t",
                          r"\approx e^{-t^2/8}\sin5t"),
    "sin t / t": (lambda t: np.sinc(t / PI), lambda t: np.where(np.abs(t) < 1e-9, 0.0, (1 - np.cos(t)) / np.where(t == 0, 1, t)), r"\frac{\sin t}{t}", r"\frac{1-\cos t}{t}"),
}


def truncated_hilbert(f, t, eps, R=1e5, n=40000):
    """H_ε f(t) = (1/π) ∫_ε^R (f(t − u) − f(t + u))/u du(|s − t| < ε を除いた積分)"""
    lu = np.linspace(np.log(eps), np.log(R), n)
    u = np.exp(lu)
    return float(trapz(f(t - u) - f(t + u), lu)) / PI


def poisson_box(x, y):
    """χ_[−1,1] のポアソン積分 u と共役ポアソン積分 v(上半平面 y > 0)"""
    u = (np.arctan((x + 1) / y) - np.arctan((x - 1) / y)) / PI
    v = np.log(((x + 1) ** 2 + y * y) / ((x - 1) ** 2 + y * y)) / (2 * PI)
    return u, v


def conj_square(th, N):
    """方形波 sgn θ のフーリエ級数の部分和と, 共役級数の部分和(奇数 n ≤ N)"""
    k = np.arange(1, N + 1, 2)[:, None]
    return (4 / PI) * np.sum(np.sin(k * th) / k, axis=0), -(4 / PI) * np.sum(np.cos(k * th) / k, axis=0)


SYSTEMS = {
    "RC 回路(1次遅れ)": (lambda w, g: 1 / (1 + 1j * w / g), "ω_c", [0.5, 1.0, 2.0, 4.0], r"H(\omega)=\frac{1}{1+i\omega/\omega_c}"),
    "減衰振動子(ローレンツ模型, ω₀ = 5)": (lambda w, g: 1 / (25 - w * w + 1j * g * w), "γ", [0.5, 1.0, 2.0, 4.0], r"H(\omega)=\frac{1}{\omega_0^2-\omega^2+i\gamma\omega}"),
}


@st.cache_data(show_spinner=False)
def kk_check(name, g):
    Hw = SYSTEMS[name][0](TT, g)        # 格子 TT を ω の格子として使う
    return Hw.real, Hw.imag, -hilbert_fft(Hw.real), hilbert_fft(Hw.imag)


def best_const(p):
    """ヒルベルト変換の L^p ノルム:tan(π/2p)(1 < p ≤ 2), cot(π/2p)(p ≥ 2)"""
    return np.tan(PI / (2 * p)) if p <= 2 else 1 / np.tan(PI / (2 * p))


def box_ratio(p):
    """‖Hχ_[−1,1]‖_p / ‖χ_[−1,1]‖_p。分布関数 |{|Hχ| > λ}| = 4 / sinh(πλ) から計算"""
    val = quad(lambda lam: p * lam ** (p - 1) * 8 * np.exp(-PI * lam) / -np.expm1(-2 * PI * lam), 0, 30, limit=200)[0]
    return (val / 2) ** (1 / p)


@st.cache_data(show_spinner=False)
def bearing(fd, noise, seed=1):
    """軸受の傷を模した信号:周期 1/fd の衝撃が共振(600 Hz)を励起 + 雑音。包絡線とそのスペクトル"""
    n, T = 8192, 1.0
    t = np.arange(n) * T / n
    x = np.zeros(n)
    for tk in np.arange(0.003, T, 1 / fd):
        s = np.clip(t - tk, 0, None)
        x += np.where(t >= tk, np.exp(-300 * s) * np.sin(2 * PI * 600 * s), 0.0)
    x += noise * np.random.default_rng(seed).standard_normal(n)
    X = np.fft.fft(x)
    h = np.zeros(n)
    h[0] = h[n // 2] = 1
    h[1:n // 2] = 2
    env = np.abs(np.fft.ifft(X * h))
    fr = np.fft.rfftfreq(n, T / n)
    return t, x, env, fr, np.abs(np.fft.rfft(env - env.mean())) / n * 2, np.abs(np.fft.rfft(x)) / n * 2


def ssb(kind):
    """メッセージ m(t) を搬送波 ω_c = 20 にのせる。DSB:m cos ω_c t, USB/LSB:m cos ω_c t ∓ Hm sin ω_c t"""
    win = np.exp(-(TT / 40) ** 2)
    m = (np.cos(2 * TT) + 0.5 * np.cos(3 * TT)) * win
    Hm = hilbert_fft(m)
    c, s = np.cos(20 * TT), np.sin(20 * TT)
    sig = m * c if kind == "DSB(両側波帯)" else (m * c - Hm * s if kind.startswith("USB") else m * c + Hm * s)
    return m, sig, np.abs(np.fft.fft(sig)) * DT


def tab_hilbert():
    st.info("**ヒルベルト変換**は, 信号のすべての周波数成分の位相を $90^\\circ$ ずらす変換である。"
            "$\\cos$ を $\\sin$ に変えるだけの単純な操作に見えるが, 信号の**包絡線**や**瞬時周波数**の取り出し, 正則関数の実部と虚部の関係, "
            "因果的な系の周波数応答(クラマース・クローニッヒの関係式)など, 多くの場面に現れる。次の 9. フーリエマルチプライヤーのいちばん大切な例でもある。")

    # ---------------------------------------------------------------- 1
    st.subheader("1. 定義:主値積分")
    st.latex(r"Hf(t)=\frac1\pi\,\mathrm{p.v.}\!\int_{-\infty}^{\infty}\frac{f(s)}{t-s}\,ds=\lim_{\varepsilon\to+0}\frac1\pi\int_{|s-t|>\varepsilon}\frac{f(s)}{t-s}\,ds")
    st.markdown("$\\dfrac{1}{\\pi t}$ との合成積 $Hf=\\dfrac{1}{\\pi t}*f$ である。$\\dfrac1t$ は $t=0$ のまわりで積分できない($\\int_0^1\\frac{dt}{t}=\\infty$)ので, "
                "特異点の左右を**対称に**切り取ってから極限をとる(コーシーの**主値**, p.v.)。$s=t\\mp u$ と置くと")
    st.latex(r"Hf(t)=\frac1\pi\int_0^{\infty}\frac{f(t-u)-f(t+u)}{u}\,du")
    st.markdown("となり, 左右の値の**差**が $u\\to0$ で小さくなるので積分が収束する。正の無限大と負の無限大が打ち消し合っている。")
    c = st.columns([1, 2])
    fname = c[0].selectbox("関数 f", list(PAIRS)[:3], key="hb_f1")
    t0 = c[0].select_slider("点 t", [-2.0, -0.5, 0.0, 0.5, 1.5, 3.0], 0.5, key="hb_t0")
    eps = c[0].select_slider("切り取る幅 ε", [1.0, 0.5, 0.2, 0.1, 0.02, 0.001], 0.2, key="hb_eps")
    f, Hex, ftex, Htex = PAIRS[fname]
    c[0].latex(f"f(t)={ftex},\\quad Hf(t)={Htex}")
    c[0].metric("ε で切り取った積分", f"{truncated_hilbert(f, t0, eps):.5f}")
    c[0].metric("Hf(t) の厳密な値", f"{float(Hex(np.array([t0]))[0]):.5f}")
    s = np.linspace(t0 - 4, t0 + 4, 4001)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = f(s) / (PI * (t0 - s))
    g = np.where(np.abs(s - t0) > eps, g, np.nan)
    fig = go.Figure(go.Scatter(x=s, y=np.where(g > 0, g, 0), fill="tozeroy", line=dict(color=C1, width=0), name="正の部分"))
    fig.add_trace(go.Scatter(x=s, y=np.where(g < 0, g, 0), fill="tozeroy", line=dict(color=C2, width=0), name="負の部分"))
    fig.add_trace(go.Scatter(x=s, y=g, line=dict(color="black", width=1.5), name="被積分関数 f(s) / (π(t − s))"))
    fig.add_vrect(x0=t0 - eps, x1=t0 + eps, fillcolor="rgba(120,120,120,0.25)", line_width=0)
    layout(fig, "被積分関数(灰色の帯 |s − t| < ε を除く)", xlab="s", height=360, yrange=[-3, 3])
    with c[1]:
        show(fig)

    # ---------------------------------------------------------------- 2
    st.subheader("2. フーリエ変換で見る:位相を $90^\\circ$ ずらす")
    st.markdown("$\\dfrac{1}{\\pi t}$ のフーリエ変換は $-i\\,\\mathrm{sgn}\\,\\omega$ である。合成積のフーリエ変換は積になるので")
    st.latex(r"\widehat{Hf}(\omega)=-i\,\mathrm{sgn}(\omega)\,\hat f(\omega)=\begin{cases}e^{-i\pi/2}\hat f(\omega)&(\omega>0)\\ e^{+i\pi/2}\hat f(\omega)&(\omega<0)\end{cases}")
    st.markdown("**振幅は変えずに, 位相だけを $90^\\circ$ ずらす**。$e^{iat}$($a>0$)は $-ie^{iat}$ に移るので, $\\cos at\\mapsto\\sin at$, $\\sin at\\mapsto-\\cos at$。")
    st.markdown(r"""
| $f(t)$ | $Hf(t)$ | | 性質 | |
|---|---|---|---|---|
| $\cos at\ (a>0)$ | $\sin at$ | | 2回行うと符号が変わる | $H(Hf)=-f$ |
| $\sin at\ (a>0)$ | $-\cos at$ | | 大きさを保つ | $\lVert Hf\rVert_2=\lVert f\rVert_2$ |
| $\dfrac{a}{t^2+a^2}\ (a>0)$ | $\dfrac{t}{t^2+a^2}$ | | 直交する | $\int f\,Hf\,dt=0$ |
| $\chi_{[a,b]}(t)$ | $\dfrac1\pi\log\left\lvert\dfrac{t-a}{t-b}\right\rvert$ | | 微分・平行移動と交換する | $H(f^{\prime})=(Hf)^{\prime}$ |
| $\dfrac{\sin t}{t}$ | $\dfrac{1-\cos t}{t}$ | | 偶関数 ↔ 奇関数 | |
""")
    c = st.columns([1, 2])
    fname = c[0].selectbox("関数 f", list(PAIRS), index=3, key="hb_f2")
    f, Hex, ftex, Htex = PAIRS[fname]
    fv = f(TT)
    Hv = hilbert_fft(fv)
    n2 = float(np.sum(fv ** 2))
    c[0].latex(f"f(t)={ftex}")
    c[0].latex(f"Hf(t)={Htex}")
    c[0].metric("‖Hf‖₂ / ‖f‖₂", f"{float(np.sqrt(np.sum(Hv ** 2) / n2)):.4f}")
    c[0].metric("∫ f Hf dt / ‖f‖₂²", f"{float(np.sum(fv * Hv) / n2):.4f}")
    sel = np.abs(TT) < 8
    fig = go.Figure(go.Scatter(x=TT[sel], y=fv[sel], name="f", line=dict(color=GRAY, width=3)))
    fig.add_trace(go.Scatter(x=TT[sel], y=Hv[sel], name="Hf(FFT で計算)", line=dict(color=C1)))
    fig.add_trace(go.Scatter(x=TT[sel], y=Hex(TT[sel]), name="Hf(式)", line=dict(color=C2, dash="dot")))
    layout(fig, "f とヒルベルト変換 Hf", xlab="t", height=360, yrange=[-1.6, 1.6])
    with c[1]:
        show(fig)
    st.caption("矩形のように $\\int f\\,dt\\neq0$ の関数では, 遠くで $Hf(t)\\approx\\dfrac{1}{\\pi t}\\int f\\,ds$ とゆっくりしか減らない。"
               "波束では $Hf$ は「$\\cos$ を $\\sin$ に変えたもの」にほぼ等しい(包絡線がゆっくり変わるとき:ベドロシアンの定理)。")

    # ---------------------------------------------------------------- 3
    st.subheader("3. 解析信号:包絡線と瞬時周波数")
    st.latex(r"z(t)=f(t)+i\,Hf(t),\qquad \hat z(\omega)=\bigl(1+\mathrm{sgn}\,\omega\bigr)\hat f(\omega)=\begin{cases}2\hat f(\omega)&(\omega>0)\\0&(\omega<0)\end{cases}")
    st.markdown("$z$ を**解析信号**という。負の周波数成分を消して正の周波数だけにした複素数値の信号で, $f=\\mathrm{Re}\\,z$。極形式 $z(t)=A(t)e^{i\\phi(t)}$ で")
    st.markdown("- $A(t)=|z(t)|=\\sqrt{f^2+(Hf)^2}$:**包絡線**(振幅のゆっくりした変化)\n"
                "- $\\phi^{\\prime}(t)$:**瞬時周波数**(その時刻での角周波数)\n\n"
                "が取り出せる。$f=a(t)\\cos\\omega_0t$ なら $z\\approx a(t)e^{i\\omega_0t}$ で, $A=|a|$, $\\phi^{\\prime}=\\omega_0$。")
    c = st.columns([1, 2])
    kind = c[0].radio("信号", ["振幅変調(AM)", "周波数が変わる信号(チャープ)"], key="hb_sig")
    w0 = c[0].select_slider("搬送波の角周波数 ω₀", [10, 20, 40], 20, key="hb_w0")
    if kind.startswith("振幅"):
        m = c[0].select_slider("変調の深さ m", [0.0, 0.3, 0.6, 0.9], 0.6, key="hb_m")
        amp = (1 + m * np.cos(1.5 * TT)) * np.exp(-(TT / 12) ** 2)
        sig, wtrue = amp * np.cos(w0 * TT), w0 + 0 * TT
        c[0].latex(r"f(t)=(1+m\cos1.5t)\,e^{-(t/12)^2}\cos\omega_0t")
    else:
        beta = c[0].select_slider("周波数の変化率 β", [0.5, 1.0, 2.0], 1.0, key="hb_beta")
        amp = np.exp(-(TT / 8) ** 2)
        sig, wtrue = amp * np.cos(w0 * TT + beta * TT ** 2 / 2), w0 + beta * TT
        c[0].latex(r"f(t)=e^{-(t/8)^2}\cos\!\left(\omega_0t+\tfrac{\beta}{2}t^2\right)")
    z = analytic(sig)
    env = np.abs(z)
    winst = np.gradient(np.unwrap(np.angle(z)), DT)
    sel = np.abs(TT) < 14
    good = sel & (env > 0.05 * env.max())
    c[0].metric("包絡線の誤差(最大)", f"{float(np.max(np.abs(env[sel] - amp[sel]))):.4f}")
    fig = go.Figure(go.Scatter(x=TT[sel], y=sig[sel], name="f(t)", line=dict(color=GRAY, width=1)))
    fig.add_trace(go.Scatter(x=TT[sel], y=env[sel], name="包絡線 |z(t)|", line=dict(color=C2, width=2.5)))
    fig.add_trace(go.Scatter(x=TT[sel], y=-env[sel], showlegend=False, line=dict(color=C2, width=2.5)))
    layout(fig, "信号と, 解析信号から求めた包絡線", xlab="t", height=300)
    fig2 = go.Figure(go.Scatter(x=TT[good], y=winst[good], name="瞬時周波数 φ′(t)", line=dict(color=C1, width=2.5)))
    fig2.add_trace(go.Scatter(x=TT[good], y=wtrue[good], name="理論値", line=dict(color="black", dash="dot")))
    layout(fig2, "瞬時周波数", xlab="t", ylab="角周波数", height=260, yrange=[0, 70])
    with c[1]:
        show(fig)
        show(fig2)

    # ---------------------------------------------------------------- 4
    st.subheader("4. 複素関数とのつながり:実部と虚部")
    st.markdown("上半平面 $\\mathrm{Im}\\,z>0$ で正則で, 遠くで $0$ になる関数 $F(z)$ の実軸上の値を $F(t)=u(t)+iv(t)$ とすると")
    st.latex(r"v=Hu,\qquad u=-Hv")
    st.markdown("が成り立つ(コーシーの積分公式から)。**正則関数の実部と虚部は, 境界の上で互いにヒルベルト変換で結ばれる**。"
                "例えば $F(z)=\\dfrac{i}{z+i}$ は $F(t)=\\dfrac{1}{1+t^2}+i\\dfrac{t}{1+t^2}$ で, 2. の表の変換対になっている。"
                "解析信号 $z(t)=f+iHf$ は, 上半平面に正則に延長できる関数の境界値である(これが「解析」信号の名前の由来)。")
    st.markdown("**ポアソン核と共役ポアソン核**  $f$ を上半平面に調和関数として延長すると")
    st.latex(r"u(x,y)=\frac1\pi\int\frac{y}{(x-s)^2+y^2}f(s)\,ds,\qquad v(x,y)=\frac1\pi\int\frac{x-s}{(x-s)^2+y^2}f(s)\,ds")
    st.markdown("で, $u+iv$ は正則になる。$y\\to+0$ のとき $u\\to f$, $v\\to Hf$。共役ポアソン核 $\\dfrac{1}{\\pi}\\dfrac{x}{x^2+y^2}$ は, $\\dfrac{1}{\\pi x}$ の特異点を $y$ でならしたものである。")
    c = st.columns([1, 2])
    y = c[0].select_slider("高さ y", [2.0, 1.0, 0.5, 0.2, 0.05, 0.01], 0.5, key="hb_y")
    x = np.linspace(-4, 4, 1601)
    uu, vv = poisson_box(x, y)
    c[0].latex(r"f=\chi_{[-1,1]}")
    c[0].caption("$y$ を小さくすると, $u$ は矩形に, $v$ は $\\frac1\\pi\\log\\left|\\frac{x+1}{x-1}\\right|$ に近づく。")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=(np.abs(x) < 1).astype(float), name="f", line=dict(color=GRAY, width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=x, y=log_ratio(x), name="Hf", line=dict(color="black", width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=x, y=uu, name="u(x, y):ポアソン積分", line=dict(color=C1, width=2.5)))
    fig.add_trace(go.Scatter(x=x, y=vv, name="v(x, y):共役ポアソン積分", line=dict(color=C2, width=2.5)))
    layout(fig, f"y = {y} での調和関数 u と共役調和関数 v", xlab="x", height=360, yrange=[-2.2, 2.2])
    with c[1]:
        show(fig)
    st.markdown("**周期関数の場合:共役関数**  フーリエ級数 $f\\sim\\dfrac{a_0}2+\\sum(a_n\\cos n\\theta+b_n\\sin n\\theta)$ に対して")
    st.latex(r"\tilde f\sim\sum_{n=1}^{\infty}(a_n\sin n\theta-b_n\cos n\theta)=\frac{1}{2\pi}\,\mathrm{p.v.}\!\int_{-\pi}^{\pi}f(\theta-\varphi)\cot\frac{\varphi}{2}\,d\varphi")
    st.markdown("を**共役関数**(共役級数)という。複素フーリエ係数に $-i\\,\\mathrm{sgn}\\,n$ を掛ける操作で, ヒルベルト変換の周期版である"
                "(第1章「7. ポアソン核とアーベル総和法」の単位円板での話に対応する)。")
    st.markdown("> **定理(M. リース, 1928)** $1<p<\\infty$ のとき $\\lVert\\tilde f\\rVert_p\\le C_p\\lVert f\\rVert_p$。"
                "その結果, フーリエ級数の部分和 $S_Nf$ は $L^p$ で $f$ に収束する。")
    st.markdown("部分和 $S_N$ は「周波数 $|n|\\le N$ だけを残す」操作で, 「$n\\ge0$ だけを残す」操作 $\\frac12(I+i\\,\\tilde{\\ })$ と $e^{\\pm iN\\theta}$ の掛け算の組合せで書けるからである。")
    c = st.columns([1, 2])
    Nc = c[0].select_slider("部分和の項(n ≤ N)", [1, 3, 7, 15, 41, 101], 15, key="hb_N")
    th = np.linspace(-PI, PI, 2001)
    sN, cN = conj_square(th, Nc)
    with np.errstate(divide="ignore"):
        exact = 2 / PI * np.log(np.abs(np.tan(th / 2)))
    c[0].latex(r"f=\mathrm{sgn}\,\theta\ \Longrightarrow\ \tilde f=\frac2\pi\log\left|\tan\frac\theta2\right|")
    c[0].caption("方形波(有界)の共役関数は, とびの位置 $\\theta=0,\\pm\\pi$ で対数的に発散する。")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=th, y=np.sign(th), name="f = sgn θ", line=dict(color=GRAY, width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=th, y=np.where(np.isfinite(exact), exact, np.nan), name="共役関数", line=dict(color="black", width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=th, y=sN, name="部分和 S_N f", line=dict(color=C1)))
    fig.add_trace(go.Scatter(x=th, y=cN, name="共役級数の部分和", line=dict(color=C2)))
    layout(fig, "方形波とその共役関数", xlab="θ", height=340, yrange=[-2.5, 2.5])
    with c[1]:
        show(fig)

    # ---------------------------------------------------------------- 5
    st.subheader("5. 因果性とクラマース・クローニッヒの関係式")
    st.markdown("入力より前に出力が出ない系(**因果的**な系)では, インパルス応答が $h(t)=0$($t<0$)をみたす。"
                "$h$ を偶関数の部分 $h_e$ と奇関数の部分 $h_o$ に分けると $h_o(t)=\\mathrm{sgn}(t)\\,h_e(t)$。周波数応答を $H(\\omega)=X(\\omega)+iY(\\omega)$ とすると, "
                "$\\hat h_e=X$, $\\hat h_o=iY$ で, $\\mathrm{sgn}(t)$ を掛けることは周波数側ではヒルベルト変換になるので")
    st.latex(r"Y(\omega)=-\frac1\pi\,\mathrm{p.v.}\!\int\frac{X(\omega^{\prime})}{\omega-\omega^{\prime}}\,d\omega^{\prime}=-\mathcal H[X](\omega),\qquad X(\omega)=\mathcal H[Y](\omega)")
    st.markdown("**因果的な系では, 周波数応答の実部と虚部は独立に選べない**。一方を測れば他方が決まる。物理・化学では, 屈折率(分散)と吸収係数の関係として"
                "**クラマース・クローニッヒの関係式**と呼ばれる(符号はフーリエ変換の規約による)。")
    c = st.columns([1, 2])
    sysname = c[0].selectbox("系", list(SYSTEMS), key="hb_sys")
    _, gname, gvals, gtex = SYSTEMS[sysname]
    g = c[0].select_slider(gname, gvals, gvals[1], key="hb_g")
    X, Y, Yk, Xk = kk_check(sysname, g)
    sel = np.abs(TT) < 12
    c[0].latex(gtex)
    c[0].metric("虚部 Y と −H[X] の差(最大)", f"{float(np.max(np.abs(Y[sel] - Yk[sel]))):.4f}")
    c[0].metric("実部 X と H[Y] の差(最大)", f"{float(np.max(np.abs(X[sel] - Xk[sel]))):.4f}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=TT[sel], y=X[sel], name="実部 X(ω)", line=dict(color=C1, width=3)))
    fig.add_trace(go.Scatter(x=TT[sel], y=Y[sel], name="虚部 Y(ω)", line=dict(color=C2, width=3)))
    fig.add_trace(go.Scatter(x=TT[sel], y=Yk[sel], name="−H[X](実部から計算した虚部)", line=dict(color="black", dash="dot")))
    layout(fig, "周波数応答の実部・虚部とヒルベルト変換", xlab="ω", height=360)
    with c[1]:
        show(fig)
    st.caption("減衰振動子では, 虚部(吸収)が共鳴 $\\omega=\\omega_0$ の近くで山になり, 実部(分散)はそこで符号を変える。分光のスペクトルで見られる形である。")

    # ---------------------------------------------------------------- 6
    st.subheader("6. $L^p$ での有界性(発展)")
    st.markdown("$|{-i}\\,\\mathrm{sgn}\\,\\omega|=1$ なので $L^2$ では $\\lVert Hf\\rVert_2=\\lVert f\\rVert_2$(プランシュレルの等式)。$p\\neq2$ ではどうなるか。")
    st.markdown("> **定理(M. リース, 1928)** $1<p<\\infty$ のとき $\\lVert Hf\\rVert_p\\le C_p\\lVert f\\rVert_p$。\n>\n"
                "> 最良の定数は $C_p=\\tan\\dfrac{\\pi}{2p}$($1<p\\le2$), $\\cot\\dfrac{\\pi}{2p}$($2\\le p<\\infty$)(ピホリデス, 1972)。")
    st.markdown("- **$p=\\infty$ では有界でない**:$f=\\chi_{[-1,1]}$ は $|f|\\le1$ だが, $Hf=\\frac1\\pi\\log\\left|\\frac{t+1}{t-1}\\right|$ は $t=\\pm1$ で発散する。\n"
                "- **$p=1$ でも有界でない**:同じ $f$ で, 遠くで $Hf(t)\\approx\\dfrac{2}{\\pi t}$ となり積分できない。\n"
                "- **$p=1$ の代わり(弱型の評価)**:$Hf$ が大きい点の集まりは小さい。")
    st.latex(r"\bigl|\{t:\ |Hf(t)|>\lambda\}\bigr|\le\frac{C}{\lambda}\lVert f\rVert_1\qquad(\lambda>0)")
    st.markdown("共役関数についてコルモゴロフ(1925)が示した。矩形 $f=\\chi_{[-1,1]}$ では左辺が**厳密に** $\\dfrac{4}{\\sinh\\pi\\lambda}$ と計算できる"
                "(一般に, 長さの合計が $|E|$ の集合 $E$ について $\\dfrac{2|E|}{\\sinh\\pi\\lambda}$ となり, $E$ の形によらない:スタイン・ワイスの公式)。")
    c = st.columns([1, 1])
    ps = np.linspace(1.05, 12, 400)
    fig = go.Figure(go.Scatter(x=ps, y=[best_const(p) for p in ps], name="最良の定数 C_p", line=dict(color=C1, width=3)))
    pl = [1.1, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0]
    fig.add_trace(go.Scatter(x=pl, y=[box_ratio(p) for p in pl], mode="markers", marker=dict(size=10, color=C2), name="矩形での比 ‖Hf‖_p / ‖f‖_p"))
    layout(fig, "L^p でのノルム:p → 1, ∞ で発散する", xlab="p", ylab="比", height=340, yrange=[0, 8])
    lam = np.linspace(0.01, 3, 300)
    fig2 = go.Figure(go.Scatter(x=lam, y=lam * 4 / np.sinh(PI * lam) / 2, name="λ |{|Hf| > λ}| / ‖f‖₁", line=dict(color=C1, width=3)))
    fig2.add_hline(y=2 / PI, line=dict(color=GRAY, dash="dot"), annotation_text="2/π")
    layout(fig2, "弱型の評価(矩形):λ を変えても有界", xlab="λ", height=340, yrange=[0, 1])
    with c[0]:
        show(fig)
    with c[1]:
        show(fig2)
    st.markdown("ヒルベルト変換は**特異積分作用素**の原型である。核 $\\dfrac1t$ の「大きさは積分できないが, 正負の打ち消しがある」という性質を高次元に一般化したのが"
                "カルデロン・ジグムントの理論(1952)で, 重みつきの評価($A_p$ 条件, ⑦ 特殊関数「9. 直交関数系どうしの移植」)や, 次の 9. のマルチプライヤー定理につながる。")

    # ---------------------------------------------------------------- 7
    st.subheader("7. 応用")
    st.markdown("**機械:包絡線による軸受の診断**  軸受の転動面に傷があると, 玉が傷を通るたびに衝撃が起き, 機械の共振(高い周波数)を励起する。"
                "波形そのもののスペクトルでは共振の周波数しか見えないが, **包絡線のスペクトル**には衝撃の繰り返しの周波数(傷の場所で決まる)が現れる。")
    c = st.columns([1, 2])
    fd = c[0].select_slider("衝撃の繰り返し(Hz)", [40, 60, 87, 120], 87, key="hb_fd")
    nz = c[0].select_slider("雑音の大きさ", [0.0, 0.2, 0.5, 1.0], 0.2, key="hb_nz")
    t, xsig, env, fr, Es, Xs = bearing(fd, nz)
    lowband = (fr > 10) & (fr < 400)
    c[0].metric("包絡線のスペクトルの最大の山", f"{float(fr[lowband][np.argmax(Es[lowband])]):.0f} Hz")
    c[0].metric("波形のスペクトルの最大の山", f"{float(fr[np.argmax(Xs)]):.0f} Hz")
    sel = t < 0.08
    fig = go.Figure(go.Scatter(x=t[sel], y=xsig[sel], name="振動の波形", line=dict(color=GRAY, width=1)))
    fig.add_trace(go.Scatter(x=t[sel], y=env[sel], name="包絡線", line=dict(color=C2, width=2.5)))
    layout(fig, "振動の波形と包絡線", xlab="時間(s)", height=280)
    fig2 = go.Figure(go.Scatter(x=fr[fr < 400], y=Es[fr < 400], name="包絡線のスペクトル", line=dict(color=C1)))
    layout(fig2, "包絡線のスペクトル", xlab="周波数(Hz)", height=260)
    with c[1]:
        show(fig)
        show(fig2)
    st.markdown("**電気電子:単側波帯(SSB)変調**  メッセージ $m(t)$ を搬送波にのせる $m(t)\\cos\\omega_ct$ のスペクトルは, $\\omega_c$ の上下に同じ情報が2つ並ぶ(両側波帯)。")
    st.latex(r"s(t)=m(t)\cos\omega_ct\mp Hm(t)\sin\omega_ct=\mathrm{Re}\bigl[(m\pm iHm)\,e^{i\omega_ct}\bigr]")
    st.markdown("とすると, 解析信号の性質から片側だけが残り, 必要な帯域が半分になる。")
    c = st.columns([1, 2])
    kind = c[0].radio("方式", ["DSB(両側波帯)", "USB(上側波帯):m cos − Hm sin", "LSB(下側波帯):m cos + Hm sin"], key="hb_ssb")
    msg, sig, S = ssb(kind)
    c[0].latex(r"m(t)=\cos2t+\tfrac12\cos3t,\quad\omega_c=20")
    wpos = np.fft.fftshift(WW)
    Sp = np.fft.fftshift(S)
    sel = (wpos > 12) & (wpos < 28)
    fig = go.Figure(go.Scatter(x=wpos[sel], y=Sp[sel], fill="tozeroy", line=dict(color=C1), name="|S(ω)|"))
    fig.add_vline(x=20, line=dict(color=GRAY, dash="dot"), annotation_text="ω_c")
    layout(fig, "変調した信号のスペクトル(ω > 0 の側)", xlab="ω", height=320)
    with c[1]:
        show(fig)
    st.markdown(r"""
| 学科 | 使われ方 |
|---|---|
| 機械 | 振動波形の包絡線解析(軸受・歯車の異常診断), 振動の減衰率の推定 |
| 電気電子 | 解析信号, 単側波帯変調, 直交復調(I/Q 信号), 最小位相フィルタ(利得と位相の関係) |
| 化学 | 分光:吸収スペクトルから屈折率(分散)を求める(クラマース・クローニッヒの関係式) |
| 都市 | 地震波・構造物の振動記録の包絡線と瞬時周波数(揺れの強さと卓越周期の時間変化) |
""")

    # ---------------------------------------------------------------- 8
    st.subheader("8. 歴史")
    st.markdown(r"""
| 年 | できごと |
|---|---|
| 1904〜1905 ごろ | ヒルベルト(D. Hilbert)が, 正則関数の実部と虚部を結ぶ積分方程式(リーマンの問題)の研究の中で扱う |
| 1924 | ハーディ(G. H. Hardy)が「ヒルベルト変換」と名づける |
| 1925 | コルモゴロフ(A. N. Kolmogorov)が共役関数の弱型の評価を示す |
| 1926〜1927 | クラマース(H. A. Kramers)とクローニッヒ(R. de L. Kronig)が独立に, 屈折率と吸収の関係式を導く |
| 1928 | M. リース(M. Riesz)が $1<p<\infty$ での $L^p$ 有界性を示す |
| 1946 | ガボール(D. Gabor)が通信の理論で解析信号を導入 |
| 1952 | カルデロンとジグムント(A. P. Calderón, A. Zygmund)が高次元の特異積分の理論を作る |
| 1972〜1974 | ピホリデスが $L^p$ の最良定数, デイヴィス(B. Davis)が弱型の最良定数を求める |
""")

    with st.expander("練習問題"):
        st.markdown("**問1** $H[3\\cos2t-\\sin5t]$ を求めよ。")
        st.markdown("**問2** $f(t)=\\dfrac{1}{t^2+4}$ のヒルベルト変換を, 表の $\\dfrac{a}{t^2+a^2}\\mapsto\\dfrac{t}{t^2+a^2}$ から求めよ。")
        st.markdown("**問3** $f(t)=3\\cos\\omega t+4\\sin\\omega t$($\\omega>0$)の解析信号 $z(t)$ と包絡線 $|z(t)|$ を求めよ。")
        st.markdown("**問4** $H\\chi_{[-1,1]}(3)$ を求めよ。")
        st.markdown("**問5** $h(t)=e^{-t}$($t\\ge0$), $0$($t<0$)の周波数応答 $H(\\omega)=\\dfrac{1}{1+i\\omega}$ の実部 $X$ と虚部 $Y$ を求め, $Y=-\\mathcal H[X]$ を表を使って確かめよ。")
        st.markdown("**問6** $H(Hf)=-f$ を, フーリエ変換を使って説明せよ。")
        if st.toggle("解答・解説を表示", key="pr_hb"):
            st.markdown("**答1** $3\\sin2t+\\cos5t$。")
            st.markdown("**答2** $a=2$ で $\\dfrac{1}{t^2+4}=\\dfrac12\\cdot\\dfrac{2}{t^2+4}$ より $Hf=\\dfrac{t}{2(t^2+4)}$。")
            st.markdown("**答3** $Hf=3\\sin\\omega t-4\\cos\\omega t$。$z=f+iHf=(3-4i)e^{i\\omega t}$, $|z|=\\sqrt{3^2+4^2}=5$(一定)。")
            st.markdown("**答4** $\\dfrac1\\pi\\log\\left|\\dfrac{3+1}{3-1}\\right|=\\dfrac{\\log2}{\\pi}$。")
            st.markdown("**答5** $X=\\dfrac{1}{1+\\omega^2}$, $Y=-\\dfrac{\\omega}{1+\\omega^2}$。表($a=1$)から $\\mathcal H[X]=\\dfrac{\\omega}{1+\\omega^2}$ なので $Y=-\\mathcal H[X]$。")
            st.markdown("**答6** $\\widehat{H(Hf)}=(-i\\,\\mathrm{sgn}\\,\\omega)^2\\hat f=-\\hat f$($\\omega\\neq0$)。")

    page_refs([("king2009", "ヒルベルト変換の専門書(歴史・公式・応用)。"), ("riesz1928", "共役関数・ヒルベルト変換の Lᵖ 有界性。"),
               ("pichorides1972", "Lᵖ ノルムの最良定数。"), ("kolmogorov1925", "共役関数の弱型の評価。"), ("davis1974", "弱型の評価の最良定数。"), ("calderonzygmund1952", "高次元の特異積分。"),
               ("kronig1926", "屈折率と吸収の関係式(クラマース・クローニッヒ)。"), ("gabor1946", "解析信号。"),
               ("stein1970", "特異積分の教科書。"), ("grafakos2014", "ヒルベルト変換の Lᵖ 理論と最良定数。"), ("zygmund2002", "共役関数・共役級数。")])
    refs([("Hilbert Transforms(F. W. King)第1巻の序論(Cambridge University Press)", "https://assets.cambridge.org/97805218/87625/excerpt/9780521887625_excerpt.pdf",
           "ヒルベルト変換の歴史(Hilbert, Hardy, Kramers と Kronig)(英語)"),
          ("Kramers–Kronig Relations(RP Photonics Encyclopedia)", "https://www.rp-photonics.com/kramers_kronig_relations.html",
           "光学でのクラマース・クローニッヒの関係式の解説(英語)"),
          ("On the weak type (1, 1) inequality for conjugate functions(Proc. Amer. Math. Soc. 1974)", "https://www.ams.org/proc/1974-044-02/S0002-9939-1974-0348381-6/",
           "コルモゴロフの弱型の評価の最良定数(英語)")])
