"""③ フーリエ解析 発展 8. フーリエマルチプライヤー(1次元)

規約:F(ω) = ∫ f(t) e^{−iωt} dt,  f(t) = (1/2π) ∫ F(ω) e^{iωt} dω。
T_m f = F^{-1}[m F] = K * f,  K = F^{-1} m(核)。
リースポテンシャル(1次元, 0 < α < 1):I_α f(t) = c_α ∫ |t − s|^{α−1} f(s) ds,  (I_α f)^ = |ω|^{−α} F,
c_α = 1 / (2 Γ(α) cos(πα/2))(∫ |t|^{α−1} e^{−iωt} dt = 2Γ(α) cos(πα/2) |ω|^{−α} から)。
"""
import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.special import gamma

from common import C1, C2, C3, C4, GRAY, layout, show
from topics.fourier_kernels import page_refs

PI = np.pi
L = 40.0
NT = 2 ** 14
TT = -L + 2 * L * np.arange(NT) / NT
DT = TT[1] - TT[0]
WW = 2 * PI * np.fft.fftfreq(NT, DT)

SIGS = {
    "矩形パルス(幅 2)": lambda t: (np.abs(t) < 1).astype(float),
    "ガウス関数 e^{−t²}": lambda t: np.exp(-t ** 2),
    "三角パルス (1 − |t|)₊": lambda t: np.maximum(1 - np.abs(t), 0.0),
    "波束 e^{−t²/4} cos 4t": lambda t: np.exp(-t ** 2 / 4) * np.cos(4 * t),
}

# 名前: (m(ω, p), パラメータ名, 範囲(最小, 最大, 初期, 刻み), m の式, 核の式, 核の関数 or None, sup|m|)
MULTS = {
    "理想ローパスフィルタ": (lambda w, p: (np.abs(w) <= p).astype(float), "Ω", (0.5, 20.0, 4.0, 0.5),
                    r"m(\omega)=\begin{cases}1&(|\omega|\le\Omega)\\0&(|\omega|>\Omega)\end{cases}",
                    r"K(t)=\frac{\sin\Omega t}{\pi t}", lambda t, p: np.where(np.abs(t) < 1e-9, p / PI, np.sin(p * t) / (PI * np.where(t == 0, 1, t))),
                    lambda p: 1.0),
    "なめらかなローパスフィルタ": (lambda w, p: np.exp(-w ** 2 / (2 * p ** 2)), "Ω", (0.5, 20.0, 4.0, 0.5),
                      r"m(\omega)=e^{-\omega^2/(2\Omega^2)}", r"K(t)=\frac{\Omega}{\sqrt{2\pi}}e^{-\Omega^2t^2/2}",
                      lambda t, p: p / math.sqrt(2 * PI) * np.exp(-(p * t) ** 2 / 2), lambda p: 1.0),
    "RC 回路(1次のローパス)": (lambda w, p: 1 / (1 + 1j * w / p), "ω_c", (0.5, 20.0, 2.0, 0.5),
                       r"m(\omega)=\frac{1}{1+i\omega/\omega_c}", r"K(t)=\omega_ce^{-\omega_ct}u(t)",
                       lambda t, p: np.where(t >= 0, p * np.exp(-p * np.maximum(t, 0)), 0.0), lambda p: 1.0),
    "時間推移": (lambda w, p: np.exp(-1j * w * p), "a", (-3.0, 3.0, 1.5, 0.5),
             r"m(\omega)=e^{-i\omega a}", r"K=\delta(t-a)\quad(T_mf(t)=f(t-a))", None, lambda p: 1.0),
    "ヒルベルト変換": (lambda w, p: -1j * np.sign(w), None, None, r"m(\omega)=-i\,\mathrm{sgn}\,\omega",
                r"K(t)=\frac{1}{\pi t}\quad(\text{主値})", lambda t, p: np.where(np.abs(t) < 0.02, np.nan, 1 / (PI * np.where(t == 0, 1, t))),
                lambda p: 1.0),
    "熱半群(熱方程式の解)": (lambda w, p: np.exp(-p * w ** 2), "時間 s", (0.01, 2.0, 0.2, 0.01),
                     r"m(\omega)=e^{-s\omega^2}", r"K(t)=\frac{1}{\sqrt{4\pi s}}e^{-t^2/(4s)}",
                     lambda t, p: np.exp(-t ** 2 / (4 * p)) / math.sqrt(4 * PI * p), lambda p: 1.0),
    "微分": (lambda w, p: 1j * w, None, None, r"m(\omega)=i\omega", r"K=\delta'(t)\quad(T_mf=f')", None, lambda p: math.inf),
    "分数冪の微分": (lambda w, p: np.abs(w) ** p, "α", (0.1, 2.0, 0.5, 0.1), r"m(\omega)=|\omega|^\alpha",
               r"(\alpha=2\ \text{で}\ T_mf=-f'')", None, lambda p: math.inf),
}


def apply(fv, m):
    return np.fft.ifft(m * np.fft.fft(fv))


def norm(v, p):
    v = np.abs(v)
    return float(np.max(v)) if p == math.inf else float((np.sum(v ** p) * DT) ** (1 / p))


def riesz_const(a):
    return 1 / (2 * gamma(a) * math.cos(PI * a / 2))


def riesz_pulse(t, a, w=1.0):
    """I_α に幅 w の矩形パルス χ_[−w/2, w/2] を入れた値(式で正確に)"""
    u, v = t + w / 2, t - w / 2
    return riesz_const(a) / a * (np.sign(u) * np.abs(u) ** a - np.sign(v) * np.abs(v) ** a)


def riesz_norm_unit(a, q):
    """‖I_α χ_[−1/2, 1/2]‖_q(数値積分 + 遠方の |t|^{α−1} の部分は式で)"""
    T = 2000.0
    t = np.concatenate([np.linspace(0, 1, 20001), np.geomspace(1.00001, T, 20001)])
    y = riesz_pulse(t, a) ** q
    core = 2 * float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(t)))
    c = riesz_const(a)
    tail = 2 * c ** q * T ** (q * (a - 1) + 1) / (q * (1 - a) - 1)
    return (core + tail) ** (1 / q)


def tab_multiplier():
    st.info("フーリエ変換した側で関数 $m(\\omega)$ をかけ, もとに戻す操作 $T_mf=\\mathcal F^{-1}[m\\hat f]$ を, "
            "**フーリエマルチプライヤー**(フーリエ乗数作用素)という。微分・フィルタ・熱方程式の解・総和法など, "
            "これまでのページの多くの操作がこの形をしている。")
    st.subheader("定義と合成積との関係")
    st.latex(r"\widehat{T_mf}(\omega)=m(\omega)\hat f(\omega),\qquad T_mf=K*f,\quad K=\mathcal F^{-1}m")
    st.markdown("合成積定理(第2章 1+.)から, マルチプライヤーをかけることは, 核 $K$ との合成積と同じ。"
                "$m$ が周波数ごとの「倍率(と位相のずれ)」, $K$ が時間の側での「ぼかし方」を表す。")
    st.markdown(r"""
| これまでのページ | 作用素 | マルチプライヤー $m$ |
|---|---|---|
| 第2章 フーリエ変換の性質 | 微分 $f'$, 時間推移 $f(t-a)$ | $i\omega$, $e^{-i\omega a}$ |
| 第1章 部分和・総和法(フーリエ級数) | $S_Nf$, $\sigma_Nf$, $A_rf$ | $\chi_{\lvert n\rvert\le N}$, $1-\frac{\lvert n\rvert}{N+1}$, $r^{\lvert n\rvert}$ |
| 第1章 良い核, ⑤ 偏微分方程式 | 熱方程式の解 | $e^{-s\omega^2}$ |
| 第3章 FFT とフィルタ | ローパスフィルタ | $\chi_{\lvert\omega\rvert\le\Omega}$ |
| 第2章 1+.(線形システム) | RC 回路 | $\dfrac{1}{1+i\omega RC}$ |
| 発展 6. ラドン変換 | ランプフィルタ $\Lambda$ | $\lvert\sigma\rvert$ |
""")
    st.caption("フーリエ級数では, 係数 $c_n$ に数列 $m(n)$ をかける操作 $\\sum m(n)c_ne^{inx}$ がマルチプライヤーになる。")

    st.subheader("動かして確かめる")
    c = st.columns(3)
    mname = c[0].selectbox("マルチプライヤー m", list(MULTS), key="mp_m")
    sname = c[1].selectbox("入力 f", list(SIGS), index=0, key="mp_f")
    mf, pname, prange, mtex, ktex, kfun, supm = MULTS[mname]
    p = c[2].slider(pname, *prange, key=f"mp_p_{mname}") if pname else None
    st.latex(mtex + r",\qquad " + ktex)
    f = SIGS[sname](TT)
    m = mf(WW, p)
    g = apply(f, m)
    real = bool(np.max(np.abs(g.imag)) < 1e-8 * max(1.0, np.max(np.abs(g.real))))
    show_t = np.abs(TT) <= 6
    cA, cB = st.columns(2)
    with cA:
        w = np.linspace(-30, 30, 1201)
        mw = mf(w, p) * np.ones_like(w)
        fig = go.Figure(go.Scatter(x=w, y=np.real(mw), mode="lines", line=dict(color=C1, width=2.5), name="Re m"))
        if np.max(np.abs(np.imag(mw))) > 1e-12:
            fig.add_trace(go.Scatter(x=w, y=np.imag(mw), mode="lines", line=dict(color=C2, width=2, dash="dash"), name="Im m"))
        layout(fig, title="マルチプライヤー m(ω)", xlab="ω", height=320)
        show(fig)
    with cB:
        fig = go.Figure(go.Scatter(x=TT[show_t], y=f[show_t], mode="lines", line=dict(color=GRAY, width=2.5), name="f"))
        fig.add_trace(go.Scatter(x=TT[show_t], y=g.real[show_t], mode="lines", line=dict(color=C2, width=2), name="T_m f"))
        if not real:
            fig.add_trace(go.Scatter(x=TT[show_t], y=g.imag[show_t], mode="lines", line=dict(color=C3, dash="dot"), name="Im T_m f"))
        layout(fig, title="入力 f と出力 T_m f", xlab="t", height=320)
        show(fig)
    if kfun is not None:
        tk = np.linspace(-6, 6, 2401)
        fig = go.Figure(go.Scatter(x=tk, y=kfun(tk, p), mode="lines", line=dict(color=C4, width=2), name="K(t)"))
        layout(fig, title="核 K(t) = 𝓕⁻¹m(T_m f = K * f)", xlab="t", height=260, legend=False)
        show(fig)
    s = supm(p)
    n2f, n2g = norm(f, 2), norm(g, 2)
    ninf_f, ninf_g = norm(f, math.inf), norm(g, math.inf)
    mm = st.columns(3)
    mm[0].metric("sup |m(ω)|", "∞(有界でない)" if s == math.inf else f"{s:g}")
    mm[1].metric("‖T_m f‖₂ / ‖f‖₂", f"{n2g / n2f:.4f}")
    mm[2].metric("‖T_m f‖∞ / ‖f‖∞", f"{ninf_g / ninf_f:.4f}")
    st.caption("数値計算は区間 [−40, 40) を周期とみなした FFT による。"
               "理想ローパスやヒルベルト変換で矩形パルスを入れると, 段差のところで出力が飛び出す(ギブス現象)・発散する(対数の特異性)。"
               "m がなめらかだと核 K が速く減衰し, 出力もなめらかになる。微分・分数冪の微分は高い周波数を大きくするので, 矩形パルスでは出力が大きくとがる。")

    st.subheader("L² での有界性:プランシュレルの等式から")
    st.caption("$L^2$, $L^p$ ノルムの定義と性質は「発展 7. 関数空間 L² と Lᵖ」を参照。")
    st.markdown("プランシュレルの等式 $\\|f\\|_2^2=\\frac{1}{2\\pi}\\|\\hat f\\|_2^2$(第2章 3.)より")
    st.latex(r"\|T_mf\|_2^2=\frac{1}{2\pi}\int|m(\omega)|^2|\hat f(\omega)|^2d\omega\le\left(\sup_\omega|m(\omega)|\right)^2\|f\|_2^2")
    st.markdown("なので, **$m$ が有界なら $T_m$ は $L^2$ で有界**で, 作用素のノルムはちょうど $\\sup|m|$ になる(逆も成り立つ)。"
                "上の「‖T_m f‖₂ / ‖f‖₂」は, どの入力でも sup|m| を超えない。")
    st.subheader("Lᵖ での有界性(p ≠ 2):調和解析の問題")
    st.markdown(
        "$p\\ne2$ ではプランシュレルの等式が使えず, $m$ が有界でも $T_m$ が $L^p$ で有界とは限らない。\n"
        "- **ヒルベルト変換**:$1<p<\\infty$ で $L^p$ 有界(M. リース, 1928)。$p=1,\\ \\infty$ では有界でない(下の図)。\n"
        "- **ミフリン・ヘルマンダーの定理**(1次元の形):$\\sup_\\omega|m(\\omega)|+\\sup_\\omega|\\omega\\,m'(\\omega)|<\\infty$ なら, "
        "$1<p<\\infty$ で $L^p$ 有界(ヘルマンダー, 1960)。なめらかなローパス・RC 回路・熱半群はこの条件をみたす。\n"
        "- **理想ローパス** $\\chi_{[-\\Omega,\\Omega]}$ は1次元では $1<p<\\infty$ で有界(ヒルベルト変換で書ける)。"
        "2次元以上の円板のマルチプライヤーは $p\\ne2$ で有界でない(フェファーマン, 1971):発展 2・3 の球面部分和の現象とつながる。")
    tk = np.linspace(-3, 3, 3001)
    with np.errstate(divide="ignore"):
        Hf = np.log(np.abs((tk + 1) / np.where(np.abs(tk - 1) < 1e-12, 1e-12, tk - 1))) / PI   # t = ±1 で発散(図では切る)
    Hf = np.where(np.isfinite(Hf), Hf, np.nan)
    fig = go.Figure(go.Scatter(x=tk, y=(np.abs(tk) < 1).astype(float), mode="lines", line=dict(color=GRAY, width=2.5), name="f = χ_[−1,1]"))
    fig.add_trace(go.Scatter(x=tk, y=Hf, mode="lines", line=dict(color=C2, width=2), name="Hf"))
    layout(fig, title="ヒルベルト変換:Hχ_[−1,1](t) = (1/π) log|(t+1)/(t−1)|", xlab="t", height=300, yrange=(-3, 3))
    show(fig)
    st.caption("有界な関数 f(|f| ≤ 1)なのに, Hf は t = ±1 で無限大になる:H は L^∞ で有界でない。"
               "ただし log の特異性は弱く, 遠くでは Hf ≈ 2/(πt) なので, Hf は 1 < p < ∞ で p 乗可積分になる(リースの定理と合う)。")

    st.subheader("リースポテンシャル(分数冪積分)")
    st.markdown("$0<\\alpha<1$ に対し, マルチプライヤー $|\\omega|^{-\\alpha}$ の作用素を**リースポテンシャル**(分数冪積分作用素)という。"
                "分数冪の微分 $|\\omega|^\\alpha$ の逆で, 「$\\alpha$ 回だけ積分する」作用素にあたる:")
    st.latex(r"I_\alpha f(t)=c_\alpha\int_{-\infty}^{\infty}\frac{f(s)}{|t-s|^{1-\alpha}}\,ds,\qquad "
             r"\widehat{I_\alpha f}(\omega)=|\omega|^{-\alpha}\hat f(\omega),\qquad c_\alpha=\frac{1}{2\Gamma(\alpha)\cos\frac{\pi\alpha}{2}}")
    st.caption("$\\int|t|^{\\alpha-1}e^{-i\\omega t}dt=2\\Gamma(\\alpha)\\cos\\frac{\\pi\\alpha}{2}\\,|\\omega|^{-\\alpha}$ から定数 $c_\\alpha$ が決まる。"
               "$m=|\\omega|^{-\\alpha}$ は $\\omega=0$ で有界でないので, $I_\\alpha$ は $L^2$ から $L^2$ への有界作用素ではない。")
    st.markdown("**例:矩形パルス** $f=\\chi_{[-w/2,\\,w/2]}$。$I_\\alpha f(t)=c_\\alpha\\displaystyle\\int_{t-w/2}^{t+w/2}|u|^{\\alpha-1}\\,du$ を計算すると")
    st.latex(r"I_\alpha f(t)=\frac{c_\alpha}{\alpha}\times\begin{cases}"
             r"\left(\dfrac w2+t\right)^{\alpha}+\left(\dfrac w2-t\right)^{\alpha} & \left(|t|\le\dfrac w2\right)\\[2mm]"
             r"\left(|t|+\dfrac w2\right)^{\alpha}-\left(|t|-\dfrac w2\right)^{\alpha} & \left(|t|\ge\dfrac w2\right)\end{cases}")
    st.markdown("- 最大値は $t=0$ で $\\dfrac{2c_\\alpha}{\\alpha}\\left(\\dfrac w2\\right)^\\alpha$, 縁 $t=\\pm\\dfrac w2$ で $\\dfrac{c_\\alpha}{\\alpha}w^\\alpha$(連続)。"
                "$I_\\alpha$ は平均ではなく「$\\alpha$ 回の積分」なので, 値は 1 を超えうるし, 幅 $w$ に対して $w^\\alpha$ 倍で大きくなる。\n"
                "- 遠方では $I_\\alpha f(t)\\approx c_\\alpha w\\,|t|^{\\alpha-1}$ とゆっくり減衰する(核そのものの形)。\n"
                "- **縁での特異性**:微分すると $f'=\\delta_{-w/2}-\\delta_{w/2}$ より")
    st.latex(r"(I_\alpha f)'(t)=c_\alpha\left(\left|t+\frac w2\right|^{\alpha-1}-\left|t-\frac w2\right|^{\alpha-1}\right)")
    st.markdown("となり, **核の特異性 $|u|^{\\alpha-1}$ がそのまま縁に現れる**:$I_\\alpha f$ は縁で連続だが接線は垂直で, "
                "$I_\\alpha f(t)\\approx\\dfrac{c_\\alpha}{\\alpha}w^\\alpha-\\dfrac{c_\\alpha}{\\alpha}\\left|t-\\dfrac w2\\right|^\\alpha$ の尖り(α 次ヘルダー連続)をもつ。"
                "$\\alpha\\to0$ では指数 $\\alpha-1\\to-1$ となって特異性は強くなり, 尖りは段差に近づく。")
    st.markdown("- **定数 $c_\\alpha$ の役割**:$\\Gamma(\\alpha)\\approx\\dfrac1\\alpha$ より $c_\\alpha\\approx\\dfrac\\alpha2\\to0$。"
                "定数なしの積分 $\\int|t-s|^{\\alpha-1}f(s)\\,ds$ は $\\alpha\\to0$ で $\\dfrac1\\alpha$ のように発散するが, "
                "$c_\\alpha$ を掛けると打ち消され, $I_\\alpha f\\to f$($m=|\\omega|^{-\\alpha}\\to1$)となる。")
    c = st.columns(3)
    a = c[0].slider("α", 0.05, 0.95, 0.5, 0.05, key="mp_a")
    wid = c[1].slider("矩形パルスの幅 w", 0.25, 4.0, 1.0, 0.25, key="mp_w")
    cmode = c[2].radio("縦軸", ["c_α あり(I_α f)", "c_α なし(核の積分のみ)"], key="mp_norm_c")
    scale = (lambda aa: 1.0) if cmode.startswith("c_α あり") else (lambda aa: 1 / riesz_const(aa))
    tr = np.linspace(-6, 6, 4801)
    fig = go.Figure(go.Scatter(x=tr, y=(np.abs(tr) < wid / 2).astype(float), mode="lines", line=dict(color=GRAY, width=2.5, shape="hv"), name="f"))
    comps = [aa for aa in (0.1, 0.9) if abs(aa - a) > 1e-9]
    for aa, col in zip(comps, (C3, C4)):
        fig.add_trace(go.Scatter(x=tr, y=scale(aa) * riesz_pulse(tr, aa, wid), mode="lines", line=dict(color=col, width=1.5, dash="dot"), name=f"α = {aa:g}"))
    fig.add_trace(go.Scatter(x=tr, y=scale(a) * riesz_pulse(tr, a, wid), mode="lines", line=dict(color=C2, width=2.5), name=f"α = {a:g}(スライダー)"))
    layout(fig, title="矩形パルスのリースポテンシャル", xlab="t", height=320)
    show(fig)
    st.caption("c_α ありでは α → 0 で f に近づき, 縁の尖りは鋭くなる。c_α なしでは α → 0 で値が 1/α のように発散する(核 |t|^{α−1} の特異性が強まるため)。"
               "逆に α → 1 では cos(πα/2) → 0 より c_α → ∞ となり, c_α ありの値は大きくなる(|ω|^{−1} は ω = 0 で可積分でない)。")

    st.markdown("**縁 $t=w/2$ の拡大**:$\\dfrac{\\alpha}{c_\\alpha}\\bigl(I_\\alpha f(w/2)-I_\\alpha f(t)\\bigr)$ を $x=t-w/2$ に対して描く(主要部は $|x|^\\alpha$)。")
    xr = np.linspace(-0.2, 0.2, 2001)
    fig = go.Figure()
    for aa, col in ((0.1, C3), (0.25, C1), (0.5, C2), (0.75, C4)):
        y = aa / riesz_const(aa) * (riesz_pulse(np.array([wid / 2]), aa, wid)[0] - riesz_pulse(wid / 2 + xr, aa, wid))
        fig.add_trace(go.Scatter(x=xr, y=y, mode="lines", line=dict(color=col, width=2), name=f"α = {aa:g}"))
    layout(fig, title="縁の尖り:α が小さいほど鋭い(接線は垂直)", xlab="x = t − w/2", height=300)
    show(fig)
    st.caption("どの α でも x = 0 で接線が垂直(導関数 ∝ |x|^{α−1} → ∞)。α が小さいほど |x|^α は 0 の近くで急に立ち上がり, 段差(α = 0)に近づく。")
    st.markdown("**ハーディ・リトルウッド・ソボレフの不等式**(1次元):$0<\\alpha<1$, $1<p<\\dfrac1\\alpha$, $\\dfrac1q=\\dfrac1p-\\alpha$ のとき")
    st.latex(r"\|I_\alpha f\|_{L^q}\le C\|f\|_{L^p}")
    st.markdown("**$q$ がこの値でなければならない理由(スケーリング)**:幅 $w$ のパルス $f_w$ では $\\|f_w\\|_p=w^{1/p}$, "
                "$I_\\alpha f_w(t)=w^{\\alpha}(I_\\alpha f_1)(t/w)$ より $\\|I_\\alpha f_w\\|_q=w^{\\alpha+1/q}\\|I_\\alpha f_1\\|_q$。比")
    st.latex(r"\frac{\|I_\alpha f_w\|_q}{\|f_w\|_p}=w^{\alpha+\frac1q-\frac1p}\cdot\frac{\|I_\alpha f_1\|_q}{\|f_1\|_p}")
    st.markdown("が $w$ によらず有界であるためには, 指数が 0, すなわち $\\frac1q=\\frac1p-\\alpha$ でなければならない。")
    p_ = st.select_slider("p", [1.25, 1.5, 1.75], value=1.5, key="mp_pp")
    if 1 / p_ - a <= 0:
        st.warning(f"p = {p_} は 1/α = {1 / a:.3g} 以上なので, この α では不等式の範囲外。α を小さくしよう。")
    else:
        qs = 1 / (1 / p_ - a)
        ws = np.geomspace(0.05, 20, 40)
        fig = go.Figure()
        for q, col, nm in ((qs, C2, f"q = {qs:.3g}(1/q = 1/p − α)"), (qs * 1.5, C3, f"q = {qs * 1.5:.3g}"), (max(1 / (1 - a) + 0.2, qs * 0.75), C1, f"q = {max(1 / (1 - a) + 0.2, qs * 0.75):.3g}")):
            r1 = riesz_norm_unit(a, q)
            fig.add_trace(go.Scatter(x=ws, y=ws ** (a + 1 / q - 1 / p_) * r1, mode="lines+markers", marker=dict(size=4),
                                     line=dict(color=col, width=2), name=nm))
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
        layout(fig, title=f"‖I_α f_w‖_q / ‖f_w‖_p(α = {a:g}, p = {p_:g})", xlab="パルスの幅 w", height=340)
        show(fig)
        st.caption("正しい q では比が一定(水平な線)。他の q では, w を小さく(または大きく)すると比がいくらでも大きくなるので, 不等式は成り立たない。")
    st.markdown("リースポテンシャルは, ソボレフの埋め込み定理(微分の可積分性から関数自身の可積分性を導く)の基礎になる。"
                "その有界性は, モレー空間(アダムスの定理)やオーリッチ空間など, いろいろな関数空間の上で現在も研究されている。")

    with st.expander("練習問題"):
        st.markdown("**問1** $m(\\omega)=i\\omega$ のとき, $f(t)=\\cos2t$ に対する $T_mf$ を求めよ。")
        st.markdown("**問2** $m(\\omega)=\\dfrac{1}{1+i\\omega}$ の $L^2$ でのノルム $\\sup|m|$ と, $f(t)=\\cos t$ に対する出力の振幅を求めよ。")
        st.markdown("**問3** $\\alpha=\\frac12$, $p=\\frac43$ のとき, ハーディ・リトルウッド・ソボレフの不等式の $q$ を求めよ。")
        if st.toggle("解答・解説を表示", key="pr_mult"):
            st.markdown("**答1** $T_me^{i\\omega_0t}=m(\\omega_0)e^{i\\omega_0t}$ なので, $\\cos2t=\\frac12(e^{2it}+e^{-2it})$ から "
                        "$T_mf=\\frac12(2i\\,e^{2it}-2i\\,e^{-2it})=-2\\sin2t$($=f'$)。")
            st.markdown("**答2** $|m(\\omega)|=\\frac{1}{\\sqrt{1+\\omega^2}}\\le1$ で, $\\sup|m|=1$($\\omega=0$)。"
                        "$\\omega=1$ では $|m(1)|=\\frac{1}{\\sqrt2}$ なので, 振幅は $\\frac{1}{\\sqrt2}$(位相は $\\frac\\pi4$ 遅れる)。")
            st.markdown("**答3** $\\frac1q=\\frac34-\\frac12=\\frac14$ より $q=4$。")
    page_refs([("riesz1928", "ヒルベルト変換(共役関数)の Lᵖ 有界性。"), ("hormander1960", "マルチプライヤー定理。"),
               ("hardylittlewood1928", "分数冪積分の不等式(ハーディ・リトルウッド・ソボレフ)。"),
               ("stein1970", "特異積分・リースポテンシャル・マルチプライヤーの古典的な教科書。"),
               ("adams1975", "モレー空間でのリースポテンシャルの有界性(アダムスの定理)。"),
               ("grafakos2014", "マルチプライヤーとヒルベルト変換・分数冪積分。")])
