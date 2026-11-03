"""③ フーリエ解析:合成積(たたみこみ)
  1-4. フーリエ級数の合成積とその性質(周期 2π)
  2-1+. フーリエ変換の合成積とその性質
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import C1, C2, C3, C4, GRAY, layout, show, trapz
from topics.fourier import wrap

PI = np.pi

# ============================================================
# 1-4. フーリエ級数の合成積
# ============================================================
M = 2048                                      # [−π, π) の分点数(FFT で円形の合成積を計算する)
XG = -PI + 2 * PI * np.arange(M) / M

PER = {
    "矩形波": (lambda x, d: np.sign(wrap(x)), r"\begin{cases}-1&(-\pi<x<0)\\ \ \ 1&(0<x<\pi)\end{cases}"),
    "のこぎり波 x": (lambda x, d: wrap(x), r"x\quad(-\pi<x<\pi)"),
    "三角波 |x|": (lambda x, d: np.abs(wrap(x)), r"|x|\quad(-\pi<x<\pi)"),
    "パルス(幅 2d)": (lambda x, d: (np.abs(wrap(x)) < d).astype(float), r"\begin{cases}1&(|x|<d)\\0&(d<|x|<\pi)\end{cases}"),
    "cos x": (lambda x, d: np.cos(x), r"\cos x"),
}


def per_conv(fv, gv):
    """(f*g)(x) = (1/2π)∫ f(x−t) g(t) dt を格子上で(円形の合成積 = 係数の積)"""
    h = np.real(np.fft.ifft(np.fft.fft(fv) * np.fft.fft(gv))) / M
    return np.roll(h, -M // 2)                # 格子の始点が −π なので, そのままだと x − π での値になる


def cn_grid(v, nmax):
    """格子の値から複素フーリエ係数 c_n(n = −nmax..nmax)"""
    c = np.fft.fft(v) / M * np.exp(1j * np.arange(M) * PI)  # x の始点 −π の分の位相を戻す
    n = np.arange(-nmax, nmax + 1)
    return n, c[n % M]


def tab_series_convolution():
    st.info("2つの周期関数を「ずらしながらかけて平均する」のが合成積。フーリエ係数で見ると、合成積は**係数の積**になる。"
            "部分和 $S_N$ やフェイェール平均も、合成積で表せる。ここでは周期 $2\\pi$ で考える。")
    st.subheader("定義")
    st.markdown("周期 $2\\pi$ の関数 $f,\\ g$ の合成積(たたみこみ)を")
    st.latex(r"(f*g)(x)=\frac{1}{2\pi}\int_{-\pi}^{\pi}f(x-t)\,g(t)\,dt")
    st.markdown("で定める。$f*g$ も周期 $2\\pi$ の関数になる。$x$ を固定すると, $f(x-t)$ は $f$ を折り返して $x$ だけずらしたもので, "
                "それに $g(t)$ をかけて $[-\\pi,\\ \\pi]$ で平均した値が $(f*g)(x)$ である。")

    st.subheader("性質")
    st.markdown(r"""
| 性質 | 式 |
|---|---|
| 交換法則 | $f*g=g*f$ |
| 結合法則 | $(f*g)*h=f*(g*h)$ |
| 分配法則・定数倍 | $f*(ag+bh)=a(f*g)+b(f*h)$ |
| 平行移動 | $f(x-a)*g=(f*g)(x-a)$ |
| 微分 | $(f*g)'=f'*g$($f$ がなめらかなとき) |
| 大きさの評価 | $\lvert(f*g)(x)\rvert\le \max\lvert f\rvert\cdot\dfrac{1}{2\pi}\displaystyle\int_{-\pi}^{\pi}\lvert g(t)\rvert dt$ |
| **フーリエ係数** | $c_n(f*g)=c_n(f)\,c_n(g)$ |
""")
    st.caption("$c_n(f)=\\dfrac{1}{2\\pi}\\displaystyle\\int_{-\\pi}^{\\pi}f(x)e^{-inx}dx$ は複素フーリエ係数。"
               "合成積をとると, なめらかな方の性質を受けつぐ(不連続な関数どうしでも, 合成積は連続になる)。")
    with st.expander("フーリエ係数の性質の証明"):
        st.markdown("積分の順序を入れかえ, $s=x-t$ とおく($f$ は周期 $2\\pi$ なので, $s$ の積分区間は1周期分のままでよい)。")
        st.latex(r"c_n(f*g)=\frac{1}{2\pi}\int_{-\pi}^{\pi}\left\{\frac{1}{2\pi}\int_{-\pi}^{\pi}f(x-t)g(t)\,dt\right\}e^{-inx}dx"
                 r"=\frac{1}{2\pi}\int_{-\pi}^{\pi}g(t)e^{-int}\left\{\frac{1}{2\pi}\int_{-\pi}^{\pi}f(x-t)e^{-in(x-t)}dx\right\}dt")
        st.latex(r"=\frac{1}{2\pi}\int_{-\pi}^{\pi}g(t)e^{-int}\,c_n(f)\,dt=c_n(f)\,c_n(g)")
        st.caption("$e^{-inx}=e^{-in(x-t)}e^{-int}$ と分けるのがポイント。")

    st.divider()
    st.subheader("ずらしてかけて平均する")
    c = st.columns(3)
    fn = c[0].selectbox("f", list(PER), index=0, key="fc_f")
    gn = c[1].selectbox("g", list(PER), index=0, key="fc_g")
    d = c[2].slider("パルスの半幅 d", 0.2, 3.0, 1.0, 0.1, key="fc_d")
    c[2].caption("(パルスを選んだときだけ使う)")
    f, g = PER[fn][0], PER[gn][0]
    fv, gv = f(XG, d), g(XG, d)
    hv = per_conv(fv, gv)
    k = st.slider("調べる点 x₀ = kπ/12 の k", -12, 12, 4, key="fc_k")
    x0 = k * PI / 12
    st.latex(rf"f(x)={PER[fn][1]},\qquad g(x)={PER[gn][1]}")
    prod = f(x0 - XG, d) * gv
    val = float(trapz(np.r_[prod, prod[0]], np.r_[XG, PI])) / (2 * PI)
    cA, cB = st.columns(2)
    with cA:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=XG, y=prod, mode="lines", fill="tozeroy", line=dict(color=C4, width=1),
                                 fillcolor="rgba(255,127,14,0.25)", name="f(x₀−t)g(t)"))
        fig.add_trace(go.Scatter(x=XG, y=f(x0 - XG, d), mode="lines", line=dict(color=C1, width=2.5), name="f(x₀−t)"))
        fig.add_trace(go.Scatter(x=XG, y=gv, mode="lines", line=dict(color=C3, width=2.5, dash="dash"), name="g(t)"))
        layout(fig, title=f"t の関数として見る(x₀ = {k}π/12)", xlab="t", height=360)
        show(fig)
    with cB:
        fig = go.Figure(go.Scatter(x=XG, y=hv, mode="lines", line=dict(color=C2, width=3), name="(f*g)(x)"))
        fig.add_trace(go.Scatter(x=[x0], y=[val], mode="markers", marker=dict(color="black", size=11), name="(f*g)(x₀)"))
        layout(fig, title="合成積 (f*g)(x)", xlab="x", height=360)
        show(fig)
    st.write(f"(f*g)(x₀) = (1/2π)∫ f(x₀−t)g(t) dt ≈ {val:.4f}(左の色のついた部分の面積 ÷ 2π)")

    st.markdown("**係数の積になることを確かめる**")
    n, cf = cn_grid(fv, 12)
    _, cg = cn_grid(gv, 12)
    _, ch = cn_grid(hv, 12)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=n, y=np.abs(cf * cg), name="|c_n(f)·c_n(g)|", marker_color=C1, opacity=0.6))
    fig.add_trace(go.Scatter(x=n, y=np.abs(ch), mode="markers", marker=dict(color=C2, size=10, symbol="x"),
                             name="|c_n(f*g)|(合成積を直接計算)"))
    layout(fig, title="c_n(f*g) = c_n(f) c_n(g)", xlab="n", height=320)
    show(fig)
    st.caption(f"最大のずれ:{np.max(np.abs(ch - cf * cg)):.1e}。高い周波数の係数は, 積になってさらに小さくなる(合成積はなめらかになる)。")

    st.divider()
    st.subheader("部分和とフェイェール平均も合成積")
    st.markdown("フーリエ級数の部分和は, **ディリクレ核** $D_N$ との合成積, フェイェール平均は**フェイェール核** $F_N$ との合成積である。")
    st.latex(r"S_N(x)=\sum_{n=-N}^{N}c_ne^{inx}=(f*D_N)(x),\qquad D_N(x)=\sum_{n=-N}^{N}e^{inx}=\frac{\sin\left(N+\frac12\right)x}{\sin\frac{x}{2}}")
    st.latex(r"\sigma_N(x)=\frac{S_0+S_1+\cdots+S_N}{N+1}=(f*F_N)(x),\qquad "
             r"F_N(x)=\sum_{n=-N}^{N}\left(1-\frac{|n|}{N+1}\right)e^{inx}=\frac{1}{N+1}\left(\frac{\sin\frac{(N+1)x}{2}}{\sin\frac{x}{2}}\right)^2")
    st.caption("$c_n(D_N)=1\\ (|n|\\le N)$ なので, 係数の積の性質から $c_n(f*D_N)=c_n(f)$($|n|\\le N$)、それ以外は 0。")
    N = st.slider("N", 1, 40, 8, key="fc_N")
    nn = np.arange(-N, N + 1)[:, None]
    DN = np.real(np.exp(1j * nn * XG)).sum(axis=0)
    FN = (np.real(np.exp(1j * nn * XG)) * (1 - np.abs(nn) / (N + 1))).sum(axis=0)
    sq = np.sign(wrap(XG))
    cA, cB = st.columns(2)
    with cA:
        fig = go.Figure(go.Scatter(x=XG, y=DN, mode="lines", line=dict(color=C1, width=2), name="D_N"))
        fig.add_trace(go.Scatter(x=XG, y=FN, mode="lines", line=dict(color=C2, width=2), name="F_N"))
        fig.add_hline(y=0, line=dict(color=GRAY, width=1))
        layout(fig, title=f"ディリクレ核とフェイェール核(N = {N})", xlab="x", height=360)
        show(fig)
    with cB:
        fig = go.Figure(go.Scatter(x=XG, y=sq, mode="lines", line=dict(color=GRAY, width=2), name="矩形波 f"))
        fig.add_trace(go.Scatter(x=XG, y=per_conv(sq, DN), mode="lines", line=dict(color=C1), name="f*D_N = S_N"))
        fig.add_trace(go.Scatter(x=XG, y=per_conv(sq, FN), mode="lines", line=dict(color=C2), name="f*F_N = σ_N"))
        layout(fig, title="矩形波との合成積", xlab="x", height=360, yrange=(-1.4, 1.4))
        show(fig)
    st.caption("$D_N$ は負の値もとるので, 不連続点のそばで $S_N$ が飛び出す(ギブス現象)。"
               "$F_N\\ge0$ で $\\dfrac{1}{2\\pi}\\int F_N=1$ なので, $\\sigma_N$ は $f$ の「重みつき平均」になり, $f$ の最大値・最小値を超えない。")

    with st.expander("練習問題"):
        st.markdown("**問1** $f$ を矩形波($-1\\ (-\\pi<x<0),\\ 1\\ (0<x<\\pi)$)とする。$c_n(f)$ を求め, "
                    "係数の積の性質を用いて $f*f$ を求めよ。")
        st.markdown("**問2** $g(x)=x\\ (-\\pi<x<\\pi)$ のとき, $(\\cos x)*g$ を求めよ。")
        st.markdown("**問3** 周期 $2\\pi$ の関数 $g$ について, $(\\cos x)*g=\\dfrac12(a_1\\cos x+b_1\\sin x)$ を示せ($a_1,\\ b_1$ は $g$ のフーリエ係数)。")
        if st.toggle("解答・解説を表示", key="pr_fourier3_series"):
            st.markdown("**答1**")
            st.latex(r"c_n(f)=\frac{1-(-1)^n}{i\pi n}=\begin{cases}\dfrac{2}{i\pi n}&(n\text{ が奇数})\\0&(n\text{ が偶数})\end{cases}"
                     r"\ \Rightarrow\ c_n(f*f)=c_n(f)^2=-\frac{4}{\pi^2n^2}\ (n\text{ が奇数})")
            st.markdown("$|x|$ の複素フーリエ係数 $c_0=\\dfrac{\\pi}{2}$, $c_n=-\\dfrac{2}{\\pi n^2}\\ (n$ が奇数$)$ と比べると, "
                        "$f*f=\\dfrac{2}{\\pi}|x|-1\\ (-\\pi\\le x\\le\\pi)$。不連続な矩形波から, 連続な三角波ができる。")
            st.markdown("**答2**")
            st.latex(r"((\cos x)*g)(x)=\frac{1}{2\pi}\int_{-\pi}^{\pi}\cos(x-t)\,t\,dt=\frac{1}{2\pi}\left(\cos x\int_{-\pi}^{\pi}t\cos t\,dt+\sin x\int_{-\pi}^{\pi}t\sin t\,dt\right)"
                     r"=\frac{1}{2\pi}(0+2\pi\sin x)=\sin x")
            st.markdown("**答3**")
            st.markdown("$\\cos x$ の複素フーリエ係数は $c_{\\pm1}=\\dfrac12$, それ以外 $0$。係数の積の性質より")
            st.latex(r"(\cos x)*g=\frac12\left(c_1(g)e^{ix}+c_{-1}(g)e^{-ix}\right)=\frac12(a_1\cos x+b_1\sin x)")
            st.caption("$\\cos x$ との合成積は, $g$ から周波数 1 の成分だけを取り出す「フィルタ」になる(問2:$b_1=2$)。")


# ============================================================
# 2-1+. フーリエ変換の合成積
# ============================================================
TG = np.linspace(-12, 12, 4801)
DT = TG[1] - TG[0]


def sig(name, p):
    """信号と、そのフーリエ変換 F(ω) = ∫ f(t) e^{−iωt} dt"""
    if name == "矩形パルス":
        return (lambda t: (np.abs(t) < p / 2).astype(float),
                lambda w: np.where(np.abs(w) < 1e-12, p, 2 * np.sin(w * p / 2) / np.where(w == 0, 1, w)),
                rf"\mathrm{{rect}}(t)=\begin{{cases}}1&(|t|<{p / 2:g})\\0&(|t|>{p / 2:g})\end{{cases}}")
    if name == "ガウス関数":
        return (lambda t: np.exp(-t ** 2 / (2 * p ** 2)),
                lambda w: p * np.sqrt(2 * PI) * np.exp(-p ** 2 * w ** 2 / 2),
                rf"e^{{-t^2/(2\cdot{p:g}^2)}}")
    return (lambda t: np.where(t >= 0, np.exp(-p * np.abs(t)), 0.0),
            lambda w: 1 / (p + 1j * w),
            rf"e^{{-{p:g}t}}u(t)\ (u\text{{ は単位ステップ関数}})")


SIGS = ["矩形パルス", "ガウス関数", "片側指数関数"]
PLAB = {"矩形パルス": ("幅", 0.4, 4.0, 2.0), "ガウス関数": ("σ", 0.3, 2.0, 0.7), "片側指数関数": ("a", 0.3, 3.0, 1.0)}


def tab_ft_convolution():
    st.info("合成積(たたみこみ)は, 一方を折り返してずらしながら他方とかけて積分したもの。"
            "フーリエ変換すると**ただの積**になる(合成積定理)。線形システムの出力は, 入力とインパルス応答の合成積で表される。")
    st.subheader("定義")
    st.latex(r"(f*g)(t)=\int_{-\infty}^{\infty}f(t-\tau)\,g(\tau)\,d\tau")
    st.caption("フーリエ級数の合成積(第1章4節)は周期 $2\\pi$ で平均($\\frac{1}{2\\pi}\\int_{-\\pi}^{\\pi}$)をとったが, ここでは全区間で積分する。"
               "フーリエ変換は $F(\\omega)=\\mathcal F[f](\\omega)=\\displaystyle\\int_{-\\infty}^{\\infty}f(t)e^{-i\\omega t}dt$。")

    st.subheader("性質")
    st.markdown(r"""
| 性質 | 式 |
|---|---|
| 交換・結合・分配法則 | $f*g=g*f$, $(f*g)*h=f*(g*h)$, $f*(g+h)=f*g+f*h$ |
| 時間推移 | $f(t-a)*g=(f*g)(t-a)$ |
| 微分 | $\dfrac{d}{dt}(f*g)=f'*g=f*g'$ |
| デルタ関数 | $f*\delta=f$(デルタ関数が合成積の「1」) |
| 面積 | $\displaystyle\int_{-\infty}^{\infty}(f*g)\,dt=\int f\,dt\cdot\int g\,dt$ |
| 幅 | $f$ が $[a,\ b]$, $g$ が $[c,\ d]$ の外で 0 なら, $f*g$ は $[a+c,\ b+d]$ の外で 0 |
| **合成積定理** | $\mathcal F[f*g]=F(\omega)G(\omega)$ |
| 積のフーリエ変換 | $\mathcal F[fg]=\dfrac{1}{2\pi}(F*G)(\omega)$ |
""")
    with st.expander("合成積定理の証明"):
        st.markdown("積分の順序を入れかえ, $s=t-\\tau$ とおく。")
        st.latex(r"\mathcal F[f*g](\omega)=\int_{-\infty}^{\infty}\left\{\int_{-\infty}^{\infty}f(t-\tau)g(\tau)d\tau\right\}e^{-i\omega t}dt"
                 r"=\int_{-\infty}^{\infty}g(\tau)e^{-i\omega\tau}\left\{\int_{-\infty}^{\infty}f(t-\tau)e^{-i\omega(t-\tau)}dt\right\}d\tau")
        st.latex(r"=\int_{-\infty}^{\infty}g(\tau)e^{-i\omega\tau}F(\omega)\,d\tau=F(\omega)G(\omega)")
        st.caption("面積の性質は, 合成積定理で $\\omega=0$ とおいたもの:$\\mathcal F[f*g](0)=F(0)G(0)$。")

    st.divider()
    st.subheader("ずらしてかけて積分する")
    c = st.columns(2)
    fn = c[0].selectbox("f", SIGS, index=0, key="ftc_f")
    lab, lo, hi, v0 = PLAB[fn]
    pf = c[0].slider(f"f の {lab}", lo, hi, v0, 0.1, key="ftc_pf")
    gn = c[1].selectbox("g", SIGS, index=0, key="ftc_g")
    lab, lo, hi, v0 = PLAB[gn]
    pg = c[1].slider(f"g の {lab}", lo, hi, v0 * 0.75 if gn != "片側指数関数" else v0, 0.1, key="ftc_pg")
    f, F, ftex = sig(fn, pf)
    g, G, gtex = sig(gn, pg)
    st.latex(rf"f(t)={ftex},\qquad g(t)={gtex}")
    fv, gv = f(TG), g(TG)
    hv = np.convolve(fv, gv, mode="same") * DT
    t0 = st.slider("t(ずらす量)", -6.0, 6.0, 0.5, 0.1, key="ftc_t")
    prod = f(t0 - TG) * gv
    val = float(trapz(prod, TG))
    cA, cB = st.columns(2)
    with cA:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=TG, y=prod, mode="lines", fill="tozeroy", line=dict(color=C4, width=1),
                                 fillcolor="rgba(255,127,14,0.25)", name="f(t−τ)g(τ)"))
        fig.add_trace(go.Scatter(x=TG, y=f(t0 - TG), mode="lines", line=dict(color=C1, width=2.5), name="f(t−τ)"))
        fig.add_trace(go.Scatter(x=TG, y=gv, mode="lines", line=dict(color=C3, width=2.5, dash="dash"), name="g(τ)"))
        layout(fig, title=f"τ の関数として見る(t = {t0:g})", xlab="τ", height=360, xrange=(-6, 6))
        show(fig)
    with cB:
        fig = go.Figure(go.Scatter(x=TG, y=hv, mode="lines", line=dict(color=C2, width=3), name="(f*g)(t)"))
        fig.add_trace(go.Scatter(x=[t0], y=[val], mode="markers", marker=dict(color="black", size=11), name="(f*g)(t)"))
        layout(fig, title="合成積 (f*g)(t)", xlab="t", height=360, xrange=(-6, 6))
        show(fig)
    st.write(f"(f*g)({t0:g}) = ∫ f(t−τ)g(τ) dτ ≈ {val:.4f}(左の色のついた部分の面積)")
    note = {("矩形パルス", "矩形パルス"): "矩形パルスどうしの合成積は台形(幅が等しければ三角形)。幅は2つの幅の和になる。",
            ("ガウス関数", "ガウス関数"): "ガウス関数どうしの合成積もガウス関数で, σ² が足される:σ² = σ_f² + σ_g²。",
            ("片側指数関数", "片側指数関数"): "a が等しいとき, e^{−at}u(t) * e^{−at}u(t) = t e^{−at} u(t)。"}
    if (fn, gn) in note:
        st.caption(note[(fn, gn)])

    st.markdown("**合成積定理を確かめる**:左は合成積 $f*g$ を数値でフーリエ変換したもの, 右は $F(\\omega)G(\\omega)$。")
    w = np.linspace(-12, 12, 481)
    Hn = (hv[None, :] * np.exp(-1j * w[:, None] * TG[None, :])).sum(axis=1) * DT
    FG = F(w) * G(w)
    cA, cB = st.columns(2)
    with cA:
        fig = go.Figure(go.Scatter(x=w, y=np.abs(Hn), mode="lines", line=dict(color=C2, width=3), name="|𝓕[f*g]|"))
        fig.add_trace(go.Scatter(x=w, y=np.abs(FG), mode="lines", line=dict(color="black", width=1.5, dash="dot"),
                                 name="|F G|"))
        layout(fig, title="振幅 |𝓕[f*g](ω)| と |F(ω)G(ω)|", xlab="ω", height=320)
        show(fig)
    with cB:
        fig = go.Figure(go.Scatter(x=w, y=np.abs(F(w)), mode="lines", line=dict(color=C1), name="|F|"))
        fig.add_trace(go.Scatter(x=w, y=np.abs(G(w)), mode="lines", line=dict(color=C3, dash="dash"), name="|G|"))
        layout(fig, title="それぞれの振幅スペクトル", xlab="ω", height=320)
        show(fig)
    st.caption(f"最大のずれ:{np.max(np.abs(Hn - FG)):.1e}(数値積分の誤差)。|F|, |G| の小さい周波数は, 合成積でさらに小さくなる → 合成積は「ぼかし」になる。")

    with st.expander("応用:線形システムと RC 回路"):
        st.markdown("入力 $x(t)$ に対する出力 $y(t)$ が線形で時間によらないシステムでは, デルタ関数を入れたときの出力 "
                    "$h(t)$(インパルス応答)を使って")
        st.latex(r"y(t)=(h*x)(t)=\int_{-\infty}^{\infty}h(t-\tau)x(\tau)\,d\tau,\qquad Y(\omega)=H(\omega)X(\omega)")
        st.markdown("と書ける。$H(\\omega)$ を**伝達関数**(周波数応答)という。RC 回路(コンデンサーの電圧を出力)では "
                    "$h(t)=\\dfrac{1}{RC}e^{-t/RC}u(t)$, $H(\\omega)=\\dfrac{1}{1+i\\omega RC}$ で, 高い周波数ほど $|H|$ が小さい(ローパスフィルタ)。"
                    "上の図で g を片側指数関数にすると, f をなめらかにする様子が見られる。")

    with st.expander("練習問題"):
        st.markdown("**問1** $f(t)=\\begin{cases}1&(|t|<\\frac12)\\\\0&(|t|>\\frac12)\\end{cases}$ のとき, $(f*f)(t)$ を求めよ。"
                    "また, 合成積定理から $\\mathcal F[f*f]$ を求めよ。")
        st.markdown("**問2** $f(t)=e^{-t}u(t)$ のとき, $(f*f)(t)$ と $\\mathcal F[f*f]$ を求めよ。")
        st.markdown("**問3** $\\displaystyle\\int_{-\\infty}^{\\infty}(f*g)\\,dt=\\int_{-\\infty}^{\\infty}f\\,dt\\int_{-\\infty}^{\\infty}g\\,dt$ を, "
                    "問1の $f$ で確かめよ。")
        if st.toggle("解答・解説を表示", key="pr_fourier3_ft"):
            st.markdown("**答1**")
            st.latex(r"(f*f)(t)=\int_{-1/2}^{1/2}f(t-\tau)\,d\tau=\begin{cases}1-|t|&(|t|\le1)\\0&(|t|>1)\end{cases},\qquad "
                     r"\mathcal F[f*f]=\left(\frac{2\sin(\omega/2)}{\omega}\right)^2=\frac{4\sin^2(\omega/2)}{\omega^2}")
            st.markdown("**答2**")
            st.latex(r"(f*f)(t)=\int_0^te^{-(t-\tau)}e^{-\tau}d\tau=te^{-t}\ (t\ge0),\ 0\ (t<0),\qquad \mathcal F[f*f]=\frac{1}{(1+i\omega)^2}")
            st.caption("$\\tau$ の積分範囲は, $f(\\tau)\\ne0$ と $f(t-\\tau)\\ne0$ の両方をみたす $0\\le\\tau\\le t$。")
            st.markdown("**答3**")
            st.write("三角形 1 − |t| の面積は 1、f の面積は 1 なので 1 = 1 × 1。")
