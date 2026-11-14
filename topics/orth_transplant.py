"""⑦ 特殊関数 第2章 直交関数系:9. 直交関数系どうしの移植

1. 移植とは  2. 三角関数系 ↔ チェビシェフ(x = cos θ)  3. ゲーゲンバウアー族(ルジャンドル ↔ チェビシェフ)
4. エルミート ↔ ラゲール(t = x²)  5. ラゲール ↔ フーリエ(双一次変換)  6. 係数の変換と経済化
7. 移植定理(L^p での有界性, ルジャンドル級数の平均収束 4/3 < p < 4 と A_p 重み)
書誌・史実は 2026 年 11 月に AMS・MSP・EuDML・DLMF などで確認。
"""
from fractions import Fraction

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.fft import dct
from scipy.special import eval_gegenbauer, eval_genlaguerre, gammaln

from common import C1, C2, C3, C4, GRAY, layout, show
from topics.fourier_apps import refs
from topics.fourier_kernels import page_refs
from topics.orth_core import POLY, expansion, p_eval, p_tex, trim

F = Fraction
FUN = {
    "e^x": (np.exp, "e^x"),
    "|x|": (np.abs, "|x|"),
    "1/(1 + 25x²)": (lambda x: 1 / (1 + 25 * x * x), r"\frac{1}{1+25x^2}"),
    "階段 H(x − 0.3)": (lambda x: (x > 0.3).astype(float), r"H(x-0.3)"),
}


# ------------------------------------------------------------
# 計算
# ------------------------------------------------------------
def cheb_coefs_integral(f, N, M=40000):
    th = (np.arange(M) + 0.5) * np.pi / M
    fc = f(np.cos(th))
    return np.array([(1 if n == 0 else 2) / M * np.sum(fc * np.cos(n * th)) for n in range(N)])


def cheb_coefs_dct(f, N, M=64):
    """チェビシェフ点 x_k = cos((k+½)π/M) の値から DCT-II で係数を求める"""
    th = (np.arange(M) + 0.5) * np.pi / M
    y = dct(f(np.cos(th)), type=2) / M
    y[0] /= 2
    return y[:N]


def gegen_fn(n, lam, th):
    """正規直交な超球関数 ψ_n^λ(θ) = (sin θ)^λ C_n^λ(cos θ) / √h_n(λ = 0 は √(2/π) cos nθ, n = 0 は 1/√π)"""
    if lam == 0:
        return np.cos(n * th) * (np.sqrt(2 / np.pi) if n else 1 / np.sqrt(np.pi))
    logh = np.log(np.pi) + (1 - 2 * lam) * np.log(2) + gammaln(n + 2 * lam) - np.log(n + lam) - 2 * gammaln(lam) - gammaln(n + 1)
    return np.sin(th) ** lam * eval_gegenbauer(n, lam, np.cos(th)) / np.exp(logh / 2)


def gegen_asym(n, lam, th):
    """セゲーの漸近形 √(2/π) cos((n+λ)θ − λπ/2)"""
    return np.sqrt(2 / np.pi) * np.cos((n + lam) * th - lam * np.pi / 2)


def lag_coef(a, n):
    """e^{−at} をラゲール関数 ℓ_n = e^{−t/2} L_n で展開した係数 (a − ½)^n / (a + ½)^{n+1}"""
    a = F(a)
    return (a - F(1, 2)) ** n / (a + F(1, 2)) ** (n + 1)


def taylor_exp(N):
    return [F(1, int(np.prod(range(1, k + 1)))) for k in range(N + 1)]


def economize(N, m):
    """e^x の N 次テイラー多項式をチェビシェフ展開し, m 次より上を捨てた多項式(厳密)"""
    c = expansion(taylor_exp(N), "chebyshev")
    q = [F(0)]
    for k in range(m + 1):
        T = POLY["chebyshev"](k)
        q = [(q[i] if i < len(q) else 0) + (c[k] * T[i] if i < len(T) else 0) for i in range(max(len(q), len(T)))]
    return trim(q), c


@st.cache_data(show_spinner=False)
def legendre_fn_matrix(N, M):
    """θ の格子(中点)と正規直交なルジャンドル関数 ψ_n^{1/2}(θ) = √((2n+1)/2) P_n(cos θ) √(sin θ), n < N"""
    th = (np.arange(M) + 0.5) * np.pi / M
    x = np.cos(th)
    P = np.zeros((N, M))
    P[0] = 1
    if N > 1:
        P[1] = x
    for n in range(1, N - 1):
        P[n + 1] = ((2 * n + 1) * x * P[n] - n * P[n - 1]) / (n + 1)
    return th, P * np.sqrt((2 * np.arange(N) + 1) / 2)[:, None] * np.sqrt(np.sin(th))


TG = {
    "区間 (π/4, 3π/4) の矩形": lambda t: ((t > np.pi / 4) & (t < 3 * np.pi / 4)).astype(float),
    "端の近くの矩形 (0, π/6)": lambda t: (t < np.pi / 6).astype(float),
    "θ(π − θ)": lambda t: t * (np.pi - t),
    "特異点つき |θ − π/2|^{−1/8}": lambda t: np.abs(t - np.pi / 2) ** -0.125,
}
PS = [1.25, 1.5, 2.0, 3.0, 4.0, 6.0]


@st.cache_data(show_spinner=False)
def transplant(gname, N=200, M=20000):
    """ルジャンドル関数での部分和 S_N g と, その移植 T S_N g = Σ a_n ψ_n^0"""
    th, Psi = legendre_fn_matrix(N, M)
    g = TG[gname](th)
    a = Psi @ g * (np.pi / M)
    Sg = a @ Psi
    Cos = np.array([gegen_fn(n, 0, th) for n in range(N)])
    Tg = a @ Cos
    ratios = {p: float((np.mean(np.abs(Tg) ** p) / np.mean(np.abs(Sg) ** p)) ** (1 / p)) for p in PS}
    return th, Sg, Tg, ratios


@st.cache_data(show_spinner=False)
def delta_norms(ns, ps):
    """1項だけを取り出す作用素 Δ_n f = (2n+1)/2 (f, P_n) P_n の L^p ノルム = (2n+1)/2 ‖P_n‖_p ‖P_n‖_{p'}"""
    x, w = np.polynomial.legendre.leggauss(3000)
    out = {}
    for p in ps:
        q = p / (p - 1)
        out[p] = [(2 * n + 1) / 2 * np.sum(w * np.abs(p_eval("legendre", n, x)) ** p) ** (1 / p) * np.sum(w * np.abs(p_eval("legendre", n, x)) ** q) ** (1 / q)
                  for n in ns]
    return out


# ------------------------------------------------------------
# ページ
# ------------------------------------------------------------
def tab_transplant():
    st.info("ある直交関数系での展開係数 $a_n$ を, そのまま別の直交関数系に付け替えることを**移植**(transplantation)という。"
            "変数変換で関数系どうしが**ぴったり移る**場合(三角関数系 ↔ チェビシェフなど)と, 形が**似ているだけ**の場合がある。"
            "後者でも, 移植した関数の大きさ($L^p$ ノルム)が元と比べられることを示すのが**移植定理**で, 調和解析の重要な道具である。")

    # ---------------------------------------------------------------- 1
    st.subheader("1. 移植とは")
    st.markdown("区間 $(0,\\pi)$ の2つの正規直交系 $\\{\\varphi_n\\}$, $\\{\\psi_n\\}$ に対して")
    st.latex(r"g=\sum_n a_n\varphi_n\quad\longmapsto\quad Tg=\sum_n a_n\psi_n\qquad\left(a_n=\int_0^\pi g\,\varphi_n\,d\theta\right)")
    st.markdown("を考える。パーセバルの等式から $\\lVert Tg\\rVert_2=\\lVert g\\rVert_2=\\bigl(\\sum|a_n|^2\\bigr)^{1/2}$ なので, **$L^2$ では大きさが変わらない**(ベクトルの正規直交基底の取りかえと同じ)。"
                "問題は $p\\neq2$ の $L^p$ ノルム $\\lVert g\\rVert_p=\\left(\\int|g|^pd\\theta\\right)^{1/p}$ で, これは係数だけでは決まらない。7. で扱う。")

    # ---------------------------------------------------------------- 2
    st.subheader("2. 三角関数系 ↔ チェビシェフ多項式:$x=\\cos\\theta$")
    st.markdown("$x=\\cos\\theta$ と置くと, 区間 $[-1,1]$ の関数 $f(x)$ は $F(\\theta)=f(\\cos\\theta)$ という**偶関数で周期 $2\\pi$ の関数**に移る。")
    st.latex(r"\cos n\theta\ \longleftrightarrow\ T_n(x),\qquad \frac{\sin(n+1)\theta}{\sin\theta}\ \longleftrightarrow\ U_n(x)\ (\text{第2種}),\qquad d\theta=-\frac{dx}{\sqrt{1-x^2}}")
    st.markdown("重みも一緒に移り, $T_n$ は $\\dfrac1{\\sqrt{1-x^2}}$, $U_n$ は $\\sqrt{1-x^2}$ の重みで直交する。"
                "**チェビシェフ係数 = $F(\\theta)$ のフーリエ余弦係数**であり, チェビシェフ点 $x_k=\\cos\\dfrac{(2k+1)\\pi}{2M}$ での値から**離散コサイン変換(DCT)**で速く計算できる。")
    c = st.columns([1, 2])
    fname = c[0].selectbox("関数 f(x)", list(FUN), key="otp_f")
    f, ftex = FUN[fname]
    ci, cd = cheb_coefs_integral(f, 8), cheb_coefs_dct(f, 8)
    c[0].markdown("| $n$ | 積分 $\\frac2\\pi\\int_0^\\pi F\\cos n\\theta\\,d\\theta$ | DCT($M=64$) |\n|---|---|---|\n" +
                  "\n".join(f"| ${n}$ | ${ci[n]:.6f}$ | ${cd[n]:.6f}$ |" for n in range(6)))
    th = np.linspace(-2 * np.pi, 2 * np.pi, 2001)
    fig = go.Figure(go.Scatter(x=th, y=f(np.cos(th)), name="F(θ) = f(cos θ)", line=dict(color=C1)))
    fig.add_vrect(x0=0, x1=np.pi, fillcolor="rgba(150,150,150,0.12)", line_width=0, annotation_text="θ ∈ [0, π] ↔ x ∈ [1, −1]")
    layout(fig, "x = cos θ で移した関数(偶関数・周期 2π)", xlab="θ", height=320)
    with c[1]:
        st.latex(f"f(x)={ftex}")
        show(fig)
    st.caption("$f$ が $[-1,1]$ でなめらかなら, $F$ は周期的につないでも**とびも折れ目もない**。"
               "これが, 8. の実験で $e^x$ のチェビシェフ展開が速く収束し, 三角関数系で直接展開するとギブス現象が起きた理由である。")

    # ---------------------------------------------------------------- 3
    st.subheader("3. ゲーゲンバウアー族:ルジャンドル ↔ チェビシェフ")
    st.markdown("重み $(1-x^2)^{\\lambda-1/2}$ で直交する多項式を**ゲーゲンバウアー(超球)多項式** $C_n^{(\\lambda)}(x)$ という。")
    st.latex(r"P_n=C_n^{(1/2)},\qquad U_n=C_n^{(1)},\qquad \lim_{\lambda\to0}\frac{n+\lambda}{\lambda}C_n^{(\lambda)}(x)=2T_n(x)\quad(n\ge1)")
    st.markdown("$\\lambda$ を動かすと, チェビシェフ($\\lambda\\to0$)→ ルジャンドル($\\lambda=\\frac12$)→ 第2種チェビシェフ($\\lambda=1$)と**連続的に移り変わる**。"
                "$\\theta$ の関数 $\\psi_n^{\\lambda}(\\theta)=c_n(\\sin\\theta)^{\\lambda}C_n^{(\\lambda)}(\\cos\\theta)$ を正規直交にとると, $n$ が大きいとき")
    st.latex(r"\psi_n^{\lambda}(\theta)\approx\sqrt{\frac2\pi}\,\cos\!\Bigl(\bigl(n+\lambda\bigr)\theta-\frac{\lambda\pi}{2}\Bigr)\qquad(\text{セゲーの漸近形})")
    st.markdown("となり, **少しずれた余弦関数**に見える($\\lambda=0,1$ では等号)。これが 7. の移植定理の出発点である。")
    c = st.columns([1, 2])
    lam = c[0].select_slider("λ", [0.0, 0.25, 0.5, 1.0, 1.5, 2.0], 0.5, key="otp_lam")
    n = c[0].select_slider("n", [1, 2, 4, 8, 16], 4, key="otp_n")
    name = {0.0: "チェビシェフ T_n", 0.5: "ルジャンドル P_n", 1.0: "第2種チェビシェフ U_n"}.get(lam, f"ゲーゲンバウアー(λ = {lam})")
    c[0].markdown(f"**{name}**")
    t = np.linspace(1e-4, np.pi - 1e-4, 1501)
    fig = go.Figure(go.Scatter(x=t, y=gegen_fn(n, lam, t), name="ψ_n^λ(θ)(正規直交)", line=dict(color=C1, width=3)))
    fig.add_trace(go.Scatter(x=t, y=gegen_asym(n, lam, t), name="√(2/π) cos((n+λ)θ − λπ/2)", line=dict(color=C2, dash="dash")))
    layout(fig, f"n = {n}, λ = {lam} の超球関数と余弦関数", xlab="θ", height=340)
    with c[1]:
        show(fig)

    # ---------------------------------------------------------------- 4
    st.subheader("4. エルミート ↔ ラゲール:$t=x^2$")
    st.markdown("実数全体の**偶関数・奇関数**は, $t=x^2$ で半直線 $[0,\\infty)$ の関数に移る。エルミート多項式は")
    st.latex(r"H_{2n}(x)=(-1)^n2^{2n}n!\,L_n^{(-1/2)}(x^2),\qquad H_{2n+1}(x)=(-1)^n2^{2n+1}n!\,x\,L_n^{(1/2)}(x^2)")
    st.markdown("とラゲールの陪多項式 $L_n^{(\\alpha)}$(重み $t^{\\alpha}e^{-t}$)で表せる。重みも $dx=\\dfrac{dt}{2\\sqrt t}$ によって $e^{-x^2}dx\\ \\to\\ \\dfrac12t^{-1/2}e^{-t}dt$ と移る。")
    c = st.columns([1, 2])
    m = c[0].select_slider("n(H_{2n} と H_{2n+1})", [0, 1, 2, 3, 4, 5], 2, key="otp_hl")
    x = np.linspace(-3, 3, 1201)
    fact = float(np.prod(range(1, m + 1)))
    lhs_e, lhs_o = p_eval("hermite", 2 * m, x), p_eval("hermite", 2 * m + 1, x)
    rhs_e = (-1) ** m * 4 ** m * fact * eval_genlaguerre(m, -0.5, x * x)
    rhs_o = (-1) ** m * 2 * 4 ** m * fact * x * eval_genlaguerre(m, 0.5, x * x)
    c[0].metric("最大の差(偶)", f"{float(np.max(np.abs(lhs_e - rhs_e))):.1e}")
    c[0].metric("最大の差(奇)", f"{float(np.max(np.abs(lhs_o - rhs_o))):.1e}")
    g = np.exp(-x * x / 2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=lhs_e * g, name=f"H_{2 * m}(x) e^(−x²/2)", line=dict(color=C1, width=4)))
    fig.add_trace(go.Scatter(x=x, y=rhs_e * g, name="ラゲールで表した式(偶)", line=dict(color=C3, dash="dash")))
    fig.add_trace(go.Scatter(x=x, y=lhs_o * g, name=f"H_{2 * m + 1}(x) e^(−x²/2)", line=dict(color=C2, width=4)))
    fig.add_trace(go.Scatter(x=x, y=rhs_o * g, name="ラゲールで表した式(奇)", line=dict(color=C4, dash="dash")))
    layout(fig, "エルミート多項式とラゲールの陪多項式(重なる)", xlab="x", height=360)
    with c[1]:
        show(fig)
    st.caption("2次元・3次元の調和振動子を極座標で解くと, 動径方向にラゲールの陪多項式が現れるのは, この関係による。")

    # ---------------------------------------------------------------- 5
    st.subheader("5. ラゲール ↔ フーリエ:双一次変換")
    st.markdown("ラゲール関数 $\\ell_n(t)=e^{-t/2}L_n(t)$ のラプラス変換は $\\mathcal L[\\ell_n](s)=\\dfrac{(s-\\frac12)^n}{(s+\\frac12)^{n+1}}$ だった。そこで")
    st.latex(r"z=\frac{s-\frac12}{s+\frac12}\qquad\Bigl(s=\frac{1+z}{2(1-z)}\Bigr)")
    st.markdown("とおくと, $\\left(s+\\frac12\\right)\\mathcal L[\\ell_n](s)=z^n$。つまり $g=\\sum c_n\\ell_n$ のとき")
    st.latex(r"\Bigl(s+\frac12\Bigr)G(s)=\sum_{n=0}^{\infty}c_n z^n")
    st.markdown("で, **ラゲール係数は $z$ のべき級数の係数**に移る。$s=i\\omega$(周波数応答)は $|z|=1$(単位円)に移るので, "
                "$z=e^{i\\phi}$ とすれば**フーリエ級数の係数**そのものである。電気回路のラゲール・ネットワークや, 連続時間 ↔ 離散時間の双一次変換(タスティン変換)と同じしくみである。")
    st.markdown("**例**:$g(t)=e^{-at}$ では $G(s)=\\dfrac1{s+a}$ から")
    st.latex(r"\Bigl(s+\frac12\Bigr)G(s)=\frac{2}{(1+2a)+(1-2a)z}\quad\Longrightarrow\quad c_n=\frac{(a-\frac12)^n}{(a+\frac12)^{n+1}}\quad(\text{等比数列})")
    c = st.columns([1, 2])
    a = c[0].select_slider("減衰の速さ a", [F(1, 8), F(1, 4), F(1, 2), F(1), F(2), F(4)], F(2), format_func=lambda v: str(v), key="otp_a")
    N = c[0].select_slider("使う項の数 N", [1, 2, 4, 8, 16], 4, key="otp_N")
    co = [lag_coef(a, k) for k in range(N)]
    r = (a - F(1, 2)) / (a + F(1, 2))
    c[0].latex(rf"c_n=\frac{{2}}{{{p_tex([1 + 2 * a])}}}\left({p_tex([r])}\right)^n" if r else r"c_0=1,\ c_n=0\ (n\ge1)")
    c[0].caption("$a=\\frac12$ なら1項で厳密。$a$ が $\\frac12$ から離れるほど公比 $\\left|\\frac{a-1/2}{a+1/2}\\right|$ が $1$ に近づき, 多くの項が必要になる"
                 "(実際には時間の尺度を $e^{-pt/2}L_n(pt)$ のように合わせる)。")
    tt = np.linspace(0, 12, 1201)
    Lf = np.array([np.exp(-tt / 2) * p_eval("laguerre", k, tt) for k in range(N)])
    fig = go.Figure(go.Scatter(x=tt, y=np.exp(-float(a) * tt), name="e^(−at)", line=dict(color=GRAY, width=4)))
    fig.add_trace(go.Scatter(x=tt, y=np.array([float(v) for v in co]) @ Lf, name=f"ラゲール関数 {N} 項", line=dict(color=C1)))
    layout(fig, "減衰する応答のラゲール展開", xlab="t", height=320)
    with c[1]:
        show(fig)

    # ---------------------------------------------------------------- 6
    st.subheader("6. 係数の変換と経済化")
    st.markdown("基底を取りかえると係数も変わる。べき $x^n$ とチェビシェフ多項式の間には")
    st.latex(r"x^n=2^{1-n}\sum_{0\le k<n/2}\binom nkT_{n-2k}(x)\ \Bigl(+\,2^{-n}\binom{n}{n/2}T_0\ \ (n\ \text{偶数})\Bigr)")
    st.latex(r"\begin{aligned}" + r"\\".join(f"x^{{{k}}}&=" + "+".join(f"{p_tex([v])}T_{{{j}}}" for j, v in enumerate(expansion([F(0)] * k + [F(1)], 'chebyshev')) if v)
                                             for k in range(2, 6)) + r"\end{aligned}")
    st.markdown("$T_n$ の最大値は $1$ なので, 係数がそのまま「その項を捨てたときの誤差」の大きさになる。"
                "そこで, テイラー多項式をチェビシェフ多項式で書き直し, 係数の小さい高次の項を捨てると, **次数を下げても最大の誤差がほとんど増えない**。これを**経済化**という。")
    c = st.columns([1, 2])
    Ntay = 6
    mdeg = c[0].select_slider("経済化した後の次数 m", [2, 3, 4, 5], 3, key="otp_m")
    q, cc = economize(Ntay, mdeg)
    xs = np.linspace(-1, 1, 2001)
    tay_m = np.polynomial.polynomial.polyval(xs, [float(v) for v in taylor_exp(mdeg)])
    eco = np.polynomial.polynomial.polyval(xs, [float(v) for v in q])
    e_tay, e_eco = float(np.max(np.abs(np.exp(xs) - tay_m))), float(np.max(np.abs(np.exp(xs) - eco)))
    c[0].latex(rf"e^x\approx {p_tex(q)}")
    c[0].metric(f"{mdeg} 次テイラー多項式の最大誤差", f"{e_tay:.2e}")
    c[0].metric(f"経済化した {mdeg} 次多項式の最大誤差", f"{e_eco:.2e}")
    fig = go.Figure(go.Scatter(x=xs, y=np.exp(xs) - tay_m, name=f"{mdeg} 次テイラー", line=dict(color=C2)))
    fig.add_trace(go.Scatter(x=xs, y=np.exp(xs) - eco, name=f"経済化({Ntay} 次 → {mdeg} 次)", line=dict(color=C1)))
    layout(fig, "誤差 e^x − 近似(区間 [−1, 1])", xlab="x", height=320)
    with c[1]:
        show(fig)
    st.caption("テイラー多項式は $x=0$ の近くだけ正確で, 端で誤差が大きい。経済化した多項式の誤差は区間全体で均等に振動する。")

    # ---------------------------------------------------------------- 7
    st.subheader("7. 移植定理:$L^p$ での大きさを比べる")
    st.markdown("**7.1 定理**  超球関数 $\\psi_n^{\\lambda}$ から余弦関数 $\\psi_n^{0}$ への移植 $T:\\sum a_n\\psi_n^{\\lambda}\\mapsto\\sum a_n\\psi_n^0$ を考える。"
                "アスキーとウェインガー(R. Askey, S. Wainger, 1966)は, $\\lambda\\ge0$, $1<p<\\infty$ のとき")
    st.latex(r"A\,\Bigl\lVert\sum a_n\psi_n^{\lambda}\Bigr\rVert_{p}\ \le\ \Bigl\lVert\sum a_n\psi_n^{0}\Bigr\rVert_{p}\ \le\ B\,\Bigl\lVert\sum a_n\psi_n^{\lambda}\Bigr\rVert_{p}")
    st.markdown("($A,B$ は $f$ によらない定数)の形の不等式を示した(係数の重みつき $\\ell^p$ 版と, 級数の $L^p$ 版)。"
                "アスキー(1967)がヤコビ係数に, マッケンハウプト(B. Muckenhoupt, 1986)がヤコビ級数に, **べき重みつきの $L^p$** まで含めて一般化した。"
                "重みの正確な範囲は原論文を参照。ハンケル変換についてのガイ(D. L. Guy, 1960)の結果が先がけである。")
    st.markdown("**7.2 数値実験**  ルジャンドル関数 $\\psi_n^{1/2}(\\theta)=\\sqrt{\\frac{2n+1}{2}}P_n(\\cos\\theta)\\sqrt{\\sin\\theta}$ で展開した部分和 $S_Ng$($N=200$)を, "
                "余弦関数に移植した $TS_Ng$ と比べる。$p=2$ では比がちょうど $1$, ほかの $p$ でも比が極端にならないことを確かめる。")
    c = st.columns([1, 2])
    gname = c[0].selectbox("関数 g(θ)", list(TG), key="otp_g")
    tht, Sg, Tg, ratios = transplant(gname)
    c[0].markdown("| $p$ | $\\lVert TS_Ng\\rVert_p\\,/\\,\\lVert S_Ng\\rVert_p$ |\n|---|---|\n" + "\n".join(f"| ${p:g}$ | ${v:.4f}$ |" for p, v in ratios.items()))
    fig = go.Figure(go.Scatter(x=tht, y=Sg, name="S_N g(ルジャンドル関数の部分和)", line=dict(color=C1)))
    fig.add_trace(go.Scatter(x=tht, y=Tg, name="T S_N g(余弦関数に移植)", line=dict(color=C2)))
    layout(fig, "移植の前と後(係数は同じ)", xlab="θ", height=360)
    with c[1]:
        show(fig)

    st.markdown("**7.3 応用:ルジャンドル級数はどの $p$ で収束するか**")
    st.markdown("フーリエ級数の部分和は $1<p<\\infty$ のすべての $L^p$ で収束する(M. リース, 1928)。ところが**ルジャンドル級数は $L^p(-1,1)$ で $\\frac43<p<4$ のときだけ**収束する"
                "(ポラード(H. Pollard, 1947)が十分性を, ニューマンとルーディン(J. Newman, W. Rudin, 1952)が必要性を示した)。範囲が狭くなる理由は, 移植で説明できる:")
    st.markdown("1. $f\\in L^p(-1,1)$ を $g(\\theta)=f(\\cos\\theta)\\sqrt{\\sin\\theta}$ に移すと, $dx=\\sin\\theta\\,d\\theta$ より")
    st.latex(r"\int_{-1}^{1}|f|^p\,dx=\int_0^{\pi}|g(\theta)|^p\,(\sin\theta)^{1-p/2}\,d\theta")
    st.markdown("   となり, **重み $w(\\theta)=(\\sin\\theta)^{1-p/2}$ つきの $L^p$** の問題になる($p=2$ のときだけ重みが消える)。\n"
                "2. 移植定理(重みつき)で, ルジャンドル関数の部分和の問題は**余弦級数の部分和**の問題に移る。\n"
                "3. 余弦級数(フーリエ級数)の部分和が重み $w$ つきの $L^p$ で有界になるのは, $w$ が **$A_p$ 条件**をみたすとき, そのときに限る"
                "(ハント・マッケンハウプト・ウィーデン, 1973)。")
    st.markdown("**べき重みの $A_p$ 条件**:原点の近くで $w(\\theta)=\\theta^{a}$ のとき, 区間 $I=(0,h)$ での平均は")
    st.latex(r"\frac1h\int_0^h\theta^{a}d\theta=\frac{h^{a}}{a+1}\ (a>-1),\qquad \frac1h\int_0^h\theta^{-\frac{a}{p-1}}d\theta=\frac{h^{-\frac{a}{p-1}}}{1-\frac{a}{p-1}}\ \Bigl(a<p-1\Bigr)")
    st.markdown("で, 積 $\\Bigl(\\frac1{|I|}\\int_Iw\\Bigr)\\Bigl(\\frac1{|I|}\\int_Iw^{-\\frac1{p-1}}\\Bigr)^{p-1}$ は $h$ によらず有界になる。"
                "よって条件は $-1<a<p-1$。いまの $a=1-\\frac p2$ を入れると")
    st.latex(r"-1<1-\frac p2<p-1\quad\Longleftrightarrow\quad\frac43<p<4")
    st.markdown("となり, ポラードの範囲がちょうど出てくる($\\theta=\\pi$ の近くも同じ)。")
    c = st.columns([1, 2])
    pp = c[0].select_slider("p", [1.1, 1.25, 4 / 3, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0], 2.0, format_func=lambda v: "4/3" if abs(v - 4 / 3) < 1e-9 else f"{v:g}", key="otp_p")
    av = 1 - pp / 2
    okp = -1 < av < pp - 1 and abs(pp - 4 / 3) > 1e-9 and pp != 4.0
    c[0].latex(rf"a=1-\frac p2={av:.3f},\qquad -1<a<p-1={pp - 1:.3f}")
    (c[0].success if okp else c[0].error)("$A_p$ 条件をみたす → ルジャンドル級数は $L^p$ で収束する" if okp else "$A_p$ 条件をみたさない → $L^p$ で収束しない関数がある")
    P = np.linspace(1.01, 8, 400)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.concatenate([P, P[::-1]]), y=np.concatenate([P - 1, -np.ones_like(P)]), fill="toself", fillcolor="rgba(46,134,193,0.15)",
                             line=dict(width=0), name="A_p の範囲 −1 < a < p − 1"))
    fig.add_trace(go.Scatter(x=P, y=1 - P / 2, name="ルジャンドルの場合 a = 1 − p/2", line=dict(color=C2, width=3)))
    fig.add_trace(go.Scatter(x=[pp], y=[av], mode="markers", marker=dict(size=13, color="black"), name="選んだ p"))
    for v in (4 / 3, 4):
        fig.add_vline(x=v, line=dict(color=GRAY, dash="dot"))
    layout(fig, "べき重み θ^a が A_p になる範囲と, ルジャンドル級数の重み", xlab="p", ylab="a", height=360, yrange=[-3.2, 4])
    with c[1]:
        show(fig)
    st.markdown("**数値で確かめる**:$n$ 番目の項だけを取り出す作用素 $\\Delta_nf=\\frac{2n+1}{2}(f,P_n)P_n$ の $L^p$ での大きさ(作用素ノルム)は, ちょうど "
                "$\\frac{2n+1}{2}\\lVert P_n\\rVert_p\\lVert P_n\\rVert_{p'}$($\\frac1p+\\frac1{p'}=1$)である。"
                "$\\Delta_n=S_n-S_{n-1}$ なので, これが $n$ とともに大きくなるなら部分和 $S_n$ も有界にはなれない。")
    ns = [4, 8, 16, 32, 64, 128, 256]
    dn = delta_norms(tuple(ns), (2.0, 3.0, 4.0, 6.0, 10.0))
    fig = go.Figure()
    for p_, col in zip((2.0, 3.0, 4.0, 6.0, 10.0), (C1, C3, GRAY, C2, C4)):
        fig.add_trace(go.Scatter(x=ns, y=dn[p_], mode="lines+markers", name=f"p = {p_:g}", line=dict(color=col)))
    layout(fig, "Δ_n の L^p 作用素ノルム(両対数)", xlab="n", ylab="(2n+1)/2 ‖P_n‖_p ‖P_n‖_p'", height=340)
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    show(fig)
    st.caption("$p<4$ では有界, $p=4$ ではゆっくり($\\log$ 程度)増え, $p>4$ では $n^{1/2-2/p}$ の速さで増える。$p$ と $p'$ を入れかえても同じ値なので, $p<\\frac43$ も同様。")

    st.markdown("**7.4 歴史**")
    st.markdown(r"""
| 年 | できごと |
|---|---|
| 1928 | M. リースが共役関数の $L^p$ 有界性を示し, フーリエ級数の部分和が $1<p<\infty$ で $L^p$ 収束することがわかる |
| 1947 | ポラードがルジャンドル級数の $L^p$ 収束を $\frac43<p<4$ で示す |
| 1952 | ニューマンとルーディンが, ポラードの条件が必要でもあることを示す |
| 1960 | ガイがハンケル変換について移植の考え方を使う |
| 1966–67 | アスキーとウェインガーが超球級数の移植定理, アスキーがヤコビ係数の移植定理を示す |
| 1972–73 | マッケンハウプトの $A_p$ 条件, ハント・マッケンハウプト・ウィーデンの重みつき不等式 |
| 1986 | マッケンハウプトがヤコビ級数の重みつき移植定理とマルチプライヤー定理をまとめる |
""")
    st.caption("③ フーリエ解析の発展「マルチプライヤー」とも関係が深い(移植でフーリエ級数のマルチプライヤー定理をヤコビ級数に移せる)。")

    # ---------------------------------------------------------------- 8
    with st.expander("練習問題"):
        st.markdown("**問1** $x=\\cos\\theta$ を使って, $T_3(x)=4x^3-3x$ と $U_2(x)=\\dfrac{\\sin3\\theta}{\\sin\\theta}=4x^2-1$ を確かめよ。")
        st.markdown("**問2** $x^4$ をチェビシェフ多項式 $T_0,T_2,T_4$ で表せ。")
        st.markdown("**問3** $H_2(x)=4x^2-2$, $L_1^{(-1/2)}(t)=\\frac12-t$ として, $H_2(x)=-4L_1^{(-1/2)}(x^2)$ を確かめよ。")
        st.markdown("**問4** $g(t)=e^{-t}$ をラゲール関数 $\\ell_n(t)=e^{-t/2}L_n(t)$ で展開した係数 $c_0,c_1,c_2$ を求めよ。")
        st.markdown("**問5** $e^x$ の3次テイラー多項式 $1+x+\\frac{x^2}2+\\frac{x^3}6$ の $x^3$ を $\\frac14(3T_1+T_3)$ で書き直し, $T_3$ の項を捨てて2次式にせよ。捨てた項の大きさは高々いくらか。")
        st.markdown("**問6** ルジャンドル級数の重み $(\\sin\\theta)^{1-p/2}$ が $A_p$ になる条件 $-1<1-\\frac p2<p-1$ を解け。")
        if st.toggle("解答・解説を表示", key="pr_otp"):
            st.markdown("**答1** $\\cos3\\theta=4\\cos^3\\theta-3\\cos\\theta$。$\\sin3\\theta=3\\sin\\theta-4\\sin^3\\theta$ より $\\dfrac{\\sin3\\theta}{\\sin\\theta}=3-4(1-\\cos^2\\theta)=4x^2-1$。")
            st.markdown("**答2** $x^4=\\dfrac38T_0+\\dfrac12T_2+\\dfrac18T_4$。")
            st.markdown("**答3** $-4\\left(\\frac12-x^2\\right)=4x^2-2$(公式 $H_{2n}=(-1)^n2^{2n}n!\\,L_n^{(-1/2)}(x^2)$ で $n=1$)。")
            st.markdown("**答4** $a=1$ で $c_n=\\dfrac{(1/2)^n}{(3/2)^{n+1}}=\\dfrac23\\left(\\dfrac13\\right)^n$:$c_0=\\dfrac23$, $c_1=\\dfrac29$, $c_2=\\dfrac2{27}$。")
            st.markdown("**答5** $1+x+\\frac{x^2}{2}+\\frac16\\cdot\\frac{3T_1+T_3}{4}=1+\\frac98x+\\frac{x^2}2+\\frac1{24}T_3$。$T_3$ の項を捨てて $1+\\frac98x+\\frac12x^2$。"
                        "$|T_3|\\le1$ なので捨てた項は高々 $\\frac1{24}$(テイラーの2次式で $x^3$ の項を捨てると最大 $\\frac16$)。")
            st.markdown("**答6** $1-\\frac p2>-1$ より $p<4$。$1-\\frac p2<p-1$ より $p>\\frac43$。よって $\\frac43<p<4$。")

    page_refs([("askeywainger1966", "超球係数とフーリエ係数の移植定理。"), ("askeywainger1966b", "超球級数の L^p 移植定理。"), ("askey1967", "ヤコビ係数の移植定理。"),
               ("muckenhoupt1986", "ヤコビ級数の重みつき移植定理とマルチプライヤー定理。"), ("guy1960", "ハンケル変換の移植(先がけ)。"),
               ("pollard1947", "ルジャンドル級数の L^p 収束(4/3 < p < 4)。"), ("newmanrudin1952", "ポラードの条件の必要性。"),
               ("riesz1928", "共役関数の L^p 有界性(フーリエ級数の L^p 収束)。"), ("muckenhoupt1972", "A_p 条件。"), ("hmw1973", "重みつき L^p でのヒルベルト変換・共役関数。"),
               ("grafakos2014", "A_p 重みと重みつき不等式の教科書。"), ("masonhandscomb2002", "チェビシェフ多項式と経済化。"), ("szego1939", "超球多項式の漸近形。"),
               ("nist2010", "第18章:直交多項式どうしの関係式。")])
    refs([("DLMF §18.7 Interrelations and Limit Relations", "https://dlmf.nist.gov/18.7", "ゲーゲンバウアー・ルジャンドル・チェビシェフ, エルミート・ラゲールの関係式(英語)"),
          ("Askey–Wainger(1966, Pacific J. Math.)の論文 PDF", "https://msp.org/pjm/1966/16-3/pjm-v16-n3-p02-p.pdf", "超球係数の移植定理(英語)"),
          ("Askey(1967, Pacific J. Math.)の論文 PDF", "https://msp.org/pjm/1967/21-3/pjm-v21-n3-p01-p.pdf", "ヤコビ係数の移植定理(英語)"),
          ("Synthesis of electric networks by means of the Fourier transforms of Laguerre's functions(MIT, 1930)", "https://dspace.mit.edu/handle/1721.1/104477",
           "Y. W. Lee の学位論文:ラゲール関数による回路の合成(英語)")])
