"""⑦ 特殊関数 第2章 直交関数系:1. 微分方程式の級数解法, 2. 重みつき内積とグラム・シュミットの直交化

1. では y = Σ a_k x^k を代入して係数の漸化式を作り, 漸化式が途中で 0 になると多項式解が現れることを見る。
2. では 1, x, x², … を重みつき内積で直交化すると, ルジャンドル・チェビシェフ・エルミート・ラゲールの多項式が出てくることを見る。
"""
from fractions import Fraction

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.integrate import solve_ivp

from common import C1, C2, C3, C4, GRAY, layout, show
from topics.fourier_apps import refs
from topics.fourier_kernels import page_refs
from topics.orth_core import F, POLY, WEIGHTS, gram_schmidt, inner, p_tex

# ------------------------------------------------------------
# 1. 級数解法
# ------------------------------------------------------------
# name: (方程式 TeX, 漸化式 TeX, 次の係数 next(a, k, par), 階数, 右辺 rhs(x, y, par), x の範囲, パラメータ名)
EQS = {
    "y′ = y": (r"y^{\prime}=y", r"a_{k+1}=\dfrac{a_k}{k+1}", 1, (-4, 4), None),
    "y″ + y = 0": (r"y^{\prime\prime}+y=0", r"a_{k+2}=-\dfrac{a_k}{(k+2)(k+1)}", 2, (-8, 8), None),
    "y″ = x y(エアリーの方程式)": (r"y^{\prime\prime}=xy", r"a_2=0,\quad a_{k+2}=\dfrac{a_{k-1}}{(k+2)(k+1)}", 2, (-8, 4), None),
    "y″ − 2x y′ + 2λ y = 0(エルミートの方程式)": (r"y^{\prime\prime}-2xy^{\prime}+2\lambda y=0", r"a_{k+2}=\dfrac{2(k-\lambda)}{(k+2)(k+1)}a_k", 2, (-3, 3), "λ"),
    "(1 − x²) y″ − 2x y′ + ν(ν + 1) y = 0(ルジャンドルの方程式)":
        (r"(1-x^2)y^{\prime\prime}-2xy^{\prime}+\nu(\nu+1)y=0", r"a_{k+2}=\dfrac{(k-\nu)(k+\nu+1)}{(k+2)(k+1)}a_k", 2, (-0.995, 0.995), "ν"),
}


def series_coefs(name, a0, a1, par, N):
    """先頭 N 個の係数(Fraction)"""
    a = [F(0)] * N
    a[0] = F(a0)
    if N > 1:
        a[1] = F(a1) if EQS[name][2] == 2 else a[0]
    for k in range(N):
        if name == "y′ = y" and k + 1 < N:
            a[k + 1] = a[k] / (k + 1)
        elif name == "y″ + y = 0" and k + 2 < N:
            a[k + 2] = -a[k] / ((k + 2) * (k + 1))
        elif name.startswith("y″ = x y") and k + 2 < N:
            a[k + 2] = (a[k - 1] if k >= 1 else F(0)) / ((k + 2) * (k + 1))
        elif name.startswith("y″ − 2x") and k + 2 < N:
            a[k + 2] = 2 * (k - par) / ((k + 2) * (k + 1)) * a[k]
        elif name.startswith("(1 − x²)") and k + 2 < N:
            a[k + 2] = (k - par) * (k + par + 1) / ((k + 2) * (k + 1)) * a[k]
    return a


def series_tex(a, nterms):
    """昇べきに nterms 個の 0 でない項を並べた TeX"""
    out = ""
    for k, c in enumerate(a):
        if c == 0:
            continue
        if nterms == 0:
            break
        nterms -= 1
        body = "" if k == 0 else ("x" if k == 1 else f"x^{{{k}}}")
        m = abs(c)
        coef = "" if (m == 1 and body) else (str(m.numerator) if m.denominator == 1 else rf"\frac{{{m.numerator}}}{{{m.denominator}}}")
        out += ("-" if c < 0 else ("+" if out else "")) + coef + body
    return out or "0"


def rhs(name, par):
    if name == "y′ = y":
        return lambda x, y: [y[0]]
    if name == "y″ + y = 0":
        return lambda x, y: [y[1], -y[0]]
    if name.startswith("y″ = x y"):
        return lambda x, y: [y[1], x * y[0]]
    if name.startswith("y″ − 2x"):
        return lambda x, y: [y[1], 2 * x * y[1] - 2 * par * y[0]]
    return lambda x, y: [y[1], (2 * x * y[1] - par * (par + 1) * y[0]) / (1 - x * x)]


@st.cache_data(show_spinner=False)
def numeric_solution(name, a0, a1, par, lo, hi):
    """x = 0 から左右に数値計算(参照用)"""
    order = EQS[name][2]
    y0 = [a0] if order == 1 else [a0, a1]
    f = rhs(name, par)
    out = []
    for end in (lo, hi):
        sol = solve_ivp(f, (0, end), y0, dense_output=True, rtol=1e-10, atol=1e-12)
        xs = np.linspace(0, end, 400)
        out.append((xs, sol.sol(xs)[0]))
    xl, yl = out[0]
    xr, yr = out[1]
    return np.concatenate([xl[::-1], xr[1:]]), np.concatenate([yl[::-1], yr[1:]])


def tab_series():
    st.info("ルジャンドル多項式やエルミート多項式は, 物理や工学に現れる2階の線形微分方程式の解として登場する。"
            "このページでは, 解を $y=\\sum a_kx^k$ と**べき級数**の形で探す方法(級数解法)を学び, "
            "係数の漸化式が途中で $0$ になると**多項式の解**が現れることを見る。この章の各ページの「微分方程式」の節は, この方法で導く。")

    st.subheader("1. 考え方:係数を比べて漸化式を作る")
    st.markdown("解を $y=a_0+a_1x+a_2x^2+\\cdots=\\sum_{k=0}^{\\infty}a_kx^k$ とおくと, 項ごとに微分して")
    st.latex(r"y^{\prime}=\sum_{k=0}^{\infty}(k+1)a_{k+1}x^k,\qquad y^{\prime\prime}=\sum_{k=0}^{\infty}(k+2)(k+1)a_{k+2}x^k")
    st.markdown("($x^k$ の係数がそろうように番号をずらした)。これを方程式に代入し, **両辺の $x^k$ の係数が等しい**とおくと, 係数の漸化式が得られる。")
    st.markdown("**例1** $y^{\\prime}=y$:$(k+1)a_{k+1}=a_k$ より $a_k=\\dfrac{a_0}{k!}$。よって $y=a_0\\sum\\dfrac{x^k}{k!}=a_0e^x$。\n\n"
                "**例2** $y^{\\prime\\prime}+y=0$:$(k+2)(k+1)a_{k+2}+a_k=0$。偶数番目は $a_0$ から, 奇数番目は $a_1$ から決まり,")
    st.latex(r"y=a_0\Bigl(1-\frac{x^2}{2!}+\frac{x^4}{4!}-\cdots\Bigr)+a_1\Bigl(x-\frac{x^3}{3!}+\frac{x^5}{5!}-\cdots\Bigr)=a_0\cos x+a_1\sin x")
    st.markdown("2階の方程式では $a_0=y(0)$, $a_1=y^{\\prime}(0)$ を自由に選べ, 2つの独立な解が得られる。")

    st.subheader("2. 部分和と数値解を比べる")
    c = st.columns([1, 2])
    name = c[0].selectbox("方程式", list(EQS), index=3, key="os_eq")
    eq_tex, rec_tex, order, (lo, hi), pname = EQS[name]
    init = c[0].radio("初期値", ["a₀ = 1, a₁ = 0", "a₀ = 0, a₁ = 1"], horizontal=True, key="os_init") if order == 2 else "a₀ = 1, a₁ = 0"
    a0, a1 = (1, 0) if init.startswith("a₀ = 1") else (0, 1)
    par = F(0)
    if pname == "λ":
        par = F(c[0].select_slider("λ", ["0", "1/2", "1", "3/2", "2", "5/2", "3", "4"], "2", key="os_lam"))
    elif pname == "ν":
        par = F(c[0].select_slider("ν", ["0", "1/2", "1", "3/2", "2", "5/2", "3", "4"], "2", key="os_nu"))
    N = c[0].select_slider("項の数 N", [2, 4, 6, 8, 12, 16, 24, 32, 48], 8, key="os_N")
    a = series_coefs(name, a0, a1, par, N)
    c[0].latex(eq_tex)
    c[0].latex(rec_tex)
    long = series_coefs(name, a0, a1, par, 64)
    last = max([k for k, v in enumerate(long) if v != 0], default=0)
    terminates = last < 40
    c[0].latex("y=" + series_tex(long[: last + 1] if terminates else long, 99 if terminates else 4) + ("" if terminates else r"+\cdots"))
    if pname and terminates:
        sym = r"\lambda" if pname == "λ" else r"\nu"
        c[0].success(f"${sym}={par}$ では $a_{{{last + 2}}}$ から先の係数がすべて $0$ になり, 解は**多項式**になる。")
    x = np.linspace(lo, hi, 801)
    S = np.polynomial.polynomial.polyval(x, [float(v) for v in a])
    xn, yn = numeric_solution(name, a0, a1, float(par), lo, hi)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xn, y=yn, name="数値解(参照)", line=dict(color=GRAY, width=4)))
    fig.add_trace(go.Scatter(x=x, y=S, name=f"級数の部分和(N = {N})", line=dict(color=C1, dash="dash")))
    ymax = float(np.nanmax(np.abs(yn))) * 1.3 + 0.5
    layout(fig, "べき級数の部分和と数値解", xlab="x", ylab="y", height=400, yrange=[-ymax, ymax])
    with c[1]:
        show(fig)
    st.caption("エルミートの方程式で $\\lambda$ を整数からずらすと, 解は $x$ が大きいところで $e^{x^2}$ のように急に増える。"
               "ルジャンドルの方程式で $\\nu$ を整数からずらすと, $x=\\pm1$ の近くで解が発散する。")

    st.subheader("3. 多項式の解が現れるしくみ")
    st.markdown("この章の4つの方程式は, どれも漸化式に $(k-n)$ という因数をもつ。$k=n$ になったところで次の係数が $0$ になり, "
                "その先がすべて $0$ になるので, **$n$ 次の多項式**が解になる(もう一方の系列は止まらないので, $a_0$ と $a_1$ の一方を $0$ にしておく)。")
    st.markdown(r"""
| 方程式 | 係数の漸化式 | 多項式の解 |
|---|---|---|
| ルジャンドル $(1-x^2)y^{\prime\prime}-2xy^{\prime}+n(n+1)y=0$ | $a_{k+2}=\dfrac{(k-n)(k+n+1)}{(k+2)(k+1)}a_k$ | $P_n(x)$ |
| チェビシェフ $(1-x^2)y^{\prime\prime}-xy^{\prime}+n^2y=0$ | $a_{k+2}=\dfrac{(k-n)(k+n)}{(k+2)(k+1)}a_k$ | $T_n(x)$ |
| エルミート $y^{\prime\prime}-2xy^{\prime}+2ny=0$ | $a_{k+2}=\dfrac{2(k-n)}{(k+2)(k+1)}a_k$ | $H_n(x)$ |
| ラゲール $xy^{\prime\prime}+(1-x)y^{\prime}+ny=0$ | $a_{k+1}=\dfrac{k-n}{(k+1)^2}a_k$ | $L_n(x)$ |
""")
    st.markdown("**なぜ整数 $n$ が大切か**:漸化式が止まらないと, 級数は")
    st.markdown("- ルジャンドル・チェビシェフでは $\\left|\\dfrac{a_{k+2}}{a_k}\\right|\\to1$ なので収束半径が $1$ で, $x=\\pm1$(方程式の係数 $1-x^2$ が $0$ になる点)で発散する。"
                "球の北極・南極に当たる点で有限な解を求めると, $n$ は整数に限られる(偏微分方程式のラプラス方程式を球座標で解くときに現れる)。\n"
                "- エルミートでは $\\dfrac{a_{k+2}}{a_k}\\approx\\dfrac2k$ で, $e^{x^2}=\\sum\\dfrac{x^{2m}}{m!}$ と同じ増え方になる。"
                "量子力学の調和振動子では $\\psi=y\\,e^{-x^2/2}$ が遠くで $0$ になる必要があるので, $n$ が整数に限られ, **エネルギーがとびとびの値になる**。")

    st.subheader("4. 確定特異点とフロベニウスの方法(紹介)")
    st.markdown("ラゲールの方程式の $x=0$ やベッセルの方程式 $x^2y^{\\prime\\prime}+xy^{\\prime}+(x^2-\\nu^2)y=0$ の $x=0$ のように, "
                "$y^{\\prime\\prime}$ の係数が $0$ になる点(**確定特異点**)では, 解を $y=x^r\\sum a_kx^k$ の形で探す。"
                "最低次の係数を比べると $r$ の2次方程式(決定方程式)が得られる。ベッセルの方程式では $r^2-\\nu^2=0$ より $r=\\pm\\nu$。"
                "ラゲールの方程式では $r^2=0$ で重解になり, ふつうのべき級数の解 $L_n$ のほかに, $\\log x$ を含む解が現れる。")
    st.caption("この方法はフロベニウス(G. Frobenius)の 1873 年の論文で整理された。")

    with st.expander("練習問題"):
        st.markdown("**問1** $y^{\\prime}=2y$, $y(0)=1$ を級数解法で解け。")
        st.markdown("**問2** $y^{\\prime\\prime}-2xy^{\\prime}+6y=0$ について, $a_0=0$, $a_1=1$ の解を求めよ。")
        st.markdown("**問3** エアリーの方程式 $y^{\\prime\\prime}=xy$ の $a_0=1$, $a_1=0$ の解を $x^6$ の項まで求めよ。")
        st.markdown("**問4** $xy^{\\prime\\prime}+(1-x)y^{\\prime}+2y=0$ の $a_0=1$ の多項式解を求めよ。")
        if st.toggle("解答・解説を表示", key="pr_os"):
            st.markdown("**答1** $(k+1)a_{k+1}=2a_k$ より $a_k=\\dfrac{2^k}{k!}$, $y=e^{2x}$。")
            st.markdown("**答2** $\\lambda=3$。$a_3=\\dfrac{2(1-3)}{3\\cdot2}a_1=-\\dfrac23$, $a_5=\\dfrac{2(3-3)}{5\\cdot4}a_3=0$ より $y=x-\\dfrac23x^3$"
                        "($H_3=8x^3-12x$ の $-\\dfrac1{12}$ 倍)。")
            st.markdown("**答3** $a_2=0$, $a_3=\\dfrac{a_0}{3\\cdot2}=\\dfrac16$, $a_6=\\dfrac{a_3}{6\\cdot5}=\\dfrac1{180}$ より $y=1+\\dfrac{x^3}{6}+\\dfrac{x^6}{180}+\\cdots$。")
            st.markdown("**答4** $a_{k+1}=\\dfrac{k-2}{(k+1)^2}a_k$ より $a_1=-2$, $a_2=\\dfrac{-1}{4}(-2)=\\dfrac12$, $a_3=0$。"
                        "$y=1-2x+\\dfrac12x^2=L_2(x)$。")

    page_refs([("inui1962", "級数解法と特殊関数。"), ("frobenius1873", "確定特異点のまわりの級数解。"), ("szego1939", "直交多項式の満たす微分方程式。")])


# ------------------------------------------------------------
# 2. 重みつき内積とグラム・シュミットの直交化
# ------------------------------------------------------------
STD = {"legendre": ("P_n", r"\dfrac{(2n)!}{2^n(n!)^2}"), "chebyshev": ("T_n", r"2^{n-1}\ (n\ge1)"),
       "hermite": ("H_n", r"2^n"), "laguerre": ("L_n", r"\dfrac{(-1)^n}{n!}")}
PLOT_RANGE = {"legendre": (-1, 1), "chebyshev": (-1, 1), "hermite": (-3, 3), "laguerre": (0, 12), "unit01": (0, 1), "x01": (0, 1)}


def weight_fn(wname, x):
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"legendre": np.ones_like(x), "chebyshev": 1 / np.sqrt(np.clip(1 - x * x, 1e-12, None)),
                "hermite": np.exp(-x * x), "laguerre": np.exp(-x), "unit01": np.ones_like(x), "x01": x}[wname]


def gram_tex(ps, wname):
    const = WEIGHTS[wname][4]
    rows = []
    for p in ps:
        rows.append("&".join(p_tex([inner(p, q, wname)]) for q in ps))
    mat = r"\begin{pmatrix}" + r"\\".join(rows) + r"\end{pmatrix}"
    return (const + mat) if const else mat


def tab_gram_schmidt():
    st.info("ベクトルの組 $\\boldsymbol a_1,\\boldsymbol a_2,\\dots$ から直交するベクトルの組を作るグラム・シュミットの方法を, "
            "関数 $1,x,x^2,\\dots$ に使う。内積に**重み** $w(x)$ を入れると, 重みの選び方に応じて, ルジャンドル・チェビシェフ・エルミート・ラゲールの多項式が出てくる。")
    st.caption("内積と $L^2$ 空間については ⑥ その他「関数の内積と $L^2$ 空間(超関数との対比)」と ③ フーリエ解析 発展 7. も参照。")

    st.subheader("1. 重みつき内積")
    st.markdown("区間 $I$ と, $I$ で正の関数 $w(x)$(**重み**)を決めて, 内積を")
    st.latex(r"(f,g)_w=\int_I f(x)g(x)\,w(x)\,dx,\qquad\lVert f\rVert_w=\sqrt{(f,f)_w}")
    st.markdown("と定める。$(f,g)_w=0$ のとき $f$ と $g$ は**直交する**という。")
    st.markdown("- 区間が無限のときは, $e^{-x^2}$ や $e^{-x}$ のように速く $0$ になる重みを掛けると, 多項式どうしの内積が有限になる。\n"
                "- チェビシェフの重み $\\dfrac1{\\sqrt{1-x^2}}$ は, $x=\\cos\\theta$ と置換すると $dx/\\sqrt{1-x^2}=-d\\theta$ となり, "
                "$\\int_{-1}^1T_mT_n\\dfrac{dx}{\\sqrt{1-x^2}}=\\int_0^\\pi\\cos m\\theta\\cos n\\theta\\,d\\theta$(フーリエ級数の直交性)になるように選ばれている。")

    st.subheader("2. グラム・シュミットの直交化")
    st.markdown("$p_0=1$ から始めて, $x^n$ からそれまでの $p_j$ の成分(正射影)を引く:")
    st.latex(r"p_n(x)=x^n-\sum_{j=0}^{n-1}\frac{(x^n,p_j)_w}{(p_j,p_j)_w}\,p_j(x)")
    st.markdown("ベクトルで $\\boldsymbol b=\\boldsymbol a-\\dfrac{\\boldsymbol a\\cdot\\boldsymbol e}{\\boldsymbol e\\cdot\\boldsymbol e}\\boldsymbol e$ "
                "とするのと同じ計算である。内積はモーメント $m_k=\\int_Ix^kw(x)\\,dx$ だけで計算できる。")
    c = st.columns([1, 2])
    wname = c[0].selectbox("重み", list(WEIGHTS), format_func=lambda k: WEIGHTS[k][0], key="ogs_w")
    nmax = c[0].select_slider("何次まで", [2, 3, 4, 5], 4, key="ogs_n")
    title, itex, wtex, mom, const = WEIGHTS[wname]
    ps = gram_schmidt(nmax, wname)
    c[0].latex(rf"I={itex},\quad w(x)={wtex}")
    c[0].latex(r"m_k=" + (const + r"\times" if const else "") + r"\left(" + ",\\ ".join(p_tex([mom(k)]) for k in range(5)) + r",\dots\right)")
    for n, p in enumerate(ps):
        c[0].latex(rf"p_{n}(x)={p_tex(p)}")
    x = np.linspace(*PLOT_RANGE[wname], 1201)
    fig = go.Figure()
    cols = [C1, C2, C3, C4, GRAY, "#8e44ad"]
    for n, p in enumerate(ps):
        y = np.polynomial.polynomial.polyval(x, [float(v) for v in p])
        fig.add_trace(go.Scatter(x=x, y=y / (np.max(np.abs(y)) or 1), name=f"p_{n}", line=dict(color=cols[n % len(cols)])))
    wv = weight_fn(wname, x)
    fig.add_trace(go.Scatter(x=x, y=wv / np.max(wv[np.isfinite(wv)]) if wname != "chebyshev" else np.minimum(wv / 4, 1), name="重み w(x)(縮小)",
                             line=dict(color="#bbb", dash="dot"), fill="tozeroy", fillcolor="rgba(150,150,150,0.12)"))
    layout(fig, "直交化で得られた多項式(区間での最大値を 1 にそろえた)", xlab="x", height=400, yrange=[-1.15, 1.15])
    with c[1]:
        show(fig)
        st.markdown("**内積の表** $(p_i,p_j)_w$(対角線以外はすべて $0$):")
        st.latex(gram_tex(ps, wname))
    if wname in STD:
        sym, lead = STD[wname]
        std_lead = [POLY[wname](n)[-1] for n in range(nmax + 1)]
        st.markdown(f"よく使われる ${sym}$ は, この $p_n$ を定数倍したもの(最高次の係数が ${lead}$)である:"
                    + ", ".join(f"${sym[0]}_{n}={p_tex([std_lead[n]])}\\,p_{n}$" if std_lead[n] != 1 else f"${sym[0]}_{n}=p_{n}$" for n in range(nmax + 1)) + "。")
    elif wname == "unit01":
        st.caption("区間 $[0,1]$, 重み $1$ の多項式は, ルジャンドル多項式で $x$ を $2x-1$ に置きかえたもの(ずらしたルジャンドル多項式)の定数倍になる。")
    else:
        st.caption("区間 $[0,1]$, 重み $x$ の多項式は, ヤコビ多項式と呼ばれるもっと一般の直交多項式の仲間である。")

    st.subheader("3. 4つの直交多項式のまとめ")
    st.markdown(r"""
| | 区間・重み | ロドリゲスの公式など | 微分方程式 | ノルムの2乗 $(p_n,p_n)_w$ |
|---|---|---|---|---|
| ルジャンドル $P_n$ | $[-1,1]$, $1$ | $P_n=\dfrac{1}{2^nn!}\dfrac{d^n}{dx^n}(x^2-1)^n$ | $(1-x^2)y^{\prime\prime}-2xy^{\prime}+n(n+1)y=0$ | $\dfrac{2}{2n+1}$ |
| チェビシェフ $T_n$ | $[-1,1]$, $\dfrac{1}{\sqrt{1-x^2}}$ | $T_n(\cos\theta)=\cos n\theta$ | $(1-x^2)y^{\prime\prime}-xy^{\prime}+n^2y=0$ | $\pi\ (n=0)$, $\dfrac\pi2\ (n\ge1)$ |
| エルミート $H_n$ | $(-\infty,\infty)$, $e^{-x^2}$ | $H_n=(-1)^ne^{x^2}\dfrac{d^n}{dx^n}e^{-x^2}$ | $y^{\prime\prime}-2xy^{\prime}+2ny=0$ | $\sqrt\pi\,2^nn!$ |
| ラゲール $L_n$ | $[0,\infty)$, $e^{-x}$ | $L_n=\dfrac{e^x}{n!}\dfrac{d^n}{dx^n}(x^ne^{-x})$ | $xy^{\prime\prime}+(1-x)y^{\prime}+ny=0$ | $1$ |
""")
    st.markdown("**共通の性質**\n"
                "- **3項漸化式**:$p_{n+1}$ は $xp_n$ と $p_{n-1}$ だけから決まる($xp_n$ と $p_j$($j\\le n-2$)の内積が $0$ になるため)。\n"
                "- **零点**:$p_n$ は区間の内部に $n$ 個の異なる実数の零点をもつ(ガウス求積法の分点になる)。\n"
                "- **最良近似**:$f\\approx\\sum_{n=0}^{N}c_np_n$, $c_n=\\dfrac{(f,p_n)_w}{(p_n,p_n)_w}$ とすると, "
                "$N$ 次以下の多項式の中で $\\lVert f-q\\rVert_w$ を最小にする($N$ 次元の平面への正射影)。")

    with st.expander("練習問題"):
        st.markdown("**問1** 区間 $[-1,1]$, 重み $1$ で, $1$, $x$, $x^2$ を直交化せよ。")
        st.markdown("**問2** 区間 $[0,\\infty)$, 重み $e^{-x}$ で, $1$, $x$ を直交化せよ($\\int_0^\\infty x^ke^{-x}dx=k!$)。")
        st.markdown("**問3** $x^2$ をルジャンドル多項式 $P_0=1$, $P_2=\\dfrac{3x^2-1}{2}$ で表せ。")
        if st.toggle("解答・解説を表示", key="pr_ogs"):
            st.markdown("**答1** $p_0=1$, $p_1=x$($(x,1)=0$), $p_2=x^2-\\dfrac{(x^2,1)}{(1,1)}=x^2-\\dfrac{2/3}{2}=x^2-\\dfrac13$($(x^2,x)=0$)。")
            st.markdown("**答2** $p_0=1$, $p_1=x-\\dfrac{(x,1)}{(1,1)}=x-\\dfrac{1!}{0!}=x-1$($=-L_1$)。")
            st.markdown("**答3** $x^2=\\dfrac{2P_2+1}{3}=\\dfrac13P_0+\\dfrac23P_2$。")

    page_refs([("szego1939", "直交多項式の一般論(3項漸化式・零点)。"), ("nist2010", "第18章:直交多項式の公式(重み・ノルム・漸化式)。")])
    refs([("DLMF §18.3 Definitions(NIST Digital Library of Mathematical Functions)", "https://dlmf.nist.gov/18.3",
           "古典直交多項式の区間・重み・ノルムの表(英語)")])
