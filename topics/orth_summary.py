"""⑦ 特殊関数 第2章 直交関数系:8. 直交関数系の使い分け(まとめ)

三角関数系・ルジャンドル・チェビシェフ・エルミート・ラゲール・ウォルシュ(ハール)を
「区間」「重み」「得意な関数」「生まれた方程式」で比べ, 同じ係数の数で近似したときの誤差を実験で比べる。
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import C1, C2, C3, C4, GRAY, layout, show
from topics.orth_core import p_eval, walsh_matrix

NS = [2, 4, 8, 16, 32, 64]
GRID = 4096
TEST = {
    "なめらか・周期的でない:e^x": (np.exp, r"e^x"),
    "なめらか・周期的:sin πx": (lambda x: np.sin(np.pi * x), r"\sin\pi x"),
    "折れ目あり:|x|": (np.abs, r"|x|"),
    "とびあり(x = 0):符号関数": (lambda x: np.where(x > 0, 1.0, np.where(x < 0, -1.0, 0.0)), r"\mathrm{sgn}\,x"),
    "とびあり(x = 0.3)": (lambda x: (x > 0.3).astype(float), r"H(x-0.3)"),
    "端の近くで急に変わる:1/(1 + 25x²)": (lambda x: 1 / (1 + 25 * x * x), r"\frac{1}{1+25x^2}"),
}
SYSTEMS = ["三角関数系(フーリエ級数)", "ルジャンドル多項式", "チェビシェフ多項式", "ウォルシュ関数(=ハール)"]
SCOL = {SYSTEMS[0]: C2, SYSTEMS[1]: C1, SYSTEMS[2]: C3, SYSTEMS[3]: C4}


def fourier_basis(N, x):
    """[−1, 1] の三角関数系(周期 2)を N 個:1, cos πx, sin πx, cos 2πx, …"""
    out = [np.ones_like(x) / np.sqrt(2)]
    k = 1
    while len(out) < N:
        out.append(np.cos(k * np.pi * x))
        if len(out) < N:
            out.append(np.sin(k * np.pi * x))
        k += 1
    return np.array(out)


@st.cache_data(show_spinner=False)
def approximations(fname, N):
    """同じ係数の数 N で近似した結果(格子 x 上の値)"""
    f = TEST[fname][0]
    x = -1 + (np.arange(GRID) + 0.5) * 2 / GRID       # 中点の格子
    fx = f(x)
    dx = 2 / GRID
    res = {}
    B = fourier_basis(N, x)                            # 正規直交(∫_{−1}^{1} φ² dx = 1)
    res[SYSTEMS[0]] = (B @ fx * dx) @ B
    g, w = np.polynomial.legendre.leggauss(200)
    xs, ws = np.concatenate([(g - 1) / 2, (g + 1) / 2]), np.concatenate([w / 2, w / 2])
    if fname.startswith("とびあり(x = 0.3)"):
        xs = np.concatenate([(g + 1) * 0.65 - 1, (g + 1) * 0.35 + 0.3])
        ws = np.concatenate([w * 0.65, w * 0.35])
    co = [(2 * n + 1) / 2 * np.sum(ws * f(xs) * p_eval("legendre", n, xs)) for n in range(N)]
    res[SYSTEMS[1]] = sum(c * p_eval("legendre", n, x) for n, c in enumerate(co))
    M = 20000
    th = (np.arange(M) + 0.5) * np.pi / M
    fc = f(np.cos(th))
    co = [(1 if n == 0 else 2) / M * np.sum(fc * np.cos(n * th)) for n in range(N)]
    res[SYSTEMS[2]] = sum(c * p_eval("chebyshev", n, x) for n, c in enumerate(co))
    W = walsh_matrix(N).astype(float)                  # [−1,1] を N 等分した上の ±1
    cells = np.repeat(np.arange(N), GRID // N)
    co = (W[:, cells] @ fx) / GRID
    res[SYSTEMS[3]] = co @ W[:, cells]
    return x, fx, res


def errors(fname, N):
    x, fx, res = approximations(fname, N)
    return {s: (float(np.sqrt(np.mean((fx - v) ** 2) * 2)), float(np.max(np.abs(fx - v)))) for s, v in res.items()}


def sci(v):
    """1.2e-05 → 1.2\\times10^{-5}(TeX)"""
    if v < 1e-15:
        return "0"
    m, e = f"{v:.1e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def recommend(domain, kind, goal):
    if domain == "実数全体 (−∞, ∞)":
        return "エルミート関数", "遠くでガウス関数のように速く $0$ になる関数に向く(調和振動子・ガウス分布・光のビーム)。フーリエ変換で形が変わらない。"
    if domain == "半直線 [0, ∞)":
        return "ラゲール関数", "時間 $t\\ge0$ で指数関数的に減衰する関数(インパルス応答など)に向く。ラプラス変換と相性がよい。"
    if domain == "周期的な現象(1周期分)":
        return "三角関数系(フーリエ級数)", "周期的でなめらかなら係数が速く小さくなる。周波数(振動数)の意味がはっきりしている。"
    if domain == "球面・角度 θ(0 ≤ θ ≤ π)":
        return "ルジャンドル多項式($x=\\cos\\theta$)", "球のまわりのポテンシャル・重力場・原子軌道。一般には球面調和関数を使う。"
    if kind == "とびや段差が多い・ディジタル(±1)の信号":
        return ("ハール関数" if goal == "場所ごとの細かい変化を知りたい" else "ウォルシュ関数"), \
            "階段関数をそのまま表せる。掛け算がいらず計算が速い。場所ごとの変化はハール(ウェーブレット), 全体の「振動の多さ」はウォルシュ。"
    if goal == "区間全体で最大の誤差を小さくしたい(補間・数値計算)":
        return "チェビシェフ多項式", "誤差が区間全体で均等に振動する(最良一様近似に近い)。$x=\\cos\\theta$ でフーリエ余弦級数になるので速く計算できる。"
    return "ルジャンドル多項式", "重み $1$ の2乗平均の誤差 $\\int_{-1}^{1}(f-q)^2dx$ が最小になる。有限要素法・ガウス求積とつながる。"


def tab_orth_summary():
    st.info("この章の直交関数系は, どれも「内積 → 係数 → 部分和が最良近似」という同じしくみで関数を展開する。"
            "違いは**区間**と**重み**, そして**どんな関数を少ない項で表せるか**である。このページでは使い分けをまとめ, 同じ関数を同じ数の係数で近似して比べる。")

    st.subheader("1. 早見表")
    st.markdown(r"""
| 関数系 | 区間 | 重み $w(x)$ | 得意な関数・現象 | 主な応用 |
|---|---|---|---|---|
| 三角関数系 $\cos nx,\ \sin nx$ | $[-\pi,\pi]$(周期) | $1$ | 周期的な関数, 振動・波 | 信号の周波数解析, 熱・波動方程式 |
| ルジャンドル $P_n$ | $[-1,1]$ | $1$ | なめらかな関数(2乗平均の誤差) | 球のまわりのポテンシャル, ガウス求積 |
| チェビシェフ $T_n$ | $[-1,1]$ | $\dfrac1{\sqrt{1-x^2}}$ | なめらかな関数(最大の誤差) | 補間・数値計算, フィルタ設計 |
| エルミート $H_ne^{-x^2/2}$ | $(-\infty,\infty)$ | $e^{-x^2}$ | ガウス関数のように減衰する関数 | 調和振動子, 確率(正規分布) |
| ラゲール $L_ne^{-x/2}$ | $[0,\infty)$ | $e^{-x}$ | 指数関数的に減衰する関数 | インパルス応答, 水素原子 |
| ウォルシュ・ハール | $[0,1)$ | $1$ | 階段関数, とび, ディジタル信号 | 通信(CDMA), 画像の圧縮・解析 |
""")

    st.subheader("2. 選び方のめやす")
    c = st.columns([1, 1])
    domain = c[0].radio("関数が定義されている範囲", ["有限の区間 [a, b]", "周期的な現象(1周期分)", "球面・角度 θ(0 ≤ θ ≤ π)", "半直線 [0, ∞)", "実数全体 (−∞, ∞)"], key="osm_dom")
    kind, goal = None, None
    if domain == "有限の区間 [a, b]":
        kind = c[0].radio("関数の性質", ["なめらか(何回も微分できる)", "とびや段差が多い・ディジタル(±1)の信号"], key="osm_kind")
        if kind.startswith("なめらか"):
            goal = c[0].radio("何を小さくしたいか", ["2乗平均の誤差(エネルギー)", "区間全体で最大の誤差を小さくしたい(補間・数値計算)"], key="osm_goal")
        else:
            goal = c[0].radio("知りたいこと", ["全体の「振動の多さ」(交番数)", "場所ごとの細かい変化を知りたい"], key="osm_goal2")
        c[0].caption("区間 $[a,b]$ は $x=\\dfrac{a+b}{2}+\\dfrac{b-a}{2}s$ で $s\\in[-1,1]$ に直してから使う。")
    name, why = recommend(domain, kind, goal)
    with c[1]:
        st.success(f"**おすすめ:{name}**")
        st.markdown(why)
        st.caption("あくまでめやす。実際には, 現象を表す微分方程式(下の 4.)から自然に決まることも多い。")

    st.subheader("3. 実験:同じ数の係数で近似して比べる")
    st.markdown("区間 $[-1,1]$ の関数を, 4つの関数系それぞれで**係数 $N$ 個**を使って近似する"
                "(三角関数系は $1,\\cos\\pi x,\\sin\\pi x,\\dots$, 多項式は $N-1$ 次まで, ウォルシュ関数は区間を $N$ 等分)。")
    c = st.columns([1, 2])
    fname = c[0].selectbox("関数", list(TEST), key="osm_f")
    N = c[0].select_slider("係数の数 N", NS, 8, key="osm_N")
    x, fx, res = approximations(fname, N)
    c[0].latex(f"f(x)={TEST[fname][1]}")
    err = errors(fname, N)
    c[0].markdown("| 関数系 | 2乗平均の誤差 | 最大の誤差 |\n|---|---|---|\n" +
                  "\n".join(f"| {s} | ${sci(e2)}$ | ${sci(em)}$ |" for s, (e2, em) in err.items()))
    fig = go.Figure(go.Scatter(x=x, y=fx, name="f(x)", line=dict(color=GRAY, width=5)))
    for s, v in res.items():
        fig.add_trace(go.Scatter(x=x, y=v, name=s, line=dict(color=SCOL[s], width=1.8, shape="hv" if "ウォルシュ" in s else "linear")))
    lo, hi = float(np.min(fx)), float(np.max(fx))
    pad = 0.35 * (hi - lo) + 0.1
    layout(fig, f"N = {N} 個の係数での近似", xlab="x", height=400, yrange=[lo - pad, hi + pad])
    with c[1]:
        show(fig)
    fig = go.Figure()
    for s in SYSTEMS:
        fig.add_trace(go.Scatter(x=NS, y=[errors(fname, n)[s][0] for n in NS], mode="lines+markers", name=s, line=dict(color=SCOL[s])))
    layout(fig, "係数の数 N と2乗平均の誤差(両対数)", xlab="係数の数 N", ylab="2乗平均の誤差", height=360)
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    show(fig)
    st.markdown("**観察のポイント**\n"
                "- $e^x$:両端の値が違うので, 三角関数系では周期的につなげたときに**とび**ができ, 端でギブス現象が起きて誤差がなかなか減らない。"
                "多項式(ルジャンドル・チェビシェフ)では誤差が**指数関数的に**減る。"
                "$N=8$ の表を見ると, 2乗平均の誤差はルジャンドルが, 最大の誤差はチェビシェフが小さい(それぞれが最小にしようとしている量)。\n"
                "- $\\sin\\pi x$:周期的でなめらかなので, 三角関数系なら係数 $3$ 個($1,\\cos\\pi x,\\sin\\pi x$)で厳密に表せる。\n"
                "- $|x|$, $\\dfrac{1}{1+25x^2}$:両端の値が等しいので, 周期的につなげても連続になる。このような関数では, $N$ が大きすぎなければ三角関数系がいちばん正確になる。"
                "$\\dfrac{1}{1+25x^2}$ は解析的なので, $N$ を大きくすると多項式が追い抜く($N=64$)。\n"
                "- 符号関数:$x=0$ がちょうど区間の分点なので, ウォルシュ関数では $N=2$ で厳密に表せる。"
                "とびの位置が分点からずれる($x=0.3$)と, どの関数系でも誤差はゆっくりしか減らない。")
    st.caption("どの関数系でも, 誤差は $N$ を増やすほど小さくなる(部分和は $N$ 次元の部分空間への正射影なので, 空間が広がれば誤差は増えない)。")

    st.subheader("4. 共通のしくみ:どの微分方程式から生まれるか")
    st.markdown("三角関数系と4つの直交多項式は, どれも**シュトゥルム・リウヴィル型**の固有値問題")
    st.latex(r"\bigl(p(x)\,y^{\prime}\bigr)^{\prime}=-\lambda\,w(x)\,y")
    st.markdown("の固有関数である。$p$ と境界の条件で区間が, $w$ で内積の重みが決まり, 異なる固有値の固有関数は自動的に直交する。")
    st.markdown(r"""
| 関数系 | $p(x)$ | $w(x)$ | 固有値 $\lambda$ | 方程式 |
|---|---|---|---|---|
| 三角関数系 | $1$ | $1$ | $n^2$ | $y^{\prime\prime}=-n^2y$ |
| ルジャンドル | $1-x^2$ | $1$ | $n(n+1)$ | $\bigl((1-x^2)y^{\prime}\bigr)^{\prime}=-n(n+1)y$ |
| チェビシェフ | $\sqrt{1-x^2}$ | $\dfrac1{\sqrt{1-x^2}}$ | $n^2$ | $\bigl(\sqrt{1-x^2}\,y^{\prime}\bigr)^{\prime}=-\dfrac{n^2}{\sqrt{1-x^2}}y$ |
| エルミート | $e^{-x^2}$ | $e^{-x^2}$ | $2n$ | $\bigl(e^{-x^2}y^{\prime}\bigr)^{\prime}=-2ne^{-x^2}y$ |
| ラゲール | $xe^{-x}$ | $e^{-x}$ | $n$ | $\bigl(xe^{-x}y^{\prime}\bigr)^{\prime}=-ne^{-x}y$ |
""")
    st.markdown("例えばエルミートでは, 左辺を展開すると $e^{-x^2}(y^{\\prime\\prime}-2xy^{\\prime})$ となり, $e^{-x^2}$ で割ると $y^{\\prime\\prime}-2xy^{\\prime}+2ny=0$(エルミートの方程式)に戻る。"
                "**直交する理由**:$y_m,y_n$ の方程式に $y_n,y_m$ を掛けて引き, 区間で積分すると")
    st.latex(r"(\lambda_m-\lambda_n)\int y_my_n\,w\,dx=\Bigl[p\,(y_my_n^{\prime}-y_ny_m^{\prime})\Bigr]_{\text{両端}}=0")
    st.markdown("($p$ が両端で $0$ になるか, 周期的であるため)。$\\lambda_m\\neq\\lambda_n$ なら $\\int y_my_nw\\,dx=0$。"
                "ウォルシュ関数・ハール関数は微分方程式からではなく, 「区間を2つに分けて符号を変える」という組合せのしくみから作られる。")

    st.subheader("5. 係数の減り方となめらかさ")
    st.markdown(r"""
| 関数の性質(展開に合った区間で) | 係数 $c_n$ の減り方のめやす | 例 |
|---|---|---|
| とびがある | $\dfrac1n$ 程度(ゆっくり) | 符号関数(とびが分点でないとき) |
| 連続だが折れ目がある | $\dfrac1{n^2}$ 程度 | $\lvert x\rvert$ |
| 何回でも微分できる | どんな $k$ についても $\dfrac1{n^k}$ より速い | |
| 解析的(テイラー展開できる) | $r^n$($0<r<1$, 指数関数的) | 多項式で $e^x$($r$ が小さく, 非常に速い), $\dfrac1{1+25x^2}$(複素数の特異点 $x=\pm\frac i5$ が区間に近いので $r$ が $1$ に近く, ゆっくり) |
""")
    st.caption("三角関数系では「周期的につなげたときに」なめらかかどうかが効く。$e^x$ は区間の中ではなめらかでも, 周期的につなげるととびができる。")

    with st.expander("練習問題"):
        st.markdown("**問1** 次の(1)〜(4)を展開するのに向いている関数系を答えよ。\n"
                    "(1) 地球のまわりの重力ポテンシャルの角度による変化  (2) コンデンサの放電電圧 $v(t)=V_0e^{-t/RC}$($t\\ge0$)に近い波形  "
                    "(3) オン・オフを繰り返す制御信号  (4) 区間 $[0,10]$ で測った気温の変化を, 最大の誤差が小さくなるように多項式で近似したい")
        st.markdown("**問2** $\\bigl(xe^{-x}y^{\\prime}\\bigr)^{\\prime}=-ne^{-x}y$ を展開し, ラゲールの方程式 $xy^{\\prime\\prime}+(1-x)y^{\\prime}+ny=0$ になることを確かめよ。")
        st.markdown("**問3** 区間 $[-1,1]$ で $f(x)=x$ を三角関数系で展開すると, 係数がゆっくりしか減らない。理由を述べよ。")
        if st.toggle("解答・解説を表示", key="pr_osm"):
            st.markdown("**答1** (1) ルジャンドル多項式($P_n(\\cos\\theta)$, 一般には球面調和関数) (2) ラゲール関数 (3) ウォルシュ関数・ハール関数 "
                        "(4) チェビシェフ多項式($t=5+5s$ で $s\\in[-1,1]$ に直す)。")
            st.markdown("**答2** $(xe^{-x}y^{\\prime})^{\\prime}=e^{-x}y^{\\prime}-xe^{-x}y^{\\prime}+xe^{-x}y^{\\prime\\prime}=e^{-x}\\bigl(xy^{\\prime\\prime}+(1-x)y^{\\prime}\\bigr)$。"
                        "$-ne^{-x}y$ と等しいとおき $e^{-x}$ で割る。")
            st.markdown("**答3** 周期 $2$ で周期的につなげると, $x=\\pm1$ で値が $1$ から $-1$ にとぶ(のこぎり波)。とびがあると係数は $\\dfrac1n$ 程度でしか減らない"
                        "($x=\\sum\\dfrac{2(-1)^{n+1}}{n\\pi}\\sin n\\pi x$)。")
