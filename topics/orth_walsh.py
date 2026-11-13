"""⑦ 特殊関数 第2章 直交関数系:7. ウォルシュ関数系

値が ±1 だけの階段関数からなる直交関数系。交番数(符号の変わる回数)の順に並べる。
N = 2^m 個のウォルシュ関数の和は, 区間を N 等分した階段関数全体(ハール関数の和と同じ範囲)を表す。
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import C1, C2, C3, C4, GRAY, layout, show
from topics.fourier_apps import refs
from topics.fourier_kernels import page_refs
from topics.orth_core import walsh_matrix

GRID = 1024   # 展開の計算に使う [0,1) の分割数
FUNCS = {
    "f(x) = x": lambda x: x,
    "f(x) = sin 2πx": lambda x: np.sin(2 * np.pi * x),
    "f(x) = e^{−3x}": lambda x: np.exp(-3 * x),
}


@st.cache_data(show_spinner=False)
def walsh_grid():
    return walsh_matrix(GRID).astype(float)


def fwht(v):
    """高速ウォルシュ・アダマール変換(アダマール行列の順, 足し算と引き算だけ)"""
    v = np.array(v, dtype=float)
    h = 1
    while h < len(v):
        for i in range(0, len(v), 2 * h):
            a, b = v[i:i + h].copy(), v[i + h:i + 2 * h].copy()
            v[i:i + h], v[i + h:i + 2 * h] = a + b, a - b
        h *= 2
    return v


def cdma(bits, noise, seed=0):
    """各利用者のビット ±1 にウォルシュ符号(長さ 8)を掛けて足し, 内積で取り出す"""
    W = walsh_matrix(8)[[1, 2, 3, 4]]
    tx = np.array(bits) @ W
    rx = tx + noise * np.random.default_rng(seed).standard_normal(8)
    return W, tx, rx, (W @ rx) / 8


def tab_walsh():
    st.info("ウォルシュ関数は, 値が $+1$ と $-1$ だけの階段関数からなる直交関数系である。"
            "三角関数系の「$\\cos n\\theta$ や $\\sin n\\theta$ の振動数 $n$」の代わりに, 符号の変わる回数(**交番数**)で並べる。"
            "掛け算が符号の反転だけで済むので, ディジタル回路やコンピュータでの計算に向いている。")

    st.subheader("1. 定義:ラーデマッハー関数の積")
    st.markdown("区間 $[0,1)$ で, $r_k(x)=\\mathrm{sgn}\\,\\sin(2^k\\pi x)$($k=1,2,\\dots$)を**ラーデマッハー関数**という($[0,1)$ を $2^k$ 等分して交互に $\\pm1$)。"
                "これらの積をとると, $[0,1)$ を $N=2^m$ 等分した上で $\\pm1$ をとる関数が $N$ 個できる。交番数の順に並べたものが**ウォルシュ関数** "
                "$w_0,w_1,\\dots,w_{N-1}$ で, $w_n$ はちょうど $n$ 回符号が変わる。")
    c = st.columns([1, 2])
    N = c[0].select_slider("区間の分割数 N", [4, 8, 16], 8, key="owl_N")
    W = walsh_matrix(N)
    c[0].latex(r"W_{" + str(N) + r"}=\begin{pmatrix}" + r"\\".join("&".join("+" if v > 0 else "-" for v in row) for row in W) + r"\end{pmatrix}")
    c[0].caption("各行が1つのウォルシュ関数の値(左の区間から順に)。")
    fig = go.Figure()
    xs = np.repeat(np.arange(N + 1) / N, 2)[1:-1]
    for n in range(N):
        ys = np.repeat(W[n], 2)
        fig.add_trace(go.Scatter(x=xs, y=-3 * n + 1.1 * ys, mode="lines", line=dict(color=[C1, C2, C3, C4][n % 4], width=2),
                                 name=f"w_{n}", showlegend=False))
        fig.add_annotation(x=-0.04, y=-3 * n, text=f"w<sub>{n}</sub>", showarrow=False)
    layout(fig, "ウォルシュ関数(交番数の順)", xlab="x", height=min(120 + 34 * N, 620), legend=False)
    fig.update_yaxes(showticklabels=False)
    with c[1]:
        show(fig)

    st.subheader("2. 直交性とアダマール行列")
    st.markdown("$[0,1)$ での内積 $(f,g)=\\int_0^1fg\\,dx$ について")
    st.latex(r"\int_0^1w_m(x)w_n(x)\,dx=\begin{cases}0&(m\neq n)\\1&(m=n)\end{cases}")
    st.markdown("これは, 上の行列 $W_N$ の異なる行どうしの内積が $0$ であること $W_NW_N^{\\mathsf T}=NI$ と同じである。"
                "行の並べ方を変えた $H_{2N}=\\begin{pmatrix}H_N&H_N\\\\H_N&-H_N\\end{pmatrix}$ を**アダマール行列**(シルベスター型)といい, 帰納的に $H_NH_N^{\\mathsf T}=NI$ が示せる。")
    st.caption(f"確認:$N={N}$ で $W_{{{N}}}W_{{{N}}}^{{\\mathsf T}}={N}I$ は {'成り立つ' if np.array_equal(W @ W.T, N * np.eye(N, dtype=int)) else '成り立たない'}。")

    st.subheader("3. ウォルシュ展開")
    st.latex(r"f(x)\approx\sum_{n=0}^{N-1}c_nw_n(x),\qquad c_n=\int_0^1f(x)w_n(x)\,dx")
    c = st.columns([1, 2])
    fname = c[0].selectbox("関数", list(FUNCS), key="owl_f")
    K = c[0].select_slider("使う項の数", [1, 2, 3, 4, 6, 8, 16, 32, 64], 8, key="owl_K")
    x = (np.arange(GRID) + 0.5) / GRID
    fv = FUNCS[fname](x)
    Wg = walsh_grid()
    coef = Wg @ fv / GRID
    S = coef[:K] @ Wg[:K]
    pow2 = K & (K - 1) == 0
    c[0].latex(", ".join(f"c_{n}\\approx{v:.3f}" for n, v in enumerate(coef[:4])) + r",\ \dots")
    c[0].metric("2乗平均の誤差", f"{float(np.sqrt(np.mean((fv - S) ** 2))):.4f}")
    if pow2:
        c[0].success(f"項の数 ${K}$ が $2$ の累乗のとき, 和は区間を ${K}$ 等分した各区間での $f$ の平均値になる(ハール関数で展開したのと同じ)。")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=fv, name="f(x)", line=dict(color=GRAY, width=4)))
    fig.add_trace(go.Scatter(x=x, y=S, name=f"{K} 項の和", line=dict(color=C1, shape="hv")))
    layout(fig, None, xlab="x", height=340)
    with c[1]:
        show(fig)
    st.markdown("**ハール関数との関係**:$[0,1)$ を $2^m$ 等分した階段関数全体は $2^m$ 次元で, 最初の $2^m$ 個のウォルシュ関数も, "
                "同じ数のハール関数(③ フーリエ解析の発展「ウェーブレット」)も, その基底になる。"
                "ハール関数は場所ごとの細かい変化を, ウォルシュ関数は区間全体での「振動の多さ」をとらえる。")

    st.subheader("4. 高速ウォルシュ・アダマール変換")
    st.markdown("$N$ 個のデータ $v$ に $H_N$ を掛ける計算は, $H_{2N}$ の形を使って半分ずつに分けると, **足し算と引き算だけ**で $N\\log_2N$ 回で済む"
                "(高速フーリエ変換と同じ考え方で, 掛け算がいらない)。")
    st.code("def fwht(v):            # len(v) = 2^m\n    h = 1\n    while h < len(v):\n        for i in range(0, len(v), 2 * h):\n"
            "            a, b = v[i:i+h], v[i+h:i+2*h]\n            v[i:i+h], v[i+h:i+2*h] = a + b, a - b\n        h *= 2\n    return v", language="python")
    v = [4, 2, 2, 0]
    st.markdown(f"例:$v=({', '.join(map(str, v))})$ に対して, 1段目 $(4+2,\\ 4-2,\\ 2+0,\\ 2-0)=(6,2,2,2)$, "
                f"2段目 $(6+2,\\ 2+2,\\ 6-2,\\ 2-2)=({', '.join(str(int(t)) for t in fwht(v))})=H_4v$。")

    st.subheader("5. 応用")
    st.markdown("**電気:符号分割多元接続(CDMA)**  何人もの利用者が同じ周波数で同時に送信するとき, 利用者ごとに異なるウォルシュ関数(符号)を割り当てる。"
                "利用者 $i$ のビット $b_i=\\pm1$ に符号 $w_i$ を掛けて足した信号 $s=\\sum b_iw_i$ を送ると, 受信側は**内積** $\\frac18(s,w_i)$ で $b_i$ を取り出せる"
                "(直交性により他の利用者の分は $0$ になる)。")
    c = st.columns([1, 2])
    bits = [1 if c[0].radio(f"利用者 {i + 1} のビット", ["+1", "−1"], horizontal=True, key=f"owl_b{i}") == "+1" else -1 for i in range(4)]
    noise = c[0].select_slider("雑音の大きさ", [0.0, 0.3, 0.6, 1.0, 1.5], 0.3, key="owl_noise")
    Wc, tx, rx, dec = cdma(bits, noise)
    c[0].markdown("\n".join(f"- 利用者 {i + 1}:送信 ${bits[i]:+d}$ → 復元 ${dec[i]:+.2f}$ → 判定 ${1 if dec[i] > 0 else -1:+d}$" for i in range(4)))
    t = np.arange(8)
    fig = go.Figure()
    for i in range(4):
        fig.add_trace(go.Bar(x=t, y=bits[i] * Wc[i], name=f"利用者 {i + 1}:b w", marker_color=[C1, C2, C3, C4][i], opacity=0.55))
    fig.add_trace(go.Scatter(x=t, y=rx, mode="lines+markers", name="受信信号(和+雑音)", line=dict(color="black", width=2.5, shape="hv")))
    fig.update_layout(barmode="relative")
    layout(fig, "各利用者の信号と, 重ね合わせた受信信号", xlab="チップの番号", height=380)
    with c[1]:
        show(fig)
    st.markdown(r"""
| 学科 | 使われ方 |
|---|---|
| 機械 | オン・オフで動く機器の信号やディジタル制御の信号の解析 |
| 電気電子 | 携帯電話などの符号分割多元接続(CDMA), 誤り訂正符号, ディジタル信号処理 |
| 化学 | アダマール変換分光法(多くの波長を同時に測って雑音を減らす測定法) |
| 都市 | 画像データ(航空写真など)の圧縮・処理 |
""")

    st.subheader("6. 歴史")
    st.markdown(r"""
| 年 | できごと |
|---|---|
| 1920年代 | ラーデマッハー(H. Rademacher)が $\pm1$ の値をとる関数列(ラーデマッハー関数)を研究 |
| 1923 | ウォルシュ(J. L. Walsh)が論文「A closed set of normal orthogonal functions」で, ウォルシュ関数系を導入 |
| 1960年代以降 | コンピュータによる信号処理が広がり, 高速ウォルシュ・アダマール変換とともに通信・画像処理に使われる |
""")

    with st.expander("練習問題"):
        st.markdown("**問1** $N=4$ のウォルシュ関数 $w_0,w_1,w_2,w_3$ の値を書き, $w_1$ と $w_3$ が直交することを確かめよ。")
        st.markdown("**問2** $[0,1)$ を4等分した各区間で $f=3,1,1,-1$ となる階段関数のウォルシュ係数 $c_0,\\dots,c_3$ を求めよ。")
        st.markdown("**問3** 符号 $w_1=(1,1,-1,-1)$, $w_2=(1,-1,-1,1)$ で2人が $b_1=-1$, $b_2=1$ を送ったときの信号 $s$ を求め, 内積で $b_1$, $b_2$ を取り出せ。")
        if st.toggle("解答・解説を表示", key="pr_owl"):
            st.markdown("**答1** $w_0=(1,1,1,1)$, $w_1=(1,1,-1,-1)$, $w_2=(1,-1,-1,1)$, $w_3=(1,-1,1,-1)$。"
                        "$w_1w_3=(1,-1,-1,1)$ の平均は $0$。")
            st.markdown("**答2** $c_n=\\frac14\\sum f_kw_n(k)$ より $c_0=1$, $c_1=\\frac14(3+1-1+1)=1$, $c_2=\\frac14(3-1-1-1)=0$, $c_3=\\frac14(3-1+1+1)=1$。")
            st.markdown("**答3** $s=-w_1+w_2=(0,-2,0,2)$。$\\frac14(s,w_1)=\\frac14(0-2+0-2)=-1$, $\\frac14(s,w_2)=\\frac14(0+2+0+2)=1$。")

    page_refs([("walsh1923", "ウォルシュ関数系を導入した論文。")])
    refs([("MacTutor:Joseph Leonard Walsh", "https://mathshistory.st-andrews.ac.uk/Biographies/Walsh_Joseph/", "ウォルシュの生涯と, ウォルシュ関数がのちに信号処理で使われたこと(英語)"),
          ("Encyclopedia of Mathematics:Walsh system", "https://encyclopediaofmath.org/wiki/Walsh_system", "ウォルシュ関数系の定義と性質(英語)")])
