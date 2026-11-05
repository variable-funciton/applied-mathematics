"""③ フーリエ解析 発展:ハール・ウェーブレットと多重解像度 / 応用:ウェーブレットの応用(4学科)

ハール変換は「平均と差の半分」の形で書く(手計算・演習と同じ):
    a = (x_{2k} + x_{2k+1}) / 2,   d = (x_{2k} − x_{2k+1}) / 2,   x_{2k} = a + d,  x_{2k+1} = a − d
関数で書くと, [0, 1) 上で  P_{j+1} f = P_j f + Σ_k d_{j,k} ψ(2^j x − k)  (ψ は正規化しないハール関数)。
係数のしきい値処理・最良 M 項近似には, 正規直交な形(√2 倍ずつ)の係数を使う。
ドブシーの D4(db2)は, 同じ枠組みのフィルタを numpy で実装している(周期境界)。
"""
import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import C1, C2, C3, C4, GRAY, layout, show
from topics.fourier_kernels import page_refs, ref_line
from topics.vec_apps import examples

PI = np.pi
J = 10
NS = 2 ** J                                   # [0, 1) の分点数
XS = (np.arange(NS) + 0.5) / NS              # 各小区間の中点

S3 = math.sqrt(3)
D4_H = np.array([1 + S3, 3 + S3, 3 - S3, 1 - S3]) / (4 * math.sqrt(2))
HAAR_H = np.array([1, 1]) / math.sqrt(2)


# ------------------------------------------------------------
# 平均と差の半分によるハール変換
# ------------------------------------------------------------
def haar_step(x):
    x = np.asarray(x, dtype=float)
    return (x[0::2] + x[1::2]) / 2, (x[0::2] - x[1::2]) / 2


def haar_full(x):
    """[a_J(1個), d(粗い順に), ...] を返す:x の長さは 2 のべき"""
    a = np.asarray(x, dtype=float)
    ds = []
    while len(a) > 1:
        a, d = haar_step(a)
        ds.insert(0, d)                        # 粗いレベルを先に
    return a, ds


def haar_inverse(a, ds):
    a = np.asarray(a, dtype=float)
    for d in ds:
        out = np.empty(2 * len(a))
        out[0::2], out[1::2] = a + d, a - d
        a = out
    return a


def proj(x, j):
    """P_j f:幅 2^{−j} の区間ごとの平均(V_j への射影)"""
    return np.repeat(np.asarray(x, float).reshape(2 ** j, -1).mean(axis=1), len(x) // 2 ** j)


# ------------------------------------------------------------
# 正規直交ウェーブレット変換(周期境界)
# ------------------------------------------------------------
def filters(h):
    L = len(h)
    g = np.array([(-1) ** n * h[L - 1 - n] for n in range(L)])
    return h, g


def dwt(x, h):
    h, g = filters(h)
    N = len(x)
    idx = (2 * np.arange(N // 2)[:, None] + np.arange(len(h))[None, :]) % N
    return x[idx] @ h, x[idx] @ g


def idwt(a, d, h):
    h, g = filters(h)
    N = 2 * len(a)
    x = np.zeros(N)
    for n in range(len(h)):
        pos = (2 * np.arange(len(a)) + n) % N
        np.add.at(x, pos, h[n] * a + g[n] * d)
    return x


def wavedec(x, h, level):
    """[a_L, d_L, …, d_1](d_1 が最も細かい)"""
    a = np.asarray(x, float)
    ds = []
    for _ in range(level):
        a, d = dwt(a, h)
        ds.insert(0, d)
    return [a] + ds


def waverec(c, h):
    a = c[0]
    for d in c[1:]:
        a = idwt(a, d, h)
    return a


def approx_level(x, h, j):
    """係数を粗い方から 2^j 個の解像度まで残した近似(ハールなら P_j f)"""
    L = J - j
    c = wavedec(x, h, L)
    return waverec([c[0]] + [np.zeros_like(d) for d in c[1:]], h)


def best_m(x, h, M):
    """正規直交係数の大きい方から M 個を残した近似"""
    c = wavedec(x, h, J)
    flat = np.concatenate(c)
    if M < len(flat):
        thr = np.sort(np.abs(flat))[-M] if M > 0 else np.inf
        keep = np.abs(flat) >= thr
        flat = np.where(keep, flat, 0.0)
    out, i = [], 0
    for v in c:
        out.append(flat[i:i + len(v)])
        i += len(v)
    return waverec(out, h)


def fourier_best_m(x, M):
    """フーリエ係数(実数の係数で数えて)大きい方から M 個を残した近似"""
    X = np.fft.rfft(x)
    amp = np.abs(X) * np.where(np.arange(len(X)) == 0, 1, math.sqrt(2))
    order = np.argsort(-amp)
    keep = np.zeros(len(X), bool)
    cnt = 0
    for i in order:
        cost = 1 if i == 0 else 2
        if cnt + cost > M:
            continue
        keep[i] = True
        cnt += cost
    return np.fft.irfft(np.where(keep, X, 0), n=len(x))


def denoise(x, h, thr, level=J):
    c = wavedec(x, h, level)
    out = [c[0]] + [np.where(np.abs(d) > thr, d, 0.0) for d in c[1:]]
    kept = 1 + sum(int(np.sum(np.abs(d) > thr)) for d in c[1:]) + len(c[0]) - 1
    return waverec(out, h), kept


FUNCS = {
    "段差(x = 0.37 でとぶ)": lambda x: np.where(x < 0.37, 0.2, 1.0),
    "段差 + 正弦波": lambda x: np.where(x < 0.6, 0.0, 1.0) + 0.4 * np.sin(4 * PI * x),
    "なめらかな山 sin πx": lambda x: np.sin(PI * x),
    "チャープ sin(40πx²)": lambda x: np.sin(40 * PI * x ** 2),
    "2つの細いパルス": lambda x: np.exp(-((x - 0.3) / 0.01) ** 2) + 0.6 * np.exp(-((x - 0.72) / 0.03) ** 2),
}


def haar_fn(x, j=0, k=0):
    u = 2 ** j * x - k
    return np.where((u >= 0) & (u < 0.5), 1.0, np.where((u >= 0.5) & (u < 1), -1.0, 0.0))


def tiling_figs():
    """時間周波数平面の分け方(フーリエ・短時間フーリエ・ウェーブレット)の3つの図"""
    figs = []
    for title, boxes, col in (
            ("フーリエ変換", [(0, 8, i, i + 1) for i in range(8)], C1),
            ("短時間フーリエ変換(窓の幅が一定)", [(2 * k, 2 * k + 2, 2 * i, 2 * i + 2) for i in range(4) for k in range(4)], C3),
            ("ウェーブレット変換", [(k * 8 / n, (k + 1) * 8 / n, y0, y0 + hgt)
                                 for y0, hgt, n in [(0, 1, 2), (1, 1, 2), (2, 2, 4), (4, 4, 8)] for k in range(n)], C2)):
        fig = go.Figure()
        for x0, x1, y0, y1 in boxes:
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, line=dict(color="black", width=1), fillcolor=col, opacity=0.55)
        fig.update_xaxes(range=[0, 8], showticklabels=False)
        fig.update_yaxes(range=[0, 8], showticklabels=False)
        layout(fig, title=title, xlab="時間", ylab="周波数", height=280, legend=False)
        figs.append(fig)
    return figs


def ref_list(items):
    """折りたたみの中で使う参考文献の一覧(expander の入れ子を避ける)"""
    st.markdown("**参考文献**")
    st.markdown("\n".join(f"- {ref_line(k)}  \n  {w}" for k, w in items))


def history():
    with st.expander("ウェーブレット解析はどのように生まれたか(フーリエ解析の何が問題だったのか)", expanded=False):
        st.markdown(
            "**1. フーリエ解析の弱点**  \n"
            "フーリエ係数 $c_n=\\frac{1}{2\\pi}\\int f(x)e^{-inx}dx$ は**区間全体**の積分で決まる。そのため\n"
            "- 一か所だけの変化(衝撃・段差)が, すべての係数に少しずつ広がる。係数を見ても「いつ・どこで」起きたかがわからない。\n"
            "- 不連続点があると係数の減り方が遅く($1/n$ 程度), 部分和にはギブス現象が残る。\n"
            "- 音声・音楽・地震波のように**周波数が時間とともに変わる信号**(非定常な信号)を, 1つのスペクトルでは表しきれない。")
        st.markdown(
            "**2. 窓をかけたフーリエ変換(ガボール, 1946)**  \n"
            "信号に短い窓をかけて, 少しずつずらしながらフーリエ変換する(短時間フーリエ変換)。"
            "しかし窓の幅は一定なので, 時間の精度と周波数の精度は不確定性 $\\Delta t\\,\\Delta\\omega\\ge\\frac12$ の範囲で**固定**されてしまう"
            "(「2+. 代表的な変換対と不確定性」の節)。短い高周波の変化と, 長く続く低周波の波を, 同じ窓で同時にうまくとらえることはできない。")
        st.markdown(
            "**3. 窓の幅を周波数に合わせて伸び縮みさせる(モルレー, 1980年前後)**  \n"
            "石油探査で地震波の反射データを解析していた地球物理学者モルレーは, 1つの波形を**伸縮・平行移動**した関数(ondelette, 小さな波)で信号を分解する方法を経験的に考えた。"
            "高い周波数には短い波, 低い周波数には長い波を使うので, 下の右の図のように時間周波数平面を分けられる。"
            "物理学者グロスマンとともに, 連続ウェーブレット変換と, そこから信号を復元する公式を示した(1984)。")
        for col_, fig_ in zip(st.columns(3), tiling_figs()):
            with col_:
                show(fig_)
        st.caption("長方形1つが, 1つの係数が表す「時間の幅 × 周波数の幅」。フーリエ変換は周波数だけを細かく分け, 時間の情報をもたない。"
                   "短時間フーリエ変換とウェーブレット変換では長方形の面積は同じ(不確定性で下限が決まる)だが, "
                   "ウェーブレットでは高い周波数ほど時間の幅が短く(細かい時刻がわかる), 低い周波数ほど周波数の幅が狭い。")
        st.markdown(
            "**4. 数学の理論へ(1985年〜)**  \n"
            "- メイエは, その復元公式が調和解析で知られていたカルデロンの公式と同じであることに気づき, **なめらかな正規直交ウェーブレット基底**を構成した。\n"
            "- マラーとメイエは, 粗い近似と細かい補正に分ける**多重解像度解析**の枠組みをつくり, 高速なウェーブレット変換を可能にした(マラー, 1989)。\n"
            "- ドブシーは, 有限の区間だけで値をもつ(台がコンパクトな)なめらかな直交ウェーブレットを構成した(1988)。\n"
            "- 実は 1910 年のハールの関数系は, 最も簡単なウェーブレット基底(このページで扱うもの)だった。")
        st.markdown(
            "**5. 応用の広がり**  \n"
            "データ圧縮(画像の JPEG 2000 など), 雑音除去, 医用画像, ハッブル宇宙望遠鏡の画像処理, "
            "重力波の検出(LIGO, 2015年)の信号解析などに使われている。メイエはウェーブレット理論の功績により 2017 年のアーベル賞を受賞した。")
        ref_list([("abel2017", "ウェーブレット理論の歴史と応用(アーベル賞の授賞理由)。"),
                   ("gabor1946", "窓をかけたフーリエ変換。"), ("grossmann1984", "連続ウェーブレット変換。"),
                   ("mallat1989", "多重解像度解析。"), ("daubechies1988", "台がコンパクトな直交ウェーブレット。"),
                   ("haar1910", "ハール関数系。")])


def tab_haar():
    st.info("フーリエ級数は, 全区間に広がる波 $\\cos nx,\\ \\sin nx$ の重ね合わせ。ウェーブレットは, **短い波を伸び縮み・平行移動したもの**の重ね合わせで, "
            "「どこで」「どのくらい細かい」変化が起きているかを同時に表せる。最も簡単なのがハール・ウェーブレット(1910)である。")
    history()
    st.subheader("ハール関数")
    st.latex(r"\varphi(x)=\begin{cases}1&(0\le x<1)\\0&(\text{それ以外})\end{cases},\qquad "
             r"\psi(x)=\begin{cases}1&(0\le x<\frac12)\\-1&(\frac12\le x<1)\\0&(\text{それ以外})\end{cases},\qquad "
             r"\psi_{j,k}(x)=\psi(2^jx-k)")
    st.markdown("$\\psi_{j,k}$ は $\\psi$ を幅 $2^{-j}$ に縮めて $k\\cdot2^{-j}$ だけずらしたもの($k=0,1,\\ldots,2^j-1$ で $[0,1)$ をうめる)。"
                "異なる $(j,k)$ どうしは**直交**する:$\\displaystyle\\int_0^1\\psi_{j,k}\\psi_{j',k'}dx=0$。")
    c = st.columns(2)
    j = c[0].slider("j(細かさ)", 0, 4, 1, key="hw_j")
    k = c[1].slider("k(位置)", 0, 2 ** j - 1, min(1, 2 ** j - 1), key="hw_k") if j > 0 else 0
    xx = np.linspace(0, 1, 2001, endpoint=False)
    fig = go.Figure(go.Scatter(x=xx, y=haar_fn(xx) * 0 + 1, mode="lines", line=dict(color=GRAY, dash="dot"), name="φ(x)"))
    fig.add_trace(go.Scatter(x=xx, y=haar_fn(xx), mode="lines", line=dict(color=C1, width=2), name="ψ(x)"))
    fig.add_trace(go.Scatter(x=xx, y=haar_fn(xx, j, k), mode="lines", line=dict(color=C2, width=3), name=f"ψ_{{{j},{k}}}(x)"))
    layout(fig, title=f"ψ_{{{j},{k}}}(x) = ψ(2^{j}x − {k})", xlab="x", height=300, yrange=(-1.4, 1.4))
    show(fig)

    st.subheader("多重解像度:粗い近似 + 細かい補正")
    st.markdown("幅 $2^{-j}$ の区間ごとに一定な関数の全体を $V_j$ とする($V_0\\subset V_1\\subset V_2\\subset\\cdots$)。"
                "$f$ を区間ごとの**平均**でおきかえた $P_jf$ が, $V_j$ での最良の近似(2乗誤差が最小)になる。$P_{j+1}f$ と $P_jf$ の差は, "
                "ハール関数 $\\psi_{j,k}$ の和で書ける:")
    st.latex(r"P_{j+1}f=P_jf+\sum_{k=0}^{2^j-1}d_{j,k}\,\psi_{j,k},\qquad "
             r"d_{j,k}=\frac{(\text{左半分の平均})-(\text{右半分の平均})}{2}=2^j\int_0^1f(x)\psi_{j,k}(x)\,dx")
    st.latex(r"f=P_0f+\sum_{j=0}^{\infty}\sum_{k}d_{j,k}\psi_{j,k}\qquad(P_0f=\int_0^1f\,dx\ \text{は全体の平均})")
    c = st.columns(2)
    fname = c[0].selectbox("関数 f", list(FUNCS), key="hw_f")
    lev = c[1].slider("近似のレベル j(区間の数 2^j)", 0, 8, 3, key="hw_lev")
    f = FUNCS[fname](XS)
    Pj, Pj1 = proj(f, lev), proj(f, lev + 1)
    cA, cB = st.columns(2)
    with cA:
        fig = go.Figure(go.Scatter(x=XS, y=f, mode="lines", line=dict(color=GRAY, width=2), name="f"))
        fig.add_trace(go.Scatter(x=XS, y=Pj1, mode="lines", line=dict(color=C3, width=1.5, shape="hv"), name=f"P_{lev + 1} f"))
        fig.add_trace(go.Scatter(x=XS, y=Pj, mode="lines", line=dict(color=C1, width=2.5, shape="hv"), name=f"P_{lev} f"))
        layout(fig, title=f"近似 P_{lev} f(2^{lev} = {2 ** lev} 区間の平均)", xlab="x", height=340)
        show(fig)
    with cB:
        fig = go.Figure(go.Scatter(x=XS, y=Pj1 - Pj, mode="lines", line=dict(color=C2, width=2, shape="hv"), name="補正"))
        layout(fig, title=f"補正 P_{lev + 1} f − P_{lev} f = Σ d_{{{lev},k}} ψ_{{{lev},k}}", xlab="x", height=340, legend=False)
        show(fig)
    st.markdown("**係数の大きさ**:レベル $j$ ごとに $|d_{j,k}|$ を並べると, $f$ が急に変わる場所(段差・パルス・振動が速い所)だけ係数が大きい。")
    a, ds = haar_full(f)
    Z = np.full((J, NS), np.nan)
    for jj, d in enumerate(ds):
        Z[jj] = np.repeat(np.abs(d), NS // len(d))
    fig = go.Figure(go.Heatmap(x=XS, y=list(range(J)), z=Z, colorscale="Viridis", colorbar=dict(title="|d|")))
    layout(fig, title="|d_{j,k}|(横:位置 x, 縦:レベル j)", xlab="x", ylab="j", height=320, legend=False)
    show(fig)
    st.caption("なめらかな部分では, 細かいレベルの係数はほとんど 0(左右の平均がほぼ等しい)。だから少ない係数で f を表せる(圧縮)。")

    st.subheader("手で計算する:4つのデータのハール変換")
    st.markdown("データ $(x_0,x_1,x_2,x_3)=(9,\\ 7,\\ 3,\\ 5)$ を, 隣どうしの**平均**と**差の半分**に分ける操作をくり返す。")
    st.markdown(r"""
| 段階 | 平均(近似) | 差の半分(補正) |
|---|---|---|
| 1回目 | $\frac{9+7}{2}=8,\ \frac{3+5}{2}=4$ | $\frac{9-7}{2}=1,\ \frac{3-5}{2}=-1$ |
| 2回目 | $\frac{8+4}{2}=6$ | $\frac{8-4}{2}=2$ |
""")
    st.markdown("ハール変換は $(6\\ |\\ 2\\ |\\ 1,\\ -1)$。逆に $6\\pm2=(8,\\ 4)$, $8\\pm1=(9,\\ 7)$, $4\\pm(-1)=(3,\\ 5)$ でもとに戻る。"
                "関数で書くと, $[0,1)$ を4等分した階段関数 $f$ について")
    st.latex(r"f(x)=6\,\varphi(x)+2\,\psi(x)+1\cdot\psi(2x)-1\cdot\psi(2x-1)")

    st.subheader("フーリエ級数とくらべる")
    c = st.columns(2)
    fname2 = c[0].selectbox("関数", list(FUNCS), key="hw_f2")
    M = c[1].slider("使う係数の数 M", 4, 128, 16, 4, key="hw_M")
    f2 = FUNCS[fname2](XS)
    yh, yd, yf = best_m(f2, HAAR_H, M), best_m(f2, D4_H, M), fourier_best_m(f2, M)
    cA, cB = st.columns(2)
    with cA:
        fig = go.Figure(go.Scatter(x=XS, y=f2, mode="lines", line=dict(color=GRAY, width=2.5), name="f"))
        fig.add_trace(go.Scatter(x=XS, y=yf, mode="lines", line=dict(color=C1), name="フーリエ"))
        fig.add_trace(go.Scatter(x=XS, y=yh, mode="lines", line=dict(color=C2, shape="hv"), name="ハール"))
        fig.add_trace(go.Scatter(x=XS, y=yd, mode="lines", line=dict(color=C3, dash="dash"), name="ドブシー D4"))
        layout(fig, title=f"大きい係数を M = {M} 個だけ残した近似", xlab="x", height=360)
        show(fig)
    with cB:
        Ms = np.array([4, 8, 16, 32, 64, 128, 256])
        err = lambda y: math.sqrt(float(np.mean((y - f2) ** 2)))  # noqa: E731
        fig = go.Figure()
        for fn_, nm, col in ((lambda m: fourier_best_m(f2, m), "フーリエ", C1), (lambda m: best_m(f2, HAAR_H, m), "ハール", C2),
                             (lambda m: best_m(f2, D4_H, m), "ドブシー D4", C3)):
            fig.add_trace(go.Scatter(x=Ms, y=[err(fn_(m)) for m in Ms], mode="lines+markers", line=dict(color=col), name=nm))
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
        layout(fig, title="2乗平均誤差(両対数)", xlab="M", height=360)
        show(fig)
    st.caption("段差やパルスのような**局所的な変化**は, ウェーブレットが少ない係数で表せる(フーリエ級数ではギブス現象が出て, 誤差が減りにくい)。"
               "なめらかな関数では, ハールより, なめらかなドブシーのウェーブレットやフーリエ級数が有利。"
               "フーリエの方はグラフの両端(x = 0 と 1)をつないだ周期関数として近似している。")
    with st.expander("ドブシーのウェーブレット D4(参考)"):
        st.markdown("ハールは $(x_{2k}+x_{2k+1})/\\sqrt2$ のように2つの値で平均をとる。ドブシー(1988)は4つの値を使う係数")
        st.latex(r"h=\frac{1}{4\sqrt2}\left(1+\sqrt3,\ 3+\sqrt3,\ 3-\sqrt3,\ 1-\sqrt3\right)")
        st.markdown("を見つけ, 直交性を保ったまま, なめらかなウェーブレットをつくった。画像圧縮(JPEG 2000)なども, このような「なめらかなウェーブレット」の考え方を使っている。"
                    "Python では PyWavelets(`pywt`)で多くのウェーブレットが使える。")

    with st.expander("練習問題"):
        st.markdown("**問1** データ $(6,\\ 2,\\ 5,\\ 7)$ のハール変換(平均と差の半分を2回)を求めよ。")
        st.markdown("**問2** $\\displaystyle\\int_0^1x\\,\\psi(2x-1)\\,dx$ を求めよ。")
        st.markdown("**問3** $\\psi$ と $\\psi(2x)$, $\\psi(2x)$ と $\\psi(2x-1)$ が直交することを確かめよ。")
        if st.toggle("解答・解説を表示", key="pr_haar"):
            st.markdown("**答1** 1回目:平均 $(4,\\ 6)$, 差の半分 $(2,\\ -1)$。2回目:平均 $5$, 差の半分 $-1$。よって $(5\\ |\\ -1\\ |\\ 2,\\ -1)$。")
            st.markdown("**答2** $\\psi(2x-1)$ は $[\\frac12,\\frac34)$ で 1, $[\\frac34,1)$ で −1 なので "
                        "$\\displaystyle\\int_{1/2}^{3/4}x\\,dx-\\int_{3/4}^{1}x\\,dx=\\frac{5}{32}-\\frac{7}{32}=-\\frac{1}{16}$。")
            st.markdown("**答3** $\\psi(2x)$ は $[0,\\frac12)$ だけで値をもち, そこで $\\psi=1$ なので $\\int\\psi\\cdot\\psi(2x)=\\int_0^{1/2}\\psi(2x)dx=0$。"
                        "$\\psi(2x)$ と $\\psi(2x-1)$ は値をもつ区間が重ならないので, 積は 0。")
    page_refs([("arai2022", "フーリエ解析からウェーブレットへ(日本語)。"),
               ("haar1910", "ハール関数系の原論文。"),
               ("daubechies1992", "なめらかで台がコンパクトな直交ウェーブレット(D4 など)。"),
               ("mallat2009", "多重解像度解析と信号処理への応用。"),
               ("pywt2019", "Python のウェーブレット変換ライブラリ。")])


# ============================================================
# 応用:ウェーブレットの応用(機械・電気電子・都市・化学)
# ============================================================
def level_bands(fs, L):
    return [(fs / 2 ** (j + 1), fs / 2 ** j) for j in range(1, L + 1)]


def detail_map(x, h, L):
    """各レベルの |d| を元の時刻の長さに引きのばした表(行 j = 1 が最も細かい)"""
    c = wavedec(x, h, L)
    rows = [np.repeat(np.abs(d), len(x) // len(d)) for d in c[:0:-1]]
    return np.array(rows)


def tab_wavelet_apps():
    st.info("ウェーブレット変換は, 信号の**どの時刻に, どの周波数帯の変化があるか**を少ない係数で表す。"
            "異常の検出・雑音の除去・データの圧縮などに使われる。ここでは 4 学科の例を, ハール(またはドブシー D4)の変換で体験する。")
    st.markdown("サンプリング周波数 $f_s$ のデータでは, レベル $j$ の係数 $d_j$ はおよそ $\\dfrac{f_s}{2^{j+1}}$〜$\\dfrac{f_s}{2^j}$ の周波数帯を受けもち, "
                "1つの係数が $\\dfrac{2^j}{f_s}$ 秒ぶんの時間を表す(細かいレベルほど時間の精度がよく, 周波数の幅が広い:不確定性)。")
    wname = st.radio("ウェーブレット", ["ハール", "ドブシー D4"], horizontal=True, key="wa_w")
    h = HAAR_H if wname == "ハール" else D4_H
    rng = np.random.default_rng(0)

    st.subheader("機械:軸受の傷による衝撃を見つける")
    fs = 1024
    t = np.arange(fs) / fs
    c = st.columns(2)
    t0 = c[0].slider("衝撃が起きる時刻 [s]", 0.05, 0.95, 0.62, 0.01, key="wa_t0")
    amp = c[1].slider("衝撃の大きさ", 0.0, 2.0, 0.8, 0.1, key="wa_amp")
    burst = amp * np.exp(-((t - t0) / 0.004) ** 2) * np.sin(2 * PI * 300 * t)
    x = np.sin(2 * PI * 50 * t) + burst + 0.05 * rng.standard_normal(fs)
    D = detail_map(x, h, 6)
    cA, cB = st.columns(2)
    with cA:
        fig = go.Figure(go.Scatter(x=t, y=x, mode="lines", line=dict(color=C1, width=1), name="振動"))
        layout(fig, title="振動の波形(50 Hz の回転 + 衝撃 + 雑音)", xlab="t [s]", height=300, legend=False)
        show(fig)
        F = np.abs(np.fft.rfft(x)) / fs * 2
        fig = go.Figure(go.Scatter(x=np.fft.rfftfreq(fs, 1 / fs), y=F, mode="lines", line=dict(color=GRAY), name="|X|"))
        layout(fig, title="フーリエスペクトル(いつ起きたかはわからない)", xlab="周波数 [Hz]", height=260, legend=False)
        show(fig)
    with cB:
        bands = level_bands(fs, 6)
        fig = go.Figure(go.Heatmap(x=t, y=[f"d{j}({lo:.0f}〜{hi:.0f} Hz)" for j, (lo, hi) in enumerate(bands, 1)], z=D,
                                   colorscale="Viridis", colorbar=dict(title="|d|")))
        fig.add_vline(x=t0, line=dict(color="white", dash="dot"))
        layout(fig, title="ウェーブレット係数 |d_j|(時刻 × 周波数帯)", xlab="t [s]", height=580, legend=False)
        show(fig)
    st.caption("フーリエスペクトルでは衝撃の成分(300 Hz 付近)が小さく広がり, 起きた時刻もわからない。"
               "ウェーブレット係数では, 256〜512 Hz の帯(d1)と 128〜256 Hz の帯(d2)の, 衝撃の時刻だけが明るくなる。")

    st.subheader("電気電子:雑音の除去と圧縮")
    c = st.columns(2)
    sig = c[0].selectbox("信号", ["パルス列の電圧", "段差 + 正弦波"], key="wa_sig")
    thr = c[1].slider("しきい値(これより小さい係数を 0 にする)", 0.0, 1.5, 0.5, 0.01, key="wa_thr")
    clean = (np.where((XS % 0.25) < 0.1, 1.0, 0.0) if sig.startswith("パルス") else FUNCS["段差 + 正弦波"](XS))
    noisy = clean + 0.15 * rng.standard_normal(NS)
    y, kept = denoise(noisy, h, thr)
    fig = go.Figure(go.Scatter(x=XS, y=noisy, mode="lines", line=dict(color=GRAY, width=1), name="雑音を含む信号"))
    fig.add_trace(go.Scatter(x=XS, y=clean, mode="lines", line=dict(color="black", width=1.5, dash="dot"), name="もとの信号"))
    fig.add_trace(go.Scatter(x=XS, y=y, mode="lines", line=dict(color=C2, width=2), name="雑音除去後"))
    layout(fig, title="しきい値処理による雑音除去", xlab="時刻(1 周期を 1 とした)", height=340)
    show(fig)
    m = st.columns(3)
    m[0].metric("残した係数", f"{kept} / {NS}")
    m[1].metric("圧縮率", f"{NS / max(kept, 1):.1f} 倍")
    m[2].metric("もとの信号との2乗平均誤差", f"{math.sqrt(float(np.mean((y - clean) ** 2))):.3f}"
                f"(処理前 {math.sqrt(float(np.mean((noisy - clean) ** 2))):.3f})")
    st.caption("雑音は細かいレベルに小さい係数として広く散らばり, 信号の段差は少数の大きい係数になる。小さい係数を 0 にすると, 段差を保ったまま雑音が消える"
               "(フーリエのローパスフィルタでは段差もなまる)。しきい値の目安は, 雑音の標準偏差 σ = 0.15 とデータ数 N から σ√(2 log N) ≈ 0.56(ドノホー–ジョンストン)。画像圧縮の JPEG 2000 も, ウェーブレット係数の多くを 0 に近い値として省くことで圧縮する。")

    st.subheader("都市:地震波の P 波・S 波の到着を読む")
    fs2 = 128
    t2 = np.arange(2048) / fs2
    c = st.columns(2)
    tp = c[0].slider("P 波の到着時刻 [s]", 1.0, 6.0, 3.0, 0.1, key="wa_tp")
    sp = c[1].slider("S − P 時間 [s]", 1.0, 8.0, 5.0, 0.1, key="wa_sp")
    ts = tp + sp
    env = lambda t, t1, tau: np.where(t >= t1, (1 - np.exp(-(t - t1) / 0.3)) * np.exp(-(t - t1) / tau), 0.0)  # noqa: E731
    w = 0.3 * env(t2, tp, 4) * np.sin(2 * PI * 9 * t2) + 1.0 * env(t2, ts, 3) * np.sin(2 * PI * 2.5 * t2 + 1) \
        + 0.03 * rng.standard_normal(len(t2))
    D2 = detail_map(w, h, 6)
    fig = go.Figure(go.Scatter(x=t2, y=w, mode="lines", line=dict(color=C1, width=1), name="揺れ"))
    for tt, nm in ((tp, "P"), (ts, "S")):
        fig.add_vline(x=tt, line=dict(color=C2, dash="dot"), annotation_text=nm)
    layout(fig, title="地震計の記録(例)", xlab="t [s]", height=280, legend=False)
    show(fig)
    bands2 = level_bands(fs2, 6)
    fig = go.Figure(go.Heatmap(x=t2, y=[f"d{j}({lo:.1f}〜{hi:.1f} Hz)" for j, (lo, hi) in enumerate(bands2, 1)], z=D2,
                               colorscale="Viridis", colorbar=dict(title="|d|")))
    layout(fig, title="ウェーブレット係数 |d_j|", xlab="t [s]", height=360, legend=False)
    show(fig)
    st.caption(f"P 波(9 Hz 付近, d3・d4)が先に, S 波(2.5 Hz 付近, d5・d6)が後から現れる。"
               f"S − P 時間 {sp:g} 秒から, 震源までの距離はおよそ {7 * sp:.0f}〜{8 * sp:.0f} km(大森公式:距離 ≈ k × (S − P 時間), k ≈ 7〜8 km/s)。")

    st.subheader("化学:スペクトルのベースライン補正と雑音除去")
    nu = np.linspace(1000, 2000, NS)
    c = st.columns(2)
    jb = c[0].slider("ベースラインとみなす近似のレベル j", 1, 6, 3, key="wa_jb")
    thr2 = c[1].slider("雑音のしきい値", 0.0, 0.3, 0.06, 0.01, key="wa_thr2")
    peaks = np.exp(-((nu - 1300) / 6) ** 2) + 0.6 * np.exp(-((nu - 1650) / 10) ** 2) + 0.4 * np.exp(-((nu - 1720) / 5) ** 2)
    base = 0.8 + 0.6 * ((nu - 1000) / 1000) ** 2 + 0.3 * np.sin((nu - 1000) / 1000 * PI)
    spec = peaks + base + 0.03 * rng.standard_normal(NS)
    bl = approx_level(spec, h, jb)
    corr, _ = denoise(spec - bl, h, thr2, level=J - jb)
    fig = go.Figure(go.Scatter(x=nu, y=spec, mode="lines", line=dict(color=GRAY, width=1), name="測定スペクトル"))
    fig.add_trace(go.Scatter(x=nu, y=bl, mode="lines", line=dict(color=C4, width=2), name=f"ベースライン(レベル {jb} の近似)"))
    fig.add_trace(go.Scatter(x=nu, y=corr, mode="lines", line=dict(color=C2, width=2), name="補正後"))
    fig.add_trace(go.Scatter(x=nu, y=peaks, mode="lines", line=dict(color="black", dash="dot", width=1), name="本当のピーク"))
    layout(fig, title="ゆっくり変わるベースラインと細いピークを分ける", xlab="波数 [cm⁻¹]", height=360)
    show(fig)
    st.caption(f"レベル {jb} の近似は幅 {1000 / 2 ** jb:.0f} cm⁻¹ ごとの(なめらかな)平均で, ゆっくり変わるベースラインを表す。"
               "j を大きくしすぎるとピークまでベースラインに入ってしまい, 小さすぎるとベースラインが残る。")
    examples("wavelet_apps", WAVE_EX)
    page_refs([("mallat2009", "時間周波数解析, しきい値による雑音除去と圧縮。"),
               ("arai2022", "ウェーブレットの理論と応用(日本語)。"),
               ("pywt2019", "Python で多くのウェーブレットを使うためのライブラリ。"),
               ("daubechies1992", "ドブシーのウェーブレット。")])


WAVE_EX = [
    (r"【機械】サンプリング周波数 $1024\ \mathrm{Hz}$ の振動データをウェーブレット変換した。レベル $3$ の係数 $d_3$ が受けもつ周波数帯と, "
     r"1つの係数が表す時間の長さを求めよ。",
     [r"レベル $j$ の帯は $\dfrac{f_s}{2^{j+1}}$〜$\dfrac{f_s}{2^j}$ なので, $j=3$ では $64$〜$128\ \mathrm{Hz}$。",
      r"時間の長さは $\dfrac{2^3}{1024}=\dfrac{1}{128}\ \mathrm{s}\approx7.8\ \mathrm{ms}$。"]),
    (r"【電気電子】データ $(8,\ 6,\ 2,\ 2)$ のハール変換(平均と差の半分を2回)を求めよ。"
     r"次に, 絶対値が最小の 0 でない係数を 0 にして逆変換し, もとのデータとの差の2乗和を求めよ。",
     [r"1回目:平均 $(7,\ 2)$, 差の半分 $(1,\ 0)$。2回目:平均 $\frac92$, 差の半分 $\frac52$。変換は $\left(\frac92\ \middle|\ \frac52\ \middle|\ 1,\ 0\right)$。",
      r"$1$ を 0 にすると, $\frac92\pm\frac52=(7,\ 2)$, $7\pm0=(7,\ 7)$, $2\pm0=(2,\ 2)$ で $(7,\ 7,\ 2,\ 2)$。",
      r"差の2乗和は $1^2+1^2=2$($=2\times1^2$:細かいレベルの係数 $d$ を消すと誤差は $2d^2$)。"]),
    (r"【都市】S − P 時間(P 波が届いてから S 波が届くまで)が $5$ 秒だった。大森公式 $D=kT$($k=7.5\ \mathrm{km/s}$)で, 震源までの距離 $D$ を求めよ。",
     [r"$D=7.5\times5=37.5\ \mathrm{km}$。",
      r"P 波と S 波は周波数帯が違うので, ウェーブレット係数のレベルごとに到着時刻を読みやすい。"]),
    (r"【化学】$1000$〜$2000\ \mathrm{cm^{-1}}$ を $1024$ 点で測ったスペクトルの, レベル $j$ の近似は何 $\mathrm{cm^{-1}}$ ごとの平均か。"
     r"幅 $10\ \mathrm{cm^{-1}}$ のピークを残してベースラインだけを取り出すには, $j$ をどのくらいにすればよいか。",
     [r"レベル $j$ の近似は $2^j$ 個の区間の平均なので, 幅は $\dfrac{1000}{2^j}\ \mathrm{cm^{-1}}$。",
      r"ピークの幅 $10\ \mathrm{cm^{-1}}$ より十分広い区間で平均すればピークは近似に入らない:$\dfrac{1000}{2^j}\ge50$ 程度として $j\le4$。"
      r"ただし平均の幅が広すぎるとベースラインの変化も追えないので, $j=3,\ 4$ 程度がよい(上の図で確かめよう)。"]),
]
