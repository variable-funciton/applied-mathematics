"""③ フーリエ解析 第3章 離散フーリエ変換
  1. 離散フーリエ変換の定義と性質
  2. 離散フーリエ変換と複素フーリエ級数(fourier2.tab_dft_series)
  3. 高速フーリエ変換(FFT)のしくみ
  4. FFT とフィルタ(fourier.tab_dft)
  5. シャノンの標本化定理(⑥ その他から移して拡充)

規約(アプリ全体と同じ):X_k = Σ_{n=0}^{N−1} x_n e^{−2πikn/N},  x_n = (1/N) Σ_k X_k e^{2πikn/N}
"""
import io
import math
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.io import wavfile

from common import C1, C2, C3, C4, GRAY, layout, show
from topics.fourier_kernels import page_refs

PI = np.pi


def ctex(z, d=4):
    """複素数を a+bi の形に(整数なら整数で)"""
    re, im = float(np.real(z)), float(np.imag(z))
    f = lambda v: (str(int(round(v))) if abs(v - round(v)) < 1e-9 else f"{v:.{d}g}")  # noqa: E731
    if abs(im) < 1e-9:
        return f(re)
    ims = ("" if abs(abs(im) - 1) < 1e-9 else f(abs(im))) + "i"
    if abs(re) < 1e-9:
        return ("-" if im < 0 else "") + ims
    return f(re) + ("-" if im < 0 else "+") + ims


def stem(n, y, name, color, fig=None):
    fig = fig or go.Figure()
    xs, ys = [], []
    for a, b in zip(n, y):
        xs += [a, a, None]
        ys += [0, b, None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=2), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=n, y=y, mode="markers", marker=dict(color=color, size=9), name=name))
    return fig


# ============================================================
# 1. 定義と性質
# ============================================================
SIGNALS = {
    "単位インパルス δ_n": lambda n, N, m: (n == 0).astype(float),
    "定数 1": lambda n, N, m: np.ones(N),
    "cos(2πmn/N)": lambda n, N, m: np.cos(2 * PI * m * n / N),
    "sin(2πmn/N)": lambda n, N, m: np.sin(2 * PI * m * n / N),
    "矩形(n < m で 1)": lambda n, N, m: (n < m).astype(float),
    "(1, 2, 3, 4, …)": lambda n, N, m: n + 1.0,
}


def tab_definition():
    st.caption("大日本図書の教科書では, 第2章のコラム「離散フーリエ変換と複素フーリエ級数」で扱われている内容を, 独立した章として詳しく扱う。")
    st.info("コンピュータで扱うデータは, 有限個の数の列 $x_0,x_1,\\ldots,x_{N-1}$。これを $N$ 個の周波数成分に分けるのが**離散フーリエ変換**(DFT)。"
            "積分を有限の和におきかえたフーリエ変換で, 逆変換も有限の和で正確に計算できる。")
    st.subheader("定義")
    st.latex(r"X_k=\sum_{n=0}^{N-1}x_n\,W^{kn}\quad(k=0,1,\ldots,N-1),\qquad W=e^{-2\pi i/N}")
    st.latex(r"x_n=\frac1N\sum_{k=0}^{N-1}X_k\,W^{-kn}\quad(\text{逆離散フーリエ変換})")
    st.markdown("$W^{kn}=e^{-2\\pi ikn/N}$ は, $n$ について $N$ 回で $k$ 周する回転。$X_k$ は「$x_n$ に $k$ 周する波がどれだけ含まれているか」を表す。")
    with st.expander("逆変換の公式が成り立つ理由(直交性)"):
        st.markdown("等比数列の和から, $W^{N}=1$ に注意して")
        st.latex(r"\sum_{n=0}^{N-1}W^{(k-l)n}=\begin{cases}N&(k\equiv l\pmod N)\\[2pt]\dfrac{1-W^{(k-l)N}}{1-W^{k-l}}=0&(\text{それ以外})\end{cases}")
        st.markdown("これを使うと $\\frac1N\\sum_kX_kW^{-kl}=\\frac1N\\sum_nx_n\\sum_kW^{k(n-l)}=x_l$。")
    st.subheader("行列で書く(N = 4)")
    st.markdown("$N=4$ では $W=e^{-\\pi i/2}=-i$ なので, $W^{kn}$ は $1,\\ -i,\\ -1,\\ i$ のくり返し:")
    st.latex(r"\begin{pmatrix}X_0\\X_1\\X_2\\X_3\end{pmatrix}=\begin{pmatrix}1&1&1&1\\1&-i&-1&i\\1&-1&1&-1\\1&i&-1&-i\end{pmatrix}"
             r"\begin{pmatrix}x_0\\x_1\\x_2\\x_3\end{pmatrix}")
    st.markdown("例:$x=(1,\\ 2,\\ 3,\\ 4)$ なら $X_0=10$, $X_1=1-2i-3+4i=-2+2i$, $X_2=1-2+3-4=-2$, $X_3=1+2i-3-4i=-2-2i$。")

    st.subheader("動かして確かめる")
    c = st.columns(3)
    N = c[0].select_slider("データの数 N", [4, 8, 16, 32], value=8, key="df_N")
    sname = c[1].selectbox("データ x_n", list(SIGNALS), index=2, key="df_s")
    m = c[2].slider("m", 0, N - 1, min(1, N - 1), key="df_m")
    n = np.arange(N)
    x = SIGNALS[sname](n, N, m)
    X = np.fft.fft(x)
    cA, cB = st.columns(2)
    with cA:
        fig = stem(n, x, "x_n", C1)
        layout(fig, title="データ x_n", xlab="n", height=300, legend=False)
        show(fig)
    with cB:
        fig = stem(n - 0.08, X.real, "Re X_k", C2)
        stem(n + 0.08, X.imag, "Im X_k", C3, fig)
        layout(fig, title="DFT X_k(実部と虚部)", xlab="k", height=300)
        show(fig)
    if N <= 8:
        st.markdown("$X_k$ = " + ", ".join(f"${ctex(v)}$" for v in X))
    xr = np.fft.ifft(X)
    st.caption(f"逆変換でもとに戻ることの確認:最大のずれ {float(np.max(np.abs(xr - x))):.1e}。"
               "cos(2πmn/N) では k = m と k = N − m に N/2 ずつ現れる($e^{\\pm i\\theta}$ の和なので)。")

    st.subheader("性質")
    st.markdown(r"""
| 性質 | データ | DFT |
|---|---|---|
| 線形性 | $ax_n+by_n$ | $aX_k+bY_k$ |
| 周期性 | — | $X_{k+N}=X_k$ |
| 実数のデータ | $x_n$ が実数 | $X_{N-k}=\overline{X_k}$(半分だけ計算すればよい) |
| 巡回シフト | $x_{n-m}$(添字は $N$ を法として) | $W^{km}X_k$ |
| 巡回たたみこみ | $(x*y)_n=\sum_{j}x_jy_{n-j}$ | $X_kY_k$ |
| パーセバルの等式 | $\sum\lvert x_n\rvert^2$ | $\frac1N\sum\lvert X_k\rvert^2$ |
""")
    st.caption("第1章・第2章のフーリエ級数・フーリエ変換の性質(平行移動・たたみこみ・パーセバル)が, そのまま有限の和の形で成り立つ。")
    with st.expander("練習問題"):
        st.markdown("**問1** $x=(2,\\ 0,\\ 1,\\ 1)$ の DFT を求めよ($N=4$)。")
        st.markdown("**問2** 問1 で, パーセバルの等式 $\\sum|x_n|^2=\\frac14\\sum|X_k|^2$ を確かめよ。")
        st.markdown("**問3** $X=(4,\\ 0,\\ 0,\\ 0)$ の逆 DFT を求めよ。")
        if st.toggle("解答・解説を表示", key="pr_dft_def"):
            st.markdown("**答1** $X_0=4$, $X_1=2+0\\cdot(-i)+1\\cdot(-1)+1\\cdot i=1+i$, $X_2=2-0+1-1=2$, $X_3=2+0\\cdot i-1+1\\cdot(-i)=1-i$。")
            st.markdown("**答2** 左辺 $4+0+1+1=6$。右辺 $\\frac14(16+2+4+2)=6$。")
            st.markdown("**答3** $x_n=\\frac14\\cdot4=1$ なので $(1,\\ 1,\\ 1,\\ 1)$(定数の列は $k=0$ の成分だけ)。")
    page_refs([("oishi2019", "離散フーリエ変換と FFT の入門。"), ("oppenheim1997", "離散時間の信号とフーリエ変換(電気電子)。"),
               ("bracewell2000", "離散フーリエ変換の性質と応用。")])


# ============================================================
# 3. FFT のしくみ
# ============================================================
def fft_rec(x):
    """クーリー–テューキー(時間間引き, 基数 2)の FFT。len(x) は 2 のべき"""
    x = np.asarray(x, dtype=complex)
    N = len(x)
    if N == 1:
        return x
    E, O = fft_rec(x[0::2]), fft_rec(x[1::2])
    w = np.exp(-2j * PI * np.arange(N // 2) / N)
    return np.concatenate([E + w * O, E - w * O])


def dft_direct(x):
    N = len(x)
    n = np.arange(N)
    return np.exp(-2j * PI * np.outer(n, n) / N) @ np.asarray(x, dtype=complex)


def bitrev(N):
    b = int(math.log2(N))
    return [int(format(i, f"0{b}b")[::-1], 2) for i in range(N)]


def butterfly_fig(N=8):
    """N 点 FFT のバタフライ図(入力はビット反転の順)"""
    L = int(math.log2(N))
    fig = go.Figure()
    for s in range(L):
        half = 2 ** s
        for start in range(0, N, 2 * half):
            for j in range(half):
                a, b = start + j, start + j + half
                for (p, q) in ((a, a), (a, b), (b, a), (b, b)):
                    fig.add_trace(go.Scatter(x=[s, s + 1], y=[-p, -q], mode="lines", showlegend=False, hoverinfo="skip",
                                             line=dict(color=C1 if p == q else C2, width=1.6)))
                k = j * (N // (2 * half))
                fig.add_annotation(x=s + 0.75, y=-b + 0.18, text=f"W<sup>{k}</sup>", showarrow=False, font=dict(size=11, color=C2))
    br = bitrev(N)
    for i in range(N):
        fig.add_annotation(x=-0.15, y=-i, text=f"x<sub>{br[i]}</sub>", showarrow=False, xanchor="right", font=dict(size=13))
        fig.add_annotation(x=L + 0.15, y=-i, text=f"X<sub>{i}</sub>", showarrow=False, xanchor="left", font=dict(size=13))
    fig.update_xaxes(visible=False, range=[-0.8, L + 0.8])
    fig.update_yaxes(visible=False)
    layout(fig, title=f"N = {N} の FFT のバタフライ図({L} 段, W = e^(−2πi/{N}))", height=360, legend=False)
    return fig


def tab_fft():
    st.info("DFT をそのまま計算すると, $N$ 個の $X_k$ のそれぞれに $N$ 回の掛け算が必要で, 全部で $N^2$ 回。"
            "**高速フーリエ変換**(FFT)は, 計算を半分ずつに分けることをくり返して, 掛け算を約 $\\frac N2\\log_2N$ 回に減らす。")
    st.subheader("しくみ:偶数番目と奇数番目に分ける")
    st.markdown("$N$ が偶数のとき, $x_n$ を偶数番目 $x_{2m}$ と奇数番目 $x_{2m+1}$ に分けると($W_N=e^{-2\\pi i/N}$, $W_N^2=W_{N/2}$)")
    st.latex(r"X_k=\sum_{m=0}^{N/2-1}x_{2m}W_{N/2}^{km}+W_N^k\sum_{m=0}^{N/2-1}x_{2m+1}W_{N/2}^{km}=E_k+W_N^kO_k")
    st.markdown("$E_k,\\ O_k$ は長さ $N/2$ の DFT で, 周期 $N/2$ をもつ。さらに $W_N^{k+N/2}=-W_N^k$ なので")
    st.latex(r"X_k=E_k+W_N^kO_k,\qquad X_{k+N/2}=E_k-W_N^kO_k\qquad\left(k=0,1,\ldots,\tfrac N2-1\right)")
    st.markdown("1回の掛け算 $W_N^kO_k$ で2つの値が求まる。この組を**バタフライ演算**という。"
                "$N=2^L$ なら, この分割を $L$ 回くり返すと長さ 1 の DFT(何もしない)になる。")
    show(butterfly_fig(8))
    st.caption("左の入力は, 添字を2進数で書いて桁を逆にした順(ビット反転の順:0, 4, 2, 6, 1, 5, 3, 7)。"
               "各段で, 青い線はそのまま, 赤い線は W をかけて足し引きする。")
    st.subheader("計算の回数")
    rows = []
    for L in (3, 6, 10, 16, 20):
        N = 2 ** L
        rows.append({"N": f"{N:,}", "直接計算 N²": f"{N * N:,}", "FFT (N/2)log₂N": f"{N // 2 * L:,}", "何倍速いか": f"{N * N / (N // 2 * L):,.0f}"})
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption("N = 1024 で約 200 倍, N = 100 万(2²⁰)では約 10 万倍。1秒の音声(44100 点)の周波数分析が, ほぼ一瞬でできるようになった。")
    st.subheader("動かして確かめる:計算時間")
    L = st.slider("N = 2^L の L", 6, 12, 10, key="fft_L")
    N = 2 ** L
    rng = np.random.default_rng(0)
    x = rng.standard_normal(N)
    t0 = time.perf_counter()
    Xd = dft_direct(x)
    t1 = time.perf_counter()
    Xr = fft_rec(x)
    t2 = time.perf_counter()
    Xn = np.fft.fft(x)
    t3 = time.perf_counter()
    m = st.columns(3)
    m[0].metric("直接計算(行列 × ベクトル)", f"{1000 * (t1 - t0):.2f} ms")
    m[1].metric("FFT(このページの再帰プログラム)", f"{1000 * (t2 - t1):.2f} ms")
    m[2].metric("FFT(numpy.fft)", f"{1000 * (t3 - t2):.3f} ms")
    st.caption(f"3つの結果の最大の差:{float(max(np.max(np.abs(Xd - Xn)), np.max(np.abs(Xr - Xn)))):.1e}。"
               "時間はサーバーの混み具合で変わる。直接計算は N を2倍にすると約4倍, FFT は約2倍になる。")
    with st.expander("FFT のプログラム(Python)"):
        st.code('''import numpy as np

def fft(x):
    """長さが 2 のべきのデータの FFT(クーリー–テューキー)"""
    N = len(x)
    if N == 1:
        return np.asarray(x, dtype=complex)
    E = fft(x[0::2])                       # 偶数番目の DFT
    O = fft(x[1::2])                       # 奇数番目の DFT
    w = np.exp(-2j * np.pi * np.arange(N // 2) / N)
    return np.concatenate([E + w * O, E - w * O])   # バタフライ演算''', language="python")

    with st.expander("FFT はどのように生まれ, 広まったか", expanded=False):
        st.markdown(
            "- **ガウス(1805年ごろ)**:小惑星の軌道を, 観測値から三角関数の和で補間する計算の中で, FFT と同じ考え方の方法を使っていた。"
            "しかし死後の全集(1866年)に, ラテン語で載っただけで, 長く知られなかった(ハイデマン, ジョンソン, バーラス, 1985)。\n"
            "- **20世紀前半**:何人もの研究者が, 同じような計算の工夫を独立に見つけていたが, 手計算の時代には広まらなかった。\n"
            "- **クーリーとテューキー(1965)**:1963年, ケネディ大統領の科学諮問委員会の会合で, 地下核実験を地震計の記録から見つける方法が話し合われた。"
            "そこでテューキーが考えていた方法に, IBM のガーウィンが注目し, IBM のクーリーがプログラムにした。"
            "1964年に IBM の計算機(IBM 7094)で実演され, 翌年に論文として発表された。\n"
            "- **広まった理由**:ちょうど計算機が使えるようになった時期で, 計算時間が何百倍・何万倍も短くなったため, "
            "スペクトル解析・音声や画像の処理・通信・CT や MRI の画像再構成などの**ディジタル信号処理**が実用になった。")
    with st.expander("練習問題"):
        st.markdown("**問1** $x=(1,\\ 2,\\ 3,\\ 4)$ の DFT を, 偶数番目 $(1,3)$ と奇数番目 $(2,4)$ に分けて, バタフライ演算で求めよ。")
        st.markdown("**問2** $N=2^{12}=4096$ のとき, 直接計算と FFT の掛け算の回数を比べよ。")
        if st.toggle("解答・解説を表示", key="pr_fft"):
            st.markdown("**答1** 長さ 2 の DFT は $(a,b)\\mapsto(a+b,\\ a-b)$ なので $E=(4,\\ -2)$, $O=(6,\\ -2)$。$W_4=-i$ より")
            st.latex(r"X_0=E_0+O_0=10,\ X_1=E_1+(-i)O_1=-2+2i,\ X_2=E_0-O_0=-2,\ X_3=E_1-(-i)O_1=-2-2i")
            st.markdown("**答2** 直接計算 $4096^2=16{,}777{,}216$ 回, FFT $2048\\times12=24{,}576$ 回(約 683 倍)。")
    page_refs([("cooleytukey1965", "FFT の論文。"), ("heideman1985", "ガウスと FFT の歴史。"),
               ("ieee_fft", "1964年の IBM での実演と, 核実験の検出という動機。"), ("oppenheim1997", "FFT と信号処理。")])


# ============================================================
# 5. シャノンの標本化定理
# ============================================================
def tone_wav(f, fs=44100, dur=1.5):
    t = np.arange(int(fs * dur)) / fs
    y = 0.6 * np.sin(2 * PI * f * t)
    ramp = int(0.01 * fs)
    y[:ramp] *= np.linspace(0, 1, ramp)
    y[-ramp:] *= np.linspace(1, 0, ramp)
    buf = io.BytesIO()
    wavfile.write(buf, fs, (y * 32767).astype(np.int16))
    return buf.getvalue()


def alias_freq(f, fs):
    return abs(f - fs * round(f / fs))


def tab_sampling():
    st.info("信号に含まれる最大周波数が $f_{\\max}$ のとき, サンプリング周波数 $f_s>2f_{\\max}$ なら, "
            "サンプル値から元の信号を完全に復元できる(シャノンの標本化定理)。")
    st.subheader("定理")
    st.markdown("$x(t)$ のフーリエ変換 $X(f)=\\int x(t)e^{-2\\pi ift}dt$ が $|f|>B$ で $0$(帯域制限)ならば, $f_s=\\dfrac1{T_s}\\ge2B$ のサンプル値 $x(nT_s)$ から")
    st.latex(r"x(t)=\sum_{n=-\infty}^{\infty}x(nT_s)\,\mathrm{sinc}\!\left(\frac{t-nT_s}{T_s}\right),\quad "
             r"T_s=\frac{1}{f_s},\quad \mathrm{sinc}\,u=\frac{\sin\pi u}{\pi u}")
    st.markdown("と完全に復元できる。$\\dfrac{f_s}{2}$ を**ナイキスト周波数**という。$f_s/2$ より高い成分は, 低い周波数の成分と区別できなくなる(**エイリアシング**, 折り返し)。")
    with st.expander("定理の証明(スペクトルの周期化)"):
        st.markdown("**1.** サンプル値でつくった列のスペクトルは, もとのスペクトルを $f_s$ ごとにずらして足したものになる(ポアソンの和公式):")
        st.latex(r"T_s\sum_{n=-\infty}^{\infty}x(nT_s)e^{-2\pi ifnT_s}=\sum_{m=-\infty}^{\infty}X(f-mf_s)")
        st.markdown("左辺は $f$ について周期 $f_s$ の関数のフーリエ級数で, その係数がサンプル値 $x(nT_s)$。")
        st.markdown("**2.** $f_s\\ge2B$ なら, ずらしたもの $X(f-mf_s)$ どうしは重ならない。$|f|<\\frac{f_s}{2}$ で右辺は $X(f)$ だけになるので")
        st.latex(r"X(f)=T_s\sum_nx(nT_s)e^{-2\pi ifnT_s}\quad\left(|f|<\tfrac{f_s}{2}\right),\qquad X(f)=0\quad\left(|f|>\tfrac{f_s}{2}\right)")
        st.markdown("**3.** 逆フーリエ変換すると, $\\int_{-f_s/2}^{f_s/2}e^{2\\pi if(t-nT_s)}df=\\dfrac{\\sin\\pi f_s(t-nT_s)}{\\pi(t-nT_s)}$ より")
        st.latex(r"x(t)=T_s\sum_nx(nT_s)\frac{\sin\pi f_s(t-nT_s)}{\pi(t-nT_s)}=\sum_nx(nT_s)\,\mathrm{sinc}\frac{t-nT_s}{T_s}")
        st.caption("$f_s<2B$ だと, ずらしたスペクトルが重なり(折り返し), もとの $X(f)$ を取り出せない。これがエイリアシング。")

    st.subheader("動かして確かめる:sinc 補間による復元")
    c = st.columns(3)
    f1 = c[0].slider("f₁ [Hz]", 0.5, 8.0, 1.0, 0.5, key="sa_f1")
    f2 = c[1].slider("f₂ [Hz]", 0.5, 8.0, 3.0, 0.5, key="sa_f2")
    fs = c[2].slider("サンプリング周波数 f_s [Hz]", 1.0, 30.0, 8.0, 0.5, key="sa_fs")
    x = lambda t: np.sin(2 * PI * f1 * t) + 0.6 * np.cos(2 * PI * f2 * t + 0.5)  # noqa: E731
    fmax = max(f1, f2)
    Ts = 1 / fs
    nn = np.arange(int(-20 * fs), int(24 * fs) + 1)
    tn = nn * Ts
    t = np.linspace(0, 4, 2001)
    xr = np.sinc((t[:, None] - tn[None, :]) / Ts) @ x(tn)
    inner = (t > 1) & (t < 3)
    err = float(np.max(np.abs(xr - x(t))[inner]))
    if fs > 2 * fmax:
        st.success(f"f_s = {fs:g} > 2 f_max = {2 * fmax:g}:復元できる(区間 1 < t < 3 での最大誤差 {err:.3f}、"
                   "有限個のサンプルで打ち切った誤差を含む)")
    else:
        al = [alias_freq(f, fs) for f in (f1, f2)]
        st.error(f"f_s = {fs:g} ≤ 2 f_max = {2 * fmax:g}:エイリアシングが起こり、"
                 f"{f1:g} Hz → {al[0]:g} Hz、{f2:g} Hz → {al[1]:g} Hz の信号と区別できない(最大誤差 {err:.3f})。")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=x(t), mode="lines", line=dict(color=GRAY, width=2), name="元の信号 x(t)"))
    fig.add_trace(go.Scatter(x=t, y=xr, mode="lines", line=dict(color=C2, width=2, dash="dash"), name="sinc 補間で復元"))
    sel = (tn >= 0) & (tn <= 4)
    fig.add_trace(go.Scatter(x=tn[sel], y=x(tn[sel]), mode="markers", marker=dict(color=C1, size=8), name="サンプル"))
    layout(fig, xlab="t [s]", height=380)
    show(fig)

    fig = go.Figure()
    fig.add_vrect(x0=-fs / 2, x1=fs / 2, fillcolor="rgba(44,160,44,0.12)", line_width=0)
    for k in range(-3, 4):
        for f, a, col in ((f1, 1.0, C1), (f2, 0.6, C4)):
            for sg in (1, -1):
                xx = sg * f + k * fs
                fig.add_trace(go.Scatter(x=[xx, xx], y=[0, a / 2], mode="lines",
                                         line=dict(color=col, width=3 if k == 0 else 1.5, dash="solid" if k == 0 else "dot"),
                                         showlegend=False))
    layout(fig, title="サンプリング後のスペクトル:元のスペクトルが f_s ごとに繰り返す(緑:|f| < f_s/2)",
           xlab="周波数 [Hz]", height=280, xrange=(-2.5 * fs, 2.5 * fs), legend=False)
    show(fig)
    st.caption("実線が元の成分、点線がサンプリングで生じる複製。緑の帯の中で複製が元の成分と重ならなければ、"
               "低域フィルタ(sinc 補間)で元に戻せる。")

    st.subheader("耳で確かめる:エイリアシング")
    c = st.columns(2)
    ft = c[0].slider("もとの音の周波数 [Hz]", 200, 12000, 5000, 100, key="sa_ft")
    fsa = c[1].select_slider("記録するサンプリング周波数 [Hz]", [4000, 8000, 16000, 44100], value=8000, key="sa_fsa")
    fa = alias_freq(ft, fsa)
    cA, cB = st.columns(2)
    with cA:
        st.markdown(f"もとの音({ft} Hz)")
        st.audio(tone_wav(ft), format="audio/wav")
    with cB:
        st.markdown(f"$f_s={fsa}$ Hz で記録して再生した音({fa:g} Hz)")
        st.audio(tone_wav(fa), format="audio/wav")
    if fa == 0:
        st.warning(f"{ft} Hz はサンプリング周波数 {fsa} Hz の整数倍なので, サンプル値がすべて同じ値になり, 音にならない(0 Hz に折り返す)。")
    elif ft > fsa / 2:
        st.warning(f"{ft} Hz はナイキスト周波数 {fsa / 2:g} Hz を超えるので, |{ft} − {fsa} × {round(ft / fsa)}| = {fa:g} Hz の音として記録される。")
    else:
        st.success(f"{ft} Hz はナイキスト周波数 {fsa / 2:g} Hz より低いので, そのまま記録できる。")
    st.caption("再生音は, 標本化定理どおり(sinc 補間)に復元した音を 44100 Hz で鳴らしている。"
               "電話(8000 Hz)で高い音がこもって聞こえるのは, 4000 Hz より上をあらかじめフィルタで消しているため(消さないと折り返しの雑音になる)。")

    with st.expander("標本化定理はどのように生まれたか"):
        st.markdown(
            "- **ホイッタカー(1915年)**:サンプル値を通る関数の中で最も「なめらかな」ものとして, sinc 関数の和(基数関数)を研究した。\n"
            "- **ナイキスト(1928年)**:電信で, 帯域幅 $B$ の回線に1秒あたり $2B$ 個の独立なパルスを送れることを示した。\n"
            "- **コテルニコフ(1933年)**:通信の立場から, 標本化と復元の定理を示した。\n"
            "- **シャノン(1949年, 書かれたのは 1940年)**:情報理論の中で定理をはっきり述べ, 証明した。\n"
            "- **染谷勲(1949年)**:著書『波形伝送』で, シャノンとは独立に同じ定理を示した。そのため日本では「シャノン=染谷の標本化定理」とも呼ばれる。")
    with st.expander("練習問題"):
        st.markdown("**問1** 音楽 CD のサンプリング周波数は 44.1 kHz。記録できる最大の周波数は?")
        st.markdown("**問2** 電話のサンプリング周波数は 8 kHz。7 kHz の音をフィルタを通さずにサンプリングすると, 何 Hz の音として記録されるか。")
        st.markdown("**問3** 最大周波数 50 Hz の脳波を記録するには, サンプリング周波数を何 Hz より大きくすればよいか。")
        if st.toggle("解答・解説を表示", key="pr_other_169"):
            st.markdown("**答1** ナイキスト周波数 22.05 kHz(人の可聴域 約 20 kHz をカバーする)。")
            st.markdown("**答2** $|7-8|=1$ kHz。")
            st.markdown("**答3** $2\\times50=100$ Hz より大きく。")
    page_refs([("shannon1949", "標本化定理(1940年に投稿, 1949年に掲載)。"), ("someya1949", "シャノンと独立に示した標本化定理。"),
               ("yoshino2015", "シャノン=染谷の標本化定理と複素解析的な方法(日本語)。"),
               ("unser2000", "標本化定理のその後の発展(スプライン・ウェーブレットとの関係)。"),
               ("oppenheim1997", "サンプリングとエイリアシング。")])
