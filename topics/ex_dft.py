"""数学IV フーリエ解析 第3章 離散フーリエ変換:演習の問題生成

規約:X_k = Σ_{n=0}^{N−1} x_n W^{kn}, W = e^{−2πi/N},  x_n = (1/N) Σ_k X_k W^{−kn}。
N = 4 では W = −i なので, 整数のデータの DFT はガウス整数(a + bi, a, b は整数)になる。
"""
import math

import numpy as np

from topics.ex_core import Choice, ExamItem, Part, Problem, ProblemType, Retry, render

DEF = (r"$X_k=\displaystyle\sum_{n=0}^{N-1}x_nW^{kn}$, $W=e^{-2\pi i/N}$, "
       r"$x_n=\dfrac1N\displaystyle\sum_{k=0}^{N-1}X_kW^{-kn}$。$N=4$ では $W=-i$。")


def gi(z):
    """ガウス整数の TeX"""
    a, b = int(round(z.real)), int(round(z.imag))
    if b == 0:
        return str(a)
    bs = ("" if abs(b) == 1 else str(abs(b))) + "i"
    if a == 0:
        return ("-" if b < 0 else "") + bs
    return f"{a}{'-' if b < 0 else '+'}{bs}"


def vt(v):
    return "(" + ",\\ ".join(gi(complex(z)) for z in v) + ")"


def dft4(x):
    return np.array([sum(x[n] * (-1j) ** (k * n) for n in range(4)) for k in range(4)])


def distinct(ok, cands):
    choices = [Choice(ok, True, value=ok)]
    for t, why in cands:
        if all(c.tex != t for c in choices) and len(choices) < 4:
            choices.append(Choice(t, why=why, value=t))
    if len(choices) < 4:
        raise Retry
    return choices


def rand_x(rng, lo=-2, hi=4):
    while True:
        x = [rng.randint(lo, hi) for _ in range(4)]
        X = dft4(x)
        if len(set(x)) >= 3 and abs(X[1].imag) > 0.5 and abs(X[1].real) > 0.5:
            return x, X


def dft_solution(x, X):
    k_lines = []
    for k in range(4):
        w = [(-1j) ** (k * n) for n in range(4)]
        terms = "+".join(f"{x[n]}\\cdot({gi(w[n])})" for n in range(4)).replace("+-", "-")
        k_lines.append(rf"$X_{k}={terms}={gi(X[k])}$")
    return [r"$N=4$ では $W^{kn}=(-i)^{kn}$ は $1,\ -i,\ -1,\ i$ をくり返す。"] + k_lines


# ------------------------------------------------------------
# 基本:N = 4 の DFT
# ------------------------------------------------------------
def g_dft4(rng):
    x, X = rand_x(rng)
    ok = vt(X)
    cands = [(vt(np.conj(X)), r"$W=e^{-2\pi i/4}=-i$。$+i$ を使うと逆変換(の $N$ 倍)になる。"),
             (vt([X[0], X[3], X[2], X[1]]), r"$X_1$ は $W^{n}=(-i)^n$, $X_3$ は $W^{3n}=i^n$ をかける。"),
             (vt([X[0], -X[1], X[2], -X[3]]), r"$W^{kn}$ の符号を確かめよう:$(-i)^2=-1$, $(-i)^3=i$。"),
             (vt([X[0], 1j * X[1], X[2], -1j * X[3]]), r"$(-i)^{kn}$ の $kn$ を確かめよう($k$ と $n$ の積)。")]
    if all(abs((v / 4).real - round((v / 4).real)) < 1e-9 and abs((v / 4).imag - round((v / 4).imag)) < 1e-9 for v in X):
        cands.insert(1, (vt(X / 4), r"DFT には $\frac1N$ をつけない($\frac1N$ は逆変換につく)。"))
    return Problem("N = 4 の離散フーリエ変換", "基本", rf"データ $x={vt(x)}$ の離散フーリエ変換 $X=(X_0,X_1,X_2,X_3)$ を求めよ。",
                   distinct(ok, cands), hints=["定義:" + DEF, r"$X_0$ はデータの和, $X_2$ は符号を交互に変えた和。"],
                   solution=dft_solution(x, X), meta=dict(x=x, X=X))


# ------------------------------------------------------------
# 基本:N = 4 の逆 DFT
# ------------------------------------------------------------
def g_idft4(rng):
    x, X = rand_x(rng)
    ok = vt(x)
    cands = [(vt([4 * v for v in x]), r"逆変換には $\frac1N=\frac14$ がつく。"),
             (vt([x[0], x[3], x[2], x[1]]), r"逆変換では $W^{-kn}=i^{kn}$ をかける($W^{kn}$ のままだと $x_{-n}$ になる)。"),
             (vt([x[1], x[2], x[3], x[0]]), r"$x_0=\frac14(X_0+X_1+X_2+X_3)$ から順に求める。")]
    sol = [r"$x_n=\dfrac14\displaystyle\sum_kX_k\,i^{kn}$($W^{-1}=i$)。"]
    for n in range(4):
        terms = "+".join(f"({gi(X[k])})\\cdot({gi(1j ** (k * n))})" for k in range(4))
        sol.append(rf"$x_{n}=\frac14\{{{terms}\}}={x[n]}$")
    return Problem("N = 4 の逆離散フーリエ変換", "基本", rf"$N=4$ の離散フーリエ変換が $X={vt(X)}$ であった。もとのデータ $x=(x_0,x_1,x_2,x_3)$ を求めよ。",
                   distinct(ok, cands), hints=["定義:" + DEF, r"$X_0=4\times(\text{データの平均})$ になっていることも確かめに使える。"],
                   solution=sol, meta=dict(x=x, X=X))


# ------------------------------------------------------------
# 標準:パーセバルの等式・巡回シフト
# ------------------------------------------------------------
def g_parseval(rng):
    x, X = rand_x(rng)
    s = sum(v * v for v in x)
    ok = str(4 * s)
    cands = [(str(s), r"パーセバルの等式は $\sum|x_n|^2=\frac1N\sum|X_k|^2$。$\sum|X_k|^2$ は $N$ 倍になる。"),
             (str(16 * s), r"$\frac1N$ を2乗していないか。$\sum|X_k|^2=N\sum|x_n|^2$。"),
             (str(sum(x) ** 2), r"$|X_0|^2$ だけでなく, すべての $k$ について足す。")]
    if str(s * 2) not in (ok, str(s), str(16 * s)):
        cands.append((str(2 * s), r"$N=4$ を使おう。"))
    return Problem("パーセバルの等式(DFT)", "標準",
                   rf"データ $x={vt(x)}$ の離散フーリエ変換を $X_k$ とする。$\displaystyle\sum_{{k=0}}^{{3}}|X_k|^2$ を求めよ。",
                   distinct(ok, cands), hints=[r"パーセバルの等式 $\displaystyle\sum_n|x_n|^2=\frac1N\sum_k|X_k|^2$。", "DFT を計算して直接確かめてもよい。"],
                   solution=[rf"$\sum|x_n|^2={'+'.join(str(v * v) for v in x)}={s}$ なので, パーセバルの等式より $\sum|X_k|^2=4\times{s}={4 * s}$。",
                             "(確かめ:$X=" + vt(X) + "$ で $" + "+".join(str(int(round(abs(v) ** 2))) for v in X) + f"={4 * s}$)"],
                   meta=dict(x=x, val=4 * s))


def g_shift(rng):
    x, X = rand_x(rng)
    y = [x[3], x[0], x[1], x[2]]                # y_n = x_{n−1}(巡回)
    Y = np.array([(-1j) ** k * X[k] for k in range(4)])
    ok = vt(Y)
    cands = [(vt([1j ** k * X[k] for k in range(4)]), r"$x_{n-1}$ の DFT は $W^{k}X_k$($W=-i$)。$i^k$ は逆向きのシフト $x_{n+1}$。"),
             (vt(X), "シフトで変わらないのは $|X_k|$ だけ。偏角は変わる。"),
             (vt([X[0], X[3], X[2], X[1]]), r"巡回シフトでは $X_k$ に $W^k$ をかける($k$ の順番は変わらない)。")]
    return Problem("巡回シフトと DFT", "標準",
                   rf"データ $x={vt(x)}$ の DFT は $X={vt(X)}$ である。1つずつ右へ巡回シフトしたデータ $y={vt(y)}$($y_n=x_{{n-1}}$)の DFT を求めよ。",
                   distinct(ok, cands), hints=[r"$y_n=x_{n-1}$ なら $Y_k=W^kX_k$。$N=4$ では $W^k=(-i)^k$。", "直接 DFT を計算してもよい。"],
                   solution=[r"$Y_k=W^kX_k=(-i)^kX_k$ より", rf"$$Y={vt(Y)}$$"], meta=dict(y=y, Y=Y))


# ------------------------------------------------------------
# 基本:エイリアシング・周波数分解能
# ------------------------------------------------------------
def g_alias(rng):
    while True:
        fs = rng.choice([100, 200, 500, 1000, 8000])
        f = rng.randint(fs // 2 + 1, 2 * fs - 1)
        f = int(round(f / (fs // 20))) * (fs // 20)
        fa = abs(f - fs * round(f / fs))
        if fs / 2 < f < 2 * fs and 0 < fa < fs / 2 and f % fs:
            break
    ok = f"{fa}\\ \\mathrm{{Hz}}"
    cands = [(f"{f - fs if f > fs else fs - f}\\ \\mathrm{{Hz}}" if (f - fs if f > fs else fs - f) != fa else f"{fs // 2}\\ \\mathrm{{Hz}}",
              r"見える周波数は $|f-mf_s|$ のうち $\frac{f_s}{2}$ 以下のもの。"),
             (f"{f}\\ \\mathrm{{Hz}}", r"$f>\frac{f_s}{2}$ なので, そのままの周波数には見えない(エイリアシング)。"),
             (f"{fs // 2}\\ \\mathrm{{Hz}}", "ナイキスト周波数そのものではない。"),
             (f"{abs(f - fs // 2)}\\ \\mathrm{{Hz}}", r"$f_s$ の整数倍を引く($\frac{f_s}{2}$ ではない)。")]
    return Problem("エイリアシング(見かけの周波数)", "基本",
                   rf"周波数 ${f}\ \mathrm{{Hz}}$ の正弦波を, サンプリング周波数 ${fs}\ \mathrm{{Hz}}$ でサンプリングした。サンプル列は何 Hz の正弦波に見えるか。",
                   distinct(ok, cands), hints=[r"ナイキスト周波数は $\frac{f_s}{2}$。", r"$t_n=\frac{n}{f_s}$ で $\cos2\pi ft_n=\cos2\pi(f-mf_s)t_n$。"],
                   solution=[rf"ナイキスト周波数は ${fs // 2}\ \mathrm{{Hz}}<{f}\ \mathrm{{Hz}}$ なので, エイリアシングが起こる。",
                             rf"$|{f}-{fs}\times{round(f / fs)}|={fa}\ \mathrm{{Hz}}\ (\le{fs // 2}\ \mathrm{{Hz}})$ の正弦波と区別できない。"],
                   meta=dict(f=f, fs=fs, fa=fa))


def g_bin(rng):
    fs = rng.choice([200, 500, 1000, 2000])
    T = rng.choice([1, 2, 4])
    N = fs * T
    k = rng.randint(3, N // 4)
    f = k * fs / N
    ftex = lambda v: (str(int(v)) if float(v).is_integer() else f"{v:g}")  # noqa: E731
    ok = rf"\Delta f={ftex(fs / N)}\ \mathrm{{Hz}},\ \ {ftex(f)}\ \mathrm{{Hz}}"
    cands = [(rf"\Delta f={ftex(N / fs)}\ \mathrm{{Hz}},\ \ {ftex(k * N / fs)}\ \mathrm{{Hz}}", r"周波数の刻みは $\frac{f_s}{N}=\frac{1}{\text{観測時間}}$。"),
             (rf"\Delta f={ftex(fs / N)}\ \mathrm{{Hz}},\ \ {k}\ \mathrm{{Hz}}", r"$X_k$ の周波数は $k\Delta f$($k$ そのものではない)。"),
             (rf"\Delta f={ftex(fs / (2 * N))}\ \mathrm{{Hz}},\ \ {ftex(f / 2)}\ \mathrm{{Hz}}", r"刻みは $\frac{f_s}{N}$(ナイキスト周波数と混同しない)。"),
             (rf"\Delta f={ftex(fs / N)}\ \mathrm{{Hz}},\ \ {ftex(f * 2)}\ \mathrm{{Hz}}", "計算を確かめよう。")]
    return Problem("DFT の周波数の刻み", "基本",
                   rf"サンプリング周波数 ${fs}\ \mathrm{{Hz}}$ で ${T}$ 秒間記録したデータ($N={N}$ 点)を DFT した。"
                   rf"周波数の刻み $\Delta f$ と, $X_{{{k}}}$ に対応する周波数を求めよ。",
                   distinct(ok, cands), hints=[r"$X_k$ は1回の観測時間 $T=\frac{N}{f_s}$ の間に $k$ 回振動する成分。", r"$\Delta f=\frac{f_s}{N}=\frac1T$。"],
                   solution=[rf"$\Delta f=\dfrac{{f_s}}{{N}}=\dfrac{{{fs}}}{{{N}}}={ftex(fs / N)}\ \mathrm{{Hz}}$。",
                             rf"$X_{{{k}}}$ は ${k}\times{ftex(fs / N)}={ftex(f)}\ \mathrm{{Hz}}$ の成分。"],
                   meta=dict(fs=fs, N=N, k=k, f=f))


def num(v):
    """3桁ごとの区切り(TeX)"""
    return f"{v:,}".replace(",", "{,}")


def g_count(rng):
    L = rng.choice([8, 9, 10, 11, 12])
    N = 2 ** L
    d, ff = N * N, N // 2 * L
    ans = lambda a, b: rf"{num(a)}\ \text{{回と}}\ {num(b)}\ \text{{回}}"  # noqa: E731
    ok = ans(d, ff)
    cands = [(ans(d, N * L), r"1段のバタフライ演算で掛け算は $\frac N2$ 回(1回で2つの値が求まる)。"),
             (ans(2 * N, ff), r"直接計算は $N$ 個の $X_k$ のそれぞれに $N$ 回:$N^2$ 回。"),
             (ans(d, N // 2 * (L - 1)), r"段の数は $\log_2N$。"),
             (ans(d, L), r"各段で $\frac N2$ 回の掛け算がある。")]
    return Problem("FFT の計算回数", "基本",
                   rf"$N=2^{{{L}}}={num(N)}$ 点の DFT を, 定義どおり直接計算するときと, FFT で計算するときの, 複素数の掛け算の回数を求めよ。",
                   distinct(ok, cands), hints=[r"直接計算:$N^2$ 回。", r"FFT:$\log_2N$ 段のそれぞれで $\frac N2$ 回(バタフライ演算)。"],
                   solution=[rf"直接計算は $N^2={num(N)}^2={num(d)}$ 回。",
                             rf"FFT は $\frac N2\log_2N={num(N // 2)}\times{L}={num(ff)}$ 回(約 ${d / ff:.0f}$ 倍速い)。"],
                   meta=dict(N=N, L=L))


# ------------------------------------------------------------
# 誘導:DFT → 巡回たたみこみ
# ------------------------------------------------------------
def guided_conv(rng):
    x, X = rand_x(rng)
    h = [1, 1, 0, 0]
    H = dft4(h)
    y = [x[n] + x[n - 1] for n in range(4)]          # y_n = x_n + x_{n−1}(巡回)
    Y = X * H
    return ExamItem("離散フーリエ変換と巡回たたみこみ",
                    rf"$N=4$ の離散フーリエ変換を $X_k=\displaystyle\sum_{{n=0}}^{{3}}x_n(-i)^{{kn}}$ とする. データ $x={vt(x)}$ と $h=(1,\ 1,\ 0,\ 0)$ について, 次の問いに答えよ.",
                    [Part(r"$x$ の離散フーリエ変換 $X$ を求めよ.", vt(X), dft_solution(x, X)),
                     Part(r"巡回たたみこみ $y_n=\displaystyle\sum_{j=0}^{3}x_jh_{n-j}$(添字は $4$ を法とする)の離散フーリエ変換 $Y$ と, $y$ を求めよ.",
                          rf"Y={vt(Y)},\ \ y={vt(y)}",
                          [rf"$h$ の DFT は $H_k=1+(-i)^k$ より $H={vt(H)}$。たたみこみの DFT は積なので $Y_k=X_kH_k$:$$Y={vt(Y)}$$",
                           rf"$h=(1,1,0,0)$ なので $y_n=x_n+x_{{n-1}}$。よって $y={vt(y)}$(逆 DFT で求めても同じ)。"])],
                    meta=dict(x=x, X=X, Y=Y, y=y))


def guided_alias(rng):
    while True:
        try:
            p = g_alias(rng).meta
            break
        except Retry:
            continue
    f, fs, fa = p["f"], p["fs"], p["fa"]
    return ExamItem("標本化定理とエイリアシング",
                    rf"周波数 ${f}\ \mathrm{{Hz}}$ の正弦波を, サンプリング周波数 $f_s$ でサンプリングする. 次の問いに答えよ.",
                    [Part(rf"$f_s={fs}\ \mathrm{{Hz}}$ のとき, サンプル列は何 Hz の正弦波に見えるか.", f"{fa}\\ \\mathrm{{Hz}}",
                          [rf"ナイキスト周波数 ${fs // 2}\ \mathrm{{Hz}}$ を超えるので, $|{f}-{fs}\times{round(f / fs)}|={fa}\ \mathrm{{Hz}}$ に見える。"]),
                     Part(r"もとの正弦波を正しく復元できるためには, $f_s$ を何 Hz より大きくすればよいか. 理由も述べよ.", f"{2 * f}\\ \\mathrm{{Hz}}",
                          [rf"標本化定理より, 信号の最大周波数 ${f}\ \mathrm{{Hz}}$ の2倍 ${2 * f}\ \mathrm{{Hz}}$ より大きくすればよい"
                           r"(このときサンプリングで生じるスペクトルの複製が, もとのスペクトルと重ならない)。"])],
                    meta=dict(f=f, fs=fs, fa=fa))


REGISTRY = [
    ProblemType("df_dft4", "N = 4 の離散フーリエ変換", "基本", g_dft4, guided=guided_conv),
    ProblemType("df_idft4", "N = 4 の逆離散フーリエ変換", "基本", g_idft4),
    ProblemType("df_alias", "エイリアシング(見かけの周波数)", "基本", g_alias, guided=guided_alias),
    ProblemType("df_bin", "DFT の周波数の刻み", "基本", g_bin),
    ProblemType("df_count", "FFT の計算回数", "基本", g_count),
    ProblemType("df_pars", "パーセバルの等式(DFT)", "標準", g_parseval),
    ProblemType("df_shift", "巡回シフトと DFT", "標準", g_shift),
]
GUIDED_NAMES = {"df_dft4": "N = 4 の DFT → 巡回たたみこみ(DFT の積)",
                "df_alias": "エイリアシングの周波数 → 復元に必要なサンプリング周波数"}


def tab():
    render("dft", REGISTRY, "フーリエ解析 第3章 離散フーリエ変換")
