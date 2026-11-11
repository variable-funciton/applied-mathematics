"""⑤ 偏微分方程式 発展5. ニュートンポテンシャルとポアソン方程式

記号(3次元):Nf(x) = ∫ f(y) / (4π|x − y|) dy,  −Δ(Nf) = f。基本解 Γ(x) = 1/(4π|x|)。
球対称な f(r) のとき  u(r) = (1/r) ∫_0^r f(s) s² ds + ∫_r^∞ f(s) s ds,  −u'(r) = (1/r²) ∫_0^r f(s) s² ds。
一様な球(半径 a, 密度 ρ₀):u = ρ₀(3a² − r²)/6 (r ≤ a),  ρ₀a³/(3r) (r ≥ a)。
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import C1, C2, C3, C4, GRAY, layout, show
from topics.fourier_apps import refs
from topics.fourier_kernels import page_refs

PI = np.pi
PROFILES = {"一様な球": "ball", "球殻(中が空洞)": "shell", "2層の球(中心ほど重い)": "two"}


def density(kind, a, r):
    """球対称な密度 f(r)"""
    r = np.asarray(r, dtype=float)
    if kind == "ball":
        return np.where(r <= a, 1.0, 0.0)
    if kind == "shell":
        return np.where((r >= 0.6 * a) & (r <= a), 1.0, 0.0)
    return np.where(r <= 0.5 * a, 3.0, np.where(r <= a, 1.0, 0.0))


def radial_potential(kind, a, r, n=4000):
    """球対称な密度のニュートンポテンシャル u(r) と, 力の大きさ −u'(r)(数値積分)"""
    s = np.linspace(0, max(a, 1e-9), n)
    f = density(kind, a, s)
    I2 = np.concatenate([[0], np.cumsum((f[1:] * s[1:] ** 2 + f[:-1] * s[:-1] ** 2) / 2 * np.diff(s))])   # ∫_0^s f t² dt
    I1 = np.concatenate([[0], np.cumsum((f[1:] * s[1:] + f[:-1] * s[:-1]) / 2 * np.diff(s))])             # ∫_0^s f t dt
    r = np.asarray(r, dtype=float)
    rc = np.minimum(r, a)
    m = np.interp(rc, s, I2)                 # 半径 r の内側の「質量」/(4π)
    outer = I1[-1] - np.interp(rc, s, I1)    # ∫_r^a f t dt
    rr = np.maximum(r, 1e-12)
    return m / rr + outer, m / rr ** 2, I2[-1]


def ball_potential(r, a, rho0=1.0):
    r = np.asarray(r, dtype=float)
    return np.where(r <= a, rho0 * (3 * a * a - r * r) / 6, rho0 * a ** 3 / (3 * np.maximum(r, 1e-12)))


def fund(n, r):
    """−Δ の基本解(n = 1, 2, 3)"""
    r = np.asarray(r, dtype=float)
    if n == 1:
        return -r / 2
    if n == 2:
        return np.log(1 / r) / (2 * PI)
    return 1 / (4 * PI * r)


def tab_newton_potential():
    st.info("地球の重力も, 電荷のまわりの電場も, 「点が作る $\\dfrac1r$ のポテンシャルを足し合わせる」ことで求められる。"
            "この足し合わせ(積分)を**ニュートンポテンシャル**といい, **ポアソン方程式の解を与える公式**になっている。"
            "① ベクトル解析のガウスの発散定理, ラプラス方程式, ③ のフーリエマルチプライヤー(リースポテンシャル)を1本につなぐ題材である。")

    st.subheader("1. 点から分布へ:足し合わせると積分になる")
    st.markdown("原点に置いた質量 $M$ の点が作る重力のポテンシャルは $-\\dfrac{GM}{r}$, 電荷 $Q$ の点が作る電位は $\\dfrac{Q}{4\\pi\\varepsilon_0r}$。"
                "どちらも**距離に反比例**する。点がたくさんあれば足せばよく, 連続的に分布していれば(密度 $f(y)$)和が積分になる:")
    st.latex(r"u(x)=\sum_k\frac{m_k}{4\pi|x-y_k|}\ \longrightarrow\ u(x)=\int_{\mathbb R^3}\frac{f(y)}{4\pi|x-y|}\,dy=:Nf(x)\qquad(\text{ニュートンポテンシャル})")
    st.caption("定数 $\\frac{1}{4\\pi}$ は, あとで $-\\Delta u=f$ がきれいな形になるように付けた(重力では $-4\\pi G$, 静電気では $\\frac{1}{\\varepsilon_0}$ を掛ける)。")

    st.subheader("2. ガウスの法則からポアソン方程式へ")
    st.markdown("$\\Gamma(x)=\\dfrac{1}{4\\pi|x|}$ とおく。$r=|x|$ だけの関数のラプラシアンは $\\Delta u=u^{\\prime\\prime}+\\dfrac2ru^{\\prime}$ なので, 原点以外では")
    st.latex(r"\Delta\frac1r=\frac{2}{r^3}+\frac2r\cdot\left(-\frac{1}{r^2}\right)=0\qquad(r\neq0)")
    st.markdown("つまり $\\Gamma$ は原点以外で**調和関数**である。一方, 原点を中心とする半径 $R$ の球面 $S_R$ を通る $\\nabla\\Gamma$ の流束は")
    st.latex(r"\int_{S_R}\nabla\Gamma\cdot\boldsymbol n\,dS=-\frac{1}{4\pi R^2}\times4\pi R^2=-1\qquad(R\text{ によらない})")
    st.markdown("ガウスの発散定理(① ベクトル解析)より, これは球の中での $\\Delta\\Gamma$ の体積積分に等しい。"
                "「原点以外では 0 なのに, 積分すると $-1$」なので, $-\\Delta\\Gamma$ は**原点だけに集中した量 1 の湧き出し**(デルタ関数 $\\delta$)を表す。"
                "$\\Gamma$ を $-\\Delta$ の**基本解**という。分布 $f$ はこの湧き出しの重ね合わせなので")
    st.latex(r"-\Delta(Nf)=f\qquad(\text{ポアソン方程式の解}),\qquad \text{重力では}\ \Delta\phi=4\pi G\rho,\quad\text{静電気では}\ \Delta\phi=-\frac{\rho}{\varepsilon_0}")
    st.markdown("導入2. のラプラス方程式は $f=0$ の場合(湧き出しのないところ)である。ニュートンポテンシャルは, 湧き出し $f$ があるときに "
                "$-\\Delta$ の**逆**を積分で書いたもの, と言える。")

    st.subheader("3. 球のポテンシャル:ニュートンの殻の定理")
    st.markdown("密度が中心からの距離 $r$ だけで決まる(球対称な)とき, 積分を計算すると")
    st.latex(r"u(r)=\frac1r\int_0^rf(s)s^2ds+\int_r^\infty f(s)s\,ds,\qquad -u^{\prime}(r)=\frac{1}{r^2}\int_0^rf(s)s^2ds=\frac{(\text{半径 }r\text{ の内側の質量})}{4\pi r^2}")
    st.markdown("- 力 $-u^{\\prime}(r)$ は**内側の質量だけ**で決まり, 外側の殻は力を及ぼさない。\n"
                "- 球の外では, **全質量が中心の1点に集まったのと同じ**(地球を点とみなしてよい理由)。\n\n"
                "半径 $a$, 密度 $\\rho_0$ の一様な球では")
    st.latex(r"u(r)=\begin{cases}\dfrac{\rho_0}{6}\left(3a^2-r^2\right)&(r\le a)\\[2mm]\dfrac{\rho_0a^3}{3r}&(r\ge a)\end{cases},\qquad"
             r"-u^{\prime}(r)=\begin{cases}\dfrac{\rho_0}{3}r&(r\le a)\\[2mm]\dfrac{\rho_0a^3}{3r^2}&(r\ge a)\end{cases}")
    c = st.columns([1, 2])
    kind = PROFILES[c[0].radio("密度の分布", list(PROFILES), key="np_kind")]
    a = c[0].slider("球の半径 a", 0.5, 2.0, 1.0, 0.1, key="np_a")
    r = np.linspace(0.01, 4, 800)
    u, g, m = radial_potential(kind, a, r)
    c[0].metric("全質量 / 4π", f"{m:.3f}")
    with c[1]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=r, y=u, mode="lines", line=dict(color=C1, width=3), name="ポテンシャル u(r)"))
        fig.add_trace(go.Scatter(x=r, y=m / r, mode="lines", line=dict(color=C1, width=1.5, dash="dot"), name="全質量を中心に集めた点のポテンシャル"))
        fig.add_trace(go.Scatter(x=r, y=g, mode="lines", line=dict(color=C3, width=3), name="力の大きさ −u′(r)"))
        fig.add_trace(go.Scatter(x=r, y=m / r ** 2, mode="lines", line=dict(color=C3, width=1.5, dash="dot"), name="点の場合の力"))
        fig.add_trace(go.Scatter(x=r, y=density(kind, a, r) * 0.3, mode="lines", line=dict(color=GRAY, width=1), fill="tozeroy", name="密度 f(r)(縮小)"))
        fig.add_vline(x=a, line=dict(color=GRAY, dash="dash"))
        ymax = float(min(max(1.2 * u.max(), 1.2 * g.max()), 6))
        layout(fig, title="球の内外のポテンシャルと力(破線が球の表面)", xlab="中心からの距離 r", height=400, yrange=(0, ymax))
        show(fig)
    st.caption("表面より外では, 実線と点線(点の場合)が重なる。球殻では空洞の中のポテンシャルは一定で, 力は 0 になる。"
               "一様な球の内部では力は $r$ に比例する(地球の中心を通るトンネルに落ちた物体は単振動する)。")

    st.subheader("4. 次元による違い")
    st.markdown("同じ考え方で, $-\\Delta$ の基本解は次元ごとに次の形になる:")
    st.latex(r"\text{1次元}:\ -\frac{|x|}{2},\qquad\text{2次元}:\ \frac{1}{2\pi}\log\frac{1}{|x|},\qquad\text{3次元}:\ \frac{1}{4\pi|x|},\qquad n\text{ 次元}\ (n\ge3):\ \frac{c_n}{|x|^{n-2}}")
    rr = np.linspace(0.02, 3, 500)
    fig = go.Figure()
    for n, col in ((1, C4), (2, C2), (3, C1)):
        fig.add_trace(go.Scatter(x=rr, y=fund(n, rr), mode="lines", line=dict(color=col, width=2.5), name=f"{n} 次元"))
    layout(fig, title="−Δ の基本解(原点からの距離の関数)", xlab="r = |x|", height=300, yrange=(-1.6, 1.6))
    show(fig)
    st.markdown("- **1次元**:$-u^{\\prime\\prime}=f$ の解 $u(x)=-\\dfrac12\\displaystyle\\int|x-y|f(y)\\,dy$。折れ線 $-\\frac{|x|}{2}$ の折れ目が, 点に集中した湧き出しを表す。\n"
                "- **2次元**:$\\log$ が現れる(無限に長い直線状の電荷のまわりの電位。④ 複素関数の $\\log z$ の実部)。\n"
                "- **3次元以上**:$|x|^{2-n}$。遠くで 0 に近づくのは3次元以上だけである。")

    st.subheader("5. フーリエ変換で見る:リースポテンシャルとのつながり")
    st.markdown("フーリエ変換では微分が掛け算になるので, $-\\Delta$ は $|\\xi|^2$ を掛ける操作になる。したがってその逆のニュートンポテンシャルは")
    st.latex(r"\widehat{-\Delta u}(\xi)=|\xi|^2\hat u(\xi)\ \Longrightarrow\ \widehat{Nf}(\xi)=\frac{\hat f(\xi)}{|\xi|^2}")
    st.markdown("つまり**マルチプライヤー $|\\xi|^{-2}$** の作用素である(③ 発展8.)。$|\\xi|^{-\\alpha}$ に対応する**リースポテンシャル**")
    st.latex(r"I_\alpha f(x)=c_{n,\alpha}\int_{\mathbb R^n}\frac{f(y)}{|x-y|^{n-\alpha}}\,dy,\qquad c_{n,\alpha}=\frac{\Gamma\left(\frac{n-\alpha}{2}\right)}{\pi^{n/2}2^\alpha\Gamma\left(\frac\alpha2\right)}")
    st.markdown("で $n=3$, $\\alpha=2$ とすると $c_{3,2}=\\dfrac{\\Gamma(\\frac12)}{\\pi^{3/2}\\cdot4\\cdot\\Gamma(1)}=\\dfrac{1}{4\\pi}$ となり, $I_2=N$(ニュートンポテンシャル)である。"
                "ハーディ・リトルウッド・ソボレフの不等式は, $n$ 次元では $\\dfrac1q=\\dfrac1p-\\dfrac\\alpha n$ のとき $\\|I_\\alpha f\\|_q\\le C\\|f\\|_p$ となり, "
                "ニュートンポテンシャルでは $\\dfrac1q=\\dfrac1p-\\dfrac23$ である。")
    st.caption("さらに, $f\\in L^p$($1<p<\\infty$)なら $Nf$ の2階の偏導関数も $L^p$ に入る(カルデロン・ジグムントの理論)。"
               "「ポアソン方程式の解は $f$ より2階分なめらか」という事実を精密に述べたもので, 特異積分作用素の理論の出発点になった。")

    st.subheader("6. 歴史")
    st.markdown(r"""
| 年 | できごと |
|---|---|
| 1687 | ニュートン(I. Newton)が『プリンキピア』で万有引力の法則を述べ, 球対称な物体は外部に対して中心の1点と同じ引力を及ぼすこと(殻の定理)を幾何学的に証明する |
| 18世紀後半 | ラグランジュ, ルジャンドル, ラプラスが, 引力を「ある1つの関数の勾配」として扱う。ラプラスは 1780年代に, この関数が物質の外でラプラス方程式をみたすことを示す |
| 19世紀初め | ポアソン(S. D. Poisson)が, 物質の内部では右辺に密度が現れること(ポアソン方程式)を示す |
| 1828 | グリーン(G. Green)が電気と磁気の論文でこの関数を「ポテンシャル」と名づけ, グリーンの公式とグリーン関数の考え方を導入 |
| 19世紀前半 | ガウス(C. F. Gauss)が, 距離の2乗に反比例する力の一般的な定理をまとめ, ポテンシャル論の基礎を築く |
| 20世紀 | M. リースがポテンシャルを分数階に一般化(リースポテンシャル)。ソボレフ, カルデロン, ジグムントらにより, 関数空間の上での作用素の理論へ発展する |
""")
    with st.expander("練習問題"):
        st.markdown("**問1** $r=|x|$(3次元)の関数 $u=r^2$ と $u=\\dfrac1r$ のラプラシアン $\\Delta u=u^{\\prime\\prime}+\\dfrac2ru^{\\prime}$ を求めよ。")
        st.markdown("**問2** 半径 $a=1$, 密度 $\\rho_0=6$ の一様な球について, $-\\Delta u=f$ の解 $u$ の中心での値と, $r=2$ での値を求めよ。")
        st.markdown("**問3** 1次元で $f=\\chi_{[-1,1]}$ のとき, $u(x)=-\\dfrac12\\displaystyle\\int|x-y|f(y)\\,dy$ の $u(0)$ を求め, $|x|<1$ で $-u^{\\prime\\prime}=1$ を確かめよ。")
        if st.toggle("解答・解説を表示", key="pr_newton"):
            st.markdown("**答1** $\\Delta r^2=2+\\frac2r\\cdot2r=6$。$\\Delta\\frac1r=\\frac{2}{r^3}-\\frac{2}{r^3}=0$($r\\ne0$)。")
            st.markdown("**答2** $u(0)=\\dfrac{\\rho_0}{6}\\cdot3a^2=3$。$u(2)=\\dfrac{\\rho_0a^3}{3\\cdot2}=1$。")
            st.markdown("**答3** $u(0)=-\\frac12\\int_{-1}^1|y|\\,dy=-\\frac12$。$|x|<1$ では $u(x)=-\\frac12\\left(\\frac{(x+1)^2}{2}+\\frac{(1-x)^2}{2}\\right)=-\\frac{x^2+1}{2}$ より $-u^{\\prime\\prime}=1$。")

    page_refs([("evans2010", "第2章:ラプラス方程式の基本解とポアソン方程式。"),
               ("strauss2008", "第7章:グリーンの公式とグリーン関数。"),
               ("stein1970", "第5章:リースポテンシャルとハーディ・リトルウッド・ソボレフの不等式(発展)。")])
    refs([("Newton's superb theorem: An elementary geometric proof(arXiv:1201.6534)", "https://arxiv.org/pdf/1201.6534", "殻の定理の初等的な証明と歴史(英語)"),
          ("MathPages:Newton's Proposition LXXI", "https://www.mathpages.com/home/kmath139/kmath139.htm", "『プリンキピア』第1巻 命題71(球殻の引力)の解説(英語)")])
