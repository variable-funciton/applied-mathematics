"""⑥ その他:シュワルツの超関数の理論(デルタ関数)

旧「デルタ関数のモデル関数」(topics/other.py の tab_delta)を 2. の節として取り込んだ。
超関数 T は試験関数 φ に数を対応させる:⟨T, φ⟩。ふつうの関数は ⟨f, φ⟩ = ∫ f φ dx, ⟨δ, φ⟩ = φ(0)。
微分は部分積分を手本に ⟨T', φ⟩ = −⟨T, φ'⟩ で定義する。
史実・書誌は 2026 年 11 月に確認(Numdam, Math-Net.Ru, MaRDI, MacTutor など)。
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import C1, C2, C3, C4, GRAY, layout, show, trapz
from topics.fourier_apps import refs
from topics.fourier_kernels import page_refs
from topics.other import DELTA, TESTF

PI = np.pi

# 試験関数(なめらかで, 遠くで十分速く 0 になる)
PHIS = {
    "φ(x) = e^{−x²}": (lambda x: np.exp(-x ** 2), lambda x: -2 * x * np.exp(-x ** 2)),
    "φ(x) = (1 + x) e^{−x²/2}": (lambda x: (1 + x) * np.exp(-x ** 2 / 2), lambda x: (1 - x - x * x) * np.exp(-x ** 2 / 2)),
    "φ(x) = cos x · e^{−x²/4}": (lambda x: np.cos(x) * np.exp(-x ** 2 / 4), lambda x: (-np.sin(x) - x / 2 * np.cos(x)) * np.exp(-x ** 2 / 4)),
}
SMOOTH = {"sin x": (np.sin, np.cos, r"\sin x", r"\cos x"), "x²/4": (lambda x: x * x / 4, lambda x: x / 2, r"\frac{x^2}{4}", r"\frac{x}{2}"),
          "0": (lambda x: 0 * x, lambda x: 0 * x, "0", "0")}


def heaviside_eps(x, eps):
    """なめらかにしたヘヴィサイド関数 H_ε(x) = ½(1 + tanh(x/ε))"""
    return 0.5 * (1 + np.tanh(x / eps))


def d_heaviside_eps(x, eps):
    e = np.exp(-2 * np.abs(x / eps))          # sech² を桁あふれなしで計算
    return 0.5 / eps * 4 * e / (1 + e) ** 2


def pair_minus_int(f, dphi, x):
    """⟨f′, φ⟩ = −∫ f φ′ dx(数値積分)"""
    return -float(trapz(f(x) * dphi(x), x))


def model_functions():
    st.markdown("**モデル関数**:面積 1 のまま幅を細くしていく関数の列 $\\delta_\\varepsilon$ は, 次の性質をもつ:")
    st.latex(r"\int_{-\infty}^{\infty}\delta_\varepsilon(t)\,dt=1,\qquad "
             r"\lim_{\varepsilon\to0}\int_{-\infty}^{\infty}\delta_\varepsilon(t)\,g(t)\,dt=g(0),\qquad "
             r"\mathcal F[\delta_\varepsilon](\omega)\to1")
    c1, c2, c3 = st.columns(3)
    name = c1.selectbox("モデル関数", list(DELTA), key="dl_m")
    gname = c2.selectbox("試験関数 g", list(TESTF), key="dl_g")
    eps = c3.slider("ε", 0.02, 1.0, 0.3, 0.01, key="dl_e")
    D = DELTA[name]
    g, g0 = TESTF[gname]
    st.latex(D["tex"] + r",\qquad \mathcal F[\delta_\varepsilon]=" + D["Ftex"])
    t = np.linspace(-60, 60, 600001)
    dv = D["d"](t, eps)
    m1, m2, m3 = st.columns(3)
    m1.metric("面積 ∫δ_ε dt", f"{float(trapz(dv, t)):.5f}")
    m2.metric("∫δ_ε g dt", f"{float(trapz(dv * g(t), t)):.5f}")
    m3.metric("g(0)", f"{g0:.5f}")
    cA, cB = st.columns(2)
    with cA:
        tp = np.linspace(-3, 3, 3001)
        fig = go.Figure()
        for e, op in ((1.0, 0.25), (0.5, 0.45)):
            fig.add_trace(go.Scatter(x=tp, y=D["d"](tp, e), mode="lines", line=dict(color=GRAY, width=1), opacity=op + 0.3, name=f"ε = {e:g}"))
        fig.add_trace(go.Scatter(x=tp, y=D["d"](tp, eps), mode="lines", line=dict(color=C1, width=2.5), name=f"ε = {eps:g}"))
        layout(fig, title="δ_ε(t)", xlab="t", height=360, yrange=(-0.4 / eps, 1.2 / eps))
        show(fig)
    with cB:
        w = np.linspace(-40, 40, 2001)
        fig = go.Figure(go.Scatter(x=w, y=D["F"](w, eps), mode="lines", line=dict(color=C2, width=2.5), name="F[δ_ε](ω)"))
        fig.add_hline(y=1, line=dict(color=GRAY, dash="dash"), annotation_text="F[δ] = 1")
        layout(fig, title="フーリエ変換(ε → 0 ですべての ω で 1 に)", xlab="ω", height=360, yrange=(-0.3, 1.2))
        show(fig)
    st.caption("ε を小さくすると, δ_ε は高く細くなり, ∫δ_ε g dt は g(0) に近づく。ガウス型は ⑤ 偏微分方程式の熱核, "
               "ローレンツ型は上半平面のポアソン核と同じ形。どのモデル関数を使っても, 試験関数との積分の極限は同じ g(0) になる。")


def tab_distribution():
    st.info("デルタ関数 $\\delta(x)$ は「$x=0$ にだけ集中した, 面積 1 のパルス」として, 工学や物理で当たり前のように使われる。"
            "しかし, そのような「関数」は実は存在しない。この矛盾を解消し, デルタ関数やヘヴィサイド関数の微分を正しく扱えるようにしたのが, "
            "シュワルツ(L. Schwartz)の**超関数(distribution)の理論**である。")

    st.subheader("1. 導入の経緯:便利だが, 関数ではない")
    st.markdown("- **ヘヴィサイドの演算子法**(19世紀末):電気回路のスイッチを入れる瞬間を, ヘヴィサイド関数(単位ステップ関数)")
    st.latex(r"H(x)=\begin{cases}0&(x<0)\\1&(x>0)\end{cases}\qquad(\text{ラプラス変換の章の }U(t)\text{ と同じ})")
    st.markdown("  で表した。その「微分」は, 一瞬だけ無限に大きい衝撃(インパルス)と考えられ, 便利に使われた。\n"
                "- **ディラックの量子力学**(1920年代後半):位置が1点に定まった状態などを表すために, 次の性質をもつ $\\delta$ を「関数」として自由に使った:")
    st.latex(r"\delta(x)=0\ (x\neq0),\qquad\int_{-\infty}^{\infty}\delta(x)\,dx=1,\qquad\int_{-\infty}^{\infty}\delta(x)f(x)\,dx=f(0)")
    st.markdown("- **矛盾**:1点を除いて 0 の関数は, (リーマン積分でもルベーグ積分でも)積分が 0 になる。1点での値をどれだけ大きくしても積分は 1 にならない。"
                "つまり, こんな**ふつうの関数は存在しない**。また $H(x)$ は $x=0$ で微分できないので, 「$H'=\\delta$」もふつうの微分の意味では成り立たない。\n\n"
                "それでも, これらを使った計算は正しい答えを出していた。では「何として」正しいのか, というのが超関数の理論の出発点である。")

    st.subheader("2. 関数の列の極限として見る(モデル関数)")
    model_functions()
    st.markdown("どのモデル関数でも同じ結果になるのは, 結果が**関数の値そのものではなく, 他の関数 $g$ との積分**で決まっているからである。これが次の考え方につながる。")

    st.subheader("3. 超関数の考え方:「点の値」から「試験関数に当てた値」へ")
    st.markdown("なめらかで, ある有界な区間の外では 0 になる関数 $\\varphi$ を**試験関数**という(ここでの図では, 遠くで速く 0 になる関数で代用する)。"
                "関数 $f$ を「各点での値」ではなく, 試験関数に当てたときの値")
    st.latex(r"\langle f,\varphi\rangle=\int_{-\infty}^{\infty}f(x)\varphi(x)\,dx")
    st.markdown("の集まりで捉え直す(発展 7. の内積と同じ形)。試験関数 $\\varphi$ に数 $\\langle T,\\varphi\\rangle$ を線形に対応させる規則 $T$ を**超関数**という。")
    st.markdown("- ふつうの関数 $f$ は, 上の積分で超関数とみなせる(点ごとの値が少し違っても積分は変わらないので, 同じ超関数になる)。\n"
                "- **デルタ関数**は「$\\varphi$ に $\\varphi(0)$ を対応させる規則」として**定義**する:")
    st.latex(r"\langle\delta,\varphi\rangle=\varphi(0),\qquad\langle\delta_a,\varphi\rangle=\varphi(a)\quad(\delta_a=\delta(x-a))")
    st.markdown("これなら矛盾はない。「$\\int\\delta(x)\\varphi(x)\\,dx=\\varphi(0)$」という書き方は, この規則の**記号**として使ってよいことになる。")
    st.caption("厳密には, 試験関数の列の収束を決めておき, 超関数はその収束について連続であることも要求する(数学科の授業で扱う内容)。")

    st.subheader("4. 超関数の微分:ヘヴィサイド関数の微分はデルタ関数")
    st.markdown("なめらかな $f$ では, 部分積分(試験関数は遠くで 0)により $\\langle f^{\\prime},\\varphi\\rangle=\\int f^{\\prime}\\varphi\\,dx=-\\int f\\varphi^{\\prime}\\,dx$。"
                "そこで, どんな超関数 $T$ にも")
    st.latex(r"\langle T^{\prime},\varphi\rangle=-\langle T,\varphi^{\prime}\rangle")
    st.markdown("で微分を**定義**する。右辺は $\\varphi^{\\prime}$ しか使わないので, $T$ がどんなに不連続でも, **超関数は何回でも微分できる**。ヘヴィサイド関数では")
    st.latex(r"\langle H^{\prime},\varphi\rangle=-\int_{-\infty}^{\infty}H(x)\varphi^{\prime}(x)\,dx=-\int_0^{\infty}\varphi^{\prime}(x)\,dx=-\bigl[\varphi(x)\bigr]_0^{\infty}=\varphi(0)=\langle\delta,\varphi\rangle")
    st.markdown("となり, **$H^{\\prime}=\\delta$ が定義どおりの計算で示される**。")
    c = st.columns([1, 2])
    pn = c[0].radio("試験関数 φ", list(PHIS), key="ds_phi")
    eps = c[0].select_slider("なめらかにする幅 ε", [1.0, 0.5, 0.2, 0.1, 0.05, 0.02], 0.2, key="ds_eps")
    phi, dphi = PHIS[pn]
    x = np.linspace(-12, 12, 240001)
    lhs = pair_minus_int(lambda s: (s > 0).astype(float), dphi, x)
    c[0].metric("−∫ H φ′ dx", f"{lhs:.6f}")
    c[0].metric("φ(0)", f"{float(phi(np.array(0.0))):.6f}")
    c[0].metric("∫ H_ε′ φ dx(なめらかにした H の微分)", f"{float(trapz(d_heaviside_eps(x, eps) * phi(x), x)):.6f}")
    with c[1]:
        xp = np.linspace(-3, 3, 1201)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xp, y=(xp > 0).astype(float), mode="lines", line=dict(color=GRAY, width=2, shape="hv"), name="H(x)"))
        fig.add_trace(go.Scatter(x=xp, y=heaviside_eps(xp, eps), mode="lines", line=dict(color=C1, width=2.5), name=f"H_ε(x)(ε = {eps})"))
        fig.add_trace(go.Scatter(x=xp, y=d_heaviside_eps(xp, eps), mode="lines", line=dict(color=C3, width=2.5), name="H_ε′(x)"))
        fig.add_trace(go.Scatter(x=xp, y=phi(xp), mode="lines", line=dict(color=C2, width=1.5, dash="dot"), name="φ(x)"))
        layout(fig, title="段差をなめらかにして微分すると, 面積 1 の細いパルスになる", xlab="x", height=360, yrange=(-0.3, min(1 / (2 * eps) + 0.3, 13)))
        show(fig)
    st.caption("$H_\\varepsilon^{\\prime}$ は面積 1 のモデル関数(2. と同じ)で, ε → 0 で ∫ H_ε′ φ dx は φ(0) に近づく。"
               "超関数の微分は, こうした極限と一致するように作られている。")

    st.markdown("**とびのある関数の微分**:$f=g+J\\,H(x-a)$($g$ はなめらか, $x=a$ で高さ $J$ のとび)の超関数としての微分は")
    st.latex(r"f^{\prime}=g^{\prime}+J\,\delta(x-a)\qquad(\text{ふつうの微分}+\text{とびの大きさ}\times\text{デルタ関数})")
    c = st.columns(4)
    gk = c[0].radio("なめらかな部分 g", list(SMOOTH), key="ds_g")
    a = c[1].slider("とびの位置 a", -2.0, 2.0, 0.5, 0.25, key="ds_a")
    J = c[2].slider("とびの大きさ J", -2.0, 2.0, 1.5, 0.5, key="ds_J")
    g, dg, gt, dgt = SMOOTH[gk]
    xp = np.linspace(-3, 3, 1201)
    cols = st.columns(2)
    with cols[0]:
        f = g(xp) + J * (xp > a)
        fig = go.Figure(go.Scatter(x=np.where(np.abs(xp - a) < 1e-9, np.nan, xp), y=f, mode="lines", line=dict(color=C1, width=2.5), name="f(x)"))
        layout(fig, title="f(x) = g(x) + J H(x − a)", xlab="x", height=300, legend=False)
        show(fig)
    with cols[1]:
        fig = go.Figure(go.Scatter(x=xp, y=dg(xp), mode="lines", line=dict(color=C3, width=2.5), name="g′(x)"))
        if J != 0:
            fig.add_annotation(x=a, y=J, ax=a, ay=0, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowwidth=3, arrowcolor=C2)
            fig.add_annotation(x=a, y=J, text=f"{J:+g} δ(x − {a:g})", showarrow=False, yshift=12 if J > 0 else -12, font=dict(color=C2))
        layout(fig, title="f′ = g′ + J δ(x − a)(矢印がデルタ関数, 長さは係数)", xlab="x", height=300, legend=False,
               yrange=(min(-2.3, float(dg(xp).min()) - 0.3), max(2.3, float(dg(xp).max()) + 0.3)))
        show(fig)
    st.latex(r"f=" + gt + (r"+" if J >= 0 else "") + f"{J:g}" + rf"H(x-{a:g})\ \Longrightarrow\ f^{{\prime}}=" + dgt + (r"+" if J >= 0 else "") + f"{J:g}" + rf"\delta(x-{a:g})")
    st.markdown("**よく使う公式**")
    st.latex(r"|x|^{\prime}=\mathrm{sgn}\,x,\quad |x|^{\prime\prime}=2\delta,\quad (xH)^{\prime\prime}=\delta,\quad f(x)\delta(x-a)=f(a)\delta(x-a),\quad "
             r"\delta(kx)=\frac{1}{|k|}\delta(x),\quad\langle\delta^{\prime},\varphi\rangle=-\varphi^{\prime}(0)")
    st.caption("$\\delta^{\\prime}$ は「とても近い正と負のパルスの組」(双極子)を表す。$\\delta(kx)$ は $\\int\\delta(kx)\\varphi(x)dx$ で $y=kx$ と置換して確かめられる。")

    st.subheader("5. これまでのページとのつながり")
    st.markdown(r"""
| 場所 | 超関数で書くと | 意味 |
|---|---|---|
| ② ラプラス変換 第2章 3. | $\mathcal L[\delta]=1$, $\mathcal L[\delta(t-a)]=e^{-as}$, $U^{\prime}=\delta$ | インパルス応答 $g=\mathcal L^{-1}[G]$ は $\delta$ を入れたときの出力 |
| ③ フーリエ解析 合成積 | $f*\delta=f$ | $\delta$ は合成積の単位元, 良い核・近似単位元は $\delta$ に近づく関数の列 |
| ⑤ 偏微分方程式 発展5. | $-\Delta\dfrac{1}{4\pi\lvert x\rvert}=\delta$ | 基本解(グリーン関数)。$\frac{1}{4\pi\lvert x\rvert}$ は原点で微分できないが, 超関数としては正しい |
| ⑤ 偏微分方程式 0. 変数分離法 | $u_{xy}=0$ などの弱い解 | 微分できない関数も「解」として扱える(ソボレフの一般化された解) |
| 機械・都市(はり) | $EI\,y^{\prime\prime\prime\prime}=P\,\delta(x-a)$ | 1点にかかる集中荷重 |
""")
    st.markdown("**フーリエ変換(紹介)**:遠くでゆっくりしか増えない超関数(緩増加超関数)にはフーリエ変換が定義でき, "
                "ふつうの意味では積分が収束しない関数のフーリエ変換も扱えるようになる:")
    st.latex(r"\mathcal F[\delta]=1,\qquad\mathcal F[1]=2\pi\delta(\omega),\qquad\mathcal F[\cos\omega_0t]=\pi\bigl(\delta(\omega-\omega_0)+\delta(\omega+\omega_0)\bigr),\qquad"
             r"\mathcal F[f^{\prime}]=i\omega\,\mathcal F[f]")
    st.caption("フーリエ級数とフーリエ変換が同じ枠組みで扱えるようになる(周期関数のフーリエ変換は, デルタ関数の列 = 線スペクトル)。"
               "フーリエ変換の規約は ③ と同じ $F(\\omega)=\\int f(t)e^{-i\\omega t}dt$。")

    st.subheader("6. 歴史")
    st.markdown(r"""
| 年 | できごと |
|---|---|
| 19世紀末 | ヘヴィサイド(O. Heaviside)が電気回路の計算で演算子法を使い, ステップ関数とその「微分」(衝撃)を自由に扱う |
| 1927 | ディラック(P. A. M. Dirac)が量子力学の論文でデルタ関数を導入。1930 年の教科書『量子力学の原理』で広める |
| 1936 | ソボレフ(S. L. Sobolev)が双曲型偏微分方程式のコーシー問題の研究で, 微分できない関数も「一般化された解」として扱う方法を示す(試験関数を使う考え方の先がけ) |
| 1945 | シュワルツ(L. Schwartz)が「関数, 微分, フーリエ変換の概念の一般化」を発表し, 超関数(distribution)を導入 |
| 1950〜1951 | シュワルツの『超関数の理論(Théorie des distributions)』全2巻が出版される。1950 年, シュワルツはこの業績でフィールズ賞を受賞 |
| その後 | ゲルファントらによる一般化関数の研究, 偏微分方程式の基本解の存在(マルグランジュ・エーレンプライスの定理)などへ発展し, 現代の偏微分方程式論・調和解析の共通言語となる |
""")

    with st.expander("練習問題"):
        st.markdown("**問1** $\\displaystyle\\int_{-\\infty}^{\\infty}\\delta(t-2)(t^2+1)\\,dt$, $\\mathcal L[\\delta(t-3)]$ を求めよ。")
        st.markdown("**問2** 超関数の微分の定義を使って, $\\langle H^{\\prime},\\varphi\\rangle=\\varphi(0)$ を示せ。")
        st.markdown("**問3** $f(x)=\\begin{cases}x&(x<1)\\\\x+3&(x>1)\\end{cases}$ を超関数として微分せよ。")
        st.markdown("**問4** $|x|$ を超関数として2回微分すると $2\\delta$ になることを示せ。")
        if st.toggle("解答・解説を表示", key="pr_dist"):
            st.markdown("**答1** $2^2+1=5$, $e^{-3s}$。")
            st.markdown("**答2** $\\langle H^{\\prime},\\varphi\\rangle=-\\int_0^\\infty\\varphi^{\\prime}dx=-(0-\\varphi(0))=\\varphi(0)$($\\varphi$ は遠くで 0)。")
            st.markdown("**答3** $f=x+3H(x-1)$ より $f^{\\prime}=1+3\\delta(x-1)$。")
            st.markdown("**答4** $|x|^{\\prime}=\\mathrm{sgn}\\,x=2H(x)-1$ なので $|x|^{\\prime\\prime}=2H^{\\prime}=2\\delta$。"
                        "(定義から:$\\langle|x|^{\\prime\\prime},\\varphi\\rangle=\\int|x|\\varphi^{\\prime\\prime}dx$ を $x<0$ と $x>0$ に分けて2回部分積分すると $2\\varphi(0)$。)")

    page_refs([("schwartz1945", ""), ("schwartz1950", ""), ("dirac1927", ""), ("sobolev1936", "")])
    refs([("Numdam:L. Schwartz, Généralisation de la notion de fonction, ...(1945)", "https://numdam.org/item/AUG_1945__21__57_0/", "超関数を導入した論文(フランス語)"),
          ("MacTutor History of Mathematics:Laurent Schwartz(The Guardian の追悼記事)", "https://mathshistory.st-andrews.ac.uk/Obituaries/Schwartz_Guardian/",
           "超関数の理論とフィールズ賞(英語)"),
          ("Sobolev and Schwartz: Two Fates and Two Fames(arXiv:0802.0533)", "https://arxiv.org/pdf/0802.0533",
           "ソボレフとシュワルツの仕事と, 超関数の理論の成り立ち(英語)"),
          ("Schwartz' Creation of the Theory of Distributions(Springer の書籍 The Prehistory of the Theory of Distributions の1章)",
           "https://link.springer.com/chapter/10.1007/978-1-4613-9472-3_7", "超関数の前史とシュワルツの理論の誕生(英語)")])
