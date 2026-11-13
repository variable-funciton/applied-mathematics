"""⑥ その他:関数の内積と L² 空間(超関数との対比)

ベクトルの内積 → 関数の内積 → L² 空間 → 超関数(⟨T, φ⟩)の順に, 「関数をどう測るか」を比べる。
L²/Lᵖ の詳しい内容は ③ フーリエ解析 発展 7.(topics/lp_spaces.py), 超関数は topics/distribution.py。
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import C1, C2, C3, GRAY, layout, show, trapz
from topics.fourier_kernels import page_refs

PI = np.pi

# 区間 [−π, π] の関数:(関数, TeX)
FG = {
    "1": (lambda x: np.ones_like(x), "1"),
    "x": (lambda x: x, "x"),
    "x²": (lambda x: x * x, "x^2"),
    "sin x": (np.sin, r"\sin x"),
    "cos x": (np.cos, r"\cos x"),
    "sin 2x": (lambda x: np.sin(2 * x), r"\sin 2x"),
}
# 試験関数の代わり(遠くで速く 0 になる)
TEST = {
    "φ(x) = e^{−x²}": (lambda x: np.exp(-x ** 2), 1.0),
    "φ(x) = (1 + x) e^{−x²/2}": (lambda x: (1 + x) * np.exp(-x ** 2 / 2), 1.0),
    "φ(x) = cos x · e^{−x²/4}": (lambda x: np.cos(x) * np.exp(-x ** 2 / 4), 1.0),
}


def riemann_inner(f, g, n, a=-PI, b=PI):
    """ベクトルの内積に Δx を掛けたもの Σ f(x_k) g(x_k) Δx(中点)"""
    dx = (b - a) / n
    xk = a + (np.arange(n) + 0.5) * dx
    return float(np.sum(f(xk) * g(xk)) * dx), xk


def int_power(eps, p):
    """∫_ε^1 x^{−p} dx(閉じた式)"""
    return -np.log(eps) if abs(p - 1) < 1e-12 else (1 - eps ** (1 - p)) / (1 - p)


def delta_eps(x, eps):
    """ガウス型のモデル関数 δ_ε(面積 1)"""
    return np.exp(-x ** 2 / (2 * eps ** 2)) / (eps * np.sqrt(2 * PI))


def delta_eps_norm2(eps):
    """‖δ_ε‖² = ∫ δ_ε² dx = 1 / (2√π ε)"""
    return 1 / (2 * np.sqrt(PI) * eps)


def tab_l2_distribution():
    st.info("関数どうしにも, ベクトルと同じように**内積**が考えられる。2乗して積分できる関数の集まりが **$L^2$ 空間**で, "
            "フーリエ級数の理論の舞台になる。一方, 超関数も $\\langle T,\\varphi\\rangle$ という「内積のような形」で定義された。"
            "このページでは, 両者が**関数の何を測っているか**を比べ, デルタ関数が $L^2$ には入らないのに超関数には入る理由を見る。")
    st.caption("$L^2$, $L^p$ 空間そのもの(ノルム, 中線定理, 最良近似など)は ③ フーリエ解析 発展 7.「関数空間 $L^2$ と $L^p$」で詳しく扱っている。"
               "超関数は1つ前のページ「シュワルツの超関数の理論(デルタ関数)」を参照。")

    # ---------------------------------------------------------------- 1
    st.subheader("1. ベクトルの内積から関数の内積へ")
    st.markdown("区間 $[-\\pi,\\pi]$ を $n$ 等分した点 $x_1,\\dots,x_n$ での値を並べると, 関数は $n$ 次元のベクトル "
                "$(f(x_1),\\dots,f(x_n))$ で近似できる。その内積に幅 $\\Delta x$ を掛けて $n\\to\\infty$ とすると積分になる:")
    st.latex(r"\sum_{k=1}^{n}f(x_k)g(x_k)\,\Delta x\ \longrightarrow\ (f,g)=\int_{-\pi}^{\pi}f(x)g(x)\,dx")
    c = st.columns([1, 2])
    fn = c[0].selectbox("f", list(FG), index=1, key="l2d_f")
    gn = c[0].selectbox("g", list(FG), index=3, key="l2d_g")
    n = c[0].select_slider("分割数 n", [4, 8, 16, 32, 64, 256], 8, key="l2d_n")
    f, ft = FG[fn]
    g, gt = FG[gn]
    approx, xk = riemann_inner(f, g, n)
    x = np.linspace(-PI, PI, 4001)
    exact = float(trapz(f(x) * g(x), x))
    nf, ng = np.sqrt(float(trapz(f(x) ** 2, x))), np.sqrt(float(trapz(g(x) ** 2, x)))
    c[0].metric(f"Σ f(x_k) g(x_k) Δx(n = {n})", f"{approx:.5f}")
    c[0].metric("(f, g) = ∫ f g dx", f"{exact:.5f}")
    c[0].metric("cos θ = (f, g) / (‖f‖ ‖g‖)", f"{exact / (nf * ng):.4f}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=f(x), name=f"f(x) = {fn}", line=dict(color=C1)))
    fig.add_trace(go.Scatter(x=x, y=g(x), name=f"g(x) = {gn}", line=dict(color=C2)))
    dx = 2 * PI / n
    fig.add_trace(go.Bar(x=xk, y=f(xk) * g(xk), width=dx * 0.9, name="f(x_k) g(x_k)(棒の面積の和 = 近似値)",
                         marker=dict(color=C3, opacity=0.35)))
    layout(fig, f"$(f,g)=\\int_{{-\\pi}}^{{\\pi}}{ft}\\cdot {gt}\\,dx$ の近似", xlab="x", height=380)
    with c[1]:
        show(fig)
    st.markdown(r"""
| ベクトル($\mathbb R^n$) | 関数($[-\pi,\pi]$ 上) |
|---|---|
| 内積 $\boldsymbol u\cdot\boldsymbol v=\sum u_kv_k$ | 内積 $(f,g)=\int f(x)g(x)\,dx$(複素数値なら $\int f\,\overline{g}\,dx$) |
| 長さ $\lvert\boldsymbol u\rvert=\sqrt{\boldsymbol u\cdot\boldsymbol u}$ | ノルム $\lVert f\rVert=\sqrt{(f,f)}=\left(\int\lvert f\rvert^2dx\right)^{1/2}$ |
| 直交 $\boldsymbol u\cdot\boldsymbol v=0$ | 直交 $(f,g)=0$(例:$(\sin mx,\sin nx)=0$, $m\neq n$) |
| 成分 $u_k=\boldsymbol u\cdot\boldsymbol e_k$ | フーリエ係数 $b_n=\dfrac{(f,\sin nx)}{(\sin nx,\sin nx)}$ |
| ピタゴラスの定理 $\lvert\boldsymbol u\rvert^2=\sum u_k^2$ | パーセバルの等式 $\dfrac1\pi\lVert f\rVert^2=\dfrac{a_0^2}{2}+\sum(a_n^2+b_n^2)$ |
""")
    st.caption("$\\sin x$ と $\\sin 2x$, $x$ と $\\cos x$ などを選ぶと内積が $0$(直交)になることが確かめられる。")

    # ---------------------------------------------------------------- 2
    st.subheader("2. $L^2$ 空間:ノルムが有限な関数の集まり")
    st.markdown("ノルム $\\lVert f\\rVert$ が有限な関数, つまり $\\displaystyle\\int\\lvert f(x)\\rvert^2dx<\\infty$ となる関数の集まりを **$L^2$ 空間**という"
                "(区間を明示して $L^2(-\\pi,\\pi)$, $L^2(0,1)$, $L^2(\\mathbb R)$ などと書く)。"
                "$L^2$ の関数どうしなら, シュワルツの不等式 $\\lvert(f,g)\\rvert\\le\\lVert f\\rVert\\,\\lVert g\\rVert$ により内積がいつも有限になる。")
    st.markdown("**例**:$f(x)=x^{-\\alpha}$($0<x<1$, $\\alpha>0$)は原点で無限大になるが, 2乗の積分")
    st.latex(r"\int_0^1x^{-2\alpha}\,dx=\lim_{\varepsilon\to+0}\int_\varepsilon^1x^{-2\alpha}\,dx")
    st.markdown("が有限なら $L^2(0,1)$ に入る。つまり $2\\alpha<1$, $\\alpha<\\dfrac12$ のとき $f\\in L^2(0,1)$。"
                "積分そのもの($L^1$)は $\\alpha<1$ で有限なので, $\\dfrac12\\le\\alpha<1$ では「積分できるが2乗は積分できない」。")
    c = st.columns([1, 2])
    al = c[0].select_slider("α", [0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.75, 0.9, 1.0, 1.2], 0.4, key="l2d_al")
    epss = np.logspace(-8, -0.01, 300)
    i1, i2 = int_power(epss, al), int_power(epss, 2 * al)
    c[0].markdown(f"- $\\int_0^1 x^{{-\\alpha}}dx$:{'有限 ' + f'$= {1 / (1 - al):.3g}$' if al < 1 else '発散'}\n"
                  f"- $\\int_0^1 x^{{-2\\alpha}}dx$:{'有限 ' + f'$= {1 / (1 - 2 * al):.3g}$' if 2 * al < 1 else '発散'}\n\n"
                  f"→ $x^{{-{al}}}$ は $L^2(0,1)$ に{'**入る**' if 2 * al < 1 else '**入らない**'}。")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epss, y=i1, name="∫_ε^1 x^(−α) dx(L¹)", line=dict(color=C1)))
    fig.add_trace(go.Scatter(x=epss, y=i2, name="∫_ε^1 x^(−2α) dx(L²)", line=dict(color=C2)))
    layout(fig, "ε → 0 での積分の値", xlab="ε(対数目盛)", ylab="積分の値(対数目盛)", height=360)
    fig.update_xaxes(type="log", autorange="reversed")
    fig.update_yaxes(type="log")
    with c[1]:
        show(fig)
    st.caption("$L^2$ の議論を数学として完成させるにはルベーグ積分と完備性(コーシー列が収束すること)が必要になるが, ここでは立ち入らない。")

    # ---------------------------------------------------------------- 3
    st.subheader("3. 超関数との対比:$L^2$ の関数は超関数の一員")
    st.markdown("$g\\in L^2$ と試験関数 $\\varphi$(なめらかで遠くで $0$)に対して, 内積 $(g,\\varphi)$ はいつも有限で, $\\varphi$ について線形である。"
                "したがって $g$ は")
    st.latex(r"\langle g,\varphi\rangle=\int g(x)\varphi(x)\,dx=(g,\varphi)\qquad(\varphi\text{ は実数値})")
    st.markdown("によって**超関数とみなせる**。超関数の記号 $\\langle\\ ,\\ \\rangle$ は, もともと内積を手本にしている。関数の集まりの広がりは次のようになる:")
    st.latex(r"\{\text{試験関数}\}\ \subset\ L^2\ \subset\ \{\text{超関数}\}")
    st.markdown("- **$L^2$**:相手も $L^2$ の関数なら内積が測れる。「2乗の積分」という**大きさ(ノルム)**がある。\n"
                "- **超関数**:相手を**なめらかな試験関数に限る**かわりに, ずっと荒いもの($\\delta$, $\\delta^{\\prime}$ など)まで扱える。")

    st.markdown("#### デルタ関数は $L^2$ に入らない")
    st.markdown("ガウス型のモデル関数 $\\delta_\\varepsilon(x)=\\dfrac{1}{\\sqrt{2\\pi}\\,\\varepsilon}e^{-x^2/(2\\varepsilon^2)}$ は面積 $1$ で, "
                "$\\varepsilon\\to0$ のとき超関数として $\\delta$ に近づく。しかしノルムは")
    st.latex(r"\lVert\delta_\varepsilon\rVert^2=\int_{-\infty}^{\infty}\frac{e^{-x^2/\varepsilon^2}}{2\pi\varepsilon^2}\,dx=\frac{1}{2\sqrt\pi\,\varepsilon}\ \longrightarrow\ \infty")
    st.markdown("と発散する。面積を $1$ に保ったまま幅を狭くすると, 高さは $\\dfrac1\\varepsilon$ に比例し, 2乗の面積は $\\dfrac1\\varepsilon$ に比例して大きくなるからである。")
    c = st.columns([1, 2])
    eps = c[0].select_slider("幅 ε", [1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02], 0.3, key="l2d_eps")
    tn = c[0].radio("試験関数 φ", list(TEST), key="l2d_phi")
    phi = TEST[tn][0]
    x = np.linspace(-8, 8, 160001)
    de = delta_eps(x, eps)
    c[0].metric("∫ δ_ε dx(面積)", f"{float(trapz(de, x)):.5f}")
    c[0].metric("⟨δ_ε, φ⟩ → φ(0)", f"{float(trapz(de * phi(x), x)):.5f}", f"φ(0) = {float(phi(np.array(0.0))):.5f}", delta_color="off")
    c[0].metric("‖δ_ε‖² = ∫ δ_ε² dx", f"{float(trapz(de ** 2, x)):.4f}", f"1/(2√π ε) = {delta_eps_norm2(eps):.4f}", delta_color="off")
    xs = np.linspace(-3, 3, 3001)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=delta_eps(xs, eps), name="δ_ε(x)(面積 1)", line=dict(color=C1)))
    fig.add_trace(go.Scatter(x=xs, y=delta_eps(xs, eps) ** 2, name="δ_ε(x)²(面積 = ‖δ_ε‖²)", line=dict(color=C2), fill="tozeroy", opacity=0.4))
    fig.add_trace(go.Scatter(x=xs, y=phi(xs), name="試験関数 φ(x)", line=dict(color=GRAY, dash="dash")))
    layout(fig, "超関数としては収束, L² のノルムは発散", xlab="x", height=380, yrange=[-0.5, max(3.0, 1.1 * float(delta_eps(0.0, eps) ** 2))])
    with c[1]:
        show(fig)
    with st.expander("極限だけでなく, δ そのものが L² の関数で表せないことの説明"):
        st.markdown("もし $\\delta$ が $L^2$ の関数 $g$ で $\\langle\\delta,\\varphi\\rangle=(g,\\varphi)$ と表せたとすると, シュワルツの不等式から, すべての試験関数で")
        st.latex(r"\lvert\varphi(0)\rvert=\lvert(g,\varphi)\rvert\le\lVert g\rVert\,\lVert\varphi\rVert")
        st.markdown("が成り立つはずである。ところが $\\varphi(x)=e^{-x^2/(2\\varepsilon^2)}$ とすると $\\varphi(0)=1$ なのに "
                    "$\\lVert\\varphi\\rVert^2=\\sqrt\\pi\\,\\varepsilon\\to0$ なので, $\\varepsilon$ を小さくすると右辺は $0$ に近づき, 不等式が破れる。"
                    "よって $\\delta$ は $L^2$ の関数では表せない。\n\n"
                    "「1点での値 $\\varphi(0)$」は $L^2$ のノルムでは**測れない量**である($L^2$ では1点の値を変えても同じ関数とみなす)。"
                    "超関数は, 相手をなめらかな関数に限ることで, この量を扱えるようにしている。")

    st.markdown("#### 逆の例:$L^2$ では収束しないが, 超関数としては $0$ に収束する")
    st.markdown("$\\sin nx$($-\\pi<x<\\pi$)は $\\lVert\\sin nx\\rVert^2=\\pi$ で, $n$ が大きくなってもノルムは小さくならない($L^2$ では $0$ に収束しない)。"
                "しかし, なめらかな $\\varphi$ に当てると, 細かく振動する正負が打ち消し合って")
    st.latex(r"\langle\sin nx,\varphi\rangle=\int_{-\pi}^{\pi}\varphi(x)\sin nx\,dx\ \longrightarrow\ 0\qquad(n\to\infty)")
    st.markdown("となる(リーマン・ルベーグの補題。$\\varphi$ のフーリエ係数が $0$ に近づくこと)。")
    c = st.columns([1, 2])
    nn = c[0].select_slider("n", [1, 2, 4, 8, 16, 32, 64], 4, key="l2d_sn")
    x = np.linspace(-PI, PI, 40001)
    phi = lambda s: np.exp(-s ** 2) * (1 + s)  # noqa: E731
    c[0].metric("‖sin nx‖²", f"{float(trapz(np.sin(nn * x) ** 2, x)):.5f}", "π = 3.14159", delta_color="off")
    c[0].metric("⟨sin nx, φ⟩", f"{float(trapz(np.sin(nn * x) * phi(x), x)):.6f}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=np.sin(nn * x), name="sin nx", line=dict(color=C1, width=1)))
    fig.add_trace(go.Scatter(x=x, y=phi(x), name="φ(x) = (1 + x) e^{−x²}", line=dict(color=GRAY, dash="dash")))
    fig.add_trace(go.Scatter(x=x, y=np.sin(nn * x) * phi(x), name="φ(x) sin nx(積分 → 0)", line=dict(color=C2), fill="tozeroy"))
    layout(fig, "振動が速くなると, 試験関数との積分は打ち消し合う", xlab="x", height=360)
    with c[1]:
        show(fig)

    # ---------------------------------------------------------------- 4
    st.subheader("4. まとめ:$L^2$ と超関数の比較")
    st.markdown(r"""
| | $L^2$ 空間 | 超関数 |
|---|---|---|
| 相手 | $L^2$ の関数 $g$ | なめらかな試験関数 $\varphi$ |
| 測るもの | 内積 $(f,g)=\int fg\,dx$, ノルム $\lVert f\rVert$ | $\langle T,\varphi\rangle$(ノルムは考えない) |
| 例 | $\sin nx$, $x^{-\alpha}\ (\alpha<\frac12)$, 不連続な関数 | $L^2$ の関数, $\delta$, $\delta^{\prime}$, $1$, $e^{x}$ など |
| 微分 | いつもできるとは限らない($H$ の微分 $\delta$ は $L^2$ に入らない) | いつでも何回でもできる($\langle T^{\prime},\varphi\rangle=-\langle T,\varphi^{\prime}\rangle$) |
| 収束 | ノルムで測る($\sin nx$ は $0$ に収束しない) | 試験関数に当てた値で測る($\delta_\varepsilon\to\delta$, $\sin nx\to0$) |
| フーリエ変換 | $L^2$ から $L^2$ への変換で, ノルムを保つ(プランシュレルの定理) | $\mathcal F[\delta]=1$, $\mathcal F[1]=2\pi\delta$ のように $L^2$ の外まで広がる |
""")
    st.markdown("**ソボレフ空間(紹介)**:両者を組み合わせると, 「$f$ も, 超関数としての微分 $f^{\\prime}$ も $L^2$ に入る」関数の集まり $H^1$ が作れる。"
                "例えば $\\lvert x\\rvert e^{-x^2}$ は折れ目があってふつうには微分できないが $H^1$ に入る。一方 $H(x)e^{-x^2}$ の微分には $\\delta$ が現れるので $H^1$ に入らない。"
                "偏微分方程式の「弱い解」は, このような空間の中で探す(前のページの歴史にあるソボレフの仕事につながる)。")

    # ---------------------------------------------------------------- 5
    with st.expander("練習問題"):
        st.markdown("**問1** $[-\\pi,\\pi]$ での内積 $(x,\\sin x)$ と, ノルム $\\lVert x\\rVert$ を求めよ。")
        st.markdown("**問2** $f(x)=x^{-1/3}$, $g(x)=x^{-2/3}$($0<x<1$)は $L^2(0,1)$ に入るか。")
        st.markdown("**問3** $\\delta_\\varepsilon(x)=\\dfrac{1}{2\\varepsilon}$($\\lvert x\\rvert<\\varepsilon$), $0$($\\lvert x\\rvert>\\varepsilon$)とする。"
                    "$\\displaystyle\\int\\delta_\\varepsilon\\,dx$ と $\\lVert\\delta_\\varepsilon\\rVert^2$ を求め, $\\varepsilon\\to+0$ のときどうなるか答えよ。")
        st.markdown("**問4** $[-\\pi,\\pi]$ で $\\displaystyle\\int_{-\\pi}^{\\pi}x\\sin nx\\,dx$ を求め, $n\\to\\infty$ の極限と $\\lVert\\sin nx\\rVert^2$ を比べよ。")
        if st.toggle("解答・解説を表示", key="pr_l2d"):
            st.markdown("**答1** 部分積分で $(x,\\sin x)=\\bigl[-x\\cos x\\bigr]_{-\\pi}^{\\pi}+\\int_{-\\pi}^{\\pi}\\cos x\\,dx=2\\pi$。"
                        "$\\lVert x\\rVert^2=\\int_{-\\pi}^{\\pi}x^2dx=\\dfrac{2\\pi^3}{3}$ より $\\lVert x\\rVert=\\sqrt{\\dfrac{2\\pi^3}{3}}$。")
            st.markdown("**答2** $f^2=x^{-2/3}$ の積分は $3$ で有限なので $f\\in L^2(0,1)$。$g^2=x^{-4/3}$ の積分は発散するので $g\\notin L^2(0,1)$"
                        "(ただし $\\int_0^1g\\,dx=3$ は有限)。")
            st.markdown("**答3** $\\int\\delta_\\varepsilon\\,dx=2\\varepsilon\\cdot\\dfrac{1}{2\\varepsilon}=1$, "
                        "$\\lVert\\delta_\\varepsilon\\rVert^2=2\\varepsilon\\cdot\\dfrac{1}{4\\varepsilon^2}=\\dfrac{1}{2\\varepsilon}\\to\\infty$。"
                        "面積は $1$ のままだが, $L^2$ のノルムは発散する(極限の $\\delta$ は $L^2$ に入らない)。")
            st.markdown("**答4** 部分積分で $\\int_{-\\pi}^{\\pi}x\\sin nx\\,dx=\\dfrac{2\\pi(-1)^{n+1}}{n}\\to0$。"
                        "一方 $\\lVert\\sin nx\\rVert^2=\\pi$ はいつも同じ。$x$ に当てた値は $0$ に近づくが, 大きさは減らない。")

    page_refs([("stein2007", "第3章:内積空間とフーリエ級数の2乗平均収束(パーセバルの等式)。"),
               ("steinreal2005", "ルベーグ積分・L² 空間・ヒルベルト空間。"),
               ("schwartz1950", "超関数の理論。L² などの関数空間と超関数の関係。"),
               ("sobolev1936", "一般化された解(ソボレフ空間の考え方の先がけ)。")])
