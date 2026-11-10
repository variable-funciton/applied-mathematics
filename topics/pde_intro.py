"""⑤ 偏微分方程式 導入:波動方程式・熱方程式の導出と歴史, ラプラス方程式の導入と歴史

史実は次で確認した(2026 年 11 月):MacTutor(Fourier の伝記, Archibald "Mathematicians and Music" 3),
Encyclopedia of Mathematics "Laplace equation", W. W. Rouse Ball "A Short Account of the History of Mathematics"(Laplace),
Wikipedia "An Essay on the Application of Mathematical Analysis to the Theories of Electricity and Magnetism"。
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import C1, C2, C3, C4, GRAY, layout, show
from topics.fourier_apps import refs
from topics.fourier_kernels import page_refs

PI = np.pi
SHAPES = {"山形(なめらか)": lambda x: np.sin(PI * x) ** 3,
          "三角形(はじいた弦)": lambda x: np.where(x < 0.3, x / 0.3, (1 - x) / 0.7),
          "正弦波 sin 2πx": lambda x: np.sin(2 * PI * x)}
TEMPS = {"中央が熱い(山形)": lambda x: np.exp(-((x - 0.5) / 0.15) ** 2),
         "左が熱く右が冷たい(直線)": lambda x: 1 - x,
         "左半分だけ熱い(段差をなめらかに)": lambda x: 0.5 * (1 - np.tanh((x - 0.5) / 0.05)),
         "正弦波 sin πx": lambda x: np.sin(PI * x)}


def d2(f, x, h=1e-3):
    return (f(x + h) - 2 * f(x) + f(x - h)) / h ** 2


def d1(f, x, h=1e-4):
    return (f(x + h) - f(x - h)) / (2 * h)


# ------------------------------------------------------------
# 1. 波動方程式・熱方程式
# ------------------------------------------------------------
def tab_wave_heat_intro():
    st.info("偏微分方程式は, 物理の法則(ニュートンの運動方程式, エネルギーの保存など)を, "
            "**小さな部分に当てはめて極限をとる**ことで導かれる。ここでは弦の振動から**波動方程式**を, "
            "棒の熱の流れから**熱方程式**を導き, その歴史をたどる。")

    st.subheader("1. 波動方程式:弦の振動")
    st.markdown("張力 $T$ で張った弦(線密度 $\\rho$:1 m あたりの質量)の, 位置 $x$, 時刻 $t$ での変位を $u(x,t)$ とする。"
                "**変位は小さい**(弦の傾き $u_x$ が小さい)とし, 重力は考えない。")
    st.markdown("区間 $[x,\\,x+\\Delta x]$ の小さな部分に働く力を考える。両端では張力 $T$ が弦の接線の向きに引っ張る。"
                "接線の傾き角を $\\theta$ とすると, 縦方向の成分は $T\\sin\\theta\\approx T\\tan\\theta=Tu_x$(傾きが小さいので)。よって")
    st.latex(r"\underbrace{\rho\,\Delta x}_{\text{質量}}\ \frac{\partial^2u}{\partial t^2}=\underbrace{T\,u_x(x+\Delta x,t)-T\,u_x(x,t)}_{\text{縦方向の力の合計}}")
    st.markdown("両辺を $\\rho\\Delta x$ で割って $\\Delta x\\to0$ とすると, 右辺は微分の定義から $\\dfrac{T}{\\rho}u_{xx}$ になる:")
    st.latex(r"\frac{\partial^2u}{\partial t^2}=c^2\frac{\partial^2u}{\partial x^2},\qquad c=\sqrt{\frac{T}{\rho}}\ (\text{波の速さ})")
    st.markdown("**意味**:右辺の $u_{xx}$ は弦の**曲がり具合**(下に凸なら正)。下に凸なところは上向きに, 上に凸なところは下向きに加速される。")
    c = st.columns([1, 2])
    shp = c[0].radio("弦の形", list(SHAPES), key="pi_shape")
    Tn = c[0].slider("張力 T [N]", 10.0, 200.0, 80.0, 10.0, key="pi_T")
    rho = c[0].select_slider("線密度 ρ [g/m]", [0.5, 1.0, 2.0, 5.0, 10.0], 2.0, key="pi_rho")
    cs = np.sqrt(Tn / (rho * 1e-3))
    c[0].metric("波の速さ c = √(T/ρ)", f"{cs:.0f} m/s")
    c[0].metric("長さ 0.65 m の弦の基本振動数 c/(2L)", f"{cs / 1.3:.0f} Hz")
    f = SHAPES[shp]
    x = np.linspace(0, 1, 401)
    xs = np.linspace(0.05, 0.95, 19)
    acc = d2(f, xs)
    sc = 0.25 / max(np.max(np.abs(acc)), 1e-9)
    with c[1]:
        fig = go.Figure(go.Scatter(x=x, y=f(x), mode="lines", line=dict(color=C1, width=3), name="弦の形 u(x)"))
        for xi, a in zip(xs, acc):
            fig.add_annotation(x=xi, y=f(xi) + sc * a, ax=xi, ay=f(xi), xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=2, arrowcolor=C2, arrowwidth=2)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=C2), name="加速度 ∝ u_xx(矢印)"))
        layout(fig, title="曲がり具合 u_xx が加速度を決める", xlab="x", ylab="u", height=360, yrange=(-1.4, 1.4))
        show(fig)
    st.caption("三角形の頂点のように折れ曲がったところでは $u_{xx}$ が非常に大きく, 強く引き戻される(はじいた弦が動き出すところ)。"
               "ギターの弦を強く張る($T$ を大きくする)ほど, 細い(軽い)弦ほど, 音は高くなる。")

    st.subheader("2. 熱方程式:棒の熱の流れ")
    st.markdown("断面積 $A$ の細い棒の温度を $u(x,t)$ とし, 側面からは熱が逃げないとする。使う法則は2つ:")
    st.markdown("- **フーリエの法則**:熱は温度の高い方から低い方へ, 温度勾配に比例して流れる。単位面積・単位時間あたりの熱の流れ(熱流束)は "
                "$q=-k\\,u_x$($k$ は熱伝導率)。\n"
                "- **エネルギーの保存**:区間 $[x,\\,x+\\Delta x]$ にたまる熱 = 左から入る熱 − 右から出る熱。")
    st.latex(r"\underbrace{c_p\rho A\Delta x}_{\text{熱容量}}\frac{\partial u}{\partial t}=A\,q(x,t)-A\,q(x+\Delta x,t)=kA\left(u_x(x+\Delta x,t)-u_x(x,t)\right)")
    st.markdown("($c_p$ は比熱, $\\rho$ は密度。)$c_p\\rho A\\Delta x$ で割って $\\Delta x\\to0$ とすると")
    st.latex(r"\frac{\partial u}{\partial t}=\kappa\frac{\partial^2u}{\partial x^2},\qquad \kappa=\frac{k}{c_p\rho}\ (\text{熱拡散率})")
    st.markdown("**意味**:温度のグラフが下に凸なところ(周りより冷たい)は温まり, 上に凸なところ(周りより熱い)は冷える。"
                "波動方程式は $u_{tt}$(加速度), 熱方程式は $u_t$(速度)なので, 波は振動し, 熱はならされていく。")
    c = st.columns([1, 2])
    tp = c[0].radio("温度分布", list(TEMPS), key="pi_temp")
    g = TEMPS[tp]
    with c[1]:
        fig = go.Figure(go.Scatter(x=x, y=g(x), mode="lines", line=dict(color=C3, width=3), name="温度 u(x)"))
        q = -d1(g, xs)
        qs = 0.06 / max(np.max(np.abs(q)), 1e-9)
        ut = d2(g, xs)
        us = 0.2 / max(np.max(np.abs(ut)), 1e-9)
        for xi, qi, ui in zip(xs, q, ut):
            if abs(qi) > 1e-6:
                fig.add_annotation(x=xi + qs * qi, y=g(xi), ax=xi, ay=g(xi), xref="x", yref="y", axref="x", ayref="y",
                                   showarrow=True, arrowhead=2, arrowcolor=C1, arrowwidth=2)
            if abs(ui) > 1e-6:
                fig.add_annotation(x=xi, y=g(xi) + us * ui, ax=xi, ay=g(xi), xref="x", yref="y", axref="x", ayref="y",
                                   showarrow=True, arrowhead=1, arrowcolor=C2, arrowwidth=1.5)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=C1), name="熱流束 q = −k u_x(横の矢印)"))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=C2), name="温度の変化 u_t ∝ u_xx(縦の矢印)"))
        layout(fig, title="熱は高温から低温へ流れ, 曲がり具合 u_xx が温度変化を決める", xlab="x", ylab="u", height=360, yrange=(-0.3, 1.4))
        show(fig)
    st.caption("直線の温度分布では $u_{xx}=0$:熱は流れているが, 入る量と出る量が等しいので温度は変わらない(定常状態)。")

    st.subheader("3. 同じ形の方程式")
    st.markdown(r"""
| 方程式 | 現象 | 係数 |
|---|---|---|
| 波動方程式 $u_{tt}=c^2u_{xx}$ | 弦の振動 | $c=\sqrt{T/\rho}$ |
| | 棒の縦振動(機械) | $c=\sqrt{E/\rho}$($E$:ヤング率) |
| | 空気中の音 | $c=\sqrt{\gamma p/\rho}$(約 340 m/s) |
| | 損失のない送電線・同軸ケーブル(電気) | $c=1/\sqrt{LC}$ |
| 熱(拡散)方程式 $u_t=\kappa u_{xx}$ | 熱伝導 | $\kappa=k/(c_p\rho)$ |
| | 物質の拡散(化学:フィックの法則) | 拡散係数 $D$ |
| | 粘土層の圧密(都市:テルツァギの理論) | 圧密係数 $c_v$ |
""")
    st.caption("2次元・3次元では $u_{xx}$ が $\\Delta u=u_{xx}+u_{yy}+u_{zz}$(ラプラシアン)に変わる。時間変化しない状態では $\\Delta u=0$(ラプラス方程式, 次のページ)。")

    st.subheader("4. 歴史")
    st.markdown(r"""
| 年 | できごと |
|---|---|
| 1713 | テイラー(B. Taylor)が, 弦が一定時間に振動する回数(振動数)を求める問題を論じる |
| 1747 | ダランベール(J. d'Alembert)が弦の振動の**偏微分方程式**を導き, 解を $f(at+x)-f(at-x)$ の形で表す |
| その後 | オイラー(L. Euler)は, 初期の弦の形は任意の曲線でよいはずだと主張。ダニエル・ベルヌーイ(D. Bernoulli)は解を**三角関数の和**で表し, それが一般の解だと主張して論争になる(弦の振動論争) |
| 1807 | フーリエ(J. Fourier)が熱の伝わり方の論文をパリの学士院で発表。熱方程式を導き, 三角級数で解く。ラグランジュとラプラスは関数の三角級数展開に異論を唱えた |
| 1811 | 熱の伝導が学士院の懸賞問題となり, フーリエが受賞 |
| 1822 | 『熱の解析的理論(Théorie analytique de la chaleur)』出版。フーリエ級数が広まり, 関数とは何か・積分とは何かという研究(リーマン, ルベーグ)につながる |
""")
    st.markdown("弦の振動論争の「どんな関数が三角関数の和で表せるか」という問いは, フーリエの仕事を経て, "
                "この応用数学の ③ フーリエ解析の中心テーマになった。物質の拡散でも同じ方程式が成り立つことは, のちにフィック(A. Fick)が示した。")

    with st.expander("練習問題"):
        st.markdown("**問1** 張力 $T=100$ N, 線密度 $\\rho=1$ g/m の弦を伝わる波の速さを求めよ。長さ $0.5$ m の弦の基本振動数 $\\dfrac{c}{2L}$ はいくらか。")
        st.markdown("**問2** 熱方程式 $u_t=\\kappa u_{xx}$ に $u=e^{-\\kappa\\pi^2t}\\sin\\pi x$ が解であることを確かめよ。")
        st.markdown("**問3** 定常状態($u_t=0$)の棒の温度分布は1次関数になることを示せ。$u(0)=100$, $u(1)=20$ のとき $u(x)$ を求めよ。")
        if st.toggle("解答・解説を表示", key="pr_piwh"):
            st.markdown("**答1** $c=\\sqrt{100/0.001}=\\sqrt{10^5}\\approx316$ m/s。基本振動数は $316/(2\\times0.5)\\approx316$ Hz。")
            st.markdown("**答2** $u_t=-\\kappa\\pi^2e^{-\\kappa\\pi^2t}\\sin\\pi x$, $u_{xx}=-\\pi^2e^{-\\kappa\\pi^2t}\\sin\\pi x$ より $u_t=\\kappa u_{xx}$。")
            st.markdown("**答3** $u_{xx}=0$ を2回積分して $u=px+q$。$u(x)=100-80x$。")

    page_refs([("fourier1822", "熱方程式と三角級数の出発点(1822)。"),
               ("matano2024", "熱・波動方程式の導出とフーリエ級数。"),
               ("strauss2008", "第1章:偏微分方程式の導出(弦の振動, 拡散)。")])
    refs([("MacTutor History of Mathematics:Jean Baptiste Joseph Fourier", "https://mathshistory.st-andrews.ac.uk/Biographies/Fourier",
           "1807 年の論文, ラグランジュ・ラプラスの異論, 1811 年の懸賞, 1822 年の出版(英語)"),
          ("MacTutor History of Mathematics:Mathematicians and Music 3(R. C. Archibald)", "https://mathshistory.st-andrews.ac.uk/Extras/Archibald_music_3",
           "テイラー・ダランベール・オイラー・ダニエル・ベルヌーイと弦の振動(英語)")])


# ------------------------------------------------------------
# 2. ラプラス方程式
# ------------------------------------------------------------
BOUNDS = {"上の辺だけ 100℃": (1.0, 0.0, 0.0, 0.0), "上と左が 100℃": (1.0, 0.0, 1.0, 0.0),
          "上 100℃, 下 50℃": (1.0, 0.5, 0.0, 0.0), "4辺とも違う温度": (1.0, 0.25, 0.75, 0.0)}


def relax(top, bottom, left, right, n=41, iters=200):
    """各点を上下左右4点の平均で置き換える反復(ヤコビ法)。境界は固定"""
    u = np.zeros((n, n))
    u[-1, :], u[0, :], u[:, 0], u[:, -1] = top, bottom, left, right
    u[-1, 0], u[-1, -1] = (top + left) / 2, (top + right) / 2
    u[0, 0], u[0, -1] = (bottom + left) / 2, (bottom + right) / 2
    for _ in range(iters):
        u[1:-1, 1:-1] = (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2]) / 4
    return u


HARMONIC = [(r"x^2-y^2", r"z^2", "実部"), (r"xy", r"\frac{1}{2}z^2", "虚部"), (r"e^x\cos y", r"e^z", "実部"),
            (r"\log\sqrt{x^2+y^2}", r"\log z", "実部, 原点以外"), (r"\frac{x}{x^2+y^2}", r"\frac{1}{z}", "実部, 原点以外")]


def tab_laplace_intro():
    st.info("熱の流れが落ち着いた金属板, 電気の通り道の電位, 渦を巻かずに流れる水。まったく違う3つの現象を数式で追いかけると, "
            "どれも同じ1つの方程式にたどり着く。この節では, 現象から出発してその方程式に名前を与える。")

    st.subheader("1. 3つの現象を数式で追いかける")
    st.markdown("**(1) 熱が釣り合った金属板**:板(2次元)の温度は熱方程式 $u_t=\\kappa(u_{xx}+u_{yy})$ に従う(導入1. の2次元版)。"
                "十分に時間がたって温度が変わらなくなると $u_t=0$ なので")
    st.latex(r"\frac{\partial^2u}{\partial x^2}+\frac{\partial^2u}{\partial y^2}=0")
    st.markdown("**(2) 電荷のない空間の電位**:電場は電位 $\\phi$ の勾配で $\\boldsymbol E=-\\nabla\\phi$, "
                "ガウスの法則は $\\nabla\\cdot\\boldsymbol E=\\dfrac{\\rho}{\\varepsilon_0}$($\\rho$ は電荷密度)。合わせると "
                "$\\nabla\\cdot(\\nabla\\phi)=-\\dfrac{\\rho}{\\varepsilon_0}$(ポアソン方程式)。電荷のないところでは")
    st.latex(r"\frac{\partial^2\phi}{\partial x^2}+\frac{\partial^2\phi}{\partial y^2}+\frac{\partial^2\phi}{\partial z^2}=0")
    st.markdown("**(3) 渦を巻かない水の流れ**:水は縮まない($\\nabla\\cdot\\boldsymbol v=0$)。渦がなければ速度は $\\boldsymbol v=\\nabla\\phi$ と書ける(速度ポテンシャル)。合わせると")
    st.latex(r"\nabla\cdot(\nabla\phi)=\frac{\partial^2\phi}{\partial x^2}+\frac{\partial^2\phi}{\partial y^2}+\frac{\partial^2\phi}{\partial z^2}=0")
    st.markdown("熱・電気・水という**まったく違う現象なのに, 共通して同じ方程式**にたどり着いた。"
                "($\\nabla$, $\\nabla\\cdot$ は ① ベクトル解析の勾配と発散で, $\\nabla\\cdot\\nabla$ を $\\Delta$ と書く。)")

    st.subheader("2. 名前を与える:ラプラス方程式と調和関数")
    st.latex(r"\Delta u=\frac{\partial^2u}{\partial x^2}+\frac{\partial^2u}{\partial y^2}\ \left(+\frac{\partial^2u}{\partial z^2}\right)=0\qquad(\text{ラプラス方程式})")
    st.markdown("この方程式を**ラプラス方程式**といい, これをみたす関数を特別に**調和関数**と呼ぶ。")
    st.markdown("**なぜ「調和」か(イメージ)**:3つの現象を振り返ると, $\\Delta u=0$ が成り立つのは")
    st.markdown("- 熱の場合:板の内部に熱が**湧き出す**ところ(ヒーター)も**吸い込まれる**ところ(冷却)もない\n"
                "- 電気の場合:内部に電荷がない(ポアソン方程式の右辺が 0)\n"
                "- 水の場合:内部に水が湧き出す泉も, 吸い込む排水口もない")
    st.markdown("という状態である。どの点も**湧き出しも吸い込みもなく, 周りと釣り合っている**。"
                "空間全体のバランスが取れて「調和」している状態, と覚えるとよい。"
                "この「周りと釣り合っている」というイメージは, 次の 3. で「**各点の値は周りの平均**」という具体的な形になる。")
    st.caption("用語の由来:英語の harmonic は, 弦の振動の倍音(ハーモニクス)を表す三角関数の仲間である「球面調和関数(spherical harmonics)」"
               "(トムソンとテイトが 1867 年に命名)とつながっており, 球面調和関数は調和関数を球面に制限したものであることが名前の由来と考えられている。")
    st.markdown("**もう1つの出会い方(④ 複素関数)**:正則関数 $f(z)=u+iv$ の実部・虚部は, コーシー・リーマンの方程式 $u_x=v_y$, $u_y=-v_x$ から "
                "$u_{xx}+u_{yy}=v_{yx}-v_{xy}=0$ をみたし, 調和関数になる。")
    st.markdown("**調和関数の例**(どれも $u_{xx}+u_{yy}=0$ を確かめてみよう):")
    st.markdown(" / ".join(f"${u}$(${z}$ の{w})" for u, z, w in HARMONIC))

    st.subheader("3. 「周りの平均」としてのラプラス方程式")
    st.markdown("2. の「周りと釣り合っている」を数式で確かめる。格子の間隔 $h$ で $u_{xx}+u_{yy}$ を差分で近似すると")
    st.latex(r"\frac{u(x+h,y)+u(x-h,y)+u(x,y+h)+u(x,y-h)-4u(x,y)}{h^2}=0"
             r"\ \Longleftrightarrow\ u(x,y)=\frac{u(x+h,y)+u(x-h,y)+u(x,y+h)+u(x,y-h)}{4}")
    st.markdown("つまり**各点の値は上下左右の平均**(周りより高すぎも低すぎもしない = 釣り合っている)。これをくり返し計算すると(ヤコビ法), 境界の温度を固定した板の定常状態が求まる。"
                "反復は「時間を進める」ことにあたり, 熱方程式の解が定常状態に近づく様子と同じである。")
    c = st.columns([1, 2])
    bd = c[0].radio("境界の温度(100℃ を 1 とする)", list(BOUNDS), key="pli_b")
    it = c[0].select_slider("反復回数", [0, 10, 50, 200, 1000, 4000], 200, key="pli_it")
    U = relax(*BOUNDS[bd], iters=it)
    Uc = relax(*BOUNDS[bd], iters=it + 1)
    c[0].metric("中心の温度", f"{100 * U[20, 20]:.1f} ℃")
    c[0].metric("1回の反復での最大の変化", f"{100 * np.max(np.abs(Uc - U)):.2e} ℃")
    with c[1]:
        g = np.linspace(0, 1, 41)
        fig = go.Figure(go.Contour(x=g, y=g, z=100 * U, colorscale="Inferno", ncontours=20, zmin=0, zmax=100,
                                   contours=dict(coloring="heatmap", showlabels=True, labelfont=dict(color="white", size=10)),
                                   colorbar=dict(title="℃", thickness=12)))
        layout(fig, title=f"{it} 回反復したときの温度分布", xlab="x", ylab="y", height=420, equal=True, legend=False)
        show(fig)
    st.caption("反復を増やすと変化が小さくなり, 定常状態(ラプラス方程式の解)に落ち着く。"
               "**最大値の原理**:調和関数の最大値・最小値は境界でとる(内部に「熱い点」はできない)。")
    st.markdown("連続の場合も同じで, 調和関数は**平均値の性質**「円の中心の値 = 円周上の平均」をもつ(発展. ラプラス方程式 のページで確かめられる)。")

    st.subheader("4. 歴史")
    st.markdown(r"""
| 年 | できごと |
|---|---|
| 18世紀半ば | オイラーとダランベールの流体力学や複素関数の研究に, この方程式が現れる |
| 1780年代 | ラプラス(P.-S. Laplace)が回転楕円体(地球や惑星)の引力を研究し, 引力のポテンシャルがこの方程式をみたすことを示す。天体力学を通して広く知られるようになり, 「ラプラス方程式」と呼ばれる |
| 19世紀初め | ポアソン(S. D. Poisson)が, 物質の内部では $\Delta V=-4\pi\rho$ の形(ポアソン方程式)になることを示す |
| 1828 | グリーン(G. Green)が電気と磁気の論文で「ポテンシャル」という名前を与え, 面積分と体積分を結ぶ公式(グリーンの定理)を示す |
| 19世紀 | 境界の値から内部を決める問題(ディリクレ問題)が研究され, 解の存在を保証する「ディリクレの原理」をめぐる議論が, 変分法や関数解析の発展につながる |
""")
    st.markdown("ラプラス方程式は, 電気(静電場・電位), 機械(定常の熱伝導・弾性), 都市(地下水の浸透流・流線網), 化学(定常の拡散)のどれにも現れる。")

    with st.expander("練習問題"):
        st.markdown("**問1** $u=x^2-y^2$, $u=e^x\\cos y$ が調和関数であることを確かめよ。$u=x^2+y^2$ はどうか。")
        st.markdown("**問2** 1次元のラプラス方程式 $u^{\\prime\\prime}=0$ の解で, $u(0)=0$, $u(d)=V$ をみたすものを求めよ(平行板コンデンサーの電位)。")
        st.markdown("**問3** 格子点の値が上 $80$, 下 $20$, 左 $50$, 右 $50$ のとき, 中央の点の値を差分のラプラス方程式から求めよ。")
        if st.toggle("解答・解説を表示", key="pr_pila"):
            st.markdown("**答1** $u_{xx}+u_{yy}=2-2=0$, $e^x\\cos y-e^x\\cos y=0$。$x^2+y^2$ は $2+2=4\\ne0$ で調和関数でない。")
            st.markdown("**答2** $u=px+q$ で $u(0)=0$, $u(d)=V$ より $u=\\dfrac{V}{d}x$(電場の大きさ $\\dfrac{V}{d}$ は一定)。")
            st.markdown("**答3** 上下左右の平均 $\\dfrac{80+20+50+50}{4}=50$。")

    page_refs([("strauss2008", "第6章:ラプラス方程式, 最大値の原理, 平均値の性質。"),
               ("evans2010", "調和関数とポテンシャル論(発展)。")])
    refs([("Encyclopedia of Mathematics:Laplace equation", "https://encyclopediaofmath.org/wiki/Laplace_equation",
           "定義と歴史(オイラー・ダランベール, ラプラスの論文)(英語)"),
          ("W. W. Rouse Ball:Pierre Simon Laplace(A Short Account of the History of Mathematics)",
           "https://www.maths.tcd.ie/pub/HistMath/People/Laplace/RouseBall/RB_Laplace.html", "回転楕円体の引力とポテンシャル(英語)"),
          ("Wikipedia:An Essay on the Application of Mathematical Analysis to the Theories of Electricity and Magnetism",
           "https://en.wikipedia.org/wiki/An_Essay_on_the_Application_of_Mathematical_Analysis_to_the_Theories_of_Electricity_and_Magnetism",
           "グリーンの 1828 年の論文:ポテンシャル関数とグリーンの定理(英語)"),
          ("Wikipedia:Spherical harmonics", "https://en.wikipedia.org/wiki/Spherical_harmonics",
           "球面調和関数の歴史(1867 年のトムソンとテイトによる命名)と, 調和多項式との関係(英語)")])
