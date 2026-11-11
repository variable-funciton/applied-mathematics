"""⑤ 偏微分方程式 発展:オイラー方程式, ナヴィエ・ストークス方程式, シュレディンガー方程式, ドリフト拡散方程式

4つとも「保存則(連続の式)+ 流れの法則」から導く。解くのは1次元の例だけ。
史実・書誌は 2026 年 11 月に確認(references.py の原論文, Bistafa "200 Years of the Navier-Stokes Equation" など)。
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.special import erfc

from common import C1, C2, C3, C4, GRAY, animate, layout, show
from topics.fourier_apps import refs
from topics.fourier_kernels import page_refs

PI = np.pi
HBAR, ME, EV, HPL = 1.054571817e-34, 9.1093837015e-31, 1.602176634e-19, 6.62607015e-34
EPS0, QE = 8.8541878128e-12, 1.602176634e-19


def overview(cur):
    with st.expander("4つの方程式の関係(保存則 + 流れの法則)", expanded=False):
        st.markdown("どれも, 量の**保存**(連続の式 $\\dfrac{\\partial\\rho}{\\partial t}+\\nabla\\cdot\\boldsymbol J=0$)と, "
                    "流れ $\\boldsymbol J$ を決める**法則**の組から導かれる。導入のページの熱方程式(エネルギーの保存 + フーリエの法則)と同じ考え方である。")
        rows = [("オイラー方程式", "質量・運動量", "圧力だけが働く(粘性なし)", "渦なしならラプラス方程式, ベルヌーイの定理"),
                ("ナヴィエ・ストークス方程式", "質量・運動量", "圧力 + 粘性(ニュートンの粘性法則)", "1次元では熱方程式と同じ形(運動量の拡散)"),
                ("シュレディンガー方程式", "確率 |ψ|²", "確率の流れ", "熱方程式の係数を虚数にした形, 変数分離・フーリエ級数"),
                ("ドリフト拡散方程式", "電子・正孔の数", "ドリフト(電場)+ 拡散(濃度勾配)", "拡散方程式 + ポアソン方程式")]
        st.markdown("| 方程式 | 保存するもの | 流れの法則 | これまでのページとのつながり |\n|---|---|---|---|\n" +
                    "\n".join(f"| {'**' + a + '**' if a == cur else a} | {b} | {c} | {d} |" for a, b, c, d in rows))
        st.caption("流体の2つは機械・都市・化学, シュレディンガー方程式は化学・電気電子, ドリフト拡散方程式は電気電子で使われる(学科別の応用のページ)。")


# ============================================================
# 1. オイラー方程式
# ============================================================
def cylinder_flow(n=161, a=1.0, U=1.0):
    x = np.linspace(-3, 3, n)
    X, Y = np.meshgrid(x, x)
    R2 = X ** 2 + Y ** 2
    psi = U * Y * (1 - a * a / np.where(R2 < a * a, np.nan, R2))
    return x, psi


def tab_euler():
    st.info("粘り気(粘性)を無視した流体の運動方程式が**オイラー方程式**である。工学で最もよく使う**ベルヌーイの定理**はここから出てくる。"
            "一方で, 粘性を無視したために説明できない現象(ダランベールのパラドックス)もあり, 次のナヴィエ・ストークス方程式につながる。")
    overview("オイラー方程式")

    st.subheader("1. 導出:流体の小さな部分にニュートンの法則")
    st.markdown("管の中を $x$ 方向に流れる流体を考え, 速度を $u(x,t)$, 圧力を $p(x,t)$, 密度を $\\rho$ とする。"
                "区間 $[x,\\,x+\\Delta x]$(断面積 $A$)の流体には, 左から $p(x)A$, 右から $p(x+\\Delta x)A$ の力が働くので")
    st.latex(r"\rho A\Delta x\,\frac{Du}{Dt}=p(x,t)A-p(x+\Delta x,t)A\ \Longrightarrow\ \rho\left(\frac{\partial u}{\partial t}+u\frac{\partial u}{\partial x}\right)=-\frac{\partial p}{\partial x}")
    st.markdown("左辺の $\\dfrac{Du}{Dt}=u_t+uu_x$ は, **流体の粒子と一緒に動きながら見た加速度**(物質微分)である。"
                "粒子は時間 $\\Delta t$ の間に $u\\Delta t$ だけ移動するので, $u(x+u\\Delta t,\\,t+\\Delta t)-u(x,t)\\approx(u_t+uu_x)\\Delta t$ となる。"
                "3次元・重力ありでは, 質量の保存(連続の式)と合わせて")
    st.latex(r"\rho\left(\frac{\partial\boldsymbol v}{\partial t}+(\boldsymbol v\cdot\nabla)\boldsymbol v\right)=-\nabla p+\rho\boldsymbol g,\qquad \frac{\partial\rho}{\partial t}+\nabla\cdot(\rho\boldsymbol v)=0")
    st.caption("$(\\boldsymbol v\\cdot\\nabla)\\boldsymbol v$ は速度どうしの積を含むので, 方程式は**非線形**になる(これまでの熱・波動方程式との大きな違い)。")

    st.subheader("2. ベルヌーイの定理")
    st.markdown("流れが時間変化しない(定常)とき, 1次元の式 $\\rho uu_x=-p_x$ は $\\dfrac{d}{dx}\\left(\\dfrac12\\rho u^2+p\\right)=0$ と書ける。重力も入れると, 流れに沿って")
    st.latex(r"p+\frac12\rho v^2+\rho gz=\text{一定}\qquad(\text{ベルヌーイの定理})")
    st.markdown("「速く流れるところほど圧力が低い」。質量の保存 $A_1v_1=A_2v_2$(連続の式)と合わせると, 管の細いところでは流速が上がり, 圧力が下がる。"
                "これを使って流量を測るのが**ベンチュリ管**, 流速を測るのが**ピトー管**(航空機の速度計)である。"
                "水槽の穴から出る水の速さ $v=\\sqrt{2gh}$(**トリチェリの定理**)もこの定理から出る。")
    c = st.columns(3)
    v1 = c[0].slider("太い部分の流速 v₁ [m/s]", 0.5, 5.0, 1.0, 0.5, key="eu_v1")
    ratio = c[1].slider("断面積の比 A₁/A₂", 1.0, 5.0, 2.0, 0.5, key="eu_r")
    rho = 1000.0
    v2 = v1 * ratio
    dp = 0.5 * rho * (v2 ** 2 - v1 ** 2)
    c[2].metric("細い部分の流速 v₂", f"{v2:.2f} m/s")
    c[2].metric("圧力差 p₁ − p₂(水)", f"{dp / 1000:.2f} kPa")
    c[2].metric("水柱の高さの差 Δp/(ρg)", f"{100 * dp / (rho * 9.8):.1f} cm")
    xx = np.linspace(0, 10, 401)
    rad = 1 - (1 - 1 / np.sqrt(ratio)) * np.exp(-((xx - 5) / 1.2) ** 2)      # 管の半径(断面積 ∝ 半径²)
    vv = v1 / rad ** 2
    pp = 0.5 * rho * (v1 ** 2 - vv ** 2) / 1000
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xx, y=rad, mode="lines", line=dict(color=GRAY, width=3), name="管の壁(半径)"))
    fig.add_trace(go.Scatter(x=xx, y=-rad, mode="lines", line=dict(color=GRAY, width=3), showlegend=False))
    fig.add_trace(go.Scatter(x=xx, y=vv / vv.max() * 0.9, mode="lines", line=dict(color=C1, width=2.5), name="流速(相対値)"))
    fig.add_trace(go.Scatter(x=xx, y=pp / max(abs(pp.min()), 1e-9) * 0.9, mode="lines", line=dict(color=C3, width=2.5, dash="dash"), name="圧力 p − p₁(相対値)"))
    layout(fig, title="ベンチュリ管:細いところで速く, 圧力が低い", xlab="管に沿った位置", height=300, yrange=(-1.1, 1.1))
    show(fig)

    st.subheader("3. 渦のない流れとラプラス方程式, ダランベールのパラドックス")
    st.markdown("縮まない流体で渦がなければ, 速度は $\\boldsymbol v=\\nabla\\phi$ と書けて $\\Delta\\phi=0$(導入2. のラプラス方程式)。"
                "2次元では複素関数が使え(④ 複素関数), 一様な流れの中に置いた半径 $a$ の円柱のまわりの流れは, "
                "複素速度ポテンシャル $w(z)=U\\left(z+\\dfrac{a^2}{z}\\right)$ の虚部(流れ関数)の等高線が流線になる。")
    x, psi = cylinder_flow()
    th = np.linspace(0, 2 * PI, 361)
    c = st.columns(2)
    with c[0]:
        fig = go.Figure(go.Contour(x=x, y=x, z=psi, contours=dict(coloring="lines", start=-2.6, end=2.6, size=0.2),
                                   line=dict(width=1.5), colorscale=[[0, C1], [1, C1]], showscale=False))
        fig.add_trace(go.Scatter(x=np.cos(th), y=np.sin(th), fill="toself", mode="lines", line=dict(color="black"), fillcolor="rgba(120,120,120,0.6)", name="円柱"))
        layout(fig, title="円柱のまわりの流線(粘性なし)", xlab="x", ylab="y", height=360, equal=True, legend=False)
        show(fig)
    with c[1]:
        cp = 1 - 4 * np.sin(th) ** 2
        fig = go.Figure(go.Scatter(x=np.degrees(th), y=cp, mode="lines", line=dict(color=C3, width=2.5), name="圧力係数"))
        layout(fig, title="円柱の表面の圧力 C_p = 1 − 4 sin²θ", xlab="角度 θ [度](0° が後ろ, 180° が前)", ylab="C_p", height=360)
        show(fig)
    st.markdown("圧力は円柱の前($\\theta=180^\\circ$)と後ろ($\\theta=0^\\circ$)で**まったく同じ**なので, 力を合計すると抗力は 0 になってしまう。"
                "実際には円柱は流されるので矛盾である(**ダランベールのパラドックス**, 1752 年)。"
                "原因は粘性を無視したことにあり, 物体の表面近くの薄い層(境界層)の粘性が抗力を生む。これがナヴィエ・ストークス方程式が必要になる理由である。")

    st.subheader("4. 歴史")
    st.markdown(r"""
| 年 | できごと |
|---|---|
| 1738 | ダニエル・ベルヌーイ(D. Bernoulli)が『流体力学(Hydrodynamica)』を出版し, 圧力と流速の関係を論じる |
| 1752 | ダランベール(J. d'Alembert)が, 理想流体では物体に抗力が働かないという結論(パラドックス)を示す |
| 1755 | オイラー(L. Euler)が「流体の運動の一般原理」を執筆(1757 年にベルリンの学士院の紀要に掲載)。流体の運動を偏微分方程式で表し, オイラー方程式と連続の式を与える |
| 1904 | プラントル(L. Prandtl)が境界層の理論を発表し, 粘性の効果が表面近くの薄い層に集中することを示して, パラドックスを解決する道を開く |
""")
    with st.expander("練習問題"):
        st.markdown("**問1** 断面積 $A_1$ の管から $A_2=\\dfrac{A_1}{3}$ の管へ水が流れる。$v_1=1$ m/s のとき $v_2$ と圧力差 $p_1-p_2$ を求めよ($\\rho=1000$ kg/m³)。")
        st.markdown("**問2** 水面から深さ $h=5$ m の穴から出る水の速さを求めよ($g=9.8$ m/s²)。")
        st.markdown("**問3** 1次元のオイラー方程式で, 定常なとき $\\rho uu_x=-p_x$ から $\\dfrac12\\rho u^2+p$ が一定であることを示せ。")
        if st.toggle("解答・解説を表示", key="pr_euler"):
            st.markdown("**答1** $v_2=3$ m/s, $p_1-p_2=\\frac12\\times1000\\times(9-1)=4000$ Pa $=4$ kPa。")
            st.markdown("**答2** $v=\\sqrt{2\\times9.8\\times5}=\\sqrt{98}\\approx9.9$ m/s。")
            st.markdown("**答3** $uu_x=\\left(\\frac12u^2\\right)_x$ より $\\left(\\frac12\\rho u^2+p\\right)_x=0$。")
    page_refs([("euler1757", "オイラー方程式の原論文(英訳は arXiv:0802.2383)。"), ("batchelor2000", "第6章:非粘性流体, ベルヌーイの定理, 円柱まわりの流れ。")])
    refs([("Translation of Leonhard Euler's: General Principles of the Motion of Fluids(arXiv:0802.2383)", "https://arxiv.org/abs/0802.2383", "オイラーの原論文の英訳"),
          ("Euler Archive(University of the Pacific):E226", "https://scholarlycommons.pacific.edu/euler-works/226", "原論文の書誌と原文")])


# ============================================================
# 2. ナヴィエ・ストークス方程式
# ============================================================
def couette_poiseuille(y, h, U, P_over_mu):
    """平行平板の間の定常流:u = U y/h + (P/(2μ)) y(h − y),  P = −dp/dx"""
    return U * y / h + 0.5 * P_over_mu * y * (h - y)


def stokes_first(y, t, U, nu):
    """急に動き出した板の上の流れ:u = U erfc(y/(2√(νt)))"""
    return U * erfc(y / (2 * np.sqrt(nu * t)))


def tab_navier_stokes():
    st.info("オイラー方程式に**粘性**(流体どうしの摩擦)を加えたものが**ナヴィエ・ストークス方程式**である。"
            "水・空気・油など身の回りのほとんどの流れを表すが, 非線形で一般には解けず, 数学の大きな未解決問題でもある。")
    overview("ナヴィエ・ストークス方程式")

    st.subheader("1. 導出:粘性の力を加える")
    st.markdown("**ニュートンの粘性法則**:層になって流れる流体では, 速さの違う層どうしが引きずり合う。"
                "単位面積あたりの力(せん断応力)は速度の勾配に比例する:$\\tau=\\mu\\dfrac{\\partial u}{\\partial y}$($\\mu$:粘性係数)。")
    st.markdown("$x$ 方向に流れ, 速度が高さ $y$ だけで変わる流れ $u(y,t)$ を考える。厚さ $\\Delta y$ の層には, 上の面で $\\tau(y+\\Delta y)$, 下の面で $-\\tau(y)$ の力が働くので, "
                "単位面積あたり")
    st.latex(r"\rho\Delta y\frac{\partial u}{\partial t}=-\frac{\partial p}{\partial x}\Delta y+\mu u_y(y+\Delta y)-\mu u_y(y)\ \Longrightarrow\ "
             r"\rho\frac{\partial u}{\partial t}=-\frac{\partial p}{\partial x}+\mu\frac{\partial^2u}{\partial y^2}")
    st.markdown("圧力の勾配がなければ $u_t=\\nu u_{yy}$($\\nu=\\mu/\\rho$:動粘性係数)で, **熱方程式と同じ形**である。"
                "粘性は「運動量の拡散」であり, $\\nu$ は運動量の拡散係数にあたる。一般の3次元の流れでは")
    st.latex(r"\rho\left(\frac{\partial\boldsymbol v}{\partial t}+(\boldsymbol v\cdot\nabla)\boldsymbol v\right)=-\nabla p+\mu\Delta\boldsymbol v+\rho\boldsymbol g,\qquad \nabla\cdot\boldsymbol v=0")
    st.caption("縮まない流体の場合。$\\mu=0$ とするとオイラー方程式に戻る。")

    st.subheader("2. 厳密に解ける例:平行な板の間の流れ")
    st.markdown("2枚の板(間隔 $h$)の間で, 下の板は止まり, 上の板が速さ $U$ で動き, 圧力の勾配 $P=-\\dfrac{dp}{dx}$ がある定常な流れでは $\\mu u^{\\prime\\prime}=-P$。"
                "境界条件 $u(0)=0$, $u(h)=U$ で2回積分すると")
    st.latex(r"u(y)=\underbrace{\frac{U}{h}y}_{\text{クエット流れ}}+\underbrace{\frac{P}{2\mu}y(h-y)}_{\text{ポアズイユ流れ(放物線)}}")
    c = st.columns([1, 2])
    U = c[0].slider("上の板の速さ U [m/s]", -1.0, 1.0, 0.0, 0.1, key="ns_U")
    Pm = c[0].slider("P/μ(圧力で押す強さ)[1/(m·s)]", -40.0, 40.0, 16.0, 4.0, key="ns_P")
    h = 0.5
    y = np.linspace(0, h, 201)
    u = couette_poiseuille(y, h, U, Pm)
    c[0].metric("平均流速", f"{float(np.mean(u)):.3f} m/s")
    with c[1]:
        fig = go.Figure(go.Scatter(x=u, y=y, mode="lines", line=dict(color=C1, width=3), name="u(y)"))
        for yi in np.linspace(0.05, h - 0.05, 9):
            ui = float(couette_poiseuille(np.array([yi]), h, U, Pm)[0])
            fig.add_annotation(x=ui, y=yi, ax=0, ay=yi, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowcolor=C2)
        fig.add_hline(y=0, line=dict(color="black", width=4))
        fig.add_hline(y=h, line=dict(color="black", width=4))
        layout(fig, title="速度分布(矢印は各高さの流速)", xlab="流速 u [m/s]", ylab="高さ y [m]", height=340, xrange=(-1.2, 1.6))
        show(fig)
    st.caption("$P=0$ なら直線(クエット流れ:軸受の潤滑油), $U=0$ なら放物線(ポアズイユ流れ:管の中の流れ)。円管では "
               "$u(r)=\\dfrac{P}{4\\mu}(R^2-r^2)$, 流量 $Q=\\dfrac{\\pi PR^4}{8\\mu}$(**ハーゲン・ポアズイユの法則**)となり, 流量は半径の4乗に比例する。")

    st.subheader("3. 粘性は運動量の拡散:急に動き出した板")
    st.markdown("静止した流体の中で, 板が時刻 0 に急に速さ $U$ で動き出す。$u_t=\\nu u_{yy}$, $u(0,t)=U$ の解は "
                "$u=U\\,\\mathrm{erfc}\\!\\left(\\dfrac{y}{2\\sqrt{\\nu t}}\\right)$ で, 化学の応用の拡散の解と同じ形である。"
                "動きは $\\sqrt{\\nu t}$ 程度の厚さまで伝わる(水 $\\nu\\approx1\\times10^{-6}$ m²/s, 空気 $\\nu\\approx1.5\\times10^{-5}$ m²/s)。")
    yy = np.linspace(0, 0.02, 301)
    ts = np.logspace(-1, 2, 40)
    frames = [[stokes_first(yy, t, 1.0, 1e-6), stokes_first(yy, t, 1.0, 1.5e-5)] for t in ts]
    fig = animate(1000 * yy, frames, ["水", "空気"], [f"{t:.3g} s" for t in ts], colors=[C1, C3],
                  title="板の動きが流体に伝わる様子(縦軸 u/U, 横軸 板からの距離)", xlab="板からの距離 y [mm]", ylab="u/U",
                  yrange=(-0.02, 1.05), height=380, duration=120, label_prefix="t = ")
    show(fig)

    st.subheader("4. レイノルズ数:層流と乱流")
    st.markdown("慣性の力 $\\rho(\\boldsymbol v\\cdot\\nabla)\\boldsymbol v$ と粘性の力 $\\mu\\Delta\\boldsymbol v$ の大きさの比が**レイノルズ数**")
    st.latex(r"Re=\frac{\rho VD}{\mu}=\frac{VD}{\nu}\qquad(V:\text{代表的な速さ},\ D:\text{代表的な長さ})")
    st.markdown("で, 円管の流れでは $Re$ がおよそ 2300 より小さいと整然とした**層流**, 大きいと不規則な**乱流**になる(レイノルズの実験, 1883 年)。")
    c = st.columns(3)
    V = c[0].select_slider("流速 V [m/s]", [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0], 0.5, key="ns_V")
    D = c[1].select_slider("管の直径 D [mm]", [1, 5, 10, 25, 50, 100], 25, key="ns_D")
    fl = c[2].radio("流体", ["水(ν = 1.0×10⁻⁶ m²/s)", "空気(ν = 1.5×10⁻⁵ m²/s)", "油(ν = 1.0×10⁻⁴ m²/s)"], key="ns_fl")
    nu = {"水": 1e-6, "空気": 1.5e-5, "油": 1e-4}[fl.split("(")[0]]
    Re = V * D / 1000 / nu
    st.metric("レイノルズ数 Re", f"{Re:,.0f}", delta="層流" if Re < 2300 else "乱流(または遷移域)", delta_color="off")

    st.subheader("5. 歴史と未解決問題")
    st.markdown(r"""
| 年 | できごと |
|---|---|
| 1822 | ナヴィエ(C.-L. Navier)が, 分子の間の力を考えて粘性流体の方程式を導き, パリの学士院で発表(論文は 1827 年刊) |
| 1845 | ストークス(G. G. Stokes)が, 連続体の応力の考え方から同じ方程式を導く。ポアソンやサン=ブナンも別の方法で導いた |
| 1839〜1840年代 | ハーゲン(G. Hagen)とポアズイユ(J. Poiseuille)が, 細い管の中の流量と圧力の関係を実験で見いだす |
| 1883 | レイノルズ(O. Reynolds)が着色した水の実験で層流と乱流の遷移を調べ, レイノルズ数の考え方を示す |
| 1904 | プラントルが境界層の理論を発表 |
| 2000 | クレイ数学研究所が, 3次元のナヴィエ・ストークス方程式のなめらかな解がいつまでも存在するかを**ミレニアム懸賞問題**の1つに選ぶ(未解決) |
""")
    with st.expander("練習問題"):
        st.markdown("**問1** $\\mu u^{\\prime\\prime}=-P$, $u(0)=u(h)=0$ を解き, 最大流速と平均流速の比を求めよ。")
        st.markdown("**問2** ハーゲン・ポアズイユの法則で, 管の半径を2倍にすると, 同じ圧力勾配での流量は何倍になるか。")
        st.markdown("**問3** 直径 $2$ cm の水道管を水が $1$ m/s で流れるときのレイノルズ数を求め, 層流か乱流か判定せよ。")
        if st.toggle("解答・解説を表示", key="pr_ns"):
            st.markdown("**答1** $u=\\dfrac{P}{2\\mu}y(h-y)$。最大は $y=\\frac h2$ で $\\dfrac{Ph^2}{8\\mu}$, 平均は $\\dfrac1h\\int_0^hu\\,dy=\\dfrac{Ph^2}{12\\mu}$。比は $3:2$(平均は最大の $\\frac23$)。")
            st.markdown("**答2** $Q\\propto R^4$ なので $16$ 倍。")
            st.markdown("**答3** $Re=\\dfrac{1\\times0.02}{1.0\\times10^{-6}}=20000$。2300 より大きいので乱流。")
    page_refs([("navier1827", ""), ("stokes1880", ""), ("reynolds1883", ""), ("batchelor2000", "第3〜5章:粘性流体, 厳密解, レイノルズ数。")])
    refs([("S. R. Bistafa:200 Years of the Navier-Stokes Equation(arXiv:2401.13669)", "https://arxiv.org/abs/2401.13669",
           "ナヴィエの 1822 年の論文からストークスまでの歴史(英語)"),
          ("Wikipedia:Navier–Stokes existence and smoothness", "https://en.wikipedia.org/wiki/Navier%E2%80%93Stokes_existence_and_smoothness",
           "ミレニアム懸賞問題の内容(英語)"),
          ("Wikipedia:Hagen–Poiseuille equation", "https://en.wikipedia.org/wiki/Hagen%E2%80%93Poiseuille_equation", "管の中の層流と圧力損失(英語)")])


# ============================================================
# 3. シュレディンガー方程式
# ============================================================
def box_E(n, L_nm):
    """箱の中の電子のエネルギー [eV]:E_n = n² h²/(8 m L²)"""
    L = L_nm * 1e-9
    return n * n * HPL ** 2 / (8 * ME * L * L) / EV


def tunnel_T(E, V0, a_nm):
    """高さ V0 [eV], 幅 a [nm] の長方形の障壁を電子が通り抜ける確率(厳密解)"""
    E = np.asarray(E, dtype=float)
    a = a_nm * 1e-9
    out = np.empty_like(E)
    lo = E < V0
    kap = np.sqrt(2 * ME * (V0 - E[lo]) * EV) / HBAR
    out[lo] = 1 / (1 + V0 ** 2 * np.sinh(kap * a) ** 2 / (4 * E[lo] * (V0 - E[lo])))
    hi = E > V0
    k2 = np.sqrt(2 * ME * (E[hi] - V0) * EV) / HBAR
    out[hi] = 1 / (1 + V0 ** 2 * np.sin(k2 * a) ** 2 / (4 * E[hi] * (E[hi] - V0)))
    eq = ~(lo | hi)
    out[eq] = 1 / (1 + ME * a * a * V0 * EV / (2 * HBAR ** 2))
    return out


def tab_schrodinger():
    st.info("電子のようなとても小さな粒子は, 粒子であると同時に**波**の性質をもつ。その波(波動関数 $\\psi$)の従う方程式が**シュレディンガー方程式**である。"
            "半導体・レーザー・化学結合など, 現代の電気電子・化学の基礎になっている。")
    overview("シュレディンガー方程式")

    st.subheader("1. 導出(発見的な説明)")
    st.markdown("光の研究から, 粒子の運動量 $p$ とエネルギー $E$ は波の波数 $k$ と角振動数 $\\omega$ で $p=\\hbar k$, $E=\\hbar\\omega$ と表される"
                "(ド・ブロイの関係, $\\hbar=h/2\\pi$:プランク定数)。平面波 $\\psi=e^{i(kx-\\omega t)}$ では")
    st.latex(r"i\hbar\frac{\partial\psi}{\partial t}=\hbar\omega\,\psi=E\psi,\qquad -\frac{\hbar^2}{2m}\frac{\partial^2\psi}{\partial x^2}=\frac{\hbar^2k^2}{2m}\psi=\frac{p^2}{2m}\psi")
    st.markdown("エネルギーの式「$E=$ 運動エネルギー $\\dfrac{p^2}{2m}$ + 位置エネルギー $V(x)$」にこれを当てはめると")
    st.latex(r"i\hbar\frac{\partial\psi}{\partial t}=-\frac{\hbar^2}{2m}\frac{\partial^2\psi}{\partial x^2}+V(x)\psi\qquad(\text{シュレディンガー方程式})")
    st.markdown("**熱方程式との比較**:$V=0$ のとき $\\psi_t=\\dfrac{i\\hbar}{2m}\\psi_{xx}$ で, 熱方程式 $u_t=\\kappa u_{xx}$ の係数 $\\kappa$ を**虚数** $\\dfrac{i\\hbar}{2m}$ にした形。"
                "そのため熱のように「ならされて消える」のではなく, 波として振動しながら広がる。"
                "$|\\psi(x,t)|^2$ は, 粒子が位置 $x$ に見つかる**確率の密度**を表す(ボルンの確率解釈)。")

    st.subheader("2. 箱の中の粒子:変数分離とエネルギーの量子化")
    st.markdown("長さ $L$ の箱の中($0<x<L$, $V=0$)に閉じこめた電子を考える。箱の外には出られないので $\\psi(0)=\\psi(L)=0$。"
                "変数分離 $\\psi=\\varphi(x)e^{-iEt/\\hbar}$(⑤ 0. の方法)で, $-\\dfrac{\\hbar^2}{2m}\\varphi^{\\prime\\prime}=E\\varphi$ となる。"
                "これは熱方程式の橋渡しで解いた $X^{\\prime\\prime}=-\\mu X$, $X(0)=X(L)=0$ と同じで")
    st.latex(r"\varphi_n(x)=\sqrt{\frac{2}{L}}\sin\frac{n\pi x}{L},\qquad E_n=\frac{n^2\pi^2\hbar^2}{2mL^2}=\frac{n^2h^2}{8mL^2}\qquad(n=1,2,3,\dots)")
    st.markdown("境界条件のために, エネルギーは飛び飛びの値しかとれない(**量子化**)。$\\sqrt{2/L}$ は $\\int_0^L|\\varphi|^2dx=1$(どこかには必ずいる)となるように決めた定数。")
    c = st.columns([1, 2])
    n = c[0].slider("量子数 n", 1, 6, 1, key="qm_n")
    L = c[0].select_slider("箱の長さ L [nm]", [0.2, 0.5, 1.0, 2.0, 5.0], 1.0, key="qm_L")
    c[0].metric(f"E_{n}", f"{box_E(n, L):.3g} eV")
    c[0].metric("E₂ − E₁(光を出すエネルギー)", f"{box_E(2, L) - box_E(1, L):.3g} eV")
    xx = np.linspace(0, L, 401)
    ph = np.sqrt(2 / L) * np.sin(n * PI * xx / L)
    with c[1]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xx, y=ph, mode="lines", line=dict(color=C1, width=2.5), name="φₙ(x)"))
        fig.add_trace(go.Scatter(x=xx, y=ph ** 2, mode="lines", line=dict(color=C3, width=2.5), fill="tozeroy", name="|φₙ(x)|²(見つかる確率の密度)"))
        fig.add_vline(x=0, line=dict(color="black", width=4))
        fig.add_vline(x=L, line=dict(color="black", width=4))
        layout(fig, title=f"n = {n} の状態", xlab="x [nm]", height=320)
        show(fig)
    st.caption("箱が小さいほどエネルギーの間隔は大きい($E_n\\propto1/L^2$)。量子ドットの発光の色が大きさで変わるのはこのためである。")
    st.markdown("**2つの状態の重ね合わせ**:$\\psi=\\frac{1}{\\sqrt2}\\left(\\varphi_1e^{-iE_1t/\\hbar}+\\varphi_2e^{-iE_2t/\\hbar}\\right)$ では, "
                "確率の密度 $|\\psi|^2$ が周期 $\\dfrac{h}{E_2-E_1}$ で左右に揺れる(▶ 再生)。")
    tt = np.linspace(0, 1, 41)
    x1 = np.linspace(0, 1, 301)
    p1, p2 = np.sqrt(2) * np.sin(PI * x1), np.sqrt(2) * np.sin(2 * PI * x1)
    frames = [[np.abs((p1 + p2 * np.exp(-2j * PI * t)) / np.sqrt(2)) ** 2] for t in tt]
    fig = animate(x1, frames, ["|ψ(x, t)|²"], [f"{t:.2f}" for t in tt], colors=[C3], title="重ね合わせた状態の確率の密度(時間は周期を 1 とした)",
                  xlab="x/L", ylab="|ψ|²(L = 1)", yrange=(0, 3.2), height=360, duration=80, label_prefix="t/周期 = ")
    show(fig)

    st.subheader("3. トンネル効果")
    st.markdown("高さ $V_0$, 幅 $a$ の壁(障壁)に, $V_0$ より小さいエネルギー $E$ の電子が当たる。古典力学では跳ね返されるだけだが, "
                "シュレディンガー方程式では壁の中で $\\psi$ が $e^{-\\kappa x}$($\\kappa=\\sqrt{2m(V_0-E)}/\\hbar$)と減りながらしみこみ, "
                "**一部が通り抜ける**(トンネル効果)。通り抜ける確率は障壁が厚いほど指数関数的に小さくなる($\\approx e^{-2\\kappa a}$)。")
    c = st.columns(2)
    V0 = c[0].slider("障壁の高さ V₀ [eV]", 0.5, 5.0, 2.0, 0.5, key="qm_V0")
    a = c[1].select_slider("障壁の幅 a [nm]", [0.1, 0.2, 0.5, 1.0, 2.0], 0.5, key="qm_a")
    E = np.linspace(0.01, 3 * V0, 600)
    T = tunnel_T(E, V0, a)
    fig = go.Figure(go.Scatter(x=E, y=T, mode="lines", line=dict(color=C1, width=2.5), name="通り抜ける確率 T(E)"))
    fig.add_vline(x=V0, line=dict(color=GRAY, dash="dash"))
    fig.update_yaxes(type="log", range=[-12, 0.1])
    layout(fig, title=f"V₀ = {V0} eV, a = {a} nm の障壁", xlab="電子のエネルギー E [eV](破線が V₀)", ylab="T(対数目盛)", height=320)
    show(fig)
    st.caption("$E<V_0$(破線の左)でも $T$ は 0 でない。幅を 2 倍にすると $T$ は大きく下がる。フラッシュメモリやトンネルダイオード, 走査型トンネル顕微鏡に使われる。")

    st.subheader("4. 歴史")
    st.markdown(r"""
| 年 | できごと |
|---|---|
| 1900 | プランク(M. Planck)が, 光のエネルギーが $h\nu$ のかたまりでやりとりされると考えて熱放射を説明 |
| 1905 | アインシュタインが光量子の考えで光電効果を説明 |
| 1913 | ボーア(N. Bohr)が水素原子のエネルギー準位の模型を発表 |
| 1924 | ド・ブロイ(L. de Broglie)が, 電子などの粒子も波の性質をもつと提唱 |
| 1925 | ハイゼンベルク(W. Heisenberg)らが行列力学を作る |
| 1926 | シュレディンガー(E. Schrödinger)が波動方程式を発表し, 水素原子のエネルギー準位を固有値問題として導く。同じ年, ボルン(M. Born)が $\lvert\psi\rvert^2$ を確率と解釈 |
""")
    with st.expander("練習問題"):
        st.markdown("**問1** 箱の中の粒子で $E_3$ は $E_1$ の何倍か。箱の長さを2倍にすると $E_1$ は何倍になるか。")
        st.markdown("**問2** $\\int_0^LA^2\\sin^2\\dfrac{\\pi x}{L}dx=1$ となる正の定数 $A$ を求めよ。")
        st.markdown("**問3** $\\psi=e^{i(kx-\\omega t)}$ が $V=0$ のシュレディンガー方程式をみたすための $\\omega$ と $k$ の関係を求めよ。")
        if st.toggle("解答・解説を表示", key="pr_qm"):
            st.markdown("**答1** $E_n\\propto n^2$ より $9$ 倍。$E\\propto1/L^2$ より $\\frac14$ 倍。")
            st.markdown("**答2** $\\int_0^L\\sin^2\\frac{\\pi x}{L}dx=\\frac L2$ より $A^2\\frac L2=1$, $A=\\sqrt{\\frac2L}$。")
            st.markdown("**答3** $i\\hbar(-i\\omega)=\\frac{\\hbar^2k^2}{2m}$ より $\\omega=\\frac{\\hbar k^2}{2m}$。")
    page_refs([("schrodinger1926", ""), ("born1926", ""), ("griffiths2018", "第1〜2章:シュレディンガー方程式, 箱の中の粒子, トンネル効果。")])


# ============================================================
# 4. ドリフト拡散方程式
# ============================================================
VT300 = 0.02585      # 300 K の kT/q [V]
NI_SI = 1.0e16       # シリコンの真性キャリア密度 [m⁻³](1.0×10¹⁰ cm⁻³)
EPS_SI = 11.7 * EPS0


def pn_junction(NA, ND, V=0.0, n=601):
    """階段接合の空乏層近似(濃度は m⁻³)。x [m], 電荷密度, 電場, 電位, 内蔵電位, 空乏層幅"""
    Vbi = VT300 * np.log(NA * ND / NI_SI ** 2)
    W = np.sqrt(2 * EPS_SI * (Vbi - V) / QE * (NA + ND) / (NA * ND))
    xp, xn = W * ND / (NA + ND), W * NA / (NA + ND)
    x = np.linspace(-1.6 * xp, 1.6 * xn, n)
    rho = np.where((x > -xp) & (x < 0), -QE * NA, 0.0) + np.where((x >= 0) & (x < xn), QE * ND, 0.0)
    E = np.where((x > -xp) & (x < 0), -QE * NA * (x + xp) / EPS_SI, 0.0) + np.where((x >= 0) & (x < xn), -QE * ND * (xn - x) / EPS_SI, 0.0)
    phi = -np.concatenate([[0], np.cumsum((E[1:] + E[:-1]) / 2 * np.diff(x))])
    return x, rho, E, phi, Vbi, W, xp, xn


def diode_I(V, Is=1e-12, T=300.0):
    VT = VT300 * T / 300
    return Is * (np.exp(V / VT) - 1)


def tab_drift_diffusion():
    st.info("トランジスタ・ダイオード・太陽電池など半導体デバイスの中の電流は, 電子と正孔(ホール)の**ドリフト**(電場で押される流れ)と"
            "**拡散**(濃度の差で広がる流れ)で決まる。これを表すのが**ドリフト拡散方程式**で, 電場を決める**ポアソン方程式**と連立させて解く。")
    overview("ドリフト拡散方程式")

    st.subheader("1. 導出:2つの流れと保存則")
    st.markdown("電子の密度を $n$, 正孔の密度を $p$, 電場を $E=-\\phi_x$($\\phi$:電位)とする。電流密度は")
    st.latex(r"J_n=\underbrace{qn\mu_nE}_{\text{ドリフト}}+\underbrace{qD_n\frac{\partial n}{\partial x}}_{\text{拡散}},\qquad J_p=qp\mu_pE-qD_p\frac{\partial p}{\partial x}")
    st.markdown("($q$:電気素量, $\\mu$:移動度, $D$:拡散係数。電子は負の電荷なので拡散の符号が正孔と逆になる。)キャリアの数の保存(連続の式)と, 電荷から電場を決めるポアソン方程式を合わせて")
    st.latex(r"\frac{\partial n}{\partial t}=\frac1q\frac{\partial J_n}{\partial x}+G-R,\qquad \frac{\partial p}{\partial t}=-\frac1q\frac{\partial J_p}{\partial x}+G-R,\qquad "
             r"\varepsilon\frac{\partial^2\phi}{\partial x^2}=-q(p-n+N_D-N_A)")
    st.markdown("($G$, $R$:キャリアの生成と再結合, $N_D$, $N_A$:ドナー・アクセプターの濃度。)電場のないところでは, 余分な少数キャリア $\\Delta n$ は "
                "$\\dfrac{\\partial\\Delta n}{\\partial t}=D\\dfrac{\\partial^2\\Delta n}{\\partial x^2}-\\dfrac{\\Delta n}{\\tau}$($\\tau$:寿命)に従い, "
                "化学の応用の「反応つき拡散方程式」と同じ形になる。定常状態では $\\Delta n\\propto e^{-x/L}$, $L=\\sqrt{D\\tau}$(**拡散長**)。")
    st.markdown("**アインシュタインの関係**:電流が流れていない平衡状態では $J_n=0$, つまりドリフトと拡散が**釣り合う**。"
                "このとき電子はボルツマン分布 $n\\propto e^{q\\phi/kT}$ に従うので $n_x=\\dfrac{q}{kT}n\\phi_x=-\\dfrac{q}{kT}nE$ となり, $J_n=0$ から")
    st.latex(r"D=\frac{kT}{q}\,\mu\qquad(300\,\mathrm K\ \text{で}\ kT/q\approx0.026\,\mathrm V)")

    st.subheader("2. pn 接合:空乏層と内蔵電位")
    st.markdown("p 形(アクセプター $N_A$)と n 形(ドナー $N_D$)を接合すると, 境目付近の電子と正孔が拡散して打ち消し合い, "
                "イオンだけが残った**空乏層**ができる。その電荷が作る電場によるドリフトが拡散と釣り合ったところで平衡になる。"
                "空乏層の中だけに電荷があるとして(空乏層近似)ポアソン方程式を2回積分すると, 電場は三角形, 電位は2次関数の形になる。")
    c = st.columns(3)
    NA = c[0].select_slider("N_A [cm⁻³](p 側)", [1e15, 1e16, 1e17, 1e18], 1e16, format_func=lambda v: f"{v:.0e}", key="dd_NA")
    ND = c[1].select_slider("N_D [cm⁻³](n 側)", [1e15, 1e16, 1e17, 1e18], 1e16, format_func=lambda v: f"{v:.0e}", key="dd_ND")
    Vb = c[2].slider("加える電圧 V [V](正が順方向)", -5.0, 0.6, 0.0, 0.1, key="dd_V")
    x, rho, E, phi, Vbi, W, xp, xn = pn_junction(NA * 1e6, ND * 1e6, Vb)
    m = st.columns(3)
    m[0].metric("内蔵電位 V_bi = (kT/q) ln(N_A N_D/n_i²)", f"{Vbi:.3f} V")
    m[1].metric("空乏層の幅 W", f"{W * 1e6:.3f} μm")
    m[2].metric("最大の電場", f"{np.max(np.abs(E)) / 1e5:.1f} kV/cm")
    cols = st.columns(3)
    for col, (y, ttl, unit, color) in zip(cols, ((rho / QE / 1e6, "電荷密度 ρ/q", "cm⁻³", C2), (E / 1e5, "電場 E", "kV/cm", C1), (phi, "電位 φ", "V", C3))):
        with col:
            fig = go.Figure(go.Scatter(x=x * 1e6, y=y, mode="lines", line=dict(color=color, width=2.5), fill="tozeroy"))
            fig.add_vline(x=0, line=dict(color=GRAY, dash="dot"))
            layout(fig, title=ttl, xlab="x [μm](左が p, 右が n)", ylab=unit, height=260, legend=False)
            show(fig)
    st.caption("逆方向の電圧(V < 0)をかけると空乏層が広がり電場が強くなる。濃度の低い側に空乏層が大きく広がる。"
               "シリコン(ε = 11.7ε₀, n_i = 1.0×10¹⁰ cm⁻³, 300 K)で計算した。")

    st.subheader("3. ダイオードの電流:ショックレーの式")
    st.markdown("順方向の電圧 $V$ をかけると, 空乏層の障壁が下がって少数キャリアが注入され, 拡散長の範囲で拡散しながら再結合する。"
                "ドリフト拡散方程式をこの考えで解くと(ショックレー, 1949 年)")
    st.latex(r"I=I_s\left(e^{qV/kT}-1\right)")
    Tk = st.slider("温度 T [K]", 250, 400, 300, 10, key="dd_T")
    Vv = np.linspace(-0.2, 0.75, 400)
    fig = go.Figure(go.Scatter(x=Vv, y=1e3 * diode_I(Vv, T=Tk), mode="lines", line=dict(color=C1, width=2.5), name=f"T = {Tk} K"))
    fig.add_trace(go.Scatter(x=Vv, y=1e3 * diode_I(Vv, T=300), mode="lines", line=dict(color=GRAY, dash="dot"), name="T = 300 K"))
    layout(fig, title="ダイオードの電流電圧特性(I_s = 10⁻¹² A)", xlab="電圧 V [V]", ylab="電流 I [mA]", height=320, yrange=(-1, 30))
    show(fig)
    st.caption("電圧を $\\frac{kT}{q}\\ln10\\approx60$ mV 上げるごとに電流は約 10 倍。逆方向にはほとんど流れない(整流作用)。"
               "ここでは $I_s$ を一定としたが, 実際には $I_s$ も温度とともに大きく増える。")

    st.subheader("4. 歴史")
    st.markdown(r"""
| 年 | できごと |
|---|---|
| 1905 | アインシュタインがブラウン運動の論文で, 拡散係数と移動度の関係(アインシュタインの関係)を導く |
| 1930年代 | 量子力学(シュレディンガー方程式)で固体中の電子のエネルギーの帯(バンド)が説明され, 半導体の理論の土台ができる |
| 1947 | ベル研究所のバーディーン(J. Bardeen)とブラッテン(W. Brattain)が点接触トランジスタを発明 |
| 1949 | ショックレー(W. Shockley)が pn 接合と接合トランジスタの理論を発表 |
| 1950 | ファン・ルースブルック(W. van Roosbroeck)が, 電子・正孔のドリフト拡散と連続の式, ポアソン方程式からなる基礎方程式系を示す |
| 1956 | ショックレー, バーディーン, ブラッテンがノーベル物理学賞を受賞 |
""")
    with st.expander("練習問題"):
        st.markdown("**問1** 電子の移動度 $\\mu_n=1400$ cm²/(V·s) のシリコンで, 300 K の拡散係数を求めよ($kT/q=0.025$ V とする)。")
        st.markdown("**問2** $D=25$ cm²/s, 寿命 $\\tau=1$ μs のときの拡散長 $L=\\sqrt{D\\tau}$ を求めよ。")
        st.markdown("**問3** ショックレーの式で, 電圧を $0.12$ V 上げると電流は約何倍になるか($kT/q\\ln10\\approx0.06$ V とする)。")
        if st.toggle("解答・解説を表示", key="pr_dd"):
            st.markdown("**答1** $D=0.025\\times1400=35$ cm²/s。")
            st.markdown("**答2** $L=\\sqrt{25\\times10^{-6}}=5\\times10^{-3}$ cm $=50$ μm。")
            st.markdown("**答3** $0.12=2\\times0.06$ なので $10^2=100$ 倍。")
    page_refs([("shockley1949", ""), ("vanroosbroeck1950", ""), ("szeng2007", "第1〜2章:キャリアの輸送, pn 接合。")])
