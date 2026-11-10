"""⑤ 偏微分方程式 応用:学科別の応用例(機械・電気電子・化学・都市)

各節は「応用例の説明 → 動かして確かめる図 → 例題(解答はトグル)→ 参考 URL」の順(③ の応用ページと同じ形式)。
例題の数値は tests で独立に検算している。参考 URL は 2026 年 11 月に内容を確認したもの。
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.special import erfc

from common import C1, C2, C3, C4, GRAY, animate, layout, show
from topics.fourier_apps import refs
from topics.vec_apps import examples

PI = np.pi


# ============================================================
# 機械:冷却フィンの熱伝導, 棒の縦振動
# ============================================================
MATERIALS = {"銅(k = 400)": 400.0, "アルミニウム(k = 237)": 237.0, "鋼(k = 50)": 50.0, "ステンレス鋼(k = 16)": 16.0}


def fin_profile(x, k, h, d, L):
    """先端断熱のピンフィン:θ/θ₀ = cosh(m(L−x))/cosh(mL), m = √(4h/(kd))"""
    m = np.sqrt(4 * h / (k * d))
    return np.cosh(m * (L - x)) / np.cosh(m * L), m


def fin_eff(mL):
    return np.tanh(mL) / mL


def tab_mech():
    st.info("機械工学では, 部品の中の**熱の伝わり方**(熱方程式・その定常状態)や, 棒・はり・板の**振動**(波動方程式)を偏微分方程式で調べる。")
    st.markdown("- **冷却フィン**:CPU のヒートシンクやエンジンのフィンは, 細い棒の中を熱が伝わりながら側面から空気へ逃げる。"
                "定常状態では, 熱方程式に「側面から逃げる熱」の項が加わった $\\theta^{\\prime\\prime}=m^2\\theta$($\\theta$ は周囲の空気との温度差)になる。\n"
                "- **棒の縦振動**:棒の伸び縮みの波は $u_{tt}=\\dfrac{E}{\\rho}u_{xx}$($E$:ヤング率, $\\rho$:密度)。"
                "固有振動数が分かると, 共振を避ける設計や打撃試験による検査ができる。はりの曲げ振動は4階の方程式 $u_{tt}+\\dfrac{EI}{\\rho A}u_{xxxx}=0$ になる。\n"
                "- **焼入れ・溶接**:急に冷やしたり局所的に熱したりするときの温度変化は熱方程式で予測する。")
    st.subheader("動かして確かめる:ピンフィンの温度分布")
    st.latex(r"kA\,\theta^{\prime\prime}-hP\,\theta=0\ \Longrightarrow\ \theta^{\prime\prime}=m^2\theta,\quad m=\sqrt{\frac{hP}{kA}}=\sqrt{\frac{4h}{kd}}\ (\text{直径 }d\text{ の丸棒})")
    st.latex(r"\frac{\theta(x)}{\theta_0}=\frac{\cosh m(L-x)}{\cosh mL}\ (\text{先端は断熱}),\qquad \text{フィン効率}\ \eta=\frac{\tanh mL}{mL}")
    c = st.columns(4)
    mat = c[0].selectbox("材料", list(MATERIALS), index=1, key="pa_mat")
    h = c[1].slider("熱伝達率 h [W/(m²·K)]", 5, 200, 25, 5, key="pa_h")
    d = c[2].slider("直径 d [mm]", 1.0, 20.0, 5.0, 0.5, key="pa_d")
    L = c[3].slider("長さ L [mm]", 10, 300, 100, 10, key="pa_L")
    k = MATERIALS[mat]
    x = np.linspace(0, L / 1000, 301)
    th, m = fin_profile(x, k, h, d / 1000, L / 1000)
    mL = m * L / 1000
    mm = st.columns(3)
    mm[0].metric("m [1/m]", f"{m:.2f}")
    mm[1].metric("先端の温度差 θ(L)/θ₀", f"{th[-1]:.3f}")
    mm[2].metric("フィン効率 η", f"{fin_eff(mL):.3f}")
    fig = go.Figure(go.Scatter(x=1000 * x, y=th, mode="lines", line=dict(color=C3, width=3), name="θ(x)/θ₀"))
    layout(fig, title=f"{mat}, h = {h}, d = {d} mm のフィン", xlab="根元からの距離 x [mm]", ylab="θ/θ₀", height=300, yrange=(0, 1.05))
    show(fig)
    st.caption("熱伝導率の小さい材料や細いフィンでは先端まで熱が届かず(先端の温度差が小さい), 長くしても効率が上がらない。"
               "$mL$ が 1 程度になるように設計するのが目安。")
    examples("p_mech", [
        (r"鋼の棒(ヤング率 $E=206\ \mathrm{GPa}$, 密度 $\rho=7850\ \mathrm{kg/m^3}$)を縦波が伝わる速さを求めよ。"
         r"長さ $1\ \mathrm{m}$ の棒の一端を固定し他端を自由にしたとき, 最も低い固有振動数 $f_1=\dfrac{c}{4L}$ はいくらか。",
         [r"$c=\sqrt{E/\rho}=\sqrt{206\times10^9/7850}\approx5.12\times10^3\ \mathrm{m/s}$。",
          r"$f_1=\dfrac{5123}{4\times1}\approx1.28\times10^3\ \mathrm{Hz}$(固定端 $u=0$, 自由端 $u_x=0$ から $\sin\dfrac{(2n-1)\pi x}{2L}$ の形の振動になる)。"]),
        (r"アルミニウムのピンフィン($k=200\ \mathrm{W/(m\cdot K)}$, 直径 $5\ \mathrm{mm}$, 長さ $0.1\ \mathrm{m}$, $h=25\ \mathrm{W/(m^2\cdot K)}$)の $m$, "
         r"先端の温度差 $\theta(L)/\theta_0$, フィン効率を求めよ。",
         [r"$m=\sqrt{\dfrac{4\times25}{200\times0.005}}=\sqrt{100}=10\ \mathrm{m^{-1}}$, $mL=1$。",
          r"$\theta(L)/\theta_0=\dfrac{1}{\cosh1}\approx0.648$, $\eta=\tanh1\approx0.762$。"]),
        (r"フィンの微小部分 $[x,\,x+\Delta x]$ でエネルギーの保存を考え, $kA\theta^{\prime\prime}=hP\theta$ を導け($P$ は周の長さ, 周囲の温度との差を $\theta$)。",
         [r"入る熱 $-kA\theta^{\prime}(x)$, 出る熱 $-kA\theta^{\prime}(x+\Delta x)$, 側面から逃げる熱 $hP\Delta x\,\theta$。"
          r"定常なので $-kA\theta^{\prime}(x)+kA\theta^{\prime}(x+\Delta x)-hP\Delta x\,\theta=0$。",
          r"$\Delta x$ で割って $\Delta x\to0$ とすると $kA\theta^{\prime\prime}=hP\theta$。"]),
    ])
    refs([("Wikipedia:Fin (extended surface)", "https://en.wikipedia.org/wiki/Fin_(extended_surface)",
           "フィンの方程式, 先端の条件ごとの温度分布(cosh)と放熱量, フィン効率(英語)")])


# ============================================================
# 電気電子:分布定数回路(電信方程式)と反射
# ============================================================
LOADS = {"短絡(Z_L = 0)": 0.0, "Z_L = Z₀/2": 0.5, "整合(Z_L = Z₀)": 1.0, "Z_L = 3Z₀": 3.0, "開放(Z_L = ∞)": np.inf}


def gamma(r):
    """反射係数 Γ = (Z_L − Z₀)/(Z_L + Z₀),  r = Z_L/Z₀"""
    return 1.0 if np.isinf(r) else (r - 1) / (r + 1)


def line_voltage(x, t, G, t0=0.25, w=0.06):
    """長さ 1, 速さ 1 の無損失線路。送端は整合。入射パルス f(t − x) と負荷での反射 Γ f(t − (2 − x))"""
    def f(s):
        return np.exp(-((s - t0) / w) ** 2)
    return f(t - x) + G * f(t - (2 - x))


def tab_elec():
    st.info("電気電子工学では, 長い電線・同軸ケーブル・基板の配線を, 単位長さあたりの抵抗 $R$・インダクタンス $L$・静電容量 $C$・コンダクタンス $G$ をもつ"
            "**分布定数回路**として扱う。電圧・電流は位置と時間の関数で, **電信方程式**(波動方程式の一種)をみたす。")
    st.markdown("- **電信方程式**:$-\\dfrac{\\partial v}{\\partial x}=L\\dfrac{\\partial i}{\\partial t}+Ri$, $-\\dfrac{\\partial i}{\\partial x}=C\\dfrac{\\partial v}{\\partial t}+Gv$。"
                "損失がない($R=G=0$)と $v_{tt}=\\dfrac{1}{LC}v_{xx}$ となり, 電圧は速さ $\\dfrac{1}{\\sqrt{LC}}$ の波として伝わる。特性インピーダンスは $Z_0=\\sqrt{L/C}$。\n"
                "- **反射**:線路の終端の負荷 $Z_L$ が $Z_0$ と違うと, 波が反射係数 $\\Gamma=\\dfrac{Z_L-Z_0}{Z_L+Z_0}$ で反射する(高速の信号でのノイズの原因, TDR による断線位置の測定)。\n"
                "- **静電場**:電荷のない空間の電位はラプラス方程式 $\\Delta\\phi=0$ をみたす(コンデンサー, 絶縁設計)。半導体中のキャリアの広がりは拡散方程式で表される。")
    st.subheader("動かして確かめる:線路を進むパルスと終端での反射")
    st.markdown("**やってみよう**(グラフ下の **▶ 再生** を押すか, 時刻のスライダーをドラッグしてパルスを動かす):\n"
                "1. 負荷を「**整合**」にして再生 → パルスは右端の負荷に**吸収され, 戻ってこない**。\n"
                "2. 負荷を「**短絡**」にして再生 → 右端でパルスが**下向き(マイナス)にひっくり返って**戻ってくる。\n"
                "3. 負荷を「**開放**」にして再生 → パルスは**上向き(プラス)のまま**戻ってくる。端では一瞬, 電圧が2倍になる。\n"
                "4. 「$Z_L=3Z_0$」「$Z_L=Z_0/2$」では, 戻ってくるパルスの高さが反射係数 $\\Gamma$ 倍になっていることを確かめる。")
    ld = st.radio("終端の負荷", list(LOADS), index=2, horizontal=True, key="pa_load")
    G = gamma(LOADS[ld])
    x = np.linspace(0, 1, 401)
    ts = np.round(np.arange(0, 2.41, 0.03), 2)

    def inc(t):
        return line_voltage(x, t, 0.0)

    frames = [[line_voltage(x, t, G), inc(t), line_voltage(x, t, G) - inc(t)] for t in ts]
    wall = go.Scatter(x=[1, 1], y=[-1.3, 2.2], mode="lines", line=dict(color=GRAY, width=6), name="終端(負荷)")
    fig = animate(x, frames, ["電圧 v(x, t)(合計)", "入射波(右へ進む)", f"反射波(左へ戻る, Γ = {G:+.2f} 倍)"], [f"{t:.2f}" for t in ts],
                  colors=["black", C1, C3], dashes=["solid", "dot", "dash"], static=[wall],
                  title=f"{ld}:反射係数 Γ = {G:+.2f}", xlab="位置 x(右端の太線が負荷)", ylab="v", yrange=(-1.3, 2.2), height=420,
                  duration=70, label_prefix="t = ")
    show(fig)
    st.caption("時刻はパルスが線路の長さを進む時間を 1 とした。入射波(青の点線)は右へ, 反射波(赤の破線)は左へ進み, "
               "実際に測れる電圧は両者の和(黒の実線)である。開放端では同じ向き(Γ = +1), 短絡端では逆向き(Γ = −1)に反射し, "
               "整合していると反射しない。パルスが戻ってくるまでの時間から, ケーブルの断線や接触不良の位置が分かる(TDR)。")
    examples("p_elec", [
        (r"同軸ケーブルの単位長さあたりのインダクタンスが $L=250\ \mathrm{nH/m}$, 静電容量が $C=100\ \mathrm{pF/m}$ のとき, "
         r"信号の伝わる速さと特性インピーダンスを求めよ。",
         [r"$LC=2.5\times10^{-7}\times1.0\times10^{-10}=2.5\times10^{-17}$ より $\dfrac{1}{\sqrt{LC}}=2.0\times10^8\ \mathrm{m/s}$(光速の約 $\frac23$)。",
          r"$Z_0=\sqrt{L/C}=\sqrt{2500}=50\ \Omega$。"]),
        (r"$Z_0=50\ \Omega$ の線路の終端に $Z_L=150\ \Omega$ をつないだときの反射係数を求めよ。また, 開放・短絡・整合($Z_L=50\ \Omega$)の場合はどうか。",
         [r"$\Gamma=\dfrac{150-50}{150+50}=0.5$。開放は $\Gamma=1$, 短絡は $\Gamma=-1$, 整合は $\Gamma=0$(反射しない)。"]),
        (r"損失のない電信方程式 $-v_x=Li_t$, $-i_x=Cv_t$ から $i$ を消去して, $v_{tt}=\dfrac{1}{LC}v_{xx}$ を導け。",
         [r"1つ目を $x$ で微分すると $-v_{xx}=Li_{tx}$。2つ目を $t$ で微分すると $-i_{xt}=Cv_{tt}$。",
          r"$i_{tx}=i_{xt}$ より $v_{xx}=LCv_{tt}$, すなわち $v_{tt}=\dfrac{1}{LC}v_{xx}$。"]),
    ])
    refs([("千葉大学 橋本研也:伝送工学 Part 4 分布定数回路(講義資料)", "https://www.te.chiba-u.jp/lab/ken/lecture/Trans-4.pdf",
           "電信方程式, 特性インピーダンス, 進行波, 反射係数・透過係数"),
          ("長崎大学学術研究成果リポジトリ:第17章 分布定数回路", "https://nagasaki-u.repo.nii.ac.jp/record/5525/files/Chapter%2017.pdf",
           "電信方程式, 伝搬定数, 進行波・定在波, 反射とインピーダンス整合(電気回路の教科書の1章)")])


# ============================================================
# 化学:拡散(フィックの法則)
# ============================================================
DIFF = {"気体中(D = 1×10⁻⁵ m²/s)": 1e-5, "水溶液中(D = 1×10⁻⁹ m²/s)": 1e-9, "高分子・ゲル中(D = 1×10⁻¹¹ m²/s)": 1e-11}


def contact_profile(x, t, D):
    """x < 0 に濃度 c₀, x > 0 に 0 の液を接触させたときの c/c₀ = ½ erfc(x/(2√(Dt)))"""
    return 0.5 * erfc(x / (2 * np.sqrt(D * t)))


def tab_chem():
    st.info("化学では, 物質が濃い方から薄い方へ広がる**拡散**を, 熱方程式と同じ形の**拡散方程式**で表す。"
            "反応の速さが拡散で決まる場合(拡散律速)や, 吸収・乾燥・薬の放出・半導体へのドーピングの解析に使われる。")
    st.markdown("- **フィックの第1法則**:物質の流れ(流束)は濃度勾配に比例する $J=-D\\dfrac{\\partial c}{\\partial x}$($D$:拡散係数)。熱のフーリエの法則と同じ形。\n"
                "- **フィックの第2法則**:物質の保存と合わせて $\\dfrac{\\partial c}{\\partial t}=D\\dfrac{\\partial^2c}{\\partial x^2}$。\n"
                "- **拡散の距離**:解は $x/\\sqrt{Dt}$ の関数になり, 広がる距離は $\\sqrt{Dt}$ 程度。距離を2倍にするには4倍の時間がかかる。\n"
                "- **反応と拡散**:反応で物質が消えると $c_t=Dc_{xx}-kc$ のように反応の項が加わる(触媒の粒子内の反応, 生体の模様の形成)。")
    st.subheader("動かして確かめる:気体・水溶液・ゲルの中の拡散を同時に比べる")
    st.latex(r"c(x,0)=\begin{cases}c_0&(x<0)\\0&(x>0)\end{cases}\ \Longrightarrow\ \frac{c(x,t)}{c_0}=\frac12\,\mathrm{erfc}\!\left(\frac{x}{2\sqrt{Dt}}\right)")
    st.caption("$\\mathrm{erfc}(z)=\\dfrac{2}{\\sqrt\\pi}\\displaystyle\\int_z^\\infty e^{-s^2}ds$(相補誤差関数)。")
    st.markdown("濃度 $c_0$ の液と濃度 0 の液を $x=0$ で接触させる。**横軸は $-1$ mm 〜 $1$ mm に固定**して, 同じ小さな範囲を3つの物質の中で覗き込む。"
                "グラフ下の **▶ 再生** を押す(またはスライダーを動かす)と, 時間が 0.01 秒から 1000 秒まで進む。")
    x = np.linspace(-1e-3, 1e-3, 401)
    ts = np.logspace(-2, 3, 51)
    names = list(DIFF)
    frames = [[contact_profile(x, t, DIFF[nm]) for nm in names] for t in ts]
    labels = [f"{t:.3g} s" for t in ts]
    fig = animate(1e3 * x, frames, ["気体中", "水溶液中", "高分子・ゲル中"], labels, colors=[C1, C2, C3],
                  title="同じ時刻での広がり方の違い(D が 100 万倍違う)", xlab="位置 x [mm](範囲は固定)", ylab="c/c₀",
                  yrange=(-0.02, 1.05), height=420, duration=120, label_prefix="t = ")
    show(fig)
    st.caption("気体ではあっという間に平らにならされるのに, ゲルの中は 1000 秒たっても段差(初めの形)がほとんど崩れない。")
    st.markdown("**1 mm 広がるのにかかる時間の目安 $x^2/D$**:")
    st.markdown(r"""
| 場所 | 拡散係数 $D$ [m²/s] | $x^2/D$($x=1$ mm) | 日常の感覚 |
|---|---|---|---|
| 気体中 | $1\times10^{-5}$ | $0.1$ 秒 | まばたきする間に広がる(においがすぐ届く) |
| 水溶液中 | $1\times10^{-9}$ | $1000$ 秒(約 17 分) | コーヒーにミルクを入れたら, かき混ぜないと待っていられない |
| 高分子・ゲル中 | $1\times10^{-11}$ | $10^5$ 秒(約 28 時間) | 1 mm しみ込むのに丸1日以上かかる |
""")
    st.caption("液体や固体の中では物質が届くのが遅いので, 反応そのものより物質の拡散が全体の速さを決めることが多い(**拡散律速**)。"
               "時間は距離の2乗に比例するので, 1 cm(10 倍)なら時間は 100 倍になる。")
    examples("p_chem", [
        (r"水中の拡散係数を $D=1\times10^{-9}\ \mathrm{m^2/s}$ とする。物質が $1\ \mathrm{mm}$ 広がるまでの時間の目安を, 拡散の距離 $\sqrt{Dt}$ から求めよ。"
         r"$1\ \mathrm{cm}$ ではどうか。",
         [r"$\sqrt{Dt}=10^{-3}$ より $t=\dfrac{(10^{-3})^2}{10^{-9}}=1000\ \mathrm{s}$(約 17 分)。",
          r"$1\ \mathrm{cm}$ は距離が 10 倍なので時間は 100 倍の $10^5\ \mathrm{s}$(約 28 時間)。"]),
        (r"接触させた2つの液で, 時刻 $t$ に $x=2\sqrt{Dt}$ の位置の濃度 $c/c_0$ を求めよ($\mathrm{erfc}(1)\approx0.157$)。",
         [r"$\dfrac{c}{c_0}=\dfrac12\mathrm{erfc}(1)\approx0.079$。この位置は時間とともに $\sqrt t$ に比例して遠ざかる。"]),
        (r"$c(x,t)=\dfrac{1}{\sqrt{4\pi Dt}}e^{-x^2/(4Dt)}$ が拡散方程式 $c_t=Dc_{xx}$ をみたすことを確かめよ(原点に置いた物質の広がり)。",
         [r"$\log c=-\frac12\log(4\pi Dt)-\dfrac{x^2}{4Dt}$ より $\dfrac{c_t}{c}=-\dfrac{1}{2t}+\dfrac{x^2}{4Dt^2}$, $\dfrac{c_x}{c}=-\dfrac{x}{2Dt}$。",
          r"$c_{xx}=c\left(\dfrac{x^2}{4D^2t^2}-\dfrac{1}{2Dt}\right)$ なので $Dc_{xx}=c\left(\dfrac{x^2}{4Dt^2}-\dfrac{1}{2t}\right)=c_t$。"]),
    ])
    refs([("DoITPoMS(University of Cambridge):Fick's second law", "https://doitpoms.ac.uk/tlplib/diffusion/fick2.php",
           "フィックの第2法則と誤差関数を用いた解(英語)"),
          ("LibreTexts Engineering:Fick's Second Law of Diffusion",
           "https://eng.libretexts.org/Bookshelves/Materials_Science/TLP_Library_I/20%3A_Diffusion/20.02%3A_Fick's_Second_Law_of_Diffusion",
           "拡散方程式と半無限の棒への拡散(英語)")])


# ============================================================
# 都市:粘土層の圧密(テルツァギ), 地下水, 振動
# ============================================================
def consol_u(z, Tv, N=200):
    """両面排水の粘土層(排水距離 H)の過剰間隙水圧 u/u₀。z/H ∈ [0, 2]"""
    out = np.zeros_like(z, dtype=float)
    for m in range(N):
        M = PI * (2 * m + 1) / 2
        out += 2 / M * np.sin(M * z) * np.exp(-M * M * Tv)
    return out


def consol_U(Tv, N=200):
    """平均圧密度 U = 1 − Σ (2/M²) e^{−M² Tv}"""
    Tv = np.asarray(Tv, dtype=float)
    s = np.zeros_like(Tv)
    for m in range(N):
        M = PI * (2 * m + 1) / 2
        s += 2 / M ** 2 * np.exp(-M * M * Tv)
    return 1 - s


def tab_city():
    st.info("都市(土木・建築)工学では, 建物の重みで粘土の地盤がゆっくり沈む**圧密**, 地盤の中の**地下水の流れ**, "
            "地震による地盤や構造物の**振動**を偏微分方程式で扱う。")
    st.markdown("- **圧密**(テルツァギの理論, 1925 年):粘土に荷重をかけると, 間隙の水が押し出されるにつれて地盤が沈む。"
                "過剰間隙水圧 $u(z,t)$ は熱方程式と同じ $\\dfrac{\\partial u}{\\partial t}=c_v\\dfrac{\\partial^2u}{\\partial z^2}$($c_v$:圧密係数)に従う。"
                "沈下の進み具合は**時間係数** $T_v=\\dfrac{c_vt}{H^2}$($H$:排水距離)だけで決まる。\n"
                "- **地下水**:定常の浸透流では水頭 $h$ がラプラス方程式 $\\Delta h=0$ をみたす。等ポテンシャル線と流線で描く**流線網**から, ダムや矢板の下の漏水量を求める。\n"
                "- **振動**:地盤を伝わる地震波や, 高い建物・橋・ケーブルの振動は波動方程式で表される。")
    st.subheader("動かして確かめる:粘土層の圧密")
    st.latex(r"\frac{u}{u_0}=\sum_{m=0}^{\infty}\frac{2}{M}\sin\frac{Mz}{H}\,e^{-M^2T_v},\quad M=\frac{(2m+1)\pi}{2},\qquad "
             r"U=1-\sum_{m=0}^{\infty}\frac{2}{M^2}e^{-M^2T_v}")
    st.caption("上下の両面から排水される厚さ $2H$ の粘土層(初期の過剰間隙水圧 $u_0$ は一様)。変数分離とフーリエ級数で解ける(⑤ 1. と同じ方法)。")
    c = st.columns([1, 2])
    Tv = c[0].select_slider("時間係数 T_v", [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.848, 1.0, 1.5], 0.2, key="pa_Tv")
    c[0].metric("平均圧密度 U", f"{100 * float(consol_U(Tv)):.1f} %")
    z = np.linspace(0, 2, 401)
    with c[1]:
        fig = go.Figure()
        for T, col in zip((0.05, 0.1, 0.2, 0.5, 0.848), (C4, C2, C3, C1, GRAY)):
            fig.add_trace(go.Scatter(x=consol_u(z, T), y=z, mode="lines", line=dict(color=col, width=1.5, dash="dot"), name=f"T_v = {T}"))
        fig.add_trace(go.Scatter(x=consol_u(z, Tv), y=z, mode="lines", line=dict(color="black", width=3), name=f"T_v = {Tv}(選択)"))
        fig.update_yaxes(autorange="reversed")
        layout(fig, title="過剰間隙水圧の分布(等時曲線)", xlab="u/u₀", ylab="深さ z/H(上下が排水面)", height=360, xrange=(0, 1.05))
        show(fig)
    Ts = np.linspace(0.001, 1.5, 400)
    fig = go.Figure(go.Scatter(x=Ts, y=100 * consol_U(Ts), mode="lines", line=dict(color=C1, width=2.5), name="U(T_v)"))
    fig.add_trace(go.Scatter(x=[Tv], y=[100 * float(consol_U(Tv))], mode="markers", marker=dict(color=C3, size=11), name="選択"))
    layout(fig, title="時間係数と平均圧密度", xlab="T_v", ylab="U [%]", height=280, yrange=(0, 100))
    show(fig)
    st.caption("圧密度 50 % は $T_v\\approx0.197$, 90 % は $T_v\\approx0.848$。$T_v=c_vt/H^2$ なので, 排水距離 $H$ が2倍になると同じ圧密度まで4倍の時間がかかる"
               "(サンドドレーンなどで排水距離を短くして沈下を早める)。")
    examples("p_city", [
        (r"厚さ $4\ \mathrm{m}$ の粘土層が上下の砂層にはさまれている(両面排水)。圧密係数 $c_v=0.02\ \mathrm{m^2/日}$ のとき, "
         r"圧密度 90 %($T_v=0.848$)に達するまでの日数を求めよ。",
         [r"両面排水なので排水距離は $H=2\ \mathrm{m}$。$t=\dfrac{T_vH^2}{c_v}=\dfrac{0.848\times4}{0.02}\approx170$ 日。"]),
        (r"同じ粘土層の下が岩盤(水を通さない)で片面排水のとき, 90 % に達する日数はどうなるか。",
         [r"排水距離が $H=4\ \mathrm{m}$ に2倍になるので, 時間は4倍の約 $678$ 日になる。"]),
        (r"$T_v$ が小さいときの近似 $U\approx\sqrt{\dfrac{4T_v}{\pi}}$ を使って, 圧密度 50 % になる $T_v$ を求めよ。",
         [r"$\sqrt{4T_v/\pi}=0.5$ より $T_v=\dfrac{\pi}{16}\approx0.196$(級数で計算した値 $0.197$ とほぼ一致)。"]),
    ])
    refs([("高知工科大学:圧密(講義資料)", "https://www.infra.kochi-tech.ac.jp/takagi/Geotech/consolidation.pdf",
           "圧密の基礎方程式, 時間係数, 圧密度と級数解"),
          ("NPTEL:Solution of Terzaghi's Theory", "https://archive.nptel.ac.in/content/storage2/courses/105103097/web/chap8final/s6.htm",
           "テルツァギの圧密理論の解と, 時間係数と圧密度の近似式(英語)")])
