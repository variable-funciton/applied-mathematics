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
    adv_mech()


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
    adv_elec()


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
    adv_chem()


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
    adv_city()



# ============================================================
# 発展(流体・量子・半導体)の応用
# ============================================================
def adv_mech():
    st.divider()
    st.subheader("発展の応用:配管の圧力損失と潤滑(ナヴィエ・ストークス方程式)")
    st.markdown("- **配管の圧力損失**:層流の円管では, 長さ $\\ell$ の区間の圧力差は $\\Delta p=\\dfrac{8\\mu\\ell Q}{\\pi R^4}$(ハーゲン・ポアズイユの法則)。"
                "半径を半分にすると, 同じ流量を流すのに 16 倍の圧力(ポンプの力)が要る。\n"
                "- **すべり軸受の潤滑**:回転する軸と軸受の狭いすき間の油の流れはクエット流れとポアズイユ流れの重ね合わせで, 油膜の圧力が軸を浮かせる(レイノルズの潤滑理論)。\n"
                "- **カルマン渦**:流れの中の円柱の後ろには渦が交互に放出され, その周期的な力が配管や熱交換器の管を振動させることがある(都市への応用の発展を参照)。")
    c = st.columns(4)
    d = c[0].select_slider("管の内径 [mm]", [2, 5, 10, 20, 50], 10, key="pa2_d")
    Q = c[1].select_slider("流量 [L/min]", [0.1, 0.5, 1.0, 2.0, 5.0], 1.0, key="pa2_Q")
    ell = c[2].select_slider("管の長さ [m]", [1, 5, 10, 50], 10, key="pa2_l")
    oil = c[3].radio("流体", ["水(μ = 1.0×10⁻³ Pa·s)", "油(μ = 0.1 Pa·s)"], key="pa2_f")
    mu, rho = (1e-3, 1000.0) if oil.startswith("水") else (0.1, 900.0)
    R = d / 2000
    Qs = Q / 1000 / 60
    dp = 8 * mu * ell * Qs / (PI * R ** 4)
    V = Qs / (PI * R * R)
    Re = rho * V * (2 * R) / mu
    m = st.columns(3)
    m[0].metric("平均流速", f"{V:.3g} m/s")
    m[1].metric("レイノルズ数", f"{Re:,.0f}", delta="層流" if Re < 2300 else "乱流:この式は使えない", delta_color="off")
    m[2].metric("圧力損失 Δp(層流の式)", f"{dp / 1000:.3g} kPa")
    st.caption("乱流になると圧力損失は層流の式より大きくなり, 実験式(ダルシー・ワイスバッハの式と管摩擦係数)を使う。")
    examples("p_mech2", [
        (r"内径 $10\ \mathrm{mm}$ の管で油($\mu=0.1\ \mathrm{Pa\cdot s}$)を流す。同じ流量で内径を $5\ \mathrm{mm}$ にすると, 圧力損失は何倍になるか(層流とする)。",
         [r"$\Delta p\propto\dfrac{1}{R^4}$ なので $2^4=16$ 倍。"]),
        (r"円管のポアズイユ流れ $u(r)=\dfrac{P}{4\mu}(R^2-r^2)$ から流量 $Q=\displaystyle\int_0^Ru\,2\pi r\,dr$ を計算し, ハーゲン・ポアズイユの法則を導け。",
         [r"$Q=\dfrac{\pi P}{2\mu}\displaystyle\int_0^R(R^2r-r^3)\,dr=\dfrac{\pi P}{2\mu}\cdot\dfrac{R^4}{4}=\dfrac{\pi PR^4}{8\mu}$。",
          r"$P=\dfrac{\Delta p}{\ell}$ とおくと $\Delta p=\dfrac{8\mu\ell Q}{\pi R^4}$。"]),
    ])
    refs([("Wikipedia:Hagen–Poiseuille equation", "https://en.wikipedia.org/wiki/Hagen%E2%80%93Poiseuille_equation", "管の中の層流, 圧力損失の式と歴史(英語)")])


def adv_elec():
    st.divider()
    st.subheader("発展の応用:pn 接合ダイオード・太陽電池とトンネル効果")
    st.markdown("- **ダイオード・太陽電池**:ドリフト拡散方程式とポアソン方程式で pn 接合を解くと, ダイオードの電流は $I=I_s(e^{qV/kT}-1)$ になる(ショックレーの式)。"
                "太陽電池では光で生まれた電子・正孔が接合の電場で分けられ, $I=I_s(e^{qV/kT}-1)-I_L$ となる($I_L$:光電流)。\n"
                "- **トンネル効果**:シュレディンガー方程式によると, 電子は薄い絶縁膜を通り抜けることがある。"
                "フラッシュメモリは, 数 nm の酸化膜を通るトンネル電流で電荷を出し入れして記録する。")
    c = st.columns(2)
    IL = c[0].slider("光電流 I_L [mA](光の強さ)", 0.0, 40.0, 30.0, 5.0, key="pa2_IL")
    Is = 1e-12
    V = np.linspace(0, 0.75, 400)
    VT = 0.02585
    I = Is * (np.exp(V / VT) - 1) - IL / 1000
    P = -V * I
    Voc = VT * np.log(IL / 1000 / Is + 1) if IL > 0 else 0.0
    k = int(np.argmax(P))
    c[1].metric("開放電圧 V_oc", f"{Voc:.3f} V")
    c[1].metric("最大出力", f"{1000 * P[k]:.2f} mW(V = {V[k]:.3f} V)")
    fig = go.Figure(go.Scatter(x=V, y=-1000 * I, mode="lines", line=dict(color=C1, width=2.5), name="取り出せる電流 −I [mA]"))
    fig.add_trace(go.Scatter(x=V, y=1000 * P, mode="lines", line=dict(color=C3, width=2, dash="dash"), name="出力 P = V·(−I) [mW]"))
    layout(fig, title="太陽電池の電流電圧特性(I_s = 10⁻¹² A, 300 K)", xlab="電圧 V [V]", height=300, yrange=(0, max(45.0, 1.1 * IL)))
    show(fig)
    st.caption("光が強いほど電流が増えるが, 開放電圧は $\\frac{kT}{q}\\ln\\frac{I_L}{I_s}$ と対数でしか増えない。")
    examples("p_elec2", [
        (r"$I_s=10^{-12}\ \mathrm A$, $I_L=30\ \mathrm{mA}$ の太陽電池の開放電圧 $V_{oc}$ を求めよ($kT/q=0.026\ \mathrm V$, $\ln(3\times10^{10})\approx24.1$)。",
         [r"$I=0$ より $e^{qV_{oc}/kT}=\dfrac{I_L}{I_s}+1\approx3\times10^{10}$。", r"$V_{oc}\approx0.026\times24.1\approx0.63\ \mathrm V$。"]),
        (r"トンネル電流が $T\approx e^{-2\kappa a}$ に比例するとし, $\kappa=10\ \mathrm{nm^{-1}}$ のとき, 絶縁膜の厚さ $a$ を $0.1\ \mathrm{nm}$ 増やすと電流は何倍になるか($e^{-2}\approx0.135$)。",
         [r"$e^{-2\kappa\times0.1}=e^{-2}\approx0.135$ 倍(約 $\frac17$)。わずか 0.1 nm で大きく変わるので, 膜の厚さの精密な制御が必要になる。"]),
    ])
    refs([("PVEducation:Ideality Factor and I0", "https://www.pveducation.org/es/fotovoltaica/idealty-factor-and-i0", "理想ダイオードの式と太陽電池(英語)"),
          ("LibreTexts Engineering:The Diode Equation",
           "https://eng.libretexts.org/Workbench/Materials_Science_for_Electrical_Engineering/03%3A_Electrical_Properties/3.03%3A_Conductors_Semiconductors_and_Diodes/3.3.07%3A_The_Diode_Equation",
           "ダイオードの式の説明(英語)")])


def adv_chem():
    st.divider()
    st.subheader("発展の応用:水素原子のスペクトル(シュレディンガー方程式)と反応器の中の流れ")
    st.markdown("- **量子化学**:水素原子のシュレディンガー方程式を解くと, エネルギーは $E_n=-\\dfrac{13.6}{n^2}\\ \\mathrm{eV}$ と飛び飛びになる。"
                "電子が高い準位から低い準位へ移るときに出す光の波長 $\\lambda=\\dfrac{hc}{E_{n_2}-E_{n_1}}$ が, 物質を見分ける分光分析の手がかりになる。"
                "分子の結合や反応の起こりやすさも, シュレディンガー方程式を(コンピューターで)解いて調べる。\n"
                "- **反応器の中の対流と拡散**:流れ(ナヴィエ・ストークス方程式)に運ばれながら拡散・反応する物質は $c_t+uc_x=Dc_{xx}-kc$ に従う。"
                "流れと拡散の比(ペクレ数 $Pe=uL/D$)で, 混ざり方が大きく変わる。")
    n1 = st.radio("移った先の準位 n₁", [1, 2, 3], index=1, horizontal=True, key="pa2_n1", format_func=lambda v: {1: "1(ライマン系列)", 2: "2(バルマー系列)", 3: "3(パッシェン系列)"}[v])
    hc = 1239.84   # eV·nm
    rows = []
    for n2 in range(n1 + 1, n1 + 6):
        dE = 13.6 * (1 / n1 ** 2 - 1 / n2 ** 2)
        rows.append((n2, dE, hc / dE))
    fig = go.Figure()
    for n2, dE, lam in rows:
        col = C3 if 380 <= lam <= 780 else GRAY
        fig.add_trace(go.Scatter(x=[lam, lam], y=[0, 1], mode="lines", line=dict(color=col, width=4), name=f"n = {n2} → {n1}:{lam:.0f} nm"))
    fig.add_vrect(x0=380, x1=780, fillcolor="rgba(255,200,0,0.12)", line_width=0)
    layout(fig, title="水素原子が出す光の波長(黄色の帯が目に見える光)", xlab="波長 λ [nm]", height=280, yrange=(0, 1.1))
    fig.update_yaxes(showticklabels=False)
    show(fig)
    st.caption("バルマー系列(n₁ = 2)の最初の線 n = 3 → 2 は約 656 nm の赤い光。")
    examples("p_chem2", [
        (r"水素原子で電子が $n=3$ から $n=2$ の準位へ移るとき, 出す光のエネルギーと波長を求めよ($E_n=-\frac{13.6}{n^2}\ \mathrm{eV}$, $hc=1240\ \mathrm{eV\cdot nm}$)。",
         [r"$\Delta E=13.6\left(\frac14-\frac19\right)=13.6\times\frac{5}{36}\approx1.89\ \mathrm{eV}$。", r"$\lambda=\dfrac{1240}{1.89}\approx656\ \mathrm{nm}$(赤い光)。"]),
        (r"管の中を流速 $u=1\ \mathrm{cm/s}$ で流れる水溶液($D=1\times10^{-9}\ \mathrm{m^2/s}$)について, 長さ $L=1\ \mathrm{m}$ でのペクレ数を求め, 流れと拡散のどちらが支配的か答えよ。",
         [r"$Pe=\dfrac{uL}{D}=\dfrac{0.01\times1}{10^{-9}}=10^7$。", r"$Pe\gg1$ なので流れ(対流)が支配的で, 拡散だけでは管の向きにはほとんど混ざらない。"]),
    ])
    refs([("LibreTexts Chemistry:Hydrogen's Atomic Emission Spectrum",
           "https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Supplemental_Modules_(Physical_and_Theoretical_Chemistry)/Electronic_Structure_of_Atoms_and_Molecules/Hydrogen%27s_Atomic_Emission_Spectrum",
           "水素原子の発光スペクトルと系列(英語)")])


def adv_city():
    st.divider()
    st.subheader("発展の応用:建物・橋にかかる風とカルマン渦, 川や地下水の流れ")
    st.markdown("- **カルマン渦**:風の中の煙突・橋のケーブル・高い建物の後ろには, 渦が左右交互に放出される(ナヴィエ・ストークス方程式の解の代表例)。"
                "放出の振動数は $f=St\\dfrac{U}{D}$($St\\approx0.2$:ストローハル数, $U$:風速, $D$:幅)で, 構造物の固有振動数と一致すると大きく揺れる(渦励振)。\n"
                "- **川・水路の流れ**:浅い水の流れ(オイラー方程式の近似)で, 洪水の伝わり方や堰の水位を計算する。\n"
                "- **地下水**:砂や土の中のゆっくりした流れでは, ナヴィエ・ストークス方程式を平均すると「流速 ∝ 水頭の勾配」(ダルシーの法則)になり, "
                "定常の浸透流は導入2. のラプラス方程式に帰着する。")
    c = st.columns(3)
    U = c[0].slider("風速 U [m/s]", 1.0, 40.0, 10.0, 1.0, key="pa2_U")
    D = c[1].select_slider("幅(直径)D [m]", [0.05, 0.1, 0.5, 1.0, 3.0, 5.0], 1.0, key="pa2_D")
    fn = c[2].select_slider("構造物の固有振動数 [Hz]", [0.5, 1.0, 2.0, 5.0, 10.0], 2.0, key="pa2_fn")
    f = 0.2 * U / D
    Ucr = fn * D / 0.2
    m = st.columns(2)
    m[0].metric("渦の放出の振動数 f = 0.2 U/D", f"{f:.3g} Hz")
    m[1].metric("共振する風速 U = f_n D / 0.2", f"{Ucr:.3g} m/s")
    Us = np.linspace(0.5, 40, 300)
    fig = go.Figure(go.Scatter(x=Us, y=0.2 * Us / D, mode="lines", line=dict(color=C1, width=2.5), name="渦の放出の振動数"))
    fig.add_hline(y=fn, line=dict(color=C3, dash="dash"))
    fig.add_trace(go.Scatter(x=[Ucr], y=[fn], mode="markers", marker=dict(color=C3, size=11), name="共振(渦励振)"))
    layout(fig, title="風速とカルマン渦の振動数(破線が固有振動数)", xlab="風速 U [m/s]", ylab="振動数 [Hz]", height=300, xrange=(0, 40))
    show(fig)
    examples("p_city2", [
        (r"直径 $1\ \mathrm m$ の煙突に風速 $10\ \mathrm{m/s}$ の風が当たる。カルマン渦の放出の振動数を求めよ($St=0.2$)。煙突の固有振動数が $1\ \mathrm{Hz}$ なら, 共振する風速はいくらか。",
         [r"$f=0.2\times\dfrac{10}{1}=2\ \mathrm{Hz}$。", r"$0.2\times\dfrac{U}{1}=1$ より $U=5\ \mathrm{m/s}$ で共振する。"]),
        (r"ダルシーの法則 $v=-K\dfrac{dh}{dx}$($K$:透水係数)と水の保存 $\dfrac{dv}{dx}=0$ から, 1次元の定常な地下水の水頭 $h(x)$ が1次関数になることを示せ。",
         [r"$\dfrac{d}{dx}\left(-K\dfrac{dh}{dx}\right)=0$ より $h^{\prime\prime}=0$(1次元のラプラス方程式)。", r"2回積分して $h=px+q$。"]),
    ])
    refs([("MONOist:構造物破壊の一要因にもなるカルマン渦の解析", "https://monoist.itmedia.co.jp/mn/articles/1601/12/news027_3.html", "カルマン渦と構造物の振動")])
