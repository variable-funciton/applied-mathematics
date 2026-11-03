"""③ フーリエ解析 応用:学科別の応用例とサンプル例題(機械・電気電子・都市・化学)

各節は「応用例の説明 → 動かして確かめる図 → 例題(問題を先に並べ、解答はトグルで表示)→ 参考 URL」の順。
例題の数値は tests で独立に検算している。参考 URL は 2026 年 11 月に内容を確認したもの。
"""
import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import C1, C2, C3, C4, GRAY, layout, show
from topics.fourier import WAVES, coefficients, partial_sum
from topics.vec_apps import examples

PI = np.pi


def refs(items):
    st.subheader("参考 URL")
    st.markdown("\n".join(f"- [{t}]({u}) — {d}" for t, u, d in items))
    st.caption("外部のサイトです。内容や URL は変わることがあります。")


def harmonic_bars(n, amps, names, colors, title, height=320, log=False):
    fig = go.Figure()
    for A, nm, col in zip(amps, names, colors):
        fig.add_trace(go.Bar(x=n, y=A, name=nm, marker_color=col))
    if log:
        fig.update_yaxes(type="log")
    layout(fig, title=title, xlab="高調波の番号 n", height=height)
    fig.update_layout(barmode="group")
    return fig


# ============================================================
# 機械:周期的な力による振動と共振, 振動の FFT 解析
# ============================================================
def sdof_gain(n, w, w0, zeta):
    """x'' + 2ζω0x' + ω0²x = e^{inωt} の定常解の複素振幅"""
    return 1 / (w0 ** 2 - (n * w) ** 2 + 2j * zeta * w0 * n * w)


def tab_mech():
    st.info("機械工学では, **周期的な力による振動**(ばね・質量・ダンパの系)をフーリエ級数で調べ, "
            "**機械の振動の測定データ**を FFT(離散フーリエ変換)で周波数ごとに分けて, 共振や故障の原因を見つける。")
    st.markdown(
        "- **周期的な力と共振**:力 $F(t)$ をフーリエ級数 $\\sum(a_n\\cos n\\omega t+b_n\\sin n\\omega t)$ に分けると, "
        "線形な振動系の応答は各項への応答の和(重ね合わせ)。固有角振動数 $\\omega_0$ が $n\\omega$ に近いと, "
        "**$n$ 番目の高調波で共振**する(基本周波数がずれていても共振する)。\n"
        "- **回転機械の診断**:軸の回転数 $f_r$ [Hz] とその整数倍の成分の大きさから, アンバランス(回転周波数の成分が大きい)や, "
        "軸受・歯車の傷を見分ける。\n"
        "- **固有振動数の測定**:ハンマーでたたいた応答を FFT すると, ピークが部材の固有振動数になる(打撃試験)。")
    st.subheader("動かして確かめる:矩形波の力で揺らされるばね")
    st.latex(r"m\ddot x+c\dot x+kx=F(t)\quad\Longleftrightarrow\quad \ddot x+2\zeta\omega_0\dot x+\omega_0^2x=\frac{F(t)}{m}")
    c = st.columns(3)
    r = c[0].slider("固有角振動数 ω₀ ÷ 力の基本角振動数 ω", 0.5, 8.0, 3.0, 0.1, key="fm_r")
    zeta = c[1].slider("減衰比 ζ", 0.01, 0.5, 0.05, 0.01, key="fm_z")
    wave = c[2].radio("力の波形", ["矩形波", "のこぎり波", "三角波"], key="fm_w")
    a, b, _ = coefficients(wave, 1.0)
    N = 25
    n = np.arange(1, N + 1)
    G = sdof_gain(n, 1.0, r, zeta)
    Fn = np.hypot(a[1:N + 1], b[1:N + 1])
    Xn = Fn * np.abs(G)
    t = np.linspace(0, 4 * PI, 1601)
    F = partial_sum(t, a, b, N) - a[0] / 2
    x = sum(np.real((a[k] - 1j * b[k]) * G[k - 1] * np.exp(1j * k * t)) for k in range(1, N + 1))
    cA, cB = st.columns(2)
    with cA:
        fig = go.Figure(go.Scatter(x=t, y=F / max(np.abs(F).max(), 1e-12), mode="lines", line=dict(color=GRAY), name="力 F(t)(最大を1に)"))
        fig.add_trace(go.Scatter(x=t, y=x / max(np.abs(x).max(), 1e-12), mode="lines", line=dict(color=C2, width=2.5),
                                 name="変位 x(t)(最大を1に)"))
        layout(fig, title="定常状態の振動(ωt で2周期)", xlab="ωt", height=340)
        show(fig)
    with cB:
        fig = harmonic_bars(n, [Fn / Fn.max(), Xn / Xn.max()], ["力の高調波", "変位の高調波"], [GRAY, C2],
                            "高調波ごとの大きさ(最大を1に)")
        show(fig)
    kmax = int(n[np.argmax(Xn)])
    st.write(f"変位でいちばん大きいのは n = {kmax} の成分(周波数は力の基本周波数の {kmax} 倍)。")
    st.caption("ω₀ ÷ ω を 3 や 5 に合わせると, 矩形波に含まれる 3 倍・5 倍の高調波で共振し, 変位が力の基本周期より速く振動する。"
               "矩形波には偶数倍の成分がないので, ω₀ ÷ ω = 2 では共振しない。")
    examples("f_mech", MECH_EX)
    refs([("MIT OpenCourseWare 18.03SC:ODE's with Periodic Input, Resonance",
           "https://ocw.mit.edu/courses/18-03sc-differential-equations-fall-2011/pages/unit-iii-fourier-series-and-laplace-transform/odes-with-periodic-input-resonance",
           "周期的な入力をフーリエ級数に分けて微分方程式を解く(英語)"),
          ("リオン:音響振動テクニカルノート「音響・振動の FFT 解析」",
           "https://svmeas.rion.co.jp/support/p38veq0000000cmg-att/FFT_07881.pdf",
           "FFT 分析器のしくみと, 軸受・アンバランス・歯車の診断例"),
          ("NTi Audio:高速フーリエ変換 - 基礎編", "https://www.nti-audio.com/ja/サポート/測定ノウハウ/高速フーリエ変換",
           "サンプリング周波数・周波数分解能・窓関数")])


MECH_EX = [
    (r"大きさ $F_0$ の矩形波の力 $F(t)=\begin{cases}-F_0&(-\pi<\omega t<0)\\F_0&(0<\omega t<\pi)\end{cases}$ の"
     r"フーリエ級数を書け。また, 減衰の小さいばね・質量系の固有角振動数が $\omega_0=3\omega$ のとき, どの成分で共振するか。",
     [r"$$F(t)=\frac{4F_0}{\pi}\left(\sin\omega t+\frac{\sin3\omega t}{3}+\frac{\sin5\omega t}{5}+\cdots\right)$$",
      r"$3\omega=\omega_0$ なので, 第3高調波 $\dfrac{4F_0}{3\pi}\sin3\omega t$ で共振する。力の基本周期の $\frac13$ の周期で大きく揺れる。"]),
    (r"減衰のない系 $\ddot x+10x=\dfrac{4}{\pi}\left(\sin t+\dfrac{\sin3t}{3}+\dfrac{\sin5t}{5}+\cdots\right)$ の定常解を, "
     r"$\sin 5t$ の項まで求めよ。いちばん大きいのはどの項か。",
     [r"$\ddot x+10x=B\sin nt$ の定常解は $x=\dfrac{B}{10-n^2}\sin nt$(重ね合わせで各項を足す)。",
      r"$$x=\frac{4}{9\pi}\sin t+\frac{4}{3\pi}\sin3t-\frac{4}{75\pi}\sin5t+\cdots$$",
      r"$10-n^2$ が小さい $n=3$ の項 $\dfrac{4}{3\pi}\sin3t$ がいちばん大きい(固有角振動数 $\sqrt{10}\approx3.16$ に近い)。"]),
    (r"回転数 $1800\ \mathrm{rpm}$ のモーターの振動を, サンプリング周波数 $1024\ \mathrm{Hz}$, $1024$ 点で FFT した。"
     r"周波数分解能と, 回転周波数とその2倍・3倍の成分が現れる周波数を求めよ。",
     [r"回転周波数は $f_r=\dfrac{1800}{60}=30\ \mathrm{Hz}$。周波数分解能は $\Delta f=\dfrac{1024}{1024}=1\ \mathrm{Hz}$。",
      r"成分は $30,\ 60,\ 90\ \mathrm{Hz}$ に現れる(どれもナイキスト周波数 $512\ \mathrm{Hz}$ より低いので正しく分析できる)。"
      r"アンバランスでは $30\ \mathrm{Hz}$ の成分が大きくなる。"]),
]


# ============================================================
# 電気電子:ひずみ波交流, フィルタ, サンプリング
# ============================================================
def thd(amp):
    """全高調波ひずみ率 √(Σ_{n≥2} A_n²) / A_1"""
    return float(np.sqrt(np.sum(amp[2:] ** 2)) / amp[1]) if amp[1] > 1e-12 else float("nan")


def tab_elec():
    st.info("電気電子工学では, 正弦波でない電圧・電流(**ひずみ波**)をフーリエ級数で**基本波と高調波**に分けて扱う。"
            "回路の応答は周波数ごとに計算して足し合わせ(重ね合わせ), 信号処理ではフーリエ変換でスペクトルを調べる。")
    st.markdown(
        "- **ひずみ波交流**:インバータや整流回路の電圧・電流は高調波を含む。**実効値**はパーセバルの等式から "
        "$V_{\\mathrm{rms}}=\\sqrt{V_0^2+\\sum_n V_n^2}$($V_n$ は各高調波の実効値)。"
        "**ひずみ率**(全高調波ひずみ, THD)は $\\dfrac{\\sqrt{V_2^2+V_3^2+\\cdots}}{V_1}$。\n"
        "- **フィルタ**:RC 回路の伝達関数 $H(\\omega)=\\dfrac{1}{1+i\\omega RC}$ をかけると, 高い周波数の成分が弱まる(ローパスフィルタ)。\n"
        "- **サンプリング**:周波数 $f_s$ で標本化すると, $f_s/2$(ナイキスト周波数)より高い成分は別の周波数に見える(エイリアシング)。"
        "**変調**:$f(t)\\cos\\omega_ct$ のスペクトルは $\\pm\\omega_c$ に移る(AM ラジオ)。")
    st.subheader("動かして確かめる:ひずみ波を RC ローパスフィルタに通す")
    c = st.columns(2)
    wave = c[0].selectbox("入力電圧の波形", ["矩形波", "のこぎり波", "三角波", "半波整流", "全波整流 |sin x|"], key="fe_w")
    rc = c[1].slider("遮断角周波数 ωc = 1/RC ÷ 基本角周波数 ω", 0.2, 20.0, 2.0, 0.1, key="fe_r")
    a, b, energy = coefficients(wave, 1.0)
    N = 60
    n = np.arange(0, N + 1)
    H = 1 / (1 + 1j * n / rc)
    amp_in = np.hypot(a[:N + 1], b[:N + 1])
    amp_in[0] = abs(a[0]) / 2
    amp_out = amp_in * np.abs(H)
    t = np.linspace(-PI, 3 * PI, 1601)
    vin = partial_sum(t, a, b, N)
    vout = a[0] / 2 + sum(np.real((a[k] - 1j * b[k]) * H[k] * np.exp(1j * k * t)) for k in range(1, N + 1))
    cA, cB = st.columns(2)
    with cA:
        fig = go.Figure(go.Scatter(x=t, y=WAVES[wave]["f"](t, 1.0), mode="lines", line=dict(color=GRAY, width=2), name="入力 v_in"))
        fig.add_trace(go.Scatter(x=t, y=vout, mode="lines", line=dict(color=C2, width=2.5), name="出力 v_out"))
        layout(fig, title="入力と出力(ωt)", xlab="ωt", height=340)
        show(fig)
    with cB:
        k = n[:16]
        fig = harmonic_bars(k, [amp_in[:16], amp_out[:16]], ["入力", "出力"], [GRAY, C2], "各成分の振幅(n = 0 は直流)")
        show(fig)
    rms_in = math.sqrt(energy / 2)
    m = st.columns(4)
    m[0].metric("入力の実効値", f"{rms_in:.4f}")
    m[1].metric("出力の実効値", f"{math.sqrt(amp_out[0] ** 2 + np.sum(amp_out[1:] ** 2) / 2):.4f}")
    m[2].metric("入力のひずみ率 THD", f"{100 * thd(amp_in):.1f} %")
    m[3].metric("出力のひずみ率 THD", f"{100 * thd(amp_out):.1f} %")
    st.caption("実効値は, 入力は $\\sqrt{\\frac{1}{2\\pi}\\int v^2dt}$(数値積分), 出力は各成分から $\\sqrt{A_0^2+\\sum A_n^2/2}$ で計算(パーセバルの等式)。"
               "ωc を小さくすると高調波が削られて正弦波に近づき(ひずみ率が下がる), 直流分は通る。")
    examples("f_elec", ELEC_EX)
    refs([("長崎大学学術研究成果リポジトリ:第14章 フーリエ級数によるひずみ波の解析",
           "https://nagasaki-u.repo.nii.ac.jp/record/5525/files/Chapter%2014.pdf",
           "ひずみ波の実効値・ひずみ率・電力・回路の計算(電気回路の教科書の1章)"),
          ("NTi Audio:高速フーリエ変換 - 基礎編", "https://www.nti-audio.com/ja/サポート/測定ノウハウ/高速フーリエ変換",
           "サンプリング・ナイキスト周波数・周波数分解能"),
          ("MIT OpenCourseWare 18.03SC:ODE's with Periodic Input, Resonance",
           "https://ocw.mit.edu/courses/18-03sc-differential-equations-fall-2011/pages/unit-iii-fourier-series-and-laplace-transform/odes-with-periodic-input-resonance",
           "周期的な入力に対する回路・振動系の応答とフーリエ級数(英語)")])


ELEC_EX = [
    (r"振幅 $E$ の矩形波電圧 $v(t)=\dfrac{4E}{\pi}\left(\sin\omega t+\dfrac{\sin3\omega t}{3}+\cdots\right)$ について, "
     r"実効値, 基本波の実効値, ひずみ率を求めよ。",
     [r"$|v|=E$ なので実効値は $V=E$。基本波は振幅 $\dfrac{4E}{\pi}$ なので実効値 $V_1=\dfrac{4E}{\sqrt2\,\pi}=\dfrac{2\sqrt2E}{\pi}\approx0.900E$。",
      r"パーセバルの等式より $V^2=V_1^2+(\text{高調波の実効値の2乗和})$ なので"
      r"$$\text{ひずみ率}=\frac{\sqrt{V^2-V_1^2}}{V_1}=\sqrt{\frac{\pi^2}{8}-1}\approx0.483\ (48.3\%)$$"]),
    (r"遮断角周波数 $\omega_c=\dfrac{1}{RC}$ の RC ローパスフィルタ $H(\omega)=\dfrac{1}{1+i\omega RC}$ に, "
     r"基本角周波数 $\omega=\omega_c$ の矩形波(例題1)を入れた。出力の基本波と第3高調波の振幅を求めよ。",
     [r"$|H(n\omega)|=\dfrac{1}{\sqrt{1+n^2}}$。",
      r"基本波:$\dfrac{4E}{\pi}\cdot\dfrac{1}{\sqrt2}\approx0.900E$, 第3高調波:$\dfrac{4E}{3\pi}\cdot\dfrac{1}{\sqrt{10}}\approx0.134E$。"
      r"入力では第3高調波は基本波の $\frac13$ だったが, 出力では約 $0.149$ 倍に減る。"]),
    (r"$f=1\ \mathrm{kHz}$ の正弦波を, サンプリング周波数 $f_s=1.5\ \mathrm{kHz}$ で標本化した。標本の列は何 Hz の正弦波に見えるか。",
     [r"ナイキスト周波数は $f_s/2=750\ \mathrm{Hz}<1\ \mathrm{kHz}$ なので, エイリアシングが起こる。",
      r"$t_k=k/f_s$ で $\cos(2\pi f t_k)=\cos(2\pi (f-f_s)t_k)$ なので, $|f-f_s|=500\ \mathrm{Hz}$ の正弦波と区別できない。"
      r"正しく記録するには $f_s>2f=2\ \mathrm{kHz}$ が必要(標本化定理)。"]),
]


# ============================================================
# 都市:潮汐の調和解析, 地震動と建物の共振
# ============================================================
TIDES = {"M2(主太陰半日周潮)": 12.4206, "S2(主太陽半日周潮)": 12.0000,
         "K1(日月合成日周潮)": 23.9345, "O1(主太陰日周潮)": 25.8193}
TIDE_DEF = {"M2(主太陰半日周潮)": 0.50, "S2(主太陽半日周潮)": 0.25, "K1(日月合成日周潮)": 0.25, "O1(主太陰日周潮)": 0.20}


def base_gain(r, zeta):
    """地面の変位に対する建物の相対変位の倍率 r²/√((1−r²)² + (2ζr)²)(r = 地震動の振動数 ÷ 固有振動数)"""
    return r ** 2 / np.sqrt((1 - r ** 2) ** 2 + (2 * zeta * r) ** 2)


def tab_city():
    st.info("都市・建設系では, **潮位の変化**を周期の決まった波(分潮)の和として予測し(潮汐の調和解析), "
            "**地震動**をフーリエスペクトルで周期ごとに分けて, 建物が共振しやすい周期を調べる。")
    st.markdown(
        "- **潮汐**:月と太陽の引力による潮位は, 周期の決まった多くの成分(**分潮**)の和で表せる。"
        "主要4分潮は M2・S2・K1・O1。観測データから各分潮の振幅と位相を求めて(調和定数), 将来の潮位(天文潮位)を予測する。"
        "港湾・海岸の設計や, 潮位表に使われる。\n"
        "- **地震動と建物**:地震の揺れをフーリエ変換すると, どの周期の揺れが強いかがわかる(フーリエスペクトル)。"
        "建物の固有周期に近い成分で共振するので, 応答スペクトルと合わせて耐震設計に使う。\n"
        "- **周期的なデータ**:1日の交通量・気温・電力需要なども, 周期 24 時間のフーリエ級数で特徴(平均・振幅・ピークの時刻)をとらえられる。")
    st.subheader("動かして確かめる:分潮を足して潮位をつくる")
    cols = st.columns(4)
    amps = {k: cols[i].slider(f"{k.split('(')[0]} の振幅 [m]", 0.0, 1.0, TIDE_DEF[k], 0.05, key=f"fc_{k[:2]}")
            for i, k in enumerate(TIDES)}
    days = st.slider("表示する日数", 3, 30, 30, 1, key="fc_days")
    th = np.arange(0, 24 * days, 0.25)
    eta = sum(A * np.cos(2 * PI * th / TIDES[k]) for k, A in amps.items())
    fig = go.Figure(go.Scatter(x=th / 24, y=eta, mode="lines", line=dict(color=C1, width=1.6), name="潮位"))
    layout(fig, title="4つの分潮の和(位相はすべて 0 とした例)", xlab="日", ylab="潮位 [m]", height=320, legend=False)
    show(fig)
    T = np.arange(0, 24 * 60, 0.25)
    e2 = sum(A * np.cos(2 * PI * T / TIDES[k]) for k, A in amps.items())
    spec = np.abs(np.fft.rfft(e2)) / len(T) * 2
    fr = np.fft.rfftfreq(len(T), 0.25) * 24
    m = (fr > 0.5) & (fr < 2.5)
    fig = go.Figure(go.Scatter(x=fr[m], y=spec[m], mode="lines", line=dict(color=C2), name="振幅"))
    for k, P in TIDES.items():
        fig.add_vline(x=24 / P, line=dict(color=GRAY, dash="dot"), annotation_text=k.split("(")[0])
    layout(fig, title="60 日分を FFT したスペクトル(周波数の単位:1日あたりの回数)", xlab="周波数 [回/日]", height=300, legend=False)
    show(fig)
    st.caption("M2 と S2 の周期が少しだけ違うため, 2つが強め合う大潮と弱め合う小潮が約 15 日ごとにくり返す(うなり)。"
               "スペクトルでは, 1日約 2 回(半日周潮)と約 1 回(日周潮)のところにピークが並ぶ。")

    st.subheader("動かして確かめる:地震動の周期と建物の共振")
    c = st.columns(2)
    Tn = c[0].slider("建物の固有周期 T [s]", 0.1, 3.0, 0.5, 0.05, key="fc_T")
    zeta = c[1].slider("減衰定数 h", 0.01, 0.3, 0.05, 0.01, key="fc_h")
    Tg = np.linspace(0.05, 4.0, 800)
    fig = go.Figure(go.Scatter(x=Tg, y=base_gain(Tn / Tg, zeta), mode="lines", line=dict(color=C2, width=2.5),
                               name="倍率"))
    fig.add_vline(x=Tn, line=dict(color=GRAY, dash="dot"), annotation_text="T")
    layout(fig, title="地面の変位に対する建物の揺れ(相対変位)の倍率", xlab="地震動の成分の周期 [s]", height=320, legend=False)
    show(fig)
    st.caption(f"地震動の周期が固有周期 T と一致すると倍率 1/(2h) = {1 / (2 * zeta):.1f} 倍になる。"
               "地震動のフーリエスペクトルが T の付近で大きいほど, 建物は大きく揺れる。")
    examples("f_city", CITY_EX)
    refs([("気象庁:潮汐・海面水位の知識 用語集", "https://www.data.jma.go.jp/kaiyou/db/tide/knowledge/tide/sakuin.html",
           "「分潮」の項:主要4分潮 M2・S2・K1・O1, 天文潮位の計算に使う分潮"),
          ("気象庁:潮汐の仕組み", "https://ds.data.jma.go.jp/gmd/kaiyou/db/tide/knowledge/tide/choseki.html",
           "満潮・干潮, 大潮・小潮が起こるしくみ"),
          ("日本地震学会 強震動委員会:新・強震動地震学基礎講座(第4回)応答スペクトル",
           "https://www.zisin.jp/kyosindo/shin_kisokoza/shinkisokoza03_04.pdf",
           "地震動のフーリエスペクトルと応答スペクトル, 耐震設計との関係"),
          ("海上保安庁 水路部報告:最小自乗法による潮汐・潮流の短期調和分解の一つの試み",
           "https://www1.kaiho.mlit.go.jp/kenkyu/report/rhr16/rhr16-04.pdf", "潮汐の調和分解の方法(研究報告)")])


CITY_EX = [
    (r"M2 分潮(周期 $12.42$ 時間)と S2 分潮(周期 $12.00$ 時間)を足すと, 振幅がゆっくり変わる「うなり」が生じる。"
     r"うなりの周期(大潮から次の大潮まで)は約何日か。",
     [r"$\cos\alpha+\cos\beta=2\cos\dfrac{\alpha-\beta}{2}\cos\dfrac{\alpha+\beta}{2}$ より, 振幅は周波数の差 $\dfrac{1}{12.00}-\dfrac{1}{12.42}$ [回/時] でくり返す。",
      r"$$\frac{1}{\frac{1}{12.00}-\frac{1}{12.42}}=\frac{12.00\times12.42}{0.42}\approx354.9\ \text{時間}\approx14.8\ \text{日}$$"
      r"(新月・満月ごとに大潮になる, 約半月の周期)"]),
    (r"ある地点の気温(時刻 $t$ 時)が $T(t)=20-3\cos\dfrac{\pi t}{12}-3\sqrt3\sin\dfrac{\pi t}{12}\ [{}^\circ\mathrm C]$ で近似できるとき, "
     r"1日の平均気温, 最高気温とその時刻を求めよ。",
     [r"周期 24 時間のフーリエ級数の定数項 $20$ が平均気温。",
      r"$-3\cos\theta-3\sqrt3\sin\theta=6\cos\left(\theta-\dfrac{4\pi}{3}\right)$($\theta=\dfrac{\pi t}{12}$, 振幅 $\sqrt{9+27}=6$)。",
      r"最高気温は $26\ {}^\circ\mathrm C$ で, $\dfrac{\pi t}{12}=\dfrac{4\pi}{3}$ すなわち $t=16$ 時(最低気温 $14\ {}^\circ\mathrm C$ は 4 時)。"]),
    (r"固有周期 $0.5\ \mathrm{s}$, 減衰定数 $h=0.05$ の建物(1質点系)が, 周期 $0.5\ \mathrm{s}$ の正弦波の地震動を受ける。"
     r"地面の変位に対する建物の相対変位の倍率 $\dfrac{r^2}{\sqrt{(1-r^2)^2+(2hr)^2}}$($r$ は振動数の比)を求めよ。"
     r"周期 $2\ \mathrm{s}$ の地震動ではどうか。",
     [r"周期が等しいので $r=1$。倍率は $\dfrac{1}{2h}=10$ 倍(共振)。",
      r"周期 $2\ \mathrm{s}$ では振動数が $\frac14$ なので $r=0.25$。倍率は $\dfrac{0.0625}{\sqrt{0.9375^2+0.025^2}}\approx0.067$ 倍(ほとんど揺れない)。"]),
]


# ============================================================
# 化学:FT-NMR・FT-IR の分光, 拡散
# ============================================================
def fid_spectrum(freqs, amps, T2, fs=200.0, T=8.0):
    """FID s(t) = Σ A_k cos(2πf_k t) e^{−t/T2} と, その FFT の実部(吸収スペクトル:ローレンツ型)"""
    t = np.arange(0, T, 1 / fs)
    s = sum(A * np.cos(2 * PI * f * t) for f, A in zip(freqs, amps)) * np.exp(-t / T2)
    Z = 8 * len(t)                                    # 0 を補って周波数の目を細かく
    S = np.real(np.fft.rfft(s, Z)) / fs
    f = np.fft.rfftfreq(Z, 1 / fs)
    return t, s, f, S


def tab_chem():
    st.info("化学では, **FT-NMR** や **FT-IR**(フーリエ変換赤外分光法)のように, 時間や光路差の関数として測った信号を"
            "フーリエ変換してスペクトルを得る分析装置が広く使われる。拡散や熱伝導の問題もフーリエ級数で解ける。")
    st.markdown(
        "- **FT-NMR**:パルスを当てたあとに観測される減衰振動(**FID**, 自由誘導減衰)は, 各原子核の共鳴周波数の波の和。"
        "フーリエ変換すると周波数ごとのピーク(NMR スペクトル)になる。減衰が速い(緩和時間 $T_2$ が短い)ほどピークは太い。\n"
        "- **FT-IR**:干渉計の鏡を動かして光路差 $\\delta$ の関数として光の強さ(**インターフェログラム**)を測り, "
        "フーリエ変換して波数 $\\tilde\\nu$ ごとの吸収スペクトルを得る。すべての波数を一度に測れるので速く, 感度が高い。\n"
        "- **拡散**:両端の濃度を 0 に保った厚さ $L$ の膜の濃度は $c=\\sum b_ne^{-D(n\\pi/L)^2t}\\sin\\dfrac{n\\pi x}{L}$。"
        "細かい凹凸(大きい $n$)ほど速く消える。")
    st.subheader("動かして確かめる:FID をフーリエ変換して NMR スペクトルをつくる")
    c = st.columns(4)
    f1 = c[0].slider("成分1 の周波数 [Hz]", 5.0, 60.0, 20.0, 0.5, key="fh_f1")
    f2 = c[1].slider("成分2 の周波数 [Hz]", 5.0, 60.0, 23.0, 0.5, key="fh_f2")
    A2 = c[2].slider("成分2 の強さ(成分1 = 1)", 0.0, 3.0, 2.0, 0.5, key="fh_A2")
    T2 = c[3].slider("緩和時間 T₂ [s]", 0.05, 2.0, 0.5, 0.05, key="fh_T2")
    t, s, f, S = fid_spectrum([f1, f2], [1.0, A2], T2)
    cA, cB = st.columns(2)
    with cA:
        fig = go.Figure(go.Scatter(x=t, y=s, mode="lines", line=dict(color=C1, width=1.2), name="FID"))
        layout(fig, title="FID(時間の関数)", xlab="t [s]", height=320, legend=False, xrange=(0, min(8.0, 6 * T2)))
        show(fig)
    with cB:
        m = (f > 0) & (f < 70)
        fig = go.Figure(go.Scatter(x=f[m], y=S[m], mode="lines", line=dict(color=C2, width=2), name="Re F"))
        layout(fig, title="スペクトル(フーリエ変換の実部, 周波数の関数)", xlab="周波数 [Hz]", height=320, legend=False)
        show(fig)
    st.caption(f"ピークの半値幅は約 1/(πT₂) = {1 / (PI * T2):.2f} Hz。2つの周波数の差がこれより小さいと, ピークが重なって分けられない"
               "(T₂ を小さくして確かめよう)。FID の振動の速さが周波数, 強さがピークの高さ(面積)に対応する。")
    examples("f_chem", CHEM_EX)
    refs([("日本分析機器工業会(JAIMA):分析の原理 No.19「フーリエ変換赤外分光光度計の原理と応用」",
           "https://www.jaima.or.jp/resource/jp/basic//pdf/basic_19.pdf",
           "干渉計とインターフェログラム, フーリエ変換で赤外スペクトルを得るしくみ(日本分光)"),
          ("Nanalysis NMR Blog:Relation between the FID and the NMR spectrum",
           "https://www.nanalysis.com/nmready-blog/2021/5/20/relation-between-the-fid-and-the-nmr-spectrum",
           "FID とフーリエ変換で NMR スペクトルが得られるしくみ(英語)"),
          ("MIT OpenCourseWare 18.03SC:ODE's with Periodic Input, Resonance",
           "https://ocw.mit.edu/courses/18-03sc-differential-equations-fall-2011/pages/unit-iii-fourier-series-and-laplace-transform/odes-with-periodic-input-resonance",
           "フーリエ級数による微分方程式の解法(英語)")])


CHEM_EX = [
    (r"FID の1つの成分 $s(t)=e^{-t/T_2}\cos2\pi f_0t\ (t\ge0)$ のフーリエ変換は, $f$ が $f_0$ に近いところで"
     r"$|S(f)|\approx\dfrac{T_2/2}{\sqrt{1+\{2\pi(f-f_0)T_2\}^2}}$ となる。$|S|^2$ が最大値の半分になる幅(半値全幅)を求め, "
     r"$T_2=0.5\ \mathrm{s}$ のときの値を計算せよ。",
     [r"$|S|^2\propto\dfrac{1}{1+\{2\pi(f-f_0)T_2\}^2}$(ローレンツ型)が半分になるのは $2\pi|f-f_0|T_2=1$ のとき。",
      r"半値全幅は $\Delta f=\dfrac{2}{2\pi T_2}=\dfrac{1}{\pi T_2}$。$T_2=0.5\ \mathrm{s}$ では $\Delta f=\dfrac{2}{\pi}\approx0.64\ \mathrm{Hz}$。"]),
    (r"FT-IR で, 波数 $\tilde\nu=1000\ \mathrm{cm^{-1}}$ の光だけが入ると, インターフェログラムは光路差 $\delta$ について "
     r"$I(\delta)\propto1+\cos2\pi\tilde\nu\delta$ となる。この周期(光路差)を求めよ。"
     r"また, 最大光路差 $\delta_{\max}=0.25\ \mathrm{cm}$ のときの波数分解能の目安 $\Delta\tilde\nu\approx\dfrac{1}{\delta_{\max}}$ を求めよ。",
     [r"周期は $\dfrac{1}{\tilde\nu}=10^{-3}\ \mathrm{cm}=10\ \mu\mathrm m$。",
      r"$\Delta\tilde\nu\approx\dfrac{1}{0.25}=4\ \mathrm{cm^{-1}}$。鏡を長く動かすほど細かく分けられる(フーリエ変換の不確定性と同じ考え方)。"]),
    (r"厚さ $L=1\ \mathrm{cm}$ の膜で, 両端の濃度を 0 に保つ。拡散係数 $D=1.0\times10^{-5}\ \mathrm{cm^2/s}$ のとき, "
     r"濃度 $c=\sum b_ne^{-D(n\pi/L)^2t}\sin\dfrac{n\pi x}{L}$ の $n=1$ の成分が $\dfrac1e$ になる時間を求めよ。$n=3$ の成分は何倍速く減るか。",
     [r"$n=1$:$\tau_1=\dfrac{L^2}{\pi^2D}=\dfrac{1}{\pi^2\times10^{-5}}\approx1.01\times10^4\ \mathrm s$(約 2.8 時間)。",
      r"減り方の速さは $n^2$ に比例するので, $n=3$ は $9$ 倍速い。時間がたつと, ほぼ $n=1$ の形 $\sin\dfrac{\pi x}{L}$ だけが残る。"]),
]
