"""⑤ 偏微分方程式 発展6. 金融とブラック・ショールズ方程式

記号:S 株価, K 権利行使価格, T 満期, τ = T − t 残り時間, r 金利(連続複利), σ ボラティリティ。
コール(買う権利)の満期の価値 max(S − K, 0), プット(売る権利)は max(K − S, 0)。
ブラック・ショールズ方程式 V_t + ½σ²S²V_SS + rSV_S − rV = 0。
x = log(S/K), τ' = σ²τ/2, V = K e^{αx + βτ'} u(x, τ'), k = 2r/σ², α = −(k−1)/2, β = −(k+1)²/4 で u_τ' = u_xx。
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.special import erfc

from common import C1, C2, C3, C4, GRAY, animate, layout, show
from topics.fourier_apps import refs
from topics.fourier_kernels import page_refs


def ncdf(x):
    """標準正規分布の累積分布関数 N(x) = ½ erfc(−x/√2)"""
    return 0.5 * erfc(-np.asarray(x, dtype=float) / np.sqrt(2))


def bs_call(S, K, r, sigma, tau):
    """ブラック・ショールズの式(コール)。τ = 0 では満期の価値"""
    S = np.asarray(S, dtype=float)
    if tau <= 0:
        return np.maximum(S - K, 0.0)
    with np.errstate(divide="ignore"):
        d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return S * ncdf(d1) - K * np.exp(-r * tau) * ncdf(d2)


def bs_put(S, K, r, sigma, tau):
    """プット = コール − S + K e^{−rτ}(プット・コール・パリティ)"""
    return bs_call(S, K, r, sigma, tau) - np.asarray(S, dtype=float) + K * np.exp(-r * tau)


def binom_one(S0, u, d, R, K, kind="call"):
    """1期間の二項モデル。R = 1 + (1期間の金利)。(Δ, B, q, 価格)"""
    pay = (lambda s: max(s - K, 0)) if kind == "call" else (lambda s: max(K - s, 0))
    Cu, Cd = pay(u * S0), pay(d * S0)
    delta = (Cu - Cd) / ((u - d) * S0)
    B = (u * Cd - d * Cu) / ((u - d) * R)
    q = (R - d) / (u - d)
    return delta, B, q, delta * S0 + B


def binom_tree(S0, K, r, sigma, T, N, kind="call"):
    """N 期間の二項モデル(コックス・ロス・ルビンシュタイン)"""
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    q = (np.exp(r * dt) - d) / (u - d)
    j = np.arange(N + 1)
    S = S0 * u ** j * d ** (N - j)
    V = np.maximum(S - K, 0) if kind == "call" else np.maximum(K - S, 0)
    disc = np.exp(-r * dt)
    for _ in range(N):
        V = disc * (q * V[1:] + (1 - q) * V[:-1])
    return float(V[0])


def gbm_paths(S0, mu, sigma, T, n_paths, n_steps, seed=0):
    """株価のでたらめな動き(幾何ブラウン運動)S = S0 exp((μ − σ²/2)t + σW)"""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    dW = rng.normal(0, np.sqrt(dt), (n_paths, n_steps))
    logS = np.cumsum((mu - sigma ** 2 / 2) * dt + sigma * dW, axis=1)
    return np.hstack([np.full((n_paths, 1), S0), S0 * np.exp(logS)])


def tab_black_scholes():
    st.info("金融の世界では, 株そのものだけでなく「株を決まった値段で買う**権利**」も売り買いされる。その権利の公平な値段を決めるのが"
            "**ブラック・ショールズ方程式**で, 変数を変えると**熱方程式**そのものになる。金融の予備知識は要らない。")
    st.caption("このページは数学の教材であり, 投資の助言ではない。実際の市場はここでのモデルの仮定(下の 6.)どおりには動かない。")

    st.subheader("1. 何を求めたいか:オプション(権利)の値段")
    st.markdown("「**1年後に, A 社の株を 1 株 1000 円で買ってもよい権利**」を考える。これを**コール・オプション**といい, "
                "1000 円を**権利行使価格** $K$, 1年後を**満期** $T$ という。")
    st.markdown("- 1年後の株価 $S$ が 1200 円なら, 権利を使って 1000 円で買い, すぐ 1200 円で売れば 200 円得する。\n"
                "- 株価が 800 円なら, 権利を使わなければよい(損はしない)。\n\n"
                "つまり満期での権利の価値は $\\max(S-K,\\,0)$。逆に「決まった値段で**売る**権利」(**プット・オプション**)の価値は $\\max(K-S,\\,0)$ である。"
                "将来の株価は分からないのに, **今この権利にいくら払うのが公平か**, が問題である。")
    c = st.columns([1, 2])
    kind = c[0].radio("種類", ["コール(買う権利)", "プット(売る権利)"], key="fi_kind")
    K = c[0].slider("権利行使価格 K [円]", 600, 1400, 1000, 50, key="fi_K")
    S = np.linspace(400, 1600, 601)
    pay = np.maximum(S - K, 0) if kind.startswith("コール") else np.maximum(K - S, 0)
    with c[1]:
        fig = go.Figure(go.Scatter(x=S, y=pay, mode="lines", line=dict(color=C1, width=3), name="満期の価値"))
        fig.add_vline(x=K, line=dict(color=GRAY, dash="dash"))
        layout(fig, title="満期での権利の価値(破線が K)", xlab="満期の株価 S [円]", ylab="価値 [円]", height=280)
        show(fig)
    st.caption("グラフは $S=K$ で折れ曲がる。この「折れ目」が, あとで熱方程式によってなめらかになる。")

    st.subheader("2. 株価は「ゆらぎ」を含む")
    st.markdown("株価は毎日でたらめに上下する。株価そのものではなく**変化の割合**が, 平均 $\\mu$(期待される伸び)とばらつき $\\sigma$(**ボラティリティ**)で"
                "でたらめに動く, と考えるのが標準的なモデルである:")
    st.latex(r"\frac{\Delta S}{S}=\mu\,\Delta t+\sigma\sqrt{\Delta t}\ Z\qquad(Z\text{ は平均 }0,\ \text{分散 }1\text{ の正規分布にしたがう乱数})")
    st.markdown("$\\sqrt{\\Delta t}$ が付くのは, 化学の拡散と同じく, でたらめな動きの広がりが**時間の平方根**に比例するからである(ブラウン運動)。")
    c = st.columns(3)
    mu = c[0].slider("μ(1年あたりの平均の伸び)", -0.2, 0.3, 0.05, 0.05, key="fi_mu")
    sg = c[1].slider("σ(ボラティリティ)", 0.05, 0.6, 0.2, 0.05, key="fi_sg")
    seed = c[2].number_input("乱数の種", 0, 999, 1, key="fi_seed")
    P = gbm_paths(1000, mu, sg, 1.0, 400, 250, int(seed))
    t = np.linspace(0, 1, 251)
    cols = st.columns([3, 2])
    with cols[0]:
        fig = go.Figure()
        for k in range(30):
            fig.add_trace(go.Scatter(x=t, y=P[k], mode="lines", line=dict(width=1), showlegend=False))
        layout(fig, title="株価のでたらめな動き(30 本の例, 1 年 = 250 営業日)", xlab="時間 [年]", ylab="株価 [円]", height=320, legend=False)
        show(fig)
    with cols[1]:
        cnt, edges = np.histogram(P[:, -1], bins=40)
        fig = go.Figure(go.Bar(x=cnt, y=(edges[1:] + edges[:-1]) / 2, orientation="h", marker_color=C2, name="1年後の株価(400 本)"))
        fig.add_hline(y=1000, line=dict(color=GRAY, dash="dash"))
        layout(fig, title="1年後の株価の分布", xlab="本数", ylab="株価 [円]", height=320, legend=False)
        show(fig)
    st.caption("σ を大きくすると広がりが大きくなる。分布は右に長く伸びる(株価は 0 より下がらず, 割合で増える)。")

    st.subheader("3. 鍵のアイデア:ゆらぎを打ち消す(二項モデル)")
    st.markdown("最も簡単な場合として, 1期間後に株価が $u$ 倍に上がるか $d$ 倍に下がるかの2通りしかないとする。"
                "銀行に預けたお金は1期間で $R$ 倍(金利の分)になるとする。")
    st.markdown("**株を $\\Delta$ 株と預金 $B$ 円を組み合わせて, 上がっても下がってもオプションと同じ値になるようにする**(複製):")
    st.latex(r"\Delta\cdot uS_0+RB=C_u,\qquad \Delta\cdot dS_0+RB=C_d\quad(C_u,\ C_d\ \text{は上がったとき・下がったときの権利の価値})")
    st.markdown("この連立1次方程式を解けば $\\Delta$ と $B$ が決まる。同じ将来をもつものは今も同じ値段のはずなので(そうでないと, 安い方を買って高い方を売れば"
                "元手なしで必ずもうかる:**裁定**), 権利の値段は $\\Delta S_0+B$ である。整理すると")
    st.latex(r"C=\frac{qC_u+(1-q)C_d}{R},\qquad q=\frac{R-d}{u-d}")
    c = st.columns(4)
    S0 = c[0].select_slider("今の株価 S₀", [800, 1000, 1200], 1000, key="fi_S0")
    up = c[1].select_slider("上がるとき u 倍", [1.1, 1.2, 1.25, 1.5], 1.2, key="fi_u")
    dn = c[2].select_slider("下がるとき d 倍", [0.5, 0.75, 0.8, 0.9], 0.8, key="fi_d")
    R = c[3].select_slider("預金が R 倍", [1.0, 1.02, 1.05, 1.1], 1.0, key="fi_R")
    if not (dn < R < up):
        st.warning("d < R < u でないと裁定(必ずもうかる方法)ができてしまうので, この範囲に設定してください。")
    else:
        delta, B, q, price = binom_one(S0, up, dn, R, K, "call" if kind.startswith("コール") else "put")
        m = st.columns(4)
        m[0].metric("株の数 Δ", f"{delta:.4g}")
        m[1].metric("預金 B [円]", f"{B:.1f}")
        m[2].metric("q", f"{q:.4g}")
        m[3].metric(f"権利の値段(K = {K})", f"{price:.2f} 円")
        st.caption(f"確かめ:上がると Δ·{up * S0:g} + {R}B = {delta * up * S0 + R * B:.2f}, 下がると Δ·{dn * S0:g} + {R}B = {delta * dn * S0 + R * B:.2f}"
                   "(それぞれ満期の権利の価値に一致)。")
    st.markdown("**驚くべき点**:株が実際に上がる**確率**はどこにも出てこない。値段は「株が上がると思うかどうか」に関係なく, "
                "ゆらぎを打ち消す組合せだけから決まる。$q$ は確率のように振る舞う数で, **リスク中立確率**と呼ばれる。")

    st.subheader("4. ブラック・ショールズ方程式")
    st.markdown("期間を細かく分けて二項モデルをくり返し, 期間の長さを 0 に近づける。株価の動きに2.のモデルを使い, 各瞬間に株 $\\Delta=\\dfrac{\\partial V}{\\partial S}$ 株を"
                "組み合わせてゆらぎを打ち消すと, 権利の値段 $V(S,t)$ について次の偏微分方程式が得られる:")
    st.latex(r"\frac{\partial V}{\partial t}+\frac12\sigma^2S^2\frac{\partial^2V}{\partial S^2}+rS\frac{\partial V}{\partial S}-rV=0,\qquad V(S,T)=\max(S-K,0)")
    st.markdown("- $r$ は金利(連続的に複利で増えるとした年利)。**期待される伸び $\\mu$ は式に残らない**(3. の驚くべき点の連続版)。\n"
                "- 条件は**満期**(未来)で与えられ, 時間を**さかのぼって**今の値段を求める。\n"
                "- 2階微分の項 $\\frac12\\sigma^2S^2V_{SS}$ は, ゆらぎから生まれる。でたらめな動きでは $(\\Delta W)^2$ が平均して $\\Delta t$ になるので, "
                "テイラー展開の2次の項が $\\Delta t$ の大きさで残る。これを正確に扱う計算規則が**伊藤の公式**(伊藤清)である。")

    st.subheader("5. 熱方程式に変換する")
    st.markdown("残り時間 $\\tau=T-t$ と $x=\\log\\dfrac SK$ を使い, さらに指数関数を掛けて余分な項を消す:")
    st.latex(r"x=\log\frac SK,\quad s=\frac{\sigma^2}{2}(T-t),\quad V=K\,e^{\alpha x+\beta s}\,u(x,s),\quad k=\frac{2r}{\sigma^2},\ \alpha=-\frac{k-1}{2},\ \beta=-\frac{(k+1)^2}{4}")
    st.latex(r"\Longrightarrow\quad\frac{\partial u}{\partial s}=\frac{\partial^2u}{\partial x^2},\qquad u(x,0)=\max\left(e^{\frac{k+1}{2}x}-e^{\frac{k-1}{2}x},\,0\right)")
    st.markdown("満期からさかのぼる時間が, 熱方程式の時間になる。満期の価値の**折れ目**は, 熱が広がるようになめらかにならされていく。"
                "熱方程式の解(化学の拡散と同じ誤差関数)を元の変数に戻すと, **ブラック・ショールズの式**")
    st.latex(r"C=S\,N(d_1)-Ke^{-r\tau}N(d_2),\quad d_1=\frac{\log\frac SK+\left(r+\frac{\sigma^2}{2}\right)\tau}{\sigma\sqrt\tau},\ d_2=d_1-\sigma\sqrt\tau,\quad N(z)=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^ze^{-\frac{w^2}{2}}dw")
    c = st.columns(2)
    r = c[0].slider("金利 r(年)", 0.0, 0.1, 0.02, 0.01, key="fi_r")
    sg2 = c[1].slider("ボラティリティ σ", 0.05, 0.6, 0.25, 0.05, key="fi_sg2")
    S = np.linspace(400, 1800, 701)
    taus = np.round(np.concatenate([[0.0], np.linspace(0.02, 2.0, 50)]), 3)
    frames = [[bs_call(S, K, r, sg2, ta), np.maximum(S - K * np.exp(-r * ta), 0)] for ta in taus]
    fig = animate(S, frames, ["権利の値段 C(S)", "下限 max(S − Ke^{−rτ}, 0)"], [f"{ta:g} 年" for ta in taus], colors=[C1, GRAY], dashes=["solid", "dot"],
                  title=f"満期までの残り時間 τ とコールの値段(K = {K}, ▶ 再生で τ を増やす)", xlab="今の株価 S [円]", ylab="値段 [円]",
                  yrange=(0, 1000), height=400, duration=90, label_prefix="τ = ")
    show(fig)
    st.caption("τ = 0(満期)の折れ線が, 残り時間が長いほどなめらかに持ち上がる。これは熱方程式で温度の折れ目がならされるのと同じ現象である。")

    st.markdown("**3つの方法で値段を比べる**(今の株価 $S_0=1000$ 円, 満期 1 年):")
    N = st.select_slider("二項モデルの期間の数 N", [1, 2, 5, 10, 50, 200, 1000], 50, key="fi_N")
    bs = float(bs_call(1000.0, K, r, sg2, 1.0))
    bt = binom_tree(1000.0, K, r, sg2, 1.0, N)
    Z = np.random.default_rng(7).standard_normal(200000)
    ST = 1000.0 * np.exp((r - sg2 ** 2 / 2) + sg2 * Z)
    mc = float(np.exp(-r) * np.mean(np.maximum(ST - K, 0)))
    m = st.columns(3)
    m[0].metric("ブラック・ショールズの式", f"{bs:.2f} 円")
    m[1].metric(f"二項モデル(N = {N})", f"{bt:.2f} 円", delta=f"{bt - bs:+.2f}", delta_color="off")
    m[2].metric("シミュレーション(20 万本の平均)", f"{mc:.2f} 円", delta=f"{mc - bs:+.2f}", delta_color="off")
    st.caption("シミュレーションでは, 株価の伸びを μ ではなく金利 r にして満期の価値を平均し, 金利の分だけ割り引く(リスク中立確率での期待値)。"
               "N を増やすと二項モデルの値はブラック・ショールズの式に近づく。")

    st.subheader("6. モデルの仮定と限界")
    st.markdown("- ボラティリティ $\\sigma$ と金利 $r$ が一定, 取引の手数料や税がない, いつでも好きな量を売買できる, 株価の変化が正規分布にしたがう, などを仮定している。\n"
                "- 実際の市場では, 暴落のような大きな変動が正規分布より頻繁に起こり, $\\sigma$ も時間とともに変わる。"
                "市場の値段から逆算した $\\sigma$(インプライド・ボラティリティ)は $K$ によって違う値になる(スマイル)。\n"
                "- それでも, 値段を決める考え方(ゆらぎを打ち消す, 裁定がない)は広く使われ, モデルを改良する研究が続いている。")

    st.subheader("7. 歴史")
    st.markdown(r"""
| 年 | できごと |
|---|---|
| 1900 | バシュリエ(L. Bachelier)が博士論文「投機の理論」で株価をブラウン運動でモデル化し, オプションの値段を論じる(アインシュタインのブラウン運動の論文より5年早い) |
| 1944 | 伊藤清が確率積分(伊藤積分)を導入。1951 年に確率微分の計算規則(伊藤の公式)を発表 |
| 1973 | ブラック(F. Black)とショールズ(M. Scholes)が価格式を発表。同じ年にマートン(R. C. Merton)が理論を厳密にし, 拡張した |
| 1979 | コックス, ロス, ルビンシュタインが二項モデルを発表(3. の考え方) |
| 1997 | ショールズとマートンがノーベル経済学賞を受賞(ブラックは 1995 年に亡くなっていた) |
""")
    st.markdown("日本の数学者である伊藤清の確率解析が, 金融工学の数学的な土台になっている。")

    with st.expander("練習問題"):
        st.markdown("**問1** $K=1000$ 円のコールとプットについて, 満期の株価が $1300$ 円のときと $900$ 円のときの価値をそれぞれ求めよ。")
        st.markdown("**問2** $S_0=1000$, $u=1.2$, $d=0.8$, $R=1$(金利 0), $K=1000$ のコールの1期間の二項モデルの値段を求めよ。")
        st.markdown("**問3** 金利 0 のとき, 満期の価値について $\\max(S-K,0)-\\max(K-S,0)=S-K$ を確かめ, "
                    "コールの値段 $C$ とプットの値段 $P$ の関係(プット・コール・パリティ)を求めよ。")
        if st.toggle("解答・解説を表示", key="pr_fin"):
            st.markdown("**答1** $1300$ 円:コール $300$ 円, プット $0$ 円。$900$ 円:コール $0$ 円, プット $100$ 円。")
            st.markdown("**答2** $C_u=200$, $C_d=0$, $q=\\dfrac{1-0.8}{1.2-0.8}=\\dfrac12$ より $C=\\dfrac{\\frac12\\times200+\\frac12\\times0}{1}=100$ 円。"
                        "($\\Delta=\\dfrac{200-0}{400}=\\dfrac12$ 株, $B=-400$ 円:500 円分の株を 400 円借りて買う。)")
            st.markdown("**答3** $S\\ge K$ なら $(S-K)-0$, $S<K$ なら $0-(K-S)$ でどちらも $S-K$。同じ将来の価値をもつので今の値段も等しく, $C-P=S_0-K$"
                        "(金利 $r$ があれば $C-P=S_0-Ke^{-rT}$)。")

    page_refs([("wilmott1995", "ブラック・ショールズ方程式の導出と熱方程式への変換。"), ("blackscholes1973", ""), ("merton1973", ""),
               ("crr1979", ""), ("bachelier1900", ""), ("ito1944", ""), ("ito1951", "")])
    refs([("The Nobel Prize:Press release 1997(経済学賞)", "https://www.nobelprize.org/prizes/economic-sciences/1997/press-release/",
           "マートンとショールズの受賞理由(英語)"),
          ("日本数学会:伊藤清 生誕100周年 業績", "https://www.mathsoc.jp/meeting/ito100/works.html", "伊藤清の業績と論文の一覧")])
