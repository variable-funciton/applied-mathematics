"""数学IV 偏微分方程式 発展6. 金融とブラック・ショールズ方程式:演習の問題生成

二項モデル(1期間):株価 S₀ → uS₀ または dS₀, 預金は R 倍。
  Δ = (C_u − C_d)/((u − d)S₀),  B = (uC_d − dC_u)/((u − d)R),  q = (R − d)/(u − d),  C = (qC_u + (1 − q)C_d)/R
"""
from fractions import Fraction

from topics.ex_core import Choice, ExamItem, Part, Problem, ProblemType, Retry, render, tex_fr

F = Fraction


def distinct(ok, cands):
    choices = [Choice(ok, True, value=ok)]
    for t, why in cands:
        if all(c.tex != t for c in choices) and len(choices) < 4:
            choices.append(Choice(t, why=why, value=t))
    if len(choices) < 4:
        raise Retry
    return choices


def yen(x):
    return rf"{tex_fr(F(x))}\ \text{{円}}"


def dec(x):
    """1.2 などの有限小数の表示(分母が 2, 5 だけの分数)"""
    x = F(x)
    s = f"{float(x):.6f}".rstrip("0").rstrip(".")
    return s


def pay(kind, S, K):
    return max(S - K, 0) if kind == "call" else max(K - S, 0)


def binom(rng):
    """係数がきれいになる1期間の二項モデルを, 条件をみたすまで引き直して作る"""
    for _ in range(200):
        try:
            return _binom(rng)
        except Retry:
            continue
    raise Retry


def _binom(rng):
    S0 = rng.choice([100, 200, 1000])
    u, d = rng.choice([(F(6, 5), F(4, 5)), (F(5, 4), F(3, 4)), (F(3, 2), F(1, 2)), (F(11, 10), F(9, 10)), (F(6, 5), F(9, 10))])
    R = rng.choice([F(1), F(1), F(21, 20), F(11, 10)])
    if not d < R < u:
        raise Retry
    kind = rng.choice(["call", "put"])
    K = S0 * rng.choice([F(1), F(9, 10), F(11, 10)])
    Cu, Cd = F(pay(kind, u * S0, K)), F(pay(kind, d * S0, K))
    if Cu == Cd:
        raise Retry
    q = (R - d) / (u - d)
    C = (q * Cu + (1 - q) * Cd) / R
    delta = (Cu - Cd) / ((u - d) * S0)
    B = (u * Cd - d * Cu) / ((u - d) * R)
    assert delta * S0 + B == C
    return dict(S0=S0, u=u, d=d, R=R, kind=kind, K=K, Cu=Cu, Cd=Cd, q=q, C=C, delta=delta, B=B)


def bstate(m):
    nm = "コール(買う権利)" if m["kind"] == "call" else "プット(売る権利)"
    rr = r"預金の金利は $0$" if m["R"] == 1 else rf"預金は1期間で ${dec(m['R'])}$ 倍"
    return (rf"今の株価が ${m['S0']}$ 円で, 1期間後に ${dec(m['u'])}$ 倍か ${dec(m['d'])}$ 倍になる株がある({rr})。"
            rf"権利行使価格 $K={tex_fr(m['K'])}$ 円, 満期が1期間後の{nm}について")


# ------------------------------------------------------------
# 基本:満期の価値
# ------------------------------------------------------------
def g_payoff(rng):
    kind = rng.choice(["call", "put"])
    K = rng.choice([800, 1000, 1200])
    S = K + rng.choice([-300, -200, -100, 100, 200, 300])
    val = pay(kind, S, K)
    other = pay("put" if kind == "call" else "call", S, K)
    ok = yen(val)
    cands = [(yen(other), "コールは「買う権利」で $\\max(S-K,0)$, プットは「売る権利」で $\\max(K-S,0)$。"),
             (yen(S - K if kind == "call" else K - S) if val == 0 else yen(-val), "損をするときは権利を使わなければよいので, 価値は $0$ より小さくならない。"),
             (yen(S), "権利の価値は株価そのものではなく, 得をする差額。"), (yen(K), "得をする差額 $|S-K|$ を考える。")]
    nm = "コール(買う権利)" if kind == "call" else "プット(売る権利)"
    return Problem("満期の価値", "基本",
                   rf"権利行使価格 $K={K}$ 円の{nm}について, 満期の株価が $S={S}$ 円のときの権利の価値を求めよ。",
                   distinct(ok, cands), hints=[r"コール $\max(S-K,0)$, プット $\max(K-S,0)$。"],
                   solution=[rf"$\max({S - K if kind == 'call' else K - S},\,0)={val}$ 円。"], meta=dict(kind=kind, K=K, S=S, val=val))


# ------------------------------------------------------------
# 基本:リスク中立確率 q, 株の数 Δ
# ------------------------------------------------------------
def g_q(rng):
    m = binom(rng)
    ok = tex_fr(m["q"])
    u, d, R = m["u"], m["d"], m["R"]
    cands = [(tex_fr(1 - m["q"]), r"$q=\dfrac{R-d}{u-d}$ は「上がる側」の数。$\dfrac{u-R}{u-d}$ は下がる側。"),
             (tex_fr((R - d) / (u + d)) if (R - d) / (u + d) != m["q"] else "1", "分母は $u-d$。"),
             (tex_fr(F(1, 2)) if m["q"] != F(1, 2) else tex_fr(F(1, 3)), "実際に上がる確率ではなく, $\\Delta$ と預金で複製した結果から決まる数。"),
             (tex_fr((u - d) / (R - d) if R != d else F(2)), r"分子と分母を取り違えていないか。")]
    rr = "" if R == 1 else rf", 預金は1期間で ${dec(R)}$ 倍"
    return Problem("リスク中立確率", "基本",
                   rf"1期間の二項モデルで, 株価が ${dec(u)}$ 倍に上がるか ${dec(d)}$ 倍に下がるとする{rr if rr else '(金利 $0$)'}。"
                   rf"リスク中立確率 $q=\dfrac{{R-d}}{{u-d}}$ を求めよ。",
                   distinct(ok, cands), hints=[r"$R$ は預金が1期間で何倍になるか(金利 $0$ なら $R=1$)。"],
                   solution=[rf"$q=\dfrac{{{dec(R)}-{dec(d)}}}{{{dec(u)}-{dec(d)}}}={ok}$。"], meta=m)


def g_delta(rng):
    m = binom(rng)
    ok = tex_fr(m["delta"])
    cands = [(tex_fr(-m["delta"]), "上がったときの価値から下がったときの価値を引く:$\\Delta=\\dfrac{C_u-C_d}{(u-d)S_0}$。"),
             (tex_fr((m["Cu"] - m["Cd"]) / (m["u"] - m["d"])), r"株価の差 $(u-d)S_0$ で割る。"),
             (tex_fr((m["Cu"] - m["Cd"]) / (m["u"] * m["S0"])), r"分母は上がったときと下がったときの株価の差。"),
             (tex_fr(m["q"]), r"これは $q$。$\Delta$ は株の数。")]
    return Problem("ゆらぎを打ち消す株の数", "基本",
                   bstate(m) + r", 上がっても下がっても権利と同じ価値になるように組み合わせる株の数 $\Delta$ を求めよ。",
                   distinct(ok, cands), hints=[r"$\Delta\cdot uS_0+RB=C_u$, $\Delta\cdot dS_0+RB=C_d$ を引き算する。"],
                   solution=[rf"$C_u={tex_fr(m['Cu'])}$, $C_d={tex_fr(m['Cd'])}$。",
                             rf"$\Delta=\dfrac{{C_u-C_d}}{{(u-d)S_0}}=\dfrac{{{tex_fr(m['Cu'] - m['Cd'])}}}{{{tex_fr((m['u'] - m['d']) * m['S0'])}}}={ok}$。"], meta=m)


# ------------------------------------------------------------
# 標準:1期間の二項モデルの値段
# ------------------------------------------------------------
def g_binom(rng):
    m = binom(rng)
    ok = yen(m["C"])
    R = m["R"]
    cands = [(yen((F(1, 2) * m["Cu"] + F(1, 2) * m["Cd"]) / R), r"確率を $\frac12$ ずつにするのではなく, $q=\dfrac{R-d}{u-d}$ を使う。"),
             (yen(m["q"] * m["Cu"] + (1 - m["q"]) * m["Cd"]) if R != 1 else yen((1 - m["q"]) * m["Cu"] + m["q"] * m["Cd"]),
              "預金の倍率 $R$ で割る(将来の価値を今の価値に直す)。" if R != 1 else r"$q$ は上がったときに掛ける。"),
             (yen(m["Cu"] / R), "下がったときの場合も考える。"), (yen((m["Cu"] + m["Cd"]) / R), "それぞれに $q$, $1-q$ を掛けて平均する。")]
    return Problem("1期間の二項モデル", "標準", bstate(m) + "の, 今の値段を求めよ。",
                   distinct(ok, cands), hints=[r"$q=\dfrac{R-d}{u-d}$, $C=\dfrac{qC_u+(1-q)C_d}{R}$。"],
                   solution=[rf"$C_u={tex_fr(m['Cu'])}$, $C_d={tex_fr(m['Cd'])}$, $q={tex_fr(m['q'])}$。",
                             rf"$C=\dfrac{{{tex_fr(m['q'])}\times{tex_fr(m['Cu'])}+{tex_fr(1 - m['q'])}\times{tex_fr(m['Cd'])}}}{{{dec(R)}}}={tex_fr(m['C'])}$ 円。"],
                   meta=m)


# ------------------------------------------------------------
# 基本:プット・コール・パリティ
# ------------------------------------------------------------
def g_parity(rng):
    S0 = rng.choice([900, 1000, 1100])
    Kd = rng.choice([950, 980, 1000])            # K e^{−rT}(割り引いた権利行使価格)
    C = rng.choice([60, 80, 100, 120])
    P = C - S0 + Kd
    if P <= 0:
        raise Retry
    want = rng.choice(["P", "C"])
    ok = yen(P if want == "P" else C)
    if want == "P":
        cands = [(yen(C + S0 - Kd), "$C-P=S_0-Ke^{-rT}$ から $P=C-S_0+Ke^{-rT}$。符号に注意。"), (yen(C), "コールとプットの値段は一般に違う。"),
                 (yen(abs(C - S0 + 1000)), r"$K$ は割り引いた値 $Ke^{-rT}$ を使う。" if Kd != 1000 else "符号を確かめよう。"),
                 (yen(S0 - Kd), r"$C$ も使う:$P=C-(S_0-Ke^{-rT})$。")]
        q = rf"コールの値段が ${C}$ 円のとき, 同じ $K$, 同じ満期のプットの値段を求めよ。"
        sol = [r"$C-P=S_0-Ke^{-rT}$ より", rf"$P={C}-{S0}+{Kd}={P}$ 円。"]
    else:
        cands = [(yen(P - S0 + Kd) if P - S0 + Kd > 0 else yen(P + 1), "$C=P+S_0-Ke^{-rT}$。符号に注意。"), (yen(P), "コールとプットの値段は一般に違う。"),
                 (yen(S0 - Kd) if S0 != Kd else yen(P + 10), r"$P$ も使う。"), (yen(P + S0 - 1000) if Kd != 1000 else yen(P + 20), r"$K$ は割り引いた値 $Ke^{-rT}$ を使う。")]
        q = rf"プットの値段が ${P}$ 円のとき, 同じ $K$, 同じ満期のコールの値段を求めよ。"
        sol = [r"$C-P=S_0-Ke^{-rT}$ より", rf"$C={P}+{S0}-{Kd}={C}$ 円。"]
    return Problem("プット・コール・パリティ", "基本",
                   rf"今の株価が $S_0={S0}$ 円, 権利行使価格 $K$ を満期まで金利で割り引いた値が $Ke^{{-rT}}={Kd}$ 円である。" + q,
                   distinct(ok, cands), hints=[r"満期では $\max(S-K,0)-\max(K-S,0)=S-K$。同じ将来の価値をもつものは今も同じ値段:$C-P=S_0-Ke^{-rT}$。"],
                   solution=sol, meta=dict(S0=S0, Kd=Kd, C=C, P=P, want=want))


# ------------------------------------------------------------
# 標準:x = log S への変換
# ------------------------------------------------------------
def coefs(sig, r):
    return sig * sig / 2, r - sig * sig / 2


def g_logx(rng):
    sig = rng.choice([F(1, 5), F(2, 5), F(1, 2), F(3, 10)])
    r = rng.choice([F(1, 50), F(1, 20), F(1, 10), F(1, 25)])
    a, b = coefs(sig, r)

    def eq(a_, b_):
        return (r"V_t+" + tex_fr(a_) + r"V_{xx}" + ("+" if b_ >= 0 else "-") + tex_fr(abs(b_)) + r"V_x-" + tex_fr(r) + "V=0")
    ok = eq(a, b)
    cands = [(eq(a, r), r"$S^2V_{SS}=V_{xx}-V_x$ なので, $V_x$ の係数は $r-\dfrac{\sigma^2}{2}$。"),
             (eq(a, r + a), r"$S^2V_{SS}=V_{xx}-V_x$:$-V_x$ の符号に注意。"),
             (eq(sig * sig, r - sig * sig), r"$\dfrac12\sigma^2$ の $\dfrac12$ を忘れていないか。"),
             (eq(2 * a, b), r"$\dfrac12\sigma^2$ をそのまま使う。")]
    return Problem("対数への変数変換", "標準",
                   rf"ブラック・ショールズ方程式 $V_t+\dfrac12\sigma^2S^2V_{{SS}}+rSV_S-rV=0$ で $\sigma={dec(sig)}$, $r={dec(r)}$ とする。"
                   r"$x=\log S$ とおくと, $SV_S=V_x$, $S^2V_{SS}=V_{xx}-V_x$ となる。$x$ と $t$ の方程式を求めよ。",
                   distinct(ok, cands), hints=[r"$\dfrac12\sigma^2(V_{xx}-V_x)+rV_x=\dfrac12\sigma^2V_{xx}+\left(r-\dfrac12\sigma^2\right)V_x$。"],
                   solution=[rf"$\dfrac12\sigma^2={tex_fr(a)}$, $r-\dfrac12\sigma^2={tex_fr(b)}$。", rf"${ok}$(係数が定数になり, 熱方程式に近い形になる)。"],
                   meta=dict(sig=sig, r=r, a=a, b=b))


# ------------------------------------------------------------
# 誘導
# ------------------------------------------------------------
def guided_binom(rng):
    m = binom(rng)
    return ExamItem("二項モデルによる複製",
                    bstate(m).replace("。", ". ") + r", 株 $\Delta$ 株と預金 $B$ 円を組み合わせて, 上がっても下がっても権利と同じ価値になるようにする. 次の問いに答えよ.",
                    [Part(r"$\Delta$ と $B$ を求めよ.", rf"\Delta={tex_fr(m['delta'])},\ B={tex_fr(m['B'])}",
                          [rf"$C_u={tex_fr(m['Cu'])}$, $C_d={tex_fr(m['Cd'])}$。",
                           rf"$\Delta\cdot{tex_fr(m['u'] * m['S0'])}+{dec(m['R'])}B={tex_fr(m['Cu'])}$, $\Delta\cdot{tex_fr(m['d'] * m['S0'])}+{dec(m['R'])}B={tex_fr(m['Cd'])}$ を解く。"]),
                     Part(r"権利の今の値段 $\Delta S_0+B$ を求めよ.", rf"{tex_fr(m['C'])}\ \text{{円}}",
                          [rf"$\Delta S_0+B={tex_fr(m['delta'] * m['S0'])}+({tex_fr(m['B'])})={tex_fr(m['C'])}$ 円。",
                           rf"($q={tex_fr(m['q'])}$ として $\dfrac{{qC_u+(1-q)C_d}}{{R}}$ でも同じ値になる。)"])], meta=m)


def guided_logx(rng):
    sig = rng.choice([F(1, 5), F(2, 5), F(1, 2)])
    r = rng.choice([F(1, 50), F(1, 20), F(1, 10)])
    a, b = coefs(sig, r)
    return ExamItem("対数への変数変換",
                    rf"ブラック・ショールズ方程式 $V_t+\dfrac12\sigma^2S^2V_{{SS}}+rSV_S-rV=0$($\sigma={dec(sig)}$, $r={dec(r)}$)で, $x=\log S$ とおく. 次の問いに答えよ.",
                    [Part(r"$V_S$ と $V_{SS}$ を $V_x$, $V_{xx}$ と $S$ で表し, $SV_S$, $S^2V_{SS}$ を求めよ.", r"SV_S=V_x,\ S^2V_{SS}=V_{xx}-V_x",
                          [r"$V_S=V_x\dfrac{dx}{dS}=\dfrac1SV_x$。", r"$V_{SS}=-\dfrac{1}{S^2}V_x+\dfrac1SV_{xx}\cdot\dfrac1S=\dfrac{1}{S^2}(V_{xx}-V_x)$。"]),
                     Part(r"$V$ が $x$, $t$ についてみたす方程式を求めよ.",
                          r"V_t+" + tex_fr(a) + r"V_{xx}" + ("+" if b >= 0 else "-") + tex_fr(abs(b)) + r"V_x-" + tex_fr(r) + "V=0",
                          [r"$\dfrac12\sigma^2(V_{xx}-V_x)+rV_x-rV$ を整理する。", rf"$\dfrac12\sigma^2={tex_fr(a)}$, $r-\dfrac12\sigma^2={tex_fr(b)}$。"])],
                    meta=dict(sig=sig, r=r, a=a, b=b))


REGISTRY = [
    ProblemType("fin_payoff", "満期の価値", "基本", g_payoff),
    ProblemType("fin_q", "リスク中立確率", "基本", g_q),
    ProblemType("fin_delta", "ゆらぎを打ち消す株の数", "基本", g_delta),
    ProblemType("fin_parity", "プット・コール・パリティ", "基本", g_parity),
    ProblemType("fin_binom", "1期間の二項モデル", "標準", g_binom, guided=guided_binom),
    ProblemType("fin_logx", "対数への変数変換", "標準", g_logx, guided=guided_logx),
]
GUIDED_NAMES = {"fin_binom": "株と預金で複製(Δ と B)→ 権利の値段",
                "fin_logx": "x = log S での微分 → 変換後の方程式"}


def tab():
    render("finance", REGISTRY, "偏微分方程式 発展6. 金融とブラック・ショールズ方程式")
