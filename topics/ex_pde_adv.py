"""数学IV 偏微分方程式 発展(流体・量子・半導体):演習の問題生成

試験向けに計算が単純になる数値を選ぶ。
"""
import math
from fractions import Fraction

from topics.ex_core import Choice, ExamItem, Part, Problem, ProblemType, Retry, render, sqrt_parts, tex_fr

F = Fraction


def distinct(ok, cands):
    choices = [Choice(ok, True, value=ok)]
    for t, why in cands:
        if all(c.tex != t for c in choices) and len(choices) < 4:
            choices.append(Choice(t, why=why, value=t))
    if len(choices) < 4:
        raise Retry
    return choices


def unit(v, u):
    return rf"{v}\ \mathrm{{{u}}}"


def num(x):
    x = F(x)
    return str(x.numerator) if x.denominator == 1 else tex_fr(x)


# ------------------------------------------------------------
# 流体:ベルヌーイの定理(ベンチュリ管)
# ------------------------------------------------------------
def venturi(rng):
    v1 = rng.choice([1, 2, 3])
    r = rng.choice([2, 3, 4])
    v2 = v1 * r
    dp = 500 * (v2 * v2 - v1 * v1)          # ρ = 1000
    return v1, r, v2, dp


def g_bern(rng):
    v1, r, v2, dp = venturi(rng)
    ok = unit(f"{dp / 1000:g}", "kPa")
    cands = [(unit(f"{1000 * (v2 * v2 - v1 * v1) / 1000:g}", "kPa"), r"$\frac12\rho v^2$ の $\frac12$ を忘れていないか。"),
             (unit(f"{500 * (v2 - v1) ** 2 / 1000:g}", "kPa"), r"$v_2^2-v_1^2$ と $(v_2-v_1)^2$ は違う。"),
             (unit(f"{500 * (v1 * r * r) ** 2 / 1000 - 500 * v1 * v1 / 1000:g}", "kPa"), r"連続の式は $A_1v_1=A_2v_2$:流速は面積の比に比例する(2乗ではない)。"),
             (unit(f"{500 * v2 * v2 / 1000:g}", "kPa"), r"$p_1+\frac12\rho v_1^2=p_2+\frac12\rho v_2^2$:$v_1$ の項も残る。")]
    return Problem("ベルヌーイの定理", "基本",
                   rf"水平な管を水(密度 $\rho=1000\ \mathrm{{kg/m^3}}$)が流れている。断面積 $A_1$ の部分の流速が $v_1={v1}\ \mathrm{{m/s}}$ で, "
                   rf"断面積 $A_2=\dfrac{{A_1}}{{{r}}}$ の細い部分へ流れ込む。圧力差 $p_1-p_2$ を求めよ。",
                   distinct(ok, cands), hints=[r"連続の式 $A_1v_1=A_2v_2$ で $v_2$ を求める。", r"ベルヌーイの定理 $p_1+\frac12\rho v_1^2=p_2+\frac12\rho v_2^2$。"],
                   solution=[rf"$v_2=\dfrac{{A_1}}{{A_2}}v_1={v2}\ \mathrm{{m/s}}$。",
                             rf"$p_1-p_2=\frac12\rho(v_2^2-v_1^2)=500\times({v2 * v2}-{v1 * v1})={dp}\ \mathrm{{Pa}}={dp / 1000:g}\ \mathrm{{kPa}}$。"],
                   meta=dict(v1=v1, r=r, dp=dp))


# ------------------------------------------------------------
# 流体:ポアズイユ流れ・ハーゲン・ポアズイユの法則
# ------------------------------------------------------------
def g_poise(rng):
    kind = rng.choice(["R", "mu", "mix"])
    if kind == "R":
        k = rng.choice([2, 3, F(1, 2)])
        ok = num(F(k) ** 4)
        cands = [(num(F(k) ** 2), r"流量は半径の4乗に比例する($Q=\frac{\pi PR^4}{8\mu}$)。"), (num(F(k) ** 3), "流量は半径の4乗に比例する。"),
                 (num(F(k)), r"速度は $R^2$, 断面積も $R^2$ に比例するので, 流量は $R^4$。")]
        q = rf"管の半径を ${num(k)}$ 倍にすると, 同じ圧力勾配で流れる流量は何倍になるか。"
        sol = [rf"$Q\propto R^4$ より $({num(k)})^4={ok}$ 倍。"]
    elif kind == "mu":
        k = rng.choice([2, 4, 10])
        ok = tex_fr(F(1, k))
        cands = [(str(k), r"粘性が大きいほど流れにくい:$Q\propto\frac{1}{\mu}$。"), (tex_fr(F(1, k * k)), r"$Q$ は $\mu$ に反比例(2乗ではない)。"),
                 ("1", r"$Q=\frac{\pi PR^4}{8\mu}$ は $\mu$ によって変わる。")]
        q = rf"粘性係数が ${k}$ 倍の液体に変えると, 同じ圧力勾配で流れる流量は何倍になるか。"
        sol = [rf"$Q\propto\dfrac1\mu$ より ${ok}$ 倍。"]
    else:
        k = rng.choice([2, 3])
        ok = tex_fr(F(1, 2))
        cands = [(tex_fr(F(2, 3)), r"円管では平均流速は最大流速の $\frac12$(平行平板では $\frac23$)。"), ("1", "流速は中心で最大, 壁で $0$。"),
                 (tex_fr(F(1, 3)), r"$\frac{1}{\pi R^2}\int_0^R\frac{P}{4\mu}(R^2-r^2)2\pi r\,dr$ を計算し直そう。")]
        q = r"円管の中のポアズイユ流れ $u(r)=\dfrac{P}{4\mu}(R^2-r^2)$ で, 断面の平均流速は最大流速(中心の流速)の何倍か。"
        sol = [r"平均 $=\dfrac{1}{\pi R^2}\displaystyle\int_0^R\frac{P}{4\mu}(R^2-r^2)\,2\pi r\,dr=\frac{PR^2}{8\mu}$, 最大 $=\dfrac{PR^2}{4\mu}$。", rf"比は ${ok}$。"]
    return Problem("ハーゲン・ポアズイユの法則", "基本", q, distinct(ok, cands),
                   hints=[r"円管の層流の流量 $Q=\dfrac{\pi PR^4}{8\mu}$($P=-\dfrac{dp}{dx}$)。"], solution=sol, meta=dict(kind=kind))


# ------------------------------------------------------------
# 流体:レイノルズ数
# ------------------------------------------------------------
def g_re(rng):
    V = rng.choice([F(1, 10), F(1, 2), F(1), F(2)])
    D = rng.choice([F(1, 100), F(2, 100), F(5, 100), F(1, 10)])
    fl, nu = rng.choice([("水", F(1, 10 ** 6)), ("油", F(1, 10 ** 4))])
    Re = V * D / nu
    if Re.denominator != 1:
        raise Retry
    Re = int(Re)
    lam = Re < 2300
    ok = rf"Re={Re},\ \text{{{'層流' if lam else '乱流'}}}"
    cands = [(rf"Re={Re},\ \text{{{'乱流' if lam else '層流'}}}", "円管では $Re$ がおよそ $2300$ より小さいと層流。"),
             (rf"Re={int(V * D * 100 / nu)},\ \text{{{'層流' if V * D * 100 / nu < 2300 else '乱流'}}}", r"直径は $\mathrm{m}$ に直してから計算する。"),
             (rf"Re={int(V / (D * nu)) if (V / (D * nu)).denominator == 1 else round(float(V / (D * nu)))},\ \text{{{'層流' if V / (D * nu) < 2300 else '乱流'}}}",
              r"$Re=\dfrac{VD}{\nu}$:$D$ は分子。")]
    Dcm = D * 100
    nut = r"1.0\times10^{-6}" if fl == "水" else r"1.0\times10^{-4}"
    return Problem("レイノルズ数", "基本",
                   rf"直径 ${num(Dcm)}\ \mathrm{{cm}}$ の円管を{fl}(動粘性係数 $\nu={nut}\ \mathrm{{m^2/s}}$)が "
                   rf"${num(V)}\ \mathrm{{m/s}}$ で流れる。レイノルズ数 $Re=\dfrac{{VD}}{{\nu}}$ を求め, 層流か乱流か判定せよ(境目を $2300$ とする)。",
                   distinct(ok, cands), hints=[r"$Re=\dfrac{VD}{\nu}$。単位を $\mathrm{m}$, $\mathrm{s}$ にそろえる。"],
                   solution=[rf"$Re=\dfrac{{{num(V)}\times{num(D)}}}{{\nu}}={Re}$。", rf"$2300$ より{'小さい' if lam else '大きい'}ので{'層流' if lam else '乱流'}。"],
                   meta=dict(V=V, D=D, nu=nu, Re=Re))


# ------------------------------------------------------------
# 流体:平行平板の間の流れ(標準)
# ------------------------------------------------------------
def g_couette(rng):
    h = rng.choice([1, 2])
    U = rng.choice([0, 1, 2])
    Pm = rng.choice([4, 8, 16])              # P/μ
    # y = h/2 での流速:U/2 + (P/μ) h²/8
    val = F(U, 2) + F(Pm * h * h, 8)
    ok = num(val)
    cands = [(num(F(U, 2) + F(Pm * h * h, 4)), r"$y(h-y)$ は $y=\frac h2$ で $\frac{h^2}{4}$, それに $\frac{P}{2\mu}$ を掛ける。"),
             (num(F(U, 2) - F(Pm * h * h, 8)) if U else num(F(Pm * h * h, 2)), r"$\mu u^{\prime\prime}=-P$ を積分したときの符号に注意(押す向きに流れる)。" if U else "係数を確かめよう。"),
             (num(F(U) + F(Pm * h * h, 8)), r"クエット流れの部分は $\frac{U}{h}y$:$y=\frac h2$ では $\frac U2$。"),
             (num(F(Pm * h * h, 8)) if U else num(F(Pm * h * h, 12)), r"上の板の動き($\frac{U}{h}y$)の分も足す。" if U else "中央の値を求める(平均ではない)。")]
    return Problem("平行平板の間の流れ", "標準",
                   rf"間隔 $h={h}$ の平行な2枚の板の間の定常な流れは $\mu\dfrac{{d^2u}}{{dy^2}}=-P$ をみたす($P=-\dfrac{{dp}}{{dx}}$)。"
                   rf"下の板は静止 $u(0)=0$, 上の板は $u(h)={U}$ で動き, $\dfrac{{P}}{{\mu}}={Pm}$ のとき, 中央 $y=\dfrac h2$ での流速を求めよ。",
                   distinct(ok, cands), hints=[r"2回積分して $u=-\frac{P}{2\mu}y^2+c_1y+c_2$, 境界条件で $c_1,c_2$ を決める。",
                                               r"$u(y)=\frac{U}{h}y+\frac{P}{2\mu}y(h-y)$。"],
                   solution=[r"$u(y)=\dfrac{U}{h}y+\dfrac{P}{2\mu}y(h-y)$。",
                             rf"$u\left(\frac h2\right)=\dfrac{{U}}{{2}}+\dfrac{{P}}{{\mu}}\cdot\dfrac{{h^2}}{{8}}={tex_fr(F(U, 2))}+{tex_fr(F(Pm * h * h, 8))}={ok}$。"],
                   meta=dict(h=h, U=U, Pm=Pm, val=val))


# ------------------------------------------------------------
# 量子:箱の中の粒子
# ------------------------------------------------------------
def g_box(rng):
    kind = rng.choice(["ratio", "diff", "L"])
    if kind == "ratio":
        n = rng.choice([2, 3, 4])
        ok = str(n * n)
        cands = [(str(n), r"$E_n=\frac{n^2h^2}{8mL^2}$:$n^2$ に比例する。"), (str(2 * n), r"$E_n\propto n^2$。"), (str(n ** 3), r"$E_n\propto n^2$。")]
        q = rf"長さ $L$ の箱の中の粒子のエネルギー $E_n=\dfrac{{n^2h^2}}{{8mL^2}}$ について, $E_{n}$ は $E_1$ の何倍か。"
        sol = [rf"$E_n\propto n^2$ より ${n}^2={ok}$ 倍。"]
    elif kind == "diff":
        a, b = sorted(rng.sample([1, 2, 3, 4], 2))
        def e1(c):
            return "E_1" if c == 1 else f"{c}E_1"
        ok = e1(b * b - a * a)
        cands = [(e1((b - a) ** 2), r"$E_b-E_a=(b^2-a^2)E_1$:$(b-a)^2$ ではない。"), (e1(b - a), r"$E_n=n^2E_1$。"),
                 (e1(b * b + a * a), "差をとる。"), (e1(b + a + 1), r"$E_n=n^2E_1$ を使う。")]
        q = rf"箱の中の粒子が $n={b}$ の状態から $n={a}$ の状態へ移るとき, 放出する光のエネルギー $E_{b}-E_{a}$ を $E_1$ で表せ。"
        sol = [rf"$E_{b}-E_{a}=({b * b}-{a * a})E_1={ok}$。"]
    else:
        k = rng.choice([2, 3, F(1, 2)])
        ok = num(1 / F(k) ** 2)
        cands = [(num(1 / F(k)), r"$E\propto\frac{1}{L^2}$:2乗に反比例する。"), (num(F(k) ** 2), "箱が大きいほどエネルギーは小さい。"), (num(F(k)), r"$E\propto\frac{1}{L^2}$。")]
        q = rf"箱の長さ $L$ を ${num(k)}$ 倍にすると, 基底状態のエネルギー $E_1$ は何倍になるか。"
        sol = [rf"$E_1\propto\dfrac{{1}}{{L^2}}$ より ${ok}$ 倍。"]
    return Problem("箱の中の粒子", "基本", q, distinct(ok, cands), hints=[r"$E_n=\dfrac{n^2h^2}{8mL^2}$ は $n^2$ に比例し, $L^2$ に反比例する。"],
                   solution=sol, meta=dict(kind=kind))


# ------------------------------------------------------------
# 量子:確率(規格化と区間の確率)
# ------------------------------------------------------------
def g_prob(rng):
    kind = rng.choice(["norm", "quarter"])
    if kind == "norm":
        L = rng.choice([1, 2, 4, 8])
        ok = r"\sqrt{2}" if L == 1 else ("1" if L == 2 else (r"\frac{1}{\sqrt{2}}" if L == 4 else r"\frac{1}{2}"))
        cands = [(r"\sqrt{2}" if L != 1 else "1", r"$\int_0^L\sin^2\frac{n\pi x}{L}dx=\frac L2$。"),
                 (r"\frac{2}{" + str(L) + "}" if L != 2 else r"\frac{1}{2}", r"$A^2\cdot\frac L2=1$ から $A^2$ を出したあと, 平方根をとる。"),
                 (r"\frac{1}{" + str(L) + "}" if L != 1 else r"\frac{1}{2}", r"$\sin^2$ の積分は $\frac L2$。"), ("2", r"$A=\sqrt{\frac2L}$。")]
        n = rng.choice([1, 2, 3])
        q = rf"長さ $L={L}$ の箱の中の粒子の波動関数 $\varphi(x)=A\sin\dfrac{{{n if n > 1 else ''}\pi x}}{{{L}}}$($0\le x\le{L}$)が $\displaystyle\int_0^{{{L}}}|\varphi|^2dx=1$ をみたす正の定数 $A$ を求めよ。"
        sol = [rf"$\int_0^{{{L}}}\sin^2\frac{{{n if n > 1 else ''}\pi x}}{{{L}}}dx=\frac{{{L}}}{{2}}$ より $A^2\cdot\frac{{{L}}}{{2}}=1$。", rf"$A=\sqrt{{\frac{{2}}{{{L}}}}}={ok}$。"]
        meta = dict(kind=kind, L=L, n=n)
    else:
        n = rng.choice([1, 2, 3])
        # P(0 ≤ x ≤ L/4) = 1/4 − sin(nπ/2)/(2nπ)
        s = [0, 1, 0, -1][n % 4]
        if s == 0:
            ok = r"\frac{1}{4}"
        else:
            ok = r"\frac{1}{4}" + ("-" if s > 0 else "+") + rf"\frac{{1}}{{{2 * n if n > 1 else 2}\pi}}"
        alt = r"\frac{1}{4}" + ("+" if s >= 0 else "-") + rf"\frac{{1}}{{{2 * n}\pi}}"
        cands = [(alt, "符号を確かめよう($\\sin^2$ の積分は $\\frac x2-\\frac{L}{4n\\pi}\\sin\\frac{2n\\pi x}{L}$)。"), (r"\frac{1}{8}", r"$|\varphi|^2=\frac2L\sin^2$:係数 $\frac2L$ を掛ける。"),
                 (r"\frac{1}{2}", "区間は $L/4$ まで。"), (r"\frac{1}{4}-\frac{1}{\pi}", r"$\int\sin^2\theta\,d\theta=\frac{\theta}{2}-\frac{\sin2\theta}{4}$。")]
        q = rf"箱の中の粒子が $\varphi_{n}=\sqrt{{\dfrac{{2}}{{L}}}}\sin\dfrac{{{n if n > 1 else ''}\pi x}}{{L}}$ の状態にあるとき, $0\le x\le\dfrac L4$ に見つかる確率を求めよ。"
        sol = [rf"$\displaystyle\int_0^{{L/4}}\frac2L\sin^2\frac{{{n if n > 1 else ''}\pi x}}{{L}}dx=\frac14-\frac{{\sin({n if n > 1 else ''}\pi/2)}}{{{2 * n}\pi}}$。", rf"よって ${ok}$。"]
        meta = dict(kind=kind, n=n)
    return Problem("波動関数と確率", "標準", q, distinct(ok, cands), hints=[r"$|\varphi|^2$ は見つかる確率の密度。$\sin^2\theta=\frac{1-\cos2\theta}{2}$。"],
                   solution=sol, meta=meta)


# ------------------------------------------------------------
# 半導体:アインシュタインの関係, 拡散長, ダイオード
# ------------------------------------------------------------
def g_einstein(rng):
    mu = rng.choice([400, 480, 1200, 1400, 1600, 2000])
    D = F(mu, 40)
    ok = unit(num(D), "cm^2/s")
    cands = [(unit(str(mu * 40), "cm^2/s"), r"$D=\frac{kT}{q}\mu$:$0.025$ を掛ける(割らない)。"), (unit(num(F(mu, 400)), "cm^2/s"), "桁を確かめよう。"),
             (unit(num(F(mu, 20)), "cm^2/s"), r"$kT/q=0.025\ \mathrm{V}$。")]
    return Problem("アインシュタインの関係", "基本",
                   rf"移動度 $\mu={mu}\ \mathrm{{cm^2/(V\cdot s)}}$ の電子の, $300\ \mathrm{{K}}$ での拡散係数 $D$ を求めよ($kT/q=0.025\ \mathrm{{V}}$ とする)。",
                   distinct(ok, cands), hints=[r"平衡状態でドリフトと拡散が釣り合う条件から $D=\dfrac{kT}{q}\mu$。"],
                   solution=[rf"$D=0.025\times{mu}={num(D)}\ \mathrm{{cm^2/s}}$。"], meta=dict(mu=mu, D=D))


def g_difflen(rng):
    D = rng.choice([1, 4, 9, 16, 25, 36])
    tau_us = rng.choice([1, 4, 100])
    L_cm2 = D * tau_us                 # L² = D τ, τ = tau_us × 10⁻⁶ s → L = √(D τ_us) × 10⁻³ cm = √(D τ_us) × 10 μm
    k, m = sqrt_parts(L_cm2)
    if m != 1:
        raise Retry
    Lum = 10 * k
    ok = unit(str(Lum), r"\mu m")
    cands = [(unit(str(Lum * 10), r"\mu m"), r"$10^{-3}\ \mathrm{cm}=10\ \mu\mathrm{m}$。単位の換算を確かめよう。"), (unit(str(D * tau_us), r"\mu m"), r"$L=\sqrt{D\tau}$:平方根をとる。"),
             (unit(str(Lum // 10) if Lum >= 10 and Lum % 10 == 0 else str(Lum * 100), r"\mu m"), r"$\sqrt{10^{-6}}=10^{-3}$。")]
    return Problem("拡散長", "標準",
                   rf"少数キャリアの拡散係数が $D={D}\ \mathrm{{cm^2/s}}$, 寿命が $\tau={tau_us}\ \mu\mathrm{{s}}$ のとき, 拡散長 $L=\sqrt{{D\tau}}$ を $\mu\mathrm{{m}}$ で求めよ。",
                   distinct(ok, cands), hints=[r"$\tau$ を秒に直す:$1\ \mu\mathrm s=10^{-6}\ \mathrm s$。", r"$1\ \mathrm{cm}=10^4\ \mu\mathrm m$。"],
                   solution=[rf"$L=\sqrt{{{D}\times{tau_us}\times10^{{-6}}}}={k}\times10^{{-3}}\ \mathrm{{cm}}={Lum}\ \mu\mathrm{{m}}$。"], meta=dict(D=D, tau=tau_us, L=Lum))


def g_diode(rng):
    k = rng.choice([1, 2, 3])
    dV = 60 * k
    ok = rf"10^{{{k}}}" if k > 1 else "10"
    cands = [(str(10 * k) if k > 1 else "2", r"$e^{qV/kT}$:$60\ \mathrm{mV}$ ごとに $10$ 倍になる(足し算ではなく掛け算)。"), (rf"{k}", "指数関数で増える。"),
             (rf"10^{{{k + 1}}}", r"$\frac{kT}{q}\ln10\approx60\ \mathrm{mV}$ で $10$ 倍。")]
    return Problem("ダイオードの電流", "基本",
                   rf"ショックレーの式 $I=I_s\left(e^{{qV/kT}}-1\right)$ で, 順方向の電圧を ${dV}\ \mathrm{{mV}}$ 上げると電流は約何倍になるか"
                   rf"($e^{{qV/kT}}\gg1$, $\dfrac{{kT}}{{q}}\ln10\approx60\ \mathrm{{mV}}$ とする)。",
                   distinct(ok, cands), hints=[r"$e^{q\Delta V/kT}$ 倍。$\Delta V=\frac{kT}{q}\ln10$ なら $e^{\ln10}=10$ 倍。"],
                   solution=[rf"${dV}\ \mathrm{{mV}}={k}\times60\ \mathrm{{mV}}$ より $10^{{{k}}}$ 倍。"], meta=dict(k=k))


# ------------------------------------------------------------
# 誘導
# ------------------------------------------------------------
def guided_venturi(rng):
    v1, r, v2, dp = venturi(rng)
    return ExamItem("連続の式とベルヌーイの定理",
                    rf"水平な管を水(密度 $\rho=1000\ \mathrm{{kg/m^3}}$)が流れている. 断面積 $A_1$ の部分の流速は $v_1={v1}\ \mathrm{{m/s}}$ で, "
                    rf"断面積 $A_2=\dfrac{{A_1}}{{{r}}}$ の部分へ流れ込む. 次の問いに答えよ.",
                    [Part(r"細い部分の流速 $v_2$ を求めよ.", rf"v_2={v2}\ \mathrm{{m/s}}", [rf"連続の式 $A_1v_1=A_2v_2$ より $v_2={r}v_1={v2}\ \mathrm{{m/s}}$。"]),
                     Part(r"ベルヌーイの定理を用いて, 圧力差 $p_1-p_2$ を求めよ.", rf"p_1-p_2={dp}\ \mathrm{{Pa}}",
                          [rf"$p_1-p_2=\frac12\rho(v_2^2-v_1^2)=500\times({v2 * v2}-{v1 * v1})={dp}\ \mathrm{{Pa}}$。"])], meta=dict(v1=v1, r=r, dp=dp))


def guided_poise(rng):
    h = rng.choice([1, 2, 3])
    return ExamItem("平行平板の間のポアズイユ流れ",
                    rf"間隔 $h={h}$ の静止した平行な2枚の板の間の定常な流れの速度 $u(y)$ は $\mu\dfrac{{d^2u}}{{dy^2}}=-P$($P>0$ は定数), $u(0)=u({h})=0$ をみたす. 次の問いに答えよ.",
                    [Part(r"$u(y)$ を求めよ.", rf"u(y)=\frac{{P}}{{2\mu}}y({h}-y)",
                          [r"2回積分して $u=-\dfrac{P}{2\mu}y^2+c_1y+c_2$。", rf"$u(0)=0$ より $c_2=0$, $u({h})=0$ より $c_1=\dfrac{{{h}P}}{{2\mu}}$。"]),
                     Part(r"最大流速と, 断面の平均流速 $\dfrac{1}{h}\displaystyle\int_0^hu\,dy$ を求めよ.",
                          rf"\text{{最大}}\ \frac{{{h * h}P}}{{8\mu}},\ \text{{平均}}\ \frac{{{h * h}P}}{{12\mu}}" if h > 1 else r"\text{最大}\ \frac{P}{8\mu},\ \text{平均}\ \frac{P}{12\mu}",
                          [rf"最大は $y=\frac{{{h}}}{{2}}$ で $\dfrac{{P}}{{2\mu}}\cdot\dfrac{{{h * h}}}{{4}}$。",
                           rf"平均は $\dfrac{{1}}{{{h}}}\displaystyle\int_0^{{{h}}}\frac{{P}}{{2\mu}}y({h}-y)\,dy=\frac{{P}}{{2\mu}}\cdot\frac{{{h * h}}}{{6}}$(最大の $\frac23$)。"])], meta=dict(h=h))


def guided_box(rng):
    L = rng.choice([1, 2, 4])
    return ExamItem("箱の中の粒子",
                    rf"長さ $L={L}$ の箱の中の粒子の波動関数 $\varphi(x)$ は $-\dfrac{{\hbar^2}}{{2m}}\varphi^{{\prime\prime}}=E\varphi$, $\varphi(0)=\varphi({L})=0$ をみたす. 次の問いに答えよ.",
                    [Part(r"$0$ でない解 $\varphi$ と, そのときのエネルギー $E$ を求めよ($k=\dfrac{\sqrt{2mE}}{\hbar}$ とおく).",
                          rf"\varphi=A\sin\frac{{n\pi x}}{{{L}}},\ E_n=\frac{{n^2\pi^2\hbar^2}}{{{2 * L * L}m}}\ (n=1,2,\dots)",
                          [r"$\varphi^{\prime\prime}=-k^2\varphi$ より $\varphi=A\sin kx+B\cos kx$, $\varphi(0)=0$ から $B=0$。",
                           rf"$\varphi({L})=A\sin {L}k=0$ より $k=\dfrac{{n\pi}}{{{L}}}$, $E=\dfrac{{\hbar^2k^2}}{{2m}}=\dfrac{{n^2\pi^2\hbar^2}}{{{2 * L * L}m}}$。"]),
                     Part(rf"$\displaystyle\int_0^{{{L}}}|\varphi|^2dx=1$ となる正の定数 $A$ を求めよ.",
                          r"A=\sqrt{2}" if L == 1 else ("A=1" if L == 2 else r"A=\frac{1}{\sqrt{2}}"),
                          [rf"$\int_0^{{{L}}}\sin^2\frac{{n\pi x}}{{{L}}}dx=\frac{{{L}}}{{2}}$ より $A^2\cdot\frac{{{L}}}{{2}}=1$。"])], meta=dict(L=L))


def guided_semi(rng):
    mu, tau, k = rng.choice([(400, 10, 10), (1600, 10, 20), (1440, 1, 6), (1000, 1, 5)])
    D = F(mu, 40)
    if F(k * k) != D * tau:
        raise Retry
    Lum = 10 * k
    return ExamItem("アインシュタインの関係と拡散長",
                    rf"シリコン中の少数キャリア(電子)の移動度を $\mu={mu}\ \mathrm{{cm^2/(V\cdot s)}}$, 寿命を $\tau={tau}\ \mu\mathrm{{s}}$ とする. "
                    r"$kT/q=0.025\ \mathrm{V}$ として, 次の問いに答えよ.",
                    [Part(r"アインシュタインの関係 $D=\dfrac{kT}{q}\mu$ から拡散係数 $D$ を求めよ.", rf"D={num(D)}\ \mathrm{{cm^2/s}}", [rf"$D=0.025\times{mu}={num(D)}\ \mathrm{{cm^2/s}}$。"]),
                     Part(r"拡散長 $L=\sqrt{D\tau}$ を $\mu\mathrm{m}$ で求めよ.", rf"L={Lum}\ \mu\mathrm{{m}}",
                          [rf"$L=\sqrt{{{num(D)}\times{tau}\times10^{{-6}}}}={k}\times10^{{-3}}\ \mathrm{{cm}}={Lum}\ \mu\mathrm{{m}}$。"])],
                    meta=dict(mu=mu, tau=tau, L=Lum))


REGISTRY = [
    ProblemType("fl_bern", "ベルヌーイの定理", "基本", g_bern, guided=guided_venturi),
    ProblemType("fl_poise", "ハーゲン・ポアズイユの法則", "基本", g_poise),
    ProblemType("fl_re", "レイノルズ数", "基本", g_re),
    ProblemType("fl_couette", "平行平板の間の流れ", "標準", g_couette, guided=guided_poise),
    ProblemType("qm_box", "箱の中の粒子", "基本", g_box, guided=guided_box),
    ProblemType("qm_prob", "波動関数と確率", "標準", g_prob),
    ProblemType("sc_einstein", "アインシュタインの関係", "基本", g_einstein, guided=guided_semi),
    ProblemType("sc_diode", "ダイオードの電流", "基本", g_diode),
    ProblemType("sc_difflen", "拡散長", "標準", g_difflen),
]
GUIDED_NAMES = {"fl_bern": "連続の式で流速 → ベルヌーイの定理で圧力差",
                "fl_couette": "ポアズイユ流れの速度分布 → 最大流速と平均流速",
                "qm_box": "箱の中の粒子の固有関数とエネルギー → 規格化",
                "sc_einstein": "アインシュタインの関係で拡散係数 → 拡散長"}


def tab():
    render("pdeadv", REGISTRY, "偏微分方程式 発展(流体・量子・半導体)")
