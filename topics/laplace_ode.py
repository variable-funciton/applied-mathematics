"""② ラプラス変換 第2章 0. 線形微分方程式の解(準備)

1階 y' + a y = f(t)        :積分因子 e^{at}
2階 y'' + a y' + b y = f(t):特性方程式 λ² + aλ + b = 0, 未定係数法, 定数変化法(ロンスキアン)
(係数 a, b は定数。独立変数は t)
"""
import math
from fractions import Fraction

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import C1, C2, C3, C4, GRAY, layout, show, trapz
from topics.fourier_kernels import page_refs

F = Fraction


# ------------------------------------------------------------
# TeX の道具
# ------------------------------------------------------------
def fr(x):
    x = F(x)
    if x.denominator == 1:
        return str(x.numerator)
    return ("-" if x < 0 else "") + rf"\frac{{{abs(x.numerator)}}}{{{x.denominator}}}"


def num_tex(x, d=4):
    """小さい分母の有理数なら分数で, それ以外は小数(≈)で"""
    q = F(x).limit_denominator(200)
    if abs(float(q) - x) < 1e-10:
        return fr(q)
    return f"{x:.{d}g}"


def ssum(terms):
    """[(係数, 本体)] の和(係数は数または TeX 文字列)"""
    out = ""
    for c, body in terms:
        if isinstance(c, str):
            out += ("" if not out or c.startswith("-") else "+") + c + body
            continue
        c = F(c)
        if c == 0:
            continue
        sgn = "-" if c < 0 else ("+" if out else "")
        m = abs(c)
        coef = fr(m) if (body == "" or m != 1) else ""
        out += sgn + coef + body
    return out or "0"


def e_t(r):
    """e^{rt}(r は TeX 文字列または有理数)"""
    if isinstance(r, str):
        return rf"e^{{{r}t}}"
    r = F(r)
    if r == 0:
        return ""
    if r == 1:
        return "e^{t}"
    if r == -1:
        return "e^{-t}"
    return rf"e^{{{fr(r)}t}}"


def trig(name, w):
    w = F(w)
    return rf"\{name} {'' if w == 1 else fr(w)}t"


# ------------------------------------------------------------
# 特性根と基本解
# ------------------------------------------------------------
def surd(m, D, sg):
    """(m + sg·√D)/2 を簡単にした TeX(D は平方数でない)"""
    k, r = 1, D
    for p in range(2, int(math.isqrt(D)) + 1):
        while r % (p * p) == 0:
            r //= p * p
            k *= p
    g = math.gcd(math.gcd(abs(m), k), 2) if m else math.gcd(k, 2)
    m2, k2, d2 = m // g, k // g, 2 // g
    root = (str(k2) if k2 != 1 else "") + rf"\sqrt{{{r}}}"
    if m2 == 0:
        body = root if d2 == 1 else rf"\frac{{{root}}}{{{d2}}}"
        return ("-" if sg < 0 else "") + body
    num = f"{m2}{'+' if sg > 0 else '-'}{root}"
    return num if d2 == 1 else rf"\frac{{{num}}}{{{d2}}}"


def char_roots(a, b):
    """('real', λ1, λ2) / ('double', λ) / ('complex', p, q)(λ = p ± qi)。数値と TeX を返す"""
    D = a * a - 4 * b
    if D > 0:
        s = math.isqrt(D)
        if s * s == D:
            l1, l2 = F(-a + s, 2), F(-a - s, 2)
            return dict(kind="real", l=(float(l1), float(l2)), tex=(fr(l1), fr(l2)), exact=(l1, l2))
        sq = math.sqrt(D)
        return dict(kind="real", l=((-a + sq) / 2, (-a - sq) / 2), tex=(surd(-a, D, 1), surd(-a, D, -1)), exact=None)
    if D == 0:
        lam = F(-a, 2)
        return dict(kind="double", l=(float(lam),), tex=(fr(lam),), exact=(lam,))
    p = F(-a, 2)
    s = math.isqrt(-D)
    qt = fr(F(s, 2)) if s * s == -D else surd(0, -D, 1)
    return dict(kind="complex", l=(float(p), math.sqrt(-D) / 2), tex=(fr(p), qt), p=p)


def basis(R):
    """基本解 y1, y2 の (数値関数, 導関数, TeX)"""
    if R["kind"] == "real":
        l1, l2 = R["l"]
        return ((lambda t: np.exp(l1 * t), lambda t: l1 * np.exp(l1 * t), e_root(R, 0)),
                (lambda t: np.exp(l2 * t), lambda t: l2 * np.exp(l2 * t), e_root(R, 1)))
    if R["kind"] == "double":
        lm = R["l"][0]
        et = e_root(R, 0)
        return ((lambda t: np.exp(lm * t), lambda t: lm * np.exp(lm * t), et or "1"),
                (lambda t: t * np.exp(lm * t), lambda t: (1 + lm * t) * np.exp(lm * t), "t" + et))
    p, q = R["l"]
    ep = e_t(R["p"])
    qt = R["tex"][1]
    qtt = "" if qt == "1" else qt
    return ((lambda t: np.exp(p * t) * np.cos(q * t), lambda t: np.exp(p * t) * (p * np.cos(q * t) - q * np.sin(q * t)), ep + rf"\cos {qtt}t"),
            (lambda t: np.exp(p * t) * np.sin(q * t), lambda t: np.exp(p * t) * (p * np.sin(q * t) + q * np.cos(q * t)), ep + rf"\sin {qtt}t"))


def e_root(R, i):
    if R["exact"]:
        return e_t(R["exact"][i])
    tx = R["tex"][i]
    if not tx.startswith("\\frac") and any(ch in tx[1:] for ch in "+-"):
        tx = "(" + tx + ")"
    return rf"e^{{{tx}t}}"


def general_tex(R):
    (_, _, t1), (_, _, t2) = basis(R)
    if R["kind"] == "complex":
        ep = e_t(R["p"])
        qt = R["tex"][1]
        qtt = "" if qt == "1" else qt
        inner = rf"C_1\cos {qtt}t+C_2\sin {qtt}t"
        return ep + rf"\left({inner}\right)" if ep else inner
    if R["kind"] == "double":
        e = e_root(R, 0)
        return rf"(C_1+C_2t){e}" if e else "C_1+C_2t"
    return rf"C_1{t1 or ''}+C_2{t2 or ''}"


# ------------------------------------------------------------
# 未定係数法による特殊解(f = c, c t, c e^{kt}, c cos ωt, c sin ωt)
# ------------------------------------------------------------
def particular(a, b, kind, c, k=1, w=1):
    """(数値関数 yp, 導関数, TeX, 置き方の説明)。a, b, c, k, w は整数"""
    a, b, c = F(a), F(b), F(c)
    if kind == "0":
        return (lambda t: 0 * t), (lambda t: 0 * t), "0", "斉次方程式なので特殊解は不要。"
    if kind == "const":
        if b != 0:
            A = c / b
            return (lambda t: float(A) + 0 * t), (lambda t: 0 * t), fr(A), rf"$y_p=A$(定数)とおくと $bA=c$。"
        if a != 0:
            A = c / a
            return (lambda t: float(A) * t), (lambda t: float(A) + 0 * t), ssum([(A, "t")]), r"$b=0$ なので $y_p=A$ では合わない($\lambda=0$ が特性根)。$y_p=At$ とおく。"
        A = c / 2
        return (lambda t: float(A) * t ** 2), (lambda t: 2 * float(A) * t), ssum([(A, "t^2")]), r"$\lambda=0$ が重解なので $y_p=At^2$ とおく。"
    if kind == "t":
        if b != 0:
            A, B = c / b, -a * c / (b * b)       # y = A t + B: aA + b(At + B) = ct
            return (lambda t: float(A) * t + float(B)), (lambda t: float(A) + 0 * t), ssum([(A, "t"), (B, "")]), r"$y_p=At+B$ とおいて係数を比べる。"
        if a != 0:
            A, B = c / (2 * a), -c / (a * a)     # y = A t² + B t: 2A + a(2At + B) = ct
            return (lambda t: float(A) * t ** 2 + float(B) * t), (lambda t: 2 * float(A) * t + float(B)), ssum([(A, "t^2"), (B, "t")]), \
                r"$\lambda=0$ が特性根なので $y_p=At^2+Bt$ とおく。"
        A = c / 6
        return (lambda t: float(A) * t ** 3), (lambda t: 3 * float(A) * t ** 2), ssum([(A, "t^3")]), r"$\lambda=0$ が重解なので $y_p=At^3$ とおく。"
    if kind == "exp":
        k = F(k)
        P, dP = k * k + a * k + b, 2 * k + a
        if P != 0:
            A = c / P
            return (lambda t: float(A) * np.exp(float(k) * t)), (lambda t: float(A * k) * np.exp(float(k) * t)), ssum([(A, e_t(k) or "")]), \
                rf"$y_p=Ae^{{{fr(k)}t}}$ とおくと $A(k^2+ak+b)=c$。"
        if dP != 0:
            A = c / dP
            return (lambda t: float(A) * t * np.exp(float(k) * t)), (lambda t: float(A) * (1 + float(k) * t) * np.exp(float(k) * t)), \
                ssum([(A, "t" + e_t(k))]), rf"$k={fr(k)}$ が特性根(単根)なので $y_p=Ate^{{{fr(k)}t}}$ とおく(共鳴)。"
        A = c / 2
        return (lambda t: float(A) * t ** 2 * np.exp(float(k) * t)), (lambda t: float(A) * (2 * t + float(k) * t ** 2) * np.exp(float(k) * t)), \
            ssum([(A, "t^2" + e_t(k))]), rf"$k={fr(k)}$ が重解なので $y_p=At^2e^{{{fr(k)}t}}$ とおく。"
    # cos / sin:複素数 e^{iωt} で考える。P(iω) = (b − ω²) + i aω
    w = F(w)
    re, im = b - w * w, a * w
    den = re * re + im * im
    if den != 0:
        # c e^{iωt} / P(iω) = c (re − i im)/den (cos + i sin)
        cr, ci = c * re / den, -c * im / den
        if kind == "cos":       # 実部:cr cos + (−ci) sin
            Ac, As = cr, -ci
        else:                   # 虚部:ci cos + cr sin
            Ac, As = ci, cr
        how = r"$y_p=A\cos\omega t+B\sin\omega t$ とおいて係数を比べる。"
        fn = (lambda t: float(Ac) * np.cos(float(w) * t) + float(As) * np.sin(float(w) * t))
        dfn = (lambda t: float(w) * (-float(Ac) * np.sin(float(w) * t) + float(As) * np.cos(float(w) * t)))
        return fn, dfn, ssum([(Ac, trig("cos", w)), (As, trig("sin", w))]), how
    # 共鳴(a = 0, b = ω²):y_p = t e^{iωt}/(2iω) の実部・虚部
    if kind == "cos":
        Ac, As = F(0), c / (2 * w)          # t sin ωt / (2ω)
    else:
        Ac, As = -c / (2 * w), F(0)         # −t cos ωt / (2ω)
    fn = (lambda t: t * (float(Ac) * np.cos(float(w) * t) + float(As) * np.sin(float(w) * t)))
    dfn = (lambda t: (float(Ac) * np.cos(float(w) * t) + float(As) * np.sin(float(w) * t))
           + t * float(w) * (-float(Ac) * np.sin(float(w) * t) + float(As) * np.cos(float(w) * t)))
    return fn, dfn, ssum([(Ac, "t" + trig("cos", w)), (As, "t" + trig("sin", w))]), \
        r"$i\omega$ が特性根なので(共鳴)$y_p=t(A\cos\omega t+B\sin\omega t)$ とおく。"


FORCING = {"0(斉次)": "0", "定数 c": "const", "c t": "t", "c e^{kt}": "exp", "c cos ωt": "cos", "c sin ωt": "sin"}


def f_tex(kind, c, k, w):
    if kind == "0":
        return "0"
    if kind == "const":
        return fr(c)
    if kind == "t":
        return ssum([(c, "t")])
    if kind == "exp":
        return ssum([(c, e_t(k) or "")])
    return ssum([(c, trig(kind, w))])


def f_num(kind, c, k, w):
    return {"0": lambda t: 0 * t, "const": lambda t: c + 0 * t, "t": lambda t: c * t, "exp": lambda t: c * np.exp(k * t),
            "cos": lambda t: c * np.cos(w * t), "sin": lambda t: c * np.sin(w * t)}[kind]


def rk4(a, b, f, y0, v0, t):
    Y = np.zeros((len(t), 2))
    Y[0] = y0, v0
    for i in range(len(t) - 1):
        h, tt, y = t[i + 1] - t[i], t[i], Y[i]
        def g(s, z):
            return np.array([z[1], f(s) - a * z[1] - b * z[0]])
        k1 = g(tt, y)
        k2 = g(tt + h / 2, y + h / 2 * k1)
        k3 = g(tt + h / 2, y + h / 2 * k2)
        k4 = g(tt + h, y + h * k3)
        Y[i + 1] = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return Y[:, 0]


def solve_ivp_closed(a, b, kind, c, k, w, y0, v0):
    """(y の数値関数, C1, C2, yp の TeX, 説明, 根の情報)"""
    R = char_roots(a, b)
    (y1, d1, _), (y2, d2, _) = basis(R)
    yp, dyp, yptex, how = particular(a, b, kind, c, k, w)
    z = np.array([0.0])
    M = np.array([[y1(z)[0], y2(z)[0]], [d1(z)[0], d2(z)[0]]])
    rhs = np.array([y0 - yp(z)[0], v0 - dyp(z)[0]])
    C1, C2 = np.linalg.solve(M, rhs)
    return (lambda t: C1 * y1(t) + C2 * y2(t) + yp(t)), C1, C2, yptex, how, R, (y1, y2, yp)


# ------------------------------------------------------------
# ページ
# ------------------------------------------------------------
def tab_linear_ode():
    st.info("ラプラス変換を微分方程式に応用する前に, **定数係数の線形微分方程式**の解き方をまとめておく。"
            "ここでは「一般解を求めてから初期条件で定数を決める」方法をとる。次の「1.」のラプラス変換では, 初期条件が最初から式に入り, "
            "一般解を経由せずに答えが出る。2つの方法を比べてみよう。")

    st.subheader("1. 1階:y′ + a y = f(t)")
    st.markdown("両辺に**積分因子** $e^{at}$ を掛けると, 左辺が1つの微分にまとまる:")
    st.latex(r"e^{at}y^{\prime}+ae^{at}y=\left(e^{at}y\right)^{\prime}=e^{at}f(t)")
    st.markdown("両辺を $0$ から $t$ まで積分して")
    st.latex(r"y(t)=\underbrace{y(0)\,e^{-at}}_{\text{初期値の影響}}+\underbrace{\int_0^te^{-a(t-\tau)}f(\tau)\,d\tau}_{\text{入力 }f\text{ の影響}}")
    st.markdown("後半の積分は $e^{-at}$ と $f$ の**合成積**で, ラプラス変換では $\\dfrac{1}{s+a}F(s)$ にあたる(第2章 2. 合成積)。"
                "一般解の形で書くと $y=Ce^{-at}+y_p$($y_p$ は特殊解の1つ)。")
    st.markdown(r"""
| $f(t)$ | 特殊解 $y_p$ の置き方 | 例($a=2$) |
|---|---|---|
| 定数 $c$ | $y_p=A$ | $y^{\prime}+2y=6$:$y_p=3$ |
| $ce^{kt}$($k\neq-a$) | $y_p=Ae^{kt}$ | $y^{\prime}+2y=e^{t}$:$y_p=\frac13e^{t}$ |
| $ce^{-at}$(共鳴) | $y_p=Ate^{-at}$ | $y^{\prime}+2y=e^{-2t}$:$y_p=te^{-2t}$ |
| $c\cos\omega t,\ c\sin\omega t$ | $y_p=A\cos\omega t+B\sin\omega t$ | $y^{\prime}+2y=5\sin t$:$y_p=2\sin t-\cos t$ |
""")
    c = st.columns(4)
    a1 = c[0].slider("a", -1.0, 3.0, 2.0, 0.5, key="lo1_a")
    f1n = c[1].selectbox("f(t)", ["1", "sin t", "e^{−t}", "0"], key="lo1_f")
    y01 = c[2].slider("y(0)", -2.0, 2.0, 0.0, 0.5, key="lo1_y0")
    T1 = c[3].slider("t の範囲", 2.0, 10.0, 6.0, 1.0, key="lo1_T")
    t = np.linspace(0, T1, 1201)
    fv = {"1": np.ones_like(t), "sin t": np.sin(t), "e^{−t}": np.exp(-t), "0": 0 * t}[f1n]
    # y = e^{−at}(y0 + ∫ e^{aτ} f)
    integ = np.concatenate([[0], np.cumsum((np.exp(a1 * t[1:]) * fv[1:] + np.exp(a1 * t[:-1]) * fv[:-1]) / 2 * np.diff(t))])
    y = np.exp(-a1 * t) * (y01 + integ)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, mode="lines", line=dict(color=C1, width=3), name="y(t)"))
    fig.add_trace(go.Scatter(x=t, y=y01 * np.exp(-a1 * t), mode="lines", line=dict(color=C3, dash="dash"), name="y(0) e^{−at}(初期値の影響)"))
    fig.add_trace(go.Scatter(x=t, y=y - y01 * np.exp(-a1 * t), mode="lines", line=dict(color=C2, dash="dot"), name="合成積(入力の影響)"))
    layout(fig, title=f"y′ + {a1:g} y = {f1n}", xlab="t", ylab="y", height=320)
    show(fig)
    st.caption("$a>0$ なら初期値の影響 $e^{-at}$ は時間とともに消え, 入力 $f$ だけで決まる状態(定常状態)に近づく。$a<0$ では初期値の影響が増大する。")

    st.subheader("2. 2階の斉次方程式:y″ + a y′ + b y = 0")
    st.markdown("$y=e^{\\lambda t}$ を代入すると $(\\lambda^2+a\\lambda+b)e^{\\lambda t}=0$。そこで**特性方程式** $\\lambda^2+a\\lambda+b=0$ の解(特性根)で場合分けする:")
    st.markdown(r"""
| 特性根 | 一般解 |
|---|---|
| 異なる2つの実数 $\lambda_1,\lambda_2$ | $y=C_1e^{\lambda_1t}+C_2e^{\lambda_2t}$ |
| 重解 $\lambda$ | $y=(C_1+C_2t)e^{\lambda t}$ |
| 虚数 $p\pm qi$ | $y=e^{pt}\left(C_1\cos qt+C_2\sin qt\right)$ |
""")
    st.markdown("- 重解のとき $te^{\\lambda t}$ も解になる:代入すると $(\\lambda^2+a\\lambda+b)te^{\\lambda t}+(2\\lambda+a)e^{\\lambda t}=0$($\\lambda$ は重解なので $2\\lambda+a=0$)。\n"
                "- 虚数解のとき $e^{(p\\pm qi)t}=e^{pt}(\\cos qt\\pm i\\sin qt)$(オイラーの公式)の和と差から, 実数の解 $e^{pt}\\cos qt$, $e^{pt}\\sin qt$ が得られる。\n"
                "- 2つの解 $y_1,y_2$ が「本当に2つ」(一方が他方の定数倍でない)であることは, **ロンスキアン** $W=y_1y_2^{\\prime}-y_1^{\\prime}y_2\\neq0$ で確かめられる。")

    st.subheader("3. 非斉次方程式:特殊解(未定係数法)")
    st.markdown("非斉次方程式 $y^{\\prime\\prime}+ay^{\\prime}+by=f(t)$ の一般解は")
    st.latex(r"y=\underbrace{C_1y_1+C_2y_2}_{\text{斉次方程式の一般解}}+\underbrace{y_p}_{\text{特殊解(1つでよい)}}")
    st.markdown(r"""
| $f(t)$ | $y_p$ の置き方 | 特性根と重なるとき |
|---|---|---|
| 定数, 多項式($n$ 次) | $n$ 次多項式 | $\lambda=0$ が特性根なら $t$ を掛ける(重解なら $t^2$) |
| $ce^{kt}$ | $Ae^{kt}$ | $k$ が単根なら $Ate^{kt}$, 重解なら $At^2e^{kt}$ |
| $c\cos\omega t,\ c\sin\omega t$ | $A\cos\omega t+B\sin\omega t$ | $\pm i\omega$ が特性根なら $t(A\cos\omega t+B\sin\omega t)$(共鳴) |
""")
    st.caption("特性根と重なると, 置いた形が斉次方程式の解になってしまい左辺が 0 になる。そこで $t$ を掛ける。")

    st.subheader("4. 特殊解:定数変化法(ロンスキアン)")
    st.markdown("未定係数法は $f$ が上の表の形のときしか使えない。どんな $f$ にも使えるのが**定数変化法**である。"
                "斉次方程式の基本解 $y_1,y_2$ の定数 $C_1,C_2$ を関数 $u_1(t),u_2(t)$ に変えて $y=u_1y_1+u_2y_2$ とおき, "
                "$u_1^{\\prime}y_1+u_2^{\\prime}y_2=0$ という条件をつけて代入すると")
    st.latex(r"u_1^{\prime}y_1+u_2^{\prime}y_2=0,\qquad u_1^{\prime}y_1^{\prime}+u_2^{\prime}y_2^{\prime}=f(t)")
    st.markdown("という連立1次方程式になる。係数の行列式がロンスキアン $W=y_1y_2^{\\prime}-y_1^{\\prime}y_2$ で, クラメルの公式から")
    st.latex(r"u_1^{\prime}=-\frac{y_2f}{W},\quad u_2^{\prime}=\frac{y_1f}{W}\qquad\Longrightarrow\qquad y_p=-y_1\int\frac{y_2f}{W}\,dt+y_2\int\frac{y_1f}{W}\,dt")
    st.markdown("**例**:$y^{\\prime\\prime}+y=\\dfrac{1}{\\cos t}$($-\\frac\\pi2<t<\\frac\\pi2$)。$y_1=\\cos t$, $y_2=\\sin t$, $W=\\cos^2t+\\sin^2t=1$ より")
    st.latex(r"u_1=-\int\frac{\sin t}{\cos t}\,dt=\log(\cos t),\quad u_2=\int1\,dt=t\qquad\Longrightarrow\qquad y_p=\cos t\,\log(\cos t)+t\sin t")
    st.caption("$\\frac{1}{\\cos t}$ は表の形でないので未定係数法では求められない。")
    st.markdown("**合成積の形**:積分を $0$ から $t$ までにとると, $y(0)=y^{\\prime}(0)=0$ をみたす特殊解は")
    st.latex(r"y_p(t)=\int_0^tg(t-\tau)f(\tau)\,d\tau,\qquad g:\ g^{\prime\prime}+ag^{\prime}+bg=0,\ g(0)=0,\ g^{\prime}(0)=1")
    st.markdown("と書ける($g$ は異なる実根なら $\\frac{e^{\\lambda_1t}-e^{\\lambda_2t}}{\\lambda_1-\\lambda_2}$, 重解なら $te^{\\lambda t}$, 虚数解なら $\\frac{1}{q}e^{pt}\\sin qt$)。"
                "$g$ はラプラス変換では $\\dfrac{1}{s^2+as+b}$ にあたり, 第2章 3. の**インパルス応答**である。")

    st.subheader("5. 初期値問題を解いてみる")
    st.markdown("係数・右辺・初期値を選ぶと, 特性根 → 斉次方程式の一般解 → 特殊解 → 初期条件で $C_1,C_2$ の順に解き, 数値解と比べる。")
    c = st.columns(4)
    a = c[0].slider("a", -4, 6, 3, key="lo_a")
    b = c[0].slider("b", -4, 13, 2, key="lo_b")
    fk = FORCING[c[1].selectbox("f(t)", list(FORCING), index=1, key="lo_f")]
    cc = c[1].select_slider("c", [-6, -4, -2, -1, 1, 2, 4, 6, 10], 2, key="lo_c")
    kk = c[2].select_slider("k(e^{kt})", [-3, -2, -1, 1, 2, 3], -1, key="lo_k")
    ww = c[2].select_slider("ω(cos, sin)", [1, 2, 3], 1, key="lo_w")
    y0 = c[3].select_slider("y(0)", [-2, -1, 0, 1, 2], 1, key="lo_y0")
    v0 = c[3].select_slider("y′(0)", [-2, -1, 0, 1, 2], 0, key="lo_v0")
    y_fn, K1, K2, yptex, how, R, (y1, y2, yp) = solve_ivp_closed(a, b, fk, cc, kk, ww, y0, v0)
    eq = ssum([(1, "y^{\\prime\\prime}"), (a, "y^{\\prime}"), (b, "y")])
    st.latex(rf"{eq}={f_tex(fk, cc, kk, ww)},\qquad y(0)={y0},\ y^{{\prime}}(0)={v0}")
    if R["kind"] == "real":
        rt = rf"\lambda={R['tex'][0]},\ {R['tex'][1]}"
    elif R["kind"] == "double":
        rt = rf"\lambda={R['tex'][0]}\ (\text{{重解}})"
    else:
        rt = rf"\lambda={R['tex'][0]}\pm {'' if R['tex'][1] == '1' else R['tex'][1]}i"
    st.latex(r"\text{特性方程式}\ " + ssum([(1, r"\lambda^2"), (a, r"\lambda"), (b, "")]) + r"=0\ \Rightarrow\ " + rt)
    st.latex(r"\text{斉次方程式の一般解}\ y_h=" + general_tex(R))
    st.markdown("特殊解:" + how)
    st.latex(r"y_p=" + yptex)
    st.latex(r"\text{初期条件から}\ C_1=" + num_tex(K1) + r",\ C_2=" + num_tex(K2))
    t = np.linspace(0, 6, 1201)
    yv = y_fn(t)
    yr = rk4(a, b, f_num(fk, cc, kk, ww), y0, v0, t)
    cols = st.columns([1, 2])
    with cols[0]:
        fig = go.Figure()
        if R["kind"] == "complex":
            p, q = R["l"]
            pts = [(p, q), (p, -q)]
        else:
            pts = [(l, 0) for l in R["l"]]
        fig.add_trace(go.Scatter(x=[x for x, _ in pts], y=[y for _, y in pts], mode="markers", marker=dict(size=13, color=C2), name="特性根"))
        fig.add_vline(x=0, line=dict(color=GRAY))
        fig.add_hline(y=0, line=dict(color=GRAY))
        layout(fig, title="特性根(複素平面)", xlab="実部", ylab="虚部", height=320, xrange=(-6, 6), yrange=(-4, 4), legend=False)
        show(fig)
        st.caption("根の実部が負なら $y_h$ は減衰し, 虚部があると振動する(ばね・ダンパーの振動とつながる)。")
    with cols[1]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=yv, mode="lines", line=dict(color=C1, width=3), name="公式の解 y(t)"))
        fig.add_trace(go.Scatter(x=t[::40], y=yr[::40], mode="markers", marker=dict(color="black", size=5), name="数値解(ルンゲ・クッタ)"))
        fig.add_trace(go.Scatter(x=t, y=yp(t), mode="lines", line=dict(color=C4, dash="dot"), name="特殊解 y_p"))
        lim = max(3.0, min(50.0, 1.2 * float(np.max(np.abs(yr)))))
        layout(fig, title="解 y(t)", xlab="t", ylab="y", height=320, yrange=(-lim, lim))
        show(fig)
    st.caption(f"公式の解と数値解の差の最大:{np.max(np.abs(yv - yr)):.1e}")

    st.subheader("6. ラプラス変換との比較(予告)")
    st.markdown("同じ問題をラプラス変換で解くと, $\\mathcal L[y^{\\prime\\prime}]=s^2Y-sy(0)-y^{\\prime}(0)$ などから")
    st.latex(r"(s^2+as+b)\,Y(s)=F(s)+(s+a)y(0)+y^{\prime}(0)")
    st.markdown("となり, 特性方程式と同じ多項式 $s^2+as+b$ が分母に現れる。$Y(s)$ を**部分分数分解**して逆変換すれば, "
                "初期条件を満たす解が直接得られる(第1章 4+. と第2章 1.)。共鳴の場合に $t$ が掛かることも, 分母の重なり(重根)から自然に出てくる。")

    with st.expander("練習問題"):
        st.markdown("**問1** $y^{\\prime}+3y=6$, $y(0)=0$ を解け。")
        st.markdown("**問2** $y^{\\prime\\prime}-y^{\\prime}-6y=0$ の一般解を求めよ。また $y^{\\prime\\prime}+4y^{\\prime}+13y=0$ の一般解を求めよ。")
        st.markdown("**問3** $y^{\\prime\\prime}-3y^{\\prime}+2y=e^{3t}$ の特殊解を, 未定係数法と定数変化法の両方で求めよ。")
        st.markdown("**問4** $y^{\\prime\\prime}+y=\\sin t$ の特殊解を求めよ(共鳴)。")
        if st.toggle("解答・解説を表示", key="pr_lode"):
            st.markdown("**答1** $y_p=2$, 一般解 $y=Ce^{-3t}+2$, $y(0)=0$ より $y=2-2e^{-3t}$。")
            st.markdown("**答2** $\\lambda^2-\\lambda-6=(\\lambda-3)(\\lambda+2)=0$ より $y=C_1e^{3t}+C_2e^{-2t}$。"
                        "$\\lambda^2+4\\lambda+13=0$ より $\\lambda=-2\\pm3i$, $y=e^{-2t}(C_1\\cos3t+C_2\\sin3t)$。")
            st.markdown("**答3** 未定係数法:$y_p=Ae^{3t}$ を代入して $(9-9+2)A=1$, $y_p=\\frac12e^{3t}$。"
                        "定数変化法:$y_1=e^{t}$, $y_2=e^{2t}$, $W=e^{3t}$, $y_p=-e^{t}\\int e^{2t}\\,dt+e^{2t}\\int e^{t}\\,dt=-\\frac12e^{3t}+e^{3t}=\\frac12e^{3t}$。")
            st.markdown("**答4** $\\pm i$ が特性根なので $y_p=t(A\\cos t+B\\sin t)$ とおく。代入して $-2A\\sin t+2B\\cos t=\\sin t$ より $A=-\\frac12$, $B=0$, "
                        "$y_p=-\\frac12t\\cos t$。")

    page_refs([("yajima2019", "定数係数の線形微分方程式, 未定係数法, 定数変化法とロンスキアン。"),
               ("tashiro2014", "ラプラス変換による微分方程式の解法との比較。"),
               ("schiff1999", "ラプラス変換と常微分方程式(発展)。")])
