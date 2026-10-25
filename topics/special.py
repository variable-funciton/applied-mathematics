"""⑦ 特殊関数:ガンマ関数/ベータ関数/ガンマ関数とベータ関数の関係"""
import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import C1, C2, C3, C4, GRAY, layout, show, trapz

PI = np.pi


def gamma_num(x, n=40001):
    """Γ(x)=∫_0^∞ t^{x-1} e^{-t} dt を t = e^s で数値積分(x > 0)"""
    s = np.linspace(-60 / max(x, 0.05), np.log(80 + 4 * x), n)
    t = np.exp(s)
    return float(trapz(np.exp(x * s - t), s))


def beta_num(p, q, n=40001):
    """B(p,q)=∫_0^1 t^{p-1}(1-t)^{q-1} dt を [0,1/2], [1/2,1] に分けて数値積分"""
    def half(a, b):  # ∫_0^{1/2} t^{a-1}(1-t)^{b-1} dt
        s = np.linspace(-60 / max(a, 0.05), np.log(0.5), n)
        t = np.exp(s)
        return trapz(t ** a * (1 - t) ** (b - 1), s)
    return float(half(p, q) + half(q, p))


def beta_exact(p, q):
    return math.exp(math.lgamma(p) + math.lgamma(q) - math.lgamma(p + q))


def gamma_signed(x):
    """負の非整数でも使える Γ(x)(極では nan)"""
    if x <= 0 and abs(x - round(x)) < 1e-9:
        return np.nan
    return math.gamma(x)


# ============================================================
# 1. ガンマ関数
# ============================================================
def tab_gamma():
    st.info("階乗 n! を、整数でない数にまで広げた関数。積分で定義され、確率・統計(ガンマ分布、カイ2乗分布)や"
            "物理のいろいろな公式に現れる。")
    st.subheader("定義")
    st.latex(r"\Gamma(x)=\int_0^{\infty}t^{x-1}e^{-t}\,dt\qquad(x>0)")
    st.markdown("$t\\to\\infty$ では $e^{-t}$ が速く 0 に近づき、$t\\to0$ では $x>0$ なら $t^{x-1}$ が積分可能なので、"
                "広義積分が収束する。")

    x = st.slider("x", 0.2, 6.0, 2.5, 0.05, key="sg_x")
    t = np.linspace(1e-4, 15, 1500)
    y = t ** (x - 1) * np.exp(-t)
    val = gamma_num(x)
    cA, cB = st.columns(2)
    with cA:
        fig = go.Figure(go.Scatter(x=t, y=np.minimum(y, 6), mode="lines", fill="tozeroy", line=dict(color=C1),
                                   name="t^{x−1} e^{−t}"))
        layout(fig, title=f"被積分関数(塗りつぶしの面積 = Γ({x:g}))", xlab="t", height=360, legend=False,
               yrange=(0, min(max(y.max(), 0.3) * 1.15, 6)))
        show(fig)
        m1, m2 = st.columns(2)
        m1.metric("数値積分", f"{val:.6f}")
        m2.metric("Γ(x)(math.gamma)", f"{math.gamma(x):.6f}")
        st.caption("被積分関数のピークは t = x − 1 にある(x > 1 のとき)。")
    with cB:
        xs = np.linspace(-4.5, 5.2, 3000)
        g = np.array([gamma_signed(v) for v in xs])
        g[np.abs(g) > 12] = np.nan
        fig = go.Figure(go.Scatter(x=xs, y=g, mode="lines", line=dict(color=C1, width=2.5), name="Γ(x)"))
        ns = np.arange(1, 6)
        fig.add_trace(go.Scatter(x=ns, y=[math.factorial(n - 1) for n in ns], mode="markers+text",
                                 text=[f"{n - 1}!" for n in ns], textposition="top left",
                                 marker=dict(color=C2, size=9), name="Γ(n) = (n−1)!"))
        fig.add_trace(go.Scatter(x=[0.5], y=[math.sqrt(PI)], mode="markers+text", text=["√π"], textposition="top right",
                                 marker=dict(color=C3, size=10, symbol="star"), name="Γ(1/2) = √π"))
        fig.add_trace(go.Scatter(x=[x], y=[math.gamma(x)], mode="markers", marker=dict(color="black", size=9),
                                 name="いまの x"))
        for k in range(0, 5):
            fig.add_vline(x=-k, line=dict(color=GRAY, dash="dot", width=1))
        layout(fig, title="Γ(x) のグラフ(0, −1, −2, … は極)", xlab="x", height=420, yrange=(-8, 12))
        show(fig)

    st.subheader("性質")
    st.latex(r"\Gamma(x+1)=x\,\Gamma(x),\qquad \Gamma(1)=1,\qquad \Gamma(n+1)=n!\quad(n=0,1,2,\ldots),\qquad "
             r"\Gamma\!\left(\tfrac12\right)=\sqrt\pi")
    with st.expander("漸化式 Γ(x+1) = xΓ(x) の導き方(部分積分)"):
        st.latex(r"\Gamma(x+1)=\int_0^\infty t^{x}e^{-t}dt=\Big[-t^xe^{-t}\Big]_0^\infty+x\int_0^\infty t^{x-1}e^{-t}dt=x\,\Gamma(x)")
        st.write("$\\Gamma(1)=\\int_0^\\infty e^{-t}dt=1$ と合わせると、$\\Gamma(n+1)=n\\Gamma(n)=\\cdots=n!$。")
        st.write("負の数へは $\\Gamma(x)=\\dfrac{\\Gamma(x+1)}{x}$ で延長する。x = 0, −1, −2, … では分母が 0 になり、極になる。")
    with st.expander("Γ(1/2) = √π の導き方"):
        st.latex(r"\Gamma\!\left(\tfrac12\right)=\int_0^\infty t^{-1/2}e^{-t}dt\ \overset{t=u^2}{=}\ 2\int_0^\infty e^{-u^2}du=\sqrt\pi")
        st.caption("最後はガウス積分 $\\int_{-\\infty}^{\\infty}e^{-u^2}du=\\sqrt\\pi$(「3. ガンマ関数とベータ関数の関係」の節)。")
        st.write("漸化式と組み合わせると、例えば $\\Gamma(\\frac52)=\\frac32\\cdot\\frac12\\Gamma(\\frac12)=\\frac34\\sqrt\\pi$。")

    st.subheader("スターリングの公式(n! の近似)")
    st.latex(r"n!=\Gamma(n+1)\sim\sqrt{2\pi n}\left(\frac ne\right)^n\qquad(n\to\infty)")
    n = np.arange(1, 31)
    exact = np.array([math.lgamma(k + 1) for k in n])
    stir = 0.5 * np.log(2 * PI * n) + n * (np.log(n) - 1)
    rel = 1 - np.exp(stir - exact)
    fig = go.Figure(go.Scatter(x=n, y=rel * 100, mode="lines+markers", line=dict(color=C4), name="相対誤差"))
    layout(fig, title="スターリングの公式の相対誤差 [%](約 1/(12n) で 0 に近づく)", xlab="n", height=300, legend=False)
    show(fig)
    st.caption(f"例:10! = {math.factorial(10):,}、スターリングの近似 ≈ {math.exp(stir[9]):,.0f}(誤差 {rel[9] * 100:.2f}%)")

    with st.expander("練習問題"):
        st.markdown("**問1** $\\Gamma(5)$, $\\Gamma\\left(\\frac72\\right)$ を求めよ。")
        st.latex(r"\Gamma(5)=4!=24,\qquad \Gamma\!\left(\tfrac72\right)=\tfrac52\cdot\tfrac32\cdot\tfrac12\sqrt\pi=\tfrac{15}{8}\sqrt\pi")
        st.markdown("**問2** $\\displaystyle\\int_0^\\infty x^3e^{-2x}dx$ を、ガンマ関数を使って求めよ。")
        st.latex(r"2x=t\ \text{とおくと}\ \int_0^\infty\left(\frac t2\right)^3e^{-t}\frac{dt}{2}=\frac{\Gamma(4)}{2^4}=\frac{6}{16}=\frac38")
        st.markdown("**問3** $a>0$ のとき $\\displaystyle\\int_0^\\infty x^{n}e^{-ax}dx$ を求めよ。")
        st.latex(r"\frac{\Gamma(n+1)}{a^{n+1}}=\frac{n!}{a^{n+1}}")


# ============================================================
# 2. ベータ関数
# ============================================================
def tab_beta():
    st.info("区間 [0, 1] での積分で定義される2変数の関数。ガンマ関数で表せ、"
            "三角関数のべきの積分(ウォリス積分)や確率(ベータ分布)に現れる。")
    st.subheader("定義")
    st.latex(r"B(p,q)=\int_0^1t^{p-1}(1-t)^{q-1}\,dt\qquad(p>0,\ q>0)")
    st.latex(r"B(p,q)=B(q,p)\quad(t=1-s\ \text{とおく}),\qquad "
             r"B(p,q)=2\int_0^{\pi/2}\sin^{2p-1}\theta\,\cos^{2q-1}\theta\,d\theta\quad(t=\sin^2\theta\ \text{とおく})")
    c1, c2 = st.columns(2)
    p = c1.slider("p", 0.3, 6.0, 2.0, 0.1, key="sb_p")
    q = c2.slider("q", 0.3, 6.0, 3.0, 0.1, key="sb_q")
    t = np.linspace(1e-4, 1 - 1e-4, 1500)
    y = t ** (p - 1) * (1 - t) ** (q - 1)
    num = beta_num(p, q)
    ex = beta_exact(p, q)
    cA, cB = st.columns(2)
    with cA:
        fig = go.Figure(go.Scatter(x=t, y=np.minimum(y, 8), mode="lines", fill="tozeroy", line=dict(color=C1),
                                   name="t^{p−1}(1−t)^{q−1}"))
        layout(fig, title=f"被積分関数(面積 = B({p:g}, {q:g}))", xlab="t", height=360, legend=False,
               yrange=(0, min(max(y.max(), 0.2) * 1.15, 8)))
        show(fig)
        st.caption("p < 1 なら t = 0 で、q < 1 なら t = 1 で被積分関数が無限大になるが、面積は有限(広義積分が収束)。")
    with cB:
        m1, m2 = st.columns(2)
        m1.metric("数値積分 B(p, q)", f"{num:.6f}")
        m2.metric("Γ(p)Γ(q)/Γ(p+q)", f"{ex:.6f}")
        st.latex(r"B(p,q)=\frac{\Gamma(p)\,\Gamma(q)}{\Gamma(p+q)}")
        if abs(p - round(p)) < 1e-9 and abs(q - round(q)) < 1e-9:
            P, Q = int(round(p)), int(round(q))
            st.latex(rf"B({P},{Q})=\frac{{{P - 1}!\,{Q - 1}!}}{{{P + Q - 1}!}}=\frac{{{math.factorial(P - 1) * math.factorial(Q - 1)}}}"
                     rf"{{{math.factorial(P + Q - 1)}}}")
        th = np.linspace(1e-4, PI / 2 - 1e-4, 1200)
        yt = 2 * np.sin(th) ** (2 * p - 1) * np.cos(th) ** (2 * q - 1)
        fig = go.Figure(go.Scatter(x=th, y=np.minimum(yt, 8), mode="lines", fill="tozeroy", line=dict(color=C3),
                                   name="三角関数型"))
        fig.update_xaxes(tickvals=[0, PI / 4, PI / 2], ticktext=["0", "π/4", "π/2"])
        layout(fig, title="三角関数型 2 sin^{2p−1}θ cos^{2q−1}θ(面積は同じ)", xlab="θ", height=260, legend=False,
               yrange=(0, min(max(yt.max(), 0.2) * 1.15, 8)))
        show(fig)

    with st.expander("練習問題"):
        st.markdown("**問1** $\\displaystyle\\int_0^1x^2(1-x)^3dx$ をベータ関数を使って求めよ。")
        st.latex(r"B(3,4)=\frac{2!\,3!}{6!}=\frac{12}{720}=\frac{1}{60}")
        st.markdown("**問2** $\\displaystyle\\int_0^1\\sqrt{x(1-x)}\\,dx$ を求めよ。")
        st.latex(r"B\!\left(\tfrac32,\tfrac32\right)=\frac{\Gamma(\frac32)^2}{\Gamma(3)}=\frac{(\frac12\sqrt\pi)^2}{2}=\frac\pi8")
        st.caption("(半径 1/2 の半円の面積と同じ。$y=\\sqrt{x(1-x)}$ は中心 (1/2, 0)、半径 1/2 の上半円)")
        st.markdown("**問3** $\\displaystyle\\int_0^{\\pi/2}\\sin^3\\theta\\cos^2\\theta\\,d\\theta$ を求めよ。")
        st.latex(r"2p-1=3,\ 2q-1=2\ \Rightarrow\ \frac12B\!\left(2,\tfrac32\right)=\frac12\cdot\frac{\Gamma(2)\Gamma(\frac32)}{\Gamma(\frac72)}"
                 r"=\frac12\cdot\frac{\frac12\sqrt\pi}{\frac{15}{8}\sqrt\pi}=\frac{2}{15}")


# ============================================================
# 3. ガンマ関数とベータ関数の関係(ガウス積分、ウォリス積分)
# ============================================================
def double_factorial(n):
    return 1 if n <= 0 else n * double_factorial(n - 2)


def tab_relation():
    st.info("2つの積分の積を2重積分に直し、変数変換する、という同じ考え方で、"
            "ガウス積分と B = ΓΓ/Γ の関係が導ける。")
    st.subheader("ガウス積分")
    st.latex(r"I=\int_{-\infty}^{\infty}e^{-x^2}dx,\qquad I^2=\iint_{\mathbb R^2}e^{-(x^2+y^2)}dxdy"
             r"=\int_0^{2\pi}\!\!\int_0^\infty e^{-r^2}r\,dr\,d\theta=\pi\quad\Longrightarrow\quad I=\sqrt\pi")
    R = st.slider("積分範囲(正方形 |x|, |y| ≤ R と円 r ≤ R)", 0.3, 3.0, 1.5, 0.1, key="sr_R")
    sq = math.erf(R) ** 2 * PI
    disk = PI * (1 - math.exp(-R * R))
    g = np.linspace(-3, 3, 121)
    X, Y = np.meshgrid(g, g)
    fig = go.Figure(go.Surface(x=g, y=g, z=np.exp(-X ** 2 - Y ** 2), colorscale="Blues", opacity=0.85, showscale=False))
    th = np.linspace(0, 2 * PI, 200)
    fig.add_trace(go.Scatter3d(x=R * np.cos(th), y=R * np.sin(th), z=0 * th, mode="lines", line=dict(color=C2, width=6),
                               name="円 r = R"))
    fig.add_trace(go.Scatter3d(x=[-R, R, R, -R, -R], y=[-R, -R, R, R, -R], z=[0] * 5, mode="lines",
                               line=dict(color=C4, width=6), name="正方形"))
    fig.update_layout(height=440, margin=dict(l=0, r=0, t=10, b=0), scene=dict(aspectmode="cube"))
    cA, cB = st.columns([3, 2])
    with cA:
        show(fig)
    with cB:
        st.metric("円の上の体積 π(1 − e^{−R²})", f"{disk:.6f}")
        st.metric("正方形の上の体積 (∫_{−R}^{R} e^{−x²}dx)²", f"{sq:.6f}")
        st.metric("R → ∞ の極限 π", f"{PI:.6f}")
        st.caption("正方形は円より大きく、半径 √2 R の円より小さい。R → ∞ で両方とも π に近づくので、I² = π。")

    st.subheader("ベータ関数とガンマ関数の関係")
    st.latex(r"B(p,q)=\frac{\Gamma(p)\,\Gamma(q)}{\Gamma(p+q)}")
    with st.expander("導き方の流れ"):
        st.latex(r"\Gamma(p)\Gamma(q)=\int_0^\infty\!\!\int_0^\infty u^{p-1}v^{q-1}e^{-(u+v)}du\,dv")
        st.write("$u=x^2,\\ v=y^2$ とおいて極座標 $x=r\\cos\\theta,\\ y=r\\sin\\theta$ に直すと、")
        st.latex(r"=4\int_0^{\pi/2}\!\!\int_0^\infty r^{2(p+q)-1}e^{-r^2}\cos^{2p-1}\theta\sin^{2q-1}\theta\,dr\,d\theta"
                 r"=\Gamma(p+q)\cdot B(p,q)")
        st.caption("r の積分が Γ(p+q)/2、θ の積分が B(p,q)/2 になる(ベータ関数の三角関数型)。")

    st.subheader("ウォリス積分")
    st.latex(r"W_n=\int_0^{\pi/2}\sin^n\theta\,d\theta=\frac12B\!\left(\frac{n+1}{2},\frac12\right)"
             r"=\begin{cases}\dfrac{(n-1)!!}{n!!}\cdot\dfrac\pi2&(n\ \text{偶数})\\[2mm]\dfrac{(n-1)!!}{n!!}&(n\ \text{奇数})\end{cases}")
    n = st.slider("n", 0, 12, 4, key="sr_n")
    th = np.linspace(0, PI / 2, 600)
    num = float(trapz(np.sin(th) ** n, th))
    frac = f"{double_factorial(n - 1)}/{double_factorial(n)}" if n > 0 else "1"
    exact = double_factorial(n - 1) / double_factorial(n) * (PI / 2 if n % 2 == 0 else 1) if n > 0 else PI / 2
    fig = go.Figure(go.Scatter(x=th, y=np.sin(th) ** n, mode="lines", fill="tozeroy", line=dict(color=C1),
                               name=f"sin^{n} θ"))
    fig.update_xaxes(tickvals=[0, PI / 4, PI / 2], ticktext=["0", "π/4", "π/2"])
    layout(fig, title=f"sin^{n} θ と W_{n}", xlab="θ", height=300, legend=False)
    show(fig)
    st.write(f"W_{n}:数値積分 **{num:.6f}**、公式 ({frac}){'·π/2' if n % 2 == 0 else ''} = **{exact:.6f}**")
    st.caption("漸化式 $W_n=\\frac{n-1}{n}W_{n-2}$(部分積分)からも同じ結果が得られる。")
