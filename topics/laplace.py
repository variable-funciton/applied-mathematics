"""② ラプラス変換:定義と収束・逆変換(部分分数)・微分方程式"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import C1, C2, C3, C4, GRAY, fmt_c, layout, show, trapz

FUNCS = {
    "1(単位ステップ)": dict(f=lambda t, a, w: np.ones_like(t), F=lambda s, a, w: 1 / s,
                         s0=lambda a, w: 0.0, tex=lambda a, w: r"1", Ftex=lambda a, w: r"\frac{1}{s}"),
    "t": dict(f=lambda t, a, w: t, F=lambda s, a, w: 1 / s ** 2, s0=lambda a, w: 0.0,
              tex=lambda a, w: r"t", Ftex=lambda a, w: r"\frac{1}{s^2}"),
    "t²": dict(f=lambda t, a, w: t ** 2, F=lambda s, a, w: 2 / s ** 3, s0=lambda a, w: 0.0,
               tex=lambda a, w: r"t^2", Ftex=lambda a, w: r"\frac{2}{s^3}"),
    "e^{at}": dict(f=lambda t, a, w: np.exp(a * t), F=lambda s, a, w: 1 / (s - a),
                   s0=lambda a, w: a, tex=lambda a, w: rf"e^{{{a:g}t}}",
                   Ftex=lambda a, w: rf"\frac{{1}}{{s-({a:g})}}"),
    "sin ωt": dict(f=lambda t, a, w: np.sin(w * t), F=lambda s, a, w: w / (s ** 2 + w ** 2),
                   s0=lambda a, w: 0.0, tex=lambda a, w: rf"\sin {w:g}t",
                   Ftex=lambda a, w: rf"\frac{{{w:g}}}{{s^2+{w * w:g}}}"),
    "cos ωt": dict(f=lambda t, a, w: np.cos(w * t), F=lambda s, a, w: s / (s ** 2 + w ** 2),
                   s0=lambda a, w: 0.0, tex=lambda a, w: rf"\cos {w:g}t",
                   Ftex=lambda a, w: rf"\frac{{s}}{{s^2+{w * w:g}}}"),
    "e^{at} sin ωt": dict(f=lambda t, a, w: np.exp(a * t) * np.sin(w * t),
                          F=lambda s, a, w: w / ((s - a) ** 2 + w ** 2), s0=lambda a, w: a,
                          tex=lambda a, w: rf"e^{{{a:g}t}}\sin {w:g}t",
                          Ftex=lambda a, w: rf"\frac{{{w:g}}}{{(s-({a:g}))^2+{w * w:g}}}"),
}




# ---------- 定義 ----------
def tab_definition():
    st.info("ラプラス変換は広義積分。$f(t)e^{-st}$ のグラフと t 軸で囲まれた(符号付き)面積が $F(s)$。")
    st.latex(r"F(s)=\mathcal L[f(t)]=\int_0^{\infty}f(t)\,e^{-st}\,dt=\lim_{T\to\infty}\int_0^{T}f(t)\,e^{-st}\,dt")
    name = st.selectbox("関数 f(t)", list(FUNCS), index=4, key="ld_f")
    D = FUNCS[name]
    c1, c2, c3 = st.columns(3)
    a = c1.slider("a", -2.0, 2.0, 0.5, 0.1, key="ld_a", disabled="a" not in name)
    w = c2.slider("ω", 0.5, 5.0, 2.0, 0.5, key="ld_w", disabled="ω" not in name)
    s = c3.slider("s", -1.0, 5.0, 1.0, 0.05, key="ld_s")
    s0 = D["s0"](a, w)
    conv = s > s0 + 1e-9

    st.latex(rf"f(t)={D['tex'](a, w)}\quad\Longrightarrow\quad F(s)={D['Ftex'](a, w)}\quad(s>{s0:g})")

    if conv:
        T_big = min(45 / (s - s0) + 10, 3000)
        Tv = min(T_big, max(10.0, 7 / (s - s0)))
    else:
        T_big = Tv = 10.0
    tt = np.linspace(0, T_big, 200001)
    g = D["f"](tt, a, w) * np.exp(-s * tt)
    cum = np.concatenate([[0], np.cumsum((g[1:] + g[:-1]) / 2 * np.diff(tt))])
    view = tt <= Tv

    m1, m2 = st.columns(2)
    if conv:
        exact = float(D["F"](s, a, w))
        m1.metric("数値積分 ∫₀^∞ f e^{−st} dt", f"{cum[-1]:.6f}")
        m2.metric("公式 F(s)", f"{exact:.6f}")
    else:
        m1.metric(f"∫₀^{Tv:g} f e^{{−st}} dt", f"{cum[view][-1]:.4g}")
        m2.metric("公式 F(s)", "収束しない")
        st.error(f"s = {s:g} ≤ {s0:g} なので、e^(−st) の減衰が f(t) の増加に負けて積分が収束しません。"
                 "右の「積分値 I(T)」のグラフが一定値に近づかないことを確認しよう。")

    cA, cB = st.columns(2)
    with cA:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tt[view][::50], y=g[view][::50], mode="lines", fill="tozeroy",
                                 line=dict(color=C1), name="f(t) e^{−st}"))
        fig.add_trace(go.Scatter(x=tt[view][::50], y=D["f"](tt[view][::50], a, w), mode="lines",
                                 line=dict(color=GRAY, dash="dot"), name="f(t)"))
        ylim = np.nanmax(np.abs(g[view])) * 1.3 + 1e-9
        layout(fig, title="被積分関数 f(t)e^{−st}(塗りつぶし=面積)", xlab="t",
               yrange=(-ylim, ylim), height=380)
        show(fig)
    with cB:
        fig = go.Figure(go.Scatter(x=tt[view][::50], y=cum[view][::50], mode="lines",
                                   line=dict(color=C3), name="I(T) = ∫₀^T"))
        if conv:
            fig.add_hline(y=exact, line=dict(color=C2, dash="dash"),
                          annotation_text=f"F(s) = {exact:.4f}")
        layout(fig, title="積分値 I(T) を T の関数として見る", xlab="T", height=380)
        show(fig)

    ss = np.linspace(s0 + 0.08, s0 + 5, 300)
    fig = go.Figure(go.Scatter(x=ss, y=D["F"](ss, a, w), mode="lines", line=dict(color=C1),
                               name="F(s)"))
    if conv and s <= s0 + 5:
        fig.add_trace(go.Scatter(x=[s], y=[exact], mode="markers",
                                 marker=dict(size=11, color=C2), name="いまの s"))
    fig.add_vline(x=s0, line=dict(color=GRAY, dash="dot"),
                  annotation_text=f"収束座標 s = {s0:g}")
    layout(fig, title="像関数 F(s)(s を動かすと赤い点が動く)", xlab="s", height=330)
    show(fig)

    with st.expander("練習問題"):
        st.markdown("**問1** 定義にしたがって $\\mathcal L[e^{at}]$ を求めよ。")
        st.markdown("**問2** $s\\le a$ で収束しないのはなぜか。")
        if st.toggle("解答・解説を表示", key="pr_laplace_100"):
            st.markdown("**答1**")
            st.latex(r"\int_0^T e^{-(s-a)t}dt=\frac{1-e^{-(s-a)T}}{s-a}\xrightarrow{T\to\infty}"
                     r"\frac{1}{s-a}\quad(s>a)")
            st.markdown("**答2**")
            st.write("$s=a$ なら被積分関数は 1 で $\\int_0^T 1\\,dt=T\\to\\infty$。"
                     "$s<a$ なら $e^{(a-s)T}\\to\\infty$。")


# ---------- 逆変換 ----------
def _parse(txt):
    vals = [float(v) for v in txt.replace("、", ",").split(",") if v.strip()]
    while vals and abs(vals[0]) < 1e-14:
        vals.pop(0)
    return np.array(vals, dtype=float)


def _poly_tex(c, var="s"):
    deg = len(c) - 1
    out = ""
    for i, v in enumerate(c):
        p = deg - i
        if abs(v) < 1e-14:
            continue
        sign = " - " if v < 0 else (" + " if out else "")
        av = abs(v)
        coef = "" if (abs(av - 1) < 1e-14 and p > 0) else f"{av:g}"
        mono = "" if p == 0 else (var if p == 1 else f"{var}^{{{p}}}")
        out += sign + coef + mono
    return out or "0"


def _num(v):
    return f"{v:.4g}"


def _exp_tex(a):
    """e^{at} の TeX(a = 0 なら空文字)"""
    if abs(a) < 1e-12:
        return ""
    if abs(a - 1) < 1e-12:
        return "e^{t}"
    if abs(a + 1) < 1e-12:
        return "e^{-t}"
    return f"e^{{{_num(a)}t}}"


def _sum_tex(terms):
    """[(係数, 関数部分の TeX)] を 2e^{-t} - e^{-2t} のような式に"""
    out = ""
    for co, fn in terms:
        if abs(co) < 1e-10:
            continue
        sign = "-" if co < 0 else ("+" if out else "")
        mag = abs(co)
        coef = "" if (abs(mag - 1) < 1e-10 and fn) else _num(mag)
        out += sign + coef + fn
    return out or "0"


def partial_fractions(N, D):
    """F = N/D(D の根はすべて単根)の逆ラプラス変換。項のリスト (根, 留数) を返す。"""
    p = np.roots(D)
    dD = np.polyder(D)
    res = np.polyval(N, p) / np.polyval(dD, p)
    return p, res


def tab_inverse():
    st.subheader("ラプラス変換の主な公式")
    st.markdown(
        r"""
| $f(t)$ | $F(s)$ | | $f(t)$ | $F(s)$ |
|---|---|---|---|---|
| $1$ | $\dfrac1s$ | | $e^{at}$ | $\dfrac{1}{s-a}$ |
| $t^n$ | $\dfrac{n!}{s^{n+1}}$ | | $t^ne^{at}$ | $\dfrac{n!}{(s-a)^{n+1}}$ |
| $\sin\omega t$ | $\dfrac{\omega}{s^2+\omega^2}$ | | $\cos\omega t$ | $\dfrac{s}{s^2+\omega^2}$ |
| $e^{at}\sin\omega t$ | $\dfrac{\omega}{(s-a)^2+\omega^2}$ | | $e^{at}\cos\omega t$ | $\dfrac{s-a}{(s-a)^2+\omega^2}$ |
| $f'(t)$ | $sF(s)-f(0)$ | | $f''(t)$ | $s^2F(s)-sf(0)-f'(0)$ |
| $e^{at}f(t)$ | $F(s-a)$(第1移動定理) | | $f(t-\tau)U(t-\tau)$ | $e^{-\tau s}F(s)$(第2移動定理) |
| $\displaystyle\int_0^t f(u)\,du$ | $\dfrac{F(s)}{s}$ | | $(f*g)(t)$ | $F(s)G(s)$ |
"""
    )
    st.divider()
    st.subheader("逆変換:部分分数分解(ヘヴィサイドの展開定理)")
    st.write("分母 $D(s)$ の根 $p_k$ がすべて異なるとき、")
    st.latex(r"F(s)=\frac{N(s)}{D(s)}=\sum_k\frac{c_k}{s-p_k},\quad c_k=\frac{N(p_k)}{D'(p_k)}"
             r"\quad\Longrightarrow\quad f(t)=\sum_k c_k\,e^{p_k t}")
    st.caption("$c_k$ は $F(s)$ の $s=p_k$ における**留数**です(④複素関数とつながる)。")

    examples = {
        "(s+3)/(s²+3s+2)": ("1, 3", "1, 3, 2"),
        "1/(s(s²+4))": ("1", "1, 0, 4, 0"),
        "(s+1)/(s²+2s+5)": ("1, 1", "1, 2, 5"),
        "1/(s³ − s)": ("1", "1, 0, -1, 0"),
        "自分で入力": None,
    }
    ex = st.selectbox("例", list(examples), key="li_ex")
    c1, c2 = st.columns(2)
    dn, dd = examples[ex] or ("1, 2", "1, 4, 13, 0")
    Ntxt = c1.text_input("分子 N(s) の係数(次数の高い順、カンマ区切り)", dn, key=f"li_N_{ex}")
    Dtxt = c2.text_input("分母 D(s) の係数(次数の高い順、カンマ区切り)", dd, key=f"li_D_{ex}")
    try:
        N, D = _parse(Ntxt), _parse(Dtxt)
        assert len(D) >= 2 and len(N) >= 1
    except Exception:
        st.error("係数は数値をカンマで区切って入力してください(例:1, 3, 2)。")
        return
    if len(N) >= len(D):
        st.error("分子の次数を分母の次数より小さくしてください(真分数)。")
        return
    st.latex(rf"F(s)=\frac{{{_poly_tex(N)}}}{{{_poly_tex(D)}}}")

    p = np.roots(D)
    dmin = min((abs(p[i] - p[j]) for i in range(len(p)) for j in range(i + 1, len(p))), default=1)
    if dmin < 1e-6:
        st.warning("分母に重根があります。この形では $\\frac{c}{(s-p)^2}$ などの項が必要です"
                   "($\\mathcal L^{-1}\\left[\\frac{1}{(s-p)^2}\\right]=te^{pt}$)。"
                   "重根のない例で試してください。")
        return
    p, c = partial_fractions(N, D)

    rows, terms = [], []
    used = np.zeros(len(p), bool)
    for i in np.argsort(-p.real):
        if used[i]:
            continue
        used[i] = True
        pi, ci = p[i], c[i]
        if abs(pi.imag) < 1e-9:
            rows.append({"根 p": fmt_c(pi.real), "留数 c": fmt_c(ci.real)})
            terms.append((ci.real, _exp_tex(pi.real)))
        else:
            j = np.argmin(np.abs(p - np.conj(pi)) + used * 1e9)
            used[j] = True
            if pi.imag < 0:
                pi, ci = np.conj(pi), np.conj(ci)
            al, be = pi.real, pi.imag
            A, B = 2 * ci.real, -2 * ci.imag
            rows.append({"根 p": f"{fmt_c(pi)}, {fmt_c(np.conj(pi))}",
                         "留数 c": f"{fmt_c(ci)}, {fmt_c(np.conj(ci))}"})
            trig = _sum_tex([(A, rf"\cos {_num(be)}t"), (B, rf"\sin {_num(be)}t")])
            ex_t = _exp_tex(al)
            if ex_t and trig.count("\\") > 1 and abs(A) > 1e-10 and abs(B) > 1e-10:
                terms.append((1.0, ex_t + r"\left(" + trig + r"\right)"))
            else:
                for co, fn in [(A, rf"\cos {_num(be)}t"), (B, rf"\sin {_num(be)}t")]:
                    terms.append((co, ex_t + fn))
    st.dataframe(rows, hide_index=True, width="stretch")
    st.latex(r"f(t)=" + _sum_tex(terms))
    st.caption("共役な複素根の組は $c\\,e^{pt}+\\bar c\\,e^{\\bar p t}=2\\,\\mathrm{Re}(c\\,e^{pt})$ "
               "として実数の三角関数にまとめています。")

    tt = np.linspace(0, 10, 1001)
    f = np.real(sum(ci * np.exp(pi * tt) for pi, ci in zip(p, c)))
    fig = go.Figure(go.Scatter(x=tt, y=f, mode="lines", line=dict(color=C1), name="f(t)"))
    layout(fig, title="逆変換した f(t)", xlab="t", height=340)
    show(fig)

    # 検算:求めた f をもう一度数値でラプラス変換
    s_chk = max(p.real.max(), 0) + 1.5
    T = 60 / (s_chk - p.real.max())
    t2 = np.linspace(0, T, 200001)
    f2 = np.real(sum(ci * np.exp(pi * t2) for pi, ci in zip(p, c)))
    num = trapz(f2 * np.exp(-s_chk * t2), t2)
    st.write(f"検算:s = {s_chk:.2f} で、求めた f(t) を数値積分でラプラス変換すると **{num:.6f}**、"
             f"もとの F(s) は **{np.polyval(N, s_chk) / np.polyval(D, s_chk):.6f}**。")

    with st.expander("練習問題"):
        st.markdown("**問3** $\\mathcal L^{-1}\\left[\\dfrac{s+3}{s^2+3s+2}\\right]$ を求めよ。")
        st.markdown("**問4** $\\mathcal L^{-1}\\left[\\dfrac{s+1}{s^2+2s+5}\\right]$ を求めよ。")
        if st.toggle("解答・解説を表示", key="pr_laplace_268"):
            st.markdown("**答3**")
            st.latex(r"\frac{s+3}{(s+1)(s+2)}=\frac{2}{s+1}-\frac{1}{s+2}\ \Longrightarrow\ "
                     r"f(t)=2e^{-t}-e^{-2t}")
            st.markdown("**答4**")
            st.latex(r"\frac{s+1}{(s+1)^2+2^2}\ \Longrightarrow\ f(t)=e^{-t}\cos 2t\quad(\text{第1移動定理})")


# ---------- 微分方程式 ----------
def rk4(fun, y0, t):
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(len(t) - 1):
        h = t[i + 1] - t[i]
        k1 = fun(t[i], y[i])
        k2 = fun(t[i] + h / 2, y[i] + h / 2 * k1)
        k3 = fun(t[i] + h / 2, y[i] + h / 2 * k2)
        k4 = fun(t[i] + h, y[i] + h * k3)
        y[i + 1] = y[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def tab_ode():
    st.info("ばね・ダンパ系や RLC 回路の方程式。ラプラス変換すると**微分が s のかけ算に**変わり、"
            "代数方程式として解ける。解の振る舞いは**極(分母の根)の位置**で決まる。")
    st.latex(r"y''+2\zeta\omega_0\,y'+\omega_0^2\,y=r(t),\qquad y(0)=y_0,\ y'(0)=v_0")

    c1, c2, c3, c4 = st.columns(4)
    w0 = c1.slider("固有角振動数 ω₀", 0.5, 4.0, 2.0, 0.1, key="lo_w0")
    zeta = c2.slider("減衰比 ζ", 0.0, 2.0, 0.2, 0.05, key="lo_z")
    y0 = c3.slider("初期変位 y₀", -2.0, 2.0, 1.0, 0.1, key="lo_y0")
    v0 = c4.slider("初速度 v₀", -4.0, 4.0, 0.0, 0.1, key="lo_v0")
    d1, d2, d3 = st.columns(3)
    inp = d1.radio("外力 r(t)", ["なし", "ステップ K", "正弦波 K sin ωt"], key="lo_inp")
    K = d2.slider("K", 0.0, 4.0, 1.0, 0.1, key="lo_K", disabled=inp == "なし")
    w = d3.slider("外力の角振動数 ω", 0.2, 6.0, 2.0, 0.1, key="lo_w", disabled="正弦" not in inp)

    if inp == "なし":
        r = lambda t: 0.0  # noqa: E731
        Rs, in_poles = "0", []
    elif inp.startswith("ステップ"):
        r = lambda t: K  # noqa: E731
        Rs, in_poles = rf"\frac{{{K:g}}}{{s}}", [0j]
    else:
        r = lambda t: K * np.sin(w * t)  # noqa: E731
        Rs, in_poles = rf"\frac{{{K * w:g}}}{{s^2+{w * w:g}}}", [1j * w, -1j * w]

    b, c = 2 * zeta * w0, w0 ** 2
    st.markdown("**ラプラス変換すると**($Y(s)=\\mathcal L[y]$)")
    st.latex(rf"(s^2+{b:.3g}s+{c:.3g})\,Y(s)={Rs}+(s+{b:.3g})\cdot({y0:g})+({v0:g})")
    st.latex(rf"Y(s)=\frac{{{Rs}+{y0:g}s+{b * y0 + v0:.3g}}}{{s^2+{b:.3g}s+{c:.3g}}}")

    disc = zeta ** 2 - 1
    poles = np.roots([1, b, c]).astype(complex)
    wd = w0 * np.sqrt(max(-disc, 0))
    if zeta == 0:
        kind, form = "単振動(減衰なし)", r"A\cos\omega_0t+B\sin\omega_0t"
    elif abs(disc) < 1e-12:
        kind, form = "臨界減衰", r"(A+Bt)e^{-\omega_0t}"
    elif disc < 0:
        kind, form = "減衰振動", rf"e^{{-{zeta * w0:.3g}t}}(A\cos {wd:.3g}t+B\sin {wd:.3g}t)"
    else:
        kind, form = "過減衰", rf"Ae^{{{poles[0].real:.3g}t}}+Be^{{{poles[1].real:.3g}t}}"
    st.write(f"特性方程式 $s^2+{b:.3g}s+{c:.3g}=0$ の根:**{fmt_c(poles[0], 3)}**, **{fmt_c(poles[1], 3)}** "
             f"→ **{kind}**")
    st.latex(r"y_{\text{自由振動}}(t)=" + form)

    T = 30.0
    tt = np.linspace(0, T, 6001)
    sol = rk4(lambda t, Y: np.array([Y[1], r(t) - b * Y[1] - c * Y[0]]), np.array([y0, v0]), tt)

    cA, cB = st.columns([3, 2])
    with cA:
        fig = go.Figure(go.Scatter(x=tt, y=sol[:, 0], mode="lines", line=dict(color=C1, width=2.5),
                                   name="y(t)"))
        if inp != "なし":
            fig.add_trace(go.Scatter(x=tt, y=[r(t) for t in tt], mode="lines",
                                     line=dict(color=GRAY, dash="dot"), name="r(t)"))
        if inp.startswith("ステップ") and c > 0:
            fig.add_hline(y=K / c, line=dict(color=C2, dash="dash"),
                          annotation_text=f"最終値 K/ω₀² = {K / c:.3f}")
        layout(fig, title=f"解 y(t)({kind})", xlab="t", height=400)
        show(fig)
    with cB:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=poles.real, y=poles.imag, mode="markers",
                                 marker=dict(symbol="x", size=14, color=C2, line=dict(width=3)),
                                 name="特性根(系の極)"))
        if in_poles:
            ip = np.array(in_poles)
            fig.add_trace(go.Scatter(x=ip.real, y=ip.imag, mode="markers",
                                     marker=dict(symbol="circle-open", size=14, color=C1,
                                                 line=dict(width=3)), name="外力の極"))
        lim = max(4.5, np.abs(poles).max() * 1.2, w * 1.2 if "正弦" in inp else 0)
        fig.add_vline(x=0, line=dict(color=GRAY))
        fig.add_hline(y=0, line=dict(color=GRAY))
        layout(fig, title="s 平面", xlab="Re s", ylab="Im s", height=400, equal=True,
               xrange=(-lim, lim), yrange=(-lim, lim))
        show(fig)
        st.caption("極が左半平面(Re s < 0)にあれば減衰、虚軸上なら持続振動。"
                   "虚部が振動の角振動数、実部が減衰の速さ。")

    if "正弦" in inp:
        st.subheader("共振:外力の角振動数 ω と振幅")
        st.latex(r"\text{定常振幅}=K\,|H(i\omega)|=\frac{K}{\sqrt{(\omega_0^2-\omega^2)^2+(2\zeta\omega_0\omega)^2}}")
        ws = np.linspace(0.05, 6, 600)
        amp = K / np.sqrt((c - ws ** 2) ** 2 + (b * ws) ** 2)
        amp_now = K / np.sqrt((c - w ** 2) ** 2 + (b * w) ** 2) if not (zeta == 0 and abs(w - w0) < 1e-9) else np.inf
        fig = go.Figure(go.Scatter(x=ws, y=np.minimum(amp, 50), mode="lines", line=dict(color=C1),
                                   name="定常振幅"))
        if np.isfinite(amp_now):
            fig.add_trace(go.Scatter(x=[w], y=[amp_now], mode="markers", marker=dict(size=11, color=C2),
                                     name="いまの ω"))
        fig.add_vline(x=w0, line=dict(color=GRAY, dash="dot"), annotation_text="ω₀")
        layout(fig, xlab="ω", ylab="振幅", height=320, yrange=(0, min(max(amp.max(), 1) * 1.1, 50)))
        show(fig)
        if zeta == 0 and abs(w - w0) < 1e-9:
            st.error("ζ = 0 かつ ω = ω₀:外力の極が系の極と重なり、Y(s) に重根が生じます。"
                     "解に $t\\sin\\omega_0 t$ の項が現れ、振幅が時間に比例して増え続けます(共振)。")

    with st.expander("練習問題"):
        st.markdown("**問5** $y''+3y'+2y=0,\\ y(0)=1,\\ y'(0)=0$ をラプラス変換で解け。")
        st.markdown("**問6** $y''+4y=\\sin 2t,\\ y(0)=y'(0)=0$ の解に $t\\cos 2t$ が現れる理由を、"
                    "Y(s) の形から説明せよ。")
        if st.toggle("解答・解説を表示", key="pr_laplace_388"):
            st.markdown("**答5**")
            st.latex(r"(s^2+3s+2)Y=s+3\ \Rightarrow\ Y=\frac{s+3}{(s+1)(s+2)}=\frac{2}{s+1}-\frac{1}{s+2}")
            st.latex(r"y(t)=2e^{-t}-e^{-2t}")
            st.write("(ω₀ = √2 ≈ 1.4、ζ = 3/(2√2) ≈ 1.06 なので、上のスライダーを ω₀ = 1.4、ζ = 1.05、"
                     "y₀ = 1、v₀ = 0 にすると、ほぼ同じ形の過減衰のグラフになる)")
            st.markdown("**答6**")
            st.latex(r"Y(s)=\frac{2}{(s^2+4)^2}\quad(\pm 2i\ \text{が2位の極})")
            st.write("2位の極からは $t e^{\\pm 2it}$ の項が出てくるため。")
