"""② ラプラス変換 第1章 4+. 部分分数分解(ヘヴィサイドの方法)

隠蔽法(cover-up):F(s) = N(s)/((s−p_1)…(s−p_n)) の 1/(s−p_k) の係数は c_k = [(s−p_k)F(s)]_{s=p_k} = N(p_k)/D'(p_k)
重解:(s−α)² を含むとき, B = [(s−α)²F]_{s=α}, A = d/ds[(s−α)²F]_{s=α}
複素数の根:s² + ω² を含むとき, 実数の形 (Bs + Cω)/(s² + ω²) を残りから求める
"""
from fractions import Fraction

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import C1, C2, C3, layout, show
from topics.fourier_kernels import page_refs

F = Fraction


# ------------------------------------------------------------
# 多項式(係数は次数の高い順, Fraction)
# ------------------------------------------------------------
def pmul(p, q):
    out = [F(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            out[i + j] += F(a) * F(b)
    return out


def pval(p, x):
    v = 0
    for c in p:
        v = v * x + c
    return v


def pder(p):
    n = len(p) - 1
    return [c * (n - i) for i, c in enumerate(p[:-1])] or [F(0)]


def pdiv(p, q):
    """p = q·商 + 余り"""
    p = [F(c) for c in p]
    out = []
    while len(p) >= len(q):
        c = p[0] / q[0]
        out.append(c)
        p = [a - c * b for a, b in zip(p, list(q) + [0] * (len(p) - len(q)))][1:]
    return out or [F(0)], p


def from_roots(rs):
    out = [F(1)]
    for r in rs:
        out = pmul(out, [F(1), -F(r)])
    return out


def fr(x):
    x = F(x)
    if x.denominator == 1:
        return str(x.numerator)
    return ("-" if x < 0 else "") + rf"\frac{{{abs(x.numerator)}}}{{{x.denominator}}}"


def ptex(p, var="s"):
    """多項式の TeX"""
    n = len(p) - 1
    out = ""
    for i, c in enumerate(p):
        c = F(c)
        if c == 0:
            continue
        d = n - i
        body = "" if d == 0 else (var if d == 1 else f"{var}^{{{d}}}")
        sgn = "-" if c < 0 else ("+" if out else "")
        m = abs(c)
        coef = fr(m) if (d == 0 or m != 1) else ""
        out += sgn + coef + body
    return out or "0"


def lin(r, var="s"):
    r = F(r)
    if r == 0:
        return var
    return f"{var}-{fr(r)}" if r > 0 else f"{var}+{fr(-r)}"


def ssum(terms):
    out = ""
    for c, body in terms:
        c = F(c)
        if c == 0:
            continue
        sgn = "-" if c < 0 else ("+" if out else "")
        m = abs(c)
        coef = fr(m) if (body == "" or m != 1) else ""
        out += sgn + coef + body
    return out or "0"


def frac_term(c, den, first):
    c = F(c)
    if c == 0:
        return ""
    sgn = "-" if c < 0 else ("" if first else "+")
    m = abs(c)
    if m.denominator == 1:
        return sgn + rf"\frac{{{m.numerator}}}{{{den}}}"
    return sgn + rf"\frac{{{m.numerator}}}{{{m.denominator}({den})}}"


def e_t(r, pre=""):
    r = F(r)
    if r == 0:
        return pre or ""
    e = "e^{t}" if r == 1 else ("e^{-t}" if r == -1 else rf"e^{{{fr(r)}t}}")
    return pre + e


# ------------------------------------------------------------
# 計算
# ------------------------------------------------------------
def cover_up(N, roots):
    """異なる根 p_k に対する係数 c_k = N(p_k)/Π_{j≠k}(p_k − p_j)"""
    out = []
    for k, p in enumerate(roots):
        den = F(1)
        for j, q in enumerate(roots):
            if j != k:
                den *= (F(p) - F(q))
        out.append(pval(N, F(p)) / den)
    return out


def double_coeffs(N, al, be):
    """N/((s−α)²(s−β)) = A/(s−α) + B/(s−α)² + C/(s−β)"""
    al, be = F(al), F(be)
    Na, dNa = pval(N, al), pval(pder(N), al)
    B = Na / (al - be)
    A = (dNa * (al - be) - Na) / (al - be) ** 2
    C = pval(N, be) / (be - al) ** 2
    return A, B, C


def complex_coeffs(N, a, w):
    """N/((s−a)(s²+ω²)) = A/(s−a) + (B s + C ω)/(s² + ω²)"""
    a, w = F(a), F(w)
    A = pval(N, a) / (a * a + w * w)
    rest = [F(x) for x in N]
    q = [F(1), F(0), w * w]
    Aq = [A * x for x in q]
    rest = [F(0)] * (len(Aq) - len(rest)) + rest
    num = [x - y for x, y in zip(rest, Aq)]           # N − A(s² + ω²) = (s − a)(B s + Cω)
    quo, rem = pdiv(num, [F(1), -a])
    assert all(r == 0 for r in rem)
    quo = [F(0)] * (2 - len(quo)) + quo
    B, Cw = quo[-2], quo[-1]
    return A, B, Cw / w


# ------------------------------------------------------------
# ページ
# ------------------------------------------------------------
def tab_partial_fraction():
    st.info("逆ラプラス変換では, $F(s)$ を $\\dfrac{c}{s-p}$ のような簡単な分数の和に分ける**部分分数分解**が要になる。"
            "ここでは係数を素早く求める**ヘヴィサイドの方法**(隠蔽法, 展開定理)をまとめる。")
    st.markdown("**歴史**:イギリスの電気技術者ヘヴィサイド(O. Heaviside)は, 電信・電気回路の計算で微分 $\\frac{d}{dt}$ を文字 $p$ のように扱う**演算子法**を使い, "
                "有理式を部分分数に分けて解を書き下す公式(展開定理)を用いた。当時は数学的な根拠が不十分とされたが, "
                "のちにラプラス変換によって正当化された($p$ がラプラス変換の $s$ にあたる)。演算子法そのものは第2章 4. を参照。")

    st.subheader("1. 係数比較で求める方法(ふつうの方法)")
    st.latex(r"\frac{s+5}{(s-1)(s+2)}=\frac{A}{s-1}+\frac{B}{s+2}\ \Longrightarrow\ s+5=A(s+2)+B(s-1)")
    st.markdown("係数を比べて $A+B=1$, $2A-B=5$ を解くと $A=2$, $B=-1$。分母の因数が増えると連立方程式が大きくなる。")

    st.subheader("2. 隠蔽法(cover-up method)")
    st.markdown("$A$ だけ欲しいときは, 両辺に $(s-1)$ を掛けて $s=1$ を代入すればよい(他の項は $(s-1)$ が掛かって 0 になる):")
    st.latex(r"A=\left[(s-1)\cdot\frac{s+5}{(s-1)(s+2)}\right]_{s=1}=\left.\frac{s+5}{s+2}\right|_{s=1}=\frac{6}{3}=2,\qquad "
             r"B=\left.\frac{s+5}{s-1}\right|_{s=-2}=\frac{3}{-3}=-1")
    st.markdown("「$(s-1)$ を指で隠して, 残りに $s=1$ を代入する」ので**隠蔽法**という。一般に, 分母の根 $p_1,\\dots,p_n$ がすべて異なるとき")
    st.latex(r"\frac{N(s)}{D(s)}=\sum_{k=1}^n\frac{c_k}{s-p_k},\qquad c_k=\Bigl[(s-p_k)F(s)\Bigr]_{s=p_k}=\frac{N(p_k)}{D^{\prime}(p_k)}")
    st.markdown("($D(s)=(s-p_k)Q(s)$ と書くと $D^{\\prime}(p_k)=Q(p_k)$ なので, 2つの式は同じ。)逆ラプラス変換すると**ヘヴィサイドの展開定理**")
    st.latex(r"\mathcal L^{-1}\left[\frac{N(s)}{D(s)}\right]=\sum_{k=1}^n\frac{N(p_k)}{D^{\prime}(p_k)}\,e^{p_kt}")
    st.markdown("が得られる。($c_k$ は複素関数で学ぶ $F(s)$ の $s=p_k$ での**留数**でもある。)")

    st.markdown("**試してみよう**:分母の根と分子を選ぶと, 隠蔽法で係数を求める。")
    c = st.columns(4)
    n = c[0].radio("分母の因数の数", [2, 3], index=1, horizontal=True, key="pf_n")
    rts = [c[1].select_slider(f"p{i + 1}", list(range(-4, 5)), v, key=f"pf_p{i}") for i, v in zip(range(n), (1, -2, 3))]
    if len(set(rts)) < n:
        st.warning("根 p₁, p₂, … は異なる値にしてください(重解は 3. で扱う)。")
    else:
        Ncoef = [F(c[2 + (i // 2)].select_slider(f"分子の s^{n - 1 - i} の係数", list(range(-5, 6)), v, key=f"pf_N{n}_{i}"))
                 for i, v in zip(range(n), (0, 1, 5))]
        D = from_roots(rts)
        if all(x == 0 for x in Ncoef):
            st.warning("分子が 0 です。")
        else:
            cs = cover_up(Ncoef, rts)
            st.latex(r"F(s)=\frac{" + ptex(Ncoef) + "}{" + "".join(lin(p) if p == 0 else "(" + lin(p) + ")" for p in sorted(rts, key=lambda p: p != 0)) + "}")
            for k, p in enumerate(rts):
                others = "".join(lin(q) if q == 0 else "(" + lin(q) + ")" for j, q in enumerate(rts) if j != k)
                st.latex(rf"c_{k + 1}=\left.\frac{{{ptex(Ncoef)}}}{{{others}}}\right|_{{s={p}}}={fr(cs[k])}")
            pf = "".join(frac_term(ck, lin(p), i == 0) for i, (ck, p) in enumerate(zip(cs, rts)) if ck != 0) or "0"
            st.latex(rf"F(s)={pf}\qquad\Longrightarrow\qquad f(t)={ssum([(ck, e_t(p) or '') for ck, p in zip(cs, rts)])}")
            ss = np.array([0.37, 5.1, -7.3])
            lhs = np.array([float(pval(Ncoef, s)) / float(pval(D, s)) for s in ss])
            rhs = np.array([sum(float(ck) / (s - p) for ck, p in zip(cs, rts)) for s in ss])
            st.caption(f"検算:いくつかの s で両辺の差 = {np.max(np.abs(lhs - rhs)):.1e}。D′(p_k) を使った値 N(p_k)/D′(p_k) = "
                       + ", ".join(f"{float(pval(Ncoef, F(p)) / pval(pder(D), F(p))):g}" for p in rts))

    st.subheader("3. 重解があるとき")
    st.markdown("分母に $(s-\\alpha)^2$ があると, $\\dfrac{A}{s-\\alpha}+\\dfrac{B}{(s-\\alpha)^2}$ の2項が必要になる。$(s-\\alpha)^2$ を掛けると")
    st.latex(r"(s-\alpha)^2F(s)=A(s-\alpha)+B+(s-\alpha)^2\times(\text{残り})")
    st.markdown("なので, $s=\\alpha$ を代入すると $B$, **微分してから** $s=\\alpha$ を代入すると $A$ が求まる:")
    st.latex(r"B=\Bigl[(s-\alpha)^2F(s)\Bigr]_{s=\alpha},\qquad A=\left[\frac{d}{ds}\Bigl\{(s-\alpha)^2F(s)\Bigr\}\right]_{s=\alpha}")
    st.markdown("逆変換は $\\mathcal L^{-1}\\left[\\dfrac{1}{(s-\\alpha)^2}\\right]=te^{\\alpha t}$(第1移動法則)。")
    c = st.columns(3)
    al = c[0].select_slider("α(重解)", list(range(-3, 4)), 1, key="pf_al")
    be = c[0].select_slider("β", list(range(-3, 4)), -2, key="pf_be")
    Nd = [F(c[1 + i // 2].select_slider(f"分子の s^{2 - i} の係数", list(range(-5, 6)), v, key=f"pf_Nd{i}")) for i, v in zip(range(3), (0, 0, 1))]
    if al == be:
        st.warning("α と β は異なる値にしてください。")
    elif all(x == 0 for x in Nd):
        st.warning("分子が 0 です。")
    else:
        A, B, Cc = double_coeffs(Nd, al, be)
        st.latex(rf"F(s)=\frac{{{ptex(Nd)}}}{{({lin(al)})^2({lin(be)})}}")
        st.latex(rf"B=\left.\frac{{{ptex(Nd)}}}{{{lin(be)}}}\right|_{{s={al}}}={fr(B)},\quad "
                 rf"A=\left.\frac{{d}}{{ds}}\frac{{{ptex(Nd)}}}{{{lin(be)}}}\right|_{{s={al}}}={fr(A)},\quad "
                 rf"C=\left.\frac{{{ptex(Nd)}}}{{({lin(al)})^2}}\right|_{{s={be}}}={fr(Cc)}")
        pf = (frac_term(A, lin(al), True) + frac_term(B, f"({lin(al)})^2", A == 0) + frac_term(Cc, lin(be), A == 0 and B == 0)) or "0"
        st.latex(rf"F(s)={pf}\qquad\Longrightarrow\qquad f(t)={ssum([(A, e_t(al) or ''), (B, e_t(al, 't') or 't'), (Cc, e_t(be) or '')])}")

    st.subheader("4. 複素数の根(s² + ω²)があるとき")
    st.markdown("$s^2+\\omega^2=(s-i\\omega)(s+i\\omega)$ なので, 隠蔽法は複素数のまま使える。ただし実数の形で答えるには, "
                "実数の根の係数を隠蔽法で求めてから, 残りを $\\dfrac{Bs+C\\omega}{s^2+\\omega^2}$ の形で求めるのが簡単である:")
    st.latex(r"\frac{N(s)}{(s-a)(s^2+\omega^2)}=\frac{A}{s-a}+\frac{Bs+C\omega}{s^2+\omega^2}\ \Longrightarrow\ f(t)=Ae^{at}+B\cos\omega t+C\sin\omega t")
    c = st.columns(3)
    a = c[0].select_slider("a", list(range(-3, 4)), 0, key="pf_a")
    w = c[0].select_slider("ω", [1, 2, 3], 2, key="pf_w")
    Nc = [F(c[1 + i // 2].select_slider(f"分子の s^{2 - i} の係数 ", list(range(-5, 6)), v, key=f"pf_Nc{i}")) for i, v in zip(range(3), (0, 0, 4))]
    if all(x == 0 for x in Nc):
        st.warning("分子が 0 です。")
    else:
        A, B, Cc = complex_coeffs(Nc, a, w)
        st.latex(rf"F(s)=\frac{{{ptex(Nc)}}}{{({lin(a)})(s^2+{w * w})}}")
        rest = [x - y for x, y in zip(Nc, [A, 0, A * w * w])]
        st.markdown(f"(1) 隠蔽法:$A=\\left.\\dfrac{{{ptex(Nc)}}}{{s^2+{w * w}}}\\right|_{{s={a}}}={fr(A)}$。\n\n"
                    f"(2) 残り:$\\dfrac{{N(s)-A(s^2+{w * w})}}{{({lin(a)})(s^2+{w * w})}}=\\dfrac{{{ptex(rest)}}}{{({lin(a)})(s^2+{w * w})}}$ の分子を "
                    f"${lin(a)}$ で割ると ${ptex([B, Cc * w])}$, つまり $B={fr(B)}$, $C\\omega={fr(Cc * w)}$。")
        st.latex(rf"f(t)={ssum([(A, e_t(a) or ''), (B, rf'\cos {w if w > 1 else ''}t'), (Cc, rf'\sin {w if w > 1 else ''}t')])}")
        iw = 1j * w
        ck = complex(sum(float(x) * iw ** (len(Nc) - 1 - i) for i, x in enumerate(Nc))) / ((iw - a) * 2 * iw)
        st.caption(f"複素数のまま隠蔽法を使うと, 1/(s − {w}i) の係数は c = N({w}i)/(({w}i − ({a}))·2·{w}i) ≈ {ck.real:.4g} {'+' if ck.imag >= 0 else '−'} {abs(ck.imag):.4g}i。"
                   f"共役な根の項と合わせて 2Re(c e^{{i{w}t}}) = {2 * ck.real:.4g} cos {w}t {'+' if -2 * ck.imag >= 0 else '−'} {abs(2 * ck.imag):.4g} sin {w}t となり, 上の B, C と一致する。")
        t = np.linspace(0, 6, 601)
        ft = float(A) * np.exp(a * t) + float(B) * np.cos(w * t) + float(Cc) * np.sin(w * t)
        fig = go.Figure(go.Scatter(x=t, y=ft, mode="lines", line=dict(color=C1, width=2.5), name="f(t)"))
        layout(fig, title="逆ラプラス変換 f(t)", xlab="t", height=280)
        show(fig)

    with st.expander("練習問題"):
        st.markdown("**問1** $\\dfrac{2s+1}{(s-1)(s+1)(s-2)}$ を隠蔽法で部分分数に分け, 逆ラプラス変換を求めよ。")
        st.markdown("**問2** $\\dfrac{1}{(s-1)^2(s+1)}$ を部分分数に分けよ。")
        st.markdown("**問3** $\\dfrac{4}{s(s^2+4)}$ の逆ラプラス変換を求めよ。")
        if st.toggle("解答・解説を表示", key="pr_pfrac"):
            st.markdown("**答1** $c_1=\\left.\\frac{2s+1}{(s+1)(s-2)}\\right|_{s=1}=\\frac{3}{-2}$, $c_2=\\left.\\frac{2s+1}{(s-1)(s-2)}\\right|_{s=-1}=\\frac{-1}{6}$, "
                        "$c_3=\\left.\\frac{2s+1}{(s-1)(s+1)}\\right|_{s=2}=\\frac{5}{3}$。"
                        "$f(t)=-\\frac32e^{t}-\\frac16e^{-t}+\\frac53e^{2t}$。")
            st.markdown("**答2** $B=\\left.\\frac{1}{s+1}\\right|_{s=1}=\\frac12$, $A=\\left.\\frac{d}{ds}\\frac{1}{s+1}\\right|_{s=1}=-\\frac14$, "
                        "$C=\\left.\\frac{1}{(s-1)^2}\\right|_{s=-1}=\\frac14$。$-\\frac{1}{4(s-1)}+\\frac{1}{2(s-1)^2}+\\frac{1}{4(s+1)}$。")
            st.markdown("**答3** $A=\\left.\\frac{4}{s^2+4}\\right|_{s=0}=1$, 残りは $\\frac{4-(s^2+4)}{s(s^2+4)}=\\frac{-s}{s^2+4}$。$f(t)=1-\\cos2t$。")

    page_refs([("tashiro2014", "逆ラプラス変換と部分分数分解。"),
               ("schiff1999", "ヘヴィサイドの展開定理と演算子法の歴史(発展)。")])
