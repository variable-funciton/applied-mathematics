"""⑤ 偏微分方程式 0. 変数分離法

扱う型(u = u(x, y), 初期条件 u(x, 0) = Σ c_i e^{a_i x}):
  1階  :A u_x + B u = g(y) u + C u_y
         → A X'/X + B = g(y) + C Y'/Y = λ,  X = e^{(λ−B)x/A},  Y = e^{(λy − G(y))/C}
         → e^{a x} の項は λ = A a + B
  2階①:u_xy + B u_y = g(y) u
         → (X'/X + B)(Y'/Y) = g(y),  X'/X + B = λ,  Y'/Y = g(y)/λ,  X = e^{(λ−B)x},  Y = e^{G(y)/λ}
         → e^{a x} の項は λ = a + B(≠ 0)
  2階②:u_xx + B u = u_y + g(y) u
         → X''/X + B = Y'/Y + g(y) = λ,  X = e^{a x}(a² = λ − B),  Y = e^{λy − G(y)}
         → e^{a x} の項は λ = a² + B
ここで G(y) = ∫_0^y g(s) ds(G(0) = 0 なので Y(0) = 1)。
"""
from fractions import Fraction

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import C1, C2, C3, GRAY, layout, show
from topics.fourier_kernels import page_refs

F = Fraction
PI = np.pi
TYPES = ("1階", "2階①(u_xy を含む)", "2階②(u_xx を含む)")


# ------------------------------------------------------------
# g(y) と G(y) = ∫_0^y g
# ------------------------------------------------------------
def gfun(kind, k=1):
    """(g の TeX, G の係数, G の本体の TeX, g の数値関数, G の数値関数)。G(y) = 係数 × 本体"""
    ky = "y" if k == 1 else f"{k}y"
    if kind == "cos":
        return rf"\cos {ky}", F(1, k), rf"\sin {ky}", (lambda y: np.cos(k * y)), (lambda y: np.sin(k * y) / k)
    if kind == "sin":
        return rf"\sin {ky}", F(1, k), rf"(1-\cos {ky})", (lambda y: np.sin(k * y)), (lambda y: (1 - np.cos(k * y)) / k)
    if kind == "y":
        return "y", F(1, 2), "y^2", (lambda y: y), (lambda y: y ** 2 / 2)
    if kind == "y2":
        return "y^2", F(1, 3), "y^3", (lambda y: y ** 2), (lambda y: y ** 3 / 3)
    if kind == "exp":
        return "e^{y}", F(1), "(e^{y}-1)", (lambda y: np.exp(y)), (lambda y: np.exp(y) - 1)
    return "0", F(0), "", (lambda y: 0 * y), (lambda y: 0 * y)


G_KINDS = {"cos ky": "cos", "sin ky": "sin", "y": "y", "y²": "y2", "e^y": "exp", "なし(g = 0)": "zero"}


# ------------------------------------------------------------
# TeX の組み立て
# ------------------------------------------------------------
def cterm(c, body, first=False):
    """係数 c(有理数)× body。c = ±1 は省き, 定数項(body = "")は数だけ"""
    c = F(c)
    if c == 0:
        return ""
    sgn = "-" if c < 0 else ("" if first else "+")
    a = abs(c)
    if body == "":
        num = str(a.numerator) if a.denominator == 1 else rf"\frac{{{a.numerator}}}{{{a.denominator}}}"
        return sgn + num
    if a == 1:
        return sgn + body
    num = str(a.numerator) if a.denominator == 1 else rf"\frac{{{a.numerator}}}{{{a.denominator}}}"
    return sgn + num + body


def lin(terms):
    """[(係数, 本体), ...] の和。すべて 0 なら 0"""
    out, first = "", True
    for c, b in terms:
        t = cterm(c, b, first)
        if t:
            out += t
            first = False
    return out or "0"


def ex(terms):
    s = lin(terms)
    return "1" if s == "0" else rf"e^{{{s}}}"


def series(cs, exps):
    """Σ c_i e^{...}"""
    out, first = "", True
    for c, e in zip(cs, exps):
        body = ex(e)
        out += cterm(c, body, first) if body != "1" else cterm(c, "", first)
        first = False
    return out


U = {"x": r"\dfrac{\partial u}{\partial x}", "y": r"\dfrac{\partial u}{\partial y}",
     "xy": r"\dfrac{\partial^2 u}{\partial x\,\partial y}", "xx": r"\dfrac{\partial^2 u}{\partial x^2}"}


def gt_paren(gt):
    """u のあとに掛ける g(y) の TeX(u\\,\\cos 2y など)"""
    return rf"\,{gt}"


def pde_tex(typ, A, B, C, gt):
    """方程式の TeX"""
    if typ == 0:
        lhs = lin([(A, U["x"]), (B, "u")])
        rhs = lin([(1, rf"u{gt_paren(gt)}"), (C, U["y"])])
    elif typ == 1:
        lhs = lin([(1, U["xy"]), (B, U["y"])])
        rhs = rf"u{gt_paren(gt)}"
    else:
        lhs = lin([(1, U["xx"]), (B, "u")])
        rhs = lin([(1, U["y"])] + ([(1, rf"u{gt_paren(gt)}")] if gt != "0" else []))
    return f"{lhs}={rhs}"


def init_tex(cs, as_):
    return series(cs, [[(a, "x")] for a in as_])


# ------------------------------------------------------------
# 解(各項の指数)
# ------------------------------------------------------------
def lam(typ, a, A=1, B=0):
    """初期条件の e^{a x} の項に対応する分離定数 λ"""
    if typ == 0:
        return A * a + B
    if typ == 1:
        return a + B
    return a * a + B


def term_exp(typ, a, A, B, C, Gc, Gb):
    """e^{a x} から作られる解の項の指数 [(係数, 本体), ...]"""
    L = F(lam(typ, a, A, B))
    if typ == 0:
        return [(a, "x"), (L / C, "y"), (-Gc / C, Gb)]
    if typ == 1:
        return [(a, "x"), (Gc / L, Gb)]
    return [(a, "x"), (L, "y"), (-Gc, Gb)]


def solution_tex(typ, cs, as_, A, B, C, Gc, Gb):
    return series(cs, [term_exp(typ, a, A, B, C, Gc, Gb) for a in as_])


def u_num(typ, cs, as_, A, B, C, Gn):
    """解の数値関数 u(x, y)"""
    def u(x, y):
        tot = 0
        for c, a in zip(cs, as_):
            L = lam(typ, a, A, B)
            if typ == 0:
                e = a * x + (L * y - Gn(y)) / C
            elif typ == 1:
                e = a * x + Gn(y) / L
            else:
                e = a * x + L * y - Gn(y)
            tot = tot + c * np.exp(e)
        return tot
    return u


def residual(typ, u, A, B, C, gn, x, y, h=1e-4):
    """方程式の左辺 − 右辺(中心差分)を, 各項の大きさで割った相対値"""
    ux = (u(x + h, y) - u(x - h, y)) / (2 * h)
    uy = (u(x, y + h) - u(x, y - h)) / (2 * h)
    uxy = (u(x + h, y + h) - u(x + h, y - h) - u(x - h, y + h) + u(x - h, y - h)) / (4 * h * h)
    uxx = (u(x + h, y) - 2 * u(x, y) + u(x - h, y)) / (h * h)
    v = u(x, y)
    if typ == 0:
        r, s = A * ux + B * v - gn(y) * v - C * uy, abs(A * ux) + abs(B * v) + abs(gn(y) * v) + abs(C * uy)
    elif typ == 1:
        r, s = uxy + B * uy - gn(y) * v, abs(uxy) + abs(B * uy) + abs(gn(y) * v)
    else:
        r, s = uxx + B * v - uy - gn(y) * v, abs(uxx) + abs(B * v) + abs(uy) + abs(gn(y) * v)
    return np.abs(r) / (s + 1e-300)


def lamb(B):
    """λ − B の TeX(B = 0 なら λ)"""
    return r"\lambda" + cterm(-B, "")


def xstep(A, B):
    lb = lamb(B)
    if A == 1:
        return rf"X^{{\prime}}=({lb})X\ \Rightarrow\ X=e^{{({lb})x}}" if B else rf"X^{{\prime}}=\lambda X\ \Rightarrow\ X=e^{{\lambda x}}"
    return rf"{A}X^{{\prime}}=({lb})X\ \Rightarrow\ X=e^{{\frac{{{lb}}}{{{A}}}x}}"


def gs(gt):
    """積分の中の g(s)"""
    return gt.replace("y", "s")


def ystep(C, gt, G):
    yp = r"Y^{\prime}"
    lhs = yp if C == 1 else ("-" + yp if C == -1 else str(C) + yp)
    body = r"\lambda y-G(y)"
    e = body if C == 1 else ("-(" + body + ")" if C == -1 else r"\frac{" + body + "}{" + str(C) + "}")
    return lhs + r"=(\lambda-" + gt + r")Y\ \Rightarrow\ Y=e^{" + e + r"},\quad G(y)=\int_0^y" + gs(gt) + r"\,ds=" + G



XPY, XYP, XPYP, XPPY = r"X^{\prime}Y", r"XY^{\prime}", r"X^{\prime}Y^{\prime}", r"X^{\prime\prime}Y"
DX, DY, DXX = r"\dfrac{X^{\prime}}{X}", r"\dfrac{Y^{\prime}}{Y}", r"\dfrac{X^{\prime\prime}}{X}"
SUB = r"u=X(x)Y(y)\ \text{を代入}:\ "
DIV = r"XY\ \text{で割る}:\ "


def last_line(lam_expr):
    return r"\text{初期条件の}\ e^{ax}\ \text{の項には}\ \lambda=" + lam_expr + r"\ \text{とする}"


def gtex(Gc, Gb):
    """G(y) の TeX(係数 1 のときは外側の括弧をとる)"""
    if not Gb:
        return "0"
    if Gc == 1 and Gb.startswith("(") and Gb.endswith(")"):
        return Gb[1:-1]
    return lin([(Gc, Gb)])


def xexp(A, B):
    """X = e^{...} の指数(1階)"""
    lb = lamb(B)
    if A == 1:
        return "(" + lb + ")x" if B else r"\lambda x"
    return r"\frac{" + lb + "}{" + str(A) + "}x"


def yexp(C):
    """Y = e^{...} の指数(1階)"""
    body = r"\lambda y-G(y)"
    return body if C == 1 else ("-(" + body + ")" if C == -1 else r"\frac{" + body + "}{" + str(C) + "}")


def steps(typ, A, B, C, gt, Gc, Gb):
    """変数分離の手順(TeX の行)"""
    G = gtex(Gc, Gb)
    gp = gt_paren(gt)
    lb = lamb(B)
    if typ == 0:
        return [SUB + lin([(A, XPY), (B, "XY")]) + "=" + lin([(1, "XY" + gp), (C, XYP)]),
                DIV + lin([(A, DX), (B, "")]) + "=" + lin([(1, gt), (C, DY)]) + r"=\lambda\ (\text{定数})",
                xstep(A, B), ystep(C, gt, G), last_line(lin([(A, "a"), (B, "")]))]
    if typ == 1:
        return [SUB + lin([(1, XPYP), (B, XYP)]) + "=XY" + gp,
                DIV + r"\left(" + lin([(1, DX), (B, "")]) + r"\right)" + DY + "=" + gt,
                lin([(1, DX), (B, "")]) + r"=\lambda\ (\neq0)\ \text{とおくと}\ " + DY + r"=\dfrac{" + gt + r"}{\lambda}",
                r"X=e^{(" + lb + r")x},\quad Y=e^{G(y)/\lambda},\quad G(y)=\int_0^y" + gs(gt) + r"\,ds=" + G,
                last_line(lin([(1, "a"), (B, "")]))]
    has_g = gt != "0"
    return [SUB + lin([(1, XPPY), (B, "XY")]) + "=" + lin([(1, XYP)] + ([(1, "XY" + gp)] if has_g else [])),
            DIV + lin([(1, DXX), (B, "")]) + "=" + lin([(1, DY)] + ([(1, gt)] if has_g else [])) + r"=\lambda",
            r"X^{\prime\prime}=(" + lb + r")X\ \Rightarrow\ X=e^{ax}\ \left(a^2=" + lb + r"\right)",
            r"Y^{\prime}=(\lambda" + ("-" + gt if has_g else "") + r")Y\ \Rightarrow\ Y=e^{\lambda y-G(y)},\quad G(y)=" + G,
            last_line(lin([(1, "a^2"), (B, "")]))]


# ------------------------------------------------------------
# ページ
# ------------------------------------------------------------
def overview():
    st.info("偏微分方程式を解く基本の方法が**変数分離法**である。解を $u=X(x)Y(y)$ の形と仮定すると, 偏微分方程式が"
            "常微分方程式2つに分かれる。分かれて出てくる簡単な解を**重ね合わせて**初期条件に合わせる。")
    st.markdown("**この章の見取り図**:変数分離で出てくる関数と, 初期条件をそれらにどう分解するかで, フーリエ級数・フーリエ変換につながる。")
    st.markdown(r"""
| 状況 | 変数分離で出てくる関数 | 初期条件の分解 | 節 |
|---|---|---|---|
| 区間の制限なし, 初期条件が指数関数の和 | $e^{ax}$ | **有限個**の和 $\sum c_ie^{a_ix}$ | 0.(この節) |
| 有限区間 $0<x<1$ + 境界条件 $u(0)=u(1)=0$ | $\sin n\pi x$($n=1,2,\dots$) | **フーリエ級数** $\sum_{n=1}^\infty b_n\sin n\pi x$ | 1. 熱伝導・波動方程式 |
| 無限に長い区間 $-\infty<x<\infty$ | $e^{i\omega x}$($\omega$ は実数全体) | **フーリエ変換** $\dfrac{1}{2\pi}\displaystyle\int\hat f(\omega)e^{i\omega x}d\omega$ | 2. フーリエ変換と偏微分方程式 |
""")
    st.markdown("指数関数 $e^{ax}$ の $a$ を純虚数 $i\\omega$ にすると $e^{i\\omega x}=\\cos\\omega x+i\\sin\\omega x$ になる。"
                "つまり, フーリエ級数やフーリエ変換は「$e^{ax}$ 型の解を, 無限個・連続的に重ね合わせる」ことにあたる。"
                "この節では, 有限個の重ね合わせで済む問題で手順を身につける。")


def tab_separation():
    overview()

    st.subheader("1. 変数分離の考え方")
    st.markdown("例として次の問題を解く:")
    st.latex(r"-u+\frac{\partial u}{\partial x}=u\cos 2y+\frac{\partial u}{\partial y},\qquad u(x,0)=8e^{12x}-9e^{7x}")
    st.markdown("**手順1(積の形を仮定する)** $u=X(x)Y(y)$ とおいて代入すると $-XY+X^{\\prime}Y=XY\\cos2y+XY^{\\prime}$。\n\n"
                "**手順2(変数を分ける)** $XY$ で割ると")
    st.latex(r"\underbrace{\frac{X^{\prime}}{X}-1}_{x\text{ だけの式}}=\underbrace{\cos 2y+\frac{Y^{\prime}}{Y}}_{y\text{ だけの式}}")
    st.markdown("左辺は $y$ を変えても変わらず, 右辺は $x$ を変えても変わらない。両辺が等しいので, 両辺は**定数** $\\lambda$(**分離定数**)に等しい。\n\n"
                "**手順3(常微分方程式を解く)**")
    st.latex(r"X^{\prime}=(\lambda+1)X\ \Rightarrow\ X=e^{(\lambda+1)x},\qquad Y^{\prime}=(\lambda-\cos2y)Y\ \Rightarrow\ Y=e^{\lambda y-\frac12\sin2y}")
    st.markdown("($Y^{\\prime}/Y=(\\log Y)^{\\prime}$ なので, $\\log Y=\\int_0^y(\\lambda-\\cos2s)\\,ds$ と積分する。定数倍はあとで決めるので $Y(0)=1$ とした。)\n\n"
                "**手順4(重ね合わせて初期条件に合わせる)** どの $\\lambda$ でも $u_\\lambda=e^{(\\lambda+1)x+\\lambda y-\\frac12\\sin2y}$ は解で, "
                "$u_\\lambda(x,0)=e^{(\\lambda+1)x}$。方程式は**線形**なので, 解の定数倍の和も解になる。"
                "初期条件の $e^{12x}$ には $\\lambda=11$, $e^{7x}$ には $\\lambda=6$ を選んで")
    st.latex(r"u(x,y)=8e^{12x+11y-\frac12\sin2y}-9e^{7x+6y-\frac12\sin2y}=e^{-\frac12\sin2y}\left(8e^{12x+11y}-9e^{7x+6y}\right)")
    st.markdown("**手順5(確かめる)** $e^{ax+by-\\frac12\\sin2y}$ を代入すると, 方程式は $a-1=b$ となる。$(a,b)=(12,11),(7,6)$ はこれをみたす。")

    st.subheader("2. いろいろな型で手順を見る")
    st.markdown("型と係数を選ぶと, 同じ手順で解を作り, 代入して確かめる。$G(y)=\\displaystyle\\int_0^yg(s)\\,ds$ とする。")
    st.latex(r"\text{1階}:\ A\frac{\partial u}{\partial x}+Bu=g(y)\,u+C\frac{\partial u}{\partial y},\qquad"
             r"\text{2階①}:\ \frac{\partial^2u}{\partial x\,\partial y}+B\frac{\partial u}{\partial y}=g(y)\,u,\qquad"
             r"\text{2階②}:\ \frac{\partial^2u}{\partial x^2}+Bu=\frac{\partial u}{\partial y}+g(y)\,u")
    c = st.columns(4)
    typ = TYPES.index(c[0].selectbox("型", TYPES, key="ps_typ"))
    gopts = list(G_KINDS) if typ == 2 else [k for k in G_KINDS if G_KINDS[k] != "zero"]
    gk = G_KINDS[c[1].selectbox("g(y)", gopts, key=f"ps_g{typ}")]
    k = c[1].select_slider("k(cos ky, sin ky のとき)", [1, 2, 3], 2, key="ps_k") if gk in ("cos", "sin") else 1
    A = c[2].select_slider("A(1階)", [1, 2, 3], 1, key="ps_A") if typ == 0 else 1
    B = c[2].select_slider("B", [-3, -2, -1, 0, 1, 2, 3], -1 if typ == 0 else 0, key="ps_B")
    C = c[2].select_slider("C(1階)", [-2, -1, 1, 2], 1, key="ps_C") if typ == 0 else 1
    c1v = c[3].select_slider("c₁", [-9, -5, -3, -2, -1, 1, 2, 3, 5, 8], 8, key="ps_c1")
    a1 = c[3].select_slider("a₁", list(range(-4, 13)), 12 if typ == 0 else 2, key="ps_a1")
    c2v = c[3].select_slider("c₂", [-9, -5, -3, -2, -1, 0, 1, 2, 3, 5, 8], -9 if typ == 0 else -1, key="ps_c2")
    a2 = c[3].select_slider("a₂", list(range(-4, 13)), 7 if typ == 0 else -1, key="ps_a2")
    cs, as_ = ([c1v, c2v], [a1, a2]) if c2v != 0 else ([c1v], [a1])
    if typ == 1 and any(a + B == 0 for a in as_):
        st.warning("2階①では $\\lambda=a+B\\ne0$ が必要(λ で割るため)。a か B を変えてください。")
        return
    gt, Gc, Gb, gn, Gn = gfun(gk, k)
    st.latex(pde_tex(typ, A, B, C, gt) + r",\qquad u(x,0)=" + init_tex(cs, as_))
    for line in steps(typ, A, B, C, gt, Gc, Gb):
        if line:
            st.latex(line)
    sol = solution_tex(typ, cs, as_, A, B, C, Gc, Gb)
    st.latex(r"\therefore\ u(x,y)=" + sol)
    u = u_num(typ, cs, as_, A, B, C, Gn)
    pts = np.random.default_rng(0).uniform([-0.5, 0.05], [0.5, 0.5], (40, 2))
    res = residual(typ, u, A, B, C, gn, pts[:, 0], pts[:, 1])
    ic = np.max(np.abs(u(pts[:, 0], 0.0) - sum(cc * np.exp(aa * pts[:, 0]) for cc, aa in zip(cs, as_))))
    m = st.columns(2)
    m[0].metric("方程式の残差(差分で代入, 相対値の最大)", f"{res.max():.1e}")
    m[1].metric("初期条件のずれ(最大)", f"{ic:.1e}")
    st.caption("残差は数値微分の誤差(10⁻⁶ 程度)まで 0 になっている:作った関数は確かに解である。")
    xs, ys = np.linspace(-0.5, 0.5, 81), np.linspace(0, 0.5, 51)
    XX, YY = np.meshgrid(xs, ys)
    fig = go.Figure(go.Contour(x=xs, y=ys, z=u(XX, YY), colorscale="RdBu", ncontours=24, colorbar=dict(title="u", thickness=12)))
    fig.add_trace(go.Scatter(x=xs, y=0 * xs, mode="lines", line=dict(color="black", width=3), name="初期条件をあたえる線 y = 0"))
    layout(fig, title="解 u(x, y) の等高線", xlab="x", ylab="y", height=360)
    show(fig)

    st.subheader("3. 重ね合わせと初期条件")
    st.markdown("- 方程式が**線形**(u とその導関数について1次)なので, 解 $u_1,u_2$ の定数倍の和 $c_1u_1+c_2u_2$ も解になる(重ね合わせの原理)。\n"
                "- 変数分離で作れる解 $u_\\lambda$ は, $y=0$ で $e^{ax}$ という1つの指数関数になる。"
                "初期条件が指数関数の**有限個の和**なら, 対応する $\\lambda$ を選んで足せば解が作れる。\n"
                "- 初期条件が $\\sin x$ や $|x|$ のような一般の関数だと, **無限個**(フーリエ級数)や**連続的に無限個**(フーリエ変換)の重ね合わせが必要になる。"
                "これが次の 1., 2. の内容である。")

    st.subheader("4. 2階への橋渡し:熱伝導方程式")
    st.markdown("長さ 1 の棒の温度 $u(x,t)$ は, 両端を 0℃ に保つと次をみたす(1. で詳しく扱う):")
    st.latex(r"\frac{\partial u}{\partial t}=\kappa\frac{\partial^2u}{\partial x^2}\ (0<x<1),\qquad u(0,t)=u(1,t)=0,\qquad u(x,0)=f(x)")
    st.markdown("$u=X(x)T(t)$ とおくと $\\dfrac{T^{\\prime}}{\\kappa T}=\\dfrac{X^{\\prime\\prime}}{X}=-\\mu$(定数)となり, "
                "$X^{\\prime\\prime}=-\\mu X$, $X(0)=X(1)=0$ を解くことになる。この節と違うのは**境界条件**があることで, "
                "これをみたす $\\mu$ は限られる:")
    st.markdown("- $\\mu<0$:$X$ は $e^{\\pm\\sqrt{-\\mu}x}$ の組合せで, $X(0)=X(1)=0$ なら $X=0$ しかない。\n"
                "- $\\mu=0$:$X=px+q$ で, やはり $X=0$。\n"
                "- $\\mu>0$:$X(0)=0$ より $X=\\sin\\sqrt\\mu\\,x$。$X(1)=\\sin\\sqrt\\mu=0$ となるのは $\\sqrt\\mu=n\\pi$ のときだけ。")
    mu_r = st.slider("√μ / π(μ > 0 のとき)", 0.2, 4.0, 1.3, 0.05, key="ps_mu")
    xx = np.linspace(0, 1, 401)
    fig = go.Figure(go.Scatter(x=xx, y=np.sin(mu_r * PI * xx), mode="lines", line=dict(color=C2, width=2.5), name="X(x) = sin √μ x"))
    fig.add_trace(go.Scatter(x=[1], y=[np.sin(mu_r * PI)], mode="markers", marker=dict(color=C3, size=12), name=f"X(1) = {np.sin(mu_r * PI):.3f}"))
    layout(fig, title="境界条件 X(1) = 0 をみたすのは √μ = nπ のときだけ", xlab="x", height=280, yrange=(-1.1, 1.1))
    show(fig)
    if abs(mu_r - round(mu_r)) < 1e-9:
        st.success(f"√μ = {int(round(mu_r))}π:X(1) = 0 となり, 境界条件をみたす。")
    st.markdown("こうして $X_n=\\sin n\\pi x$, $T_n=e^{-\\kappa(n\\pi)^2t}$ が得られ, 初期条件に合わせて重ね合わせると")
    st.latex(r"u(x,t)=\sum_{n=1}^{\infty}b_ne^{-\kappa(n\pi)^2t}\sin n\pi x,\qquad f(x)=\sum_{n=1}^{\infty}b_n\sin n\pi x\ (\text{フーリエ正弦級数})")
    st.markdown("初期条件が $\\sin$ の有限和(例:$f=3\\sin\\pi x-\\sin 4\\pi x$)なら, この節と同じく有限個の和で解ける:"
                "$u=3e^{-\\kappa\\pi^2t}\\sin\\pi x-e^{-16\\kappa\\pi^2t}\\sin4\\pi x$。一般の $f$ では無限級数になる(1. へ)。"
                "区間が無限に長いと $n\\pi$ が飛び飛びでなくなり, $\\sum$ が $\\int$ に変わる(2. のフーリエ変換)。")
    st.markdown("**参考**:2階②で $B=0$, $g=0$ とした $u_{xx}=u_y$ は熱伝導方程式の形である。"
                "初期条件 $e^{ax}$ からは $e^{ax+a^2y}$ と増えていく解が出るが, $a=i\\omega$ とすると "
                "$e^{i\\omega x-\\omega^2y}$ となって減衰する。これが 2. のフーリエ変換による解の各成分である。")

    st.subheader("5. 補足:変数分離で作った解のほかに解はないか")
    st.markdown("1階の型 $A u_x+Bu=g(y)u+Cu_y$($C\\ne0$)は, 次のように**一般の初期条件** $u(x,0)=f(x)$ に対して解ける:")
    st.latex(r"u(x,y)=e^{\frac{By-G(y)}{C}}\,f\!\left(x+\frac{A}{C}y\right)")
    st.markdown("(代入すると $f^{\\prime}$ の項が打ち消し合う。直線 $Cx+Ay=$ 一定 の上で $u$ が常微分方程式をみたす, という**特性曲線の方法**による。)"
                "この形の解はほかになく, 初期条件から解はただ1つに決まる。"
                "$f=\\sum c_ie^{a_ix}$ を代入すると, 変数分離で作った解と一致する(1. の例では "
                "$e^{-y-\\frac12\\sin2y}\\left(8e^{12(x+y)}-9e^{7(x+y)}\\right)$ が上の答えと同じになることを確かめよう)。")
    st.markdown("一方, 2階の型では条件 $u(x,0)=f(x)$ だけでは解は1つに決まらない。"
                "たとえば $u_{xy}=0$, $u(x,0)=0$ には $u=0$ のほかに $u=y$, $u=\\sin y$ なども解になる。"
                "解を1つに決めるには $u(0,y)$ や境界条件などの条件を追加する必要がある(2階の問題では「変数分離形の解を1つ求めよ」という問い方をする)。")

    with st.expander("練習問題"):
        st.markdown("**問1** $\\dfrac{\\partial u}{\\partial x}=2\\dfrac{\\partial u}{\\partial y}$, $u(x,0)=e^{4x}$ を変数分離で解け。")
        st.markdown("**問2** $\\dfrac{\\partial u}{\\partial x}+u=y\\,u+\\dfrac{\\partial u}{\\partial y}$, $u(x,0)=3e^{2x}+e^{-x}$ を変数分離で解け。")
        st.markdown("**問3** $\\dfrac{\\partial^2u}{\\partial x\\partial y}=u\\cos y$ の解で $u(x,0)=e^{2x}$ となる変数分離形の解を求めよ。")
        if st.toggle("解答・解説を表示", key="pr_psep"):
            st.markdown("**答1** $X^{\\prime}/X=2Y^{\\prime}/Y=\\lambda$ より $u=e^{\\lambda x+\\frac\\lambda2y}$。$\\lambda=4$ として $u=e^{4x+2y}$。")
            st.markdown("**答2** $\\dfrac{X^{\\prime}}{X}+1=y+\\dfrac{Y^{\\prime}}{Y}=\\lambda$ より $X=e^{(\\lambda-1)x}$, $Y=e^{\\lambda y-\\frac{y^2}{2}}$。"
                        "$e^{2x}$ には $\\lambda=3$, $e^{-x}$ には $\\lambda=0$ で $u=3e^{2x+3y-\\frac{y^2}{2}}+e^{-x-\\frac{y^2}{2}}$。")
            st.markdown("**答3** $\\dfrac{X^{\\prime}}{X}\\cdot\\dfrac{Y^{\\prime}}{Y}=\\cos y$, $\\dfrac{X^{\\prime}}{X}=\\lambda$ とおくと $Y=e^{\\frac{\\sin y}{\\lambda}}$。"
                        "$\\lambda=2$ として $u=e^{2x+\\frac12\\sin y}$。")

    page_refs([("strauss2008", "第1章:1階線形偏微分方程式と特性曲線, 第4章:変数分離法。"),
               ("matano2024", "熱・波動方程式とフーリエ級数・フーリエ変換。"),
               ("fourier1822", "熱伝導方程式を変数分離と三角級数で解いた出発点。"),
               ("evans2010", "1階の偏微分方程式(特性曲線)と解の一意性(発展)。")])
