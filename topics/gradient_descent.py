"""① ベクトル解析 発展:勾配降下法

前提:第2章の勾配と方向微分係数。
  1. 考え方(−∇f の向きが最も急に下る)
  2. 更新式とステップ幅 η(進む向きに沿った断面で見る)
  3. 動かして確かめる(等高線の上で点が下る)
  4. 「何倍になるか」で収束を読む(数列の知識は前提にしない)
  5. 使う場面   6. 例題
"""
import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import C1, C2, C3, C4, GRAY, layout
from topics.player import player
from topics.vec_apps import examples

E = np.exp
FUNCS = {
    "丸い谷 x² + y²": dict(
        f=lambda x, y: x ** 2 + y ** 2, g=lambda x, y: (2 * x, 2 * y), tex=r"f=x^2+y^2",
        grad=r"\nabla f=(2x,\ 2y)", R=3.0, start=(2.5, 1.5), eta=0.2),
    "細長い谷 x² + 5y²": dict(
        f=lambda x, y: x ** 2 + 5 * y ** 2, g=lambda x, y: (2 * x, 10 * y), tex=r"f=x^2+5y^2",
        grad=r"\nabla f=(2x,\ 10y)", R=3.0, start=(2.5, 1.5), eta=0.15),
    "斜めの谷 x² + xy + y²": dict(
        f=lambda x, y: x ** 2 + x * y + y ** 2, g=lambda x, y: (2 * x + y, x + 2 * y), tex=r"f=x^2+xy+y^2",
        grad=r"\nabla f=(2x+y,\ x+2y)", R=3.0, start=(2.5, -0.5), eta=0.2),
    "谷が2つ (x² − 1)² + y²": dict(
        f=lambda x, y: (x ** 2 - 1) ** 2 + y ** 2, g=lambda x, y: (4 * x * (x ** 2 - 1), 2 * y),
        tex=r"f=(x^2-1)^2+y^2", grad=r"\nabla f=(4x(x^2-1),\ 2y)", R=2.0, start=(0.3, 1.5), eta=0.1),
    "平らな所がある x e^{−(x²+y²)}": dict(
        f=lambda x, y: x * E(-(x ** 2 + y ** 2)), g=lambda x, y: ((1 - 2 * x ** 2) * E(-(x ** 2 + y ** 2)),
                                                                 -2 * x * y * E(-(x ** 2 + y ** 2))),
        tex=r"f=x\,e^{-(x^2+y^2)}", grad=r"\nabla f=e^{-(x^2+y^2)}(1-2x^2,\ -2xy)", R=2.5, start=(0.6, 1.2), eta=0.5),
}
ETAS = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2]


def descend(F, x0, eta, n, limit=1e6):
    """勾配降下法の点列(発散したら途中で止める)"""
    pts = [np.array(x0, float)]
    for _ in range(n):
        p = pts[-1]
        gx, gy = F["g"](p[0], p[1])
        q = p - eta * np.array([gx, gy], float)
        if not np.all(np.isfinite(q)) or np.max(np.abs(q)) > limit:
            break
        pts.append(q)
    return np.array(pts)


@st.cache_data(show_spinner=False, max_entries=16)
def _grid(name, R, m=121):
    F = FUNCS[name]
    g = np.linspace(-R, R, m)
    X, Y = np.meshgrid(g, g)
    return g, F["f"](X, Y)


def contour_fig(name, height=480, title=None, levels=18):
    F = FUNCS[name]
    g, Z = _grid(name, F["R"])
    zmin, zmax = float(np.min(Z)), float(np.percentile(Z, 85))
    fig = go.Figure(go.Contour(x=g, y=g, z=Z, colorscale="Viridis", ncontours=levels, opacity=0.75,
                               contours=dict(start=zmin, end=zmax, size=(zmax - zmin) / levels),
                               colorbar=dict(title="f", thickness=12), hoverinfo="skip"))
    layout(fig, title=title, height=height, equal=True, xlab="x", ylab="y", xrange=(-F["R"], F["R"]), yrange=(-F["R"], F["R"]))
    fig.update_layout(uirevision="keep")
    return fig


def _arrow2d(fig, p, v, color, text=None, width=3):
    fig.add_annotation(x=p[0] + v[0], y=p[1] + v[1], ax=p[0], ay=p[1], xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=3, arrowsize=1.4, arrowwidth=width, arrowcolor=color, text="")
    if text:
        fig.add_annotation(x=p[0] + 1.12 * v[0], y=p[1] + 1.12 * v[1], text=text, showarrow=False,
                           font=dict(size=15, color=color))


# ------------------------------------------------------------
# 1. 考え方
# ------------------------------------------------------------
def fig_idea():
    name = "楕円の谷"
    f = lambda x, y: x ** 2 + 2 * y ** 2  # noqa: E731
    g = np.linspace(-3, 3, 121)
    X, Y = np.meshgrid(g, g)
    fig = go.Figure(go.Contour(x=g, y=g, z=f(X, Y), colorscale="Viridis", ncontours=14, opacity=0.7, showscale=False,
                               hoverinfo="skip", name=name))
    p = np.array([2.0, 1.0])
    gr = np.array([2 * p[0], 4 * p[1]])
    u = gr / np.linalg.norm(gr)
    _arrow2d(fig, p, 0.9 * u, C2, "∇f(最も急に登る)")
    _arrow2d(fig, p, -0.9 * u, C1, "−∇f(最も急に下る)")
    t = np.array([-u[1], u[0]])
    fig.add_trace(go.Scatter(x=[p[0] - 1.2 * t[0], p[0] + 1.2 * t[0]], y=[p[1] - 1.2 * t[1], p[1] + 1.2 * t[1]], mode="lines",
                             line=dict(color="white", width=3, dash="dash"), name="等高線の接線"))
    s = 0.18
    fig.add_trace(go.Scatter(x=[p[0] + s * t[0], p[0] + s * t[0] + s * u[0], p[0] + s * u[0]],
                             y=[p[1] + s * t[1], p[1] + s * t[1] + s * u[1], p[1] + s * u[1]], mode="lines",
                             line=dict(color="white", width=2), showlegend=False))
    fig.add_trace(go.Scatter(x=[p[0]], y=[p[1]], mode="markers+text", text=["x₀"], textposition="bottom left",
                             marker=dict(size=11, color="black"), showlegend=False))
    layout(fig, height=430, equal=True, xlab="x", ylab="y", xrange=(-3, 3), yrange=(-3, 3),
           title="f = x² + 2y² の等高線と、点 x₀ = (2, 1) での勾配")
    return fig


def _sec_idea():
    st.subheader("1. 考え方:山を下るなら、どちらへ進むか")
    a, b = st.columns([1.1, 1])
    with a:
        st.markdown(
            "霧の中の山で、足もとの傾きだけを頼りに谷底へ下りたい。どちらへ一歩を出せばよいか。\n\n"
            "第2章で学んだように、単位ベクトル $\\vec u$ の向きの**方向微分係数**は")
        st.latex(r"\nabla f\cdot\vec u=|\nabla f|\,|\vec u|\cos\theta=|\nabla f|\cos\theta")
        st.markdown(
            "($\\theta$ は $\\nabla f$ と $\\vec u$ のなす角)。これが最も小さくなるのは $\\cos\\theta=-1$、"
            "つまり $\\theta=180^\\circ$ のときで、そのとき値は $-|\\nabla f|$ になる。\n\n"
            "**最も急に下る向きは $-\\nabla f$**(勾配と逆向き)。第2章の「最も増加する向き $\\nabla f$」の裏返しである。"
            "図のように、$\\nabla f$ と $-\\nabla f$ は等高線に垂直になっている。")
    with b:
        st.plotly_chart(fig_idea(), width="stretch", key="gd_idea")


# ------------------------------------------------------------
# 2. 更新式と η
# ------------------------------------------------------------
def _sec_update():
    st.subheader("2. 更新式と、ステップ幅 η の意味")
    st.markdown("$-\\nabla f$ の向きに少し進み、着いた所でまた勾配を計算して進む。これをくり返すのが**勾配降下法**である。")
    st.latex(r"\vec x_{k+1}=\vec x_k-\eta\,\nabla f(\vec x_k)\qquad(k=0,1,2,\ldots)")
    st.markdown("$\\eta>0$ は1歩の大きさを決める数で、**ステップ幅**(機械学習では**学習率**)という。"
                "大きすぎても小さすぎてもうまくいかない。下の図は、点 $\\vec x_0=(2,1)$ から $-\\nabla f$ の向きに進んだときの"
                "$f$ の値(進む向きに沿った断面)である。")
    eta = st.select_slider("ステップ幅 η", [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5], value=0.25, key="gd_eta1")
    f = lambda x, y: x ** 2 + 2 * y ** 2  # noqa: E731
    p = np.array([2.0, 1.0])
    gr = np.array([4.0, 4.0])
    t = np.linspace(0, 0.6, 200)
    phi = f(p[0] - t * gr[0], p[1] - t * gr[1])
    lin = f(*p) - t * (gr @ gr)
    fig = go.Figure(go.Scatter(x=t, y=phi, mode="lines", line=dict(color=C1, width=3), name="f(x₀ − t∇f)(断面)"))
    fig.add_trace(go.Scatter(x=t, y=lin, mode="lines", line=dict(color=GRAY, dash="dot"), name="1次近似 f(x₀) − t|∇f|²"))
    v = f(p[0] - eta * gr[0], p[1] - eta * gr[1])
    fig.add_trace(go.Scatter(x=[0, eta], y=[f(*p), v], mode="markers+text", text=["x₀", "x₁"], textposition="top center",
                             marker=dict(size=11, color=[GRAY, C2]), showlegend=False))
    fig.add_hline(y=f(*p), line=dict(color=GRAY, width=1))
    layout(fig, height=380, xlab="進んだ量 t(1歩のときの t が η)", ylab="f の値", yrange=(-1, 12),
           title=f"η = {eta}:f(x₀) = 6 → f(x₁) = {v:.3f}")
    a, b = st.columns([1.3, 1])
    with a:
        st.plotly_chart(fig, width="stretch", key="gd_sec")
    with b:
        st.markdown(
            "- $f=x^2+2y^2$, $\\nabla f=(2x,\\ 4y)$, $\\vec x_0=(2,\\ 1)$ で $\\nabla f(\\vec x_0)=(4,\\ 4)$。\n"
            f"- $\\eta={eta}$ なら $\\vec x_1=(2,1)-{eta}(4,4)=({2 - 4 * eta:g},\\ {1 - 4 * eta:g})$。\n"
            "- 断面は放物線。はじめは下がるが、進みすぎると上がる。**$\\eta$ が小さいうちは必ず下がる**:"
            "点線の1次近似 $f(\\vec x_0)-t|\\nabla f|^2$ のとおり、下がる速さは $|\\nabla f|^2$。\n"
            "- $\\eta$ を大きくしすぎると、谷を飛び越えて反対側の斜面に着き、かえって $f$ が大きくなる。")
        if v < f(*p):
            st.success(f"f は {f(*p):g} から {v:.3f} に下がった。")
        else:
            st.error(f"f は {f(*p):g} から {v:.3f} に上がってしまった(η が大きすぎる)。")


# ------------------------------------------------------------
# 3. 動かして確かめる
# ------------------------------------------------------------
def _sec_sim():
    st.subheader("3. 動かして確かめる")
    c1, c2, c3, c4 = st.columns([1.6, 1, 1, 1])
    name = c1.selectbox("関数", list(FUNCS), key="gd_f")
    F = FUNCS[name]
    eta = c2.select_slider("ステップ幅 η", ETAS, value=F["eta"], key=f"gd_eta_{name}")
    x0 = c3.slider("出発点 x₀", -F["R"] * 0.95, F["R"] * 0.95, F["start"][0], 0.05, key=f"gd_x_{name}")
    y0 = c4.slider("出発点 y₀", -F["R"] * 0.95, F["R"] * 0.95, F["start"][1], 0.05, key=f"gd_y_{name}")
    st.latex(F["tex"] + r",\qquad " + F["grad"])
    N = 40
    pts = descend(F, (x0, y0), eta, N)
    vals = [float(F["f"](p[0], p[1])) for p in pts]
    diverged = len(pts) < N + 1 or np.max(np.abs(pts[-1])) > F["R"] * 3
    labels = [f"k = {k}" for k in range(len(pts))]
    R = F["R"]

    def draw(k):
        fig = contour_fig(name, height=500, title=f"k = {k}:x = ({pts[k][0]:.3f}, {pts[k][1]:.3f}),  f = {vals[k]:.4f}")
        P = np.clip(pts[:k + 1], -R * 1.5, R * 1.5)
        fig.add_trace(go.Scatter(x=P[:, 0], y=P[:, 1], mode="lines+markers", line=dict(color="white", width=2.5),
                                 marker=dict(size=6, color="white"), name="点の道すじ"))
        fig.add_trace(go.Scatter(x=[P[-1, 0]], y=[P[-1, 1]], mode="markers", marker=dict(size=14, color=C2,
                                 line=dict(color="white", width=2)), name="現在の点"))
        return fig

    a, b = st.columns([3, 2])
    with a:
        player("gd_sim", labels, draw)
    with b:
        st.markdown("**各ステップの点と f の値**")
        rows = [{"k": k, "x": f"{p[0]:.4f}", "y": f"{p[1]:.4f}", "f": f"{v:.5f}"} for k, (p, v) in enumerate(zip(pts, vals))]
        st.dataframe(rows[:21], hide_index=True, width="stretch", height=420)
        if diverged:
            st.error("点が遠くへ飛んでいく(発散)。η を小さくしよう。")
        elif vals[-1] > vals[0]:
            st.warning("f が下がっていない。η が大きすぎる。")
    with st.expander("観察のポイント"):
        st.markdown(
            "- **丸い谷**:$-\\nabla f$ がいつも谷底を向くので、まっすぐ下る。$\\eta=0.5$ なら1歩で谷底に着く。\n"
            "- **細長い谷**:勾配は等高線に垂直なので谷底を向かず、**ジグザグ**になる。$\\eta$ を少し大きくすると発散する"
            "($y$ 方向の傾きが急なため)。\n"
            "- **谷が2つ**:出発点の $x$ が正なら右の谷 $(1,0)$、負なら左の谷 $(-1,0)$ に落ちる。$x_0=0$ だと真ん中の"
            "**鞍点** $(0,0)$ で止まってしまう。勾配降下法が見つけるのは「近くの谷底」で、一番低い所とは限らない。\n"
            "- **平らな所**:勾配が小さい所では1歩が小さく、なかなか進まない。")


# ------------------------------------------------------------
# 4. 何倍になるか
# ------------------------------------------------------------
def _sec_ratio():
    st.subheader("4. 「何倍になるか」で、近づくかどうかを読む")
    st.markdown("$f=ax^2+by^2$($a,b>0$、谷底は原点)なら $\\nabla f=(2ax,\\ 2by)$ なので、更新式は成分ごとに")
    st.latex(r"x_{k+1}=x_k-\eta\cdot2ax_k=(1-2\eta a)\,x_k,\qquad y_{k+1}=(1-2\eta b)\,y_k")
    st.markdown("となる。**1歩ごとに、$x$ 座標は毎回同じ $(1-2\\eta a)$ 倍、$y$ 座標は $(1-2\\eta b)$ 倍される。**")
    c1, c2, c3 = st.columns(3)
    a = c1.select_slider("a", [1, 2, 3, 4, 5], value=1, key="gd_ra")
    b = c2.select_slider("b", [1, 2, 3, 4, 5], value=4, key="gd_rb")
    eta = c3.select_slider("η", [0.05, 0.1, 0.125, 0.2, 0.25, 0.3, 0.4, 0.5], value=0.2, key="gd_reta")
    rx, ry = 1 - 2 * eta * a, 1 - 2 * eta * b
    xs, ys = [4.0], [4.0]
    for _ in range(6):
        xs.append(xs[-1] * rx)
        ys.append(ys[-1] * ry)
    st.markdown(f"$f={'' if a == 1 else a}x^2+{'' if b == 1 else b}y^2$, $\\eta={eta}$, 出発点 $(4,\\ 4)$ のとき、"
                f"倍率は $x$ が ${rx:g}$、$y$ が ${ry:g}$。")
    st.dataframe([{"k": k, "x": f"{x:.4g}", "y": f"{y:.4g}"} for k, (x, y) in enumerate(zip(xs, ys))],
                 hide_index=True, width="stretch")

    def verdict(r):
        if abs(r) < 1:
            return "0 に近づく" + ("(符号を変えながら)" if r < 0 else "")
        if abs(r) == 1:
            return "近づかない(同じ大きさのまま)"
        return "どんどん大きくなる(発散)"
    st.markdown(f"- $x$:倍率 ${rx:g}$ → {verdict(rx)}\n- $y$:倍率 ${ry:g}$ → {verdict(ry)}")
    st.markdown(
        "**まとめ**:同じ数を何回もかけると、その数の絶対値が 1 より小さければ 0 に近づき、1 より大きければ大きくなっていく。"
        "したがって点が谷底に近づく条件は")
    st.latex(r"|1-2\eta a|<1\ \text{かつ}\ |1-2\eta b|<1\iff 0<\eta<\frac1a\ \text{かつ}\ 0<\eta<\frac1b"
             r"\iff 0<\eta<\frac{1}{\max(a,b)}")
    st.caption(f"この例では 0 < η < {1 / max(a, b):g}。急な方向($a,b$ の大きい方)で η の上限が決まる。"
               "3 の「細長い谷」で、少し η を大きくすると発散したのはこのため。"
               "(数列の言葉では、$x_k$ は公比 $1-2\\eta a$ の等比数列。)")


SCENES = [
    ("機械学習(AI)", "誤差を表す関数を、勾配降下法で小さくしていくことを「学習」という。変数(パラメータ)が何百万個あっても、更新式は "
     "$\\vec x_{k+1}=\\vec x_k-\\eta\\nabla f$ のまま。$\\eta$ は「学習率」と呼ばれ、選び方が性能を大きく左右する。"),
    ("機械", "部材の寸法を変数にして、重さやたわみを最小にする最適設計。"),
    ("化学", "分子のエネルギーを原子の位置の関数とみて、勾配(原子にはたらく力の逆向き)にそって下り、安定な形を探す。"),
    ("都市", "いくつかの地点への距離の和が最小になる施設の位置を探す。距離の和の勾配は「各地点から向かう単位ベクトルの和」になる。"),
    ("データに直線を合わせる", "測定した点の列に一番近い直線 $y=px+q$ を探すことも、誤差の2乗の和を $p,\\ q$ の関数とみて勾配降下法で解ける"
     "(5年の確率統計で扱う内容の入口)。"),
    ("$n$ 変数のとき", "$\\nabla f=\\left(\\dfrac{\\partial f}{\\partial x_1},\\ldots,\\dfrac{\\partial f}{\\partial x_n}\\right)$ として、"
     "同じ更新式がそのまま使える。"),
]

EXAMPLES = [
    (r"$f(x,y)=x^2+2y^2$ に、出発点 $\vec x_0=(2,\ 1)$、ステップ幅 $\eta=\dfrac14$ で勾配降下法を使う。$\vec x_1,\ \vec x_2$ を求め、"
     r"$f(\vec x_0),\ f(\vec x_1),\ f(\vec x_2)$ を比べよ。",
     [r"$\nabla f=(2x,\ 4y)$。$\nabla f(2,1)=(4,\ 4)$ より $\vec x_1=(2,1)-\dfrac14(4,4)=(1,\ 0)$。",
      r"$\nabla f(1,0)=(2,\ 0)$ より $\vec x_2=(1,0)-\dfrac14(2,0)=\left(\dfrac12,\ 0\right)$。",
      r"$f(\vec x_0)=6,\ f(\vec x_1)=1,\ f(\vec x_2)=\dfrac14$ と下がっていく(最小値は $f(0,0)=0$)。"]),
    (r"$f(x,y)=x^2+4y^2$ に勾配降下法を使うとき、どんな出発点からでも点が谷底 $(0,0)$ に近づくための $\eta$ の範囲を求めよ。",
     [r"$\nabla f=(2x,\ 8y)$ なので $x_{k+1}=(1-2\eta)x_k,\ y_{k+1}=(1-8\eta)y_k$。",
      r"近づく条件は $|1-2\eta|<1$ かつ $|1-8\eta|<1$。$|1-2\eta|<1\iff0<\eta<1$、$|1-8\eta|<1\iff0<\eta<\dfrac14$。",
      r"両方を満たすのは $0<\eta<\dfrac14$。急な $y$ 方向で上限が決まる。"]),
    (r"$f(x,y)=(x-1)^2+(y+2)^2$ では、どの出発点からでも1歩で最小点 $(1,\ -2)$ に着く $\eta$ がある。その値を求めよ。"
     r"また、$f=(x-1)^2+3(y+2)^2$ ではそのような $\eta$ がない理由を説明せよ。",
     [r"$\nabla f=(2(x-1),\ 2(y+2))$ より $x_1-1=(1-2\eta)(x_0-1),\ y_1+2=(1-2\eta)(y_0+2)$。",
      r"$1-2\eta=0$、つまり $\eta=\dfrac12$ なら $x_1=1,\ y_1=-2$ で、1歩で着く(丸い谷では $-\nabla f$ がちょうど最小点を向く)。",
      r"$f=(x-1)^2+3(y+2)^2$ では倍率が $1-2\eta$ と $1-6\eta$ で、同時に $0$ にはできない。"
      r"$-\nabla f$ が最小点を向かない(細長い谷)ので、ふつうは1歩では着かない。"]),
]


def tab_gradient_descent():
    st.info("関数の値が一番小さくなる点を、**勾配の逆向きに少しずつ進んで**探す方法。前提:第2章の勾配と方向微分係数"
            "($\\nabla f$ は最も増加する向き、方向微分係数は $\\nabla f\\cdot\\vec u$)。")
    _sec_idea()
    _sec_update()
    _sec_sim()
    _sec_ratio()
    st.subheader("5. どんな場面で使うか")
    for nm, txt in SCENES:
        st.markdown(f"- **{nm}**:{txt}")
    examples("gd", EXAMPLES)
