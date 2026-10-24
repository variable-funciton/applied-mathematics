"""全分野で共通に使う描画・数値計算の部品"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# 色(系列の区別用)
C1, C2, C3, C4 = "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"
GRAY = "#888888"


def trapz(y, x=None, dx=1.0, axis=-1):
    """numpy のバージョン差(trapz / trapezoid)を吸収する台形公式"""
    f = getattr(np, "trapezoid", None) or np.trapz
    return f(y, x=x, dx=dx, axis=axis)


def layout(fig, title=None, height=420, equal=False, xlab=None, ylab=None,
           xrange=None, yrange=None, legend=True):
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=45, r=15, t=55 if title else 25, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        showlegend=legend,
    )
    if xlab:
        fig.update_xaxes(title_text=xlab)
    if ylab:
        fig.update_yaxes(title_text=ylab)
    if xrange is not None:
        fig.update_xaxes(range=list(xrange))
    if yrange is not None:
        fig.update_yaxes(range=list(yrange))
    if equal:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch")


def quiver(X, Y, U, V, length=0.25, normalize=True, color=C1, name="ベクトル場",
           width=1.4, head=0.35):
    """矢印の集まりを1本の Scatter(線)として返す。
    normalize=True なら向きだけ(長さを揃える)、False なら最大の長さが length になるよう縮尺。"""
    X, Y, U, V = (np.ravel(a).astype(float) for a in (X, Y, U, V))
    mag = np.hypot(U, V)
    ok = np.isfinite(mag) & (mag > 1e-12)
    X, Y, U, V, mag = X[ok], Y[ok], U[ok], V[ok], mag[ok]
    if normalize:
        U, V = U / mag * length, V / mag * length
    else:
        s = length / mag.max() if mag.size else 1.0
        U, V = U * s, V * s
    xs, ys = [], []
    for x, y, u, v in zip(X, Y, U, V):
        x1, y1 = x + u, y + v
        xs += [x, x1, None]
        ys += [y, y1, None]
        # 矢じり(±25°)
        L = np.hypot(u, v) * head
        ang = np.arctan2(v, u)
        for d in (np.pi - 0.45, np.pi + 0.45):
            xs += [x1, x1 + L * np.cos(ang + d), None]
            ys += [y1, y1 + L * np.sin(ang + d), None]
    return go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=width),
                      name=name, hoverinfo="skip")


def arrow(x0, y0, dx, dy, color=C2, name="", width=3):
    """1本の太い矢印"""
    return quiver([x0], [y0], [dx], [dy], length=1.0, normalize=False, color=color,
                  name=name, width=width, head=0.18) if np.hypot(dx, dy) > 1e-12 else \
        go.Scatter(x=[x0], y=[y0], mode="markers", marker=dict(color=color), name=name)


def animate(x, frames, names, labels, colors=None, dashes=None, static=(),
            title=None, xlab=None, ylab=None, yrange=None, height=430,
            duration=60, label_prefix="t = ", modes=None):
    """折れ線のアニメーション。
    frames : [フレーム][系列] の y 配列
    static : アニメーションしない追加トレース(初期形状など)"""
    k = len(names)
    colors = colors or [C1, C2, C3, C4][:k]
    dashes = dashes or ["solid"] * k
    modes = modes or ["lines"] * k

    def traces(ys):
        return [go.Scatter(x=x, y=y, mode=md, name=nm, marker=dict(size=5),
                           line=dict(color=c, dash=d, width=2.5 if i == 0 else 1.8))
                for i, (y, nm, c, d, md) in enumerate(zip(ys, names, colors, dashes, modes))]

    fig = go.Figure(data=traces(frames[0]) + list(static),
                    frames=[go.Frame(data=traces(ys), name=str(i), traces=list(range(k)))
                            for i, ys in enumerate(frames)])
    steps = [dict(method="animate", label=lab,
                  args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=False),
                                       transition=dict(duration=0))])
             for i, lab in enumerate(labels)]
    fig.update_layout(
        updatemenus=[dict(type="buttons", showactive=False, x=0, y=-0.12, xanchor="left",
                          yanchor="top", direction="left",
                          buttons=[dict(label="▶ 再生", method="animate",
                                        args=[None, dict(frame=dict(duration=duration, redraw=False),
                                                         fromcurrent=True, transition=dict(duration=0))]),
                                   dict(label="■ 停止", method="animate",
                                        args=[[None], dict(mode="immediate",
                                                           frame=dict(duration=0, redraw=False))])])],
        sliders=[dict(active=0, x=0.22, y=-0.08, len=0.78, steps=steps,
                      currentvalue=dict(prefix=label_prefix))],
    )
    layout(fig, title=title, height=height, xlab=xlab, ylab=ylab, yrange=yrange)
    fig.update_layout(margin=dict(b=110))
    return fig


def hsv_to_rgb(h, s, v):
    """numpy 配列版 HSV→RGB(0〜1)"""
    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def fmt_c(z, d=4):
    """複素数を a + bi の形の文字列に"""
    z = complex(z)
    re, im = z.real, z.imag
    if abs(re) < 10 ** (-d):
        re = 0.0
    if abs(im) < 10 ** (-d):
        im = 0.0
    if im == 0:
        return f"{re:.{d}f}"
    if re == 0:
        return f"{im:.{d}f} i"
    return f"{re:.{d}f} {'+' if im > 0 else '−'} {abs(im):.{d}f} i"


# ------------------------------------------------------------
# 3D 用
# ------------------------------------------------------------
def arrow3d(origin, vec, color=C1, name="", width=6, head=0.18, showlegend=True):
    """3D の矢印(線 + 円錐)。トレースのリストを返す"""
    o = np.asarray(origin, float)
    v = np.asarray(vec, float)
    L = np.linalg.norm(v)
    if L < 1e-12:
        return [go.Scatter3d(x=[o[0]], y=[o[1]], z=[o[2]], mode="markers",
                             marker=dict(size=4, color=color), name=name, showlegend=showlegend)]
    tip = o + v
    hl = min(head * max(L, 0.5), 0.45 * L)
    base = tip - v / L * hl
    return [
        go.Scatter3d(x=[o[0], base[0]], y=[o[1], base[1]], z=[o[2], base[2]], mode="lines",
                     line=dict(color=color, width=width), name=name, showlegend=showlegend,
                     legendgroup=name, hoverinfo="skip"),
        go.Cone(x=[tip[0]], y=[tip[1]], z=[tip[2]], u=[v[0] / L], v=[v[1] / L], w=[v[2] / L],
                sizemode="absolute", sizeref=hl, anchor="tip", showscale=False,
                colorscale=[[0, color], [1, color]], legendgroup=name, hoverinfo="skip"),
    ]


def layout3d(fig, height=520, title=None, aspect="data", ranges=None):
    scene = dict(aspectmode=aspect, xaxis_title="x", yaxis_title="y", zaxis_title="z")
    if ranges is not None:
        scene.update(xaxis=dict(range=ranges[0], title="x"), yaxis=dict(range=ranges[1], title="y"),
                     zaxis=dict(range=ranges[2], title="z"))
        scene["aspectmode"] = "cube"
    fig.update_layout(height=height, title=title, scene=scene,
                      margin=dict(l=0, r=0, t=40 if title else 10, b=0),
                      legend=dict(orientation="h", yanchor="bottom", y=0.98, x=0))
    return fig


def cones(X, Y, Z, U, V, W, size=0.5, colorscale="Viridis", name="F"):
    """ベクトル場を円錐で描く(色 = 大きさ)"""
    return go.Cone(x=np.ravel(X), y=np.ravel(Y), z=np.ravel(Z), u=np.ravel(U), v=np.ravel(V),
                   w=np.ravel(W), sizemode="scaled", sizeref=size, anchor="tail",
                   colorscale=colorscale, showscale=False, name=name, hoverinfo="skip")


def laplace_num(g, s, T=60.0, n=60001):
    """数値ラプラス変換 ∫_0^T g(t) e^{-st} dt(g は配列を受け取る関数)"""
    t = np.linspace(0, T, n)
    return trapz(g(t) * np.exp(-s * t), t)


def vec_tex(v, d=3):
    r"""(1,2,3) → (1,\ 2,\ 3)"""
    return r"\left(" + r",\ ".join(f"{float(x):.{d}g}" for x in v) + r"\right)"
