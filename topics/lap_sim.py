"""② ラプラス変換 第2章:ばね・ダンパーの振動(アニメーション)

微分方程式  y'' + 2ζω₀ y' + ω₀² y = r(t)  の解を、質点の動きとグラフで同時に再生する。
表示は「ばね・ダンパー」か「1層の建物(床=質量、柱=ばね)」を選べる。
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import C1, C2, C4, GRAY, fmt_c, layout
from topics.laplace import rk4
from topics.player import player

NF = 161


@st.cache_data(show_spinner=False, max_entries=32)
def solve(w0, zeta, y0, v0, inp, K, w, T, n=8001):
    b, c = 2 * zeta * w0, w0 ** 2
    if inp == "インパルス":
        v0 = v0 + K                          # r = Kδ(t):速度が K だけ跳ぶ
    tt = np.linspace(0, T, n)

    def r(t):
        if inp == "ステップ":
            return K
        if inp == "正弦波":
            return K * np.sin(w * t)
        return 0.0
    sol = rk4(lambda t, Y: np.array([Y[1], r(t) - b * Y[1] - c * Y[0]]), np.array([y0, v0], float), tt)
    rr = np.array([r(t) for t in tt])
    return tt, sol[:, 0], rr


def _zigzag(x0, x1, y, coils=9, amp=0.17):
    xs = np.linspace(x0, x1, 2 * coils + 3)
    ys = np.full_like(xs, y)
    ys[1:-1:2] += amp
    ys[2:-1:2] -= amp
    ys[1], ys[-2] = y, y
    return xs, ys


def _line(xs, ys, color="#333", width=2.5, name=None, dash=None, fill=None, fillcolor=None):
    return go.Scatter(x=list(xs), y=list(ys), mode="lines", line=dict(color=color, width=width, dash=dash),
                      showlegend=False, hoverinfo="skip", fill=fill, fillcolor=fillcolor, name=name)


def spring_traces(d, f):
    """ばね・ダンパー:d は表示用に縮めた変位、f は外力の向き(-1〜1)"""
    X = 3.5 + d                                 # 質点の中心
    tr = [_line([0, 0], [-1.3, 1.3], width=6)]  # 壁
    for yy in np.linspace(-1.2, 1.2, 9):
        tr.append(_line([0, -0.25], [yy, yy - 0.2], width=1.2))
    zx, zy = _zigzag(0, X - 0.6, 0.5)
    tr.append(_line(zx, zy, color=C1, width=2.5))
    # ダンパー(シリンダーは壁に固定、ピストンは質点といっしょに動く)
    tr.append(_line([0, 0.5], [-0.5, -0.5]))
    tr.append(_line([2.0, 0.5, 0.5, 2.0], [-0.25, -0.25, -0.75, -0.75], color=C4, width=2.5))
    xp = X - 2.25
    tr.append(_line([xp, X - 0.6], [-0.5, -0.5], color=C4))
    tr.append(_line([xp, xp], [-0.33, -0.67], color=C4, width=5))
    # 質点
    tr.append(_line([X - 0.6, X + 0.6, X + 0.6, X - 0.6, X - 0.6], [-0.9, -0.9, 0.9, 0.9, -0.9],
                    color="#222", fill="toself", fillcolor="rgba(214,39,40,0.75)"))
    tr.append(_line([-0.3, 5.6], [-1.0, -1.0], color=GRAY, width=1.5))       # 床
    tr.append(_line([3.5, 3.5], [-1.25, 1.25], color=GRAY, width=1, dash="dot"))  # つり合いの位置
    if abs(f) > 1e-3:
        tr.append(_line([X + 0.6, X + 0.6 + 1.0 * f], [0, 0], color=C2, width=4))
        tr.append(go.Scatter(x=[X + 0.6 + 1.0 * f], y=[0], mode="markers", showlegend=False, hoverinfo="skip",
                             marker=dict(symbol="triangle-right" if f > 0 else "triangle-left", size=14, color=C2)))
    return tr, (-0.6, 6.4), (-1.6, 1.6)


def building_traces(d, f):
    """1層の建物:床スラブ = 質量、柱 = ばね。柱は両端固定のはりの変形 3ξ²−2ξ³ で描く"""
    H = 2.5
    z = np.linspace(0, H, 30)
    xi = z / H
    shape = 3 * xi ** 2 - 2 * xi ** 3
    tr = [_line([-2.4, 2.4], [0, 0], width=4)]
    for xx in np.linspace(-2.3, 2.1, 12):
        tr.append(_line([xx, xx + 0.25], [0, -0.25], width=1.2))
    for x0 in (-1.1, 1.1):
        tr.append(_line(x0 + d * shape, z, color=C1, width=6))
    tr.append(_line([-1.4 + d, 1.4 + d, 1.4 + d, -1.4 + d, -1.4 + d], [H, H, H + 0.35, H + 0.35, H],
                    color="#222", fill="toself", fillcolor="rgba(214,39,40,0.75)"))
    tr.append(_line([0, 0], [0, H + 0.6], color=GRAY, width=1, dash="dot"))
    if abs(f) > 1e-3:
        tr.append(_line([1.4 + d, 1.4 + d + 0.5 * f], [H + 0.17, H + 0.17], color=C2, width=4))
        tr.append(go.Scatter(x=[1.4 + d + 0.5 * f], y=[H + 0.17], mode="markers", showlegend=False, hoverinfo="skip",
                             marker=dict(symbol="triangle-right" if f > 0 else "triangle-left", size=14, color=C2)))
    return tr, (-3.0, 3.0), (-0.5, 3.4)


def frame_fig(view, tt, y, rr, i_full, scale, fscale, title):
    d = float(np.clip(y[i_full] * scale, -1.0, 1.0))
    f = float(rr[i_full] * fscale)
    mech, xr, yr = (spring_traces if view.startswith("ばね") else building_traces)(d, f)
    fig = go.Figure()
    for tr in mech:
        fig.add_trace(tr)
    # 右:y(t) のグラフ(ここまでの道のり)
    fig.add_trace(go.Scatter(x=tt, y=y, mode="lines", line=dict(color="#cccccc", width=1.5),
                             xaxis="x2", yaxis="y2", name="y(t)(全体)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=tt[:i_full + 1], y=y[:i_full + 1], mode="lines", line=dict(color=C1, width=2.8),
                             xaxis="x2", yaxis="y2", name="y(t)"))
    if np.any(rr != 0):
        fig.add_trace(go.Scatter(x=tt, y=rr, mode="lines", line=dict(color=C2, width=1.2, dash="dot"),
                                 xaxis="x2", yaxis="y2", name="外力 r(t)"))
    fig.add_trace(go.Scatter(x=[tt[i_full]], y=[y[i_full]], mode="markers", marker=dict(size=11, color=C2),
                             xaxis="x2", yaxis="y2", showlegend=False))
    lo = min(float(y.min()), float(rr.min()) if np.any(rr != 0) else 0.0)
    hi = max(float(y.max()), float(rr.max()) if np.any(rr != 0) else 0.0)
    pad = 0.08 * (hi - lo + 1e-9)
    layout(fig, title=title, height=430)
    fig.update_layout(
        xaxis=dict(domain=[0, 0.4], range=list(xr), visible=False),
        yaxis=dict(range=list(yr), visible=False, scaleanchor="x", scaleratio=1),
        xaxis2=dict(domain=[0.48, 1], range=[0, float(tt[-1])], title_text="t", anchor="y2"),
        yaxis2=dict(range=[lo - pad, hi + pad], anchor="x2", title_text="y"),
        uirevision="keep",
    )
    return fig


def tab_spring():
    st.info("ラプラス変換で解いた $y(t)$ を、質点の動きとして再生する。"
            "減衰比 $\\zeta$ と外力を変えて「▶ シミュレーション開始」を押そう。")
    st.latex(r"y''+2\zeta\omega_0\,y'+\omega_0^2\,y=r(t),\qquad y(0)=y_0,\ y'(0)=v_0")
    view = st.radio("表示", ["ばね・ダンパー", "1層の建物(床 = 質量, 柱 = ばね)"], horizontal=True, key="ls_view")
    c1, c2, c3, c4 = st.columns(4)
    w0 = c1.slider("固有角振動数 ω₀", 0.5, 4.0, 2.0, 0.1, key="ls_w0")
    zeta = c2.slider("減衰比 ζ", 0.0, 2.0, 0.1, 0.05, key="ls_z")
    y0 = c3.slider("初期変位 y₀", -2.0, 2.0, 1.0, 0.1, key="ls_y0")
    v0 = c4.slider("初速度 v₀", -4.0, 4.0, 0.0, 0.1, key="ls_v0")
    d1, d2, d3, d4 = st.columns(4)
    inp = d1.selectbox("外力 r(t)", ["なし", "ステップ", "正弦波", "インパルス"], key="ls_inp")
    K = d2.slider("大きさ K", 0.0, 4.0, 1.0, 0.1, key="ls_K", disabled=inp == "なし")
    w = d3.slider("外力の角振動数 ω", 0.2, 6.0, 2.0, 0.1, key="ls_w", disabled=inp != "正弦波")
    T = d4.select_slider("時間 T", [10.0, 20.0, 30.0, 40.0], value=20.0, key="ls_T")

    b, c = 2 * zeta * w0, w0 ** 2
    Rs = {"なし": "0", "ステップ": rf"\frac{{{K:g}}}{{s}}", "正弦波": rf"\frac{{{K * w:g}}}{{s^2+{w * w:g}}}",
          "インパルス": f"{K:g}"}[inp]
    st.latex(rf"Y(s)=\frac{{{Rs}+{y0:g}s+{b * y0 + v0:.3g}}}{{s^2+{b:.3g}s+{c:.3g}}}".replace("+-", "-"))
    poles = np.roots([1, b, c]).astype(complex)
    disc = zeta ** 2 - 1
    kind = ("単振動(減衰なし)" if zeta == 0 else "臨界減衰" if abs(disc) < 1e-12
            else "減衰振動" if disc < 0 else "過減衰")
    msg = f"極(分母の根):{fmt_c(poles[0], 3)}, {fmt_c(poles[1], 3)} → **{kind}**"
    if inp == "正弦波" and zeta < 0.3 and abs(w - w0) < 0.3:
        msg += "  外力の角振動数が ω₀ に近いので**共振**する。"
    st.markdown(msg)

    tt, y, rr = solve(w0, zeta, y0, v0, inp, K, w, T)
    idx = np.linspace(0, len(tt) - 1, NF).round().astype(int)
    scale = 1.0 / max(float(np.abs(y).max()), 1e-9) * (0.6 if view.startswith("ばね") else 0.7)
    fscale = 0.9 / max(float(np.abs(rr).max()), 1e-9) if np.any(rr != 0) else 0.0
    labels = [f"t = {tt[i]:.2f}" for i in idx]

    def draw(j):
        return frame_fig(view, tt, y, rr, int(idx[j]), scale, fscale, f"{kind}  {labels[j]}")

    player("springsim", labels, draw)
    st.caption("左の図の変位は見やすいように縮尺している(最大の変位が一定の幅になる)。赤い矢印は外力。")
    with st.expander("土木・建築での見かた"):
        st.markdown(
            "- 建物を「床の質量 $m$ と柱の水平剛性 $k$」の1質点系とみなすと、$my''+cy'+ky=F(t)$ となり、"
            "両辺を $m$ で割るとこの式になる($\\omega_0=\\sqrt{k/m}$, 固有周期 $T_0=2\\pi/\\omega_0$)。\n"
            "- 建物の減衰比は $\\zeta=0.02$〜$0.05$ 程度と小さく、外力の周期が固有周期に近いと**共振**して大きく揺れる。"
            "制振ダンパーは $\\zeta$ を大きくして揺れを抑える工夫。\n"
            "- 伝達関数 $G(s)=\\dfrac{1}{s^2+2\\zeta\\omega_0s+\\omega_0^2}$ の極が左半平面にあれば揺れは収まる(安定)。")
