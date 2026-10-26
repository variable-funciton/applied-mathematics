"""① ベクトル解析 第2章:ベクトル場の例 ― 天気予報の風

1. 風向・風速から風のベクトル (u, v) を作る(風向は「吹いてくる方向」)
2. モデルの渦(ランキン渦 + 吹き込み):回転・発散と、閉曲線に沿う循環・流出量
3. 実際の予報データ(Open-Meteo、APIキー不要)の風:差分で rot・div を計算する
"""
import json
import math
import urllib.request

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import C1, C2, C3, GRAY, layout, quiver

DIR16 = ["北", "北北東", "北東", "東北東", "東", "東南東", "南東", "南南東",
         "南", "南南西", "南西", "西南西", "西", "西北西", "北西", "北北西"]
EARTH_R = 6.371e6   # m


def wind_uv(speed, deg):
    """風速 V と風向 θ(北から時計回り。風が「吹いてくる」方向)→ (u 東向き, v 北向き)"""
    th = np.radians(deg)
    return -speed * np.sin(th), -speed * np.cos(th)


# ------------------------------------------------------------
# 1. 風向・風速 → ベクトル
# ------------------------------------------------------------
def _compass(deg, V):
    u, v = wind_uv(V, deg)
    fig = go.Figure()
    t = np.linspace(0, 2 * np.pi, 120)
    fig.add_trace(go.Scatter(x=np.cos(t), y=np.sin(t), mode="lines", line=dict(color=GRAY, width=1), hoverinfo="skip"))
    for k, nm in enumerate(DIR16[::2]):
        a = np.radians(45 * k)
        fig.add_annotation(x=1.18 * np.sin(a), y=1.18 * np.cos(a), text=nm, showarrow=False, font=dict(size=13))
    th = np.radians(deg)
    up = (np.sin(th), np.cos(th))                    # 風上(吹いてくる方)
    m = max(math.hypot(u, v), 1e-9)
    fig.add_annotation(x=0.9 * u / m * 0.9, y=0.9 * v / m * 0.9, ax=up[0] * 0.9, ay=up[1] * 0.9,
                       xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.5,
                       arrowwidth=4, arrowcolor=C2, text="")
    fig.add_trace(go.Scatter(x=[up[0] * 0.9], y=[up[1] * 0.9], mode="markers+text", text=["風上"],
                             textposition="top center", marker=dict(size=10, color=C1), hoverinfo="skip"))
    layout(fig, height=360, equal=True, xrange=(-1.4, 1.4), yrange=(-1.4, 1.4), legend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def _section_vector():
    st.subheader("1. 風向・風速から風のベクトルを作る")
    st.markdown(
        "天気予報の「北の風 5 m/s」は、**北から吹いてくる**風のこと。空気は**南へ**動くので、"
        "ベクトルは**南向き**になる(風向は「行き先」ではなく「来る方向」)。")
    c1, c2 = st.columns([1, 1.2])
    name = c1.select_slider("風向(16方位)", DIR16, value="北西", key="wd_dir")
    V = c1.slider("風速 V [m/s]", 0.0, 30.0, 8.0, 0.5, key="wd_V")
    deg = 22.5 * DIR16.index(name)
    u, v = wind_uv(V, deg)
    c1.latex(rf"\theta={deg:g}^\circ\ (\text{{北から時計回り}})")
    c1.latex(r"u=-V\sin\theta,\quad v=-V\cos\theta")
    c1.latex(rf"\vec v=(u,\ v)=({u:.2f},\ {v:.2f})\ \mathrm{{m/s}}")
    c1.caption("u は東向き、v は北向きの成分。気象では x 軸を東、y 軸を北にとる。")
    with c2:
        st.plotly_chart(_compass(deg, V), width="stretch", key="wd_compass")
        st.caption("青い点が風上(風が吹いてくる方)、赤い矢印が空気の動く向き(ベクトル)。")


# ------------------------------------------------------------
# 2. モデルの渦
# ------------------------------------------------------------
def rankine(X, Y, Vm, rm, alpha, sign):
    """ランキン渦 + 吹き込み角 α。sign=+1 低気圧(反時計回り・吹き込み)、-1 高気圧(時計回り・吹き出し)
    戻り値 u, v, rot, div(X, Y, rm は km、速さは m/s、rot・div は 1/s)"""
    r = np.hypot(X, Y)
    with np.errstate(divide="ignore", invalid="ignore"):
        vt = np.where(r < rm, Vm * r / rm, Vm * rm / np.where(r > 0, r, 1))
        k = np.tan(np.radians(alpha))
        vr = -k * vt                                       # 中心向き(吹き込み)
        ct, sn = np.where(r > 0, X / np.where(r > 0, r, 1), 0), np.where(r > 0, Y / np.where(r > 0, r, 1), 0)
    u = sign * (-vt * sn) + sign * vr * ct
    v = sign * (vt * ct) + sign * vr * sn
    inside = r < rm
    rot = np.where(inside, 2 * Vm / (rm * 1e3), 0.0) * sign
    div = np.where(inside, -2 * k * Vm / (rm * 1e3), 0.0) * sign
    return u, v, rot, div


def circle_integrals(Vm, rm, alpha, sign, rho, n=720):
    """半径 rho [km] の円(反時計回り)に沿う循環 ∮ v·dr と、外向きの流出量 ∮ v·n ds [m²/s] を数値積分で"""
    t = np.linspace(0, 2 * np.pi, n + 1)
    X, Y = rho * np.cos(t), rho * np.sin(t)
    u, v, _, _ = rankine(X, Y, Vm, rm, alpha, sign)
    tx, ty = -np.sin(t), np.cos(t)
    ds = rho * 1e3 * (t[1] - t[0])
    circ = float(np.sum((u * tx + v * ty)[:-1]) * ds)
    flux = float(np.sum((u * np.cos(t) + v * np.sin(t))[:-1]) * ds)
    return circ, flux


def model_exact(Vm, rm, alpha, sign, rho):
    """∬ rot dA と ∬ div dA(厳密値)"""
    k = math.tan(math.radians(alpha))
    a = min(rho, rm) * 1e3
    area_in = math.pi * a * a
    return sign * 2 * Vm / (rm * 1e3) * area_in, -sign * 2 * k * Vm / (rm * 1e3) * area_in


def _section_model():
    st.subheader("2. モデルの渦で発散と回転を見る")
    st.markdown(
        "台風を簡単にしたモデル(**ランキン渦**)。中心から半径 $r_m$ までは剛体のように回り、外側では風速が "
        "$1/r$ で弱まる。地表付近では摩擦のため、風は等圧線から**吹き込み角** $\\alpha$ だけ中心側へずれて吹き込む。")
    c1, c2, c3, c4 = st.columns(4)
    kind = c1.radio("気圧", ["低気圧・台風(北半球)", "高気圧(北半球)"], key="wm_kind")
    sign = 1 if kind.startswith("低") else -1
    Vm = c2.slider("最大風速 Vₘ [m/s]", 10, 60, 40, 5, key="wm_V")
    rm = c3.slider("最大風速半径 rₘ [km]", 20, 150, 60, 10, key="wm_rm")
    alpha = c4.slider("吹き込み角 α [°]", 0, 45, 20, 5, key="wm_a")
    d1, d2 = st.columns(2)
    color = d1.radio("背景の色", ["回転 rot", "発散 div", "風速"], horizontal=True, key="wm_col")
    rho = d2.slider("閉曲線(円)の半径 ρ [km]", 10, 300, 100, 10, key="wm_rho")

    L = 300
    g = np.linspace(-L, L, 17)
    X, Y = np.meshgrid(g, g)
    u, v, _, _ = rankine(X, Y, Vm, rm, alpha, sign)
    gb = np.linspace(-L, L, 161)
    XB, YB = np.meshgrid(gb, gb)
    ub, vb, rotb, divb = rankine(XB, YB, Vm, rm, alpha, sign)
    if color.startswith("回転"):
        z, cs, title, zmid = rotb * 1e5, "RdBu_r", "rot [×10⁻⁵ /s]", 0
    elif color.startswith("発散"):
        z, cs, title, zmid = divb * 1e5, "PuOr", "div [×10⁻⁵ /s]", 0
    else:
        z, cs, title, zmid = np.hypot(ub, vb), "Blues", "風速 [m/s]", None
    fig = go.Figure(go.Heatmap(x=gb, y=gb, z=z, colorscale=cs, zmid=zmid, opacity=0.6,
                               colorbar=dict(title=title, thickness=12), hoverinfo="skip"))
    fig.add_trace(quiver(X, Y, u, v, length=0.9 * (g[1] - g[0]), normalize=False, color="#222", name="風"))
    t = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(go.Scatter(x=rho * np.cos(t), y=rho * np.sin(t), mode="lines", line=dict(color=C3, width=3),
                             name="閉曲線 C(反時計回り)"))
    layout(fig, height=560, equal=True, xlab="東西 x [km]", ylab="南北 y [km]", xrange=(-L, L), yrange=(-L, L))
    cA, cB = st.columns([3, 2])
    with cA:
        st.plotly_chart(fig, width="stretch", key="wm_fig")
    circ, flux = circle_integrals(Vm, rm, alpha, sign, rho)
    R_ex, D_ex = model_exact(Vm, rm, alpha, sign, rho)
    with cB:
        st.markdown("**このモデルの回転・発散**($r<r_m$ の内側)")
        k = math.tan(math.radians(alpha))
        st.latex(rf"\mathrm{{rot}}\,\vec v={sign * 2 * Vm / (rm * 1e3) * 1e5:+.2f}\times10^{{-5}}\ /\mathrm{{s}}")
        st.latex(rf"\mathrm{{div}}\,\vec v={-sign * 2 * k * Vm / (rm * 1e3) * 1e5:+.2f}\times10^{{-5}}\ /\mathrm{{s}}")
        st.markdown("外側 $r>r_m$ では、風が回っているのに $\\mathrm{rot}=0$、$\\mathrm{div}=0$ になる。")
        st.markdown("**閉曲線 C で確かめる**(単位 m²/s)")
        st.markdown(
            f"| | 線積分(数値) | 2重積分 |\n|---|---|---|\n"
            f"| 循環 $\\oint_C\\vec v\\cdot d\\vec r$ と $\\iint\\mathrm{{rot}}\\,dA$ | {circ:.4g} | {R_ex:.4g} |\n"
            f"| 流出量 $\\oint_C\\vec v\\cdot\\vec n\\,ds$ と $\\iint\\mathrm{{div}}\\,dA$ | {flux:.4g} | {D_ex:.4g} |")
        st.caption("どちらも一致する(グリーンの定理。ストークスの定理・ガウスの発散定理の平面版)。"
                   "ρ を rₘ より大きくしても循環は変わらない:回転は内側だけにある。")
    with st.expander("読み取ってみよう"):
        st.markdown(
            "- 低気圧(台風)では $\\mathrm{rot}>0$(反時計回り)、$\\mathrm{div}<0$(吹き込み = 収束)。"
            "収束した空気は上昇し、雲ができて雨になる。\n"
            "- 高気圧ではその逆で、$\\mathrm{rot}<0$(時計回り)、$\\mathrm{div}>0$(吹き出し = 発散)。"
            "空気は上から降りてくる(下降気流)ので晴れやすい。\n"
            "- 吹き込み角 $\\alpha=0$ にすると $\\mathrm{div}=0$ になり、風は等圧線(円)に沿って回るだけになる。")


# ------------------------------------------------------------
# 3. 実際の予報データ
# ------------------------------------------------------------
REGIONS = {
    "日本付近(北緯 26〜46°、東経 126〜148°)": (26, 46, 126, 148, 2.0),
    "本州付近(北緯 31〜42°、東経 130〜143°)": (31, 42, 130, 143, 1.0),
    "九州・沖縄付近(北緯 24〜34°、東経 123〜133°)": (24, 34, 123, 133, 1.0),
}


@st.cache_data(ttl=3600, show_spinner="予報データを取得しています…")
def fetch_wind(lat0, lat1, lon0, lon1, step):
    lats = np.arange(lat0, lat1 + 1e-9, step)
    lons = np.arange(lon0, lon1 + 1e-9, step)
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    q = ("https://api.open-meteo.com/v1/forecast?latitude=" + ",".join(f"{x:g}" for x in LA.ravel())
         + "&longitude=" + ",".join(f"{x:g}" for x in LO.ravel())
         + "&hourly=wind_speed_10m,wind_direction_10m&wind_speed_unit=ms&forecast_days=2&timezone=Asia%2FTokyo")
    req = urllib.request.Request(q, headers={"User-Agent": "applied-math-lab (education)"})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read().decode("utf-8"))
    return parse_wind(data, lats, lons)


def parse_wind(data, lats, lons):
    items = data if isinstance(data, list) else [data]
    if len(items) != len(lats) * len(lons):
        raise ValueError("地点の数が合わない")
    times = items[0]["hourly"]["time"]
    S = np.array([it["hourly"]["wind_speed_10m"] for it in items], dtype=float)
    D = np.array([it["hourly"]["wind_direction_10m"] for it in items], dtype=float)
    S = S.reshape(len(lats), len(lons), -1)
    D = D.reshape(len(lats), len(lons), -1)
    return lats, lons, times, S, D


def div_rot_sphere(lats, lons, u, v):
    """格子点の風から差分で div・rot [1/s](球面上の距離 dx = R cosφ dλ, dy = R dφ。曲率の小さな項は省く)"""
    phi = np.radians(lats)[:, None]
    dlam = np.radians(lons[1] - lons[0])
    dphi = np.radians(lats[1] - lats[0])
    du_dl = np.gradient(u, dlam, axis=1)
    dv_dl = np.gradient(v, dlam, axis=1)
    du_dp = np.gradient(u, dphi, axis=0)
    dv_dp = np.gradient(v, dphi, axis=0)
    div = du_dl / (EARTH_R * np.cos(phi)) + dv_dp / EARTH_R
    rot = dv_dl / (EARTH_R * np.cos(phi)) - du_dp / EARTH_R
    return div, rot


def geo_quiver(lats, lons, u, v, step):
    """地図上の矢印(長さは風速に比例、最大で格子間隔の 0.9 倍)"""
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    m = np.nanmax(np.hypot(u, v)) or 1.0
    s = 0.9 * step / m
    xs, ys = [], []
    for la, lo, uu, vv in zip(LA.ravel(), LO.ravel(), u.ravel(), v.ravel()):
        if not (np.isfinite(uu) and np.isfinite(vv)):
            continue
        dlo, dla = s * uu / max(math.cos(math.radians(la)), 0.2), s * vv
        x1, y1 = lo + dlo, la + dla
        xs += [lo, x1, None]
        ys += [la, y1, None]
        ang = math.atan2(dla, dlo)
        h = 0.3 * math.hypot(dlo, dla)
        for d in (math.pi - 0.45, math.pi + 0.45):
            xs += [x1, x1 + h * math.cos(ang + d), None]
            ys += [y1, y1 + h * math.sin(ang + d), None]
    return go.Scattergeo(lon=xs, lat=ys, mode="lines", line=dict(color="#222", width=1.3), name="風", hoverinfo="skip")


def _section_real():
    st.subheader("3. 実際の予報データで見る")
    st.markdown(
        "Open-Meteo の数値予報(地上 10 m の風)を、格子状の地点で取り出して描く。"
        "格子点の値しかないので、$\\mathrm{div}$ と $\\mathrm{rot}$ は**差分**で近似する(偏微分方程式の差分法と同じ考え方)。")
    st.caption("教材用に予報データを表示しているだけで、このアプリが天気を予報するものではない。"
               "データ:Open-Meteo.com(CC BY 4.0)。")
    c1, c2 = st.columns([1.4, 1])
    reg = c1.selectbox("範囲", list(REGIONS), key="wr_reg")
    color = c2.radio("色", ["回転 rot", "発散 div", "風速"], horizontal=True, key="wr_col")
    if not st.toggle("予報データを取得して表示する", value=False, key="wr_on"):
        st.caption("オンにすると、インターネットから最新の予報データを取得する(1時間ごとに更新)。")
        return
    lat0, lat1, lon0, lon1, step = REGIONS[reg]
    try:
        lats, lons, times, S, D = fetch_wind(lat0, lat1, lon0, lon1, step)
    except Exception as e:  # ネットワーク・API の不調
        st.warning(f"予報データを取得できなかった({type(e).__name__})。時間をおいて試すか、上のモデルの渦で確かめよう。")
        return
    k = st.select_slider("時刻(日本時間)", list(range(len(times))), value=min(12, len(times) - 1),
                         format_func=lambda i: times[i].replace("T", " "), key="wr_t")
    u, v = wind_uv(S[:, :, k], D[:, :, k])
    div, rot = div_rot_sphere(lats, lons, u, v)
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    if color.startswith("回転"):
        z, cs, title, zmid = rot * 1e5, "RdBu_r", "rot [×10⁻⁵ /s]", 0
    elif color.startswith("発散"):
        z, cs, title, zmid = div * 1e5, "PuOr", "div [×10⁻⁵ /s]", 0
    else:
        z, cs, title, zmid = np.hypot(u, v), "Blues", "風速 [m/s]", None
    lim = float(np.nanpercentile(np.abs(z), 95)) if zmid == 0 else None
    fig = go.Figure(go.Scattergeo(lon=LO.ravel(), lat=LA.ravel(), mode="markers",
                                  marker=dict(size=22 if step >= 2 else 16, color=z.ravel(), colorscale=cs,
                                              cmid=zmid, cmin=-lim if lim else None, cmax=lim if lim else None,
                                              opacity=0.75, colorbar=dict(title=title, thickness=12)),
                                  text=[f"風速 {s:.1f} m/s, 風向 {d:.0f}°" for s, d in zip(S[:, :, k].ravel(), D[:, :, k].ravel())],
                                  hoverinfo="text", name="格子点"))
    fig.add_trace(geo_quiver(lats, lons, u, v, step))
    fig.update_geos(projection_type="mercator", lataxis_range=[lat0 - 1, lat1 + 1], lonaxis_range=[lon0 - 1, lon1 + 1],
                    showcoastlines=True, coastlinecolor="#555", showland=True, landcolor="#f2efe6",
                    showocean=True, oceancolor="#e8f1f8", resolution=50)
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=30, b=0), showlegend=False,
                      title=f"地上 10 m の風  {times[k].replace('T', ' ')}(日本時間)")
    st.plotly_chart(fig, width="stretch", key="wr_fig")
    st.caption("赤(rot>0)は反時計回りの渦(低気圧性)、青(rot<0)は時計回り(高気圧性)。"
               "発散を表示すると、吹き込む所(div<0)と吹き出す所(div>0)がわかる。"
               "陸の上は地形の影響で風が乱れるので、海上の方が渦の形がはっきりする。")


def tab_wind():
    st.info("天気予報の風は、身近な**ベクトル場**の例。地図上の各地点に「風のベクトル」が1つずつ対応している。")
    _section_vector()
    _section_model()
    _section_real()
