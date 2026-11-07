"""③ フーリエ解析 発展 5. ウェーブレットによる画像の類似度判定

2次元ハール変換(1次元と同じ「平均と差の半分」を行と列に):2×2 のブロック [[a, b], [c, d]] に対して
    A = (a+b+c+d)/4,  H = ((a+c)−(b+d))/4(左右の差),  V = ((a+b)−(c+d))/4(上下の差),  D = ((a+d)−(b+c))/4(斜め)
    a = A+H+V+D,  b = A−H+V−D,  c = A+H−V−D,  d = A−H−V+D,   a²+b²+c²+d² = 4(A²+H²+V²+D²)
特徴量:各レベル・各方向の係数の2乗平均の平方根(RMS)。比較はコサイン類似度。
"""
import io

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import ndimage

from common import C1, C2, C3, C4, GRAY, layout, show
from topics.fourier_kernels import page_refs

N = 256                 # 画像の大きさ(N×N)
LMAX = 5                # 特徴量に使うレベル数
BG = 0.85               # 見本の背景の明るさ
DIRS = ("左右の差 H", "上下の差 V", "斜め D")


# ------------------------------------------------------------
# 見本の図形画像(ボトル形の図形。ラベルの縞の向きだけが違う)
# ------------------------------------------------------------
@st.cache_data
def sample_bottle(pattern):
    """pattern = 'v'(縦縞)または 'h'(横縞)。縞の太さと明暗は同じなので明るさの分布(ヒストグラム)は同じ"""
    y, x = np.mgrid[0:N, 0:N]
    cx = N / 2
    img = np.full((N, N), BG)
    body = (np.abs(x - cx) < 44) & (y > 70) & (y < 240)
    shoulder = (y >= 40) & (y <= 70) & (np.abs(x - cx) < 16 + 28 * (y - 40) / 30)
    neck = (np.abs(x - cx) < 16) & (y >= 22) & (y < 40)
    cap = (np.abs(x - cx) < 19) & (y >= 8) & (y < 22)
    img[body | shoulder | neck] = 0.55
    img[cap] = 0.25
    x0, y0 = int(cx) - 40, 112                 # ラベルは 80×80 の正方形(縞は幅 8 画素で 10 本)
    label = (x >= x0) & (x < x0 + 80) & (y >= y0) & (y < y0 + 80)
    stripe = ((x - x0) // 8) % 2 == 0 if pattern == "v" else ((y - y0) // 8) % 2 == 0
    img[label] = np.where(stripe[label], 0.15, 0.95)
    return img


def transform(img, dx=0, dy=0, rot=0.0, scale=1.0, bright=0.0, noise=0.0, seed=0):
    """平行移動・回転・拡大縮小・明るさ・ノイズを加えた画像(見本の A' を作る)"""
    out = ndimage.rotate(img, rot, reshape=False, order=1, mode="nearest") if rot else img.copy()
    if scale != 1.0:
        z = ndimage.zoom(out, scale, order=1)
        h = z.shape[0]
        if h >= N:
            o = (h - N) // 2
            out = z[o:o + N, o:o + N]
        else:
            o = (N - h) // 2
            out = np.full((N, N), BG)
            out[o:o + h, o:o + h] = z
    if dx or dy:
        out = ndimage.shift(out, (dy, dx), order=1, mode="nearest")
    out = out + bright + (np.random.default_rng(seed).normal(0, noise, out.shape) if noise > 0 else 0)
    return np.clip(out, 0, 1)


def load_upload(data):
    """アップロードされた画像(バイト列)→ N×N×3 の配列(値は 0〜1)。縦横比を保って白で余白をつける"""
    from PIL import Image, ImageOps
    im = Image.open(io.BytesIO(data))
    im = ImageOps.exif_transpose(im).convert("RGB")
    im = ImageOps.pad(im, (N, N), color=(255, 255, 255))
    return np.asarray(im, dtype=float) / 255


def to_gray(img):
    return img if img.ndim == 2 else img @ np.array([0.299, 0.587, 0.114])


# ------------------------------------------------------------
# 2次元ハール変換
# ------------------------------------------------------------
def haar2(x):
    """1段階(間引きあり):(A, H, V, D) はそれぞれ大きさが半分"""
    a, b, c, d = x[0::2, 0::2], x[0::2, 1::2], x[1::2, 0::2], x[1::2, 1::2]
    return (a + b + c + d) / 4, ((a + c) - (b + d)) / 4, ((a + b) - (c + d)) / 4, ((a + d) - (b + c)) / 4


def ihaar2(A, H, V, D):
    x = np.empty((2 * A.shape[0], 2 * A.shape[1]))
    x[0::2, 0::2], x[0::2, 1::2] = A + H + V + D, A - H + V - D
    x[1::2, 0::2], x[1::2, 1::2] = A + H - V - D, A - H - V + D
    return x


def uhaar2(x, s):
    """間引かない(シフト不変な)ハール変換の1段階:距離 s 離れた画素どうしで平均と差の半分をとる(周期的に延長)"""
    r = np.roll(x, -s, axis=1)
    lo, hi = (x + r) / 2, (x - r) / 2

    def col(z):
        c = np.roll(z, -s, axis=0)
        return (z + c) / 2, (z - c) / 2
    A, V = col(lo)
    H, D = col(hi)
    return A, H, V, D


def mosaic(x, L):
    """L 段階の分解を1枚に並べた図(左上が近似, 右上 H, 左下 V, 右下 D)。細部は絶対値を拡大して表示"""
    out = np.zeros_like(x)
    a = x
    n = x.shape[0]
    for _ in range(L):
        a, H, V, D = haar2(a)
        h = n // 2
        for blk, (r0, c0) in ((H, (0, h)), (V, (h, 0)), (D, (h, h))):
            m = np.abs(blk)
            out[r0:r0 + h, c0:c0 + h] = m / (m.max() + 1e-12)
        n = h
    out[:n, :n] = (a - a.min()) / (np.ptp(a) + 1e-12)
    return out


def features(img, L=LMAX, shift_inv=True, skip1=False, color=False):
    """各レベル・各方向の係数の RMS を並べた特徴ベクトルと, その名前"""
    g = to_gray(img)
    mu, sd = g.mean(), g.std() + 1e-12
    chans = [img[..., k] for k in range(3)] if (color and img.ndim == 3) else [g]
    names = (" R", " G", " B") if len(chans) == 3 else ("",)
    f, lab = [], []
    for ch, nm in zip(chans, names):
        z = (ch - mu) / sd                  # 明るさ・コントラストの違いを取り除く
        for j in range(L):
            if shift_inv:
                z, H, V, D = uhaar2(z, 2 ** j)
            else:
                z, H, V, D = haar2(z)
            if skip1 and j == 0:
                continue
            for blk, dn in zip((H, V, D), "HVD"):
                f.append(float(np.sqrt(np.mean(blk ** 2))))
                lab.append(f"L{j + 1}-{dn}{nm}")
    return np.array(f), lab


def cos_sim(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


def pixel_sim(a, b):
    """画素どうしの相関係数(画像をそのままベクトルとみたときのなす角)"""
    a, b = to_gray(a).ravel(), to_gray(b).ravel()
    return cos_sim(a - a.mean(), b - b.mean())


def hist_sim(a, b, bins=32):
    """明るさのヒストグラムどうしのコサイン類似度"""
    ha = np.histogram(to_gray(a), bins, (0, 1))[0].astype(float)
    hb = np.histogram(to_gray(b), bins, (0, 1))[0].astype(float)
    return cos_sim(ha, hb)


def heat(z, title, height=300):
    fig = go.Figure(go.Heatmap(z=z, colorscale="Gray", showscale=False, zmin=0, zmax=1))
    fig.update_yaxes(autorange="reversed", showticklabels=False)
    fig.update_xaxes(showticklabels=False)
    layout(fig, title=title, height=height, equal=True, legend=False)
    return fig


# ------------------------------------------------------------
# ページ
# ------------------------------------------------------------
def basis_patterns():
    pats = {"A(平均)": [[1, 1], [1, 1]], "H(左右の差)": [[1, -1], [1, -1]], "V(上下の差)": [[1, 1], [-1, -1]], "D(斜め)": [[1, -1], [-1, 1]]}
    cols = st.columns(4)
    for col, (nm, p) in zip(cols, pats.items()):
        z = (np.array(p, dtype=float) + 1) / 2
        fig = go.Figure(go.Heatmap(z=z, colorscale="Gray", showscale=False, zmin=0, zmax=1,
                                   text=[["+1" if v > 0 else "−1" for v in row] for row in p], texttemplate="%{text}",
                                   textfont=dict(size=18, color="red")))
        fig.update_yaxes(autorange="reversed", showticklabels=False)
        fig.update_xaxes(showticklabels=False)
        layout(fig, title=nm, height=200, equal=True, legend=False)
        with col:
            show(fig)


def get_images(mode):
    """3枚の画像 {名前: 配列} を返す(そろわないときは None)"""
    if mode == "見本の図形":
        A, B = sample_bottle("v"), sample_bottle("h")
        with st.expander("A′ の作り方(A を動かす・回す・明るくする・ノイズを加える)", expanded=True):
            c = st.columns(3)
            dx = c[0].slider("横に平行移動(画素)", -20, 20, 5, key="wi_dx")
            dy = c[0].slider("縦に平行移動(画素)", -20, 20, 3, key="wi_dy")
            rot = c[1].slider("回転(度)", -90, 90, 3, key="wi_rot")
            sc = c[1].slider("拡大・縮小(倍)", 0.7, 1.3, 1.0, 0.05, key="wi_sc")
            br = c[2].slider("明るさ", -0.2, 0.2, 0.1, 0.05, key="wi_br")
            nz = c[2].slider("ノイズの大きさ", 0.0, 0.2, 0.03, 0.01, key="wi_nz")
        return {"A": A, "A′": transform(A, dx, dy, rot, sc, br, nz), "B": B}
    st.caption("画像はこのページの計算に使うだけで, 保存はしない。写真は対象が大きく写るように切り抜き, "
               "背景は無地, 向き(縦向き)をそろえると判定しやすい。")
    c = st.columns(3)
    out = {}
    for col, nm, help_ in zip(c, ("A", "A′", "B"), ("基準の画像", "A と同じものを別に撮った画像", "A と比べたい別のもの")):
        up = col.file_uploader(f"画像 {nm}({help_})", type=["png", "jpg", "jpeg"], key=f"wi_up_{nm}")
        try:
            out[nm] = load_upload(up.getvalue()) if up is not None else None
        except Exception:
            out[nm] = None
            col.error("この画像は読み込めなかった。")
    if any(v is None for v in out.values()):
        st.info("3枚の画像(A, A′, B)をアップロードすると判定する。")
        return None
    return out


def tab_wavelet_image():
    st.info("2枚の画像が「同じもの」を写しているかを, ウェーブレットで判定する仕組みを, 実際に計算しながら見る。"
            "流れは **前処理 → 2次元ハール変換 → 特徴量(方向・細かさごとのエネルギー)→ コサイン類似度 → しきい値で判定**。"
            "画像の分類に使われる AI(畳み込みニューラルネットワーク)も, 内部では似たことをしている。")

    st.subheader("1. 2次元ハール変換")
    st.markdown("1次元の「隣どうしの平均と差の半分」(発展 4.)を, 画像の**横方向と縦方向**の両方に行う。"
                "画像を $2\\times2$ のブロックに分け, ブロック $\\begin{pmatrix}a&b\\\\c&d\\end{pmatrix}$ ごとに4つの数を作る:")
    st.latex(r"A=\frac{a+b+c+d}{4},\quad H=\frac{(a+c)-(b+d)}{4},\quad V=\frac{(a+b)-(c+d)}{4},\quad D=\frac{(a+d)-(b+c)}{4}")
    basis_patterns()
    st.markdown("- $A$ は平均(画像を半分に縮めた粗い近似), $H$ は**左右の明るさの差**(縦の線・縦縞に反応), "
                "$V$ は**上下の差**(横の線・横縞に反応), $D$ は斜めの差。\n"
                "- 4つの模様は互いに直交するので, もとに戻せる:$a=A+H+V+D$, $b=A-H+V-D$, $c=A+H-V-D$, $d=A-H-V+D$。\n"
                "- エネルギー(2乗和)も保たれる:$a^2+b^2+c^2+d^2=4\\left(A^2+H^2+V^2+D^2\\right)$(パーセバルの等式の2次元版)。\n"
                "- $A$ でできた半分の大きさの画像に同じ操作をくり返すと, **多重解像度分解**(レベル 1, 2, 3, …)になる。"
                "レベル $j$ の係数は, $2^j$ 画素くらいの大きさの模様に反応する。")

    st.subheader("2. 比べる3枚の画像")
    st.markdown("「同じ」かどうかのしきい値は2枚だけでは決められないので, **A**(基準), **A′**(A と同じものを別に撮ったもの), "
                "**B**(別のもの)の3枚を比べ, 「A と A′ は似ている, A と B は似ていない」と判定できるかを見る。")
    mode = st.radio("画像", ["見本の図形", "画像をアップロード"], horizontal=True, key="wi_mode")
    imgs = get_images(mode)
    if imgs is None:
        return
    cols = st.columns(3)
    for col, (nm, im) in zip(cols, imgs.items()):
        col.image(np.clip(im, 0, 1), caption=nm, width="stretch")
    if mode == "見本の図形":
        st.caption("A と B は形が同じで, ラベルの縞の向き(縦・横)だけが違う。縞の太さと明暗は同じなので, "
                   "**明るさの分布(ヒストグラム)はまったく同じ**になる。A′ は A をずらし・少し回し・明るくし・ノイズを加えたもの。")

    st.subheader("3. 多重解像度分解を見る")
    c = st.columns(2)
    which = c[0].radio("画像", list(imgs), horizontal=True, key="wi_which")
    lev = c[1].slider("レベル j(反応する模様の大きさ 約 2^j 画素)", 1, LMAX, 3, key="wi_lev")
    g = to_gray(imgs[which])
    z = (g - g.mean()) / (g.std() + 1e-12)
    for jj in range(lev):
        z, H, V, D = uhaar2(z, 2 ** jj)
    mx = max(np.abs(H).max(), np.abs(V).max(), np.abs(D).max()) + 1e-12
    cols = st.columns(4)
    panels = [((z - z.min()) / (np.ptp(z) + 1e-12), f"近似(レベル {lev})")] + [(np.abs(b) / mx, f"|{nm}|") for b, nm in zip((H, V, D), DIRS)]
    for col, (im, ttl) in zip(cols, panels):
        with col:
            show(heat(im, ttl, height=260))
    st.caption("間引かない変換で, レベル j の係数を元の画像と同じ大きさで表示した(細部は絶対値。3方向で明るさの基準をそろえている)。"
               "縦縞は H に, 横縞は V に強く出る。レベル 1 では縞の境目だけが光り, レベル 4 では縞全体が光る(縞の幅 8 画素 ≒ 2^{4−1})。")
    with st.expander("通常の(間引く)分解を1枚に並べた図"):
        L = st.slider("レベル数", 1, 4, 4, key="wi_L")
        show(heat(mosaic(g, L), f"{which} の {L} 段階の分解(左上:近似, 右上:H, 左下:V, 右下:D)", height=420))
        st.caption("通常の変換では, レベルが上がるごとに画像が半分の大きさになる(ブロックの区切りが固定)。"
                   "見本の縞は幅 8 画素でブロックの区切りとそろっているので, レベル 1〜3 の H には縞の内部がほとんど現れず, レベル 4 で強く現れる。"
                   "このように, 通常の変換は区切りと模様の位置関係に左右される。")

    st.subheader("4. 特徴量と類似度")
    st.markdown("位置が少しずれただけで係数の値そのものは大きく変わるので, 係数を直接比べるのではなく, "
                "**レベル $j$・方向(H, V, D)ごとの係数の大きさ**(2乗平均の平方根, つまりエネルギー)を並べたベクトルを特徴量にする。"
                f"レベル 1〜{LMAX} の3方向で {3 * LMAX} 個の数になる。2つの特徴ベクトル $\\boldsymbol u,\\boldsymbol v$ の近さは, "
                "ベクトルのなす角で測る(コサイン類似度。発展 7. の関数のなす角と同じ):")
    st.latex(r"\cos\theta=\frac{\boldsymbol u\cdot\boldsymbol v}{|\boldsymbol u|\,|\boldsymbol v|}\qquad(1\text{ に近いほど似ている})")
    st.markdown("前処理として, 画像の明るさの平均を 0, 標準偏差を 1 にそろえる(明るさ・コントラストの違いを消す)。")
    c = st.columns(4)
    shift_inv = c[0].radio("変換", ["間引かない(シフトに強い)", "通常(間引く)"], key="wi_kind").startswith("間引かない")
    skip1 = c[1].checkbox("レベル 1 を使わない(ノイズ対策)", key="wi_skip1")
    color = c[2].checkbox("色も使う(RGB ごと)", key="wi_color", disabled=(mode == "見本の図形"))
    tau = c[3].slider("しきい値 τ", 0.80, 0.99, 0.90, 0.01, key="wi_tau")
    fe = {nm: features(im, shift_inv=shift_inv, skip1=skip1, color=color) for nm, im in imgs.items()}
    labels = fe["A"][1]
    fig = go.Figure()
    for (nm, (f, _)), col in zip(fe.items(), (C1, C3, C2)):
        fig.add_trace(go.Bar(x=labels, y=f, name=nm, marker_color=col))
    layout(fig, title="特徴ベクトル(レベル・方向ごとの係数の大きさ)", xlab="レベル − 方向", height=330)
    fig.update_layout(barmode="group")
    show(fig)

    pairs = [("A", "A′"), ("A", "B"), ("A′", "B")]
    rows = []
    for p, q in pairs:
        s = cos_sim(fe[p][0], fe[q][0])
        rows.append({"組": f"{p} と {q}", "ウェーブレット特徴の類似度": round(s, 3),
                     "判定": "同じ" if s >= tau else "違う",
                     "(比較)画素どうしの相関": round(pixel_sim(imgs[p], imgs[q]), 3),
                     "(比較)ヒストグラムの類似度": round(hist_sim(imgs[p], imgs[q]), 3)})
    st.dataframe(rows, hide_index=True, width="stretch")
    ok = rows[0]["判定"] == "同じ" and rows[1]["判定"] == "違う" and rows[2]["判定"] == "違う"
    if ok:
        st.success(f"しきい値 τ = {tau:.2f} で, A と A′ は「同じ」, B は「違う」と正しく判定できた。")
    else:
        st.warning(f"しきい値 τ = {tau:.2f} では正しく判定できていない。しきい値や設定を変えるか, 何が原因かを下の図で調べてみよう。")
    st.markdown("**比較の2つの方法がうまくいかない理由**(見本の図形で確かめられる):\n"
                "- **画素どうしの相関**:画像をそのまま $N^2$ 次元のベクトルとみる方法。縞が半周期ずれるだけで明暗が逆になり, "
                "A と A′ の相関は大きく下がる(位置のずれに弱い)。\n"
                "- **ヒストグラム**:明るさの分布だけを見る方法。位置のずれには強いが, 縞の向きの違いが分からないので "
                "A と B を区別できない(見本では 1.000)。また明るさを変えた A′ とは似ていないと判断してしまう。\n"
                "- **ウェーブレット特徴**は「どの細かさで, どの向きの変化が, どれだけあるか」を測るので, 位置のずれに強く, 向きの違いは区別できる。")
    st.markdown("**どこで差がついたか**:特徴ごとの差の2乗 $(u_k-v_k)^2$。")
    fig = go.Figure()
    for (p, q), col in zip(pairs[:2], (C3, C2)):
        fig.add_trace(go.Bar(x=labels, y=(fe[p][0] - fe[q][0]) ** 2, name=f"{p} と {q}", marker_color=col))
    layout(fig, title="特徴ごとの差の2乗", xlab="レベル − 方向", height=300)
    fig.update_layout(barmode="group")
    show(fig)
    with st.expander("「間引かない」変換とは"):
        st.markdown("通常の2次元ハール変換は $2\\times2$ のブロックの区切りが固定されているので, 画像が1画素ずれると区切りが変わり, "
                    "係数の大きさまで変わってしまう(「通常(間引く)」に切り替えて, A′ の平行移動を動かしてみよう)。"
                    "**間引かない変換**では, すべての位置から始まるブロックで平均と差の半分を計算する"
                    "(レベル $j$ では $2^{j-1}$ 画素離れた画素どうし)。計算量は増えるが, 係数の大きさは平行移動で変わらなくなる。")

    st.subheader("5. 仕組みの限界と AI へのつながり")
    st.markdown("- **回転に弱い**:A′ の回転を 90° にすると, 縦縞が横縞になって B と「同じ」と判定される。"
                "H と V を区別していることの裏返しで, 回転しても変わらない特徴にするには工夫が要る。\n"
                "- **大きさ・背景・写り方**:大きさが大きく違うとレベルがずれる。実際の写真では背景や照明, 写る角度の影響が大きいので, "
                "対象を切り抜いて大きさと向きをそろえることが大切になる。\n"
                "- **しきい値**:どこまでを「同じ」とするかは, 同じもの・違うものの例をたくさん集めて決める必要がある。"
                "この「例から決める」部分を大規模に自動化したのが機械学習である。\n"
                "- **AI(畳み込みニューラルネットワーク, CNN)**:小さなフィルター(ここでの H, V, D の模様にあたる)を画像全体に当て, "
                "絶対値のような非線形な処理と縮小をくり返して特徴を作る。違いは, フィルターを**データから学習する**ことである。\n"
                "- **散乱変換**:マラ(S. Mallat)は, ウェーブレット変換 → 絶対値 → 平均 を何段も重ねた**散乱変換**を考え, "
                "平行移動に強く, 小さな変形にも安定な特徴が作れることを示した。ここで使った「係数の大きさの平均」は, その最も簡単な形にあたる。"
                "CNN がなぜうまくいくのかを, 調和解析の立場から説明しようとする研究につながっている。")

    with st.expander("練習問題"):
        st.markdown("**問1** ブロック $\\begin{pmatrix}6&2\\\\4&0\\end{pmatrix}$ の $A,H,V,D$ を求め, "
                    "$a^2+b^2+c^2+d^2=4(A^2+H^2+V^2+D^2)$ を確かめよ。")
        st.markdown("**問2** 左半分が明るく右半分が暗い画像(縦の境界線)を2次元ハール変換すると, $H,V,D$ のうちどれが大きくなるか。")
        st.markdown("**問3** 特徴ベクトル $\\boldsymbol u=(1,2,2)$, $\\boldsymbol v=(2,1,2)$ のコサイン類似度を求めよ。"
                    "しきい値 $\\tau=0.9$ で「同じ」と判定されるか。")
        if st.toggle("解答・解説を表示", key="pr_wvimg"):
            st.markdown("**答1** $A=\\frac{6+2+4+0}{4}=3$, $H=\\frac{(6+4)-(2+0)}{4}=2$, $V=\\frac{(6+2)-(4+0)}{4}=1$, $D=\\frac{(6+0)-(2+4)}{4}=0$。"
                        "$36+4+16+0=56=4(9+4+1+0)$。")
            st.markdown("**答2** $H$(左右の差)。境界線をまたぐブロックでは左の列が明るいので $(a+c)-(b+d)>0$ となり, 上下の差 $V$ と斜め $D$ は $0$ になる(上下の画素が同じ明るさのため)。")
            st.markdown("**答3** $\\cos\\theta=\\dfrac{2+2+4}{3\\cdot3}=\\dfrac{8}{9}\\approx0.889<0.9$ なので「違う」と判定される。")

    page_refs([("mallat1989", "多重解像度解析と2次元の分解(画像の近似と縦・横・斜めの細部)。"),
               ("mallat2009", "ウェーブレットと画像処理の教科書。平行移動に強い(間引かない)ウェーブレット変換も扱う。"),
               ("changkuo1993", "ウェーブレット係数のエネルギーを特徴量にしたテクスチャ(模様)の分類。"),
               ("mallat2012", "散乱変換:平行移動に強く, 変形に安定な特徴。"),
               ("brunamallat2013", "散乱変換による画像分類(手書き数字・テクスチャ)と CNN との比較。")])
