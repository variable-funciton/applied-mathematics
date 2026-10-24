import streamlit as st

st.set_page_config(page_title="応用数学ラボ", page_icon="∮", layout="wide")

from topics import (complex2 as c2, complex_fn as cf, fourier, fourier2 as f2, laplace,  # noqa: E402
                    laplace2 as l2, other, pde, pde2 as p2, vec_functions as vf, vec_integrals as vi, vector)

# 分野 → 章 → [(節の名前, 描画する関数)] (教科書の目次の順)
CURRICULUM = {
    "① ベクトル解析": {
        "第1章 ベクトル関数": [
            ("1. 内積・外積", vf.tab_products),
            ("2. ベクトル関数", vf.tab_vector_function),
            ("3. 曲線", vf.tab_curve),
            ("4. 曲面(曲面積・接平面)", vf.tab_surface),
        ],
        "第2章 スカラー場とベクトル場": [
            ("1. 勾配", vector.tab_gradient),
            ("2. 発散と回転", vector.tab_div_rot),
        ],
        "第3章 線積分・面積分": [
            ("1. スカラー場の線積分", vi.tab_scalar_line),
            ("2. ベクトル場の線積分", vi.tab_vector_line),
            ("3. グリーンの定理", vector.tab_green),
            ("4. 面積分", vi.tab_surface_integral),
            ("5. ガウスの発散定理", vi.tab_gauss),
            ("6. ストークスの定理", vi.tab_stokes),
            ("7. 勾配の意味", vi.tab_gradient_meaning),
            ("8. 楕円の内部での光の反射", vi.tab_ellipse),
        ],
    },
    "② ラプラス変換": {
        "第1章 ラプラス変換の定義と性質": [
            ("1. ラプラス変換の定義", laplace.tab_definition),
            ("2. 相似性と移動法則", l2.tab_shift),
            ("3. 微分法則と積分法則", l2.tab_diff_int),
            ("4. 逆ラプラス変換", laplace.tab_inverse),
        ],
        "第2章 ラプラス変換の応用": [
            ("1. 微分方程式への応用", laplace.tab_ode),
            ("2. 合成積", l2.tab_convolution),
            ("3. 線形システムの伝達関数とデルタ関数", l2.tab_transfer),
            ("4. ラプラス変換と演算子法", l2.tab_operator),
        ],
    },
    "③ フーリエ解析": {
        "第1章 フーリエ級数": [
            ("1. 一般周期のフーリエ級数", fourier.tab_series),
            ("2. 複素フーリエ級数", f2.tab_complex_series),
        ],
        "第2章 フーリエ変換": [
            ("1. フーリエ変換と積分定理", f2.tab_integral_theorem),
            ("2. フーリエ変換の性質と公式", f2.tab_properties),
            ("2+. 代表的な変換対と不確定性", fourier.tab_transform),
            ("3. スペクトル", f2.tab_spectrum),
            ("4. 離散フーリエ変換と複素フーリエ級数", f2.tab_dft_series),
            ("4+. FFT とフィルタ", fourier.tab_dft),
        ],
    },
    "④ 複素関数": {
        "第1章 正則関数": [
            ("1. 複素数と極形式", c2.tab_polar),
            ("2. 絶対値と偏角", c2.tab_abs_arg),
            ("3. 複素関数(写像)", cf.tab_mapping),
            ("3+. 定義域の色付け", cf.tab_domain_coloring),
            ("4. 正則関数", c2.tab_holomorphic),
            ("5. コーシー・リーマンの関係式", c2.tab_cauchy_riemann),
            ("6. 逆関数", c2.tab_inverse_fn),
        ],
        "第2章 積分": [
            ("1. 複素積分", c2.tab_complex_integral),
            ("2. コーシーの積分定理", c2.tab_cauchy_theorem),
            ("3. コーシーの積分表示", c2.tab_cauchy_formula),
            ("4. 関数の展開", c2.tab_expansion),
            ("5. 留数と留数定理", cf.tab_residue),
            ("5+. 実積分への応用", cf.tab_real_integral),
            ("6. 代数学の基本定理", c2.tab_fundamental),
        ],
    },
    "⑤ 偏微分方程式": {
        "1次元波動方程式・1次元熱伝導方程式": [
            ("1. フーリエ級数と熱伝導方程式", pde.tab_heat),
            ("1. フーリエ級数と波動方程式", pde.tab_wave),
            ("1+. 差分法と安定性", pde.tab_ftcs),
            ("2. フーリエ変換と偏微分方程式", p2.tab_fourier_pde),
            ("発展. ラプラス方程式", pde.tab_laplace),
        ],
    },
    "⑥ その他": {
        "その他": [
            ("デルタ関数のモデル関数", other.tab_delta),
            ("フーリエ変換とフーリエ級数の関係", other.tab_series_vs_transform),
            ("サンプリング定理", other.tab_sampling),
        ],
    },
}

with st.sidebar:
    st.title("応用数学ラボ")
    field = st.radio("分野", ["ホーム"] + list(CURRICULUM), key="nav_field")
    if field != "ホーム":
        chapters = list(CURRICULUM[field])
        chapter = st.radio("章", chapters, key=f"nav_ch_{field}") if len(chapters) > 1 else chapters[0]
    st.divider()
    st.caption("スライダーを動かして、式とグラフの変化を観察しよう。各節の最後に練習問題があります。"
               "「+」のついた節は教科書の節を補う発展的な内容です。")

if field == "ホーム":
    st.title("応用数学ラボ")
    st.caption("ベクトル解析・ラプラス変換・フーリエ解析・複素関数・偏微分方程式を、動かして学ぶ")
    st.info("左のメニューで分野と章を選び、上に並ぶ節を選んでください。"
            "スマートフォンでは左上の「>」でメニューが開きます。")
    cols = st.columns(3)
    for i, (fname, chs) in enumerate(CURRICULUM.items()):
        with cols[i % 3]:
            st.subheader(fname)
            for ch, secs in chs.items():
                st.markdown(f"**{ch}**")
                st.markdown("\n".join(f"- {s}" for s, _ in secs))
    st.subheader("分野のつながり")
    st.markdown(
        """
- **フーリエ級数・変換 → 偏微分方程式**:熱・波動方程式は、初期条件をフーリエ級数(有限区間)やフーリエ変換(無限区間)で分解して解く。
- **ラプラス変換 ↔ 複素関数**:逆変換の部分分数は留数の計算そのもの。極の位置が解の振る舞いを決める。
- **ベクトル解析 ↔ 複素関数**:コーシーの積分定理は、グリーンの定理とコーシー・リーマンの関係式から導ける。
- **デルタ関数**:ラプラス変換(インパルス応答)、フーリエ変換(白色スペクトル)、熱方程式(熱核)に共通して現れる。
"""
    )
else:
    sections = CURRICULUM[field][chapter]
    st.title(field)
    st.caption(chapter)
    labels = [s for s, _ in sections]
    sec = st.radio("節", labels, horizontal=True, key=f"nav_sec_{field}_{chapter}", label_visibility="collapsed")
    st.header(sec)
    dict(sections)[sec]()
