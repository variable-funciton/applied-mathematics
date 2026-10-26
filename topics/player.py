"""「シミュレーション開始」ボタンで動かす再生部品(熱・波動・ばねの振動で共用)

- st.fragment の中で動くので、再生中も再生後も、この部分だけが再実行される(ページ全体は再計算しない)。
- 計算は呼び出し側で先に済ませておき(st.cache_data)、ここでは描画だけをくり返す。
- 1コマの描画にかかった時間を測り、待ち時間から差し引くので、速さが一定になる。
- 再生後は「時刻」のスライダーで好きな瞬間に戻して確かめられる。
"""
import time

import streamlit as st

SLEEP = time.sleep          # テストでは差しかえる
CLOCK = time.perf_counter
SPEEDS = {"ゆっくり": 0.14, "ふつう": 0.07, "はやい": 0.03}
MAX_LIVE = 160              # 再生で描くコマ数の上限(多いとブラウザの描画が追いつかない)


def _fragment(f):
    frag = getattr(st, "fragment", None)
    return frag(f) if callable(frag) else f


@_fragment
def player(key, labels, draw, height=None):
    """labels: 各コマの時刻の表示(例 "t = 0.25")、draw(i): i コマ目の plotly の図を返す関数"""
    n = len(labels)
    pos_key = f"{key}_pos"
    if pos_key not in st.session_state or st.session_state[pos_key] not in labels:
        st.session_state[pos_key] = labels[0]
    ph = st.empty()
    c1, c2 = st.columns([1, 1.4])
    start = c1.button("▶ シミュレーション開始", key=f"{key}_start", type="primary")
    speed = c2.select_slider("再生速度", list(SPEEDS), value="ふつう", key=f"{key}_speed")

    if start:
        skip = max(1, round(n / MAX_LIVE))
        idx = list(range(0, n, skip))
        if idx[-1] != n - 1:
            idx.append(n - 1)
        wait = SPEEDS[speed]
        for i in idx:
            t0 = CLOCK()
            ph.plotly_chart(draw(i), width="stretch", key=f"{key}_live_{i}")
            SLEEP(max(0.0, wait - (CLOCK() - t0)))
        st.session_state[pos_key] = labels[-1]
    lab = st.select_slider("時刻(再生のあとで動かすと、その瞬間を確かめられる)", labels, key=pos_key)
    ph.plotly_chart(draw(labels.index(lab)), width="stretch", key=f"{key}_still")
