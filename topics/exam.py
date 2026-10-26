"""教員用:試験問題(記述式)の作成

演習と同じ生成関数から、章をまたいで問題の型と題数を選び、
数学IB の試験と同じ構成(本体 + problems.tex / answers.tex)の LaTeX 一式を作る。
"""
import io
import random
import zipfile

import streamlit as st

from topics import ex_ch1, ex_ch2, ex_ch3, ex_laplace
from topics.ex_core import CLASS_LINE, NENDO, PREAMBLE, Retry, build, exam_item_from_problem, md_to_tex, punct, rng_for

# 試験範囲として選べる章(章を増やしたらここに足す)
CHAPTERS = [
    ("ベクトル解析 第1章 ベクトル関数", ex_ch1.REGISTRY),
    ("ベクトル解析 第2章 スカラー場とベクトル場", ex_ch2.REGISTRY),
    ("ベクトル解析 第3章 線積分・面積分", ex_ch3.REGISTRY),
    ("ラプラス変換 第1章 定義と性質", ex_laplace.REGISTRY1),
    ("ラプラス変換 第2章 応用", ex_laplace.REGISTRY2),
]
GUIDED_NAMES = {**getattr(ex_ch1, "GUIDED_NAMES", {}), **ex_ch2.GUIDED_NAMES, **ex_ch3.GUIDED_NAMES,
                **ex_laplace.GUIDED_NAMES}


def catalog():
    """選べる問題の一覧:(表示名, 章, 型, 誘導問題にするか)"""
    out = []
    for ch, reg in CHAPTERS:
        for t in reg:
            out.append((f"{ch}|{t.name}", ch, t, False))
            if t.guided is not None:
                gname = GUIDED_NAMES.get(t.key, t.name)
                out.append((f"{ch}|【誘導 5点×2】{gname}", ch, t, True))
    return out


def make_items(selection, seed):
    """selection: [(表示名, 題数)] → ExamItem のリスト"""
    cat = {c[0]: c for c in catalog()}
    items = []
    k = 0
    for name, cnt in selection:
        _, ch, t, guided = cat[name]
        for j in range(cnt):
            rng = rng_for("exam", seed, k, t.key, guided)
            if guided:
                for _ in range(200):
                    try:
                        items.append(t.guided(rng))
                        break
                    except Retry:
                        continue
                else:
                    raise RuntimeError(f"{t.name}: 誘導問題を生成できませんでした")
            else:
                items.append(exam_item_from_problem(build(t, rng)))
            k += 1
    return items


# ------------------------------------------------------------
# LaTeX
# ------------------------------------------------------------
def page_sizes(n):
    """1ページの問題数(10題なら 3, 4, 3 と昨年度の期末試験と同じ)"""
    if n == 10:
        return [3, 4, 3]
    sizes = []
    while n > 0:
        k = 3 if n != 4 else 4
        k = min(k, n)
        sizes.append(k)
        n -= k
    return sizes


def _item_tex(it, answers):
    from topics.ex_fig import tikz
    fig = tikz(it.figure)
    afig = tikz(it.answer_figure) if answers else ""
    head = rf"\item {md_to_tex(it.intro)}"
    if len(it.parts) == 1 and not it.parts[0].statement:
        p = it.parts[0]
        body = head + rf" \hfill ({it.points}点)" + "\n" + (fig + "\n" if fig else "")
        if answers:
            body += _answer_box([("", p)]) + (afig + "\n" if afig else "")
        else:
            body += "\\vspace{4cm}\\vfill\n"
        return body
    body = head + rf" \hfill (各{it.parts[0].points}点)" + "\n" + (fig + "\n" if fig else "") + "  \\begin{enumerate}\n"
    for p in it.parts:
        body += rf"  \item {md_to_tex(p.statement)}" + "\n"
        if answers:
            body += _answer_box([("", p)], indent="  ")
        else:
            body += "  \\vspace{2.5cm}\\vfill\n"
    body += "  \\end{enumerate}\n" + (afig + "\n" if afig else "")
    return body


def _answer_box(parts, indent=""):
    out = indent + "  \\begin{itembox}[l]{解答・解説}\n"
    for _, p in parts:
        if p.answer:
            out += indent + rf"    \textbf{{答：${p.answer}$}}\\" + "\n\n"
        out += "\n\n".join(indent + "    " + md_to_tex(s) for s in p.solution) + "\n"
    out += indent + "  \\end{itembox}\n" + indent + "\\vfill\n"
    return out


def problems_file(items, answers=False):
    out, i = [], 0
    for size in page_sizes(len(items)):
        page = items[i:i + size]
        pts = sum(it.points for it in page)
        out += [_item_tex(it, answers) for it in page]
        out.append(rf"\vfill \hfill \underline{{小計：\hspace{{1.5cm}}点/{pts}点}}\clearpage")
        i += size
    kind = "解答・解説" if answers else "問題"
    return f"% {kind}(本体から \\input する)\n" + "\n".join(out) + "\n"


def main_file(title, inner, answer, seed, total):
    t = title + (" 解答" if answer else "")
    score = ("" if answer else
             "\\begin{flushright}\n \\setlength{\\fboxsep}{0.3cm}\\fbox{\\emph{総合得点}\\hspace{2.5cm} /\\emph{"
             + f"{total}点" + "}}\n\\end{flushright}\n")
    return (PREAMBLE + "\\makeatletter\n" + rf"\title{{{t}}}" + "\n" + rf"\author{{{CLASS_LINE}}}" + "\n"
            + rf"\date{{配点:各10点\quad (問題番号 {seed})}}" + "\n"
            + "\\begin{document}\n\\maketitle\n" + score + "\\begin{enumerate}\n"
            + rf"\input{{{inner}}}" + "\n\\end{enumerate}\n\\end{document}\n")


def exam_files(items, title, seed):
    total = sum(it.points for it in items)
    return {
        "試験問題.tex": main_file(title, "problems.tex", False, seed, total),
        "試験問題-Ans.tex": main_file(title, "answers.tex", True, seed, total),
        "problems.tex": problems_file(items, False),
        "answers.tex": problems_file(items, True),
    }


def exam_zip(items, title, seed):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, text in exam_files(items, title, seed).items():
            z.writestr(name, text)
    return buf.getvalue()


# ------------------------------------------------------------
# 画面
# ------------------------------------------------------------
def _password():
    try:
        return st.secrets["teacher_password"]
    except Exception:
        return None


def _new_seed():
    st.session_state["exam_seed"] = random.randint(10000, 99999)


def tab():
    pw = _password()
    if pw is None:
        st.warning("教員用パスワードが設定されていないため、このページは使えません。")
        st.markdown(
            "**設定のしかた**\n"
            "- Streamlit Community Cloud:アプリの Settings → Secrets に `teacher_password = \"(パスワード)\"` と書く。\n"
            "- 手元で動かす場合:`.streamlit/secrets.toml` に同じ1行を書く(このファイルは GitHub に上げない)。"
        )
        return
    if not st.session_state.get("exam_ok"):
        given = st.text_input("教員用パスワード", type="password", key="exam_pw")
        if given and given == pw:
            st.session_state["exam_ok"] = True
        elif given:
            st.error("パスワードが違います。")
            return
        else:
            return

    st.info("試験範囲に合わせて、章をまたいで問題の型と題数を選ぶ。1題10点(誘導問題は5点×2で1題)。"
            "数値は問題番号から決まるので、同じ番号なら同じ試験が再現できる。")
    title = st.text_input("試験のタイトル", f"{NENDO} 数学IV 後期中間試験", key="exam_title")

    cat = catalog()
    names = [c[0] for c in cat]
    st.markdown("**出題する問題の型**(上から選んだ順に並ぶ)")
    chosen = []
    for ch, reg in CHAPTERS:
        opts = [n for n in names if n.startswith(ch + "|")]
        with st.expander(ch, expanded=ch.startswith("ベクトル解析")):
            sel = st.multiselect("型を選ぶ", opts, format_func=lambda s: s.split("|", 1)[1], key=f"exam_sel_{ch}")
            cols = st.columns(3)
            for i, n in enumerate(sel):
                cnt = cols[i % 3].number_input(n.split("|", 1)[1], 1, 5, 1, key=f"exam_cnt_{n}")
                chosen.append((n, int(cnt)))
    total_q = sum(c for _, c in chosen)
    if total_q == 0:
        st.caption("型を選ぶと、ここに試験の構成が表示される。")
        return
    msg = f"合計 {total_q} 題({total_q * 10} 点)"
    (st.success if total_q == 10 else st.warning)(msg + ("" if total_q == 10 else "。100点にするには10題にする。"))

    if "exam_seed" not in st.session_state:
        st.session_state["exam_seed"] = 10000
    c1, c2 = st.columns(2)
    seed = int(c1.number_input("問題番号", 1, 99999, step=1, key="exam_seed"))
    c2.button("別の数値で作り直す", on_click=_new_seed, key="exam_new")
    shuffle = c2.checkbox("出題順をシャッフルする", key="exam_shuffle")

    items = make_items(chosen, seed)
    if shuffle:
        random.Random(seed).shuffle(items)

    st.subheader("プレビュー")
    for i, it in enumerate(items, 1):
        with st.container(border=True):
            st.markdown(punct(f"**{i}.** {it.intro}　({it.points}点)"))
            if it.figure is not None:
                from topics.ex_fig import plotly_fig
                st.plotly_chart(plotly_fig(it.figure, height=320), width="stretch", key=f"exam_fig_{seed}_{i}")
            if len(it.parts) > 1 or it.parts[0].statement:
                for j, p in enumerate(it.parts, 1):
                    st.markdown(punct(f"({j}) {p.statement}　({p.points}点)"))
            if st.checkbox("解答を表示", key=f"exam_show_{seed}_{i}"):
                for j, p in enumerate(it.parts, 1):
                    if len(it.parts) > 1:
                        st.markdown(f"({j})")
                    if p.answer:
                        st.markdown(f"答:${p.answer}$")
                    for s in p.solution:
                        st.markdown(punct(s))
                if it.answer_figure is not None:
                    from topics.ex_fig import plotly_fig
                    st.plotly_chart(plotly_fig(it.answer_figure, height=320), width="stretch", key=f"exam_afig_{seed}_{i}")
    st.download_button("LaTeX 一式をダウンロード(.zip)", exam_zip(items, title, seed),
                       file_name=f"試験問題_{seed}.zip", mime="application/zip", key="exam_dl")
    st.caption("zip の中身:試験問題.tex / 試験問題-Ans.tex(本体)、problems.tex / answers.tex(\\input される問題・解答)。"
               "uplatex + dvipdfmx でコンパイルする。")
