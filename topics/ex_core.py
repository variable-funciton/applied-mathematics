"""演習(自習用の類題)の共通部品:問題の型、4択の表示、ヒント、解答、LaTeX 出力

問題を増やすときは、各章のファイル(ex_ch1.py など)に「生成関数」を書き、REGISTRY に登録する。
生成関数は random.Random を受け取り、Problem を返す。条件に合わない数値が出たら Retry を投げる。
"""
import random
import re
from dataclasses import dataclass, field
from fractions import Fraction
from math import gcd, isqrt

import streamlit as st

LABELS = ["(1)", "(2)", "(3)", "(4)"]


class Retry(Exception):
    """数値がよくない(答えが自明・選択肢が重なる等)ので作り直す"""


@dataclass
class Choice:
    tex: str                 # 選択肢(LaTeX、$ なし)
    correct: bool = False
    why: str = ""            # 誤答を選んだときの指摘
    value: object = None     # 自動テスト用の数値


@dataclass
class Problem:
    title: str
    level: str               # "基本" / "標準"
    statement: str           # 問題文(Markdown。数式は $..$ / $$..$$。** は使わない)
    choices: list
    hints: list = field(default_factory=list)
    solution: list = field(default_factory=list)   # 解答の途中式(段落ごと)
    meta: dict = field(default_factory=dict)       # 自動テスト用の元データ
    # 試験(記述式)で使うとき:問題文・答えを4択用から変えたい場合だけ指定する
    exam_statement: str = None   # None なら statement をそのまま使う
    exam_answer: str = None      # None なら正解の選択肢、"" なら「答」の行を出さない(証明問題など)
    exam_solution: list = None   # None なら solution をそのまま使う
    figure: dict = None          # 図の仕様(ex_fig.py)。問題文の下と解答に表示する
    answer_figure: dict = None   # 解答だけに入れる図(None なら解答には figure を使わない)


@dataclass
class Part:
    """試験の小問(誘導問題の (1), (2) など)"""
    statement: str
    answer: str                  # LaTeX($ なし)。"" なら「答」の行を出さない
    solution: list
    points: int = 5


@dataclass
class ExamItem:
    """試験の大問1つ(単問なら parts は1つ)"""
    title: str
    intro: str                   # 大問の問題文(小問がある場合は共通の設定)
    parts: list
    points: int = 10
    meta: dict = field(default_factory=dict)       # 自動テスト用
    figure: dict = None
    answer_figure: dict = None


def exam_item_from_problem(p, points=10):
    """4択の問題を記述式の単問にする"""
    ans = p.exam_answer if p.exam_answer is not None else [c.tex for c in p.choices if c.correct][0]
    return ExamItem(p.title, p.exam_statement or p.statement,
                    [Part("", ans, p.exam_solution if p.exam_solution is not None else p.solution, points)], points,
                    meta=p.meta, figure=p.figure, answer_figure=p.answer_figure)


@dataclass
class ProblemType:
    key: str
    name: str
    level: str
    gen: object              # gen(rng) -> Problem
    guided: object = None    # 試験用の誘導問題(5点×2):guided(rng) -> ExamItem。ない型は None


def build(ptype, rng, tries=200):
    """生成関数を、条件を満たすまで繰り返し呼ぶ。選択肢を並べかえて返す"""
    for _ in range(tries):
        try:
            p = ptype.gen(rng)
        except Retry:
            continue
        texs = [c.tex for c in p.choices]
        if len(p.choices) != 4 or len(set(texs)) != 4 or sum(c.correct for c in p.choices) != 1:
            continue
        rng.shuffle(p.choices)
        return p
    raise RuntimeError(f"{ptype.name}: 問題を生成できませんでした")


def rng_for(*parts):
    return random.Random("-".join(str(x) for x in parts))


# ------------------------------------------------------------
# 数式の文字列を作る道具
# ------------------------------------------------------------
def sqrt_parts(n):
    """n = k² m(m は平方因子なし)の (k, m)"""
    n = int(n)
    k, m, p = 1, n, 2
    while p * p <= m:
        while m % (p * p) == 0:
            m //= p * p
            k *= p
        p += 1
    return k, m


def tex_root(n):
    """√n を簡単にした形"""
    if n == 0:
        return "0"
    k, m = sqrt_parts(n)
    if m == 1:
        return str(k)
    return (str(k) if k != 1 else "") + rf"\sqrt{{{m}}}"


def tex_fr(x):
    """分数(Fraction / int)"""
    x = Fraction(x)
    if x.denominator == 1:
        return str(x.numerator)
    s = "-" if x < 0 else ""
    return rf"{s}\frac{{{abs(x.numerator)}}}{{{x.denominator}}}"


def tex_frac_root(p, q, n):
    """p√n / q を約分・簡単化した形(q > 0)"""
    if p == 0:
        return "0"
    k, m = sqrt_parts(n)
    p *= k
    g = gcd(abs(p), q)
    p, q = p // g, q // g
    s = "-" if p < 0 else ""
    p = abs(p)
    if m == 1:
        return tex_fr(Fraction(-p if s else p, q))
    num = (str(p) if p != 1 else "") + rf"\sqrt{{{m}}}"
    return s + (num if q == 1 else rf"\frac{{{num}}}{{{q}}}")


def vtex(v):
    return r"\left(" + r",\ ".join(tex_fr(c) for c in v) + r"\right)"


def is_square(n):
    return n >= 0 and isqrt(n) ** 2 == n


def lin_tex(terms, const=0):
    """[(係数, 'x'), ...] + 定数 → 3x - 2y + 5"""
    out = ""
    for c, v in terms:
        if c == 0:
            continue
        s = "-" if c < 0 else ("+" if out else "")
        m = abs(c)
        out += s + ("" if m == 1 else tex_fr(m)) + v
    if const != 0 or not out:
        out += ("-" if const < 0 else ("+" if out else "")) + tex_fr(abs(const))
    return out


def poly_tex(coefs, var="t"):
    """[c0, c1, c2, c3] → 2t^3 - t + 1"""
    terms = []
    for k in range(len(coefs) - 1, 0, -1):
        terms.append((coefs[k], var if k == 1 else f"{var}^{{{k}}}"))
    return lin_tex(terms, coefs[0])


# ------------------------------------------------------------
# 画面表示
# ------------------------------------------------------------
def _inc(key, reset=None):
    st.session_state[key] = st.session_state.get(key, 0) + 1
    for r in reset or []:
        st.session_state.pop(r, None)


def _new_seed(key):
    st.session_state[key] = random.randint(1, 9999)


def make_set(registry, names, n, seed):
    """出題セット:型を順にくり返して n 問。各問の数値はセット番号と問番号から決まる"""
    types = [t for t in registry if t.name in names]
    out = []
    for i in range(n):
        t = types[i % len(types)]
        out.append((i, t))
    return out


def render(chapter, registry, title):
    """章の演習ページ"""
    ck = f"ex_{chapter}"
    st.info("自習用の演習です。数値を変えた類題が何度でも作れます。"
            "答えを選ぶと正誤と、誤答なら間違えやすい点が表示されます。ヒントは1段ずつ開けられます。")
    c1, c2 = st.columns([1, 2])
    level = c1.radio("難易度", ["基本", "標準", "基本+標準"], key=f"{ck}_lv")
    avail = [t for t in registry if level == "基本+標準" or t.level == level]
    names = c2.multiselect("問題の型(複数選べる)", [t.name for t in avail], default=[t.name for t in avail],
                           key=f"{ck}_types_{level}")
    if not names:
        st.warning("問題の型を1つ以上選んでください。")
        return
    d1, d2, d3 = st.columns([1, 1, 1])
    n = d1.slider("問題数", 1, 12, min(len(names), 6), key=f"{ck}_n")
    seed_key = f"{ck}_seed"
    if seed_key not in st.session_state:
        st.session_state[seed_key] = 1
    seed = int(d2.number_input("セット番号", 1, 9999, step=1, key=seed_key))
    d3.button("新しいセットを作る", on_click=_new_seed, args=(seed_key,), key=f"{ck}_new")
    d3.caption("同じセット番号なら同じ問題が出る(質問するときに番号を伝えよう)。")

    probs = []
    for i, t in make_set(registry, names, n, seed):
        vkey = f"{ck}_{seed}_{i}_v"
        v = st.session_state.get(vkey, 0)
        p = build(t, rng_for(seed, i, v, t.key))
        probs.append((i, t, v, vkey, p))

    st.divider()
    for i, t, v, vkey, p in probs:
        base = f"{ck}_{seed}_{i}_{v}"
        st.markdown(f"#### 問{i + 1}　{p.title}　〔{p.level}〕" + (f"(類題 {v})" if v else ""))
        st.markdown(punct(p.statement))
        if p.figure is not None:
            from topics.ex_fig import plotly_fig
            st.plotly_chart(plotly_fig(p.figure, height=360), width="stretch", key=f"{base}_fig")
        for lab, c in zip(LABELS, p.choices):
            st.markdown(f"{lab}　${c.tex}$")
        ans = st.radio("答え", LABELS, index=None, horizontal=True, key=f"{base}_ans")
        if ans is not None:
            c = p.choices[LABELS.index(ans)]
            if c.correct:
                st.success("正解! ")
            else:
                st.error(punct("不正解. " + (c.why or "もう一度計算を確かめよう。")))
        hkey = f"{base}_h"
        k = st.session_state.get(hkey, 0)
        for j, h in enumerate(p.hints[:k], 1):
            st.markdown(punct(f"ヒント{j}:{h}"))
        b1, b2, b3 = st.columns(3)
        if k < len(p.hints):
            b1.button(f"ヒントを見る({k + 1}/{len(p.hints)})", on_click=_inc, args=(hkey,), key=f"{base}_hb")
        show = b2.checkbox("解答を表示", key=f"{base}_sol")
        b3.button("同じ型でもう一問", on_click=_inc, args=(vkey,), key=f"{base}_more")
        if show:
            correct = LABELS[[c.correct for c in p.choices].index(True)]
            with st.container(border=True):
                st.markdown(f"正解:{correct}")
                for s in p.solution:
                    st.markdown(punct(s))
                if p.answer_figure is not None:
                    from topics.ex_fig import plotly_fig
                    st.plotly_chart(plotly_fig(p.answer_figure, height=340), width="stretch", key=f"{base}_afig")
        st.divider()

    st.subheader("紙で解く(LaTeX でダウンロード)")
    st.caption("いま表示している問題(類題に切りかえた問題を含む)を、数学IB の試験と同じ構成の LaTeX 一式"
               "(本体2つ + problems.tex + answers.tex)にして zip でダウンロードする。uplatex + dvipdfmx でコンパイルする。")
    plist = [p for *_, p in probs]
    e1, e2 = st.columns(2)
    with e1:
        st.markdown("**LaTeX ファイル**(先生・自分で編集したい人)")
        st.download_button("LaTeX 一式をダウンロード(.zip)", latex_zip(plist, title, seed),
                           file_name=f"数学IV_{title.replace(' ', '_')}_演習_{seed}.zip", mime="application/zip",
                           key=f"{ck}_dl")
    with e2:
        st.markdown("**すぐに取り組みたい学生はこちらでダウンロード**")
        student_pdf_ui(plist, title, seed, ck)


# ------------------------------------------------------------
# LaTeX 出力(数学IB 前期期末試験と同じ構成:本体 + problems.tex / answers.tex)
# ------------------------------------------------------------
NENDO = "令和8年度"
CLASS_LINE = r"4年M, E, C, T工学科 \hspace{10truemm}番\ 氏名\hspace{50truemm}"

PREAMBLE = r"""\documentclass[a4paper,12pt,dvipdfmx,fleqn,uplatex]{jsarticle}
\newcommand{\f}[1]{\framebox[1cm]{\textgt{\small #1}}}
\def\labelenumi{\framebox[0.8cm]{\bf\theenumi}}
\def\theenumii{\arabic{enumii}}
\usepackage[margin=0.5cm,dvipdfmx]{geometry}
\usepackage{amsmath, amssymb}
\usepackage{arydshln}
\usepackage{multicol,enumerate,multienum}
\usepackage{setspace}
\usepackage{cancel}
\usepackage{ascmac}
\usepackage{tikz}
\newcommand{\Set}[1]{ \left\{ #1 \right\} }
\pagestyle{empty}
"""


def punct(s):
    """句読点を「, 」「. 」(小文字のコンマ・ピリオドと半角スペース)にそろえる(数式の外だけ)"""
    if not s:
        return s
    parts = re.split(r"(\$\$.+?\$\$|\\\[.+?\\\]|\$[^$]+\$)", s, flags=re.S)
    for k in range(0, len(parts), 2):
        parts[k] = parts[k].replace("、", ", ").replace("。", ". ").replace("，", ", ").replace("．", ". ")
    return "".join(parts).rstrip()


def md_to_tex(s):
    """問題文・解答の Markdown を LaTeX に(数式はそのまま、$$..$$ は \\[..\\])"""
    s = re.sub(r"\$\$(.+?)\$\$", lambda m: r"\[" + m.group(1) + r"\]", s, flags=re.S)
    s = s.replace("**", "")
    # 句読点を試験問題の書式(「, 」「. 」)にそろえる(数式の外だけ)
    parts = re.split(r"(\$\$.+?\$\$|\\\[.+?\\\]|\$[^$]+\$)", s, flags=re.S)
    for k in range(0, len(parts), 2):
        parts[k] = parts[k].replace("、", ", ").replace("。", ". ").replace("，", ", ").replace("．", ". ")
    return "".join(parts).rstrip()


def choices_tex(choices):
    """4択の選択肢。短ければ1行に4つ、長ければ1行に2つ"""
    cells = [rf"{lab}\ ${c.tex}$" for lab, c in zip(LABELS, choices)]
    if max(len(c.tex) for c in choices) <= 28:
        return r"\noindent" + r"\hfill ".join(cells) + r"\hfill\null"
    rows = []
    for k in (0, 2):
        rows.append(r"\noindent\makebox[0.48\linewidth][l]{" + cells[k] + r"}\makebox[0.48\linewidth][l]{"
                    + cells[k + 1] + "}")
    return r"\\[2mm]".join(rows)


def main_tex(title, seed, inner, answer=False):
    head = rf"{NENDO} 数学IV {title} 演習" + (" 解答" if answer else "")
    return (PREAMBLE + "\\makeatletter\n"
            + rf"\title{{{head}}}" + "\n"
            + rf"\author{{{CLASS_LINE}}}" + "\n"
            + rf"\date{{セット番号 {seed}}}" + "\n"
            + "\\begin{document}\n\\maketitle\n\\begin{enumerate}\n"
            + rf"\input{{{inner}}}" + "\n\\end{enumerate}\n\\end{document}\n")


def _fig(p):
    from topics.ex_fig import tikz
    t = tikz(getattr(p, "figure", None))
    return t + "\n" if t else ""


def _afig(p):
    from topics.ex_fig import tikz
    return tikz(getattr(p, "answer_figure", None))


def problems_tex(problems, per_page=4):
    out = []
    for k, p in enumerate(problems, 1):
        out.append(rf"\item {md_to_tex(p.statement)}" + "\n\n" + _fig(p) + choices_tex(p.choices) + "\n\\vfill")
        if k % per_page == 0 and k < len(problems):
            out.append(r"\clearpage")
    return "% 問題(本体から \\input する)\n" + "\n\n".join(out) + "\n"


def answers_tex(problems, per_page=2):
    out = []
    for k, p in enumerate(problems, 1):
        j = [c.correct for c in p.choices].index(True)
        sol = "\n\n".join("    " + md_to_tex(s) for s in p.solution)
        afig = _afig(p)
        out.append(rf"\item {md_to_tex(p.statement)}" + "\n\n" + _fig(p) + choices_tex(p.choices) + "\n\n"
                   + "  \\begin{itembox}[l]{解答・解説}\n"
                   + rf"    \textbf{{答：{LABELS[j]}\ ${p.choices[j].tex}$}}\\" + "\n\n"
                   + sol + "\n" + (afig + "\n" if afig else "") + "  \\end{itembox}\n\\vfill")
        if k % per_page == 0 and k < len(problems):
            out.append(r"\clearpage")
    return "% 解答・解説(本体から \\input する)\n" + "\n\n".join(out) + "\n"


def latex_files(problems, title, seed):
    """{ファイル名: 内容}。本体の2ファイルは problems.tex / answers.tex を \\input する"""
    return {
        "数学IV演習.tex": main_tex(title, seed, "problems.tex"),
        "数学IV演習-Ans.tex": main_tex(title, seed, "answers.tex", answer=True),
        "problems.tex": problems_tex(problems),
        "answers.tex": answers_tex(problems),
    }


def latex_zip(problems, title, seed):
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, text in latex_files(problems, title, seed).items():
            z.writestr(name, text)
    return buf.getvalue()


# ------------------------------------------------------------
# 学生用 PDF(問題 → 解答・解説 の順に1つの PDF)
# サーバーに uplatex と dvipdfmx が必要(Streamlit Community Cloud では packages.txt で入れる)
# ------------------------------------------------------------
STUDENT_PREAMBLE = r"""\documentclass[a4paper,11pt,dvipdfmx,fleqn,uplatex]{jsarticle}
\def\labelenumi{\framebox[0.8cm]{\bf\theenumi}}
\def\theenumii{\arabic{enumii}}
\usepackage[margin=1.5cm]{geometry}
\usepackage{amsmath, amssymb}
\usepackage{ascmac}
\usepackage{tikz}
\pagestyle{empty}
"""


def student_tex(problems, title, seed):
    return (STUDENT_PREAMBLE
            + rf"\title{{数学IV　{title}　演習}}" + "\n"
            + rf"\author{{4年\hspace{{1.5cm}}科\hspace{{1.5cm}}番\ 氏名\hspace{{4cm}}}}" + "\n"
            + rf"\date{{セット番号 {seed}(同じ番号で Web の演習と同じ問題が出ます)}}" + "\n"
            + "\\begin{document}\n\\maketitle\n\\begin{enumerate}\n" + problems_tex(problems)
            + "\\end{enumerate}\n\\clearpage\n\\begin{center}{\\Large 解答・解説}\\end{center}\n"
            + "\\begin{enumerate}\n" + answers_tex(problems) + "\\end{enumerate}\n\\end{document}\n")


def latex_available():
    import shutil
    return shutil.which("uplatex") is not None and shutil.which("dvipdfmx") is not None


@st.cache_data(show_spinner=False, max_entries=200)
def compile_pdf(tex):
    """uplatex → dvipdfmx。(PDF のバイト列, エラーの抜粋) を返す"""
    import os
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "main.tex"), "w", encoding="utf-8") as f:
            f.write(tex)
        try:
            r = subprocess.run(["uplatex", "-interaction=nonstopmode", "-halt-on-error", "-kanji=utf8", "main.tex"],
                               cwd=d, capture_output=True, text=True, timeout=90)
            if r.returncode != 0:
                return None, "\n".join(l for l in r.stdout.splitlines() if l.startswith("!") or "l." in l)[-1500:]
            r = subprocess.run(["dvipdfmx", "-q", "main.dvi"], cwd=d, capture_output=True, text=True, timeout=90)
            path = os.path.join(d, "main.pdf")
            if r.returncode != 0 or not os.path.exists(path):
                return None, (r.stderr or r.stdout)[-1500:]
            with open(path, "rb") as f:
                return f.read(), ""
        except Exception as e:  # タイムアウトなど
            return None, str(e)


def student_html(problems, title, seed):
    """LaTeX が使えないとき用:ブラウザで開いて印刷(PDF に保存)できるページ(数式は KaTeX)"""
    import html as H

    def para(s):
        return "".join(f"<p>{H.escape(x)}</p>" for x in s.split("\n\n") if x.strip())

    body = []
    for i, p in enumerate(problems, 1):
        ch = "".join(f'<span class="ch">{lab} ${H.escape(c.tex)}$</span>' for lab, c in zip(LABELS, p.choices))
        fig = '<p class="note">(この問題には図があります。Web の演習画面で確認してください)</p>' if p.figure else ""
        body.append(f'<div class="q"><b class="n">{i}</b>{para(p.statement)}{fig}<div class="chs">{ch}</div>'
                    f'<div class="space"></div></div>')
    ans = []
    for i, p in enumerate(problems, 1):
        j = [c.correct for c in p.choices].index(True)
        ans.append(f'<div class="a"><b class="n">{i}</b><p><b>答:{LABELS[j]} ${H.escape(p.choices[j].tex)}$</b></p>'
                   + "".join(para(s) for s in p.solution) + "</div>")
    return f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>数学IV {H.escape(title)} 演習 {seed}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
 onload="renderMathInElement(document.body,{{delimiters:[{{left:'$$',right:'$$',display:true}},{{left:'$',right:'$',display:false}}]}})"></script>
<style>body{{font-family:serif;max-width:46em;margin:2em auto;line-height:1.7}} .n{{border:1px solid #000;padding:0 .6em;margin-right:.6em}}
.chs{{display:flex;flex-wrap:wrap;gap:.5em 2.5em;margin:.6em 0}} .space{{height:9em}} .note{{color:#666;font-size:.9em}}
.q,.a{{break-inside:avoid;margin-bottom:1.2em}} .answers{{break-before:page}} @media print{{.noprint{{display:none}}}}</style></head><body>
<p class="noprint">ブラウザの「印刷」から「PDF に保存」を選ぶと PDF になります。</p>
<h2>数学IV {H.escape(title)} 演習</h2><p>セット番号 {seed}　　4年　　科　　番　氏名</p>
{''.join(body)}<div class="answers"><h2>解答・解説</h2>{''.join(ans)}</div></body></html>"""


def student_pdf_ui(problems, title, seed, ck):
    if latex_available():
        key = f"{ck}_pdf_{seed}_" + str(abs(hash(tuple(p.statement for p in problems))) % 10 ** 8)
        if st.button("PDF を作る(問題 → 解答・解説)", key=f"{key}_make"):
            with st.spinner("PDF を作成中…"):
                st.session_state[key] = compile_pdf(student_tex(problems, title, seed))
        if key in st.session_state:
            pdf, err = st.session_state[key]
            if pdf:
                st.download_button("PDF をダウンロード", pdf, file_name=f"数学IV_{title.replace(' ', '_')}_演習_{seed}.pdf",
                                   mime="application/pdf", key=f"{key}_dl")
            else:
                st.error("PDF の作成に失敗しました。LaTeX 一式をダウンロードして手元でコンパイルしてください。")
                st.code(err)
        st.caption("印刷して解き、最後のページの解答・解説で答え合わせができます。")
    else:
        st.download_button("印刷用ページをダウンロード(.html)", student_html(problems, title, seed),
                           file_name=f"数学IV_{title.replace(' ', '_')}_演習_{seed}.html", mime="text/html", key=f"{ck}_html")
        st.caption("このサーバーでは PDF を作れないため、印刷用ページを出しています。"
                   "ブラウザで開いて「印刷」→「PDF に保存」で PDF になります(図は Web 画面で確認)。")
