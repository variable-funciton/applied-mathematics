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


@dataclass
class ProblemType:
    key: str
    name: str
    level: str
    gen: object              # gen(rng) -> Problem


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
        st.markdown(p.statement)
        for lab, c in zip(LABELS, p.choices):
            st.markdown(f"{lab}　${c.tex}$")
        ans = st.radio("答え", LABELS, index=None, horizontal=True, key=f"{base}_ans")
        if ans is not None:
            c = p.choices[LABELS.index(ans)]
            if c.correct:
                st.success("正解!")
            else:
                st.error("不正解。" + (c.why or "もう一度計算を確かめよう。"))
        hkey = f"{base}_h"
        k = st.session_state.get(hkey, 0)
        for j, h in enumerate(p.hints[:k], 1):
            st.markdown(f"ヒント{j}:{h}")
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
                    st.markdown(s)
        st.divider()

    st.subheader("紙で解く(LaTeX でダウンロード)")
    st.caption("いま表示している問題(類題に切りかえた問題を含む)を .tex ファイルにする。"
               "手元の LaTeX 環境でコンパイルして使う。")
    e1, e2 = st.columns(2)
    plist = [p for *_, p in probs]
    e1.download_button("問題のみ(.tex)", to_latex(plist, title, seed, False),
                       file_name=f"数学IV_{title.replace(' ', '_')}_演習_{seed}.tex", mime="text/x-tex", key=f"{ck}_dl1")
    e2.download_button("解答つき(.tex)", to_latex(plist, title, seed, True),
                       file_name=f"数学IV_{title.replace(' ', '_')}_演習_{seed}_解答.tex", mime="text/x-tex", key=f"{ck}_dl2")


# ------------------------------------------------------------
# LaTeX 出力(仮のテンプレート。体裁は latex_document だけを差し替えればよい)
# ------------------------------------------------------------
def md_to_tex(s):
    """問題文・解答の Markdown を LaTeX に(数式はそのまま、$$..$$ は \\[..\\])"""
    s = re.sub(r"\$\$(.+?)\$\$", lambda m: r"\[" + m.group(1) + r"\]", s, flags=re.S)
    s = s.replace("**", "")
    return s


def latex_document(title, seed, body):
    return rf"""\documentclass{{jlreq}}
\usepackage{{amsmath,amssymb}}
\usepackage[inline]{{enumitem}}
\usepackage[margin=1.5cm]{{geometry}}
\pagestyle{{empty}}
\title{{数学IV　{title}　演習}}
\date{{セット番号 {seed}}}
\begin{{document}}
\maketitle
{body}
\end{{document}}
"""


def to_latex(problems, title, seed, with_answers):
    items = []
    for p in problems:
        ch = "\n".join(rf"    \item ${c.tex}$" for c in p.choices)
        item = (rf"\item {md_to_tex(p.statement)}\\" + "\n"
                + r"\begin{enumerate*}[label=(\arabic*), itemjoin={\hfill}]" + "\n" + ch + "\n"
                + r"\end{enumerate*}")
        if with_answers:
            k = [c.correct for c in p.choices].index(True)
            sol = "\n\n".join(md_to_tex(s) for s in p.solution)
            item += "\n\n" + rf"\noindent 正解:({k + 1})" + "\n\n" + sol
        else:
            item += "\n\\vfill"
        items.append(item)
    body = r"\begin{enumerate}" + "\n" + "\n\n".join(items) + "\n" + r"\end{enumerate}"
    return latex_document(title, seed, body)
