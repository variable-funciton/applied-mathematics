"""参考文献:分野ごとの文献リスト(教科書・入門書・発展・応用・原論文)と BibTeX

書誌情報(出版社・年・ISBN)は 2026 年 11 月に国立国会図書館サーチ・CiNii・出版社のページなどで確認した。
ISBN は確認できたものだけ載せている。
"""
import re

import streamlit as st

# kind: book / article / web    level: 教科書 / 入門 / 発展 / 応用 / 原論文 / 資料
REFS = {
    "共通": [
        dict(key="takato2023", kind="book", level="教科書", authors="高遠節夫 ほか", title="新応用数学 改訂版",
             publisher="大日本図書", year=2023, isbn="978-4-477-03502-4",
             note="高専の応用数学の教科書。ベクトル解析・ラプラス変換・フーリエ解析・複素関数。"),
        dict(key="kreyszig2011", kind="book", level="入門", authors="E. Kreyszig", title="Advanced Engineering Mathematics, 10th ed.",
             publisher="Wiley", year=2011, note="工学系の応用数学の定番(英語)。この分野のすべてを含む。"),
        dict(key="terasawa1954", kind="book", level="発展", authors="寺沢寛一", title="自然科学者のための数学概論 増訂版",
             publisher="岩波書店", year=1954, note="応用数学の古典的な名著。特殊関数・偏微分方程式まで詳しい。"),
    ],
    "① ベクトル解析": [
        dict(key="toda2019", kind="book", level="入門", authors="戸田盛和", title="ベクトル解析 新装版(理工系の数学入門コース)",
             publisher="岩波書店", year=2019, isbn="978-4-00-029885-8", note="物理の例が豊富な入門書。"),
        dict(key="todawatanabe", kind="book", level="入門", authors="戸田盛和, 渡辺慎介", title="ベクトル解析演習 新装版(理工系の数学入門コース/演習)",
             publisher="岩波書店", year=2020, note="上の本の演習書。"),
        dict(key="schey2005", kind="book", level="入門", authors="H. M. Schey", title="div, grad, curl, and all that: An Informal Text on Vector Calculus, 4th ed.",
             publisher="W. W. Norton", year=2005, note="電磁気学の例で div・grad・rot の意味を説明する(英語)。"),
        dict(key="griffiths2017", kind="book", level="応用", authors="D. J. Griffiths", title="Introduction to Electrodynamics, 4th ed.",
             publisher="Cambridge University Press", year=2017, note="電磁気学の教科書。第1章がベクトル解析のまとめ(英語)。"),
        dict(key="rodrigues1840", kind="article", level="原論文", authors="O. Rodrigues",
             title="Des lois géométriques qui régissent les déplacements d'un système solide dans l'espace",
             journal="J. Math. Pures Appl.", volume="5", pages="380–440", year=1840, note="ロドリゲスの回転公式の原論文(フランス語)。"),
        dict(key="goodfellow2016", kind="book", level="応用", authors="I. Goodfellow, Y. Bengio, A. Courville", title="Deep Learning",
             publisher="MIT Press", year=2016, url="https://www.deeplearningbook.org/",
             note="深層学習の教科書。勾配降下法とその改良(第4章・第8章)。Web で読める(英語)。"),
    ],
    "② ラプラス変換": [
        dict(key="tashiro2014", kind="book", level="入門", authors="田代嘉宏", title="ラプラス変換とフーリエ解析要論 第2版(新装版)",
             publisher="森北出版", year=2014, isbn="978-4-627-02613-1", note="ラプラス変換とフーリエ解析を1冊で。"),
        dict(key="morimoto2015", kind="book", level="入門", authors="森本義廣 監修, 村上純 著", title="基礎から応用までのラプラス変換・フーリエ解析(実用理工学入門講座)",
             publisher="日新出版", year=2015, isbn="978-4-8173-0250-2", note="工学への応用まで扱う。"),
        dict(key="yajima2019", kind="book", level="入門", authors="矢嶋信男", title="常微分方程式 新装版(理工系の数学入門コース)",
             publisher="岩波書店", year=2019, isbn="978-4-00-029886-5", note="ラプラス変換で解く常微分方程式の基礎。"),
        dict(key="schiff1999", kind="book", level="発展", authors="J. L. Schiff", title="The Laplace Transform: Theory and Applications",
             publisher="Springer (Undergraduate Texts in Mathematics)", year=1999, note="理論(逆変換の複素積分)と応用(英語)。"),
    ],
    "③ フーリエ解析": [
        dict(key="oishi2019", kind="book", level="入門", authors="大石進一", title="フーリエ解析 新装版(理工系の数学入門コース)",
             publisher="岩波書店", year=2019, isbn="978-4-00-029888-9", note="フーリエ級数・変換の入門書。"),
        dict(key="nakamura2003", kind="book", level="入門", authors="中村周", title="フーリエ解析(応用数学基礎講座 4)",
             publisher="朝倉書店", year=2003, isbn="978-4-254-11574-1", note="数学的にきちんとした入門書。"),
        dict(key="stein2007", kind="book", level="発展", authors="E. M. スタイン, R. シャカルチ(新井仁之, 杉本充, 高木啓行, 千原浩之 訳)",
             title="フーリエ解析入門(プリンストン解析学講義 I)", publisher="日本評論社", year=2007, isbn="978-4-535-60891-7",
             note="原書 Fourier Analysis: An Introduction (Princeton University Press, 2003)。収束定理・フーリエ変換・応用。"),
        dict(key="arai2022", kind="book", level="発展", authors="新井仁之", title="フーリエ解析とウェーブレット",
             publisher="朝倉書店", year=2022, isbn="978-4-254-11761-5", note="フーリエ解析からウェーブレット解析へ。"),
        dict(key="korner1988", kind="book", level="発展", authors="T. W. Körner", title="Fourier Analysis",
             publisher="Cambridge University Press", year=1988, note="歴史や応用の話題が豊富な読み物風の本(英語)。"),
        dict(key="pinsky2002", kind="book", level="発展", authors="M. A. Pinsky", title="Introduction to Fourier Analysis and Wavelets",
             publisher="Brooks/Cole", year=2002, note="ピンスキー現象の解説を含む(英語。AMS から再刊)。"),
        dict(key="grafakos2014", kind="book", level="発展", authors="L. Grafakos", title="Classical Fourier Analysis, 3rd ed. (Graduate Texts in Mathematics 249)",
             publisher="Springer", year=2014, note="大学院レベルの調和解析の標準的な教科書(英語)。"),
        dict(key="katznelson2004", kind="book", level="発展", authors="Y. Katznelson", title="An Introduction to Harmonic Analysis, 3rd ed.",
             publisher="Cambridge University Press", year=2004, isbn="978-0-521-54359-0",
             note="総和核(summability kernel)・フェイェール核・ポアソン核を第1章で扱う調和解析の教科書(英語)。"),
        dict(key="zygmund2002", kind="book", level="発展", authors="A. Zygmund", title="Trigonometric Series, 3rd ed. (Cambridge Mathematical Library)",
             publisher="Cambridge University Press", year=2002, note="三角級数論の古典。チェザロ・アーベル総和法の詳しい理論(英語)。"),
        dict(key="hardy1949", kind="book", level="発展", authors="G. H. Hardy", title="Divergent Series",
             publisher="Clarendon Press, Oxford", year=1949, note="発散級数と総和法(チェザロ・アーベル・タウバー型定理)の古典(英語)。"),
        dict(key="fejer1904", kind="article", level="原論文", authors="L. Fejér", title="Untersuchungen über Fouriersche Reihen",
             journal="Math. Ann.", volume="58", pages="51–69", year=1904, note="フェイェールの定理(フーリエ級数のチェザロ平均の収束)の原論文(ドイツ語)。"),
        dict(key="daubechies1992", kind="book", level="発展", authors="I. Daubechies", title="Ten Lectures on Wavelets (CBMS-NSF Regional Conference Series in Applied Mathematics 61)",
             publisher="SIAM", year=1992, isbn="978-0-89871-274-2", note="ドブシーのウェーブレット(なめらかで台がコンパクトな直交ウェーブレット)の講義録(英語)。"),
        dict(key="mallat2009", kind="book", level="発展", authors="S. Mallat", title="A Wavelet Tour of Signal Processing: The Sparse Way, 3rd ed.",
             publisher="Academic Press", year=2009, note="多重解像度解析と信号処理(雑音除去・圧縮)への応用(英語)。"),
        dict(key="haar1910", kind="article", level="原論文", authors="A. Haar", title="Zur Theorie der orthogonalen Funktionensysteme",
             journal="Math. Ann.", volume="69", pages="331–371", year=1910, note="ハール関数系の原論文(ドイツ語)。"),
        dict(key="pywt2019", kind="article", level="資料", authors="G. R. Lee, R. Gommers, F. Waselewski, K. Wohlfahrt, A. O'Leary",
             title="PyWavelets: A Python package for wavelet analysis", journal="J. Open Source Softw.", volume="4",
             pages="1237", year=2019, url="https://pywavelets.readthedocs.io/", note="Python のウェーブレット変換ライブラリ PyWavelets(pywt)の論文。リンクは解説書。"),
        dict(key="bracewell2000", kind="book", level="応用", authors="R. N. Bracewell", title="The Fourier Transform and Its Applications, 3rd ed.",
             publisher="McGraw-Hill", year=2000, note="工学でのフーリエ変換の使い方(英語)。"),
        dict(key="wilbraham1848", kind="article", level="原論文", authors="H. Wilbraham", title="On a certain periodic function",
             journal="Cambridge Dublin Math. J.", volume="3", pages="198–201", year=1848, note="ギブス現象を最初に指摘した論文。"),
        dict(key="gibbs1899", kind="article", level="原論文", authors="J. W. Gibbs", title="Fourier's series",
             journal="Nature", volume="59", pages="606", year=1899, note="ギブスによる説明(Nature への手紙)。"),
        dict(key="hewitt1979", kind="article", level="発展", authors="E. Hewitt, R. E. Hewitt",
             title="The Gibbs–Wilbraham phenomenon: An episode in Fourier analysis",
             journal="Arch. Hist. Exact Sci.", volume="21", pages="129–160", year=1979, note="ギブス現象の歴史。"),
        dict(key="pst1993", kind="article", level="原論文", authors="M. A. Pinsky, N. K. Stanton, P. E. Trapa",
             title="Fourier series of radial functions in several variables", journal="J. Funct. Anal.", volume="116",
             pages="111–132", year=1993, url="https://doi.org/10.1006/jfan.1993.1106", note="ピンスキー現象(球の中心での発散)。"),
        dict(key="taylor2002", kind="article", level="発展", authors="M. E. Taylor",
             title="The Gibbs phenomenon, the Pinsky phenomenon, and variants for eigenfunction expansions",
             journal="Comm. Partial Differential Equations", volume="27", pages="565–605", year=2002, note="ギブス現象・ピンスキー現象の一般化。"),
        dict(key="kuratsubo2010", kind="article", level="原論文", authors="S. Kuratsubo", title="On pointwise convergence of Fourier series of the indicator function of D dimensional ball",
             journal="J. Fourier Anal. Appl.", volume="16", pages="52–59", year=2010, url="https://doi.org/10.1007/s00041-009-9071-3",
             note="倉坪現象(d ≥ 5 で有理点で収束しない)。"),
        dict(key="kno2010", kind="article", level="原論文", authors="S. Kuratsubo, E. Nakai, K. Ootsubo",
             title="Generalized Hardy identity and relations to Gibbs–Wilbraham and Pinsky phenomena", journal="J. Funct. Anal.",
             volume="259", pages="315–342", year=2010, note="3つの現象と格子点問題(ハーディの恒等式)。"),
        dict(key="kn2022", kind="article", level="原論文", authors="S. Kuratsubo, E. Nakai", title="Multiple Fourier series and lattice point problems",
             journal="J. Funct. Anal.", volume="282", pages="109272", year=2022, url="https://arxiv.org/abs/2011.06178",
             note="多重フーリエ級数と格子点問題。数値例の図を含む(arXiv で読める)。"),
    ],
    "④ 複素関数": [
        dict(key="omote2019", kind="book", level="入門", authors="表実", title="複素関数 新装版(理工系の数学入門コース)",
             publisher="岩波書店", year=2019, isbn="978-4-00-029887-2", note="工学系向けの入門書。"),
        dict(key="jimbo2024", kind="book", level="発展", authors="神保道夫", title="複素関数入門 新装版(現代数学への入門)",
             publisher="岩波書店", year=2024, isbn="978-4-00-029926-8", note="数学的にていねいな入門書。"),
        dict(key="ahlfors1979", kind="book", level="発展", authors="L. V. Ahlfors", title="Complex Analysis, 3rd ed.",
             publisher="McGraw-Hill", year=1979, note="複素解析の古典的な教科書(英語。日本語訳:笠原乾吉 訳『複素解析』現代数学社)。"),
        dict(key="needham1997", kind="book", level="発展", authors="T. Needham", title="Visual Complex Analysis",
             publisher="Oxford University Press", year=1997, note="図で考える複素解析。写像・定義域の色付けの考え方(英語)。"),
    ],
    "⑤ 偏微分方程式": [
        dict(key="matano2024", kind="book", level="入門", authors="俣野博, 神保道夫", title="熱・波動と微分方程式 新装版(現代数学への入門)",
             publisher="岩波書店", year=2024, isbn="978-4-00-029928-2", note="熱方程式・波動方程式とフーリエ解析。"),
        dict(key="strauss2008", kind="book", level="入門", authors="W. A. Strauss", title="Partial Differential Equations: An Introduction, 2nd ed.",
             publisher="Wiley", year=2008, note="偏微分方程式の入門書の定番(英語)。"),
        dict(key="evans2010", kind="book", level="発展", authors="L. C. Evans", title="Partial Differential Equations, 2nd ed. (Graduate Studies in Mathematics 19)",
             publisher="American Mathematical Society", year=2010, note="大学院レベルの標準的な教科書(英語)。"),
        dict(key="crank1975", kind="book", level="応用", authors="J. Crank", title="The Mathematics of Diffusion, 2nd ed.",
             publisher="Oxford University Press", year=1975, note="拡散方程式の解法と化学・材料への応用(英語)。"),
        dict(key="fourier1822", kind="book", level="原論文", authors="J. Fourier", title="Théorie analytique de la chaleur",
             publisher="Firmin Didot", year=1822, note="熱伝導方程式とフーリエ級数の出発点(フランス語)。"),
    ],
    "⑦ 特殊関数": [
        dict(key="inui1962", kind="book", level="発展", authors="犬井鉄郎", title="特殊函数(岩波全書 252)",
             publisher="岩波書店", year=1962, note="ガンマ関数・ベッセル関数などの古典的な教科書。"),
        dict(key="aar1999", kind="book", level="発展", authors="G. E. Andrews, R. Askey, R. Roy", title="Special Functions",
             publisher="Cambridge University Press", year=1999, note="特殊関数の標準的な教科書(英語)。"),
        dict(key="nist2010", kind="book", level="資料", authors="F. W. J. Olver, D. W. Lozier, R. F. Boisvert, C. W. Clark (eds.)",
             title="NIST Handbook of Mathematical Functions", publisher="Cambridge University Press", year=2010,
             url="https://dlmf.nist.gov/", note="公式集。Web 版(DLMF)は無料で使え, 更新されている(英語)。"),
    ],
    "応用(学科別)": [
        dict(key="denhartog1985", kind="book", level="応用", authors="J. P. Den Hartog", title="Mechanical Vibrations",
             publisher="Dover", year=1985, note="【機械】機械振動の古典。共振・振動の絶縁(英語)。"),
        dict(key="oppenheim1997", kind="book", level="応用", authors="A. V. Oppenheim, A. S. Willsky, S. H. Nawab",
             title="Signals and Systems, 2nd ed.", publisher="Prentice Hall", year=1997,
             note="【電気電子】信号とシステム。フーリエ変換・ラプラス変換・サンプリング(英語)。"),
        dict(key="chopra2017", kind="book", level="応用", authors="A. K. Chopra", title="Dynamics of Structures: Theory and Applications to Earthquake Engineering, 5th ed.",
             publisher="Pearson", year=2017, isbn="978-0-13-455512-6", note="【都市】構造物の振動と地震応答・応答スペクトル(英語)。"),
        dict(key="pugh2014", kind="book", level="応用", authors="D. Pugh, P. Woodworth", title="Sea-Level Science: Understanding Tides, Surges, Tsunamis and Mean Sea-Level Changes",
             publisher="Cambridge University Press", year=2014, note="【都市】潮汐の調和解析を含む海面水位の教科書(英語)。"),
        dict(key="keeler2010", kind="book", level="応用", authors="J. Keeler", title="Understanding NMR Spectroscopy, 2nd ed.",
             publisher="Wiley", year=2010, note="【化学】FT-NMR のしくみ(FID とフーリエ変換)(英語)。"),
        dict(key="griffiths2007", kind="book", level="応用", authors="P. R. Griffiths, J. A. de Haseth", title="Fourier Transform Infrared Spectrometry, 2nd ed.",
             publisher="Wiley", year=2007, note="【化学】FT-IR の原理と応用(英語)。"),
    ],
}

LEVELS = ["教科書", "入門", "発展", "応用", "原論文", "資料"]


def is_ja(s):
    return bool(re.search(r"[぀-ヿ一-鿿]", s))


def fmt(r):
    """数学IB の参考文献の書き方に近い形:和書は『』, 洋書はイタリック, 論文は「」/ 引用符"""
    if r["kind"] == "article":
        t = f"「{r['title']}」" if is_ja(r["title"]) else f"“{r['title']}”"
        s = f"{r['authors']}, {t}, *{r['journal']}* **{r['volume']}** ({r['year']}), {r['pages']}."
    elif is_ja(r["title"]):
        s = f"{r['authors']}『{r['title']}』{r['publisher']}, {r['year']}."
    else:
        s = f"{r['authors']}, *{r['title']}*, {r['publisher']}, {r['year']}."
    if r.get("isbn"):
        s += f" ISBN {r['isbn']}"
    if r.get("url"):
        s += f" [リンク]({r['url']})"
    return s


def bib_authors(a):
    """「A, B, C」→「A and B and C」。括弧の中(訳者・eds.)は別に返す"""
    m = re.search(r"[((](.+?)[))]\s*$", a)
    extra = m.group(1) if m else ""
    a = a[:m.start()].strip() if m else a
    return " and ".join(x.strip() for x in a.split(",")), extra


def bibtex(refs):
    out = []
    for r in refs:
        auth, extra = bib_authors(r["authors"])
        if r["kind"] == "article":
            f = dict(author=auth, title=r["title"], journal=r["journal"], volume=r["volume"],
                     pages=r["pages"].replace("–", "--"), year=r["year"])
            typ = "article"
        else:
            f = {("editor" if extra == "eds." else "author"): auth, "title": r["title"], "publisher": r["publisher"],
                 "year": r["year"]}
            typ = "book"
        if r.get("isbn"):
            f["isbn"] = r["isbn"]
        if r.get("url"):
            f["url"] = r["url"]
        f["note"] = (extra + "。" if extra and extra != "eds." else "") + r["note"]
        body = ",\n".join(f"  {k} = {{{v}}}" for k, v in f.items())
        out.append(f"@{typ}{{{r['key']},\n{body}\n}}")
    return "\n\n".join(out) + "\n"


def all_refs():
    return [r for rs in REFS.values() for r in rs]


def section(field):
    def tab():
        st.caption("書誌情報は 2026 年 11 月に国立国会図書館サーチ・CiNii・出版社のページなどで確認しました。"
                   "ISBN は確認できたものだけ載せています。版が新しくなっている場合があります。")
        rs = REFS[field]
        for lv in LEVELS:
            items = [r for r in rs if r["level"] == lv]
            if not items:
                continue
            st.markdown(f"**{lv}**")
            st.markdown("\n".join(f"- {fmt(r)}  \n  {r['note']}" for r in items))
        st.download_button(f"この分野の BibTeX(.bib)をダウンロード", bibtex(rs), file_name="refs_" + field.split()[-1] + ".bib",
                           mime="text/plain", key=f"bib_{field}")
    return tab


def tab_all_bib():
    st.markdown("すべての分野の文献を1つの BibTeX ファイルにまとめたもの。LaTeX で `\\bibliography{refs}` として使えます"
                "(和文の文献は pBibTeX/upBibTeX を使ってください)。")
    rs = all_refs()
    st.write(f"{len(rs)} 件")
    st.download_button("すべての BibTeX(refs.bib)をダウンロード", bibtex(rs), file_name="refs.bib", mime="text/plain", key="bib_all")
    with st.expander("BibTeX を表示"):
        st.code(bibtex(rs), language="bibtex")


SECTIONS = [(f, section(f)) for f in REFS] + [("BibTeX(まとめ)", tab_all_bib)]
