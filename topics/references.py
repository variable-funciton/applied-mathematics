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
        dict(key="gabor1946", kind="article", level="原論文", authors="D. Gabor", title="Theory of communication",
             journal="J. Inst. Electr. Eng. Part III", volume="93", pages="429–457", year=1946,
             note="窓をかけたフーリエ変換(ガボール変換・短時間フーリエ変換)と時間周波数の不確定性。"),
        dict(key="grossmann1984", kind="article", level="原論文", authors="A. Grossmann, J. Morlet",
             title="Decomposition of Hardy functions into square integrable wavelets of constant shape", journal="SIAM J. Math. Anal.",
             volume="15", pages="723–736", year=1984, url="https://doi.org/10.1137/0515056", note="連続ウェーブレット変換と逆変換の公式。"),
        dict(key="daubechies1988", kind="article", level="原論文", authors="I. Daubechies",
             title="Orthonormal bases of compactly supported wavelets", journal="Comm. Pure Appl. Math.", volume="41",
             pages="909–996", year=1988, url="https://doi.org/10.1002/cpa.3160410705", note="台がコンパクトでなめらかな直交ウェーブレット。"),
        dict(key="mallat1989", kind="article", level="原論文", authors="S. G. Mallat",
             title="A theory for multiresolution signal decomposition: the wavelet representation",
             journal="IEEE Trans. Pattern Anal. Mach. Intell.", volume="11", pages="674–693", year=1989,
             url="https://doi.org/10.1109/34.192463", note="多重解像度解析と高速ウェーブレット変換。"),
        dict(key="changkuo1993", kind="article", level="原論文", authors="T. Chang, C.-C. J. Kuo",
             title="Texture analysis and classification with tree-structured wavelet transform", journal="IEEE Trans. Image Process.",
             volume="2", pages="429–441", year=1993, note="ウェーブレット係数のエネルギーによるテクスチャ分類。"),
        dict(key="mallat2012", kind="article", level="原論文", authors="S. Mallat", title="Group invariant scattering",
             journal="Comm. Pure Appl. Math.", volume="65", pages="1331–1398", year=2012, url="https://doi.org/10.1002/cpa.21413",
             note="散乱変換の理論(平行移動に不変, 変形に安定)。"),
        dict(key="brunamallat2013", kind="article", level="原論文", authors="J. Bruna, S. Mallat", title="Invariant scattering convolution networks",
             journal="IEEE Trans. Pattern Anal. Mach. Intell.", volume="35", pages="1872–1886", year=2013,
             url="https://doi.org/10.1109/TPAMI.2012.230", note="散乱変換による画像分類。"),
        dict(key="abel2017", kind="web", level="資料", authors="The Norwegian Academy of Science and Letters",
             title="The Abel Prize 2017: Yves Meyer (citation)", publisher="International Mathematical Union", year=2017,
             url="https://www.mathunion.org/fileadmin/IMU/Prizes/Abel/2017/Abelprize_2017_Meyer_citation.pdf",
             note="ウェーブレット理論の歴史(ハール, モルレー, カルデロン, メイエ, マラー, ドブシー)と応用。"),
        dict(key="radon1917", kind="article", level="原論文", authors="J. Radon",
             title="Über die Bestimmung von Funktionen durch ihre Integralwerte längs gewisser Mannigfaltigkeiten",
             journal="Ber. Verh. Sächs. Akad. Wiss. Leipzig, Math.-Phys. Kl.", volume="69", pages="262–277", year=1917,
             note="ラドン変換と逆変換の原論文(ドイツ語)。"),
        dict(key="cormack1963", kind="article", level="原論文", authors="A. M. Cormack",
             title="Representation of a function by its line integrals, with some radiological applications",
             journal="J. Appl. Phys.", volume="34", pages="2722–2727", year=1963, note="X 線 CT の数学的な基礎(続編 II は 35 (1964), 2908–2913)。"),
        dict(key="nobel1979", kind="web", level="資料", authors="Nobel Prize Outreach", title="The Nobel Prize in Physiology or Medicine 1979",
             publisher="nobelprize.org", year=1979, url="https://www.nobelprize.org/prizes/medicine/1979/summary/",
             note="コーマックとハウンスフィールド「コンピュータ断層撮影の開発」。"),
        dict(key="natterer2001", kind="book", level="発展", authors="F. Natterer", title="The Mathematics of Computerized Tomography (Classics in Applied Mathematics)",
             publisher="SIAM", year=2001, isbn="978-0-89871-493-7", note="CT の数学の標準的な教科書。ラドン変換の性質・逆変換・数値解法(英語)。"),
        dict(key="helgason1999", kind="book", level="発展", authors="S. Helgason", title="The Radon Transform, 2nd ed. (Progress in Mathematics 5)",
             publisher="Birkhäuser", year=1999, note="積分幾何としてのラドン変換(英語)。"),
        dict(key="kakslaney2001", kind="book", level="応用", authors="A. C. Kak, M. Slaney", title="Principles of Computerized Tomographic Imaging (Classics in Applied Mathematics)",
             publisher="SIAM", year=2001, url="https://engineering.purdue.edu/~malcolm/pct",
             note="フィルタ補正逆投影法などの再構成アルゴリズム。個人利用なら Web で読める(英語, 原著 IEEE Press 1988)。"),
        dict(key="epstein2007", kind="book", level="発展", authors="C. L. Epstein", title="Introduction to the Mathematics of Medical Imaging, 2nd ed.",
             publisher="SIAM", year=2007, isbn="978-0-89871-642-9", note="医用画像の数学。フーリエ解析からラドン変換・CT・MRI まで(英語)。"),
        dict(key="grangeat1991", kind="article", level="応用", authors="P. Grangeat",
             title="Mathematical framework of cone beam 3D reconstruction via the first derivative of the Radon transform",
             journal="Lecture Notes in Math.", volume="1497", pages="66–97", year=1991, url="https://doi.org/10.1007/BFb0084509",
             note="円錐ビーム CT のデータを3次元ラドン変換の微分に結びつける。"),
        dict(key="cooleytukey1965", kind="article", level="原論文", authors="J. W. Cooley, J. W. Tukey",
             title="An algorithm for the machine calculation of complex Fourier series", journal="Math. Comp.", volume="19",
             pages="297–301", year=1965, url="https://doi.org/10.1090/S0025-5718-1965-0178586-1", note="高速フーリエ変換(FFT)の論文。"),
        dict(key="heideman1985", kind="article", level="発展", authors="M. T. Heideman, D. H. Johnson, C. S. Burrus",
             title="Gauss and the history of the fast Fourier transform", journal="Arch. Hist. Exact Sci.", volume="34",
             pages="265–277", year=1985, url="https://doi.org/10.1007/BF00348431", note="ガウスによる FFT の先駆けと FFT の歴史。"),
        dict(key="ieee_fft", kind="web", level="資料", authors="IEEE History Center",
             title="Milestones: First Demonstration of the Fast Fourier Transform (FFT), 1964", publisher="Engineering and Technology History Wiki",
             year=None, url="https://ethw.org/Milestones:First_Demonstration_of_the_Fast_Fourier_Transform_(FFT),_1964",
             note="IBM での FFT の実演と, 地下核実験の検出という開発の動機。"),
        dict(key="shannon1949", kind="article", level="原論文", authors="C. E. Shannon", title="Communication in the presence of noise",
             journal="Proc. IRE", volume="37", pages="10–21", year=1949, url="https://doi.org/10.1109/JRPROC.1949.232969",
             note="標本化定理(1940年投稿)。"),
        dict(key="someya1949", kind="book", level="原論文", authors="染谷勲", title="波形伝送(高周波科学論叢 第3輯)",
             publisher="修教社", year=1949, note="シャノンと独立に標本化定理を示した。"),
        dict(key="yoshino2015", kind="article", level="発展", authors="吉野邦生", title="標本化定理における複素解析的手法",
             journal="数理解析研究所講究録", volume="1972", pages="67–83", year=2015,
             url="https://www.kurims.kyoto-u.ac.jp/~kyodo/kokyuroku/contents/pdf/1972-05.pdf", note="シャノン=染谷の標本化定理の解説と発展(日本語)。"),
        dict(key="unser2000", kind="article", level="発展", authors="M. Unser", title="Sampling—50 years after Shannon",
             journal="Proc. IEEE", volume="88", pages="569–587", year=2000, note="標本化定理のその後の発展(スプライン・ウェーブレット)。"),
        dict(key="riesz1928", kind="article", level="原論文", authors="M. Riesz", title="Sur les fonctions conjuguées",
             journal="Math. Z.", volume="27", pages="218–244", year=1928, url="https://doi.org/10.1007/BF01171098",
             note="共役関数(ヒルベルト変換)の Lᵖ 有界性(1 < p < ∞)。"),
        dict(key="hormander1960", kind="article", level="原論文", authors="L. Hörmander",
             title="Estimates for translation invariant operators in L^p spaces", journal="Acta Math.", volume="104",
             pages="93–140", year=1960, url="https://doi.org/10.1007/BF02547187", note="フーリエマルチプライヤーの Lᵖ 有界性の定理。"),
        dict(key="hardylittlewood1928", kind="article", level="原論文", authors="G. H. Hardy, J. E. Littlewood",
             title="Some properties of fractional integrals. I", journal="Math. Z.", volume="27", pages="565–606", year=1928,
             note="分数冪積分(リースポテンシャル)の不等式(1次元)。"),
        dict(key="adams1975", kind="article", level="発展", authors="D. R. Adams", title="A note on Riesz potentials",
             journal="Duke Math. J.", volume="42", pages="765–778", year=1975, url="https://doi.org/10.1215/S0012-7094-75-04265-9",
             note="モレー空間でのリースポテンシャルの有界性(アダムスの定理)。"),
        dict(key="stein1970", kind="book", level="発展", authors="E. M. Stein",
             title="Singular Integrals and Differentiability Properties of Functions (Princeton Mathematical Series 30)",
             publisher="Princeton University Press", year=1970, isbn="978-0-691-08079-6",
             note="特異積分・マルチプライヤー・リースポテンシャル・ソボレフ空間の古典的な教科書(英語)。"),
        dict(key="steinreal2005", kind="book", level="発展", authors="E. M. Stein, R. Shakarchi",
             title="Real Analysis: Measure Theory, Integration, and Hilbert Spaces (Princeton Lectures in Analysis III)",
             publisher="Princeton University Press", year=2005, isbn="978-0-691-11386-9",
             note="ルベーグ積分・L² 空間・ヒルベルト空間(英語)。"),
        dict(key="riesz1910", kind="article", level="原論文", authors="F. Riesz", title="Untersuchungen über Systeme integrierbarer Funktionen",
             journal="Math. Ann.", volume="69", pages="449–497", year=1910, note="Lᵖ 空間の導入。"),
        dict(key="jordanvonneumann1935", kind="article", level="原論文", authors="P. Jordan, J. von Neumann",
             title="On inner products in linear, metric spaces", journal="Ann. of Math. (2)", volume="36", pages="719–723", year=1935,
             url="https://doi.org/10.2307/1968653", note="中線定理をみたすノルムは内積から作られる。"),
        dict(key="birnbaumorlicz1931", kind="article", level="原論文", authors="Z. W. Birnbaum, W. Orlicz",
             title="Über die Verallgemeinerung des Begriffes der zueinander konjugierten Potenzen", journal="Studia Math.",
             volume="3", pages="1–67", year=1931, note="オーリッチ空間の始まり。"),
        dict(key="morrey1938", kind="article", level="原論文", authors="C. B. Morrey",
             title="On the solutions of quasi-linear elliptic partial differential equations", journal="Trans. Amer. Math. Soc.",
             volume="43", pages="126–166", year=1938, url="https://doi.org/10.1090/S0002-9947-1938-1501936-8", note="モレー空間の始まり。"),
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
        dict(key="batchelor2000", kind="book", level="発展", authors="G. K. Batchelor", title="An Introduction to Fluid Dynamics (Cambridge Mathematical Library)",
             publisher="Cambridge University Press", year=2000, isbn="978-0-521-66396-0", note="流体力学の標準的な教科書。オイラー方程式・ナヴィエ・ストークス方程式(英語)。"),
        dict(key="griffiths2018", kind="book", level="入門", authors="D. J. Griffiths, D. F. Schroeter", title="Introduction to Quantum Mechanics, 3rd ed.",
             publisher="Cambridge University Press", year=2018, isbn="978-1-107-18963-8", note="シュレディンガー方程式, 箱の中の粒子, トンネル効果(英語)。"),
        dict(key="szeng2007", kind="book", level="応用", authors="S. M. Sze, K. K. Ng", title="Physics of Semiconductor Devices, 3rd ed.",
             publisher="Wiley", year=2007, isbn="978-0-471-14323-9", note="半導体デバイスの標準的な教科書。ドリフト拡散, pn 接合(英語)。"),
        dict(key="euler1757", kind="article", level="原論文", authors="L. Euler", title="Principes généraux du mouvement des fluides",
             journal="Mémoires de l'académie des sciences de Berlin", volume="11", pages="274–315", year=1757,
             url="https://scholarlycommons.pacific.edu/euler-works/226", note="オイラー方程式(1755 年執筆, E226)。"),
        dict(key="navier1827", kind="article", level="原論文", authors="C.-L. Navier", title="Mémoire sur les lois du mouvement des fluides",
             journal="Mémoires de l'Académie Royale des Sciences de l'Institut de France", volume="6", pages="389–440", year=1827,
             note="1822 年に学士院で発表(巻の日付は 1823 年)。粘性流体の方程式。"),
        dict(key="stokes1880", kind="book", level="原論文", authors="G. G. Stokes", title="On the theories of the internal friction of fluids in motion (1845), in: Mathematical and Physical Papers, Vol. 1, pp. 75–129",
             publisher="Cambridge University Press", year=1880, note="粘性流体の方程式の導出(1845 年の論文の再録)。"),
        dict(key="reynolds1883", kind="article", level="原論文", authors="O. Reynolds",
             title="An experimental investigation of the circumstances which determine whether the motion of water shall be direct or sinuous, and of the law of resistance in parallel channels",
             journal="Phil. Trans. R. Soc. Lond.", volume="174", pages="935–982", year=1883, url="https://doi.org/10.1098/rstl.1883.0029",
             note="層流と乱流の遷移(レイノルズ数)。"),
        dict(key="schrodinger1926", kind="article", level="原論文", authors="E. Schrödinger", title="Quantisierung als Eigenwertproblem",
             journal="Ann. Phys.", volume="384", pages="361–376", year=1926, url="https://doi.org/10.1002/andp.19263840404",
             note="シュレディンガー方程式(固有値問題としての量子化)。"),
        dict(key="born1926", kind="article", level="原論文", authors="M. Born", title="Zur Quantenmechanik der Stoßvorgänge",
             journal="Z. Phys.", volume="37", pages="863–867", year=1926, url="https://doi.org/10.1007/BF01397477", note="波動関数の確率解釈。"),
        dict(key="shockley1949", kind="article", level="原論文", authors="W. Shockley", title="The theory of p-n junctions in semiconductors and p-n junction transistors",
             journal="Bell Syst. Tech. J.", volume="28", pages="435–489", year=1949, note="pn 接合とダイオードの電流の理論。"),
        dict(key="vanroosbroeck1950", kind="article", level="原論文", authors="W. van Roosbroeck",
             title="Theory of the flow of electrons and holes in germanium and other semiconductors", journal="Bell Syst. Tech. J.",
             volume="29", pages="560–607", year=1950, url="https://doi.org/10.1002/j.1538-7305.1950.tb03653.x", note="ドリフト拡散方程式の基礎方程式系。"),
        dict(key="wilmott1995", kind="book", level="入門", authors="P. Wilmott, S. Howison, J. Dewynne",
             title="The Mathematics of Financial Derivatives: A Student Introduction", publisher="Cambridge University Press", year=1995,
             isbn="978-0-521-49789-3", note="ブラック・ショールズ方程式と, 熱方程式への変換(英語)。"),
        dict(key="bachelier1900", kind="article", level="原論文", authors="L. Bachelier", title="Théorie de la spéculation",
             journal="Ann. Sci. École Norm. Sup. (3)", volume="17", pages="21–86", year=1900, url="https://doi.org/10.24033/asens.476",
             note="株価をブラウン運動でモデル化した最初の論文。"),
        dict(key="ito1944", kind="article", level="原論文", authors="K. Itô", title="Stochastic integral", journal="Proc. Imp. Acad. Tokyo",
             volume="20", pages="519–524", year=1944, url="https://doi.org/10.3792/pia/1195572786", note="伊藤積分(確率積分)。"),
        dict(key="ito1951", kind="article", level="原論文", authors="K. Itô", title="On a formula concerning stochastic differentials",
             journal="Nagoya Math. J.", volume="3", pages="55–65", year=1951, note="伊藤の公式。"),
        dict(key="blackscholes1973", kind="article", level="原論文", authors="F. Black, M. Scholes", title="The pricing of options and corporate liabilities",
             journal="J. Polit. Econ.", volume="81", pages="637–654", year=1973, url="https://doi.org/10.1086/260062", note="ブラック・ショールズ方程式と価格式。"),
        dict(key="merton1973", kind="article", level="原論文", authors="R. C. Merton", title="Theory of rational option pricing",
             journal="Bell J. Econ. Manag. Sci.", volume="4", pages="141–183", year=1973, url="https://doi.org/10.2307/3003143", note="オプション価格理論の厳密化と拡張。"),
        dict(key="crr1979", kind="article", level="原論文", authors="J. C. Cox, S. A. Ross, M. Rubinstein", title="Option pricing: A simplified approach",
             journal="J. Financ. Econ.", volume="7", pages="229–263", year=1979, url="https://doi.org/10.1016/0304-405X(79)90015-1", note="二項モデル。"),
        dict(key="schwartz1945", kind="article", level="原論文", authors="L. Schwartz",
             title="Généralisation de la notion de fonction, de dérivation, de transformation de Fourier et applications mathématiques et physiques",
             journal="Annales de l'Université de Grenoble", volume="21", pages="57–74", year=1945,
             url="https://numdam.org/item/AUG_1945__21__57_0/", note="超関数(distribution)を導入した論文。"),
        dict(key="schwartz1950", kind="book", level="発展", authors="L. Schwartz", title="Théorie des distributions, Tomes I, II",
             publisher="Hermann, Paris", year=1950, note="超関数の理論の基本文献(第 I 巻 1950 年, 第 II 巻 1951 年)。"),
        dict(key="dirac1927", kind="article", level="原論文", authors="P. A. M. Dirac", title="The physical interpretation of the quantum dynamics",
             journal="Proc. R. Soc. Lond. A", volume="113", pages="621–641", year=1927, note="デルタ関数を量子力学に導入。"),
        dict(key="sobolev1936", kind="article", level="原論文", authors="S. L. Sobolev",
             title="Méthode nouvelle à résoudre le problème de Cauchy pour les équations linéaires hyperboliques normales",
             journal="Mat. Sb.", volume="1(43)", pages="39–72", year=1936, url="https://mathnet.ru/eng/sm5358", note="一般化された解(超関数の先がけ)。"),
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
        dict(key="szego1939", kind="book", level="発展", authors="G. Szegő", title="Orthogonal Polynomials (AMS Colloquium Publications 23)",
             publisher="American Mathematical Society", year=1939, isbn="978-0-8218-1023-1", note="直交多項式の古典的な教科書(英語)。"),
        dict(key="frobenius1873", kind="article", level="原論文", authors="G. Frobenius", title="Ueber die Integration der linearen Differentialgleichungen durch Reihen",
             journal="J. Reine Angew. Math.", volume="76", pages="214–235", year=1873, note="確定特異点のまわりの級数解(フロベニウスの方法)。"),
        dict(key="rodrigues1816", kind="article", level="原論文", authors="O. Rodrigues", title="Mémoire sur l'attraction des sphéroïdes",
             journal="Correspondance sur l'École Impériale Polytechnique", volume="3", pages="361–385", year=1816, note="ルジャンドル多項式のロドリゲスの公式。"),
        dict(key="hermite1864", kind="article", level="原論文", authors="C. Hermite", title="Sur un nouveau développement en série des fonctions",
             journal="C. R. Acad. Sci. Paris", volume="58", pages="93–100, 266–273", year=1864, note="エルミート多項式による関数の展開。"),
        dict(key="laguerre1879", kind="article", level="原論文", authors="E. Laguerre", title="Sur l'intégrale ∫_x^∞ e^{−x} dx/x",
             journal="Bull. Soc. Math. France", volume="7", pages="72–81", year=1879, url="https://www.numdam.org/item/BSMF_1879__7__72_1",
             note="ラゲール多項式と直交性(連分数による積分の近似から)。"),
        dict(key="walsh1923", kind="article", level="原論文", authors="J. L. Walsh", title="A closed set of normal orthogonal functions",
             journal="Amer. J. Math.", volume="45", pages="5–24", year=1923, note="ウォルシュ関数系を導入した論文。"),
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
    elif r["kind"] == "web":
        s = f"{r['authors']}, *{r['title']}*, {r['publisher']}."
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
        elif r["kind"] == "web":
            f = dict(author=auth, title=r["title"], howpublished=r["publisher"])
            if r.get("year"):
                f["year"] = r["year"]
            typ = "misc"
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
