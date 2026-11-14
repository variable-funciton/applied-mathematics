"""数学IV フーリエ解析 発展 ヒルベルト変換:演習の問題生成

規約:Hf(t) = (1/π) p.v. ∫ f(s)/(t − s) ds,  (Hf)^(ω) = −i sgn(ω) F(ω)。
H[cos at] = sin at, H[sin at] = −cos at (a > 0), H[a/(t²+a²)] = t/(t²+a²), Hχ_[a,b] = (1/π) log|(t−a)/(t−b)|。
"""
from fractions import Fraction

from topics.ex_core import Choice, ExamItem, Part, Problem, ProblemType, Retry, render, sqrt_parts, tex_fr

F = Fraction


def distinct(ok, cands):
    choices = [Choice(ok, True, value=ok)]
    for t, why in cands:
        if all(c.tex != t for c in choices) and len(choices) < 4:
            choices.append(Choice(t, why=why, value=t))
    if len(choices) < 4:
        raise Retry
    return choices


def term(c, body, first=False):
    """係数 c(0 でない整数)と本体 → 符号つきの TeX"""
    sgn = "-" if c < 0 else ("" if first else "+")
    return sgn + ("" if abs(c) == 1 else str(abs(c))) + body


def trig(A, a, B, b):
    """A cos at + B sin bt の TeX(0 の項は省く)"""
    out = ""
    if A:
        out += term(A, rf"\cos {a if a != 1 else ''}t", True)
    if B:
        out += term(B, rf"\sin {b if b != 1 else ''}t", not out)
    return out or "0"


def trig2(S, a, C, b):
    """S sin at + C cos bt の TeX"""
    out = ""
    if S:
        out += term(S, rf"\sin {a if a != 1 else ''}t", True)
    if C:
        out += term(C, rf"\cos {b if b != 1 else ''}t", not out)
    return out or "0"


def root_tex(n):
    k, m = sqrt_parts(n)
    if m == 1:
        return str(k)
    return (str(k) if k != 1 else "") + rf"\sqrt{{{m}}}"


# ------------------------------------------------------------
# 基本:三角関数のヒルベルト変換
# ------------------------------------------------------------
def g_trig(rng):
    A, B = rng.choice([1, 2, 3, -2, 4]), rng.choice([1, -1, 2, -3, 5])
    a, b = rng.sample([1, 2, 3, 4, 5], 2)
    ok = trig2(A, a, -B, b)                 # H[A cos at + B sin bt] = A sin at − B cos bt
    cands = [(trig2(A, a, B, b), r"$H[\sin bt]=-\cos bt$:符号はマイナス。"), (trig2(-A, a, B, b), r"$H[\cos at]=\sin at$(プラス), $H[\sin bt]=-\cos bt$(マイナス)。"),
             (trig2(-A * a, a, B * b, b), "微分ではない。振幅は変わらず, 位相だけが $90^\\circ$ ずれる。"),
             (trig(A, a, B, b), "ヒルベルト変換は恒等変換ではない。$\\cos$ と $\\sin$ が入れかわる。"), (trig2(-A, a, -B, b), r"$H[\cos at]=\sin at$ の符号を確かめよう。")]
    return Problem("三角関数のヒルベルト変換", "基本", rf"$f(t)={trig(A, a, B, b)}$ のヒルベルト変換 $Hf(t)$ を求めよ。ただし $H[\cos at]=\sin at$, $H[\sin at]=-\cos at$($a>0$)である。",
                   distinct(ok, cands), hints=[r"$H$ は線形。各項の位相を $90^\circ$ 遅らせる:$\cos\to\sin$, $\sin\to-\cos$。"],
                   solution=[rf"$Hf={A if A != 1 else ''}H[\cos {a if a != 1 else ''}t]{'+' if B > 0 else '-'}{abs(B) if abs(B) != 1 else ''}H[\sin {b if b != 1 else ''}t]={ok}$。"],
                   meta=dict(A=A, a=a, B=B, b=b))


# ------------------------------------------------------------
# 基本:2回のヒルベルト変換・マルチプライヤー
# ------------------------------------------------------------
def g_twice(rng):
    A, B = rng.choice([1, 2, 3, -2]), rng.choice([1, -1, 2, 4])
    a, b = rng.sample([1, 2, 3, 5], 2)
    n = rng.choice([2, 3, 4])
    # H^n:2 回で −1 倍, 3 回で −H, 4 回で恒等
    res = {2: trig(-A, a, -B, b), 3: trig2(-A, a, B, b), 4: trig(A, a, B, b)}[n]
    ok = res
    cands = [(trig(A, a, B, b) if n != 4 else trig(-A, a, -B, b), r"$(-i\,\mathrm{sgn}\,\omega)^2=-1$ なので $H^2=-I$。回数を確かめよう。"),
             (trig2(A, a, -B, b), "これは1回だけ変換したもの。"), (trig2(-A, a, B, b) if n != 3 else trig(-A, a, -B, b), "回数を確かめよう($H^2=-I$, $H^3=-H$, $H^4=I$)。"),
             (trig(-A, a, -B, b) if n != 2 else trig2(-A, a, B, b), "回数を確かめよう。")]
    hn = {2: "H(Hf)", 3: "H(H(Hf))", 4: "H(H(H(Hf)))"}[n]
    return Problem("ヒルベルト変換を繰り返す", "基本",
                   rf"$f(t)={trig(A, a, B, b)}$ のとき ${hn}$ を求めよ。ただし $H[\cos at]=\sin at$, $H[\sin at]=-\cos at$($a>0$)である。",
                   distinct(ok, cands), hints=[r"フーリエ変換では $-i\,\mathrm{sgn}\,\omega$ を掛ける操作。$(-i\,\mathrm{sgn}\,\omega)^2=-1$。"],
                   solution=[r"$H^2=-I$(2回で符号が変わる), $H^3=-H$, $H^4=I$。", rf"よって ${hn}={ok}$。"], meta=dict(A=A, a=a, B=B, b=b, n=n))


# ------------------------------------------------------------
# 標準:ローレンツ型の関数
# ------------------------------------------------------------
def g_lorentz(rng):
    a = rng.choice([1, 2, 3])
    c = rng.choice([1, 2, 3, 6])
    t0 = rng.choice([1, 2, 3, -1, -2])
    val = F(c * t0, a * (t0 * t0 + a * a))            # H[c/(t²+a²)] = c t / (a (t²+a²))
    ok = tex_fr(val)
    cands = [(tex_fr(F(c * t0, t0 * t0 + a * a)), r"表は分子が $a$ の形。$\dfrac{c}{t^2+a^2}=\dfrac ca\cdot\dfrac{a}{t^2+a^2}$ と直す。"),
             (tex_fr(F(c, t0 * t0 + a * a)), "これは $f(t_0)$ の値。"), (tex_fr(-val), "符号を確かめよう($t>0$ で正)。"),
             (tex_fr(F(c * t0 * a, t0 * t0 + a * a)), r"$a$ を掛けるのではなく割る。")]
    return Problem("ローレンツ型の関数のヒルベルト変換", "標準",
                   rf"$f(t)=\dfrac{{{c}}}{{t^2+{a * a}}}$ のヒルベルト変換 $Hf$ の $t={t0}$ での値を求めよ。ただし $H\!\left[\dfrac{{a}}{{t^2+a^2}}\right]=\dfrac{{t}}{{t^2+a^2}}$($a>0$)である。",
                   distinct(ok, cands), hints=[rf"$a={a}$。$f=\dfrac{{{c}}}{{{a}}}\cdot\dfrac{{{a}}}{{t^2+{a * a}}}$。"],
                   solution=[rf"$Hf(t)=\dfrac{{{c}}}{{{a}}}\cdot\dfrac{{t}}{{t^2+{a * a}}}$。", rf"$Hf({t0})={ok}$。"], meta=dict(a=a, c=c, t0=t0, val=val))


# ------------------------------------------------------------
# 標準:解析信号の包絡線
# ------------------------------------------------------------
def g_env(rng):
    A, B = rng.choice([(3, 4), (5, 12), (6, 8), (1, 1), (1, 2), (2, 3), (1, 3), (2, 2), (8, 15), (4, 4)])
    if rng.random() < 0.5:
        A, B = B, A
    B = B * rng.choice([1, -1])
    w = rng.choice([2, 3, 5])
    n = A * A + B * B
    ok = root_tex(n)
    cands = [(str(abs(A) + abs(B)), "振幅は足し算ではない。$|z|=\\sqrt{A^2+B^2}$。"), (str(n), "平方根をとる。"),
             (str(max(abs(A), abs(B))), "2つの成分を合わせた振幅を求める。"), (root_tex(abs(A * A - B * B)) if abs(A) != abs(B) else "0", "$A^2+B^2$(和)の平方根。"),
             (str(abs(abs(A) - abs(B))) if abs(A) != abs(B) else str(2 * abs(A)), "$|z|=\\sqrt{A^2+B^2}$。")]
    f = trig(A, w, B, w)
    return Problem("解析信号と包絡線", "標準",
                   rf"$f(t)={f}$ の解析信号 $z(t)=f(t)+i\,Hf(t)$ の絶対値(包絡線)$|z(t)|$ を求めよ。ただし $H[\cos at]=\sin at$, $H[\sin at]=-\cos at$($a>0$)である。",
                   distinct(ok, cands), hints=[r"$Hf$ を求め, $|z|^2=f^2+(Hf)^2$ を計算する。", r"$z=(A-iB)e^{i\omega t}$ の形になる。"],
                   solution=[rf"$Hf={trig2(A, w, -B, w)}$。", rf"$z=({A}{'-' if B > 0 else '+'}{abs(B) if abs(B) != 1 else ''}i)e^{{{w}it}}$ より $|z|=\sqrt{{{A * A}+{B * B}}}={ok}$。"],
                   meta=dict(A=A, B=B, w=w, n=n))


# ------------------------------------------------------------
# 標準:区間の特性関数
# ------------------------------------------------------------
def log_tex(r):
    """(1/π) log r(r は正の有理数)。r < 1 のときは −log(1/r) の形にする"""
    r = F(r)
    if r == 1:
        return "0"
    if r > 1:
        return rf"\dfrac{{\log {tex_fr(r)}}}{{\pi}}" if r.denominator == 1 else rf"\dfrac{{1}}{{\pi}}\log{tex_fr(r)}"
    q = 1 / r
    return rf"-\dfrac{{\log {tex_fr(q)}}}{{\pi}}" if q.denominator == 1 else rf"-\dfrac{{1}}{{\pi}}\log{tex_fr(q)}"


def g_box(rng):
    a, b = rng.choice([(-1, 1), (0, 1), (0, 2), (-2, 2), (1, 3), (-1, 2)])
    t0 = rng.choice([t for t in [-4, -3, -2, 3, 4, 5, 6] if t < a or t > b])
    r = abs(F(t0 - a, t0 - b))
    if r == 1:
        raise Retry
    ok = log_tex(r)
    cands = [(log_tex(1 / r), r"$\left|\dfrac{t-a}{t-b}\right|$:分子が左端 $a$, 分母が右端 $b$。"),
             (rf"\log {tex_fr(r)}", r"係数 $\dfrac1\pi$ を忘れていないか。"),
             (log_tex(abs(F(t0 - a) * F(t0 - b))) if abs(F(t0 - a) * F(t0 - b)) != 1 else log_tex(F(b - a) + 1), "積ではなく比の対数。"),
             (log_tex(F(b - a)) if b - a != 1 else "0", r"区間の長さではなく, $t$ から両端までの距離の比。"), ("0", "区間の外でも $0$ にはならない。")]
    return Problem("区間の特性関数のヒルベルト変換", "標準",
                   rf"$f=\chi_{{[{a},{b}]}}$(区間 $[{a},{b}]$ で $1$, ほかで $0$)のとき $Hf({t0})$ を求めよ。ただし $H\chi_{{[a,b]}}(t)=\dfrac1\pi\log\left|\dfrac{{t-a}}{{t-b}}\right|$ である。",
                   distinct(ok, cands), hints=[rf"$\left|\dfrac{{{t0}-({a})}}{{{t0}-({b})}}\right|$ を計算する。"],
                   solution=[rf"$\left|\dfrac{{t-a}}{{t-b}}\right|=\left|\dfrac{{{t0 - a}}}{{{t0 - b}}}\right|={tex_fr(r)}$。", rf"$Hf({t0})={ok}$。"], meta=dict(a=a, b=b, t0=t0, r=r))


# ------------------------------------------------------------
# 標準:因果的な系の実部と虚部
# ------------------------------------------------------------
def g_causal(rng):
    a = rng.choice([1, 2, 3])
    c = rng.choice([1, 2, 3, 4])
    w0 = rng.choice([1, 2, 3, -1, -2])
    # h = c e^{−at} u(t):H(ω) = c/(a + iω), X = ca/(ω²+a²), Y = −cω/(ω²+a²)
    val = F(-c * w0, w0 * w0 + a * a)
    ok = tex_fr(val)
    cands = [(tex_fr(-val), r"$Y=-\mathcal H[X]$:マイナスが付く(因果的な系では虚部は $\omega>0$ で負)。"),
             (tex_fr(F(c * a, w0 * w0 + a * a)), "これは実部 $X(\\omega_0)$ の値。"), (tex_fr(F(-c * w0, a * (w0 * w0 + a * a))), r"$X=c\cdot\dfrac{a}{\omega^2+a^2}$ なので, 表がそのまま使える。"),
             (tex_fr(F(-c * w0 * a, w0 * w0 + a * a)), r"$\mathcal H\!\left[\dfrac{a}{\omega^2+a^2}\right]=\dfrac{\omega}{\omega^2+a^2}$:$a$ は掛からない。")]
    return Problem("因果的な系の周波数応答", "標準",
                   rf"因果的な系($t<0$ でインパルス応答が $0$)の周波数応答 $H(\omega)=X(\omega)+iY(\omega)$ は $Y=-\mathcal H[X]$ をみたす($\mathcal H$ はヒルベルト変換)。"
                   rf"実部が $X(\omega)=\dfrac{{{c * a}}}{{\omega^2+{a * a}}}$ のとき, $Y({w0})$ を求めよ。ただし $\mathcal H\!\left[\dfrac{{a}}{{\omega^2+a^2}}\right]=\dfrac{{\omega}}{{\omega^2+a^2}}$($a>0$)である。",
                   distinct(ok, cands), hints=[rf"$X={c}\cdot\dfrac{{{a}}}{{\omega^2+{a * a}}}$ と見る($a={a}$)。"],
                   solution=[rf"$\mathcal H[X]=\dfrac{{{c}\omega}}{{\omega^2+{a * a}}}$ より $Y(\omega)=-\dfrac{{{c}\omega}}{{\omega^2+{a * a}}}$。", rf"$Y({w0})={ok}$。"
                             rf"($h(t)={c}e^{{-{a if a != 1 else ''}t}}$($t\ge0$)の系で, $H(\omega)=\dfrac{{{c}}}{{{a}+i\omega}}$。)"], meta=dict(a=a, c=c, w0=w0, val=val))


# ------------------------------------------------------------
# 誘導
# ------------------------------------------------------------
def guided_analytic(rng):
    A, B = rng.choice([(3, 4), (4, 3), (5, 12), (12, 5), (6, 8), (8, 6)])
    B = B * rng.choice([1, -1])
    w = rng.choice([2, 3, 5])
    f = trig(A, w, B, w)
    Hf = trig2(A, w, -B, w)
    return ExamItem("解析信号と包絡線",
                    rf"ヒルベルト変換 $H$ は線形で, $H[\cos at]=\sin at$, $H[\sin at]=-\cos at$($a>0$)をみたす. $f(t)={f}$ について, 次の問いに答えよ.",
                    [Part(r"$Hf(t)$ を求めよ.", Hf, [r"$\cos\to\sin$, $\sin\to-\cos$ と置きかえる。", rf"$Hf={Hf}$。"]),
                     Part(r"解析信号 $z(t)=f(t)+i\,Hf(t)$ の絶対値 $|z(t)|$ を求めよ.", root_tex(A * A + B * B),
                          [rf"$|z|^2=f^2+(Hf)^2={A * A}+{B * B}$($\cos^2+\sin^2=1$ を使う)。", rf"$|z|={root_tex(A * A + B * B)}$。"])], meta=dict(A=A, B=B, w=w))


def guided_causal(rng):
    a = rng.choice([1, 2, 3])
    c = rng.choice([1, 2, 4])
    at = "" if a == 1 else str(a)
    return ExamItem("因果的な系の周波数応答とヒルベルト変換",
                    rf"インパルス応答が $h(t)={c if c != 1 else ''}e^{{-{at}t}}$($t\ge0$), $h(t)=0$($t<0$)の系を考える. "
                    r"ヒルベルト変換について $\mathcal H\!\left[\dfrac{a}{\omega^2+a^2}\right]=\dfrac{\omega}{\omega^2+a^2}$($a>0$)が成り立つ. 次の問いに答えよ.",
                    [Part(r"周波数応答 $H(\omega)=\displaystyle\int_0^\infty h(t)e^{-i\omega t}dt$ を求め, 実部 $X(\omega)$ と虚部 $Y(\omega)$ に分けよ.",
                          rf"H=\dfrac{{{c}}}{{{a}+i\omega}},\ X=\dfrac{{{c * a}}}{{\omega^2+{a * a}}},\ Y=-\dfrac{{{c}\omega}}{{\omega^2+{a * a}}}",
                          [rf"$\int_0^\infty e^{{-({a}+i\omega)t}}dt=\dfrac{{1}}{{{a}+i\omega}}$。", rf"分母を実数にして $\dfrac{{{c}({a}-i\omega)}}{{\omega^2+{a * a}}}$。"]),
                     Part(r"$\mathcal H[X](\omega)$ を求め, $Y=-\mathcal H[X]$ が成り立つことを確かめよ.", rf"\mathcal H[X]=\dfrac{{{c}\omega}}{{\omega^2+{a * a}}}",
                          [rf"$X={c}\cdot\dfrac{{{a}}}{{\omega^2+{a * a}}}$ より $\mathcal H[X]=\dfrac{{{c}\omega}}{{\omega^2+{a * a}}}$。", r"(1) の $Y$ と比べると $Y=-\mathcal H[X]$。"])],
                    meta=dict(a=a, c=c))


REGISTRY = [
    ProblemType("hb_trig", "三角関数のヒルベルト変換", "基本", g_trig),
    ProblemType("hb_twice", "ヒルベルト変換を繰り返す", "基本", g_twice),
    ProblemType("hb_lorentz", "ローレンツ型の関数のヒルベルト変換", "標準", g_lorentz),
    ProblemType("hb_env", "解析信号と包絡線", "標準", g_env, guided=guided_analytic),
    ProblemType("hb_box", "区間の特性関数のヒルベルト変換", "標準", g_box),
    ProblemType("hb_causal", "因果的な系の周波数応答", "標準", g_causal, guided=guided_causal),
]
GUIDED_NAMES = {"hb_env": "ヒルベルト変換 → 解析信号の包絡線",
                "hb_causal": "周波数応答の実部・虚部 → ヒルベルト変換による関係"}


def tab():
    render("hilbert", REGISTRY, "フーリエ解析 発展 ヒルベルト変換")
