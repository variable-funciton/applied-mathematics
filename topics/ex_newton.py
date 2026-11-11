"""数学IV 偏微分方程式 発展5. ニュートンポテンシャルとポアソン方程式:演習の問題生成

規約:−Δu = f の解 u = Nf。3次元の球対称な関数では Δu = u'' + (2/r)u'。
一様な球(半径 a, 密度 ρ₀):u = ρ₀(3a² − r²)/6 (r ≤ a),  ρ₀a³/(3r) (r ≥ a)。
"""
from fractions import Fraction

from topics.ex_core import Choice, ExamItem, Part, Problem, ProblemType, Retry, render, tex_fr

F = Fraction


def distinct(ok, cands):
    choices = [Choice(ok, True, value=ok)]
    for t, why in cands:
        if all(c.tex != t for c in choices) and len(choices) < 4:
            choices.append(Choice(t, why=why, value=t))
    if len(choices) < 4:
        raise Retry
    return choices


def mono(c, k):
    """c r^k の TeX"""
    c = F(c)
    if c == 0:
        return "0"
    body = "" if k == 0 else ("r" if k == 1 else f"r^{{{k}}}")
    if body == "":
        return tex_fr(c)
    if c == 1:
        return body
    if c == -1:
        return "-" + body
    return tex_fr(c) + body


# ------------------------------------------------------------
# 基本:球対称な関数のラプラシアン
# ------------------------------------------------------------
def g_lap(rng):
    k = rng.choice([-2, -1, 1, 2, 3, 4])
    ok = mono(k * (k + 1), k - 2)
    cands = [(mono(k * (k - 1), k - 2), r"3次元では $\Delta u=u^{\prime\prime}+\dfrac2ru^{\prime}$:$\dfrac2ru^{\prime}$ の項を忘れていないか。"),
             (mono(k * (k + 2), k - 2), r"$u^{\prime\prime}=k(k-1)r^{k-2}$, $\dfrac2ru^{\prime}=2kr^{k-2}$ を足す。"),
             (mono(k * k, k - 2), r"これは2次元の場合($\Delta u=u^{\prime\prime}+\dfrac1ru^{\prime}$)。"),
             (mono(k * (k + 1), k - 1), r"2回微分するので $r$ の指数は $2$ 下がる。")]
    ut = "r" if k == 1 else (rf"\dfrac{{1}}{{r}}" if k == -1 else (rf"\dfrac{{1}}{{r^{-k}}}" if k < 0 else f"r^{k}"))
    return Problem("球対称な関数のラプラシアン", "基本",
                   rf"3次元空間で $r=|x|$ とする。$u={ut}$ のラプラシアン $\Delta u=u^{{\prime\prime}}+\dfrac{{2}}{{r}}u^{{\prime}}$ を求めよ($r\neq0$)。",
                   distinct(ok, cands), hints=[r"$u=r^k$ なら $u^{\prime}=kr^{k-1}$, $u^{\prime\prime}=k(k-1)r^{k-2}$。"],
                   solution=[rf"$\Delta r^k=k(k-1)r^{{k-2}}+2kr^{{k-2}}=k(k+1)r^{{k-2}}$。$k={k}$ として ${ok}$。" +
                             (r"($\frac1r$ は原点以外で調和関数。)" if k == -1 else "")], meta=dict(k=k))


# ------------------------------------------------------------
# 基本:球の外のポテンシャル(殻の定理), 標準:球の内部
# ------------------------------------------------------------
def ball(rng):
    a = rng.choice([1, 2, 3])
    rho = rng.choice([3, 6, 12])
    return a, rho


def g_outside(rng):
    a, rho = ball(rng)
    r = a * rng.choice([2, 3])
    val = F(rho * a ** 3, 3 * r)
    ok = tex_fr(val)
    cands = [(tex_fr(F(rho * (3 * a * a - r * r), 6)), r"$r>a$ は球の外。外では全質量が中心にある点と同じ $\dfrac{\rho_0a^3}{3r}$。"),
             (tex_fr(F(rho * a ** 3, 3 * r * r)), r"$\dfrac{1}{r^2}$ は力の大きさ。ポテンシャルは $\dfrac1r$ に比例する。"),
             (tex_fr(F(rho * a ** 3, r)), r"全質量は $\dfrac{4\pi a^3}{3}\rho_0$。$\dfrac{1}{4\pi r}$ を掛けると $\dfrac{\rho_0a^3}{3r}$。"),
             (tex_fr(F(rho * a * a, 3 * r)), r"体積は $a^3$ に比例する。")]
    return Problem("球の外のポテンシャル", "基本",
                   rf"半径 $a={a}$, 密度 $\rho_0={rho}$ の一様な球のニュートンポテンシャル $u=Nf$($-\Delta u=f$)の, 中心からの距離 $r={r}$ の点での値を求めよ。",
                   distinct(ok, cands), hints=[r"球の外では, 全質量 $M=\dfrac{4\pi a^3}{3}\rho_0$ が中心に集まった点のポテンシャル $\dfrac{M}{4\pi r}$ と同じ(殻の定理)。"],
                   solution=[rf"$u(r)=\dfrac{{\rho_0a^3}}{{3r}}=\dfrac{{{rho}\times{a ** 3}}}{{3\times{r}}}={ok}$。"], meta=dict(a=a, rho=rho, r=r, val=val))


def g_inside(rng):
    a, rho = ball(rng)
    r = rng.choice([0] + list(range(1, a)))
    val = F(rho * (3 * a * a - r * r), 6)
    ok = tex_fr(val)
    cands = [(tex_fr(F(rho * a * a, 3)), r"これは表面($r=a$)での値。内部では $\dfrac{\rho_0}{6}(3a^2-r^2)$。"),
             (tex_fr(F(rho * (a * a - r * r), 6)), r"表面で外側の値 $\dfrac{\rho_0a^2}{3}$ とつながるように定数を決める:$3a^2-r^2$。"),
             (tex_fr(F(rho * (3 * a * a + r * r), 6)), r"$-\Delta u=\rho_0>0$ なので, 中心から離れるとポテンシャルは下がる。"),
             (tex_fr(F(rho * (3 * a * a - r * r), 3)), r"$u^{\prime\prime}+\dfrac2ru^{\prime}=-\rho_0$ に $u=c-kr^2$ を代入すると $6k=\rho_0$。")]
    return Problem("球の内部のポテンシャル", "標準",
                   rf"半径 $a={a}$, 密度 $\rho_0={rho}$ の一様な球のニュートンポテンシャル $u=Nf$($-\Delta u=f$)の, "
                   rf"{'中心' if r == 0 else rf'中心からの距離 $r={r}$ の点'}での値を求めよ。",
                   distinct(ok, cands), hints=[r"内部では $u^{\prime\prime}+\dfrac2ru^{\prime}=-\rho_0$。$u=c-\dfrac{\rho_0}{6}r^2$ の形。",
                                               r"表面 $r=a$ で外側の $\dfrac{\rho_0a^3}{3r}$ と値が一致するように $c$ を決める。"],
                   solution=[r"$u(r)=\dfrac{\rho_0}{6}\left(3a^2-r^2\right)$($r\le a$)。",
                             rf"$u({r})=\dfrac{{{rho}}}{{6}}\left({3 * a * a}-{r * r}\right)={ok}$。"], meta=dict(a=a, rho=rho, r=r, val=val))


# ------------------------------------------------------------
# 基本:力の大きさの比
# ------------------------------------------------------------
def g_force(rng):
    kind = rng.choice(["in", "out"])
    k = rng.choice([2, 3, 4])
    if kind == "in":
        ok = tex_fr(F(1, k))
        cands = [(tex_fr(F(1, k * k)), r"球の内部では力は $r$ に比例する(内側の質量 $\propto r^3$ を $r^2$ で割る)。"), (str(k), "中心に近いほど内側の質量が少ないので, 力は小さくなる。"),
                 (tex_fr(F(1, k ** 3)), r"内側の質量は $r^3$ に比例し, それを $r^2$ で割る。"), (str(k * k), r"内部では力は $r$ に比例する。")]
        q = rf"一様な球(半径 $a$)の内部で, 中心からの距離が $\dfrac{{a}}{{{k}}}$ の点での引力の大きさは, 表面での引力の大きさの何倍か。"
        sol = [r"内部では $-u^{\prime}(r)=\dfrac{\rho_0}{3}r$ で $r$ に比例する。", rf"よって ${ok}$ 倍。"]
    else:
        ok = tex_fr(F(1, k * k))
        cands = [(tex_fr(F(1, k)), r"球の外では力は $\dfrac{1}{r^2}$ に比例する。"), (tex_fr(F(1, k ** 3)), r"力は距離の2乗に反比例する。"),
                 (str(k * k), "遠いほど力は小さくなる。"), (str(k), r"力は $\dfrac{1}{r^2}$ に比例する。")]
        q = rf"一様な球(半径 $a$)の外で, 中心からの距離が ${k}a$ の点での引力の大きさは, 表面での引力の大きさの何倍か。"
        sol = [r"外では $-u^{\prime}(r)=\dfrac{\rho_0a^3}{3r^2}$ で $\dfrac{1}{r^2}$ に比例する。", rf"よって ${ok}$ 倍。"]
    return Problem("引力の大きさ", "基本", q, distinct(ok, cands),
                   hints=[r"力の大きさは $\dfrac{(\text{半径 }r\text{ の内側の質量})}{4\pi r^2}$(外側の殻は力を及ぼさない)。"], solution=sol, meta=dict(kind=kind, k=k))


# ------------------------------------------------------------
# 基本:1次元のポテンシャル
# ------------------------------------------------------------
def g_one(rng):
    a, c = rng.choice([1, 2, 3]), rng.choice([1, 2, 4])
    x0 = rng.choice([0, a + 1, 2 * a])
    val = F(-c * a * a, 2) if x0 == 0 else F(-c * a * x0)
    ok = tex_fr(val)
    alt = F(-c * a * x0) if x0 == 0 else F(-c * a * a, 2)
    cands = [(tex_fr(-val), r"基本解は $-\dfrac{|x|}{2}$:負の符号がつく。"), (tex_fr(2 * val), r"$\dfrac12$ を忘れていないか。"),
             (tex_fr(val / 2) if x0 else tex_fr(F(-c * a * a)), r"$\int_{-a}^{a}|x_0-y|\,dy$ を計算し直そう。"),
             (tex_fr(alt) if alt != 0 else tex_fr(F(-c * a)), r"$x_0$ が区間の中か外かで $|x_0-y|$ の積分が変わる。")]
    ft = (str(c) if c > 1 else "") + rf"\chi_{{[-{a},{a}]}}"
    integ = rf"\int_{{-{a}}}^{{{a}}}|y|\,dy={a * a}" if x0 == 0 else rf"\int_{{-{a}}}^{{{a}}}({x0}-y)\,dy={2 * a * x0}"
    return Problem("1次元のポテンシャル", "基本",
                   rf"1次元で $f={ft}$(区間 $[-{a},{a}]$ で ${c}$, それ以外で $0$)とする。$-u^{{\prime\prime}}=f$ の解 "
                   rf"$u(x)=-\dfrac12\displaystyle\int_{{-\infty}}^{{\infty}}|x-y|f(y)\,dy$ の $x={x0}$ での値を求めよ。",
                   distinct(ok, cands), hints=[r"$f$ が $0$ でない区間だけで積分する。", r"$x$ が区間の外にあれば, $|x-y|$ の絶対値はそのまま外せる。"],
                   solution=[rf"${integ}$ より $u({x0})=-\dfrac12\times{c}\times{a * a if x0 == 0 else 2 * a * x0}={ok}$。"], meta=dict(a=a, c=c, x0=x0, val=val))


# ------------------------------------------------------------
# 標準:ハーディ・リトルウッド・ソボレフの指数
# ------------------------------------------------------------
def g_hls(rng):
    p = rng.choice([F(6, 5), F(9, 7), F(12, 11), F(5, 4), F(4, 3), F(15, 13)])
    inv = 1 / p - F(2, 3)
    if inv <= 0:
        raise Retry
    q = 1 / inv
    ok = tex_fr(q)
    cands = [(tex_fr(1 / (1 / p - F(1, 3))), r"$\alpha=2$, $n=3$ なので $\dfrac\alpha n=\dfrac23$。"),
             (tex_fr(1 / (1 / p + F(2, 3))), r"ポテンシャルをとると積分できる指数は**大きく**なる:$\dfrac1q=\dfrac1p-\dfrac23$。"),
             (tex_fr(p / (p - 1)), r"これは共役指数。ここでは $\dfrac1q=\dfrac1p-\dfrac\alpha n$ を使う。"),
             (tex_fr(1 / abs(1 / p - 2)) if 1 / p != 2 else "1", r"$n=3$ で割るのを忘れていないか。")]
    return Problem("ハーディ・リトルウッド・ソボレフの指数", "標準",
                   rf"3次元のニュートンポテンシャル $N=I_2$ について, $\|Nf\|_q\le C\|f\|_p$ が成り立つ指数の関係は $\dfrac1q=\dfrac1p-\dfrac23$ である。"
                   rf"$p={tex_fr(p)}$ のときの $q$ を求めよ。",
                   distinct(ok, cands), hints=[r"リースポテンシャル $I_\alpha$ では $\dfrac1q=\dfrac1p-\dfrac\alpha n$。ニュートンポテンシャルは $n=3$, $\alpha=2$。"],
                   solution=[rf"$\dfrac1q={tex_fr(1 / p)}-\dfrac23={tex_fr(inv)}$ より $q={ok}$。"], meta=dict(p=p, q=q))


# ------------------------------------------------------------
# 誘導
# ------------------------------------------------------------
def guided_ball(rng):
    a, rho = ball(rng)
    return ExamItem("一様な球のポテンシャル",
                    rf"3次元空間で, 半径 $a={a}$ の球の内部で $f={rho}$, 外部で $f=0$ とする. $r=|x|$ だけの関数 $u(r)$ が $-\Delta u=f$, "
                    rf"すなわち $u^{{\prime\prime}}+\dfrac{{2}}{{r}}u^{{\prime}}=-f$ をみたし, $r\to\infty$ で $u\to0$ となるとする. 次の問いに答えよ.",
                    [Part(rf"球の外($r>{a}$)では $u=\dfrac{{C}}{{r}}$ の形である. 球の内部($r<{a}$)で $u=c-kr^2$ の形の解の定数 $k$ を求めよ.",
                          rf"k={tex_fr(F(rho, 6))}",
                          [rf"$u^{{\prime}}=-2kr$, $u^{{\prime\prime}}=-2k$ より $-2k-4k=-{rho}$, $k={tex_fr(F(rho, 6))}$。"]),
                     Part(rf"$r={a}$ で $u$ と $u^{{\prime}}$ が連続になるように $C$ と $c$ を定めよ.",
                          rf"C={tex_fr(F(rho * a ** 3, 3))},\ c={tex_fr(F(rho * a * a, 2))}",
                          [rf"$u^{{\prime}}$ の連続:$-2k\cdot{a}=-\dfrac{{C}}{{{a * a}}}$ より $C={tex_fr(F(rho * a ** 3, 3))}$。",
                           rf"$u$ の連続:$c-k\cdot{a * a}=\dfrac{{C}}{{{a}}}$ より $c={tex_fr(F(rho * a * a, 2))}$。"])], meta=dict(a=a, rho=rho))


def guided_fund(rng):
    R = rng.choice([1, 2, 3])
    return ExamItem("基本解と流束",
                    r"3次元空間で $r=|x|$, $\Gamma=\dfrac{1}{4\pi r}$ とする. 次の問いに答えよ.",
                    [Part(r"$r\neq0$ で $\Delta\Gamma=\Gamma^{\prime\prime}+\dfrac2r\Gamma^{\prime}$ を計算せよ.", r"\Delta\Gamma=0",
                          [r"$\Gamma^{\prime}=-\dfrac{1}{4\pi r^2}$, $\Gamma^{\prime\prime}=\dfrac{1}{2\pi r^3}$ より $\Delta\Gamma=\dfrac{1}{2\pi r^3}-\dfrac{1}{2\pi r^3}=0$。"]),
                     Part(rf"原点を中心とする半径 ${R}$ の球面 $S$(外向き単位法線 $\boldsymbol n$)について, 流束 $\displaystyle\int_S\nabla\Gamma\cdot\boldsymbol n\,dS$ を求めよ.", "-1",
                          [rf"球面上で $\nabla\Gamma\cdot\boldsymbol n=\Gamma^{{\prime}}({R})=-\dfrac{{1}}{{4\pi\cdot{R * R}}}$。",
                           rf"球面の面積 $4\pi\cdot{R * R}$ を掛けて $-1$(半径によらない)。"])], meta=dict(R=R))


REGISTRY = [
    ProblemType("np_lap", "球対称な関数のラプラシアン", "基本", g_lap, guided=guided_fund),
    ProblemType("np_out", "球の外のポテンシャル", "基本", g_outside),
    ProblemType("np_force", "引力の大きさ", "基本", g_force),
    ProblemType("np_one", "1次元のポテンシャル", "基本", g_one),
    ProblemType("np_in", "球の内部のポテンシャル", "標準", g_inside, guided=guided_ball),
    ProblemType("np_hls", "ハーディ・リトルウッド・ソボレフの指数", "標準", g_hls),
]
GUIDED_NAMES = {"np_lap": "基本解のラプラシアン → 球面を通る流束",
                "np_in": "球の内部の解 → 表面で外部の解とつなぐ"}


def tab():
    render("newton", REGISTRY, "偏微分方程式 発展5. ニュートンポテンシャルとポアソン方程式")
