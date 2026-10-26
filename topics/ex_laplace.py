"""数学IV ラプラス変換:演習の問題生成

第1章 定義と性質:公式、第1・第2移動法則、t f(t)、逆変換(部分分数・平方完成)
第2章 応用      :微分方程式、合成積、伝達関数とインパルス応答

各選択肢の value には「t の関数」または「s の関数」(numpy で計算できる)を入れておき、
自動テストで数値積分・数値解と照合する。
"""
import math
from fractions import Fraction

import numpy as np

from topics.ex_core import Choice, ExamItem, Part, Problem, ProblemType, Retry, render, tex_fr

F = Fraction


# ------------------------------------------------------------
# 文字列の道具
# ------------------------------------------------------------
def sh(a, var="s"):
    """s − a の形(a = 0 なら s)"""
    a = F(a)
    if a == 0:
        return var
    return f"{var}-{tex_fr(a)}" if a > 0 else f"{var}+{tex_fr(-a)}"


def expt(a):
    """e^{at}(a = 0 なら空)"""
    a = F(a)
    if a == 0:
        return ""
    if a == 1:
        return "e^{t}"
    if a == -1:
        return "e^{-t}"
    return f"e^{{{tex_fr(a)}t}}"


def signed_sum(terms):
    """[(係数, 本体), ...] → 3e^{t}-2\\sin 2t。本体が '' なら定数"""
    out = ""
    for c, body in terms:
        c = F(c)
        if c == 0:
            continue
        sgn = "-" if c < 0 else ("+" if out else "")
        m = abs(c)
        if body == "":
            coef = tex_fr(m)
        elif m == 1:
            coef = ""
        elif m.denominator != 1:
            coef = rf"\frac{{{m.numerator}}}{{{m.denominator}}}"
        else:
            coef = str(m.numerator)
        out += sgn + coef + body
    return out or "0"


def frac_sum(terms):
    """[(係数, 分子の本体, 分母), ...] → \\frac{3}{s-2}-\\frac{2s}{s^2+4}"""
    out = ""
    for c, numb, den in terms:
        c = F(c)
        if c == 0:
            continue
        sgn = "-" if c < 0 else ("+" if out else "")
        m = abs(c)
        if numb == "":
            num = tex_fr(m) if m.denominator == 1 else str(m.numerator)
            den2 = den if m.denominator == 1 else f"{m.denominator}({den})"
        else:
            num = (str(m.numerator) if m.numerator != 1 else "") + numb
            den2 = den if m.denominator == 1 else f"{m.denominator}({den})"
        out += sgn + rf"\frac{{{num}}}{{{den2}}}"
    return out or "0"


def inv(d):
    """1/d を係数として前に置く形(1 → 空, -1 → -)"""
    if d == 1:
        return ""
    if d == -1:
        return "-"
    return rf"\frac{{1}}{{{d}}}" if d > 0 else rf"-\frac{{1}}{{{-d}}}"


def _top_split(inner):
    """かっこ・波かっこの外にある + / - の位置"""
    depth, pos = 0, []
    for i, ch in enumerate(inner):
        depth += ch in "{("
        depth -= ch in "})"
        if depth == 0 and i > 0 and ch in "+-":
            pos.append(i)
    return pos


def paren(d, inner):
    """inv(d)(inner)。d = ±1 ならかっこを外す。
    d < 0 で inner が X-Y の形なら、-(X-Y)/|d| ではなく (Y-X)/|d| と書く"""
    pos = _top_split(inner)
    if d < 0 and len(pos) == 1 and inner[pos[0]] == "-" and not inner.startswith("-"):
        d, inner = -d, inner[pos[0] + 1:] + "-" + inner[:pos[0]]
    if d == 1:
        return inner
    if d == -1:
        return r"-\left(" + inner + r"\right)"
    return inv(d) + r"\left(" + inner + r"\right)"


def sin_t(w):
    return rf"\sin {'' if w == 1 else w}t"


def cos_t(w):
    return rf"\cos {'' if w == 1 else w}t"


def tpow(n):
    return "t" if n == 1 else f"t^{{{n}}}"


# ============================================================
# 第1章 基本1:公式によるラプラス変換
# ============================================================
def g_table(rng):
    kinds = [rng.choice(["const", "tn"]), "exp", rng.choice(["sin", "cos"])]
    rng.shuffle(kinds)
    terms_t, terms_s, bad_sincos, bad_exp, bad_pow = [], [], [], [], []
    fns, Fs, Fb1, Fb2, Fb3 = [], [], [], [], []
    for k in kinds:
        c = rng.choice([-3, -2, -1, 1, 2, 3, 4])
        if k == "const":
            terms_t.append((c, ""))
            ts = [(c, "", "s")]
            terms_s += ts; bad_sincos += ts; bad_exp += ts; bad_pow += [(c, "", "s^{2}")]
            fns.append(lambda t, c=c: c + 0 * t)
            Fs.append(lambda s, c=c: c / s); Fb3.append(lambda s, c=c: c / s ** 2)
            Fb1.append(lambda s, c=c: c / s); Fb2.append(lambda s, c=c: c / s)
        elif k == "tn":
            n = rng.choice([1, 2, 3])
            fac = math.factorial(n)
            terms_t.append((c, tpow(n)))
            ts = [(c * fac, "", f"s^{{{n + 1}}}")]
            terms_s += ts; bad_sincos += ts; bad_exp += ts; bad_pow += [(c * fac, "", f"s^{{{n}}}" if n > 1 else "s")]
            fns.append(lambda t, c=c, n=n: c * t ** n)
            Fs.append(lambda s, c=c, n=n, f=fac: c * f / s ** (n + 1))
            Fb3.append(lambda s, c=c, n=n, f=fac: c * f / s ** n)
            Fb1.append(Fs[-1]); Fb2.append(Fs[-1])
        elif k == "exp":
            a = rng.choice([-3, -2, -1, 1, 2, 3])
            terms_t.append((c, expt(a)))
            ts = [(c, "", sh(a))]
            terms_s += ts; bad_sincos += ts; bad_pow += ts; bad_exp += [(c, "", sh(-a))]
            fns.append(lambda t, c=c, a=a: c * np.exp(a * t))
            Fs.append(lambda s, c=c, a=a: c / (s - a)); Fb2.append(lambda s, c=c, a=a: c / (s + a))
            Fb1.append(Fs[-1]); Fb3.append(Fs[-1])
        else:
            w = rng.choice([1, 2, 3])
            den = f"s^{{2}}+{w * w}"
            if k == "sin":
                terms_t.append((c, sin_t(w)))
                terms_s.append((c * w, "", den)); bad_sincos.append((c, "s", den))
                fns.append(lambda t, c=c, w=w: c * np.sin(w * t))
                Fs.append(lambda s, c=c, w=w: c * w / (s * s + w * w)); Fb1.append(lambda s, c=c, w=w: c * s / (s * s + w * w))
            else:
                terms_t.append((c, cos_t(w)))
                terms_s.append((c, "s", den)); bad_sincos.append((c * w, "", den))
                fns.append(lambda t, c=c, w=w: c * np.cos(w * t))
                Fs.append(lambda s, c=c, w=w: c * s / (s * s + w * w)); Fb1.append(lambda s, c=c, w=w: c * w / (s * s + w * w))
            bad_exp.append(terms_s[-1]); bad_pow.append(terms_s[-1])
            Fb2.append(Fs[-1]); Fb3.append(Fs[-1])

    def tot(fl):
        return lambda s: sum(f(s) for f in fl)
    ft = signed_sum(terms_t)
    choices = [Choice(frac_sum(terms_s), True, value=tot(Fs)),
               Choice(frac_sum(bad_sincos), why=r"$\mathcal L[\sin\omega t]=\frac{\omega}{s^2+\omega^2}$, $\mathcal L[\cos\omega t]=\frac{s}{s^2+\omega^2}$。分子を取り違えていないか。", value=tot(Fb1)),
               Choice(frac_sum(bad_exp), why=r"$\mathcal L[e^{at}]=\frac{1}{s-a}$。分母の符号を確かめよう。", value=tot(Fb2)),
               Choice(frac_sum(bad_pow), why=r"$\mathcal L[t^n]=\frac{n!}{s^{n+1}}$, $\mathcal L[1]=\frac1s$。$s$ の指数を確かめよう。", value=tot(Fb3))]
    return Problem(
        "公式によるラプラス変換", "基本",
        rf"$f(t)={ft}$ のラプラス変換 $F(s)=\mathcal L[f(t)]$ を求めよ。",
        choices,
        hints=[r"ラプラス変換は線形:$\mathcal L[af+bg]=a\mathcal L[f]+b\mathcal L[g]$。項ごとに公式を使う。",
               r"$\mathcal L[1]=\frac1s,\ \mathcal L[t^n]=\frac{n!}{s^{n+1}},\ \mathcal L[e^{at}]=\frac{1}{s-a},\ "
               r"\mathcal L[\sin\omega t]=\frac{\omega}{s^2+\omega^2},\ \mathcal L[\cos\omega t]=\frac{s}{s^2+\omega^2}$"],
        solution=[r"線形性より、項ごとに公式を使う。", rf"$$F(s)={frac_sum(terms_s)}$$"],
        meta=dict(f=lambda t: sum(f(t) for f in fns)))


# ============================================================
# 第1章 基本2:第1移動法則
# ============================================================
def g_shift1(rng):
    a = rng.choice([-3, -2, -1, 1, 2, 3])
    kind = rng.choice(["cos", "sin", "tn"])
    if kind == "tn":
        n = rng.choice([1, 2, 3])
        fac = math.factorial(n)
        ft = tpow(n) + expt(a)
        ok = rf"\frac{{{fac}}}{{({sh(a)})^{{{n + 1}}}}}"
        b1 = rf"\frac{{{fac}}}{{({sh(-a)})^{{{n + 1}}}}}"
        b2 = rf"\frac{{{fac}}}{{({sh(a)})^{{{n}}}}}" if n > 1 else rf"\frac{{1}}{{{sh(a)}}}"
        b3 = rf"\frac{{{fac}}}{{s^{{{n + 1}}}}}"
        vals = [lambda s: fac / (s - a) ** (n + 1), lambda s: fac / (s + a) ** (n + 1),
                lambda s: fac / (s - a) ** n, lambda s: fac / s ** (n + 1)]
        whys = [r"$e^{at}$ をかけると $s$ が $s-a$ にずれる(第1移動法則)。符号を確かめよう。",
                r"$\mathcal L[t^n]=\frac{n!}{s^{n+1}}$。指数は $n+1$。",
                r"$e^{at}$ をかけた効果(第1移動法則)を入れ忘れていないか。"]
        f = lambda t: t ** n * np.exp(a * t)  # noqa: E731
        base = rf"\mathcal L[{tpow(n)}]=\frac{{{fac}}}{{s^{{{n + 1}}}}}"
    else:
        w = rng.choice([1, 2, 3])
        den = rf"({sh(a)})^2+{w * w}"
        denb = rf"({sh(-a)})^2+{w * w}"
        if kind == "cos":
            ft = expt(a) + cos_t(w)
            ok = rf"\frac{{{sh(a)}}}{{{den}}}"
            b1 = rf"\frac{{{sh(-a)}}}{{{denb}}}"
            b2 = rf"\frac{{s}}{{{den}}}"
            b3 = rf"\frac{{{w}}}{{{den}}}"
            vals = [lambda s: (s - a) / ((s - a) ** 2 + w * w), lambda s: (s + a) / ((s + a) ** 2 + w * w),
                    lambda s: s / ((s - a) ** 2 + w * w), lambda s: w / ((s - a) ** 2 + w * w)]
            whys = [r"$e^{at}$ をかけると $s$ が $s-a$ にずれる。符号を確かめよう。",
                    r"分子の $s$ も $s-a$ に置きかえる(すべての $s$ をずらす)。",
                    r"$\cos$ の変換の分子は $s$($\omega$ は $\sin$ のとき)。"]
            f = lambda t: np.exp(a * t) * np.cos(w * t)  # noqa: E731
            base = rf"\mathcal L[{cos_t(w)}]=\frac{{s}}{{s^2+{w * w}}}"
        else:
            ft = expt(a) + sin_t(w)
            ok = rf"\frac{{{w}}}{{{den}}}"
            b1 = rf"\frac{{{w}}}{{{denb}}}"
            b2 = rf"\frac{{{sh(a)}}}{{{den}}}"
            b3 = rf"\frac{{{w}}}{{s^2+{w * w}}}"
            vals = [lambda s: w / ((s - a) ** 2 + w * w), lambda s: w / ((s + a) ** 2 + w * w),
                    lambda s: (s - a) / ((s - a) ** 2 + w * w), lambda s: w / (s * s + w * w)]
            whys = [r"$e^{at}$ をかけると $s$ が $s-a$ にずれる。符号を確かめよう。",
                    r"$\sin$ の変換の分子は $\omega$($s$ は $\cos$ のとき)。",
                    r"$e^{at}$ をかけた効果(第1移動法則)を入れ忘れていないか。"]
            f = lambda t: np.exp(a * t) * np.sin(w * t)  # noqa: E731
            base = rf"\mathcal L[{sin_t(w)}]=\frac{{{w}}}{{s^2+{w * w}}}"
    choices = [Choice(ok, True, value=vals[0])] + [Choice(t, why=y, value=v) for t, y, v in zip((b1, b2, b3), whys, vals[1:])]
    return Problem(
        "第1移動法則", "基本",
        rf"$f(t)={ft}$ のラプラス変換を求めよ。",
        choices,
        hints=[r"第1移動法則:$\mathcal L[e^{at}g(t)]=G(s-a)$($G=\mathcal L[g]$)。",
               r"まず $e^{at}$ を除いた部分を変換し、$s$ を $s-a$ に置きかえる。"],
        solution=[rf"$${base}$$", rf"第1移動法則より $s$ を ${sh(a)}$ に置きかえて", rf"$$\mathcal L[f(t)]={ok}$$"],
        meta=dict(f=f, sig=a))


# ============================================================
# 第1章 基本3:第2移動法則
# ============================================================
def g_shift2(rng):
    tau = rng.choice([1, 2, 3])
    kind = rng.choice(["one", "t", "t2", "exp"])
    if kind == "one":
        g, G, gt = (lambda u: 1 + 0 * u), (lambda s: 1 / s), "1"
        Gt = r"\frac{1}{s}"
        ft = rf"U(t-{tau})"
    elif kind == "t":
        g, G, gt = (lambda u: u), (lambda s: 1 / s ** 2), "t"
        Gt = r"\frac{1}{s^2}"
        ft = rf"(t-{tau})\,U(t-{tau})"
    elif kind == "t2":
        g, G, gt = (lambda u: u * u), (lambda s: 2 / s ** 3), "t^2"
        Gt = r"\frac{2}{s^3}"
        ft = rf"(t-{tau})^2\,U(t-{tau})"
    else:
        a = rng.choice([-2, -1, 1, 2])
        g, G, gt = (lambda u, a=a: np.exp(a * u)), (lambda s, a=a: 1 / (s - a)), expt(a)
        Gt = rf"\frac{{1}}{{{sh(a)}}}"
        ft = rf"e^{{{tex_fr(a)}(t-{tau})}}\,U(t-{tau})".replace("e^{1(", "e^{(").replace("e^{-1(", "e^{-(")
    ok = rf"e^{{-{tau}s}}{Gt}"
    choices = [Choice(ok, True, value=lambda s: np.exp(-tau * s) * G(s)),
               Choice(rf"e^{{{tau}s}}{Gt}", why="$t$ 軸方向に $\\tau$ 遅らせると $e^{-\\tau s}$ がかかる。指数の符号を確かめよう。",
                      value=lambda s: np.exp(tau * s) * G(s)),
               Choice(Gt, why="$U(t-\\tau)$ によって $\\tau$ だけ遅れた効果 $e^{-\\tau s}$ を入れ忘れていないか。", value=G),
               Choice(rf"e^{{-{tau}s}}" + (r"\frac{1}{s^2}" if kind != "t" else r"\frac{1}{s}"),
                      why="遅らせる前の関数 $g(t)$ の変換 $G(s)$ を確かめよう。",
                      value=(lambda s: np.exp(-tau * s) / s ** 2) if kind != "t" else (lambda s: np.exp(-tau * s) / s))]
    return Problem(
        "第2移動法則", "基本",
        rf"$U(t)$ を単位ステップ関数とする. $f(t)={ft}$ のラプラス変換を求めよ。",
        choices,
        hints=[r"第2移動法則:$\mathcal L[g(t-\tau)U(t-\tau)]=e^{-\tau s}G(s)$。",
               r"$f(t)$ は $g(t)$ を $t$ 軸方向に $\tau$ だけ遅らせたもの。まず $g$ を見つける。"],
        solution=[rf"$f(t)=g(t-{tau})U(t-{tau})$, $g(t)={gt}$ で $G(s)={Gt}$", rf"$$\mathcal L[f(t)]=e^{{-{tau}s}}G(s)={ok}$$"],
        meta=dict(f=lambda t: np.where(t >= tau, g(np.maximum(t - tau, 0)), 0.0), sig=2))


# ============================================================
# 第1章 標準1:t f(t) の変換(像関数の微分)
# ============================================================
def g_tfun(rng):
    w = rng.choice([1, 2, 3])
    kind = rng.choice(["tsin", "tcos", "texp"])
    if kind == "tsin":
        ft, ok = "t" + sin_t(w), rf"\frac{{{2 * w}s}}{{(s^2+{w * w})^2}}"
        v = lambda s: 2 * w * s / (s * s + w * w) ** 2  # noqa: E731
        bads = [(rf"-\frac{{{2 * w}s}}{{(s^2+{w * w})^2}}", "$\\mathcal L[tf(t)]=-F'(s)$。マイナスを忘れていないか。", lambda s: -2 * w * s / (s * s + w * w) ** 2),
                (rf"\frac{{{2 * w}s}}{{s^2+{w * w}}}", "$\\frac{1}{s^2+\\omega^2}$ を微分すると分母は2乗になる。", lambda s: 2 * w * s / (s * s + w * w)),
                (rf"\frac{{{w}}}{{s^2(s^2+{w * w})}}", "$t$ をかけることは $\\frac1s$ をかけることではない。$-\\frac{d}{ds}$ を使う。", lambda s: w / (s * s * (s * s + w * w)))]
        f = lambda t: t * np.sin(w * t)  # noqa: E731
        Ft, dF = rf"\frac{{{w}}}{{s^2+{w * w}}}", rf"-\frac{{{2 * w}s}}{{(s^2+{w * w})^2}}"
    elif kind == "tcos":
        ft, ok = "t" + cos_t(w), rf"\frac{{s^2-{w * w}}}{{(s^2+{w * w})^2}}"
        v = lambda s: (s * s - w * w) / (s * s + w * w) ** 2  # noqa: E731
        bads = [(rf"\frac{{{w * w}-s^2}}{{(s^2+{w * w})^2}}", "$\\mathcal L[tf(t)]=-F'(s)$。マイナスを忘れていないか。", lambda s: (w * w - s * s) / (s * s + w * w) ** 2),
                (rf"\frac{{s^2+{w * w}}}{{(s^2+{w * w})^2}}", "商の微分 $\\left(\\frac{u}{v}\\right)'=\\frac{u'v-uv'}{v^2}$ の計算を確かめよう。", lambda s: 1 / (s * s + w * w)),
                (rf"\frac{{2s}}{{(s^2+{w * w})^2}}", "$\\frac{s}{s^2+\\omega^2}$ を正しく $s$ で微分しよう(商の微分)。", lambda s: 2 * s / (s * s + w * w) ** 2)]
        f = lambda t: t * np.cos(w * t)  # noqa: E731
        Ft, dF = rf"\frac{{s}}{{s^2+{w * w}}}", rf"\frac{{{w * w}-s^2}}{{(s^2+{w * w})^2}}"
    else:
        a = rng.choice([-3, -2, -1, 1, 2, 3])
        ft, ok = "t" + expt(a), rf"\frac{{1}}{{({sh(a)})^2}}"
        v = lambda s: 1 / (s - a) ** 2  # noqa: E731
        bads = [(rf"-\frac{{1}}{{({sh(a)})^2}}", "$\\mathcal L[tf(t)]=-F'(s)$。マイナスを忘れていないか。", lambda s: -1 / (s - a) ** 2),
                (rf"\frac{{1}}{{({sh(-a)})^2}}", "$\\mathcal L[e^{at}]=\\frac{1}{s-a}$。符号を確かめよう。", lambda s: 1 / (s + a) ** 2),
                (rf"\frac{{1}}{{s({sh(a)})}}", "$t$ をかけることは $\\frac1s$ をかけることではない。", lambda s: 1 / (s * (s - a)))]
        f = lambda t: t * np.exp(a * t)  # noqa: E731
        Ft, dF = rf"\frac{{1}}{{{sh(a)}}}", rf"-\frac{{1}}{{({sh(a)})^2}}"
    choices = [Choice(ok, True, value=v)] + [Choice(t, why=y, value=vv) for t, y, vv in bads]
    return Problem(
        "像関数の微分法則", "標準",
        rf"$f(t)={ft}$ のラプラス変換を求めよ。",
        choices,
        hints=[r"$\mathcal L[t\,g(t)]=-\dfrac{d}{ds}G(s)$($G=\mathcal L[g]$)。",
               r"$t$ を除いた部分 $g(t)$ を変換してから、$s$ で微分して $-1$ 倍する。"],
        solution=[rf"$G(s)={Ft}$", rf"$G'(s)={dF}$", rf"$$\mathcal L[f(t)]=-G'(s)={ok}$$"],
        meta=dict(f=f, sig=3))


# ============================================================
# 第1章 基本4:逆変換(部分分数)
# ============================================================
def pf_parts(rng):
    r1, r2 = rng.sample([-4, -3, -2, -1, 1, 2, 3], 2)
    A, B = rng.choice([-3, -2, -1, 1, 2, 3, 4]), rng.choice([-3, -2, -1, 1, 2, 3])
    # F = A/(s-r1) + B/(s-r2) = ((A+B)s - (A r2 + B r1)) / ((s-r1)(s-r2))
    p, q = A + B, -(A * r2 + B * r1)
    if p == 0 and q == 0:
        raise Retry
    num = signed_sum([(p, "s"), (q, "")])
    den = rf"({sh(r1)})({sh(r2)})"
    return r1, r2, A, B, num, den


def g_inv_pf(rng):
    r1, r2, A, B, num, den = pf_parts(rng)
    ok = signed_sum([(A, expt(r1)), (B, expt(r2))])

    def val(a, ra, b, rb):
        return lambda t: a * np.exp(ra * t) + b * np.exp(rb * t)
    choices = [Choice(ok, True, value=val(A, r1, B, r2)),
               Choice(signed_sum([(A, expt(-r1)), (B, expt(-r2))]), why="$\\frac{1}{s-a}$ の逆変換は $e^{at}$。指数の符号を確かめよう。", value=val(A, -r1, B, -r2)),
               Choice(signed_sum([(B, expt(r1)), (A, expt(r2))]), why="部分分数の係数の対応を確かめよう($s=a$ を代入して求める)。", value=val(B, r1, A, r2)),
               Choice(signed_sum([(A, expt(r1)), (-B, expt(r2))]), why="部分分数の係数の符号を確かめよう。", value=val(A, r1, -B, r2))]
    return Problem(
        "逆ラプラス変換(部分分数)", "基本",
        rf"$F(s)=\dfrac{{{num}}}{{{den}}}$ の逆ラプラス変換 $f(t)=\mathcal L^{{-1}}[F(s)]$ を求めよ。",
        choices,
        hints=[r"$\dfrac{ps+q}{(s-a)(s-b)}=\dfrac{A}{s-a}+\dfrac{B}{s-b}$ と部分分数に分ける。",
               r"$A$ は両辺に $(s-a)$ をかけて $s=a$ を代入すると求まる。$\mathcal L^{-1}\left[\frac{1}{s-a}\right]=e^{at}$。"],
        solution=[rf"$$F(s)={frac_sum([(A, '', sh(r1)), (B, '', sh(r2))])}$$",
                  rf"($s={r1}$ を代入して $A={A}$, $s={r2}$ を代入して $B={B}$)",
                  rf"$$f(t)={ok}$$"],
        meta=dict(F=lambda s: A / (s - r1) + B / (s - r2)))


def guided_inv_pf(rng):
    r1, r2, A, B, num, den = pf_parts(rng)
    pfx = frac_sum([(A, '', sh(r1)), (B, '', sh(r2))])
    ok = signed_sum([(A, expt(r1)), (B, expt(r2))])
    return ExamItem("逆ラプラス変換", rf"$F(s)=\dfrac{{{num}}}{{{den}}}$ について, 次の問いに答えよ.",
                    [Part(r"$F(s)$ を部分分数に分解せよ.", pfx,
                          [rf"$\dfrac{{{num}}}{{{den}}}=\dfrac{{A}}{{{sh(r1)}}}+\dfrac{{B}}{{{sh(r2)}}}$ とおき, "
                           rf"両辺に分母をかけて $s={r1},\ {r2}$ を代入すると $A={A},\ B={B}$"]),
                     Part(r"逆ラプラス変換 $f(t)=\mathcal L^{-1}[F(s)]$ を求めよ.", ok,
                          [r"$\mathcal L^{-1}\left[\dfrac{1}{s-a}\right]=e^{at}$ より", rf"$$f(t)={ok}$$"])],
                    meta=dict(F=lambda s: A / (s - r1) + B / (s - r2)))


# ============================================================
# 第1章 標準2:逆変換(平方完成)
# ============================================================
def g_inv_sq(rng):
    a = rng.choice([-3, -2, -1, 1, 2, 3])
    w = rng.choice([1, 2, 3])
    k, m = rng.choice([-2, -1, 1, 2, 3]), rng.choice([-2, -1, 1, 2, 3])
    if k == m or -k * a + m * w == m:
        raise Retry   # 誤答が正解と同じ関数になる
    # F = (k(s-a) + m w) / ((s-a)^2 + w^2)
    num = signed_sum([(k, "s"), (-k * a + m * w, "")])
    den = signed_sum([(1, "s^{2}"), (-2 * a, "s"), (a * a + w * w, "")])

    def body(kk, mm, aa, sw=False):
        tri = signed_sum([(kk, cos_t(w)), (mm, sin_t(w))]) if not sw else signed_sum([(kk, sin_t(w)), (mm, cos_t(w))])
        e = expt(aa)
        return (e + r"\left(" + tri + r"\right)") if e else tri

    def val(kk, mm, aa, sw=False):
        if sw:
            return lambda t: np.exp(aa * t) * (kk * np.sin(w * t) + mm * np.cos(w * t))
        return lambda t: np.exp(aa * t) * (kk * np.cos(w * t) + mm * np.sin(w * t))
    frac_sa = rf"\frac{{{sh(a)}}}{{({sh(a)})^2+{w}^2}}"
    frac_w = rf"\frac{{{w}}}{{({sh(a)})^2+{w}^2}}"
    choices = [Choice(body(k, m, a), True, value=val(k, m, a)),
               Choice(body(k, m, -a), why="平方完成 $(s-a)^2+\\omega^2$ から $e^{at}$ が出る。符号を確かめよう。", value=val(k, m, -a)),
               Choice(body(k, m, a, sw=True), why="$\\frac{s-a}{(s-a)^2+\\omega^2}\\to e^{at}\\cos\\omega t$, $\\frac{\\omega}{(s-a)^2+\\omega^2}\\to e^{at}\\sin\\omega t$。",
                      value=val(k, m, a, sw=True)),
               Choice(body(k, -k * a + m * w, a), why="分子を $k(s-a)+m\\omega$ の形に直してから使う($s$ のままではない)。",
                      value=val(k, -k * a + m * w, a))]
    return Problem(
        "逆ラプラス変換(平方完成)", "標準",
        rf"$F(s)=\dfrac{{{num}}}{{{den}}}$ の逆ラプラス変換を求めよ。",
        choices,
        hints=[r"分母を平方完成して $(s-a)^2+\omega^2$ の形にする。",
               r"分子を $k(s-a)+m\omega$ の形に直し、$\mathcal L^{-1}\left[\frac{s-a}{(s-a)^2+\omega^2}\right]=e^{at}\cos\omega t$, "
               r"$\mathcal L^{-1}\left[\frac{\omega}{(s-a)^2+\omega^2}\right]=e^{at}\sin\omega t$ を使う。"],
        solution=[rf"$${den}=({sh(a)})^2+{w * w}$$",
                  rf"$${num}={signed_sum([(k, '(' + sh(a) + ')'), (m * w, '')])}$$",
                  rf"$$F(s)={signed_sum([(k, frac_sa), (m, frac_w)])}$$",
                  rf"$$f(t)={body(k, m, a)}$$"],
        meta=dict(F=lambda s: (k * (s - a) + m * w) / ((s - a) ** 2 + w * w)))


# ============================================================
# 第2章 標準1:微分方程式
# ============================================================
def ode_parts(rng):
    r1, r2 = rng.sample([-4, -3, -2, -1, 1, 2], 2)
    A, B = rng.choice([-3, -2, -1, 1, 2, 3]), rng.choice([-3, -2, -1, 1, 2, 3])
    y0, v0 = A + B, A * r1 + B * r2
    p, q = -(r1 + r2), r1 * r2
    eq = signed_sum([(1, "y''"), (p, "y'"), (q, "y")]) + "=0"
    char = signed_sum([(1, "s^{2}"), (p, "s"), (q, "")])
    numY = signed_sum([(y0, "s"), (v0 + p * y0, "")])
    return r1, r2, A, B, y0, v0, p, q, eq, char, numY


def ode_solution(r1, r2, A, B, y0, v0, p, q, char, numY):
    return [rf"両辺をラプラス変換すると $s^2Y-sy(0)-y'(0)" + (rf"{'+' if p > 0 else '-'}{abs(p)}(sY-y(0))" if p else "")
            + (rf"{'+' if q > 0 else '-'}{abs(q)}Y" if q else "") + "=0$",
            rf"$$({char})\,Y(s)={numY}\quad\Longrightarrow\quad Y(s)=\frac{{{numY}}}{{({sh(r1)})({sh(r2)})}}$$",
            rf"$$Y(s)={frac_sum([(A, '', sh(r1)), (B, '', sh(r2))])}$$",
            rf"$$y(t)={signed_sum([(A, expt(r1)), (B, expt(r2))])}$$"]


def g_ode(rng):
    r1, r2, A, B, y0, v0, p, q, eq, char, numY = ode_parts(rng)
    ok = signed_sum([(A, expt(r1)), (B, expt(r2))])

    def val(a, ra, b, rb):
        return lambda t: a * np.exp(ra * t) + b * np.exp(rb * t)
    # 誤答:y'(0) を無視(v0 = 0 として解く)/ 特性根の符号 / 係数入れかえ
    A0 = F(-r2 * y0, r1 - r2)
    B0 = F(r1 * y0, r1 - r2)
    choices = [Choice(ok, True, value=val(A, r1, B, r2)),
               Choice(signed_sum([(A, expt(-r1)), (B, expt(-r2))]), why="$Y(s)$ の分母 $(s-a)$ から $e^{at}$ が出る。符号を確かめよう。", value=val(A, -r1, B, -r2)),
               Choice(signed_sum([(B, expt(r1)), (A, expt(r2))]), why="部分分数の係数を確かめよう。$y(0),\\ y'(0)$ を代入して検算できる。", value=val(B, r1, A, r2)),
               Choice(signed_sum([(A0, expt(r1)), (B0, expt(r2))]), why="$\\mathcal L[y'']=s^2Y-sy(0)-y'(0)$。$y'(0)$ の項を落としていないか。",
                      value=val(float(A0), r1, float(B0), r2))]
    return Problem(
        "ラプラス変換による微分方程式の解法", "標準",
        rf"微分方程式 ${eq},\ y(0)={y0},\ y'(0)={v0}$ を, ラプラス変換を用いて解け。",
        choices,
        hints=[r"$\mathcal L[y']=sY-y(0),\ \mathcal L[y'']=s^2Y-sy(0)-y'(0)$ で、微分方程式を $Y(s)$ の方程式にする。",
               r"$Y(s)$ を部分分数に分けて逆変換する。最後に $y(0),\ y'(0)$ で検算しよう。"],
        solution=ode_solution(r1, r2, A, B, y0, v0, p, q, char, numY),
        meta=dict(pq=(p, q), y0=y0, v0=v0))


def guided_ode(rng):
    r1, r2, A, B, y0, v0, p, q, eq, char, numY = ode_parts(rng)
    sol = ode_solution(r1, r2, A, B, y0, v0, p, q, char, numY)
    return ExamItem("ラプラス変換による微分方程式の解法",
                    rf"微分方程式 ${eq},\ y(0)={y0},\ y'(0)={v0}$ について, $Y(s)=\mathcal L[y(t)]$ とする. 次の問いに答えよ.",
                    [Part(r"$Y(s)$ を求めよ.", rf"\frac{{{numY}}}{{({sh(r1)})({sh(r2)})}}", sol[:2]),
                     Part(r"$y(t)$ を求めよ.", signed_sum([(A, expt(r1)), (B, expt(r2))]), sol[2:])],
                    meta=dict(pq=(p, q), y0=y0, v0=v0))


# ============================================================
# 第2章 基本1:合成積
# ============================================================
FUN = {"1exp": lambda a, b: (lambda t: 1 + 0 * t, lambda t: np.exp(a * t)),
       "expexp": lambda a, b: (lambda t: np.exp(a * t), lambda t: np.exp(b * t)),
       "tt": lambda a, b: (lambda t: t, lambda t: t),
       "t1": lambda a, b: (lambda t: t, lambda t: 1 + 0 * t),
       "1sin": lambda a, b: (lambda t: 1 + 0 * t, lambda t: np.sin(t))}


def g_conv(rng):
    a, b = rng.sample([-3, -2, -1, 1, 2, 3], 2)
    kind = rng.choice(["1exp", "expexp", "tt", "1sin", "t1"])
    if kind == "1exp":
        if abs(a) == 1:
            raise Retry   # 誤答 a(e^{at}-1) が正解と一致する
        ft, gt = "1", expt(a)
        ok = paren(a, rf"{expt(a)}-1")
        v = lambda t: (np.exp(a * t) - 1) / a  # noqa: E731
        FG = rf"\frac{{1}}{{s}}\cdot\frac{{1}}{{{sh(a)}}}"
        bads = [(expt(a), "それは $f(t)g(t)$(ただの積)。合成積は積分 $\\int_0^t f(\\tau)g(t-\\tau)d\\tau$。", lambda t: np.exp(a * t)),
                (paren(a, rf"1-{expt(a)}"), "積分の上端・下端の代入順を確かめよう。", lambda t: (1 - np.exp(a * t)) / a),
                (rf"{a}\left({expt(a)}-1\right)", "$\\int e^{a\\tau}d\\tau=\\frac{1}{a}e^{a\\tau}$。$a$ で割る。", lambda t: a * (np.exp(a * t) - 1))]
    elif kind == "expexp":
        ft, gt = expt(a), expt(b)
        ok = paren(a - b, rf"{expt(a)}-{expt(b)}")
        v = lambda t: (np.exp(a * t) - np.exp(b * t)) / (a - b)  # noqa: E731
        FG = rf"\frac{{1}}{{({sh(a)})({sh(b)})}}"
        bads = [(expt(a + b) if a + b else "1", "それは $f(t)g(t)$(ただの積)。合成積は積分。", lambda t: np.exp((a + b) * t)),
                (paren(a - b, rf"{expt(b)}-{expt(a)}"), "部分分数の符号を確かめよう。", lambda t: (np.exp(b * t) - np.exp(a * t)) / (a - b)),
                (paren(a + b, rf"{expt(a)}+{expt(b)}") if a + b else rf"{expt(a)}+{expt(b)}",
                 "$F(s)G(s)$ を部分分数に分けて逆変換しよう。", (lambda t: (np.exp(a * t) + np.exp(b * t)) / (a + b)) if a + b else (lambda t: np.exp(a * t) + np.exp(b * t)))]
    elif kind == "tt":
        ft, gt = "t", "t"
        ok, v = r"\frac{t^3}{6}", (lambda t: t ** 3 / 6)
        FG = r"\frac{1}{s^2}\cdot\frac{1}{s^2}=\frac{1}{s^4}"
        bads = [("t^2", "それは $f(t)g(t)$(ただの積)。合成積は積分。", lambda t: t * t),
                (r"\frac{t^3}{3}", "$\\mathcal L^{-1}\\left[\\frac{1}{s^4}\\right]=\\frac{t^3}{3!}$。", lambda t: t ** 3 / 3),
                (r"\frac{t^4}{24}", "$\\frac{1}{s^4}$ の逆変換は $\\frac{t^3}{3!}$($\\frac{n!}{s^{n+1}}\\leftrightarrow t^n$)。", lambda t: t ** 4 / 24)]
    elif kind == "t1":
        ft, gt = "t", "1"
        ok, v = r"\frac{t^2}{2}", (lambda t: t * t / 2)
        FG = r"\frac{1}{s^2}\cdot\frac{1}{s}=\frac{1}{s^3}"
        bads = [("t", "それは $f(t)g(t)$(ただの積)。合成積は積分。", lambda t: t),
                ("t^2", "$\\mathcal L^{-1}\\left[\\frac{1}{s^3}\\right]=\\frac{t^2}{2!}$。", lambda t: t * t),
                (r"\frac{t^3}{6}", "$\\frac{1}{s^3}$ の逆変換は $\\frac{t^2}{2}$。", lambda t: t ** 3 / 6)]
    else:
        ft, gt = "1", r"\sin t"
        ok, v = r"1-\cos t", (lambda t: 1 - np.cos(t))
        FG = r"\frac{1}{s}\cdot\frac{1}{s^2+1}"
        bads = [(r"\sin t", "それは $f(t)g(t)$(ただの積)。合成積は積分。", lambda t: np.sin(t)),
                (r"\cos t-1", "$\\int_0^t\\sin\\tau\\,d\\tau=\\left[-\\cos\\tau\\right]_0^t$。符号を確かめよう。", lambda t: np.cos(t) - 1),
                (r"1-\sin t", "$\\sin$ の原始関数は $-\\cos$。", lambda t: 1 - np.sin(t))]
    choices = [Choice(ok, True, value=v)] + [Choice(t, why=y, value=vv) for t, y, vv in bads]
    return Problem(
        "合成積", "基本",
        rf"$f(t)={ft},\ g(t)={gt}$ の合成積 $(f*g)(t)=\displaystyle\int_0^tf(\tau)g(t-\tau)\,d\tau$ を求めよ。",
        choices,
        hints=[r"定義どおり積分してもよいし、$\mathcal L[f*g]=F(s)G(s)$ を使って逆変換してもよい。"],
        solution=[rf"$$\mathcal L[f*g]=F(s)G(s)={FG}$$", rf"逆変換して $(f*g)(t)={ok}$"],
        meta=dict(kind=kind, f=FUN[kind](a, b)[0], g=FUN[kind](a, b)[1]))


# ============================================================
# 第2章 標準2:伝達関数とインパルス応答
# ============================================================
def tf_parts(rng):
    r1, r2 = rng.sample([-4, -3, -2, -1], 2)
    p, q = -(r1 + r2), r1 * r2
    eq = signed_sum([(1, "y''"), (p, "y'"), (q, "y")]) + "=r(t)"
    char = signed_sum([(1, "s^{2}"), (p, "s"), (q, "")])
    d = r1 - r2
    return r1, r2, p, q, eq, char, d


def g_transfer(rng):
    r1, r2, p, q, eq, char, d = tf_parts(rng)
    pf12 = rf"\frac{{1}}{{{sh(r1)}}}-\frac{{1}}{{{sh(r2)}}}"
    ok = paren(d, rf"{expt(r1)}-{expt(r2)}")

    def val(sgn=1, dd=d, sw=False):
        if sw:
            return lambda t: (np.exp(-r1 * t) - np.exp(-r2 * t)) / dd
        return lambda t: sgn * (np.exp(r1 * t) - np.exp(r2 * t)) / dd
    choices = [Choice(ok, True, value=val()),
               Choice(paren(d, rf"{expt(-r1)}-{expt(-r2)}"),
                      why="$G(s)$ の極(分母の根)が $e^{at}$ の指数になる。符号を確かめよう。", value=val(sw=True)),
               Choice(rf"{expt(r1)}-{expt(r2)}", why="部分分数の係数 $\\frac{1}{a-b}$ を忘れていないか。", value=val(dd=1)),
               Choice(paren(d, rf"{expt(r2)}-{expt(r1)}"),
                      why="部分分数の符号を確かめよう。", value=val(sgn=-1))]
    return Problem(
        "伝達関数とインパルス応答", "標準",
        rf"システム ${eq}$(初期値はすべて $0$)の伝達関数を $G(s)$ とするとき, インパルス応答 $g(t)=\mathcal L^{{-1}}[G(s)]$ を求めよ。",
        choices,
        hints=[r"初期値 $0$ でラプラス変換すると $({s^2+ps+q})Y=R$ なので $G(s)=\dfrac{Y}{R}=\dfrac{1}{s^2+ps+q}$。",
               r"$r(t)=\delta(t)$ なら $R(s)=1$ なので、出力は $\mathcal L^{-1}[G(s)]$(インパルス応答)。"],
        solution=[rf"$$G(s)=\frac{{1}}{{{char}}}=\frac{{1}}{{({sh(r1)})({sh(r2)})}}={paren(d, pf12)}$$",
                  rf"$$g(t)={ok}$$"],
        meta=dict(pq=(p, q)))


def guided_transfer(rng):
    r1, r2, p, q, eq, char, d = tf_parts(rng)
    pf12 = rf"\frac{{1}}{{{sh(r1)}}}-\frac{{1}}{{{sh(r2)}}}"
    ok = paren(d, rf"{expt(r1)}-{expt(r2)}")
    return ExamItem("伝達関数とインパルス応答", rf"システム ${eq}$(初期値はすべて $0$)について, 次の問いに答えよ.",
                    [Part(r"伝達関数 $G(s)=\dfrac{Y(s)}{R(s)}$ を求めよ.", rf"\frac{{1}}{{{char}}}",
                          [rf"初期値 $0$ でラプラス変換すると $({char})Y(s)=R(s)$"]),
                     Part(r"インパルス応答 $g(t)=\mathcal L^{-1}[G(s)]$ を求めよ.", ok,
                          [rf"$$G(s)={paren(d, pf12)}$$", rf"$$g(t)={ok}$$"])],
                    meta=dict(pq=(p, q)))


def _display(gen):
    """選択肢の分数を \\dfrac にして読みやすくする(4択の選択肢・答の行)"""
    def g(rng):
        p = gen(rng)
        for c in p.choices:
            c.tex = c.tex.replace(r"\frac", r"\dfrac")
        return p
    g.__name__ = gen.__name__
    return g


for _n in ("g_table", "g_shift1", "g_shift2", "g_tfun", "g_inv_pf", "g_inv_sq", "g_conv", "g_ode", "g_transfer"):
    globals()[_n] = _display(globals()[_n])


REGISTRY1 = [
    ProblemType("lp_table", "公式によるラプラス変換", "基本", g_table),
    ProblemType("lp_shift1", "第1移動法則", "基本", g_shift1),
    ProblemType("lp_shift2", "第2移動法則", "基本", g_shift2),
    ProblemType("lp_inv_pf", "逆ラプラス変換(部分分数)", "基本", g_inv_pf, guided=guided_inv_pf),
    ProblemType("lp_tfun", "像関数の微分法則", "標準", g_tfun),
    ProblemType("lp_inv_sq", "逆ラプラス変換(平方完成)", "標準", g_inv_sq),
]
REGISTRY2 = [
    ProblemType("lp_conv", "合成積", "基本", g_conv),
    ProblemType("lp_ode", "ラプラス変換による微分方程式の解法", "標準", g_ode, guided=guided_ode),
    ProblemType("lp_tf", "伝達関数とインパルス応答", "標準", g_transfer, guided=guided_transfer),
]
GUIDED_NAMES = {"lp_inv_pf": "部分分数 → 逆ラプラス変換", "lp_ode": "Y(s) → 微分方程式の解",
                "lp_tf": "伝達関数 → インパルス応答"}


def tab1():
    render("lap1", REGISTRY1, "ラプラス変換 第1章 定義と性質")


def tab2():
    render("lap2", REGISTRY2, "ラプラス変換 第2章 応用")
