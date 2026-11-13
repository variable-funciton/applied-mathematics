"""数学IV その他 シュワルツの超関数の理論(デルタ関数):演習の問題生成

⟨δ(x−a), φ⟩ = φ(a), ⟨T', φ⟩ = −⟨T, φ'⟩, H' = δ, |x|'' = 2δ, δ(kx) = δ(x)/|k|, ⟨δ', φ⟩ = −φ'(0)。
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


def poly(cs, var="x"):
    """係数(高次から)の多項式の TeX"""
    n = len(cs) - 1
    out = ""
    for i, c in enumerate(cs):
        c = F(c)
        if c == 0:
            continue
        d = n - i
        body = "" if d == 0 else (var if d == 1 else f"{var}^{d}")
        sgn = "-" if c < 0 else ("+" if out else "")
        m = abs(c)
        coef = tex_fr(m) if (d == 0 or m != 1) else ""
        out += sgn + coef + body
    return out or "0"


def pval(cs, x):
    v = F(0)
    for c in cs:
        v = v * x + c
    return v


def pder(cs):
    n = len(cs) - 1
    return [c * (n - i) for i, c in enumerate(cs[:-1])] or [0]


def shift(a, var="x"):
    return var if a == 0 else (f"{var}-{a}" if a > 0 else f"{var}+{-a}")


# ------------------------------------------------------------
# 基本:δ(x − a) との積分
# ------------------------------------------------------------
def g_shift(rng):
    cs = [rng.choice([1, 2, -1]), rng.choice([-3, -2, 0, 1, 2]), rng.choice([1, 3, -4, 5])]
    a = rng.choice([-2, -1, 1, 2, 3])
    inside = rng.random() < 0.8
    lo, hi = (a - 2, a + 3) if inside else ((a + 1, a + 4) if rng.random() < 0.5 else (a - 4, a - 1))
    val = pval(cs, a) if inside else F(0)
    ok = tex_fr(val)
    cands = [(tex_fr(pval(cs, -a)), r"$\delta(x-a)$ は $x=a$ に集中している($-a$ ではない)。"), (tex_fr(pval(cs, 0)), r"$\delta(x-a)$ なので代入するのは $x=a$。"),
             (tex_fr(pval(cs, a)) if not inside else "0", r"$x=a$ が積分区間に入っているかを確かめる。" if not inside else r"$x=a$ は積分区間に入っている。"),
             ("1", r"$\int\delta(x-a)\,dx=1$ だが, $f(x)$ が掛かっているので $f(a)$。")]
    return Problem("デルタ関数との積分", "基本",
                   rf"$\displaystyle\int_{{{lo}}}^{{{hi}}}\delta({shift(a)})\left({poly(cs)}\right)dx$ を求めよ。",
                   distinct(ok, cands), hints=[r"$\int\delta(x-a)f(x)\,dx=f(a)$($x=a$ が積分区間の内部にあるとき)。区間の外なら $0$。"],
                   solution=[rf"$x={a}$ は区間 $[{lo},\,{hi}]$ の{'内部' if inside else '外'}にある。", rf"値は ${ok}$。"], meta=dict(cs=cs, a=a, lo=lo, hi=hi, val=val))


# ------------------------------------------------------------
# 標準:δ(kx − b)
# ------------------------------------------------------------
def g_scale(rng):
    cs = [rng.choice([1, 2, 3]), rng.choice([-1, 0, 1]), rng.choice([1, 2])]
    k = rng.choice([2, 3, -2, 4])
    b = k * rng.choice([1, -1, 2])
    x0 = F(b, k)
    val = pval(cs, x0) / abs(k)
    ok = tex_fr(val)
    cands = [(tex_fr(pval(cs, x0)), r"$\delta(kx)=\dfrac{1}{|k|}\delta(x)$:$\dfrac{1}{|k|}$ を掛ける。"), (tex_fr(pval(cs, b) / abs(k)), r"$kx-b=0$ となる $x=\dfrac bk$ を代入する。"),
             (tex_fr(pval(cs, x0) / k) if k < 0 else tex_fr(pval(cs, x0) * abs(k)), r"$\dfrac{1}{|k|}$ は正(絶対値)。" if k < 0 else r"$y=kx$ と置換すると $dx=\dfrac{dy}{k}$:割る。"),
             (tex_fr(pval(cs, 0) / abs(k)), r"代入するのは $x=\dfrac bk$。")]
    inner = f"{k}x" + (f"-{b}" if b > 0 else f"+{-b}")
    return Problem("デルタ関数の変数変換", "標準",
                   rf"$\displaystyle\int_{{-\infty}}^{{\infty}}\delta({inner})\left({poly(cs)}\right)dx$ を求めよ。",
                   distinct(ok, cands), hints=[r"$y=kx-b$ と置換する($k<0$ のときは積分の向きが逆になる)。", r"$\delta(kx-b)=\dfrac{1}{|k|}\delta\!\left(x-\dfrac bk\right)$。"],
                   solution=[rf"$\delta({inner})=\dfrac{{1}}{{{abs(k)}}}\delta\!\left(x-{tex_fr(x0) if x0 >= 0 else '(' + tex_fr(x0) + ')'}\right)$。",
                             rf"値は $\dfrac{{1}}{{{abs(k)}}}\times{tex_fr(pval(cs, x0))}={ok}$。"], meta=dict(cs=cs, k=k, b=b, val=val))


# ------------------------------------------------------------
# 基本:とびのある関数の微分
# ------------------------------------------------------------
def jump_case(rng):
    while True:   # 誘導問題は Retry なしで呼ばれるので, ここで作り直す
        a = rng.choice([-1, 1, 2])
        p = [rng.choice([1, 2, -1]), rng.choice([0, 1, 3])]          # 左:px + q
        q = [rng.choice([1, 2, -2, 3]), rng.choice([-2, 0, 2, 4])]   # 右
        J = pval(q, a) - pval(p, a)
        if J != 0 and p != q:
            return a, p, q, J


def piece(p, q, a):
    return rf"\begin{{cases}}{poly(p)}&(x<{a})\\{poly(q)}&(x>{a})\end{{cases}}"


def deriv_tex(p, q, a, J):
    dp, dq = pder(p)[0], pder(q)[0]
    base = rf"\begin{{cases}}{tex_fr(F(dp))}&(x<{a})\\{tex_fr(F(dq))}&(x>{a})\end{{cases}}" if dp != dq else tex_fr(F(dp))
    return base, rf"{'+' if J > 0 else '-'}{'' if abs(J) == 1 else tex_fr(abs(J))}\delta({shift(a)})"


def g_jump(rng):
    a, p, q, J = jump_case(rng)
    base, dl = deriv_tex(p, q, a, J)
    ok = base + dl
    _, dl_neg = deriv_tex(p, q, a, -J)
    dl_slope = rf"+{tex_fr(F(pder(q)[0] - pder(p)[0]))}\delta({shift(a)})" if pder(q)[0] - pder(p)[0] > 0 else rf"-{tex_fr(F(abs(pder(q)[0] - pder(p)[0]) or 1))}\delta({shift(a)})"
    cands = [(base + dl_neg, "とびの大きさは「右の値 − 左の値」。"), (base, "とびのところで, 超関数の微分にはデルタ関数が現れる。"),
             (base + dl_slope, "デルタ関数の係数は傾きの差ではなく, 関数の値のとび。"),
             (base + rf"{'+' if J > 0 else '-'}{'' if abs(J) == 1 else tex_fr(abs(J))}\delta(x)", r"とびの位置 $x=a$ にデルタ関数を置く。")]
    return Problem("とびのある関数の微分", "基本",
                   rf"関数 $f(x)={piece(p, q, a)}$ を超関数として微分せよ。",
                   distinct(ok, cands), hints=[r"$f=g+J\,H(x-a)$($J$ はとびの大きさ)と書くと $f^{\prime}=g^{\prime}+J\,\delta(x-a)$。"],
                   solution=[rf"$x={a}$ でのとび:右 ${tex_fr(pval(q, a))}$ − 左 ${tex_fr(pval(p, a))}$ $={tex_fr(J)}$。", rf"$f^{{\prime}}={ok}$。"],
                   meta=dict(a=a, p=p, q=q, J=J))


# ------------------------------------------------------------
# 標準:折れ線の2階微分
# ------------------------------------------------------------
def g_kink(rng):
    kind = rng.choice(["abs", "ramp"])
    c = rng.choice([1, 2, 3, -1, F(1, 2)])
    a = rng.choice([0, 1, -2, 3])
    cs = tex_fr(F(c)) if c not in (1, -1) else ("" if c == 1 else "-")
    if kind == "abs":
        ft = rf"{cs}|{shift(a)}|"
        val = 2 * F(c)
        hint = r"$|x|^{\prime}=\mathrm{sgn}\,x=2H(x)-1$。"
    else:
        ft = rf"{cs}({shift(a)})H({shift(a)})"
        val = F(c)
        hint = r"$(xH(x))^{\prime}=H(x)$($x\,\delta(x)=0\cdot\delta(x)=0$)。"

    def dl(v, pos=a):
        v = F(v)
        return ("" if v == 1 else ("-" if v == -1 else tex_fr(v))) + rf"\delta({shift(pos)})"
    ok = dl(val)
    cands = [(dl(val / 2) if kind == "abs" else dl(2 * val), r"$|x|$ の傾きは $-1$ から $1$ へ, $2$ だけ変わる。" if kind == "abs" else r"傾きは $0$ から $c$ へ変わる($2$ 倍しない)。"),
             (dl(val, -a) if a != 0 else dl(-val), "デルタ関数の位置(または符号)を確かめよう。"),
             ("0", "折れ目では1階微分がとぶので, 2階微分にデルタ関数が現れる。"),
             (rf"{tex_fr(val) if val != 1 else ''}H({shift(a)})", "これは1階微分の形。もう1回微分する。")]
    return Problem("折れ線の2階微分", "標準", rf"$f(x)={ft}$ を超関数として2回微分せよ。",
                   distinct(ok, cands), hints=[hint, r"$H^{\prime}=\delta$。"],
                   solution=[rf"1階微分は傾きの関数で, $x={a}$ で傾きが ${tex_fr(val)}$ だけとぶ。", rf"よって $f^{{\prime\prime}}={ok}$。"],
                   meta=dict(kind=kind, c=c, a=a, val=val))


# ------------------------------------------------------------
# 標準:δ′ を当てる
# ------------------------------------------------------------
def g_dprime(rng):
    cs = [rng.choice([1, 2, -1]), rng.choice([-3, 0, 2]), rng.choice([1, 4, -2]), rng.choice([0, 1, 5])]
    a = rng.choice([0, 1, -1, 2])
    d = pder(cs)
    val = -pval(d, a)
    if val == 0:
        raise Retry
    ok = tex_fr(val)
    cands = [(tex_fr(-val), r"$\langle\delta^{\prime},\varphi\rangle=-\langle\delta,\varphi^{\prime}\rangle$:マイナスが付く。"),
             (tex_fr(-pval(cs, a)), "微分するのは $\\varphi$ の方。"), (tex_fr(pval(cs, a)), r"$\delta^{\prime}$ は $\varphi$ の値ではなく, 微分係数(の $-1$ 倍)を取り出す。"),
             (tex_fr(-pval(d, 0)) if a != 0 else tex_fr(-pval(pder(d), 0)), r"$x=a$ での値をとる。" if a != 0 else "1回だけ微分する。")]
    dd = rf"\delta^{{\prime}}({shift(a)})" if a else r"\delta^{\prime}"
    return Problem("デルタ関数の微分", "標準",
                   rf"$\varphi(x)={poly(cs)}$ とする(遠くでは $0$ になるように変形してあるとしてよい)。$\langle {dd},\,\varphi\rangle$ を求めよ。",
                   distinct(ok, cands), hints=[r"超関数の微分の定義 $\langle T^{\prime},\varphi\rangle=-\langle T,\varphi^{\prime}\rangle$。", r"$\langle\delta^{\prime}(x-a),\varphi\rangle=-\varphi^{\prime}(a)$。"],
                   solution=[rf"$\varphi^{{\prime}}(x)={poly(d)}$。", rf"$\langle {dd},\varphi\rangle=-\varphi^{{\prime}}({a})={ok}$。"], meta=dict(cs=cs, a=a, val=val))


# ------------------------------------------------------------
# 基本:ラプラス変換・フーリエ変換
# ------------------------------------------------------------
def g_transform(rng):
    kind = rng.choice(["L", "F"])
    a = rng.choice([1, 2, 3])
    c = rng.choice([1, 2, 3, 5])
    cc = "" if c == 1 else str(c)
    if kind == "L":
        ok = rf"{cc}e^{{-{a if a > 1 else ''}s}}"
        cands = [(rf"{cc}e^{{{a if a > 1 else ''}s}}", r"第2移動法則:$\mathcal L[f(t-a)U(t-a)]=e^{-as}F(s)$。符号はマイナス。"), (cc or "1", r"$\mathcal L[\delta]=1$ だが, $t=a$ にずれると $e^{-as}$ が掛かる。"),
                 (rf"\frac{{{c}}}{{s}}e^{{-{a if a > 1 else ''}s}}", r"$\frac1s$ はステップ関数の変換。$\mathcal L[\delta]=1$。")]
        q = rf"$\mathcal L[{cc}\delta(t-{a})]$ を求めよ。"
        sol = [rf"$\int_0^\infty {cc}\delta(t-{a})e^{{-st}}dt={cc}e^{{-{a}s}}$。"]
    else:
        ok = rf"{cc}e^{{-i{a if a > 1 else ''}\omega}}"
        cands = [(rf"{cc}e^{{i{a if a > 1 else ''}\omega}}", r"$F(\omega)=\int f(t)e^{-i\omega t}dt$:符号はマイナス。"), (cc or "1", r"$t=a$ にずれると $e^{-ia\omega}$ が掛かる。"),
                 (rf"{2 * c}\pi\delta(\omega-{a})", r"これは $e^{iat}$ のフーリエ変換の形。")]
        q = rf"フーリエ変換 $F(\omega)=\displaystyle\int_{{-\infty}}^{{\infty}}f(t)e^{{-i\omega t}}dt$ で, $f(t)={cc}\delta(t-{a})$ のフーリエ変換を求めよ。"
        sol = [rf"$\int {cc}\delta(t-{a})e^{{-i\omega t}}dt={cc}e^{{-i{a}\omega}}$。"]
    return Problem("デルタ関数の変換", "基本", q, distinct(ok, cands), hints=[r"$\int\delta(t-a)g(t)\,dt=g(a)$ で $g$ を $e^{-st}$ や $e^{-i\omega t}$ とする。"],
                   solution=sol, meta=dict(kind=kind, a=a, c=c))


# ------------------------------------------------------------
# 誘導
# ------------------------------------------------------------
def guided_heaviside(rng):
    a = rng.choice([0, 1, 2, -1])
    c = rng.choice([1, 2, 3])
    cc = "" if c == 1 else str(c)
    return ExamItem("ヘヴィサイド関数とランプ関数の微分",
                    rf"$H(x)$ をヘヴィサイド関数($x<0$ で $0$, $x>0$ で $1$)とし, $r(x)={cc}({shift(a)})H({shift(a)})$ とする. "
                    r"超関数の微分は, 遠くで $0$ になるなめらかな関数 $\varphi$ に対して $\langle T^{\prime},\varphi\rangle=-\langle T,\varphi^{\prime}\rangle$ で定める. 次の問いに答えよ.",
                    [Part(rf"$\langle H^{{\prime}}({shift(a)}),\varphi\rangle=-\displaystyle\int_{{{a}}}^{{\infty}}\varphi^{{\prime}}(x)\,dx$ を計算し, $H^{{\prime}}({shift(a)})$ を求めよ.",
                          rf"\varphi({a}),\ H^{{\prime}}({shift(a)})=\delta({shift(a)})",
                          [rf"$-\int_{{{a}}}^{{\infty}}\varphi^{{\prime}}dx=-\bigl[\varphi\bigr]_{{{a}}}^{{\infty}}=\varphi({a})$($\varphi$ は遠くで $0$)。", rf"よって $H^{{\prime}}({shift(a)})=\delta({shift(a)})$。"]),
                     Part(r"$r^{\prime}$ と $r^{\prime\prime}$ を求めよ.", rf"r^{{\prime}}={cc}H({shift(a)}),\ r^{{\prime\prime}}={cc}\delta({shift(a)})",
                          [rf"$x>{a}$ では $r={cc}({shift(a)})$ で傾き ${c}$, $x<{a}$ では $0$。$r$ は連続なのでとびはなく $r^{{\prime}}={cc}H({shift(a)})$。",
                           rf"(1) より $r^{{\prime\prime}}={cc}\delta({shift(a)})$。"])], meta=dict(a=a, c=c))


def guided_jump(rng):
    a, p, q, J = jump_case(rng)
    base, dl = deriv_tex(p, q, a, J)
    return ExamItem("とびのある関数の超関数としての微分",
                    rf"関数 $f(x)={piece(p, q, a)}$ について, 次の問いに答えよ.",
                    [Part(rf"$x\neq{a}$ でのふつうの微分 $f^{{\prime}}(x)$ と, $x={a}$ でのとびの大きさ $J=f({a}+0)-f({a}-0)$ を求めよ.",
                          rf"f^{{\prime}}={base}\ (x\neq{a}),\ J={tex_fr(J)}",
                          [rf"右の値 ${tex_fr(pval(q, a))}$, 左の値 ${tex_fr(pval(p, a))}$ より $J={tex_fr(J)}$。"]),
                     Part(r"$f$ を超関数として微分せよ.", base + dl, [r"$f=g+J\,H(x-a)$ の形に書くと $f^{\prime}=g^{\prime}+J\,\delta(x-a)$。", rf"$f^{{\prime}}={base}{dl}$。"])],
                    meta=dict(a=a, p=p, q=q, J=J))


REGISTRY = [
    ProblemType("ds_shift", "デルタ関数との積分", "基本", g_shift),
    ProblemType("ds_trans", "デルタ関数の変換", "基本", g_transform),
    ProblemType("ds_jump", "とびのある関数の微分", "基本", g_jump, guided=guided_jump),
    ProblemType("ds_scale", "デルタ関数の変数変換", "標準", g_scale),
    ProblemType("ds_kink", "折れ線の2階微分", "標準", g_kink, guided=guided_heaviside),
    ProblemType("ds_dprime", "デルタ関数の微分", "標準", g_dprime),
]
GUIDED_NAMES = {"ds_jump": "ふつうの微分ととびの大きさ → 超関数としての微分",
                "ds_kink": "ヘヴィサイド関数の微分 → ランプ関数の1階・2階微分"}


def tab():
    render("dist", REGISTRY, "その他 シュワルツの超関数の理論(デルタ関数)")
