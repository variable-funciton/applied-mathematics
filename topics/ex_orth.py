"""数学IV ⑦ 特殊関数 第2章 直交関数系:演習の問題生成

すべて Fraction で厳密に計算する(チェビシェフは π, エルミートは √π をくくり出す)。
"""
from fractions import Fraction

from topics.ex_core import Choice, ExamItem, Part, Problem, ProblemType, Retry, render, tex_fr
from topics.orth_core import POLY, WEIGHTS, gram_schmidt, inner, norm2, p_add, p_scale, p_tex, p_val, trim, walsh_coefs, walsh_matrix

F = Fraction
NAME = {"legendre": ("ルジャンドル多項式", "P"), "chebyshev": ("チェビシェフ多項式", "T"), "hermite": ("エルミート多項式", "H"), "laguerre": ("ラゲール多項式", "L")}
RECTEX = {
    "legendre": r"(n+1)P_{n+1}(x)=(2n+1)xP_n(x)-nP_{n-1}(x)",
    "hermite": r"H_{n+1}(x)=2xH_n(x)-2nH_{n-1}(x)",
    "laguerre": r"(n+1)L_{n+1}(x)=(2n+1-x)L_n(x)-nL_{n-1}(x)",
}
FIRST = {"legendre": r"P_0=1,\ P_1=x", "hermite": r"H_0=1,\ H_1=2x", "laguerre": r"L_0=1,\ L_1=1-x"}
BOUNDS = {"legendre": ("-1", "1"), "chebyshev": ("-1", "1"), "hermite": (r"-\infty", r"\infty"), "laguerre": ("0", r"\infty")}


def distinct(ok, cands):
    choices = [Choice(ok, True, value=ok)]
    for t, why in cands:
        if all(c.tex != t for c in choices) and len(choices) < 4:
            choices.append(Choice(t, why=why, value=t))
    if len(choices) < 4:
        raise Retry
    return choices


def with_const(v, const):
    """v × 定数(π, √π)の TeX"""
    v = F(v)
    if v == 0 or not const:
        return tex_fr(v)
    if v == 1:
        return const
    if v == -1:
        return "-" + const
    return tex_fr(v) + const


def wrong_rec(fam, n, x0):
    """漸化式の第2項の符号を間違えた値"""
    p0, p1 = F(1), {"legendre": x0, "hermite": 2 * x0, "laguerre": 1 - x0}[fam]
    for k in range(1, n):
        if fam == "legendre":
            p0, p1 = p1, ((2 * k + 1) * x0 * p1 + k * p0) / (k + 1)
        elif fam == "hermite":
            p0, p1 = p1, 2 * x0 * p1 + 2 * k * p0
        else:
            p0, p1 = p1, ((2 * k + 1 - x0) * p1 + k * p0) / (k + 1)
    return p1


# ------------------------------------------------------------
# 基本:漸化式で値を求める
# ------------------------------------------------------------
def g_rec(rng):
    fam = rng.choice(["legendre", "hermite", "laguerre"])
    n = rng.choice([3, 4]) if fam != "laguerre" else rng.choice([2, 3])
    x0 = F(rng.choice({"legendre": ["1/2", "-1/2", "2", "1/3"], "hermite": ["1", "-1", "1/2", "2"], "laguerre": ["1", "2", "3", "4"]}[fam]))
    nm, s = NAME[fam]
    val = p_val(POLY[fam](n), x0)
    ok = tex_fr(val)
    cands = [(tex_fr(wrong_rec(fam, n, x0)), "漸化式の最後の項はマイナス。"),
             (tex_fr(p_val(POLY[fam](n - 1), x0)), f"これは ${s}_{{{n - 1}}}({tex_fr(x0)})$。もう1回漸化式を使う。"),
             (tex_fr(p_val(POLY[fam](n + 1), x0)), f"これは ${s}_{{{n + 1}}}({tex_fr(x0)})$。"),
             (tex_fr(-val), "符号を確かめよう。")]
    return Problem(f"{nm}の漸化式", "基本",
                   rf"{nm}は ${FIRST[fam]}$ と漸化式 ${RECTEX[fam]}$ で決まる。${s}_{{{n}}}\left({tex_fr(x0)}\right)$ の値を求めよ。",
                   distinct(ok, cands), hints=["$n=1,2,\\dots$ と順に漸化式に代入する(多項式を求めてから代入してもよい)。"],
                   solution=[rf"${s}_{{{k}}}(x)={p_tex(POLY[fam](k))}$" + ("," if k < n else "。") for k in range(2, n + 1)] +
                            [rf"${s}_{{{n}}}\left({tex_fr(x0)}\right)={ok}$。"], meta=dict(fam=fam, n=n, x0=x0, val=val))


# ------------------------------------------------------------
# 基本:T_n(cos θ) = cos nθ
# ------------------------------------------------------------
COSV = {F(1): (0, 1), F(1, 2): (1, 3), F(0): (1, 2), F(-1, 2): (2, 3), F(-1): (1, 1)}   # x0 = cos(pπ/q)


def cos_rational(p, q):
    """cos(pπ/q)(値が有理数のものだけ)"""
    r = F(p, q) % 2
    table = {F(0): F(1), F(1, 3): F(1, 2), F(1, 2): F(0), F(2, 3): F(-1, 2), F(1): F(-1), F(4, 3): F(-1, 2), F(3, 2): F(0), F(5, 3): F(1, 2)}
    if r not in table:
        raise Retry
    return table[r]


def g_cheb(rng):
    x0 = rng.choice([F(1, 2), F(-1, 2), F(0)])
    n = rng.choice([3, 4, 5, 6, 7])
    p, q = COSV[x0]
    val = cos_rational(n * p, q)
    if val != p_val(POLY["chebyshev"](n), x0):
        raise Retry
    th = {F(1, 2): r"\dfrac{\pi}{3}", F(-1, 2): r"\dfrac{2\pi}{3}", F(0): r"\dfrac{\pi}{2}"}[x0]
    ok = tex_fr(val)
    cands = [(tex_fr(-val), r"$\cos n\theta$ の符号を確かめよう。"), (tex_fr(cos_rational((n - 1) * p, q)), rf"これは $T_{{{n - 1}}}$ の値。"),
             (tex_fr(cos_rational((n + 1) * p, q)), rf"これは $T_{{{n + 1}}}$ の値。"), (tex_fr(x0 ** n), r"$T_n(x)=x^n$ ではない。"),
             ("1", r"$\cos n\theta$ の値を単位円で確かめよう。")]
    return Problem("チェビシェフ多項式の値", "基本",
                   rf"チェビシェフ多項式は $T_n(\cos\theta)=\cos n\theta$ をみたす。これを使って $T_{{{n}}}\left({tex_fr(x0)}\right)$ を求めよ。",
                   distinct(ok, cands), hints=[rf"${tex_fr(x0)}=\cos{th}$。"],
                   solution=[rf"$\theta={th}$ とすると $T_{{{n}}}\left({tex_fr(x0)}\right)=\cos\left({n}\cdot{th}\right)={ok}$。"],
                   meta=dict(n=n, x0=x0, val=val))


# ------------------------------------------------------------
# 基本:級数解法の係数
# ------------------------------------------------------------
ODE = {
    "hermite": (r"y^{\prime\prime}-2xy^{\prime}+{c}y=0", lambda k, lam: F(2 * (k - lam), (k + 2) * (k + 1)), 2,
                r"(k+2)(k+1)a_{k+2}-2ka_k+2\lambda a_k=0", r"a_{k+2}=\dfrac{2(k-\lambda)}{(k+2)(k+1)}a_k"),
    "legendre": (r"(1-x^2)y^{\prime\prime}-2xy^{\prime}+{c}y=0", lambda k, lam: F((k - lam) * (k + lam + 1), (k + 2) * (k + 1)), 2,
                 r"(k+2)(k+1)a_{k+2}-k(k+1)a_k+\nu(\nu+1)a_k=0", r"a_{k+2}=\dfrac{(k-\nu)(k+\nu+1)}{(k+2)(k+1)}a_k"),
    "laguerre": (r"xy^{\prime\prime}+(1-x)y^{\prime}+{c}y=0", lambda k, lam: F(k - lam, (k + 1) ** 2), 1,
                 r"(k+1)^2a_{k+1}-ka_k+\lambda a_k=0", r"a_{k+1}=\dfrac{k-\lambda}{(k+1)^2}a_k"),
}


def ode_tex(fam, lam):
    c = {"hermite": 2 * lam, "legendre": lam * (lam + 1), "laguerre": lam}[fam]
    return ODE[fam][0].replace("{c}", str(c) if c != 1 else "")


def series(fam, lam, a0, a1, K):
    r, step = ODE[fam][1], ODE[fam][2]
    a = [F(0)] * (K + 1)
    a[0] = F(a0)
    if step == 2:
        a[1] = F(a1)
    for k in range(K + 1 - step):
        a[k + step] = r(k, lam) * a[k]
    return a


def g_series(rng):
    fam = rng.choice(["hermite", "legendre", "laguerre"])
    lam = rng.choice([F(1, 2), F(5, 2), F(3), F(4), F(5)]) if fam != "laguerre" else rng.choice([F(1, 2), F(3), F(4), F(5)])
    if fam == "laguerre":
        a0, a1, K = 1, 0, 2
    else:
        a0, a1 = rng.choice([(1, 0), (0, 1)])
        K = 4 if a0 else 5
    a = series(fam, lam, a0, a1, K)
    val = a[K]
    if val == 0:
        raise Retry
    ok = tex_fr(val)
    step = ODE[fam][2]
    r = ODE[fam][1]
    # 誤答
    w1 = series(fam, -lam if fam == "hermite" else lam + 1, a0, a1, K)[K]
    prev = a[K - step]
    w3 = prev * r(K - step, lam) * K if step == 2 else prev * F(K - 1 - lam, K)   # 分母の因数を1つ落とした
    cands = [(tex_fr(w1), "漸化式の $\\lambda$(または $\\nu$)の入れ方を確かめよう。"), (tex_fr(prev), rf"これは $a_{{{K - step}}}$。もう1回漸化式を使う。"),
             (tex_fr(w3), "分母の $(k+2)(k+1)$ をすべて使ったか確かめよう。" if step == 2 else "分母は $(k+1)^2$。"),
             (tex_fr(-val), "符号を確かめよう。")]
    init = r"a_0=1" if fam == "laguerre" else (r"a_0=1,\ a_1=0" if a0 else r"a_0=0,\ a_1=1")
    return Problem("級数解法の係数", "基本",
                   rf"微分方程式 ${ode_tex(fam, lam)}$ の解を $y=\sum_{{k=0}}^{{\infty}}a_kx^k$ とおく。${init}$ のとき, $a_{{{K}}}$ を求めよ。",
                   distinct(ok, cands), hints=["$y=\\sum a_kx^k$ を代入して $x^k$ の係数を比べる。", f"漸化式は ${ODE[fam][4]}$ の形になる。"],
                   solution=[f"$x^k$ の係数を比べると ${ODE[fam][4]}$" + (rf"($\lambda={tex_fr(lam)}$)。" if fam != "legendre" else rf"($\nu={tex_fr(lam)}$)。"),
                             "$" + ",\\ ".join(f"a_{{{k}}}={tex_fr(a[k])}" for k in range(K % step, K + 1, step)) + "$。"],
                   meta=dict(fam=fam, lam=lam, a0=a0, a1=a1, K=K, val=val))


# ------------------------------------------------------------
# 標準:直交多項式で展開した係数
# ------------------------------------------------------------
def g_expand(rng):
    fam = rng.choice(["legendre", "chebyshev", "hermite", "laguerre"])
    deg = rng.choice([2, 3]) if fam != "laguerre" else 2
    f = trim([rng.choice([-2, -1, 0, 1, 2, 3]) for _ in range(deg)] + [rng.choice([1, 2, -1, 3])])
    n = rng.choice([deg - 1, deg]) if deg == 3 else rng.choice([0, 2])
    nm, s = NAME[fam]
    co = [inner(f, POLY[fam](k), fam) / norm2(fam, k) for k in range(deg + 1)]
    val = co[n]
    ok = tex_fr(val)
    raw = inner(f, POLY[fam](n), fam)
    cands = [(tex_fr(raw), rf"$(f,{s}_n)_w$ をノルムの2乗 $({s}_n,{s}_n)_w$ で割るのを忘れていないか。"),
             (tex_fr(f[n] if n < len(f) else 0), rf"$x^{n}$ の係数そのものではない。${s}_n$ にも低い次数の項がある。"),
             (tex_fr(co[n - 1] if n > 0 else co[n + 1]), "別の番号の係数。"),
             (tex_fr(val * 2), "ノルムの2乗の値を確かめよう。"), (tex_fr(-val), "符号を確かめよう。")]
    polys = ",\\ ".join(f"{s}_{k}={p_tex(POLY[fam](k))}" for k in range(deg + 1))
    return Problem("直交多項式による展開", "標準",
                   rf"$f(x)={p_tex(f)}$ を{nm}で $f=\sum c_k{s}_k$ と表すとき, $c_{{{n}}}$ を求めよ。ただし ${polys}$ である。",
                   distinct(ok, cands), hints=[r"最高次から順に係数を決めていくと速い(または $c_k=\dfrac{(f,p_k)_w}{(p_k,p_k)_w}$)。"],
                   solution=[f"$f=" + "+".join(f"\\left({tex_fr(co[k])}\\right){s}_{k}" for k in range(deg + 1) if co[k] != 0) + "$ と表せる"
                             f"(最高次の係数から順に決める)。", rf"よって $c_{{{n}}}={ok}$。"], meta=dict(fam=fam, f=f, n=n, val=val))


# ------------------------------------------------------------
# 標準:重みつきの積分(直交性とモーメント)
# ------------------------------------------------------------
def g_moment(rng):
    fam = rng.choice(["legendre", "chebyshev", "hermite", "laguerre"])
    n = rng.choice([1, 2, 3])
    k = rng.choice([n - 1, n, n + 1, n + 2]) if fam != "laguerre" else rng.choice([n - 1, n, n + 1])
    if k < 0:
        raise Retry
    nm, s = NAME[fam]
    _, _, wtex, _, const = WEIGHTS[fam]
    xk = [F(0)] * k + [F(1)]
    val = inner(xk, POLY[fam](n), fam)
    if k >= n and val == 0:
        raise Retry
    ok = with_const(val, const)
    cands = [(with_const(norm2(fam, n), const), rf"これは $({s}_{n},{s}_{n})_w$。$x^{k}$ と ${s}_{n}$ の積分とは違う。"),
             (tex_fr(val) if const else with_const(val, r"\pi"), "くくり出される定数(ガウス積分 $\\sqrt\\pi$ など)を確かめよう。" if const else "定数 $\\pi$ は出てこない。"),
             ("0" if val != 0 else with_const(norm2(fam, n), const), rf"$k<n$ のときだけ直交性で $0$ になる。" if val != 0 else rf"$x^{k}$ は ${s}_0,\dots,{s}_{k}$ で表せる。$k<n$ なら直交性で $0$。"),
             (with_const(-val, const) if val else with_const(1, const), "符号を確かめよう。" if val else "直交性を使う。"),
             (with_const(val * 2, const) if val else with_const(2, const), "モーメントの値を確かめよう。")]
    a, b = BOUNDS[fam]
    mom = {"legendre": r"\int_{-1}^1x^{m}dx=\dfrac{2}{m+1}\ (m\ \text{偶数})", "chebyshev": r"\int_{-1}^1\dfrac{x^{2j}}{\sqrt{1-x^2}}dx=\pi\dfrac{(2j-1)!!}{(2j)!!}",
           "hermite": r"\int_{-\infty}^{\infty}x^{2j}e^{-x^2}dx=\sqrt\pi\dfrac{(2j-1)!!}{2^j}", "laguerre": r"\int_0^\infty x^me^{-x}dx=m!"}[fam]
    return Problem("重みつきの積分", "標準",
                   rf"${s}_{{{n}}}(x)={p_tex(POLY[fam](n))}$({nm})のとき, $\displaystyle\int_{{{a}}}^{{{b}}}x^{{{k}}}{s}_{{{n}}}(x)\,{wtex}\,dx$ を求めよ。"
                   if k > 1 else rf"${s}_{{{n}}}(x)={p_tex(POLY[fam](n))}$({nm})のとき, $\displaystyle\int_{{{a}}}^{{{b}}}{'x' if k == 1 else ''}{s}_{{{n}}}(x)\,{wtex}\,dx$ を求めよ。",
                   distinct(ok, cands), hints=[rf"${mom}$ を使う。", rf"$k<n$ なら, $x^k$ は ${s}_0,\dots,{s}_k$ で表せるので直交性から $0$。"],
                   solution=[rf"展開して ${mom}$ を使うと ${ok}$。" + ("(直交性から $0$ になることもわかる。)" if val == 0 else "")],
                   meta=dict(fam=fam, n=n, k=k, val=val))


# ------------------------------------------------------------
# 標準:グラム・シュミットの直交化
# ------------------------------------------------------------
GS_W = ["legendre", "hermite", "laguerre", "unit01", "x01", "chebyshev"]
GS_DESC = {"legendre": r"区間 $[-1,1]$, 重み $w(x)=1$", "chebyshev": r"区間 $[-1,1]$, 重み $w(x)=\dfrac{1}{\sqrt{1-x^2}}$",
           "hermite": r"区間 $(-\infty,\infty)$, 重み $w(x)=e^{-x^2}$", "laguerre": r"区間 $[0,\infty)$, 重み $w(x)=e^{-x}$",
           "unit01": r"区間 $[0,1]$, 重み $w(x)=1$", "x01": r"区間 $[0,1]$, 重み $w(x)=x$"}
GS_MOM = {"legendre": r"m_k=\dfrac{2}{k+1}\ (k\ \text{偶数}),\ 0\ (k\ \text{奇数})", "chebyshev": r"m_0=\pi,\ m_2=\dfrac{\pi}{2},\ m_4=\dfrac{3\pi}{8},\ m_{\text{奇数}}=0",
          "hermite": r"m_0=\sqrt\pi,\ m_2=\dfrac{\sqrt\pi}{2},\ m_4=\dfrac{3\sqrt\pi}{4},\ m_{\text{奇数}}=0", "laguerre": r"m_k=k!",
          "unit01": r"m_k=\dfrac{1}{k+1}", "x01": r"m_k=\dfrac{1}{k+2}"}


def g_gs(rng):
    w = rng.choice(GS_W)
    n = rng.choice([1, 2]) if w in ("laguerre", "unit01", "x01") else 2
    ps = gram_schmidt(n, w)
    ok = p_tex(ps[n])
    m = WEIGHTS[w][3]
    others = [p_tex(gram_schmidt(n, v)[n]) for v in GS_W if v != w]
    wrong1 = p_tex(p_add([F(0)] * n + [F(1)], [-m(n)])) if n == 2 else p_tex([-m(1), 1])
    cands = [(wrong1, "$(x^n,1)_w$ を $(1,1)_w=m_0$ で割るのを忘れていないか。"),
             (p_tex(p_add(ps[n], p_scale(ps[n - 1], 2))) if n == 2 else p_tex([m(1) / m(0), 1]), "引く成分の符号を確かめよう。")] + \
            [(t, "別の重みで直交化した結果。モーメントを確かめよう。") for t in others]
    if n == 2 and w in ("unit01", "x01", "laguerre"):
        cands.insert(1, (p_tex([-(m(2) / m(0)), 0, 1]), "$p_1$ の成分も引く必要がある(区間が対称でないので $(x^2,p_1)_w\\neq0$)。"))
    return Problem("グラム・シュミットの直交化", "標準",
                   rf"{GS_DESC[w]} の内積 $(f,g)_w=\int fg\,w\,dx$ で $1,x{',x^2' if n == 2 else ''}$ を直交化し, 最高次の係数が $1$ の多項式 $p_0=1,\ p_1{',\\ p_2' if n == 2 else ''}$ を作る。"
                   rf"$p_{{{n}}}(x)$ を求めよ。ただし $m_k=\int x^kw\,dx$ は ${GS_MOM[w]}$ である。",
                   distinct(ok, cands), hints=[r"$p_n=x^n-\sum_{j<n}\dfrac{(x^n,p_j)_w}{(p_j,p_j)_w}p_j$。内積はモーメント $m_k$ で計算できる。"],
                   solution=[rf"$p_1=x-\dfrac{{m_1}}{{m_0}}={p_tex(ps[1])}$。"] +
                            ([rf"$p_2=x^2-\dfrac{{(x^2,1)_w}}{{(1,1)_w}}-\dfrac{{(x^2,p_1)_w}}{{(p_1,p_1)_w}}p_1={ok}$。"] if n == 2 else []),
                   meta=dict(w=w, n=n))


# ------------------------------------------------------------
# 基本:ウォルシュ係数
# ------------------------------------------------------------
def g_walsh(rng):
    vals = [rng.choice([-2, -1, 0, 1, 2, 3, 4]) for _ in range(4)]
    co = walsh_coefs(vals)
    n = rng.choice([1, 2, 3])
    W = walsh_matrix(4)
    val = co[n]
    ok = tex_fr(val)
    had = [sum(int(r[k]) * F(vals[k]) for k in range(4)) / 4 for r in [[1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]]
    cands = [(tex_fr(val * 4), r"$\int_0^1$ なので, 4つの区間の幅 $\frac14$ を掛ける。"),
             (tex_fr(co[n % 3 + 1]), "交番数の順に並べた番号を確かめよう。"), (tex_fr(had[n - 1]), "交番数の順に並べた番号を確かめよう。"),
             (tex_fr(-val), "符号を確かめよう。"), (tex_fr(co[0]), "これは $c_0$(平均値)。")]
    rows = ",\\ ".join(f"w_{k}=({','.join('1' if v > 0 else '-1' for v in W[k])})" for k in range(4))
    return Problem("ウォルシュ係数", "基本",
                   rf"$[0,1)$ を4等分した区間で順に $f={','.join(map(str, vals))}$ となる階段関数 $f$ がある。ウォルシュ関数の値が(区間の順に)${rows}$ のとき, "
                   rf"$c_{{{n}}}=\displaystyle\int_0^1f(x)w_{{{n}}}(x)\,dx$ を求めよ。",
                   distinct(ok, cands), hints=[r"区間の幅は $\frac14$。$c_n=\frac14\sum_k f_kw_n(k)$。"],
                   solution=[rf"$c_{{{n}}}=\frac14\left(" + "+".join(f"({vals[k]})({int(W[n][k])})" for k in range(4)) + rf"\right)={ok}$。"],
                   meta=dict(vals=vals, n=n, val=val))


# ------------------------------------------------------------
# 誘導
# ------------------------------------------------------------
def guided_series(rng):
    lam = rng.choice([2, 3, 4])
    a0, a1 = (1, 0) if lam % 2 == 0 else (0, 1)
    a = series("hermite", F(lam), a0, a1, lam + 3)
    poly = trim(a)
    return ExamItem("級数解法でエルミートの方程式の多項式解を求める",
                    r"微分方程式 $y^{\prime\prime}-2xy^{\prime}+2\lambda y=0$ の解を $y=\sum_{k=0}^{\infty}a_kx^k$ とおく. 次の問いに答えよ.",
                    [Part(r"$y^{\prime}$, $y^{\prime\prime}$ を級数で表して方程式に代入し, $a_{k+2}$ を $a_k$ で表せ.", r"a_{k+2}=\dfrac{2(k-\lambda)}{(k+2)(k+1)}a_k",
                          [r"$y^{\prime\prime}=\sum(k+2)(k+1)a_{k+2}x^k$, $xy^{\prime}=\sum ka_kx^k$。", r"$x^k$ の係数から $(k+2)(k+1)a_{k+2}-2ka_k+2\lambda a_k=0$。"]),
                     Part(rf"$\lambda={lam}$, $a_0={a0}$, $a_1={a1}$ のときの解 $y$ を求めよ.", p_tex(poly),
                          ["$" + ",\\ ".join(f"a_{{{k}}}={tex_fr(a[k])}" for k in range(lam % 2, lam + 3, 2)) + "$。",
                           rf"$a_{{{lam + 2}}}=0$ より先はすべて $0$ で, $y={p_tex(poly)}$(多項式解)。"])], meta=dict(lam=lam))


def guided_gs(rng):
    w = rng.choice(["unit01", "x01", "laguerre"])
    ps = gram_schmidt(2, w)
    m = WEIGHTS[w][3]
    return ExamItem("グラム・シュミットの直交化",
                    rf"{GS_DESC[w]} の内積 $(f,g)_w=\int fg\,w\,dx$ を考える. $m_k=\int x^kw\,dx$ は ${GS_MOM[w]}$ である. "
                    r"$1,x,x^2$ を直交化して, 最高次の係数が $1$ の多項式 $p_0=1,\ p_1,\ p_2$ を作る. 次の問いに答えよ.",
                    [Part(r"$p_1(x)=x-\dfrac{(x,1)_w}{(1,1)_w}$ を求めよ.", p_tex(ps[1]), [rf"$\dfrac{{m_1}}{{m_0}}={tex_fr(m(1) / m(0))}$ より $p_1={p_tex(ps[1])}$。"]),
                     Part(r"$p_2(x)=x^2-\dfrac{(x^2,1)_w}{(1,1)_w}-\dfrac{(x^2,p_1)_w}{(p_1,p_1)_w}p_1$ を求めよ.", p_tex(ps[2]),
                          [rf"$(x^2,1)_w=m_2={tex_fr(m(2))}$, $(x^2,p_1)_w={tex_fr(inner([0, 0, 1], ps[1], w))}$, $(p_1,p_1)_w={tex_fr(inner(ps[1], ps[1], w))}$。",
                           rf"$p_2={p_tex(ps[2])}$。"])], meta=dict(w=w))


def guided_walsh(rng):
    while True:   # 誘導問題は Retry なしで呼ばれる
        vals = [rng.choice([-2, -1, 0, 1, 2, 3, 4, 5]) for _ in range(4)]
        co = walsh_coefs(vals)
        if co[1] != 0 and len(set(vals)) > 2:
            break
    W = walsh_matrix(4)
    rows = ",\\ ".join(f"w_{k}=({','.join('1' if v > 0 else '-1' for v in W[k])})" for k in range(4))
    half = [co[0] + co[1]] * 2 + [co[0] - co[1]] * 2
    return ExamItem("ウォルシュ展開",
                    rf"$[0,1)$ を4等分した区間で順に $f={','.join(map(str, vals))}$ となる階段関数 $f$ がある. ウォルシュ関数の値は(区間の順に)${rows}$ である. 次の問いに答えよ.",
                    [Part(r"ウォルシュ係数 $c_n=\displaystyle\int_0^1f(x)w_n(x)\,dx$($n=0,1,2,3$)を求めよ.",
                          ",\\ ".join(f"c_{n}={tex_fr(co[n])}" for n in range(4)), [r"$c_n=\frac14\sum_kf_kw_n(k)$ で計算する。"]),
                     Part(r"最初の2項の和 $c_0w_0+c_1w_1$ の, 4つの区間での値を求めよ.", "(" + ",".join(tex_fr(v) for v in half) + ")",
                          [rf"左半分は $c_0+c_1={tex_fr(half[0])}$, 右半分は $c_0-c_1={tex_fr(half[2])}$。", "これは左半分・右半分それぞれでの $f$ の平均値になっている。"])],
                    meta=dict(vals=vals))


REGISTRY = [
    ProblemType("or_rec", "直交多項式の漸化式", "基本", g_rec),
    ProblemType("or_cheb", "チェビシェフ多項式の値", "基本", g_cheb),
    ProblemType("or_series", "級数解法の係数", "基本", g_series, guided=guided_series),
    ProblemType("or_walsh", "ウォルシュ係数", "基本", g_walsh, guided=guided_walsh),
    ProblemType("or_expand", "直交多項式による展開", "標準", g_expand),
    ProblemType("or_moment", "重みつきの積分", "標準", g_moment),
    ProblemType("or_gs", "グラム・シュミットの直交化", "標準", g_gs, guided=guided_gs),
]
GUIDED_NAMES = {"or_series": "漸化式 → 級数解法で多項式解(エルミートの方程式)",
                "or_walsh": "ウォルシュ係数 → 最初の2項の和",
                "or_gs": "p₁ → p₂(グラム・シュミットの直交化)"}


def tab():
    render("orth", REGISTRY, "特殊関数 第2章 直交関数系")
