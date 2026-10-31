"""数学IV ベクトル解析 発展:勾配降下法の演習

関数はすべて整数係数の2変数2次関数。η = 1/2, 1/4, 1/8 などで、計算は整数か簡単な分数で終わる。
  基本:1ステップ後の点 / 最も急に下る向きの単位ベクトルと方向微分係数
  標準:2ステップ後の点 / 近づく η の範囲 / 1ステップで最小点に着く η
  誘導(5点×2):A 勾配 → 1ステップと f の減少、B 倍率 → η の範囲
"""
from fractions import Fraction

from topics.ex_core import Choice, ExamItem, Part, Problem, ProblemType, Retry, lin_tex, render, tex_fr, vtex

F = Fraction
UPDATE = r"$\vec x_{k+1}=\vec x_k-\eta\,\nabla f(\vec x_k)$"


def quad_tex(a, c, b, d, e, k=0):
    """a x² + c xy + b y² + d x + e y + k"""
    return lin_tex([(a, "x^2"), (c, "xy"), (b, "y^2"), (d, "x"), (e, "y")], k)


def grad(a, c, b, d, e, P):
    x, y = P
    return (F(2 * a) * x + c * y + d, F(c) * x + 2 * b * y + e)


def fval(a, c, b, d, e, k, P):
    x, y = P
    return F(a) * x * x + F(c) * x * y + F(b) * y * y + F(d) * x + F(e) * y + k


def grad_tex(a, c, b, d, e):
    return (rf"\nabla f=\left({lin_tex([(2 * a, 'x'), (c, 'y')], d)},\ {lin_tex([(c, 'x'), (2 * b, 'y')], e)}\right)")


def step(P, g, eta):
    return (P[0] - eta * g[0], P[1] - eta * g[1])


def pick(correct, cands):
    choices = [Choice(vtex(correct), True, value=tuple(float(x) for x in correct))]
    seen = [tuple(correct)]
    for v, why in cands:
        v = tuple(F(x) for x in v)
        if v in seen:
            continue
        choices.append(Choice(vtex(v), why=why, value=tuple(float(x) for x in v)))
        seen.append(v)
        if len(choices) == 4:
            return choices
    raise Retry


def eta_tex(eta):
    return tex_fr(eta)


def one_step_parts(rng, need_decrease=True):
    a, b = rng.choice([1, 2, 3]), rng.choice([1, 2, 3])
    c = rng.choice([0, 0, 0, 1, -1, 2])
    d, e = rng.randint(-4, 4), rng.randint(-4, 4)
    eta = rng.choice([F(1, 2), F(1, 4), F(1, 4), F(1, 8)])
    if 4 * a * b - c * c <= 0:
        raise Retry
    P = (rng.randint(-3, 3), rng.randint(-3, 3))
    g = grad(a, c, b, d, e, P)
    if g == (0, 0) or 0 in g and rng.random() < 0.7:
        raise Retry
    P1 = step(P, g, eta)
    f0, f1 = fval(a, c, b, d, e, 0, P), fval(a, c, b, d, e, 0, P1)
    if need_decrease and not f1 < f0:
        raise Retry
    return a, c, b, d, e, eta, P, g, P1, f0, f1


def one_step_solution(a, c, b, d, e, eta, P, g, P1):
    return [rf"${grad_tex(a, c, b, d, e)}$ より $\nabla f(\vec x_0)={vtex(g)}$。",
            rf"$$\vec x_1=\vec x_0-\eta\nabla f(\vec x_0)={vtex(P)}-{eta_tex(eta)}{vtex(g)}={vtex(P1)}$$"]


def g_one_step(rng):
    a, c, b, d, e, eta, P, g, P1, f0, f1 = one_step_parts(rng)
    bad_g = grad(F(a, 2), F(c), F(b, 2), d, e, P)          # x² の微分を x としてしまう
    cands = [(step(P, g, -eta), "それは $+\\nabla f$(最も急に登る向き)に進んだ点。下るには $-\\nabla f$ の向き。"),
             (step(P, g, 1), "ステップ幅 $\\eta$ をかけ忘れていないか。"),
             (step(P, bad_g, eta), "偏微分を確かめよう:$\\dfrac{\\partial}{\\partial x}x^2=2x$。"),
             (step(P, (g[1], g[0]), eta), "$\\nabla f=\\left(\\dfrac{\\partial f}{\\partial x},\\ \\dfrac{\\partial f}{\\partial y}\\right)$ の成分の順を確かめよう。")]
    return Problem("勾配降下法の1ステップ", "基本",
                   rf"関数 $f(x,y)={quad_tex(a, c, b, d, e)}$ に、出発点 $\vec x_0={vtex(P)}$, ステップ幅 $\eta={eta_tex(eta)}$ で"
                   rf"勾配降下法を1回使ったときの点 $\vec x_1$ を求めよ。",
                   pick(P1, cands), hints=["勾配降下法の更新式:" + UPDATE,
                                           r"まず $\nabla f=\left(\dfrac{\partial f}{\partial x},\ \dfrac{\partial f}{\partial y}\right)$ を計算し、$\vec x_0$ を代入する。"],
                   solution=one_step_solution(a, c, b, d, e, eta, P, g, P1) +
                   [rf"(確かめ:$f(\vec x_0)={tex_fr(f0)}$, $f(\vec x_1)={tex_fr(f1)}$ で、値が下がっている。)"],
                   meta=dict(coef=(a, c, b, d, e), eta=eta, P=P, n=1))


TRIPLES = [(3, 4), (4, 3), (5, 12), (12, 5), (6, 8), (8, 6), (0, 5), (5, 0), (0, 3), (4, 0)]


def g_descent_dir(rng):
    gx, gy = rng.choice(TRIPLES)
    gx, gy = gx * rng.choice([1, -1]), gy * rng.choice([1, -1])
    L = int(round((gx * gx + gy * gy) ** 0.5))
    a, b = rng.choice([1, 2, 3]), rng.choice([1, 2, 3])
    P = (rng.randint(-2, 2), rng.randint(-2, 2))
    d, e = gx - 2 * a * P[0], gy - 2 * b * P[1]
    g = (F(gx), F(gy))
    u = (-g[0] / L, -g[1] / L)

    def ch(v, m):
        return rf"\vec u={vtex(v)},\ \text{{最小値}}\ {tex_fr(m)}"
    cands = [(ch((-x for x in u), -L), "それは最も急に**登る**向き $\\dfrac{\\nabla f}{|\\nabla f|}$。下る向きは $-\\nabla f$。"),
             (ch(u, -L * L), "方向微分係数の最小値は $-|\\nabla f|$($|\\nabla f|^2$ ではない)。"),
             (ch(u, L), "最も急に下る向きでは方向微分係数は負になる:$\\nabla f\\cdot\\vec u=-|\\nabla f|$。"),
             (ch((-g[0], -g[1]), -L), "$\\vec u$ は単位ベクトル。$-\\nabla f$ を $|\\nabla f|$ で割る。")]
    ok = ch(u, -L)
    choices = [Choice(ok, True, value=(float(u[0]), float(u[1]), -L))]
    for t, why in cands:
        if t != ok and all(c.tex != t for c in choices):
            choices.append(Choice(t, why=why, value=t))
        if len(choices) == 4:
            break
    if len(choices) < 4:
        raise Retry
    return Problem("最も急に下る向き", "基本",
                   rf"関数 $f(x,y)={quad_tex(a, 0, b, d, e)}$ について, 点 ${vtex(P)}$ で $f$ が最も急に減少する向きの単位ベクトル $\vec u$ と, "
                   r"そのときの方向微分係数 $\nabla f\cdot\vec u$(最小値)を求めよ。",
                   choices, hints=[r"方向微分係数 $\nabla f\cdot\vec u=|\nabla f|\cos\theta$ が最小になるのは $\theta=180^\circ$。",
                                   r"$\vec u=-\dfrac{\nabla f}{|\nabla f|}$, 最小値は $-|\nabla f|$。"],
                   solution=[rf"${grad_tex(a, 0, b, d, e)}$ より $\nabla f{vtex(P)}={vtex(g)}$, $|\nabla f|={L}$。",
                             rf"最も急に減少する向きは $\vec u=-\dfrac{{\nabla f}}{{|\nabla f|}}={vtex(u)}$, "
                             rf"方向微分係数は $\nabla f\cdot\vec u=-|\nabla f|={-L}$。"],
                   meta=dict(coef=(a, 0, b, d, e), P=P, u=u, L=L))


# ------------------------------------------------------------
# 分かれた2次関数 f = a(x−p)² + b(y−q)²(倍率で考える型)
# ------------------------------------------------------------
def sep_tex(a, p, b, q):
    def sq(v, s):
        return f"{v}^2" if s == 0 else rf"({v}{'-' if s > 0 else '+'}{abs(s)})^2"
    return lin_tex([(a, sq("x", p)), (b, sq("y", q))], 0)


def g_two_steps(rng):
    a, b = rng.sample([1, 2, 3], 2)
    p, q = rng.randint(-2, 2), rng.randint(-2, 2)
    eta = rng.choice([F(1, 4), F(1, 8)])
    P = (p + rng.choice([-4, 4, -8, 8]), q + rng.choice([-4, 4, -8, 8]))
    coef = (a, 0, b, -2 * a * p, -2 * b * q)
    g0 = grad(*coef, P)
    P1 = step(P, g0, eta)
    P2 = step(P1, grad(*coef, P1), eta)
    cands = [(step(P1, g0, eta), "2回目は、新しい点 $\\vec x_1$ での勾配 $\\nabla f(\\vec x_1)$ を使う。"),
             (P1, "それは1ステップ後の点 $\\vec x_1$。もう1回くり返す。"),
             (step(step(P, g0, -eta), grad(*coef, step(P, g0, -eta)), -eta), "$-\\nabla f$ の向きに進む。"),
             (step(P, g0, 2 * eta), "2回目は $\\vec x_1$ での勾配を計算し直す($\\vec x_0$ での勾配を2倍するのではない)。")]
    rx, ry = 1 - 2 * eta * a, 1 - 2 * eta * b
    return Problem("勾配降下法の2ステップ", "標準",
                   rf"関数 $f(x,y)={sep_tex(a, p, b, q)}$ に, 出発点 $\vec x_0={vtex(P)}$, ステップ幅 $\eta={eta_tex(eta)}$ で"
                   r"勾配降下法を2回使ったときの点 $\vec x_2$ を求めよ。",
                   pick(P2, cands), hints=["更新式:" + UPDATE + "。1回ごとに、その点での勾配を計算し直す。",
                                           r"$\nabla f=(2a(x-p),\ 2b(y-q))$ の形。"],
                   solution=[rf"$\nabla f=\left({lin_tex([(2 * a, 'x')], -2 * a * p)},\ {lin_tex([(2 * b, 'y')], -2 * b * q)}\right)$。"
                             rf"$\nabla f(\vec x_0)={vtex(g0)}$ より $\vec x_1={vtex(P1)}$。",
                             rf"$\nabla f(\vec x_1)={vtex(grad(*coef, P1))}$ より $\vec x_2={vtex(P2)}$。",
                             rf"(1歩ごとに、最小点 $({p},\ {q})$ からのずれが $x$ 方向は ${tex_fr(rx)}$ 倍、$y$ 方向は ${tex_fr(ry)}$ 倍になっている。)"],
                   meta=dict(coef=coef, eta=eta, P=P, n=2))


def range_parts(rng):
    a, b = rng.sample([1, 2, 3, 4, 5, 6], 2)
    p, q = rng.randint(-3, 3), rng.randint(-3, 3)
    return a, b, p, q


def range_tex(ub):
    return rf"0<\eta<{tex_fr(ub)}"


def g_eta_range(rng):
    a, b, p, q = range_parts(rng)
    M, m = max(a, b), min(a, b)
    ok = F(1, M)
    cands = [(F(1, m), "急な方向(係数の大きい方)で上限が決まる。両方の条件を満たす範囲を考える。"),
             (F(2, M), "$\\nabla f$ の成分は $2a(x-p)$。倍率は $1-2\\eta a$ で、$|1-2\\eta a|<1\\iff0<\\eta<\\frac1a$。"),
             (F(1, 2 * M), "$|1-2\\eta a|<1$ は $-1<1-2\\eta a<1$。$1-2\\eta a$ が負でもよい(符号を変えながら近づく)。"),
             (F(1, 2 * m), "急な方向(係数の大きい方)で上限が決まる。$|1-2\\eta a|<1$ を正しく解こう。")]
    choices = [Choice(range_tex(ok), True, value=float(ok))]
    for v, why in cands:
        if all(abs(c.value - float(v)) > 1e-12 for c in choices):
            choices.append(Choice(range_tex(v), why=why, value=float(v)))
        if len(choices) == 4:
            break
    if len(choices) < 4:
        raise Retry
    return Problem("近づくステップ幅の範囲", "標準",
                   rf"関数 $f(x,y)={sep_tex(a, p, b, q)}$ に勾配降下法を使う。どんな出発点からでも, 点が最小点 $({p},\ {q})$ に"
                   r"近づくための $\eta$ の範囲を求めよ。",
                   choices, hints=[r"1歩で、最小点からのずれ $x_k-p$ が何倍になるかを求める。",
                                   r"同じ数を何回もかけて $0$ に近づくのは、その数の絶対値が $1$ より小さいとき。"],
                   solution=range_solution(a, b, p, q), meta=dict(a=a, b=b, ub=ok))


def range_solution(a, b, p, q):
    M = max(a, b)
    return [rf"$\nabla f=(2\cdot{a}(x{'-' if p > 0 else '+'}{abs(p)}),\ 2\cdot{b}(y{'-' if q > 0 else '+'}{abs(q)}))$ より".replace("+0", "").replace("-0", ""),
            rf"$x_{{k+1}}-p=(1-{2 * a}\eta)(x_k-p),\ y_{{k+1}}-q=(1-{2 * b}\eta)(y_k-q)$($p={p},\ q={q}$)。",
            rf"近づく条件は $|1-{2 * a}\eta|<1$ かつ $|1-{2 * b}\eta|<1$, つまり $0<\eta<{tex_fr(F(1, a))}$ かつ $0<\eta<{tex_fr(F(1, b))}$。",
            rf"両方を満たすのは $${range_tex(F(1, M))}$$"]


def g_one_shot(rng):
    a = rng.choice([1, 2, 3, 4])
    p, q = rng.randint(-3, 3), rng.randint(-3, 3)
    k = rng.randint(-3, 3)
    ok = F(1, 2 * a)
    cands = [(F(1, a), "$\\nabla f=2a(x-p,\\ y-q)$。$2$ を忘れていないか。"), (F(2 * a), "$1-2a\\eta=0$ を解く。"),
             (F(1, 4 * a), "$1-2a\\eta=0$ を解く。"), (F(a, 2), "$1-2a\\eta=0$ を $\\eta$ について解こう。")]
    choices = [Choice(rf"\eta={tex_fr(ok)}", True, value=float(ok))]
    for v, why in cands:
        if all(abs(c.value - float(v)) > 1e-12 for c in choices):
            choices.append(Choice(rf"\eta={tex_fr(v)}", why=why, value=float(v)))
        if len(choices) == 4:
            break
    fstr = (("" if a == 1 else str(a)) + r"\left\{" + sep_tex(1, p, 1, q) + r"\right\}") + (f"{'+' if k > 0 else '-'}{abs(k)}" if k else "")
    return Problem("1ステップで最小点に着くステップ幅", "標準",
                   rf"関数 $f(x,y)={fstr}$ に勾配降下法を使う。どんな出発点からでも1回で最小点に着く $\eta$ を求めよ。",
                   choices, hints=[r"1歩で、最小点からのずれが何倍になるかを求める。それが $0$ 倍になればよい。"],
                   solution=[rf"$\nabla f={2 * a}(x-p,\ y-q)$($p={p},\ q={q}$)より、"
                             rf"$\vec x_1-(p,q)=(1-{2 * a}\eta)(\vec x_0-(p,q))$。",
                             rf"$1-{2 * a}\eta=0$ より $\eta={tex_fr(ok)}$。丸い谷では $-\nabla f$ がちょうど最小点を向くので、1歩で着ける。"],
                   meta=dict(a=a, p=p, q=q, eta=ok))


# ------------------------------------------------------------
# 誘導問題(5点×2)
# ------------------------------------------------------------
def guided_one_step(rng):
    a, c, b, d, e, eta, P, g, P1, f0, f1 = one_step_parts(rng)
    sol = one_step_solution(a, c, b, d, e, eta, P, g, P1)
    return ExamItem("勾配降下法",
                    rf"関数 $f(x,y)={quad_tex(a, c, b, d, e)}$ に, 出発点 $\vec x_0={vtex(P)}$, ステップ幅 $\eta={eta_tex(eta)}$ で"
                    r"勾配降下法 $\vec x_{k+1}=\vec x_k-\eta\nabla f(\vec x_k)$ を使う. 次の問いに答えよ.",
                    [Part(r"$\nabla f(\vec x_0)$ を求めよ.", vtex(g), sol[:1]),
                     Part(r"$\vec x_1$ を求め, $f(\vec x_1)<f(\vec x_0)$ であることを確かめよ.", vtex(P1),
                          sol[1:] + [rf"$f(\vec x_0)={tex_fr(f0)}$, $f(\vec x_1)={tex_fr(f1)}$ なので $f(\vec x_1)<f(\vec x_0)$."])],
                    meta=dict(coef=(a, c, b, d, e), eta=eta, P=P, P1=P1))


def guided_range(rng):
    a, b, p, q = range_parts(rng)
    sol = range_solution(a, b, p, q)
    return ExamItem("勾配降下法のステップ幅",
                    rf"関数 $f(x,y)={sep_tex(a, p, b, q)}$ に, ステップ幅 $\eta>0$ の勾配降下法 "
                    r"$\vec x_{k+1}=\vec x_k-\eta\nabla f(\vec x_k)$ を使う. $\vec x_k=(x_k,\ y_k)$ とする. 次の問いに答えよ.",
                    [Part(rf"$x_{{k+1}}{'-' if p > 0 else '+'}{abs(p)}$ を $x_k{'-' if p > 0 else '+'}{abs(p)}$ で, "
                          rf"$y_{{k+1}}{'-' if q > 0 else '+'}{abs(q)}$ を $y_k{'-' if q > 0 else '+'}{abs(q)}$ で表せ.".replace("+0", "").replace("-0", ""),
                          rf"(1-{2 * a}\eta)\ \text{{倍}},\ (1-{2 * b}\eta)\ \text{{倍}}", sol[:2]),
                     Part(r"どんな出発点からでも点が最小点に近づく $\eta$ の範囲を求めよ.", range_tex(F(1, max(a, b))), sol[2:])],
                    meta=dict(a=a, b=b))


REGISTRY = [
    ProblemType("gd_step", "勾配降下法の1ステップ", "基本", g_one_step, guided=guided_one_step),
    ProblemType("gd_dir", "最も急に下る向き", "基本", g_descent_dir),
    ProblemType("gd_two", "勾配降下法の2ステップ", "標準", g_two_steps),
    ProblemType("gd_range", "近づくステップ幅の範囲", "標準", g_eta_range, guided=guided_range),
    ProblemType("gd_shot", "1ステップで最小点に着くステップ幅", "標準", g_one_shot),
]
GUIDED_NAMES = {"gd_step": "勾配 → 1ステップと f の減少(勾配降下法)", "gd_range": "倍率 → ステップ幅の範囲(勾配降下法)"}


def tab():
    render("gradient", REGISTRY, "ベクトル解析 発展 勾配降下法")
