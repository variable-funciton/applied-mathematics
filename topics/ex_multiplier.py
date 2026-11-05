"""数学IV フーリエ解析 発展 フーリエマルチプライヤー:演習の問題生成(簡単なパターン)

規約:F(ω) = ∫ f(t) e^{−iωt} dt,  T_m f = F^{-1}[m F] = K * f,  T_m e^{iω₀t} = m(ω₀) e^{iω₀t}。
"""
from fractions import Fraction

from topics.ex_core import Choice, ExamItem, Part, Problem, ProblemType, Retry, render, tex_fr

F = Fraction
BASIC = r"$e^{i\omega_0t}$ に $T_m$ を作用させると $m(\omega_0)e^{i\omega_0t}$ になる。$\cos\omega_0t=\frac12(e^{i\omega_0t}+e^{-i\omega_0t})$。"


def distinct(ok, cands):
    choices = [Choice(ok, True, value=ok)]
    for t, why in cands:
        if all(c.tex != t for c in choices) and len(choices) < 4:
            choices.append(Choice(t, why=why, value=t))
    if len(choices) < 4:
        raise Retry
    return choices


def k_(c, s):
    """係数 c と式 s の積(c = 1, −1 を省く)"""
    return s if c == 1 else ("-" + s if c == -1 else f"{c}{s}")


def arg(w0):
    return "t" if w0 == 1 else f"{w0}t"


# ------------------------------------------------------------
# 基本:cos ω₀t に作用させる
# ------------------------------------------------------------
def g_cos(rng):
    w0 = rng.choice([1, 2, 3])
    a = rng.choice([1, 2])
    th = arg(w0)
    ith = "it" if w0 == 1 else f"{w0}it"
    cos, sin = rf"\cos {th}", rf"\sin {th}"
    kind = rng.choice(["d", "dd", "hil", "abs", "shift"])
    if kind == "d":
        mt, ok, sol = r"i\omega", k_(-w0, sin), rf"$\frac12\{{i{w0}e^{{{ith}}}+(-i{w0})e^{{-{ith}}}\}}=-{w0}\sin {th}$(微分 $f'$ と同じ)"
        cands = [(k_(w0, sin), r"$\frac12(i\omega_0e^{i\omega_0t}-i\omega_0e^{-i\omega_0t})=-\omega_0\sin\omega_0t$。符号に注意。"),
                 (k_(-w0, cos), r"$i$ がかかると $\cos$ が $\sin$ に変わる。"),
                 (k_(-1, sin), r"$m(\omega_0)=i\omega_0$ の $\omega_0$ をかけ忘れていないか。"),
                 (k_(w0, cos), r"$i\omega$ は微分:$(\cos\omega_0t)'=-\omega_0\sin\omega_0t$。")]
    elif kind == "dd":
        mt, ok, sol = r"-\omega^2", k_(-w0 * w0, cos), rf"$m(\pm{w0})=-{w0 * w0}$ なので $-{w0 * w0}\cos {th}$(2階微分 $f''$)"
        cands = [(k_(w0 * w0, cos), r"$m(\omega_0)=-\omega_0^2$ は負。"), (k_(-w0 * w0, sin), r"$m$ は実数なので $\cos$ のまま。"),
                 (k_(-w0, cos), r"$\omega_0^2$ をかける。"), (k_(-2 * w0, cos), r"$m(\omega_0)=-\omega_0^2$ を確かめよう。")]
    elif kind == "hil":
        mt, ok, sol = r"-i\,\mathrm{sgn}\,\omega", sin, rf"$\frac12\{{(-i)e^{{{ith}}}+i\,e^{{-{ith}}}\}}=\sin {th}$(ヒルベルト変換)"
        cands = [(k_(-1, sin), r"$m(\omega_0)=-i$, $m(-\omega_0)=i$ を確かめよう:$\frac{-ie^{i\theta}+ie^{-i\theta}}{2}=\sin\theta$。"),
                 (k_(-1, cos), r"$-i$ がかかると $\cos$ は $\sin$ になる。"), (cos, r"$m$ は $\omega$ の符号で値が変わる($\pm i$)。"),
                 (k_(w0, sin), r"$|m|=1$ なので振幅は変わらない。")]
    elif kind == "abs":
        mt, ok, sol = r"|\omega|", k_(w0, cos), rf"$m(\pm{w0})={w0}$ なので ${k_(w0, cos)}$"
        cands = [(k_(-w0, sin), r"$|\omega|$ は実数で偶関数なので, $\cos$ のまま。"), (cos, r"$|\omega_0|$ をかける。"),
                 (k_(w0 * w0, cos), r"$|\omega|$ であって $\omega^2$ ではない。"), (k_(-w0, cos), r"$|\omega_0|>0$。")]
    else:
        mt = rf"e^{{-{'' if a == 1 else a}i\omega}}"
        sh = w0 * a
        ok = rf"\cos({th}-{sh})"
        sol = rf"$m(\omega)=e^{{-i\omega a}}$($a={a}$)は時間推移:$T_mf(t)=f(t-{a})=\cos {w0 if w0 > 1 else ''}(t-{a})=\cos({th}-{sh})$"
        cands = [(rf"\cos({th}+{sh})", r"$e^{-i\omega a}$ は $f(t-a)$(右にずらす)。"),
                 (rf"\cos({th}-{a})" if w0 > 1 else rf"\cos({th}-{2 * a})", r"$\cos\omega_0(t-a)=\cos(\omega_0t-\omega_0a)$。"),
                 (rf"e^{{-{sh}}}\cos {th}" if sh > 1 else rf"e^{{-1}}\cos {th}", r"$|e^{-i\omega a}|=1$:大きさは変わらず, ずれるだけ。"),
                 (rf"\sin({th}-{sh})", r"$m$ を作用させても $\cos$ の形は変わらない(ずれるだけ)。")]
    return Problem("cos に作用させる", "基本",
                   rf"マルチプライヤー $m(\omega)={mt}$ の作用素 $T_m$ を, $f(t)=\cos {th}$ に作用させた $T_mf(t)$ を求めよ。",
                   distinct(ok, cands), hints=[BASIC, r"$m(\omega_0)$ と $m(-\omega_0)$ を計算する。"],
                   solution=[rf"$T_m\cos {th}=\frac12\{{m({w0})e^{{{ith}}}+m(-{w0})e^{{-{ith}}}\}}$。", sol + "。"], meta=dict(kind=kind, w0=w0, a=a))


# ------------------------------------------------------------
# 基本:マルチプライヤーが表す作用素
# ------------------------------------------------------------
OPS = [(r"i\omega", r"f'(t)"), (r"-\omega^2", r"f''(t)"), (r"(i\omega)^3", r"f'''(t)"),
       (r"e^{-2i\omega}", r"f(t-2)"), (r"e^{2i\omega}", r"f(t+2)"), (r"3", r"3f(t)"), (r"-i\,\mathrm{sgn}\,\omega", r"Hf(t)\ (\text{ヒルベルト変換})")]
WHY = {r"f'(t)": r"$(f')^\wedge=i\omega\hat f$(第2章の微分の公式)。", r"f''(t)": r"$f''$ は $(i\omega)^2=-\omega^2$。",
       r"f'''(t)": r"$f'''$ は $(i\omega)^3$。", r"f(t-2)": r"$f(t-a)$ は $e^{-i\omega a}\hat f$(時間推移)。",
       r"f(t+2)": r"$f(t+a)$ は $e^{i\omega a}\hat f$。", r"3f(t)": "定数倍のマルチプライヤーは定数倍。",
       r"Hf(t)\ (\text{ヒルベルト変換})": r"ヒルベルト変換のマルチプライヤーは $-i\,\mathrm{sgn}\,\omega$。"}


def g_op(rng):
    i = rng.randrange(len(OPS))
    mt, ok = OPS[i]
    others = [o for j, (_, o) in enumerate(OPS) if j != i]
    rng.shuffle(others)
    return Problem("マルチプライヤーと作用素", "基本",
                   rf"$\widehat{{T_mf}}(\omega)=m(\omega)\hat f(\omega)$, $m(\omega)={mt}$ のとき, $T_mf$ はどんな作用素か。"
                   r"(フーリエ変換は $\hat f(\omega)=\int f(t)e^{-i\omega t}dt$)",
                   distinct(ok, [(o, WHY[o]) for o in others]),
                   hints=[r"$(f')^\wedge(\omega)=i\omega\hat f(\omega)$, $(f(t-a))^\wedge=e^{-i\omega a}\hat f(\omega)$。", "部分積分・変数変換で確かめられる。"],
                   solution=[rf"$m(\omega)={mt}$ は ${ok}$ に対応する。", WHY[ok]], meta=dict(i=i))


# ------------------------------------------------------------
# 標準:マルチプライヤーから核を求める
# ------------------------------------------------------------
def g_kernel(rng):
    kind = rng.choice(["poisson", "lowpass", "rc", "heat"])
    c = rng.choice([1, 2, 3])
    ct = "" if c == 1 else str(c)
    if kind == "poisson":
        mt = rf"e^{{-{ct}|\omega|}}"
        ok = rf"\frac{{{c}}}{{\pi(t^2+{c * c})}}"
        cands = [(rf"\frac{{{c}}}{{t^2+{c * c}}}", r"逆変換の $\frac{1}{2\pi}$ を確かめよう。"),
                 (rf"\frac{{{2 * c}}}{{\pi(t^2+{c * c})}}", r"$\frac{1}{2\pi}\int e^{-c|\omega|}e^{i\omega t}d\omega=\frac{1}{2\pi}\cdot\frac{2c}{c^2+t^2}$。"),
                 (rf"\frac{{1}}{{\pi(t^2+{c * c})}}" if c > 1 else r"\frac{1}{2\pi(t^2+1)}", r"分子の $c$ を確かめよう。"),
                 (rf"e^{{-{ct}|t|}}", "核は m と同じ形とは限らない(逆フーリエ変換する)。")]
        sol = rf"$\frac{{1}}{{2\pi}}\int_{{-\infty}}^{{\infty}}e^{{-{ct}|\omega|}}e^{{i\omega t}}d\omega=\frac{{1}}{{2\pi}}\cdot\frac{{{2 * c}}}{{{c * c}+t^2}}={ok}$(ポアソン核)"
    elif kind == "lowpass":
        mt = rf"\begin{{cases}}1&(|\omega|\le{c})\\0&(|\omega|>{c})\end{{cases}}"
        ok = rf"\frac{{\sin {ct}t}}{{\pi t}}"
        cands = [(rf"\frac{{\sin {ct}t}}{{t}}", r"逆変換の $\frac{1}{2\pi}$ を確かめよう。"),
                 (rf"\frac{{2\sin {ct}t}}{{\pi t}}", r"$\int_{-c}^{c}e^{i\omega t}d\omega=\frac{2\sin ct}{t}$, これに $\frac1{2\pi}$。"),
                 (rf"\frac{{\cos {ct}t}}{{\pi t}}", r"$\frac{e^{ict}-e^{-ict}}{it}=\frac{2\sin ct}{t}$。"),
                 (rf"\frac{{\sin t}}{{{c}\pi t}}" if c > 1 else r"\frac{\sin 2t}{\pi t}", r"$\sin ct$ の $c$ を確かめよう。")]
        sol = rf"$\frac{{1}}{{2\pi}}\int_{{-{c}}}^{{{c}}}e^{{i\omega t}}d\omega=\frac{{1}}{{2\pi}}\cdot\frac{{2\sin {ct}t}}{{t}}={ok}$"
    elif kind == "rc":
        mt = rf"\frac{{{c}}}{{{c}+i\omega}}"
        ok = rf"{ct}e^{{-{ct}t}}u(t)"
        cands = [(rf"e^{{-{ct}t}}u(t)" if c > 1 else r"2e^{-t}u(t)", r"$\int_0^\infty ce^{-ct}e^{-i\omega t}dt=\frac{c}{c+i\omega}$ の $c$ を確かめよう。"),
                 (rf"{ct}e^{{{ct}t}}u(-t)", r"$\frac{1}{c+i\omega}$ は $t\ge0$ の $e^{-ct}$ の変換。"),
                 (rf"{ct}e^{{-{ct}|t|}}", r"$e^{-c|t|}$ の変換は $\frac{2c}{c^2+\omega^2}$(実数)。"),
                 (rf"\frac{{{ct}}}{{2\pi}}e^{{-{ct}t}}u(t)", r"フーリエ変換を直接計算して確かめよう($\frac1{2\pi}$ はつかない)。")]
        sol = rf"$\int_0^\infty {ct}e^{{-{ct}t}}e^{{-i\omega t}}dt=\frac{{{c}}}{{{c}+i\omega}}$ なので $K(t)={ok}$($u$ は単位ステップ関数)"
    else:
        mt = rf"e^{{-{ct}\omega^2}}"
        ok = rf"\frac{{1}}{{\sqrt{{{4 * c}\pi}}}}e^{{-t^2/{4 * c}}}"
        cands = [(rf"\frac{{1}}{{\sqrt{{{4 * c}\pi}}}}e^{{-t^2/{c}}}" if c > 1 else r"\frac{1}{\sqrt{4\pi}}e^{-t^2}", r"$e^{-s\omega^2}$ の核は $\frac{1}{\sqrt{4\pi s}}e^{-t^2/(4s)}$。"),
                 (rf"\frac{{1}}{{\sqrt{{{2 * c}\pi}}}}e^{{-t^2/{4 * c}}}", r"定数は $\frac{1}{\sqrt{4\pi s}}$。"),
                 (rf"e^{{-t^2/{4 * c}}}", r"$\frac{1}{2\pi}\int e^{-s\omega^2}e^{i\omega t}d\omega$ の定数を忘れていないか。"),
                 (rf"\frac{{1}}{{\sqrt{{{4 * c}\pi}}}}e^{{-{4 * c}t^2}}", r"$t^2$ は $4s$ で割る。")]
        sol = rf"熱核:$\frac{{1}}{{2\pi}}\int e^{{-s\omega^2}}e^{{i\omega t}}d\omega=\frac{{1}}{{\sqrt{{4\pi s}}}}e^{{-t^2/(4s)}}$ に $s={c}$ を代入して $K(t)={ok}$"
    return Problem("マルチプライヤーの核", "標準",
                   rf"マルチプライヤー $m(\omega)={mt}$ の核 $K=\mathcal F^{{-1}}m$, すなわち $K(t)=\dfrac{{1}}{{2\pi}}\displaystyle\int m(\omega)e^{{i\omega t}}d\omega$ を求めよ。",
                   distinct(ok, cands), hints=[r"$T_mf=K*f$, $K(t)=\frac{1}{2\pi}\int m(\omega)e^{i\omega t}d\omega$。",
                                               r"逆に, 答えの候補をフーリエ変換して $m$ になるか確かめてもよい。"],
                   solution=[sol + "。"], meta=dict(kind=kind, c=c))


# ------------------------------------------------------------
# 基本:L² での作用素ノルム sup|m|
# ------------------------------------------------------------
def g_norm(rng):
    kind = rng.choice(["lor", "odd", "gauss", "rc"])
    c = rng.choice([1, 2, 3, 4])
    if kind == "lor":
        b = rng.choice([1, 2, 4])
        mt, val, at = rf"\frac{{{c}}}{{{b}+\omega^2}}", F(c, b), r"\omega=0"
        cands = [(tex_fr(F(c, b + 1)), r"最大になるのは $\omega=0$。"), (str(c), r"$\omega=0$ で $\frac{c}{b}$。"), (tex_fr(F(c, 2 * b)), "計算を確かめよう。")]
    elif kind == "odd":
        mt, val, at = rf"\frac{{{c if c > 1 else ''}\omega}}{{1+\omega^2}}", F(c, 2), r"\omega=\pm1"
        cands = [(str(c), r"$\frac{\omega}{1+\omega^2}$ は $\omega=1$ で最大 $\frac12$。"), ("0", r"$m(0)=0$ は最小値。最大値を求める。"),
                 (tex_fr(F(c, 4)), r"$\left(\frac{\omega}{1+\omega^2}\right)'=\frac{1-\omega^2}{(1+\omega^2)^2}=0$ から $\omega=1$。")]
    elif kind == "gauss":
        mt, val, at = rf"{c if c > 1 else ''}\omega^2e^{{-\omega^2}}", None, r"\omega=\pm1"
        ok = rf"\frac{{{c}}}{{e}}"
        cands = [(str(c), r"$\omega^2e^{-\omega^2}$ は $\omega=\pm1$ で最大 $e^{-1}$。"), ("0", r"$m(0)=0$ は最小値。"),
                 (rf"\frac{{{c}}}{{e^2}}", r"$(\omega^2e^{-\omega^2})'=2\omega(1-\omega^2)e^{-\omega^2}$。")]
    else:
        mt, val, at = rf"\frac{{{c}}}{{1+i\omega}}", F(c), r"\omega=0"
        cands = [(tex_fr(F(c, 2)), r"$|m(\omega)|=\frac{c}{\sqrt{1+\omega^2}}$ の最大値。"), (rf"\frac{{{c}}}{{\sqrt2}}", r"$\omega=1$ での値ではなく, 最大値。"),
                 ("0", r"$\omega\to\infty$ の極限ではなく最大値。")]
    if val is not None:
        ok = tex_fr(val)
    return Problem("L² での作用素のノルム", "基本",
                   rf"マルチプライヤー $m(\omega)={mt}$ の作用素 $T_m$ の $L^2$ での作用素ノルム $\displaystyle\sup_{{f\ne0}}\frac{{\|T_mf\|_2}}{{\|f\|_2}}=\sup_\omega|m(\omega)|$ を求めよ。",
                   distinct(ok, cands), hints=[r"プランシュレルの等式から, $L^2$ での作用素ノルムは $\sup|m|$。", r"$|m(\omega)|$ の最大値を微分などで求める。"],
                   solution=[rf"$|m(\omega)|$ は ${at}$ で最大になり, $\sup|m|={ok}$。"], meta=dict(kind=kind, c=c))


# ------------------------------------------------------------
# 標準:ハーディ・リトルウッド・ソボレフの指数
# ------------------------------------------------------------
HLS = [(F(1, 2), F(4, 3)), (F(1, 2), F(3, 2)), (F(1, 3), F(3, 2)), (F(1, 3), F(6, 5)), (F(1, 4), F(2)), (F(1, 4), F(4, 3)),
       (F(2, 3), F(6, 5)), (F(1, 2), F(6, 5)), (F(1, 3), F(2))]


def hls_parts(rng):
    a, p = rng.choice(HLS)
    q = 1 / (1 / p - a)
    return a, p, q


def g_hls(rng):
    a, p, q = hls_parts(rng)
    cands = []
    if 1 / p + a < 1:
        cands.append((tex_fr(1 / (1 / p + a)), r"$I_\alpha$ は関数をなめらかにして, より大きい $q$ の空間に入れる:$\frac1q=\frac1p-\alpha$。"))
    cands += [(tex_fr(1 / a), r"$q$ は $p$ と $\alpha$ の両方で決まる。"), (tex_fr(p / (1 - a)), r"$\frac1q=\frac1p-\alpha$ を $q$ について解き直そう。"),
              (tex_fr(p + a), r"指数の関係は逆数で:$\frac1q=\frac1p-\alpha$。"), (tex_fr(2 * q), "計算を確かめよう。")]
    return Problem("ハーディ・リトルウッド・ソボレフの指数", "標準",
                   rf"1次元のリースポテンシャル $I_\alpha$($\alpha={tex_fr(a)}$)について, ハーディ・リトルウッド・ソボレフの不等式 "
                   rf"$\|I_\alpha f\|_{{L^q}}\le C\|f\|_{{L^p}}$ が $p={tex_fr(p)}$ で成り立つときの $q$ を求めよ。",
                   distinct(tex_fr(q), cands), hints=[r"$\frac1q=\frac1p-\alpha$(1次元)。", "幅 $w$ の矩形パルスでスケーリングを考えると導ける。"],
                   solution=[rf"$\frac1q=\frac1p-\alpha={tex_fr(1 / p)}-{tex_fr(a)}={tex_fr(1 / q)}$ より $q={tex_fr(q)}$。"], meta=dict(a=a, p=p, q=q))


# ------------------------------------------------------------
# 誘導
# ------------------------------------------------------------
def guided_rc(rng):
    c = rng.choice([1, 2, 3])
    ct = "" if c == 1 else str(c)
    return ExamItem("RC 回路のマルチプライヤー",
                    rf"マルチプライヤー $m(\omega)=\dfrac{{{c}}}{{{c}+i\omega}}$ の作用素 $T_m$ について, 次の問いに答えよ. "
                    r"ただし $\hat f(\omega)=\int f(t)e^{-i\omega t}dt$, $u(t)$ は単位ステップ関数とする.",
                    [Part(r"核 $K(t)$(フーリエ変換が $m$ になる関数)を求めよ.", rf"K(t)={ct}e^{{-{ct}t}}u(t)",
                          [rf"$\int_0^\infty {ct}e^{{-{ct}t}}e^{{-i\omega t}}dt={ct}\left[\frac{{e^{{-({c}+i\omega)t}}}}{{-({c}+i\omega)}}\right]_0^\infty=\frac{{{c}}}{{{c}+i\omega}}$ より $K(t)={ct}e^{{-{ct}t}}u(t)$。"]),
                     Part(rf"$f(t)=\cos {ct}t$ に対する出力 $T_mf(t)$ を求めよ.", rf"\frac{{1}}{{\sqrt2}}\cos\left({ct}t-\frac{{\pi}}{{4}}\right)",
                          [rf"$m({c})=\frac{{1}}{{1+i}}=\frac{{1}}{{\sqrt2}}e^{{-i\pi/4}}$, $m(-{c})=\overline{{m({c})}}$。",
                           rf"$T_m\cos {ct}t=\mathrm{{Re}}\left(m({c})e^{{i{ct}t}}\right)=\frac{{1}}{{\sqrt2}}\cos\left({ct}t-\frac{{\pi}}{{4}}\right)$"
                           r"(振幅が $\frac{1}{\sqrt2}$ 倍, 位相が $\frac\pi4$ 遅れる)。"])],
                    meta=dict(c=c))


def guided_hls(rng):
    a, p, q = hls_parts(rng)
    return ExamItem("リースポテンシャルとスケーリング",
                    rf"$0<\alpha<1$ とし, $I_\alpha f(t)=c_\alpha\displaystyle\int\frac{{f(s)}}{{|t-s|^{{1-\alpha}}}}ds$ とする. "
                    rf"$\alpha={tex_fr(a)}$, $p={tex_fr(p)}$ について, 次の問いに答えよ.",
                    [Part(r"$w>0$ に対し $f_w(t)=f(t/w)$ とおく. $I_\alpha f_w(t)=w^{\alpha}(I_\alpha f)(t/w)$ を示し, "
                          r"$\|f_w\|_p=w^{1/p}\|f\|_p$, $\|I_\alpha f_w\|_q=w^{\alpha+1/q}\|I_\alpha f\|_q$ を導け.", "",
                          [r"$s=w\sigma$ とおくと $I_\alpha f_w(t)=c_\alpha\int\frac{f(\sigma)}{|t-w\sigma|^{1-\alpha}}w\,d\sigma"
                           r"=w^{\alpha}c_\alpha\int\frac{f(\sigma)}{|t/w-\sigma|^{1-\alpha}}d\sigma=w^{\alpha}(I_\alpha f)(t/w)$。",
                           r"ノルムは $t=w\tau$ とおいて $\left(\int|g(t/w)|^rdt\right)^{1/r}=w^{1/r}\|g\|_r$ から。"]),
                     Part(r"不等式 $\|I_\alpha f\|_q\le C\|f\|_p$ がすべての $f$ で成り立つための $q$ の条件を求め, $q$ の値を答えよ.", tex_fr(q),
                          [r"(1) から $\frac{\|I_\alpha f_w\|_q}{\|f_w\|_p}=w^{\alpha+\frac1q-\frac1p}\frac{\|I_\alpha f\|_q}{\|f\|_p}$。"
                           r"これが $w\to0$ と $w\to\infty$ の両方で有界であるためには, 指数が $0$ でなければならない。",
                           rf"$\frac1q=\frac1p-\alpha={tex_fr(1 / p)}-{tex_fr(a)}={tex_fr(1 / q)}$ より $q={tex_fr(q)}$。"])],
                    meta=dict(a=a, p=p, q=q))


REGISTRY = [
    ProblemType("mp_cos", "cos に作用させる", "基本", g_cos, guided=guided_rc),
    ProblemType("mp_op", "マルチプライヤーと作用素", "基本", g_op),
    ProblemType("mp_norm", "L² での作用素のノルム", "基本", g_norm),
    ProblemType("mp_kernel", "マルチプライヤーの核", "標準", g_kernel),
    ProblemType("mp_hls", "ハーディ・リトルウッド・ソボレフの指数", "標準", g_hls, guided=guided_hls),
]
GUIDED_NAMES = {"mp_cos": "RC 回路の核 → cos への応答(振幅と位相)",
                "mp_hls": "リースポテンシャルのスケーリング → HLS の指数 q"}


def tab():
    render("multiplier", REGISTRY, "フーリエ解析 発展 フーリエマルチプライヤー")
