"""演習・試験の図:同じ仕様(dict)から、Web 用の plotly 図と LaTeX 用の TikZ を作る

仕様の例
  {"kind": "polygon", "pts": [(0,0),(3,0),(3,4),(0,4)], "labels": ["O","A_0","A_1","A_2"]}
  {"kind": "sector", "R": 1, "deg": 60, "labels": ["O","A","B"]}
  {"kind": "curve3d", "pts": [[x...],[y...],[z...]]}      (Web のみ。LaTeX には出さない)
"""
import math

import numpy as np
import plotly.graph_objects as go

from common import layout

ANG_TEX = {30: r"\frac{\pi}{6}", 45: r"\frac{\pi}{4}", 60: r"\frac{\pi}{3}", 90: r"\frac{\pi}{2}",
           120: r"\frac{2\pi}{3}", 180: r"\pi"}


def _boundary(spec, n=60):
    """反時計回りの境界の点列"""
    if spec["kind"] == "polygon":
        P = list(spec["pts"]) + [spec["pts"][0]]
        return np.array(P, float)
    R, th = spec["R"], math.radians(spec["deg"])
    arc = [(R * math.cos(t), R * math.sin(t)) for t in np.linspace(0, th, n)]
    return np.array([(0, 0)] + arc + [(0, 0)], float)


def _corners(spec):
    if spec["kind"] == "polygon":
        return [tuple(p) for p in spec["pts"]]
    R, th = spec["R"], math.radians(spec["deg"])
    return [(0, 0), (R, 0), (R * math.cos(th), R * math.sin(th))]


def _edges(spec):
    """向きの矢印をつける辺(始点, 終点)"""
    c = _corners(spec)
    if spec["kind"] == "polygon":
        return [(c[i], c[(i + 1) % len(c)]) for i in range(len(c))]
    R, th = spec["R"], math.radians(spec["deg"])
    mid = th / 2
    return [(c[0], c[1]), ((R * math.cos(mid - 0.05), R * math.sin(mid - 0.05)), (R * math.cos(mid + 0.05), R * math.sin(mid + 0.05))),
            (c[2], c[0])]


def _extent(spec):
    pts = np.array(_corners(spec), float)
    hi = max(1.0, float(np.ceil(pts.max()))) + 1
    lo = min(0.0, float(np.floor(pts.min()))) - 1
    return lo, hi


def plotly_fig(spec, height=420):
    if spec is None:
        return None
    if spec["kind"] == "curve2d":
        x, y = spec["pts"]
        m = max(max(map(abs, x)), max(map(abs, y))) * 1.2 + 0.2
        fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", fill="toself", fillcolor="rgba(31,119,180,0.15)",
                                   line=dict(color="black", width=3), name="C"))
        k = len(x) // 8
        fig.add_annotation(x=x[k + 1], y=y[k + 1], ax=x[k - 1], ay=y[k - 1], xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=3, arrowsize=1.6, arrowwidth=2, arrowcolor="#d62728", text="")
        fig.update_xaxes(zeroline=True, zerolinecolor="black", zerolinewidth=2)
        fig.update_yaxes(zeroline=True, zerolinecolor="black", zerolinewidth=2)
        layout(fig, height=height, equal=True, xlab="x", ylab="y", xrange=(-m, m), yrange=(-m, m), legend=False)
        return fig
    if spec["kind"] == "curve3d":
        x, y, z = spec["pts"]
        fig = go.Figure(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color="#d62728", width=6), name="C"))
        fig.add_trace(go.Scatter3d(x=[x[0], x[-1]], y=[y[0], y[-1]], z=[z[0], z[-1]], mode="markers+text",
                                   text=["始点", "終点"], marker=dict(size=4, color="black"), name="端点"))
        fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), scene=dict(aspectmode="data"),
                          showlegend=False)
        return fig
    B = _boundary(spec)
    lo, hi = _extent(spec)
    fig = go.Figure(go.Scatter(x=B[:, 0], y=B[:, 1], mode="lines", fill="toself", fillcolor="rgba(31,119,180,0.15)",
                               line=dict(color="black", width=3), name="C", hoverinfo="skip"))
    for (p, q) in _edges(spec):
        m = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
        fig.add_annotation(x=m[0] + (q[0] - p[0]) * 0.08, y=m[1] + (q[1] - p[1]) * 0.08,
                           ax=m[0] - (q[0] - p[0]) * 0.08, ay=m[1] - (q[1] - p[1]) * 0.08,
                           xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.6,
                           arrowwidth=2, arrowcolor="#d62728", text="")
    for (x, y), lab in zip(_corners(spec), spec["labels"]):
        fig.add_annotation(x=x, y=y, text=_html(lab), showarrow=False, xshift=-14 if x <= 0 else 14,
                           yshift=-14 if y <= 0 else 14, font=dict(size=18))
    C = np.array(_corners(spec), float)
    fig.add_trace(go.Scatter(x=C[:, 0], y=C[:, 1], mode="markers", marker=dict(color="black", size=7), hoverinfo="skip"))
    if spec["kind"] == "sector":
        fig.add_annotation(x=0.35 * spec["R"] * math.cos(math.radians(spec["deg"] / 2)) + 0.05 * spec["R"],
                           y=0.35 * spec["R"] * math.sin(math.radians(spec["deg"] / 2)),
                           text=f"{spec['deg']}°", showarrow=False, font=dict(size=14))
    step = 1 if hi - lo <= 9 else 2
    fig.update_xaxes(dtick=step, zeroline=True, zerolinecolor="black", zerolinewidth=2, showgrid=True)
    fig.update_yaxes(dtick=step, zeroline=True, zerolinecolor="black", zerolinewidth=2, showgrid=True)
    if spec["kind"] == "sector" and spec["R"] <= 2:
        fig.update_xaxes(dtick=0.5)
        fig.update_yaxes(dtick=0.5)
        lo, hi = -0.25 * spec["R"], 1.4 * spec["R"]
    layout(fig, height=height, equal=True, xlab="x", ylab="y", xrange=(lo, hi), yrange=(lo, hi), legend=False)
    return fig


def _html(lab):
    """A_0 → A<sub>0</sub>"""
    if "_" in lab:
        a, b = lab.split("_", 1)
        return f"{a}<sub>{b.strip('{}')}</sub>"
    return lab


def _tikz_label(lab):
    return r"$\mathrm{" + lab.split("_")[0] + "}" + ("_{" + lab.split("_", 1)[1].strip("{}") + "}" if "_" in lab else "") + "$"


def tikz(spec):
    """方眼つきの図(数学IB の道路網の図と同じく tikzpicture で描く)"""
    if spec is None or spec["kind"] == "curve3d":
        return ""
    if spec["kind"] == "curve2d":
        x, y = spec["pts"]
        m = math.ceil(max(max(map(abs, x)), max(map(abs, y)))) + 1
        sc = 4.0 / (2 * m)
        coords = " ".join(f"({a:.3f},{b:.3f})" for a, b in zip(x[::4], y[::4])) + f" ({x[0]:.3f},{y[0]:.3f})"
        k = len(x) // 8
        return "\n".join([r"\begin{center}", rf"\begin{{tikzpicture}}[scale={sc:.3g}]",
                          rf"  \draw[step=1, gray!35, very thin] ({-m},{-m}) grid ({m},{m});",
                          rf"  \draw[->, thick] ({-m},0) -- ({m + 0.4},0) node[right] {{$x$}};",
                          rf"  \draw[->, thick] (0,{-m}) -- (0,{m + 0.4}) node[above] {{$y$}};",
                          rf"  \draw[very thick, fill=blue!8] plot coordinates {{{coords}}};",
                          rf"  \draw[-latex, very thick, red] ({x[k - 1]:.3f},{y[k - 1]:.3f}) -- ({x[k + 1]:.3f},{y[k + 1]:.3f});",
                          r"\end{tikzpicture}", r"\end{center}"])
    lo, hi = _extent(spec)
    if spec["kind"] == "sector":
        R = spec["R"]
        lo, hi = -0.25 * R, 1.35 * R
        sc = 4.0 / (hi - lo)
        grid = R / 4
    else:
        sc = min(0.6, 5.0 / (hi - lo))
        grid = 1
    out = [r"\begin{center}", rf"\begin{{tikzpicture}}[scale={sc:.3g}, baseline=(current bounding box.center)]",
           rf"  \draw[step={grid:g}, gray!35, very thin] ({lo:g},{lo:g}) grid ({hi:g},{hi:g});",
           rf"  \draw[->, thick] ({lo:g},0) -- ({hi + 0.3 * (hi - lo) / 8:g},0) node[right] {{$x$}};",
           rf"  \draw[->, thick] (0,{lo:g}) -- (0,{hi + 0.3 * (hi - lo) / 8:g}) node[above] {{$y$}};"]
    if spec["kind"] == "polygon":
        path = " -- ".join(f"({x:g},{y:g})" for x, y in spec["pts"]) + " -- cycle"
    else:
        R, d = spec["R"], spec["deg"]
        path = rf"(0,0) -- ({R:g},0) arc (0:{d}:{R:g}) -- cycle"
    out.append(rf"  \fill[blue!8] {path};")
    out.append(rf"  \draw[very thick] {path};")
    for (p, q) in _edges(spec):
        m = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
        d = (q[0] - p[0], q[1] - p[1])
        L = math.hypot(*d) or 1
        e = 0.18 * (hi - lo) / 8 if spec["kind"] == "polygon" else 0.06 * spec["R"]
        a = (m[0] - d[0] / L * e, m[1] - d[1] / L * e)
        b = (m[0] + d[0] / L * e, m[1] + d[1] / L * e)
        out.append(rf"  \draw[-latex, very thick, red] ({a[0]:.3f},{a[1]:.3f}) -- ({b[0]:.3f},{b[1]:.3f});")
    for (x, y), lab in zip(_corners(spec), spec["labels"]):
        pos = ("below" if y <= 0 else "above") + (" left" if x <= 0 else " right")
        out.append(rf"  \fill ({x:.3f},{y:.3f}) circle ({2 / sc:.2f}pt) node[{pos}] {{{_tikz_label(lab)}}};")
    if spec["kind"] == "sector":
        r0 = 0.3 * spec["R"]
        out.append(rf"  \draw ({r0:g},0) arc (0:{spec['deg']}:{r0:g});")
        mid = math.radians(spec["deg"] / 2)
        ang = ANG_TEX.get(spec["deg"], str(spec["deg"]) + r"^\circ")
        out.append(rf"  \node at ({(r0 + 0.12 * spec['R']) * math.cos(mid):.3f},{(r0 + 0.12 * spec['R']) * math.sin(mid):.3f}) "
                   rf"{{$\scriptstyle {ang}$}};")
    out += [r"\end{tikzpicture}", r"\end{center}"]
    return "\n".join(out)


# ============================================================
# 空間図形(斜投影の TikZ / plotly 3D)と楕円の光線
# ============================================================
# TikZ の3次元座標:x 軸は左下、y 軸は右、z 軸は上
TIKZ3D = r"x={(-0.55cm,-0.45cm)}, y={(1cm,0cm)}, z={(0cm,1cm)}"


def _axes3d(L, sc):
    return [rf"\begin{{tikzpicture}}[{TIKZ3D}, scale={sc:.3g}]",
            rf"  \draw[->, thick] (0,0,0) -- ({L[0]:g},0,0) node[below left] {{$x$}};",
            rf"  \draw[->, thick] (0,0,0) -- (0,{L[1]:g},0) node[right] {{$y$}};",
            rf"  \draw[->, thick] (0,0,0) -- (0,0,{L[2]:g}) node[above] {{$z$}};",
            r"  \node[below right] at (0,0,0) {$\mathrm{O}$};"]


def _p3(p):
    return "(" + ",".join(f"{float(v):.3f}" for v in p) + ")"


def tikz3d(spec):
    k = spec["kind"]
    out = []
    if k == "segment3d":
        A, B = spec["A"], spec["B"]
        m = max(max(abs(v) for v in A + B), 1)
        L = [m + 1.5] * 3
        sc = 3.2 / (m + 1.5)
        out += _axes3d(L, sc)
        for P, lab in ((A, spec["labels"][0]), (B, spec["labels"][1])):
            out.append(rf"  \draw[dashed, gray] {_p3(P)} -- {_p3((P[0], P[1], 0))} -- {_p3((P[0], 0, 0))};")
            out.append(rf"  \draw[dashed, gray] {_p3((P[0], P[1], 0))} -- {_p3((0, P[1], 0))};")
        out.append(rf"  \draw[-latex, very thick, red] {_p3(A)} -- {_p3(B)};")
        for P, lab, pos in ((A, spec["labels"][0], "left"), (B, spec["labels"][1], "above right")):
            out.append(rf"  \fill {_p3(P)} circle ({2 / sc:.2f}pt) node[{pos}] {{{lab}}};")
    elif k == "plane_tri3d":
        a, b, c = spec["abc"]
        sc = 2.6 / max(a, b, c)
        out += _axes3d([a * 1.35, b * 1.35, c * 1.35], sc)
        out.append(rf"  \fill[blue!12] ({a},0,0) -- (0,{b},0) -- (0,0,{c}) -- cycle;")
        out.append(rf"  \draw[very thick] ({a},0,0) -- (0,{b},0) -- (0,0,{c}) -- cycle;")
        out.append(rf"  \draw[dashed] (0,0,0) -- ({a},0,0) -- (0,{b},0) -- cycle;")
        out.append(rf"  \node[left] at ({a},0,0) {{${a}$}};  \node[below] at (0,{b},0) {{${b}$}};  \node[left] at (0,0,{c}) {{${c}$}};")
        if spec.get("normal"):
            g = (a + b + c) / 3
            ctr = (a / 3, b / 3, c / 3)
            n = (c / a, c / b, 1)
            s = 0.7 * max(a, b, c) / math.sqrt(sum(x * x for x in n))
            out.append(rf"  \draw[-latex, very thick, red] {_p3(ctr)} -- {_p3(tuple(ctr[i] + s * n[i] for i in range(3)))} "
                       r"node[above right] {$\vec n$};")
            del g
    elif k == "box3d":
        a, b, c = spec["abc"]
        sc = 2.4 / max(a, b, c)
        out += _axes3d([a * 1.4, b * 1.3, c * 1.4], sc)
        V = lambda x, y, z: f"({x},{y},{z})"  # noqa: E731
        out.append(rf"  \fill[blue!8] {V(a,0,0)} -- {V(a,b,0)} -- {V(a,b,c)} -- {V(a,0,c)} -- cycle;")
        out.append(rf"  \fill[blue!14] {V(0,0,c)} -- {V(a,0,c)} -- {V(a,b,c)} -- {V(0,b,c)} -- cycle;")
        out.append(rf"  \draw[very thick] {V(a,0,0)} -- {V(a,b,0)} -- {V(a,b,c)} -- {V(a,0,c)} -- cycle;")
        out.append(rf"  \draw[very thick] {V(a,0,c)} -- {V(0,0,c)} -- {V(0,b,c)} -- {V(a,b,c)};")
        out.append(rf"  \draw[very thick] {V(a,b,0)} -- {V(0,b,0)} -- {V(0,b,c)};")
        out.append(rf"  \draw[dashed] {V(0,0,0)} -- {V(a,0,0)};  \draw[dashed] {V(0,0,0)} -- {V(0,b,0)};  \draw[dashed] {V(0,0,0)} -- {V(0,0,c)};")
        out.append(rf"  \node[left] at ({a},0,0) {{${a}$}};  \node[below] at (0,{b},0) {{${b}$}};  \node[left] at (0,0,{c}) {{${c}$}};")
    elif k in ("sphere3d", "hemisphere3d", "disk3d"):
        # 球・半球・円板は、教科書によくある見取り図(輪郭の円 + 赤道の楕円)で描く
        R = spec["R"]
        sc = 1.5 / R
        e = 0.32 * R
        out += [rf"\begin{{tikzpicture}}[scale={sc:.3g}]",
                rf"  \draw[->, thick] (0,0) -- ({1.6 * R:.2f},0) node[right] {{$y$}};",
                rf"  \draw[->, thick] (0,0) -- (0,{1.6 * R:.2f}) node[above] {{$z$}};",
                rf"  \draw[->, thick] (0,0) -- ({-0.75 * R:.2f},{-0.75 * R:.2f}) node[below left] {{$x$}};"]
        if k == "disk3d":
            out.append(rf"  \fill[blue!12] (0,0) ellipse ({R} and {e:.3f});")
            out.append(rf"  \draw[very thick] (0,0) ellipse ({R} and {e:.3f});")
        else:
            if k == "sphere3d":
                out.append(rf"  \fill[blue!10, opacity=0.8] (0,0) circle ({R});")
                out.append(rf"  \draw[very thick] (0,0) circle ({R});")
            else:
                out.append(rf"  \fill[blue!10, opacity=0.8] ({-R},0) arc (180:0:{R}) arc (0:-180:{R} and {e:.3f});")
                out.append(rf"  \draw[very thick] ({-R},0) arc (180:0:{R});")
            out.append(rf"  \draw[thick, dashed] ({R},0) arc (0:180:{R} and {e:.3f});")
            out.append(rf"  \draw[thick] ({-R},0) arc (180:360:{R} and {e:.3f});")
        if spec.get("orient"):
            out.append(rf"  \draw[-latex, very thick, red] ({-0.25 * R:.3f},{-e * 0.968:.3f}) -- ({0.25 * R:.3f},{-e * 0.968:.3f});")
            out.append(rf"  \node[below] at (0,{-e:.3f}) {{$C$}};")
        out.append(rf"  \node[below right] at ({R},0) {{${R}$}};")
        out.append(r"\end{tikzpicture}")
        return out
    out.append(r"\end{tikzpicture}")
    return out


def tikz_ellipse(spec):
    a, b = spec["ab"]
    c = math.sqrt(a * a - b * b)
    sc = 5.5 / (2 * a)
    out = [rf"\begin{{tikzpicture}}[scale={sc:.3g}]",
           rf"  \draw[->, thick] ({-a * 1.2:.2f},0) -- ({a * 1.25:.2f},0) node[right] {{$x$}};",
           rf"  \draw[->, thick] (0,{-b * 1.25:.2f}) -- (0,{b * 1.3:.2f}) node[above] {{$y$}};",
           rf"  \draw[very thick] (0,0) ellipse ({a} and {b});",
           rf"  \fill ({-c:g},0) circle ({2 / sc:.2f}pt) node[below] {{$\mathrm{{F}}_1$}};",
           rf"  \fill ({c:g},0) circle ({2 / sc:.2f}pt) node[below] {{$\mathrm{{F}}_2$}};"]
    for p, q in spec.get("rays", []):
        out.append(rf"  \draw[-latex, very thick, red] ({p[0]:.3f},{p[1]:.3f}) -- ({q[0]:.3f},{q[1]:.3f});")
    Q = spec["Q"]
    out.append(rf"  \fill ({Q[0]:.3f},{Q[1]:.3f}) circle ({2 / sc:.2f}pt) node[{'above' if Q[1] >= 0 else 'below'} "
               rf"{'right' if Q[0] >= 0 else 'left'}] {{$\mathrm{{Q}}$}};")
    if spec.get("normal"):
        n = (2 * Q[0] / a ** 2, 2 * Q[1] / b ** 2)
        L = math.hypot(*n)
        s = 0.35 * a / L
        out.append(rf"  \draw[-latex, thick, blue] ({Q[0]:.3f},{Q[1]:.3f}) -- ({Q[0] - s * n[0]:.3f},{Q[1] - s * n[1]:.3f}) "
                   r"node[left] {$-\nabla g$};")
    out.append(r"\end{tikzpicture}")
    return out


def plotly3d(spec, height=420):
    k = spec["kind"]
    fig = go.Figure()
    if k == "segment3d":
        A, B = spec["A"], spec["B"]
        fig.add_trace(go.Scatter3d(x=[A[0], B[0]], y=[A[1], B[1]], z=[A[2], B[2]], mode="lines+markers+text",
                                   text=spec.get("web_labels", ["始点", "終点"]),
                                   line=dict(color="#d62728", width=7), marker=dict(size=5, color="black")))
        fig.add_trace(go.Cone(x=[B[0]], y=[B[1]], z=[B[2]], u=[B[0] - A[0]], v=[B[1] - A[1]], w=[B[2] - A[2]],
                              sizemode="absolute", sizeref=0.6, anchor="tip", showscale=False,
                              colorscale=[[0, "#d62728"], [1, "#d62728"]]))
    elif k == "plane_tri3d":
        a, b, c = spec["abc"]
        fig.add_trace(go.Mesh3d(x=[a, 0, 0], y=[0, b, 0], z=[0, 0, c], i=[0], j=[1], k=[2], color="#1f77b4", opacity=0.45))
        fig.add_trace(go.Scatter3d(x=[a, 0, 0, a], y=[0, b, 0, 0], z=[0, 0, c, 0], mode="lines", line=dict(color="black", width=5)))
    elif k == "box3d":
        a, b, c = spec["abc"]
        E = [((0, 0, 0), (a, 0, 0)), ((0, 0, 0), (0, b, 0)), ((0, 0, 0), (0, 0, c)), ((a, 0, 0), (a, b, 0)), ((a, 0, 0), (a, 0, c)),
             ((0, b, 0), (a, b, 0)), ((0, b, 0), (0, b, c)), ((0, 0, c), (a, 0, c)), ((0, 0, c), (0, b, c)),
             ((a, b, 0), (a, b, c)), ((a, 0, c), (a, b, c)), ((0, b, c), (a, b, c))]
        for p, q in E:
            fig.add_trace(go.Scatter3d(x=[p[0], q[0]], y=[p[1], q[1]], z=[p[2], q[2]], mode="lines", line=dict(color="black", width=5)))
    else:
        R = spec["R"]
        if k == "disk3d":
            r, t = np.meshgrid(np.linspace(0, R, 10), np.linspace(0, 2 * np.pi, 60))
            fig.add_trace(go.Surface(x=r * np.cos(t), y=r * np.sin(t), z=0 * r, opacity=0.5, showscale=False, colorscale="Blues"))
        else:
            u, v = np.meshgrid(np.linspace(0, np.pi / 2 if k == "hemisphere3d" else np.pi, 30), np.linspace(0, 2 * np.pi, 60))
            fig.add_trace(go.Surface(x=R * np.sin(u) * np.cos(v), y=R * np.sin(u) * np.sin(v), z=R * np.cos(u), opacity=0.45,
                                     showscale=False, colorscale="Blues"))
        t = np.linspace(0, 2 * np.pi, 120)
        fig.add_trace(go.Scatter3d(x=R * np.cos(t), y=R * np.sin(t), z=0 * t, mode="lines", line=dict(color="#d62728", width=6)))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), scene=dict(aspectmode="data"), showlegend=False)
    return fig


def plotly_ellipse(spec, height=380):
    a, b = spec["ab"]
    c = math.sqrt(a * a - b * b)
    t = np.linspace(0, 2 * np.pi, 300)
    fig = go.Figure(go.Scatter(x=a * np.cos(t), y=b * np.sin(t), mode="lines", line=dict(color="black", width=3)))
    for p, q in spec.get("rays", []):
        fig.add_annotation(x=q[0], y=q[1], ax=p[0], ay=p[1], xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                           arrowhead=3, arrowsize=1.5, arrowwidth=2.5, arrowcolor="#d62728", text="")
    Q = spec["Q"]
    fig.add_trace(go.Scatter(x=[-c, c, Q[0]], y=[0, 0, Q[1]], mode="markers+text", text=["F₁", "F₂", "Q"],
                             textposition="bottom center", marker=dict(size=9, color="black")))
    if spec.get("normal"):
        n = np.array([2 * Q[0] / a ** 2, 2 * Q[1] / b ** 2])
        n = n / np.linalg.norm(n) * 0.35 * a
        fig.add_annotation(x=Q[0] - n[0], y=Q[1] - n[1], ax=Q[0], ay=Q[1], xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=3, arrowwidth=2, arrowcolor="#1f77b4", text="")
    layout(fig, height=height, equal=True, xlab="x", ylab="y", legend=False, xrange=(-1.2 * a, 1.2 * a), yrange=(-1.3 * b, 1.3 * b))
    return fig


_plotly2d = plotly_fig
_tikz2d = tikz


# ============================================================
# 平面曲線をつないだ道(円・円弧・線分。各部分に向きの矢印と名前)
#   {"kind": "path2d", "pieces": [{"pts": [[x..], [y..]], "label": "C_1"}, ...],
#    "points": [(x, y, "A"), ...], "fill": True}
# ============================================================
PIECE_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
PIECE_TIKZ = ["red", "blue", "green!60!black", "orange"]


def _path_extent(spec):
    xs = [v for pc in spec["pieces"] for v in pc["pts"][0]] + [p[0] for p in spec.get("points", [])]
    ys = [v for pc in spec["pieces"] for v in pc["pts"][1]] + [p[1] for p in spec.get("points", [])]
    lo = math.floor(min(min(xs), min(ys), 0)) - 1
    hi = math.ceil(max(max(xs), max(ys), 0)) + 1
    return lo, hi


def _mid_arrow(pc):
    x, y = pc["pts"]
    k = len(x) // 2
    k0, k1 = max(k - 2, 0), min(k + 2, len(x) - 1)
    return (x[k0], y[k0]), (x[k1], y[k1]), (x[k], y[k])


def plotly_path2d(spec, height=380):
    fig = go.Figure()
    if spec.get("fill"):
        X = [v for pc in spec["pieces"] for v in pc["pts"][0]]
        Y = [v for pc in spec["pieces"] for v in pc["pts"][1]]
        fig.add_trace(go.Scatter(x=X, y=Y, mode="lines", fill="toself", fillcolor="rgba(31,119,180,0.12)",
                                 line=dict(width=0), hoverinfo="skip", showlegend=False))
    lo, hi = _path_extent(spec)
    for i, pc in enumerate(spec["pieces"]):
        col = PIECE_COLORS[i % 4]
        x, y = pc["pts"]
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=col, width=3.5), name=pc.get("label", "C"),
                                 hoverinfo="skip"))
        a, b, m = _mid_arrow(pc)
        fig.add_annotation(x=b[0], y=b[1], ax=a[0], ay=a[1], xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                           arrowhead=3, arrowsize=1.8, arrowwidth=2.5, arrowcolor=col, text="")
        if pc.get("label"):
            dx, dy = b[0] - a[0], b[1] - a[1]
            L = math.hypot(dx, dy) or 1
            off = 0.06 * (hi - lo)
            fig.add_annotation(x=m[0] - dy / L * off, y=m[1] + dx / L * off, text=_html(pc["label"]), showarrow=False,
                               font=dict(size=17, color=col))
    for (x, y, lab) in spec.get("points", []):
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers", marker=dict(color="black", size=8), hoverinfo="skip",
                                 showlegend=False))
        fig.add_annotation(x=x, y=y, text=_html(lab), showarrow=False, xshift=-14 if x <= 0 else 14,
                           yshift=-14 if y <= 0 else 14, font=dict(size=17))
    fig.update_xaxes(dtick=1, zeroline=True, zerolinecolor="black", zerolinewidth=2, showgrid=True)
    fig.update_yaxes(dtick=1, zeroline=True, zerolinecolor="black", zerolinewidth=2, showgrid=True)
    layout(fig, height=height, equal=True, xlab="x", ylab="y", xrange=(lo, hi), yrange=(lo, hi), legend=False)
    return fig


def tikz_path2d(spec):
    lo, hi = _path_extent(spec)
    sc = min(0.7, 5.0 / (hi - lo))
    out = [rf"\begin{{tikzpicture}}[scale={sc:.3g}]",
           rf"  \draw[step=1, gray!35, very thin] ({lo},{lo}) grid ({hi},{hi});",
           rf"  \draw[->, thick] ({lo},0) -- ({hi + 0.4},0) node[right] {{$x$}};",
           rf"  \draw[->, thick] (0,{lo}) -- (0,{hi + 0.4}) node[above] {{$y$}};"]
    if spec.get("fill"):
        X = [v for pc in spec["pieces"] for v in pc["pts"][0]]
        Y = [v for pc in spec["pieces"] for v in pc["pts"][1]]
        out.append(r"  \fill[blue!8] plot coordinates {" + " ".join(f"({a:.3f},{b:.3f})" for a, b in zip(X, Y)) + "} -- cycle;")
    for i, pc in enumerate(spec["pieces"]):
        col = PIECE_TIKZ[i % 4]
        x, y = pc["pts"]
        step = max(1, len(x) // 40)
        pts = list(zip(x[::step], y[::step]))
        if (pts[-1][0], pts[-1][1]) != (x[-1], y[-1]):
            pts.append((x[-1], y[-1]))
        out.append(rf"  \draw[very thick, {col}] plot coordinates {{" + " ".join(f"({a:.3f},{b:.3f})" for a, b in pts) + "};")
        a, b, m = _mid_arrow(pc)
        out.append(rf"  \draw[-latex, very thick, {col}] ({a[0]:.3f},{a[1]:.3f}) -- ({b[0]:.3f},{b[1]:.3f});")
        if pc.get("label"):
            dx, dy = b[0] - a[0], b[1] - a[1]
            L = math.hypot(dx, dy) or 1
            off = 0.07 * (hi - lo)
            out.append(rf"  \node[{col}] at ({m[0] - dy / L * off:.3f},{m[1] + dx / L * off:.3f}) {{{_tikz_label(pc['label'])}}};")
    for (x, y, lab) in spec.get("points", []):
        pos = ("below" if y <= 0 else "above") + (" left" if x <= 0 else " right")
        out.append(rf"  \fill ({x:.3f},{y:.3f}) circle ({2 / sc:.2f}pt) node[{pos}] {{{_tikz_label(lab)}}};")
    out.append(r"\end{tikzpicture}")
    return out


def arc_pts(cx, cy, R, t0, t1, n=81):
    t = np.linspace(t0, t1, n)
    return [list(map(float, cx + R * np.cos(t))), list(map(float, cy + R * np.sin(t)))]


def seg_pts(A, B, n=21):
    t = np.linspace(0, 1, n)
    return [list(map(float, A[0] + (B[0] - A[0]) * t)), list(map(float, A[1] + (B[1] - A[1]) * t))]


def plotly_fig(spec, height=420):  # noqa: F811
    if spec is None:
        return None
    if spec["kind"] in ("segment3d", "plane_tri3d", "box3d", "sphere3d", "hemisphere3d", "disk3d"):
        return plotly3d(spec, height)
    if spec["kind"] == "ellipse":
        return plotly_ellipse(spec, height)
    if spec["kind"] == "path2d":
        return plotly_path2d(spec, height)
    return _plotly2d(spec, height)


def tikz(spec):  # noqa: F811
    """図の TikZ。spec["complex"] が真なら、全体をコメントアウトして出す(必要なら % を外して使う)"""
    if spec is None:
        return ""
    if spec["kind"] in ("segment3d", "plane_tri3d", "box3d", "sphere3d", "hemisphere3d", "disk3d"):
        lines = [r"\begin{center}"] + tikz3d(spec) + [r"\end{center}"]
    elif spec["kind"] == "ellipse":
        lines = [r"\begin{center}"] + tikz_ellipse(spec) + [r"\end{center}"]
    elif spec["kind"] == "path2d":
        lines = [r"\begin{center}"] + tikz_path2d(spec) + [r"\end{center}"]
    else:
        code = _tikz2d(spec)
        if not code:
            return ""
        lines = code.split("\n")
    if spec.get("complex"):
        return "% 図が複雑なためコメントアウトしています(必要なら各行の % を外す)\n" + "\n".join("% " + s for s in lines)
    return "\n".join(lines)
