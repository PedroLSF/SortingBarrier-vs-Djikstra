#!/usr/bin/env python3
"""Generate the report figures from the benchmark CSVs.

Usage:
    .venv/bin/python src/plots.py   # reads results/csv/*.csv, writes results/figures/<stage>/*.png and *.pdf
    .venv/bin/python src/plots.py --flat --output <folder>   # every figure in a single folder

Folders: stage1 (real graphs), stage2 (synthetic ladder, with the prediction subfolder), stage3
(fixed k, t) and supplement (heap in Python). Figures 1 to 8 use the rows with heap="c".
"""

import argparse
import csv
import math
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "csv")
OUT = os.path.join(ROOT, "results", "figures")
FLAT = False
COLOR = {"bmssp": "#1f77b4", "dijkstra": "#d62728"}
LABEL = {"bmssp": "BMSSP (Duan et al.)", "dijkstra": "Dijkstra"}
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 100})


def read(path):
    """Read a benchmark CSV, converting numbers to int/float and empty cells to None."""
    out = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        for k, v in r.items():
            if v == "":
                r[k] = None
            else:
                try:
                    r[k] = float(v) if "." in v else int(v)
                except ValueError:
                    pass
        out.append(r)
    return out


def groups(rows):
    """Group the rows by graph, in order of appearance. Return [(name, rows)]."""
    order, g = [], {}
    for r in rows:
        if r["graph"] not in g:
            order.append(r["graph"])
            g[r["graph"]] = []
        g[r["graph"]].append(r)
    return [(name, g[name]) for name in order]


def short_name(graph):
    """Short graph label for the axes (NY, COL, 2^20, ...)."""
    return graph.replace("USA-road-d.", "").replace("-m3n", "").replace("rand-", "")


def med(rows, key, fn=None):
    """(median, minimum, maximum) of r[key], or of fn(r) if fn is given, over the rows."""
    vals = [fn(r) if fn else r[key] for r in rows]
    return statistics.median(vals), min(vals), max(vals)


def save(fig, name, folder):
    """Write the figure as PNG and PDF to OUT/<folder>/ (or to OUT itself with --flat) and close it."""
    fig.tight_layout()
    d = OUT if FLAT else os.path.join(OUT, folder)
    os.makedirs(d, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(d, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ", os.path.join(folder, name))


def error_bars(ax, x, rows_per_graph, key, alg, offset, width):
    """Bars of the median of `key` per graph, with an error bar from the minimum to the
    maximum over the sources. Return the medians."""
    meds, lo, hi = zip(*(med(rows, key) for _, rows in rows_per_graph))
    err = [[m - l for m, l in zip(meds, lo)], [h - m for m, h in zip(meds, hi)]]
    ax.bar(x + offset, meds, width, yerr=err, capsize=3, color=COLOR[alg], label=LABEL[alg])
    return meds


def fig_time_real(real):
    """Figure 1: time per source on the road networks, with the BMSSP/Dijkstra ratio."""
    g = groups(real)
    x = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    mb = error_bars(ax, x, g, "t_bmssp_s", "bmssp", -0.2, 0.4)
    md = error_bars(ax, x, g, "t_dijkstra_s", "dijkstra", 0.2, 0.4)
    for i, (b, d) in enumerate(zip(mb, md)):
        ax.text(i, b * 1.25, f"{b / d:.1f}×", ha="center", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(top=ax.get_ylim()[1] * 2)
    ax.set_xticks(x, [f"{short_name(n)}\nn={rows[0]['n'] / 1e6:.2f}M\n{len(rows)} sources" for n, rows in g])
    ax.set_ylabel("Time per source (s), median over sources")
    ax.set_title("DIMACS road networks: running time")
    ax.legend(loc="upper left")
    save(fig, "fig1_time_real", "stage1")


def fig_comparisons_real(real):
    """Figure 2: comparisons per arc on the road networks, split by origin (explicit, heap,
    structure D), and how many times each arc is examined."""
    g = groups(real)
    x = np.arange(len(g))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [3, 2]})
    parts_b = [("bmssp_explicit_comparisons", "explicit (relaxations, bounds)", 1.0),
               ("bmssp_heap_comparisons", "heap of the base cases", 0.65),
               ("bmssp_D_comparisons", "structure D (Lemma 3.3)", 0.35)]
    parts_d = [("dijkstra_explicit_comparisons", "explicit (relaxations)", 1.0),
               ("dijkstra_heap_comparisons", "binary heap", 0.45)]
    for offset, alg, parts in ((-0.2, "bmssp", parts_b), (0.2, "dijkstra", parts_d)):
        base = np.zeros(len(g))
        for key, lab, alpha in parts:
            vals = np.array([med(rows, None, lambda r: r[key] / r["m"])[0] for _, rows in g])
            ax.bar(x + offset, vals, 0.4, bottom=base, color=COLOR[alg], alpha=alpha,
                   edgecolor="white", label=f"{LABEL[alg].split()[0]}: {lab}")
            base += vals
        if alg == "bmssp":
            tot_b = base.copy()
        else:
            for i, (b, d) in enumerate(zip(tot_b, base)):
                ax.text(i, max(b, d) + 0.4, f"{b / d:.2f}×", ha="center", fontsize=9)
    ax.set_xticks(x, [short_name(n) for n, _ in g])
    ax.set_ylabel("Comparisons per arc (total / m)")
    ax.set_title("Comparisons in the comparison-addition model")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
    ax.legend(fontsize=7.5, loc="upper left", ncol=2)

    ex_b = [med(rows, None, lambda r: r["bmssp_arcs_examined"] / r["m"])[0] for _, rows in g]
    ex_d = [med(rows, None, lambda r: r["dijkstra_arcs_examined"] / r["m"])[0] for _, rows in g]
    ax2.bar(x - 0.2, ex_b, 0.4, color=COLOR["bmssp"], label=LABEL["bmssp"])
    ax2.bar(x + 0.2, ex_d, 0.4, color=COLOR["dijkstra"], label=LABEL["dijkstra"])
    for i, v in enumerate(ex_b):
        ax2.text(i - 0.2, v + 0.05, f"{v:.1f}", ha="center", fontsize=9)
    ax2.set_xticks(x, [short_name(n) for n, _ in g])
    ax2.set_ylim(0, max(ex_b) * 1.4)
    ax2.set_ylabel("Arc examinations / m")
    ax2.set_title("How many times each arc is examined")
    ax2.legend(fontsize=8, loc="upper left", ncol=2)
    save(fig, "fig2_comparisons_real", "stage1")


def power_fit(xs, ys):
    """Least-squares fit of y = a·x^b on a log-log scale. Return (a, b)."""
    b, loga = np.polyfit(np.log(xs), np.log(ys), 1)
    return math.exp(loga), b


def series(g, fn):
    """Series over log₂ n: (xs, medians, minima, maxima) of fn(r) for each graph of group g."""
    xs = np.array([math.log2(rows[0]["n"]) for _, rows in g])
    m, lo, hi = zip(*(med(rows, None, fn) for _, rows in g))
    return xs, np.array(m), np.array(lo), np.array(hi)


def fig_ladder_comparisons(synth):
    """Figure 3: comparisons per arc and comparison ratio along the synthetic ladder."""
    gs = groups(synth)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    xx = np.linspace(14.5, 24.5, 50)
    for alg in ("bmssp", "dijkstra"):
        xs, m, lo, hi = series(gs, lambda r, a=alg: r[f"{a}_total_comparisons"] / r["m"])
        a, b = power_fit(xs, m)
        ax.errorbar(xs, m, yerr=[m - lo, hi - m], fmt="o-", color=COLOR[alg], capsize=3,
                    label=f"{LABEL[alg]}")
        ax.plot(xx, a * xx ** b, "--", color=COLOR[alg], alpha=0.6, lw=1,
                label=f"fit ∝ (log₂ n)^{b:.2f}")
    ax.set_xlabel("log₂ n   (stage 2: synthetic ladder of random graphs, m = 3n)")
    ax.set_ylabel("Comparisons per arc")
    ax.set_title("Growth of comparisons with n")
    ax.set_xticks(range(15, 25))
    ax.legend(fontsize=8)

    xs, m, lo, hi = series(gs, lambda r: r["comparison_ratio"])
    ax2.errorbar(xs, m, yerr=[m - lo, hi - m], fmt="o-", color="k", capsize=3, label="synthetic ladder (m = 3n)")
    ax2.axhline(1, ls=":", color="gray", label="tie")
    ax2.set_xlabel("log₂ n")
    ax2.set_ylabel("BMSSP comparisons / Dijkstra comparisons")
    ax2.set_title("Comparison ratio (< 1: BMSSP compares less)")
    ax2.set_ylim(0, hi.max() * 1.15)
    ax2.set_xticks(range(15, 25))
    ax2.legend(fontsize=8)
    save(fig, "fig3_ladder_comparisons", "stage2")


def fig_ladder_time(synth):
    """Figure 4: absolute time and time ratio along the synthetic ladder."""
    gs = groups(synth)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ns = np.array([rows[0]["n"] for _, rows in gs])
    for alg in ("bmssp", "dijkstra"):
        _, m, lo, hi = series(gs, lambda r, a=alg: r[f"t_{a}_s"])
        ax.errorbar(ns, m, yerr=[m - lo, hi - m], fmt="o-", color=COLOR[alg], capsize=3, label=LABEL[alg])
    ax.plot(ns, m[0] * ns / ns[0], ":", color="gray", lw=1, label="linear reference in n")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("n   (stage 2: synthetic ladder of random graphs, m = 3n)")
    ax.set_ylabel("Time per source (s)")
    ax.set_title("Absolute time")
    ax.legend(fontsize=8)

    xs, m, lo, hi = series(gs, lambda r: r["time_ratio"])
    ax2.errorbar(xs, m, yerr=[m - lo, hi - m], fmt="o-", color="k", capsize=3, label="synthetic ladder (m = 3n)")
    ax2.axhline(1, ls=":", color="gray", label="tie")
    ax2.set_ylim(0, max(hi) * 1.1)
    ax2.set_xlabel("log₂ n")
    ax2.set_ylabel("BMSSP time / Dijkstra time")
    ax2.set_title("Wall-clock time ratio (Python)")
    ax2.set_xticks(range(15, 25))
    ax2.legend(fontsize=8)
    save(fig, "fig4_ladder_time", "stage2")


def fig_constants(synth, real):
    """Figure 5: comparisons divided by each algorithm's asymptotic bound
    (m·log^(2/3) n and m·log n), that is, the hidden constants."""
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    norms = {"bmssp": ("m · (log₂ n)^(2/3)", lambda r: r["bmssp_total_comparisons"] / (r["m"] * r["log2n_23"])),
             "dijkstra": ("m · log₂ n", lambda r: r["dijkstra_total_comparisons"] / (r["m"] * r["log2n"]))}
    for alg, (lab, fn) in norms.items():
        for rows, fmt, fam in ((synth, "o-", "synthetic ladder"), (real, "s", "road networks")):
            xs, m, lo, hi = series(groups(rows), fn)
            ax.errorbar(xs, m, yerr=[m - lo, hi - m], fmt=fmt, color=COLOR[alg], capsize=3,
                        mfc=COLOR[alg] if fam == "synthetic ladder" else "white",
                        label=f"{LABEL[alg].split()[0]} / [{lab}]  —  {fam}")
    ax.set_ylim(0, 3.0)
    ax.set_xlabel("log₂ n")
    ax.set_ylabel("Comparisons / asymptotic bound")
    ax.set_title("Hidden constants: comparisons divided by the theoretical bound")
    ax.set_xticks(range(15, 26))
    ax.legend(fontsize=7.5, loc="upper center", ncol=2)
    save(fig, "fig5_constants", "stage2")


def fig_levels(real):
    """Figure 6: maximum |U| and |S| per recursion level as a fraction of the Lemma 3.9 bounds."""
    g = groups(real)
    fig, axes = plt.subplots(1, len(g), figsize=(2.6 * len(g) + 0.6, 3.4), sharey=True)
    for ax, (name, rows) in zip(axes, g):
        top = rows[0]["top_level"]
        levels = list(range(top + 1))
        u = [max(r[f"level{l}_U_max"] / r[f"level{l}_U_bound"] for r in rows) for l in levels]
        s = [max(r[f"level{l}_S_max"] / r[f"level{l}_S_bound"] for r in rows) for l in levels]
        x = np.arange(len(levels))
        ax.bar(x - 0.2, u, 0.4, color=COLOR["bmssp"], label="max |U| / 4k·2^(lt)")
        ax.bar(x + 0.2, s, 0.4, color=COLOR["bmssp"], alpha=0.45, label="max |S| / 2^(lt)")
        ax.axhline(1, ls="--", color="gray", lw=1)
        ax.set_xticks(x, [str(l) for l in levels])
        ax.set_xlabel("level l")
        ax.set_title(f"{short_name(name)}  (k={rows[0]['k']}, t={rows[0]['t']})", fontsize=10)
    axes[0].set_ylabel("fraction of the bound (Lemma 3.9)")
    axes[0].set_ylim(0, 1.15)
    axes[-1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Lemma 3.9 bounds per recursion level (worst source of each graph)", fontsize=11)
    save(fig, "fig6_levels_lemma39", "stage1")


def fig_kt_fixed(real, fixed):
    """Figure 7: time ratio and base cases with the paper's k, t and with fixed k=2, t=6."""
    gf = groups(fixed)
    ga = [(n, rows) for n, rows in groups(real) if n in dict(gf)]
    x = np.arange(len(ga))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for offset, g, color, lab in ((-0.2, ga, COLOR["bmssp"], "k, t of the paper"), (0.2, gf, "#7fb3d5", "k=2, t=6 fixed")):
        m, lo, hi = zip(*(med(rows, "time_ratio") for _, rows in g))
        ax.bar(x + offset, m, 0.4, yerr=[[a - b for a, b in zip(m, lo)], [b - a for a, b in zip(m, hi)]],
               capsize=3, color=color, label=lab)
        cb = [med(rows, "bmssp_basecase_calls")[0] for _, rows in g]
        ax2.bar(x + offset, cb, 0.4, color=color, label=lab)
    for i, (name, rows) in enumerate(ga):
        ax.text(i - 0.2, med(rows, "time_ratio")[2] + 0.3, f"t={rows[0]['t']}", ha="center", fontsize=8)
    ax.set_xticks(x, [short_name(n) for n, _ in ga])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.set_ylabel("BMSSP time / Dijkstra time")
    ax.set_title("Time ratio")
    ax.legend(fontsize=8)
    ax2.set_xticks(x, [short_name(n) for n, _ in ga])
    ax2.set_yscale("log")
    ax2.set_ylabel("BaseCase calls")
    ax2.set_title("Base cases executed")
    ax2.legend(fontsize=8)
    fig.suptitle("Parameter control: t of the paper (⌊log^(2/3) n⌋) vs fixed t", fontsize=11)
    save(fig, "fig7_kt_fixed", "stage3")


def fig_source_spread(real):
    """Figure 8: spread over the sources of the time and comparison ratios."""
    g = groups(real)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.4))
    labels = [f"{short_name(n)}\n({len(rows)} sources)" for n, rows in g]
    ax.boxplot([[r["time_ratio"] for r in rows] for _, rows in g], tick_labels=labels)
    ax.set_ylabel("BMSSP time / Dijkstra time")
    ax.set_title("Time ratio per source")
    ax2.boxplot([[r["comparison_ratio"] for r in rows] for _, rows in g], tick_labels=labels)
    ax2.set_ylabel("BMSSP / Dijkstra comparisons")
    ax2.set_title("Comparison ratio per source")
    save(fig, "fig8_source_spread", "stage1")


def fig_heap_python(synth, synth_py, real, real_py):
    """Figure 9: time ratio with heapq in C and with the pure heapq.py."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for rs, rr, color, lab in ((synth, real, "k", "heapq in C"), (synth_py, real_py, "#ff7f0e", "pure heapq.py")):
        xs, m, lo, hi = series(groups(rs), lambda r: r["time_ratio"])
        ax.errorbar(xs, m, yerr=[m - lo, hi - m], fmt="o-", color=color, capsize=3, label=f"synthetic ladder, {lab}")
        if rr:
            xr, mr, lr, hr = series(groups(rr), lambda r: r["time_ratio"])
            ax.errorbar(xr, mr, yerr=[mr - lr, hr - mr], fmt="s", mfc="white", color=color, capsize=3,
                        label=f"road networks, {lab}")
    ax.axhline(1, ls=":", color="gray")
    ax.set_xlabel("log₂ n")
    ax.set_ylabel("BMSSP time / Dijkstra time")
    ax.set_title("Time ratio with the C heap and with the Python heap")
    ax.set_xticks(range(15, 26))
    ax.legend(fontsize=8)
    save(fig, "fig9_heap_python", "supplement")


def linear_fit(xs, ys):
    """Least-squares fit of y = a + b·x. Return (a, b, standard deviation of b)."""
    (b, a), cov = np.polyfit(xs, ys, 1, cov=True)
    return a, b, math.sqrt(cov[0, 0])


def fig_prediction(synth, synth_py):
    """Extrapolations (curiosity): linear fits in log₂ n for the comparisons and power laws
    in n for the time, extended to the point where BMSSP and Dijkstra meet."""
    gs = groups(synth)
    xx = np.linspace(15, 48, 100)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    fits = {}
    for alg in ("bmssp", "dijkstra"):
        xs, m, _, _ = series(gs, lambda r, a=alg: r[f"{a}_total_comparisons"] / r["m"])
        fits[alg] = linear_fit(xs, m)
        ax.plot(xs, m, "o", color=COLOR[alg])
    (ab, bb, sb), (ad, bd, sd) = fits["bmssp"], fits["dijkstra"]
    if bd > bb:
        x_cross = (ab - ad) / (bd - bb)
        dx = x_cross * math.hypot(sd, sb) / (bd - bb)
        note = f"  →  meeting at n ≈ 2^{x_cross:.0f} (±{dx:.0f})"
        ax.axvline(x_cross, ls=":", color="gray", lw=1)
    else:
        note = "  →  no meeting"
    ax.plot(xx, ab + bb * xx, "--", color=COLOR["bmssp"], lw=1, label=f"BMSSP: {ab:.1f} + {bb:.2f}·log₂n{note}")
    ax.plot(xx, ad + bd * xx, "--", color=COLOR["dijkstra"], lw=1, label=f"Dijkstra: {ad:.1f} + {bd:.2f}·log₂n")
    ax.axvspan(15, 24, color="gray", alpha=0.08)
    ax.text(19.5, ax.get_ylim()[1] * 0.97, "measured", ha="center", va="top", fontsize=9, color="gray")
    ax.text(36, ax.get_ylim()[1] * 0.97, "extrapolation (linear fit in log₂ n)", ha="center", va="top", fontsize=9, color="gray")
    ax.set_xlim(15, 48)
    ax.set_xlabel("log₂ n   (synthetic ladder, m = 3n)")
    ax.set_ylabel("Comparisons per arc")
    ax.set_title("Prediction: where BMSSP starts to compare less than Dijkstra")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "pred1_comparisons", "stage2/prediction")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ns = np.array([rows[0]["n"] for _, rows in gs])
    curves = [("BMSSP", synth, "t_bmssp_s", COLOR["bmssp"], "o-"),
              ("Dijkstra, heapq in C", synth, "t_dijkstra_s", COLOR["dijkstra"], "o-")]
    if synth_py:
        curves.append(("Dijkstra, pure heapq.py", synth_py, "t_dijkstra_s", "#ff7f0e", "s-"))
    nn = np.logspace(15, 34, 100, base=2)
    laws = {}
    for lab, rows, key, color, fmt in curves:
        _, m, _, _ = series(groups(rows), lambda r, c=key: r[c])
        laws[lab] = power_fit(ns, m)
        ax.plot(ns, m, fmt, color=color)
        ax.plot(nn, laws[lab][0] * nn ** laws[lab][1], "--", color=color, lw=1, label=f"{lab}: t ∝ n^{laws[lab][1]:.2f}")
    cb, ab_ = laws["BMSSP"]
    notes = []
    for lab, (cd, ad_) in laws.items():
        if lab != "BMSSP" and ad_ > ab_:
            n_cross = (cb / cd) ** (1 / (ad_ - ab_))
            ax.axvline(n_cross, ls=":", color="gray", lw=1)
            notes.append(f"{lab}: meeting at n ≈ 2^{math.log2(n_cross):.0f}, ~{cb * n_cross ** ab_ / 3600:.0f} h per source")
    ax.axvspan(2 ** 15, 2 ** 24, color="gray", alpha=0.08)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(2 ** 15, 2 ** 34)
    ax.set_xlabel("n   (synthetic ladder, m = 3n)")
    ax.set_ylabel("Time per source (s)")
    ax.set_title("Prediction: where the running times meet (fitted power laws)")
    ax.legend(fontsize=7.5, loc="upper left", title="\n".join(notes) or None, title_fontsize=7.5)
    save(fig, "pred2_time", "stage2/prediction")


def main():
    """Read the CSVs in results/csv/ and generate every figure; figures 1 to 8 use heap="c" only."""
    global OUT, FLAT
    ap = argparse.ArgumentParser(description="benchmark figures")
    ap.add_argument("--flat", action="store_true", help="every figure in one folder, no per-stage subfolders")
    ap.add_argument("--output", default=OUT, help="output folder")
    a = ap.parse_args()
    OUT, FLAT = a.output, a.flat
    os.makedirs(OUT, exist_ok=True)
    everything = read(os.path.join(CSV, "results.csv"))
    if os.path.exists(os.path.join(CSV, "results_usa.csv")):
        everything += read(os.path.join(CSV, "results_usa.csv"))
    everything_synth = read(os.path.join(CSV, "results_synthetic.csv"))
    fixed = read(os.path.join(CSV, "results_kt_fixed.csv"))
    heap = lambda rows, h: [r for r in rows if r.get("heap", "c") == h]
    real, synth = heap(everything, "c"), heap(everything_synth, "c")
    print(f"figures in {OUT}/")
    fig_time_real(real)
    fig_comparisons_real(real)
    fig_ladder_comparisons(synth)
    fig_ladder_time(synth)
    fig_constants(synth, real)
    fig_levels(real)
    fig_kt_fixed(real, fixed)
    fig_source_spread(real)
    if heap(everything_synth, "python"):
        fig_heap_python(synth, heap(everything_synth, "python"), real, heap(everything, "python"))
    fig_prediction(synth, heap(everything_synth, "python"))


if __name__ == "__main__":
    main()
