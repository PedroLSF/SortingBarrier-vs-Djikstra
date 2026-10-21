#!/usr/bin/env python3
"""Gera as figuras do relatório a partir dos CSVs do benchmark.

Uso:
    .venv/bin/python src/graficos.py   # lê resultados/csv/*.csv, escreve resultados/figuras/<etapa>/*.png e *.pdf
    .venv/bin/python src/graficos.py --en --plano --saida <pasta>   # versão em inglês, numa pasta só

Pastas: etapa1 (grafos reais), etapa2 (escada sintética, com a subpasta previsao), etapa3
(k, t fixos) e complemento (heap em Python). As figuras 1 a 8 usam as linhas com heap="c".
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

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(RAIZ, "resultados", "csv")
SAIDA = os.path.join(RAIZ, "resultados", "figuras")
COR = {"bmssp": "#1f77b4", "dijkstra": "#d62728"}
ROTULO = {"bmssp": "BMSSP (Duan et al.)", "dijkstra": "Dijkstra"}
PLANO = False
LANG = "pt"
EN = {
    "{n} fontes": "{n} sources",
    "Tempo por fonte (s), mediana entre fontes": "Time per source (s), median over sources",
    "Malhas rodoviárias DIMACS: tempo de execução": "DIMACS road networks: running time",
    "explícitas (relaxações, cotas)": "explicit (relaxations, bounds)",
    "heap dos casos base": "heap of the base cases",
    "estrutura D (Lema 3.3)": "structure D (Lemma 3.3)",
    "explícitas (relaxações)": "explicit (relaxations)",
    "heap binário": "binary heap",
    "Comparações por aresta (total / m)": "Comparisons per arc (total / m)",
    "Comparações no modelo comparação-adição": "Comparisons in the comparison-addition model",
    "Exames de aresta / m": "Arc examinations / m",
    "Quantas vezes cada aresta é examinada": "How many times each arc is examined",
    "ajuste ∝ (log₂ n)^{b:.2f}": "fit ∝ (log₂ n)^{b:.2f}",
    "log₂ n   (etapa 2: escada sintética de grafos aleatórios, m = 3n)": "log₂ n   (stage 2: synthetic ladder of random graphs, m = 3n)",
    "Comparações por aresta": "Comparisons per arc",
    "Crescimento das comparações com n": "Growth of comparisons with n",
    "escada sintética (m = 3n)": "synthetic ladder (m = 3n)",
    "empate": "tie",
    "Comparações BMSSP / comparações Dijkstra": "BMSSP comparisons / Dijkstra comparisons",
    "Razão de comparações (< 1: BMSSP compara menos)": "Comparison ratio (< 1: BMSSP compares less)",
    "referência linear em n": "linear reference in n",
    "n   (etapa 2: escada sintética de grafos aleatórios, m = 3n)": "n   (stage 2: synthetic ladder of random graphs, m = 3n)",
    "Tempo por fonte (s)": "Time per source (s)",
    "Tempo absoluto": "Absolute time",
    "Tempo BMSSP / tempo Dijkstra": "BMSSP time / Dijkstra time",
    "Razão de tempo de parede (Python)": "Wall-clock time ratio (Python)",
    "escada sintética": "synthetic ladder",
    "rodoviários": "road networks",
    "Comparações / cota assintótica": "Comparisons / asymptotic bound",
    "Constantes ocultas: comparações divididas pela cota teórica": "Hidden constants: comparisons divided by the theoretical bound",
    "|U| máx / 4k·2^(lt)": "max |U| / 4k·2^(lt)",
    "|S| máx / 2^(lt)": "max |S| / 2^(lt)",
    "nível l": "level l",
    "fração da cota (Lema 3.9)": "fraction of the bound (Lemma 3.9)",
    "Cotas do Lema 3.9 por nível de recursão (pior fonte de cada grafo)": "Lemma 3.9 bounds per recursion level (worst source of each graph)",
    "k, t do artigo": "k, t of the paper",
    "k=2, t=6 fixos": "k=2, t=6 fixed",
    "Razão de tempo": "Time ratio",
    "Chamadas de BaseCase": "BaseCase calls",
    "Casos base executados": "Base cases executed",
    "Controle de parâmetros: t do artigo (⌊log^(2/3) n⌋) vs t fixo": "Parameter control: t of the paper (⌊log^(2/3) n⌋) vs fixed t",
    "{nome}\n({n} fontes)": "{nome}\n({n} sources)",
    "Razão de tempo por fonte": "Time ratio per source",
    "Comparações BMSSP / Dijkstra": "BMSSP / Dijkstra comparisons",
    "Razão de comparações por fonte": "Comparison ratio per source",
    "heapq em C": "heapq in C",
    "heapq.py puro": "pure heapq.py",
    "escada sintética, {rot}": "synthetic ladder, {rot}",
    "rodoviários, {rot}": "road networks, {rot}",
    "Razão de tempo com o heap em C e com o heap em Python": "Time ratio with the C heap and with the Python heap",
    "  →  encontro em n ≈ 2^{x:.0f} (±{dx:.0f})": "  →  meeting at n ≈ 2^{x:.0f} (±{dx:.0f})",
    "  →  sem encontro": "  →  no meeting",
    "medido": "measured",
    "extrapolação (ajuste linear em log₂ n)": "extrapolation (linear fit in log₂ n)",
    "log₂ n   (escada sintética, m = 3n)": "log₂ n   (synthetic ladder, m = 3n)",
    "Previsão: onde o BMSSP passa a comparar menos que o Dijkstra": "Prediction: where BMSSP starts to compare less than Dijkstra",
    "Dijkstra, heapq em C": "Dijkstra, heapq in C",
    "Dijkstra, heapq.py puro": "Dijkstra, pure heapq.py",
    "{rot}: encontro em n ≈ 2^{k:.0f}, ~{h:.0f} h por fonte": "{rot}: meeting at n ≈ 2^{k:.0f}, ~{h:.0f} h per source",
    "n   (escada sintética, m = 3n)": "n   (synthetic ladder, m = 3n)",
    "Previsão: onde os tempos se encontram (leis de potência ajustadas)": "Prediction: where the running times meet (fitted power laws)",
}


def T(texto):
    return EN.get(texto, texto) if LANG == "en" else texto
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 100})


def le(path):
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


def grupos(rows):
    ordem, g = [], {}
    for r in rows:
        if r["grafo"] not in g:
            ordem.append(r["grafo"])
            g[r["grafo"]] = []
        g[r["grafo"]].append(r)
    return [(nome, g[nome]) for nome in ordem]


def nome_curto(grafo):
    return grafo.replace("USA-road-d.", "").replace("-m3n", "").replace("rand-", "")


def med(rows, chave, fn=None):
    vals = [fn(r) if fn else r[chave] for r in rows]
    return statistics.median(vals), min(vals), max(vals)


def salva(fig, nome, pasta):
    fig.tight_layout()
    d = SAIDA if PLANO else os.path.join(SAIDA, pasta)
    os.makedirs(d, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(d, f"{nome}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ", os.path.join(pasta, nome))


def barras_err(ax, x, rows_por_grafo, chave, alg, desloc, largura):
    meds, lo, hi = zip(*(med(rows, chave) for _, rows in rows_por_grafo))
    err = [[m - l for m, l in zip(meds, lo)], [h - m for m, h in zip(meds, hi)]]
    ax.bar(x + desloc, meds, largura, yerr=err, capsize=3, color=COR[alg], label=ROTULO[alg])
    return meds


def fig_tempo_reais(reais):
    g = grupos(reais)
    x = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    mb = barras_err(ax, x, g, "t_bmssp_s", "bmssp", -0.2, 0.4)
    md = barras_err(ax, x, g, "t_dijkstra_s", "dijkstra", 0.2, 0.4)
    for i, (b, d) in enumerate(zip(mb, md)):
        ax.text(i, b * 1.25, f"{b / d:.1f}×", ha="center", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(top=ax.get_ylim()[1] * 2)
    ax.set_xticks(x, [f"{nome_curto(n)}\nn={rows[0]['n'] / 1e6:.2f}M\n" + T("{n} fontes").format(n=len(rows)) for n, rows in g])
    ax.set_ylabel(T("Tempo por fonte (s), mediana entre fontes"))
    ax.set_title(T("Malhas rodoviárias DIMACS: tempo de execução"))
    ax.legend(loc="upper left")
    salva(fig, "fig1_tempo_reais", "etapa1")


def fig_comparacoes_reais(reais):
    g = grupos(reais)
    x = np.arange(len(g))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [3, 2]})
    partes_b = [("bmssp_comparacoes_explicitas", T("explícitas (relaxações, cotas)"), 1.0),
                ("bmssp_comparacoes_heap", T("heap dos casos base"), 0.65),
                ("bmssp_comparacoes_D", T("estrutura D (Lema 3.3)"), 0.35)]
    partes_d = [("dijkstra_comparacoes_explicitas", T("explícitas (relaxações)"), 1.0),
                ("dijkstra_comparacoes_heap", T("heap binário"), 0.45)]
    for desloc, alg, partes in ((-0.2, "bmssp", partes_b), (0.2, "dijkstra", partes_d)):
        base = np.zeros(len(g))
        for chave, rot, alfa in partes:
            vals = np.array([med(rows, None, lambda r: r[chave] / r["m"])[0] for _, rows in g])
            ax.bar(x + desloc, vals, 0.4, bottom=base, color=COR[alg], alpha=alfa,
                   edgecolor="white", label=f"{ROTULO[alg].split()[0]}: {rot}")
            base += vals
        if alg == "bmssp":
            tot_b = base.copy()
        else:
            for i, (b, d) in enumerate(zip(tot_b, base)):
                ax.text(i, max(b, d) + 0.4, f"{b / d:.2f}×", ha="center", fontsize=9)
    ax.set_xticks(x, [nome_curto(n) for n, _ in g])
    ax.set_ylabel(T("Comparações por aresta (total / m)"))
    ax.set_title(T("Comparações no modelo comparação-adição"))
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
    ax.legend(fontsize=7.5, loc="upper left", ncol=2)

    ex_b = [med(rows, None, lambda r: r["bmssp_arestas_examinadas"] / r["m"])[0] for _, rows in g]
    ex_d = [med(rows, None, lambda r: r["dijkstra_arestas_examinadas"] / r["m"])[0] for _, rows in g]
    ax2.bar(x - 0.2, ex_b, 0.4, color=COR["bmssp"], label=ROTULO["bmssp"])
    ax2.bar(x + 0.2, ex_d, 0.4, color=COR["dijkstra"], label=ROTULO["dijkstra"])
    for i, v in enumerate(ex_b):
        ax2.text(i - 0.2, v + 0.05, f"{v:.1f}", ha="center", fontsize=9)
    ax2.set_xticks(x, [nome_curto(n) for n, _ in g])
    ax2.set_ylim(0, max(ex_b) * 1.4)
    ax2.set_ylabel(T("Exames de aresta / m"))
    ax2.set_title(T("Quantas vezes cada aresta é examinada"))
    ax2.legend(fontsize=8, loc="upper left", ncol=2)
    salva(fig, "fig2_comparacoes_reais", "etapa1")


def ajuste_potencia(xs, ys):
    b, loga = np.polyfit(np.log(xs), np.log(ys), 1)
    return math.exp(loga), b


def serie(g, fn):
    xs = np.array([math.log2(rows[0]["n"]) for _, rows in g])
    m, lo, hi = zip(*(med(rows, None, fn) for _, rows in g))
    return xs, np.array(m), np.array(lo), np.array(hi)


def fig_escada_comparacoes(sint):
    gs = grupos(sint)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    xx = np.linspace(14.5, 24.5, 50)
    for alg in ("bmssp", "dijkstra"):
        xs, m, lo, hi = serie(gs, lambda r, a=alg: r[f"{a}_comparacoes_total"] / r["m"])
        a, b = ajuste_potencia(xs, m)
        ax.errorbar(xs, m, yerr=[m - lo, hi - m], fmt="o-", color=COR[alg], capsize=3,
                    label=f"{ROTULO[alg]}")
        ax.plot(xx, a * xx ** b, "--", color=COR[alg], alpha=0.6, lw=1,
                label=T("ajuste ∝ (log₂ n)^{b:.2f}").format(b=b))
    ax.set_xlabel(T("log₂ n   (etapa 2: escada sintética de grafos aleatórios, m = 3n)"))
    ax.set_ylabel(T("Comparações por aresta"))
    ax.set_title(T("Crescimento das comparações com n"))
    ax.set_xticks(range(15, 25))
    ax.legend(fontsize=8)

    xs, m, lo, hi = serie(gs, lambda r: r["razao_comparacoes"])
    ax2.errorbar(xs, m, yerr=[m - lo, hi - m], fmt="o-", color="k", capsize=3, label=T("escada sintética (m = 3n)"))
    ax2.axhline(1, ls=":", color="gray", label=T("empate"))
    ax2.set_xlabel("log₂ n")
    ax2.set_ylabel(T("Comparações BMSSP / comparações Dijkstra"))
    ax2.set_title(T("Razão de comparações (< 1: BMSSP compara menos)"))
    ax2.set_ylim(0, hi.max() * 1.15)
    ax2.set_xticks(range(15, 25))
    ax2.legend(fontsize=8)
    salva(fig, "fig3_escada_comparacoes", "etapa2")


def fig_escada_tempo(sint):
    gs = grupos(sint)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ns = np.array([rows[0]["n"] for _, rows in gs])
    for alg in ("bmssp", "dijkstra"):
        _, m, lo, hi = serie(gs, lambda r, a=alg: r[f"t_{a}_s"])
        ax.errorbar(ns, m, yerr=[m - lo, hi - m], fmt="o-", color=COR[alg], capsize=3, label=ROTULO[alg])
    ax.plot(ns, m[0] * ns / ns[0], ":", color="gray", lw=1, label=T("referência linear em n"))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(T("n   (etapa 2: escada sintética de grafos aleatórios, m = 3n)"))
    ax.set_ylabel(T("Tempo por fonte (s)"))
    ax.set_title(T("Tempo absoluto"))
    ax.legend(fontsize=8)

    xs, m, lo, hi = serie(gs, lambda r: r["razao"])
    ax2.errorbar(xs, m, yerr=[m - lo, hi - m], fmt="o-", color="k", capsize=3, label=T("escada sintética (m = 3n)"))
    ax2.axhline(1, ls=":", color="gray", label=T("empate"))
    ax2.set_ylim(0, max(hi) * 1.1)
    ax2.set_xlabel("log₂ n")
    ax2.set_ylabel(T("Tempo BMSSP / tempo Dijkstra"))
    ax2.set_title(T("Razão de tempo de parede (Python)"))
    ax2.set_xticks(range(15, 25))
    ax2.legend(fontsize=8)
    salva(fig, "fig4_escada_tempo", "etapa2")


def fig_constantes(sint, reais):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    normas = {"bmssp": ("m · (log₂ n)^(2/3)", lambda r: r["bmssp_comparacoes_total"] / (r["m"] * r["log2n_23"])),
              "dijkstra": ("m · log₂ n", lambda r: r["dijkstra_comparacoes_total"] / (r["m"] * r["log2n"]))}
    for alg, (rot, fn) in normas.items():
        for rows, fmt, fam in ((sint, "o-", T("escada sintética")), (reais, "s", T("rodoviários"))):
            xs, m, lo, hi = serie(grupos(rows), fn)
            ax.errorbar(xs, m, yerr=[m - lo, hi - m], fmt=fmt, color=COR[alg], capsize=3,
                        mfc=COR[alg] if fam == T("escada sintética") else "white",
                        label=f"{ROTULO[alg].split()[0]} / [{rot}]  —  {fam}")
    ax.set_ylim(0, 3.0)
    ax.set_xlabel("log₂ n")
    ax.set_ylabel(T("Comparações / cota assintótica"))
    ax.set_title(T("Constantes ocultas: comparações divididas pela cota teórica"))
    ax.set_xticks(range(15, 26))
    ax.legend(fontsize=7.5, loc="upper center", ncol=2)
    salva(fig, "fig5_constantes", "etapa2")


def fig_niveis(reais):
    g = grupos(reais)
    fig, axes = plt.subplots(1, len(g), figsize=(2.6 * len(g) + 0.6, 3.4), sharey=True)
    for ax, (nome, rows) in zip(axes, g):
        topo = rows[0]["nivel_topo"]
        niveis = list(range(topo + 1))
        u = [max(r[f"nivel{l}_U_max"] / r[f"nivel{l}_U_cota"] for r in rows) for l in niveis]
        s = [max(r[f"nivel{l}_S_max"] / r[f"nivel{l}_S_cota"] for r in rows) for l in niveis]
        x = np.arange(len(niveis))
        ax.bar(x - 0.2, u, 0.4, color=COR["bmssp"], label=T("|U| máx / 4k·2^(lt)"))
        ax.bar(x + 0.2, s, 0.4, color=COR["bmssp"], alpha=0.45, label=T("|S| máx / 2^(lt)"))
        ax.axhline(1, ls="--", color="gray", lw=1)
        ax.set_xticks(x, [str(l) for l in niveis])
        ax.set_xlabel(T("nível l"))
        ax.set_title(f"{nome_curto(nome)}  (k={rows[0]['k']}, t={rows[0]['t']})", fontsize=10)
    axes[0].set_ylabel(T("fração da cota (Lema 3.9)"))
    axes[0].set_ylim(0, 1.15)
    axes[-1].legend(fontsize=8, loc="upper right")
    fig.suptitle(T("Cotas do Lema 3.9 por nível de recursão (pior fonte de cada grafo)"), fontsize=11)
    salva(fig, "fig6_niveis_lema39", "etapa1")


def fig_kt_fixo(reais, fixo):
    gf = grupos(fixo)
    ga = [(n, rows) for n, rows in grupos(reais) if n in dict(gf)]
    x = np.arange(len(ga))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for desloc, g, cor, rot in ((-0.2, ga, COR["bmssp"], T("k, t do artigo")), (0.2, gf, "#7fb3d5", T("k=2, t=6 fixos"))):
        m, lo, hi = zip(*(med(rows, "razao") for _, rows in g))
        ax.bar(x + desloc, m, 0.4, yerr=[[a - b for a, b in zip(m, lo)], [b - a for a, b in zip(m, hi)]],
               capsize=3, color=cor, label=rot)
        cb = [med(rows, "bmssp_chamadas_base")[0] for _, rows in g]
        ax2.bar(x + desloc, cb, 0.4, color=cor, label=rot)
    for i, (nome, rows) in enumerate(ga):
        ax.text(i - 0.2, med(rows, "razao")[2] + 0.3, f"t={rows[0]['t']}", ha="center", fontsize=8)
    ax.set_xticks(x, [nome_curto(n) for n, _ in ga])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.set_ylabel(T("Tempo BMSSP / tempo Dijkstra"))
    ax.set_title(T("Razão de tempo"))
    ax.legend(fontsize=8)
    ax2.set_xticks(x, [nome_curto(n) for n, _ in ga])
    ax2.set_yscale("log")
    ax2.set_ylabel(T("Chamadas de BaseCase"))
    ax2.set_title(T("Casos base executados"))
    ax2.legend(fontsize=8)
    fig.suptitle(T("Controle de parâmetros: t do artigo (⌊log^(2/3) n⌋) vs t fixo"), fontsize=11)
    salva(fig, "fig7_kt_fixo", "etapa3")


def fig_dispersao(reais):
    g = grupos(reais)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.4))
    rotulos = [T("{nome}\n({n} fontes)").format(nome=nome_curto(n), n=len(rows)) for n, rows in g]
    ax.boxplot([[r["razao"] for r in rows] for _, rows in g], tick_labels=rotulos)
    ax.set_ylabel(T("Tempo BMSSP / tempo Dijkstra"))
    ax.set_title(T("Razão de tempo por fonte"))
    ax2.boxplot([[r["razao_comparacoes"] for r in rows] for _, rows in g], tick_labels=rotulos)
    ax2.set_ylabel(T("Comparações BMSSP / Dijkstra"))
    ax2.set_title(T("Razão de comparações por fonte"))
    salva(fig, "fig8_dispersao_fontes", "etapa1")


def fig_heap_python(sint, sint_py, reais, reais_py):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for rs, rr, cor, rot in ((sint, reais, "k", T("heapq em C")), (sint_py, reais_py, "#ff7f0e", T("heapq.py puro"))):
        xs, m, lo, hi = serie(grupos(rs), lambda r: r["razao"])
        ax.errorbar(xs, m, yerr=[m - lo, hi - m], fmt="o-", color=cor, capsize=3, label=T("escada sintética, {rot}").format(rot=rot))
        if rr:
            xr, mr, lr, hr = serie(grupos(rr), lambda r: r["razao"])
            ax.errorbar(xr, mr, yerr=[mr - lr, hr - mr], fmt="s", mfc="white", color=cor, capsize=3,
                        label=T("rodoviários, {rot}").format(rot=rot))
    ax.axhline(1, ls=":", color="gray")
    ax.set_xlabel("log₂ n")
    ax.set_ylabel(T("Tempo BMSSP / tempo Dijkstra"))
    ax.set_title(T("Razão de tempo com o heap em C e com o heap em Python"))
    ax.set_xticks(range(15, 26))
    ax.legend(fontsize=8)
    salva(fig, "fig9_heap_python", "complemento")


def ajuste_linear(xs, ys):
    (b, a), cov = np.polyfit(xs, ys, 1, cov=True)
    return a, b, math.sqrt(cov[0, 0])


def fig_previsao(sint, sint_py):
    """Extrapolações (curiosidade): ajustes lineares em log₂ n para as comparações e leis de
    potência em n para o tempo, prolongados até o ponto em que BMSSP e Dijkstra se encontram."""
    gs = grupos(sint)
    xx = np.linspace(15, 48, 100)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ajustes = {}
    for alg in ("bmssp", "dijkstra"):
        xs, m, _, _ = serie(gs, lambda r, a=alg: r[f"{a}_comparacoes_total"] / r["m"])
        ajustes[alg] = ajuste_linear(xs, m)
        ax.plot(xs, m, "o", color=COR[alg])
    (ab, bb, sb), (ad, bd, sd) = ajustes["bmssp"], ajustes["dijkstra"]
    if bd > bb:
        x_cruz = (ab - ad) / (bd - bb)
        dx = x_cruz * math.hypot(sd, sb) / (bd - bb)
        nota = T("  →  encontro em n ≈ 2^{x:.0f} (±{dx:.0f})").format(x=x_cruz, dx=dx)
        ax.axvline(x_cruz, ls=":", color="gray", lw=1)
    else:
        nota = T("  →  sem encontro")
    ax.plot(xx, ab + bb * xx, "--", color=COR["bmssp"], lw=1, label=f"BMSSP: {ab:.1f} + {bb:.2f}·log₂n{nota}")
    ax.plot(xx, ad + bd * xx, "--", color=COR["dijkstra"], lw=1, label=f"Dijkstra: {ad:.1f} + {bd:.2f}·log₂n")
    ax.axvspan(15, 24, color="gray", alpha=0.08)
    ax.text(19.5, ax.get_ylim()[1] * 0.97, T("medido"), ha="center", va="top", fontsize=9, color="gray")
    ax.text(36, ax.get_ylim()[1] * 0.97, T("extrapolação (ajuste linear em log₂ n)"), ha="center", va="top", fontsize=9, color="gray")
    ax.set_xlim(15, 48)
    ax.set_xlabel(T("log₂ n   (escada sintética, m = 3n)"))
    ax.set_ylabel(T("Comparações por aresta"))
    ax.set_title(T("Previsão: onde o BMSSP passa a comparar menos que o Dijkstra"))
    ax.legend(fontsize=8, loc="lower right")
    salva(fig, "prev1_comparacoes", "etapa2/previsao")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ns = np.array([rows[0]["n"] for _, rows in gs])
    series = [("BMSSP", sint, "t_bmssp_s", COR["bmssp"], "o-"), (T("Dijkstra, heapq em C"), sint, "t_dijkstra_s", COR["dijkstra"], "o-")]
    if sint_py:
        series.append((T("Dijkstra, heapq.py puro"), sint_py, "t_dijkstra_s", "#ff7f0e", "s-"))
    nn = np.logspace(15, 34, 100, base=2)
    leis = {}
    for rot, rows, chave, cor, fmt in series:
        _, m, _, _ = serie(grupos(rows), lambda r, c=chave: r[c])
        leis[rot] = ajuste_potencia(ns, m)
        ax.plot(ns, m, fmt, color=cor)
        ax.plot(nn, leis[rot][0] * nn ** leis[rot][1], "--", color=cor, lw=1, label=f"{rot}: t ∝ n^{leis[rot][1]:.2f}")
    cb, ab_ = leis["BMSSP"]
    notas = []
    for rot, (cd, ad_) in leis.items():
        if rot != "BMSSP" and ad_ > ab_:
            n_cruz = (cb / cd) ** (1 / (ad_ - ab_))
            ax.axvline(n_cruz, ls=":", color="gray", lw=1)
            notas.append(T("{rot}: encontro em n ≈ 2^{k:.0f}, ~{h:.0f} h por fonte").format(rot=rot, k=math.log2(n_cruz), h=cb * n_cruz ** ab_ / 3600))
    ax.axvspan(2 ** 15, 2 ** 24, color="gray", alpha=0.08)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(2 ** 15, 2 ** 34)
    ax.set_xlabel(T("n   (escada sintética, m = 3n)"))
    ax.set_ylabel(T("Tempo por fonte (s)"))
    ax.set_title(T("Previsão: onde os tempos se encontram (leis de potência ajustadas)"))
    ax.legend(fontsize=7.5, loc="upper left", title="\n".join(notas) or None, title_fontsize=7.5)
    salva(fig, "prev2_tempo", "etapa2/previsao")


def main():
    global SAIDA, PLANO, LANG
    ap = argparse.ArgumentParser(description="figuras do benchmark")
    ap.add_argument("--en", action="store_true", help="textos em inglês")
    ap.add_argument("--plano", action="store_true", help="tudo numa pasta só, sem subpastas por etapa")
    ap.add_argument("--saida", default=SAIDA, help="pasta de saída")
    a = ap.parse_args()
    SAIDA, PLANO, LANG = a.saida, a.plano, "en" if a.en else "pt"
    os.makedirs(SAIDA, exist_ok=True)
    todos = le(os.path.join(CSV, "resultados.csv"))
    if os.path.exists(os.path.join(CSV, "resultados_usa.csv")):
        todos += le(os.path.join(CSV, "resultados_usa.csv"))
    todos_sint = le(os.path.join(CSV, "resultados_sintetico.csv"))
    fixo = le(os.path.join(CSV, "resultados_kt_fixo.csv"))
    heap = lambda rows, h: [r for r in rows if r.get("heap", "c") == h]
    reais, sint = heap(todos, "c"), heap(todos_sint, "c")
    print(f"figuras em {SAIDA}/")
    fig_tempo_reais(reais)
    fig_comparacoes_reais(reais)
    fig_escada_comparacoes(sint)
    fig_escada_tempo(sint)
    fig_constantes(sint, reais)
    fig_niveis(reais)
    fig_kt_fixo(reais, fixo)
    fig_dispersao(reais)
    if heap(todos_sint, "python"):
        fig_heap_python(sint, heap(todos_sint, "python"), reais, heap(todos, "python"))
    fig_previsao(sint, heap(todos_sint, "python"))


if __name__ == "__main__":
    main()
