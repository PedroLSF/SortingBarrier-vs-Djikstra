#!/usr/bin/env python3
"""Gera as figuras do relatório a partir dos CSVs do benchmark.

Uso:
    ../.venv/bin/python graficos.py            # lê resultados*.csv, escreve figuras/*.png e *.pdf
"""

import csv
import math
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SAIDA = "figuras"
COR = {"bmssp": "#1f77b4", "dijkstra": "#d62728"}
ROTULO = {"bmssp": "BMSSP (Duan et al.)", "dijkstra": "Dijkstra"}
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


def salva(fig, nome):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(SAIDA, f"{nome}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ", nome)


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
    ax.set_xticks(x, [f"{nome_curto(n)}\nn={rows[0]['n'] / 1e6:.2f}M\n{len(rows)} fontes" for n, rows in g])
    ax.set_ylabel("Tempo por fonte (s), mediana entre fontes")
    ax.set_title("Malhas rodoviárias DIMACS: tempo de execução")
    ax.legend(loc="upper left")
    salva(fig, "fig1_tempo_reais")


def fig_comparacoes_reais(reais):
    g = grupos(reais)
    x = np.arange(len(g))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [3, 2]})
    partes_b = [("bmssp_comparacoes_explicitas", "explícitas (relaxações, cotas)", 1.0),
                ("bmssp_comparacoes_heap_est", "heap dos casos base", 0.65),
                ("bmssp_comparacoes_D_est", "estrutura D (Lema 3.3)", 0.35)]
    partes_d = [("dijkstra_comparacoes_explicitas", "explícitas (relaxações)", 1.0),
                ("dijkstra_comparacoes_heap_est", "heap binário", 0.45)]
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
    ax.set_ylabel("Comparações por aresta (total / m)")
    ax.set_title("Comparações no modelo comparação-adição")
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
    ax2.set_ylabel("Exames de aresta / m")
    ax2.set_title("Quantas vezes cada aresta é examinada")
    ax2.legend(fontsize=8, loc="upper left", ncol=2)
    salva(fig, "fig2_comparacoes_reais")


def ajuste_potencia(xs, ys):
    b, loga = np.polyfit(np.log(xs), np.log(ys), 1)
    return math.exp(loga), b


def serie(g, fn):
    xs = np.array([math.log2(rows[0]["n"]) for _, rows in g])
    m, lo, hi = zip(*(med(rows, None, fn) for _, rows in g))
    return xs, np.array(m), np.array(lo), np.array(hi)


def fig_escada_comparacoes(sint, reais):
    gs, gr = grupos(sint), grupos(reais)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    xx = np.linspace(14.5, 22.5, 50)
    for alg in ("bmssp", "dijkstra"):
        xs, m, lo, hi = serie(gs, lambda r, a=alg: r[f"{a}_comparacoes_total"] / r["m"])
        a, b = ajuste_potencia(xs, m)
        ax.errorbar(xs, m, yerr=[m - lo, hi - m], fmt="o-", color=COR[alg], capsize=3,
                    label=f"{ROTULO[alg]}")
        ax.plot(xx, a * xx ** b, "--", color=COR[alg], alpha=0.6, lw=1,
                label=f"ajuste ∝ (log₂ n)^{b:.2f}")
    ax.set_xlabel("log₂ n   (grafos aleatórios, m = 3n)")
    ax.set_ylabel("Comparações por aresta")
    ax.set_title("Crescimento das comparações com n")
    ax.set_xticks(range(15, 22))
    ax.legend(fontsize=8)

    xs, m, lo, hi = serie(gs, lambda r: r["razao_comparacoes"])
    ax2.errorbar(xs, m, yerr=[m - lo, hi - m], fmt="o-", color="k", capsize=3, label="aleatórios (m = 3n)")
    xr, mr, lr, hr = serie(gr, lambda r: r["razao_comparacoes"])
    ax2.errorbar(xr, mr, yerr=[mr - lr, hr - mr], fmt="s", mfc="white", color="k", capsize=3,
                 label="malhas rodoviárias")
    for x0, y0, (nome, _) in zip(xr, mr, gr):
        ax2.annotate(nome_curto(nome), (x0, y0), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax2.axhline(1, ls=":", color="gray")
    ax2.set_xlabel("log₂ n")
    ax2.set_ylabel("Comparações BMSSP / comparações Dijkstra")
    ax2.set_title("Razão de comparações (< 1: BMSSP compara menos)")
    ax2.set_xticks(range(15, 25))
    ax2.legend(fontsize=8)
    salva(fig, "fig3_escada_comparacoes")


def fig_escada_tempo(sint, reais):
    gs, gr = grupos(sint), grupos(reais)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ns = np.array([rows[0]["n"] for _, rows in gs])
    for alg in ("bmssp", "dijkstra"):
        _, m, lo, hi = serie(gs, lambda r, a=alg: r[f"t_{a}_s"])
        ax.errorbar(ns, m, yerr=[m - lo, hi - m], fmt="o-", color=COR[alg], capsize=3, label=ROTULO[alg])
    ax.plot(ns, m[0] * ns / ns[0], ":", color="gray", lw=1, label="referência linear em n")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("n   (grafos aleatórios, m = 3n)")
    ax.set_ylabel("Tempo por fonte (s)")
    ax.set_title("Tempo absoluto")
    ax.legend(fontsize=8)

    xs, m, lo, hi = serie(gs, lambda r: r["razao"])
    ax2.errorbar(xs, m, yerr=[m - lo, hi - m], fmt="o-", color="k", capsize=3, label="aleatórios (m = 3n)")
    xr, mr, lr, hr = serie(gr, lambda r: r["razao"])
    ax2.errorbar(xr, mr, yerr=[mr - lr, hr - mr], fmt="s", mfc="white", color="k", capsize=3,
                 label="malhas rodoviárias")
    for x0, y0, (nome, _) in zip(xr, mr, gr):
        ax2.annotate(nome_curto(nome), (x0, y0), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax2.axhline(1, ls=":", color="gray")
    ax2.set_ylim(0, max(hr) * 1.1)
    ax2.set_xlabel("log₂ n")
    ax2.set_ylabel("Tempo BMSSP / tempo Dijkstra")
    ax2.set_title("Razão de tempo de parede (Python)")
    ax2.set_xticks(range(15, 25))
    ax2.legend(fontsize=8)
    salva(fig, "fig4_escada_tempo")


def fig_constantes(sint, reais):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    normas = {"bmssp": ("m · (log₂ n)^(2/3)", lambda r: r["bmssp_comparacoes_total"] / (r["m"] * r["log2n_23"])),
              "dijkstra": ("m · log₂ n", lambda r: r["dijkstra_comparacoes_total"] / (r["m"] * r["log2n"]))}
    for alg, (rot, fn) in normas.items():
        for rows, fmt, fam in ((sint, "o-", "aleatórios"), (reais, "s", "rodoviários")):
            xs, m, lo, hi = serie(grupos(rows), fn)
            ax.errorbar(xs, m, yerr=[m - lo, hi - m], fmt=fmt, color=COR[alg], capsize=3,
                        mfc=COR[alg] if fam == "aleatórios" else "white",
                        label=f"{ROTULO[alg].split()[0]} / [{rot}]  —  {fam}")
    ax.set_ylim(0, 4.8)
    ax.set_xlabel("log₂ n")
    ax.set_ylabel("Comparações / cota assintótica")
    ax.set_title("Constantes ocultas: comparações divididas pela cota teórica")
    ax.set_xticks(range(15, 25))
    ax.legend(fontsize=7.5, loc="upper center", ncol=2)
    salva(fig, "fig5_constantes")


def fig_niveis(reais):
    g = grupos(reais)
    fig, axes = plt.subplots(1, len(g), figsize=(2.6 * len(g) + 0.6, 3.4), sharey=True)
    for ax, (nome, rows) in zip(axes, g):
        topo = rows[0]["nivel_topo"]
        niveis = list(range(topo + 1))
        u = [max(r[f"nivel{l}_U_max"] / r[f"nivel{l}_U_cota"] for r in rows) for l in niveis]
        s = [max(r[f"nivel{l}_S_max"] / r[f"nivel{l}_S_cota"] for r in rows) for l in niveis]
        x = np.arange(len(niveis))
        ax.bar(x - 0.2, u, 0.4, color=COR["bmssp"], label="|U| máx / 4k·2^(lt)")
        ax.bar(x + 0.2, s, 0.4, color=COR["bmssp"], alpha=0.45, label="|S| máx / 2^(lt)")
        ax.axhline(1, ls="--", color="gray", lw=1)
        ax.set_xticks(x, [str(l) for l in niveis])
        ax.set_xlabel("nível l")
        ax.set_title(f"{nome_curto(nome)}  (k={rows[0]['k']}, t={rows[0]['t']})", fontsize=10)
    axes[0].set_ylabel("fração da cota (Lema 3.9)")
    axes[0].set_ylim(0, 1.15)
    axes[-1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Cotas do Lema 3.9 por nível de recursão (pior fonte de cada grafo)", fontsize=11)
    salva(fig, "fig6_niveis_lema39")


def fig_kt_fixo(reais, fixo):
    gf = grupos(fixo)
    ga = [(n, rows) for n, rows in grupos(reais) if n in dict(gf)]
    x = np.arange(len(ga))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for desloc, g, cor, rot in ((-0.2, ga, COR["bmssp"], "k, t do artigo"), (0.2, gf, "#7fb3d5", "k=2, t=6 fixos")):
        m, lo, hi = zip(*(med(rows, "razao") for _, rows in g))
        ax.bar(x + desloc, m, 0.4, yerr=[[a - b for a, b in zip(m, lo)], [b - a for a, b in zip(m, hi)]],
               capsize=3, color=cor, label=rot)
        cb = [med(rows, "bmssp_chamadas_base")[0] for _, rows in g]
        ax2.bar(x + desloc, cb, 0.4, color=cor, label=rot)
    for i, (nome, rows) in enumerate(ga):
        ax.text(i - 0.2, med(rows, "razao")[2] + 0.3, f"t={rows[0]['t']}", ha="center", fontsize=8)
    ax.set_xticks(x, [nome_curto(n) for n, _ in ga])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.set_ylabel("Tempo BMSSP / tempo Dijkstra")
    ax.set_title("Razão de tempo")
    ax.legend(fontsize=8)
    ax2.set_xticks(x, [nome_curto(n) for n, _ in ga])
    ax2.set_yscale("log")
    ax2.set_ylabel("Chamadas de BaseCase")
    ax2.set_title("Casos base executados")
    ax2.legend(fontsize=8)
    fig.suptitle("Controle de parâmetros: t do artigo (⌊log^(2/3) n⌋) vs t fixo", fontsize=11)
    salva(fig, "fig7_kt_fixo")


def fig_dispersao(reais):
    g = grupos(reais)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.4))
    rotulos = [f"{nome_curto(n)}\n({len(rows)} fontes)" for n, rows in g]
    ax.boxplot([[r["razao"] for r in rows] for _, rows in g], tick_labels=rotulos)
    ax.set_ylabel("Tempo BMSSP / tempo Dijkstra")
    ax.set_title("Razão de tempo por fonte")
    ax2.boxplot([[r["razao_comparacoes"] for r in rows] for _, rows in g], tick_labels=rotulos)
    ax2.set_ylabel("Comparações BMSSP / Dijkstra")
    ax2.set_title("Razão de comparações por fonte")
    salva(fig, "fig8_dispersao_fontes")


def main():
    os.makedirs(SAIDA, exist_ok=True)
    reais = le("resultados.csv")
    if os.path.exists("resultados_usa.csv"):
        reais += le("resultados_usa.csv")
    sint = le("resultados_sintetico.csv")
    fixo = le("resultados_kt_fixo.csv")
    print(f"figuras em {SAIDA}/")
    fig_tempo_reais(reais)
    fig_comparacoes_reais(reais)
    fig_escada_comparacoes(sint, reais)
    fig_escada_tempo(sint, reais)
    fig_constantes(sint, reais)
    fig_niveis(reais)
    fig_kt_fixo(reais, fixo)
    fig_dispersao(reais)


if __name__ == "__main__":
    main()
