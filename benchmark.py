#!/usr/bin/env python3
"""Benchmark BMSSP (Duan et al. 2025) vs Dijkstra em grafos DIMACS e sintéticos.

Para cada grafo e cada fonte sorteada (seed fixa), os dois algoritmos partem da
MESMA fonte. Por fonte há duas execuções de cada algoritmo:
  1. cronometrada, sem instrumentação extra (ordem alternada a cada fonte para
     não favorecer o segundo com o cache aquecido);
  2. de contagem, com contar=True, que estima as comparações internas de heap,
     bisect, sort e min no modelo comparação-adição; não entra no tempo.
As distâncias são conferidas a cada par; a carga do grafo fica fora da medição
e o garbage collector fica desligado durante cada execução cronometrada.

Uso:
    python3 benchmark.py ../data/USA-road-d.NY.gr.gz ../data/USA-road-d.COL.gr.gz --fontes 10
    python3 benchmark.py --sintetico 15 16 17 18 19 20 --grau 3 --fontes 3 --saida sintetico.csv
    python3 benchmark.py ../data/USA-road-d.NY.gr.gz -k 2 -t 6      # k e t fixos
"""

import argparse
import csv
import gc
import logging
import math
import os
import statistics
import sys
import time

from bmssp import DuanSSSP, Graph, dijkstra, random_graph
from dimacs import load_gr, random_sources

MAX_NIVEL = 6
CAMPOS_NIVEL = ("chamadas", "parciais", "iteracoes", "S_max", "S_cota", "U_max", "U_cota", "U_medio")


def cronometra(fn):
    gc.collect()
    gc.disable()
    try:
        t0 = time.perf_counter()
        r = fn()
        return r, time.perf_counter() - t0
    finally:
        gc.enable()


def iguais(a, b):
    if a == b:
        return True
    return all(x == y or math.isclose(x, y, rel_tol=1e-9) for x, y in zip(a, b))


def nome_grafo(path):
    base = os.path.basename(path)
    for suf in (".gr.gz", ".gr"):
        if base.endswith(suf):
            return base[:-len(suf)]
    return base


def colunas_nivel(niveis):
    out = {}
    for l in range(MAX_NIVEL + 1):
        lv = niveis.get(l)
        for c in CAMPOS_NIVEL:
            if lv is None:
                out[f"nivel{l}_{c}"] = ""
            elif c == "U_medio":
                out[f"nivel{l}_{c}"] = round(lv["U_total"] / lv["chamadas"], 2)
            else:
                out[f"nivel{l}_{c}"] = lv[c]
    return out


def carrega_grafos(a):
    for path in a.grafos:
        t0 = time.perf_counter()
        g = load_gr(path, Graph)
        yield nome_grafo(path), g, time.perf_counter() - t0
    for e in a.sintetico:
        n = 2 ** e
        m = a.grau * n
        t0 = time.perf_counter()
        g, _ = random_graph(n, m, a.seed_grafo, 1, a.peso_max, False)
        yield f"rand-2^{e}-m{a.grau}n", g, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description="BMSSP vs Dijkstra")
    ap.add_argument("grafos", nargs="*", help="arquivos DIMACS .gr ou .gr.gz")
    ap.add_argument("--sintetico", nargs="*", type=int, default=[], metavar="E",
                    help="grafos aleatórios com n = 2^E vértices e m = grau·n arcos")
    ap.add_argument("--grau", type=int, default=3, help="m/n dos grafos sintéticos")
    ap.add_argument("--peso-max", type=int, default=10000, help="pesos inteiros em [1, peso-max] nos sintéticos")
    ap.add_argument("--seed-grafo", type=int, default=7, help="seed dos grafos sintéticos")
    ap.add_argument("--fontes", type=int, default=10, help="fontes sorteadas por grafo")
    ap.add_argument("--seed", type=int, default=42, help="seed do sorteio das fontes")
    ap.add_argument("--repeticoes", type=int, default=1, help="execuções cronometradas por fonte")
    ap.add_argument("-k", type=int, default=None, help="fixa o parâmetro k em todos os grafos")
    ap.add_argument("-t", type=int, default=None, help="fixa o parâmetro t em todos os grafos")
    ap.add_argument("--sem-aquecimento", action="store_true")
    ap.add_argument("--sem-contagem", action="store_true", help="pula a execução de contagem")
    ap.add_argument("--saida", default="resultados.csv")
    a = ap.parse_args()
    if not a.grafos and not a.sintetico:
        ap.error("informe arquivos DIMACS e/ou --sintetico")
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    with open(a.saida, "w", newline="", encoding="utf-8") as f:
        writer = None
        for nome, g, t_carga in carrega_grafos(a):
            print(f"\n== {nome}: n={g.n} m={g.m} (carregado em {t_carga:.1f}s)")
            fontes = random_sources(g.n, a.fontes, a.seed)
            novo = lambda s, contar=False: DuanSSSP(g, s, k=a.k, t=a.t, contar=contar)

            if not a.sem_aquecimento:
                dijkstra(g, fontes[0])
                novo(fontes[0]).run()

            tempos_bm, tempos_dj = [], []
            for idx, s in enumerate(fontes):
                for rep in range(a.repeticoes):
                    def roda_bmssp():
                        solver = novo(s)
                        return solver, solver.run()
                    roda_dij = lambda: dijkstra(g, s)
                    if (idx + rep) % 2 == 0:
                        (solver, d_bm), t_bm = cronometra(roda_bmssp)
                        (d_dj, st_dj), t_dj = cronometra(roda_dij)
                    else:
                        (d_dj, st_dj), t_dj = cronometra(roda_dij)
                        (solver, d_bm), t_bm = cronometra(roda_bmssp)
                    if not iguais(d_bm, d_dj):
                        bad = next(v for v in range(g.n) if d_bm[v] != d_dj[v])
                        print(f"DIVERGÊNCIA em {nome}, fonte {s + 1}: vértice {bad + 1} "
                              f"BMSSP={d_bm[bad]} Dijkstra={d_dj[bad]}")
                        sys.exit(1)
                    if not a.sem_contagem and rep == 0:
                        del solver, d_bm, d_dj   # libera 3 vetores de n antes do 2º solver (USA: ~2 GB)
                        solver = novo(s, contar=True)
                        solver.run()
                        _, st_dj = dijkstra(g, s, contar=True)
                    tempos_bm.append(t_bm)
                    tempos_dj.append(t_dj)

                    log2n = math.log2(g.n)
                    row = dict(grafo=nome, n=g.n, m=g.m, log2n=round(log2n, 3),
                               log2n_23=round(log2n ** (2 / 3), 3),
                               k=solver.k, t=solver.t, nivel_topo=solver.top_level,
                               fonte=s + 1, rep=rep, primeiro="bmssp" if (idx + rep) % 2 == 0 else "dijkstra",
                               t_bmssp_s=round(t_bm, 4), t_dijkstra_s=round(t_dj, 4),
                               razao=round(t_bm / t_dj, 3))
                    row.update({f"bmssp_{k}": v for k, v in solver.stats.items()})
                    row.update({f"dijkstra_{k}": v for k, v in st_dj.items()})
                    row["razao_comparacoes"] = ("" if a.sem_contagem else
                                                round(solver.stats["comparacoes_total"]
                                                      / max(1, st_dj["comparacoes_total"]), 3))
                    row.update(colunas_nivel(solver.niveis))
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    f.flush()
                    comp = "" if a.sem_contagem else f"   comparações {row['razao_comparacoes']:5.2f}x"
                    print(f"  fonte {s + 1:>9}  BMSSP {t_bm * 1000:9.1f} ms   Dijkstra {t_dj * 1000:8.1f} ms   "
                          f"razão {t_bm / t_dj:5.2f}{comp}   ok")

            med_bm, med_dj = statistics.median(tempos_bm), statistics.median(tempos_dj)
            print(f"  mediana: BMSSP {med_bm * 1000:.1f} ms   Dijkstra {med_dj * 1000:.1f} ms   "
                  f"razão {med_bm / med_dj:.2f}   (k={solver.k}, t={solver.t}, l={solver.top_level})")
            del g, solver
            gc.collect()

    print(f"\nResultados em {a.saida}")


if __name__ == "__main__":
    main()
