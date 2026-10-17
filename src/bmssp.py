#!/usr/bin/env python3
"""
Command line: run DuanSSSP (duan.py) and Dijkstra (dijkstra.py) on one graph, log the
recursion of BMSSP and check that both algorithms give the same distances.

Examples:
    python3 src/bmssp.py --exemplo
    python3 src/bmssp.py -n 50 --seed 3 --log DEBUG
    python3 src/bmssp.py --dimacs datasets/USA-road-d.NY.gr.gz --fonte 1 --log WARNING
"""

import argparse
import logging
import math
import sys
import time

from algoritmos.comum import TRACE, log
from algoritmos.dijkstra import dijkstra
from algoritmos.duan import DuanSSSP
from grafos.dimacs import load_gr
from grafos.grafo import Graph, example_graph, random_graph, to_constant_degree

def main():
    """Command line: run BMSSP and Dijkstra on one graph (fixed example, random or DIMACS),
    log the recursion and check that both give the same distances (exit code 1 if not)."""
    ap = argparse.ArgumentParser(description="BMSSP (Duan et al. 2025) vs Dijkstra, com logs")
    ap.add_argument("--exemplo", action="store_true", help="usa o grafo pequeno fixo")
    ap.add_argument("--dimacs", metavar="ARQUIVO",
                    help="carrega um grafo DIMACS .gr/.gr.gz (ignora -n, -m, --seed)")
    ap.add_argument("--fonte", type=int, default=1,
                    help="vértice fonte do grafo DIMACS, 1-indexado como no arquivo")
    ap.add_argument("-n", type=int, default=50, help="vértices do grafo aleatório")
    ap.add_argument("-m", type=int, default=None, help="arestas (padrão: 3n)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--peso-min", type=float, default=1)
    ap.add_argument("--peso-max", type=float, default=20)
    ap.add_argument("--pesos-reais", action="store_true", help="pesos reais em vez de inteiros")
    ap.add_argument("--grau-constante", action="store_true",
                    help="aplica a transformação de grau constante da Seção 2")
    ap.add_argument("-k", type=int, default=None, help="força o parâmetro k")
    ap.add_argument("-t", type=int, default=None, help="força o parâmetro t")
    ap.add_argument("--log", default="INFO", help="TRACE, DEBUG, INFO, WARNING")
    ap.add_argument("--log-arquivo", default=None, help="grava o log também neste arquivo")
    ap.add_argument("--empates", action="store_true", help="mostra o desempate ⟨h,v⟩ nos valores")
    ap.add_argument("--preview", type=int, default=8, help="nº de vértices exibidos por conjunto")
    a = ap.parse_args()

    level = TRACE if a.log.upper() == "TRACE" else getattr(logging, a.log.upper())
    handlers = [logging.StreamHandler(sys.stdout)]
    if a.log_arquivo:
        handlers.append(logging.FileHandler(a.log_arquivo, mode="w", encoding="utf-8"))
    logging.basicConfig(level=level, format="%(message)s", handlers=handlers)

    if a.exemplo:
        g, s = example_graph()
    elif a.dimacs:
        t0 = time.perf_counter()
        g = load_gr(a.dimacs, Graph)
        if not 1 <= a.fonte <= g.n:
            ap.error(f"--fonte deve estar em [1, {g.n}]")
        s = a.fonte - 1
        log.info("DIMACS %s: n=%d, m=%d, carregado em %.1fs", a.dimacs, g.n, g.m, time.perf_counter() - t0)
    else:
        wmin, wmax = (a.peso_min, a.peso_max) if a.pesos_reais else (int(a.peso_min), int(a.peso_max))
        g, s = random_graph(a.n, a.m or 3 * a.n, a.seed, wmin, wmax, a.pesos_reais)

    work, ws, rep = g, s, list(range(g.n))
    if a.grau_constante:
        work, ws, rep = to_constant_degree(g, s)
        log.info("Transformação de grau constante: n %d -> %d, m %d -> %d", g.n, work.n, g.m, work.m)

    t0 = time.perf_counter()
    solver = DuanSSSP(work, ws, k=a.k, t=a.t, show_ties=a.empates, preview=a.preview)
    dw = solver.run()
    t_bmssp = time.perf_counter() - t0
    d_bmssp = [dw[rep[v]] for v in range(g.n)]

    t0 = time.perf_counter()
    d_dij, st_dij = dijkstra(g, s)
    t_dij = time.perf_counter() - t0

    bad = [v for v in range(g.n)
           if not (d_bmssp[v] == d_dij[v] or math.isclose(d_bmssp[v], d_dij[v], rel_tol=1e-9))]

    print("\n=== Resultado ===")
    if g.n <= 30:
        print(f"{'vértice':>8} {'BMSSP':>10} {'Dijkstra':>10}")
        for v in range(g.n):
            print(f"{g.name(v):>8} {d_bmssp[v]:>10g} {d_dij[v]:>10g}")
    print(f"BMSSP:    {t_bmssp * 1000:9.2f} ms  {solver.stats}")
    print(f"Dijkstra: {t_dij * 1000:9.2f} ms  {st_dij}")
    if bad:
        print(f"DIVERGÊNCIA em {len(bad)} vértice(s), ex.: "
              + ", ".join(f"{g.name(v)}: {d_bmssp[v]} vs {d_dij[v]}" for v in bad[:5]))
        sys.exit(1)
    print(f"OK: as {g.n} distâncias coincidem com Dijkstra.")


if __name__ == "__main__":
    main()
