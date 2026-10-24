#!/usr/bin/env python3
"""
Command line: run DuanSSSP (duan.py) and Dijkstra (dijkstra.py) on one graph, log the
recursion of BMSSP and check that both algorithms give the same distances.

Examples:
    python3 src/bmssp.py --example
    python3 src/bmssp.py -n 50 --seed 3 --log DEBUG
    python3 src/bmssp.py --dimacs datasets/test/example.gr --source 1
    python3 src/bmssp.py --dimacs datasets/USA-road-d.NY.gr.gz --source 1 --log WARNING
"""

import argparse
import logging
import sys
import time

from algorithms.common import TRACE, log, mismatches
from algorithms.dijkstra import dijkstra
from algorithms.duan import DuanSSSP
from graphs.dimacs import load_gr
from graphs.graph import Graph, example_graph, random_graph, to_constant_degree


def main():
    """Command line: run BMSSP and Dijkstra on one graph (fixed example, random or DIMACS),
    log the recursion and check that both give the same distances (exit code 1 if not)."""
    ap = argparse.ArgumentParser(description="BMSSP (Duan et al. 2025) vs Dijkstra, with logs")
    ap.add_argument("--example", action="store_true", help="use the small fixed graph")
    ap.add_argument("--dimacs", metavar="FILE",
                    help="load a DIMACS .gr/.gr.gz graph (ignores -n, -m, --seed)")
    ap.add_argument("--source", type=int, default=1,
                    help="source vertex of the DIMACS graph, 1-indexed as in the file")
    ap.add_argument("-n", type=int, default=50, help="vertices of the random graph")
    ap.add_argument("-m", type=int, default=None, help="arcs (default: 3n)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-weight", type=float, default=1)
    ap.add_argument("--max-weight", type=float, default=20)
    ap.add_argument("--real-weights", action="store_true", help="real weights instead of integers")
    ap.add_argument("--constant-degree", action="store_true",
                    help="apply the constant-degree transformation of Section 2")
    ap.add_argument("-k", type=int, default=None, help="force parameter k")
    ap.add_argument("-t", type=int, default=None, help="force parameter t")
    ap.add_argument("--log", default="INFO", help="TRACE, DEBUG, INFO, WARNING")
    ap.add_argument("--log-file", default=None, help="also write the log to this file")
    ap.add_argument("--ties", action="store_true", help="show the tie-break ⟨h,v⟩ next to the values")
    ap.add_argument("--preview", type=int, default=8, help="number of vertices shown per set")
    a = ap.parse_args()

    level = TRACE if a.log.upper() == "TRACE" else getattr(logging, a.log.upper())
    handlers = [logging.StreamHandler(sys.stdout)]
    if a.log_file:
        handlers.append(logging.FileHandler(a.log_file, mode="w", encoding="utf-8"))
    logging.basicConfig(level=level, format="%(message)s", handlers=handlers)

    if a.example:
        g, s = example_graph()
    elif a.dimacs:
        t0 = time.perf_counter()
        g = load_gr(a.dimacs, Graph)
        if not 1 <= a.source <= g.n:
            ap.error(f"--source must be in [1, {g.n}]")
        s = a.source - 1
        log.info("DIMACS %s: n=%d, m=%d, loaded in %.1fs", a.dimacs, g.n, g.m, time.perf_counter() - t0)
    else:
        wmin, wmax = (a.min_weight, a.max_weight) if a.real_weights else (int(a.min_weight), int(a.max_weight))
        g, s = random_graph(a.n, a.m or 3 * a.n, a.seed, wmin, wmax, a.real_weights)

    work, ws, rep = g, s, list(range(g.n))
    if a.constant_degree:
        work, ws, rep = to_constant_degree(g, s)
        log.info("Constant-degree transformation: n %d -> %d, m %d -> %d", g.n, work.n, g.m, work.m)

    t0 = time.perf_counter()
    solver = DuanSSSP(work, ws, k=a.k, t=a.t, show_ties=a.ties, preview=a.preview)
    dw = solver.run()
    t_bmssp = time.perf_counter() - t0
    d_bmssp = [dw[rep[v]] for v in range(g.n)]

    t0 = time.perf_counter()
    d_dij, st_dij = dijkstra(g, s)
    t_dij = time.perf_counter() - t0

    bad = mismatches(d_bmssp, d_dij)

    print("\n=== Result ===")
    if g.n <= 30:
        print(f"{'vertex':>8} {'BMSSP':>10} {'Dijkstra':>10}")
        for v in range(g.n):
            print(f"{g.name(v):>8} {d_bmssp[v]:>10g} {d_dij[v]:>10g}")
    print(f"BMSSP:    {t_bmssp * 1000:9.2f} ms  {solver.stats}")
    print(f"Dijkstra: {t_dij * 1000:9.2f} ms  {st_dij}")
    if bad:
        print(f"MISMATCH in {len(bad)} vertex(es), e.g.: "
              + ", ".join(f"{g.name(v)}: {d_bmssp[v]} vs {d_dij[v]}" for v in bad[:5]))
        sys.exit(1)
    print(f"OK: all {g.n} distances match Dijkstra.")


if __name__ == "__main__":
    main()
