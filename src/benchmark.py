#!/usr/bin/env python3
"""Benchmark BMSSP (Duan et al. 2025) vs Dijkstra on DIMACS and synthetic graphs.

For each graph and each sampled source (fixed seed), both algorithms start from the
SAME source. Each algorithm runs twice per source:
  1. timed, with no extra instrumentation (the order alternates from source to source
     so that the second one does not benefit from a warm cache);
  2. counting, with count=True, which counts every comparison executed in the heap and
     in the structure D (comparison-addition model); it is not timed.
The distances are checked for every pair; loading the graph is not timed and the
garbage collector is disabled during each timed run.

Usage:
    python3 src/benchmark.py datasets/USA-road-d.NY.gr.gz datasets/USA-road-d.COL.gr.gz --sources 10
    python3 src/benchmark.py --synthetic 15 16 17 18 19 20 --degree 3 --sources 3 --output results/csv/synthetic.csv
    python3 src/benchmark.py datasets/USA-road-d.NY.gr.gz -k 2 -t 6      # fixed k and t
    python3 src/benchmark.py datasets/USA-road-d.NY.gr.gz --heap both    # heapq in C and pure heapq.py
    python3 src/benchmark.py datasets/test/example.gr --sources 3        # quick test on the test instance
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

from algorithms.common import mismatches, use_heap
from algorithms.dijkstra import dijkstra
from algorithms.duan import DuanSSSP
from graphs.dimacs import load_gr, random_sources
from graphs.graph import Graph, random_graph

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_LEVEL = 6
LEVEL_FIELDS = ("calls", "partials", "iterations", "S_max", "S_bound", "U_max", "U_bound", "U_mean")


def timed(fn):
    """Run fn() with the garbage collector disabled and return (result, seconds)."""
    gc.collect()
    gc.disable()
    try:
        t0 = time.perf_counter()
        r = fn()
        return r, time.perf_counter() - t0
    finally:
        gc.enable()


def graph_name(path):
    """Graph name taken from the file path, without the .gr / .gr.gz extension."""
    base = os.path.basename(path)
    for suf in (".gr.gz", ".gr"):
        if base.endswith(suf):
            return base[:-len(suf)]
    return base


def level_columns(levels):
    """Flatten DuanSSSP.levels into CSV columns level<l>_<field>, for l = 0..MAX_LEVEL;
    levels that do not exist in this graph are left empty."""
    out = {}
    for l in range(MAX_LEVEL + 1):
        lv = levels.get(l)
        for c in LEVEL_FIELDS:
            if lv is None:
                out[f"level{l}_{c}"] = ""
            elif c == "U_mean":
                out[f"level{l}_{c}"] = round(lv["U_total"] / lv["calls"], 2)
            else:
                out[f"level{l}_{c}"] = lv[c]
    return out


def load_graphs(a):
    """Yield (name, graph, load seconds) for each DIMACS file and each --synthetic exponent,
    one at a time, so that two large graphs are never in memory together."""
    for path in a.graphs:
        t0 = time.perf_counter()
        g = load_gr(path, Graph)
        yield graph_name(path), g, time.perf_counter() - t0
    for e in a.synthetic:
        n = 2 ** e
        m = a.degree * n
        t0 = time.perf_counter()
        g, _ = random_graph(n, m, a.graph_seed, 1, a.max_weight, False)
        yield f"rand-2^{e}-m{a.degree}n", g, time.perf_counter() - t0


def main():
    """Run the protocol described at the top of the file and write one CSV row per
    (graph, source, repetition, heap). Exit with code 1 if the distances disagree."""
    ap = argparse.ArgumentParser(description="BMSSP vs Dijkstra")
    ap.add_argument("graphs", nargs="*", help="DIMACS .gr or .gr.gz files")
    ap.add_argument("--synthetic", nargs="*", type=int, default=[], metavar="E",
                    help="random graphs with n = 2^E vertices and m = degree·n arcs")
    ap.add_argument("--degree", type=int, default=3, help="m/n of the synthetic graphs")
    ap.add_argument("--max-weight", type=int, default=10000,
                    help="integer weights in [1, max-weight] in the synthetic graphs")
    ap.add_argument("--graph-seed", type=int, default=7, help="seed of the synthetic graphs")
    ap.add_argument("--sources", type=int, default=10, help="sources sampled per graph")
    ap.add_argument("--seed", type=int, default=42, help="seed of the source sampling")
    ap.add_argument("--repetitions", type=int, default=1, help="timed runs per source")
    ap.add_argument("-k", type=int, default=None, help="fix parameter k on every graph")
    ap.add_argument("-t", type=int, default=None, help="fix parameter t on every graph")
    ap.add_argument("--heap", choices=("c", "python", "both"), default="c",
                    help="heapq in C (default), pure heapq.py or both, in Dijkstra and BaseCase")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--no-count", action="store_true", help="skip the counting run")
    ap.add_argument("--output", default=os.path.join(ROOT, "results", "csv", "results.csv"),
                    help="output CSV (default: results/csv/results.csv)")
    a = ap.parse_args()
    if not a.graphs and not a.synthetic:
        ap.error("give DIMACS files and/or --synthetic")
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    heaps = ("c", "python") if a.heap == "both" else (a.heap,)
    use_heap(heaps[0])

    with open(a.output, "w", newline="", encoding="utf-8") as f:
        writer = None
        for name, g, t_load in load_graphs(a):
            print(f"\n== {name}: n={g.n} m={g.m} (loaded in {t_load:.1f}s)")
            sources = random_sources(g.n, a.sources, a.seed)
            new = lambda s, count=False: DuanSSSP(g, s, k=a.k, t=a.t, count=count)

            if not a.no_warmup:
                dijkstra(g, sources[0])
                new(sources[0]).run()

            times = {h: ([], []) for h in heaps}
            for idx, s in enumerate(sources):
                counting = None
                for rep in range(a.repetitions):
                    for heap in heaps:
                        use_heap(heap)
                        def run_bmssp():
                            solver = new(s)
                            return solver, solver.run()
                        run_dij = lambda: dijkstra(g, s)
                        if (idx + rep) % 2 == 0:
                            (solver, d_bm), t_bm = timed(run_bmssp)
                            (d_dj, st_dj), t_dj = timed(run_dij)
                        else:
                            (d_dj, st_dj), t_dj = timed(run_dij)
                            (solver, d_bm), t_bm = timed(run_bmssp)
                        bad = mismatches(d_bm, d_dj)
                        if bad:
                            bad = bad[0]
                            print(f"MISMATCH in {name}, source {s + 1}: vertex {bad + 1} "
                                  f"BMSSP={d_bm[bad]} Dijkstra={d_dj[bad]}")
                            sys.exit(1)
                        if not a.no_count:
                            if counting is None:   # the counts do not depend on the heap: one per source
                                del solver, d_bm, d_dj   # frees 3 vectors of size n before the 2nd solver (USA: ~2 GB)
                                solver = new(s, count=True)
                                solver.run()
                                _, st_dj = dijkstra(g, s, count=True)
                                counting = solver, st_dj
                            solver, st_dj = counting
                        times[heap][0].append(t_bm)
                        times[heap][1].append(t_dj)

                        log2n = math.log2(g.n)
                        row = dict(graph=name, n=g.n, m=g.m, log2n=round(log2n, 3),
                                   log2n_23=round(log2n ** (2 / 3), 3),
                                   k=solver.k, t=solver.t, top_level=solver.top_level, heap=heap,
                                   source=s + 1, rep=rep, first="bmssp" if (idx + rep) % 2 == 0 else "dijkstra",
                                   t_bmssp_s=round(t_bm, 4), t_dijkstra_s=round(t_dj, 4),
                                   time_ratio=round(t_bm / t_dj, 3))
                        row.update({f"bmssp_{k}": v for k, v in solver.stats.items()})
                        row.update({f"dijkstra_{k}": v for k, v in st_dj.items()})
                        row["comparison_ratio"] = ("" if a.no_count else
                                                   round(solver.stats["total_comparisons"]
                                                         / max(1, st_dj["total_comparisons"]), 3))
                        row.update(level_columns(solver.levels))
                        if writer is None:
                            writer = csv.DictWriter(f, fieldnames=list(row))
                            writer.writeheader()
                        writer.writerow(row)
                        f.flush()
                        comp = "" if a.no_count else f"   comparisons {row['comparison_ratio']:5.2f}x"
                        print(f"  source {s + 1:>9} [{heap:>6}]  BMSSP {t_bm * 1000:9.1f} ms   "
                              f"Dijkstra {t_dj * 1000:8.1f} ms   ratio {t_bm / t_dj:5.2f}{comp}   ok")

            for heap in heaps:
                med_bm, med_dj = statistics.median(times[heap][0]), statistics.median(times[heap][1])
                print(f"  median [{heap:>6}]: BMSSP {med_bm * 1000:.1f} ms   Dijkstra {med_dj * 1000:.1f} ms   "
                      f"ratio {med_bm / med_dj:.2f}   (k={solver.k}, t={solver.t}, l={solver.top_level})")
            del g, solver
            gc.collect()

    print(f"\nResults in {a.output}")


if __name__ == "__main__":
    main()
