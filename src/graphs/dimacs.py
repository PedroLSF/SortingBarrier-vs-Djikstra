#!/usr/bin/env python3
"""Reader and validator of the DIMACS format (9th Challenge) for .gr / .gr.gz graphs.

Format: `c` lines are comments, `p sp <n> <m>` is the header and each
`a <u> <v> <w>` is a directed arc with 1-indexed vertices and an integer weight.
The graph is returned 0-indexed.

Usage as a validator:
    python3 src/graphs/dimacs.py --check datasets/USA-road-d.NY.gr.gz [more files...]
"""

import argparse
import gzip
import hashlib
import random
import sys
import time
from array import array
from itertools import accumulate


def open_text(path):
    """Open the file in text mode, decompressing it if it ends in .gz."""
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")


def load_gr(path, graph_cls):
    """Read the arcs into arrays, sort them by tail (counting sort) and build adj[u] as an
    interleaved tuple (v0, w0, v1, w1, ...) with shared int objects. No per-vertex lists
    along the way: peak memory stays at ~1.6x the final size."""
    n = m = None
    us, vs, ws = array("i"), array("i"), array("q")
    with open_text(path) as f:
        for line in f:
            c = line[0]
            if c == "a":
                _, u, v, w = line.split()
                us.append(int(u) - 1)
                vs.append(int(v) - 1)
                ws.append(int(w))
            elif c == "p":
                _, _, n, m = line.split()
                n, m = int(n), int(m)
    if n is None:
        raise ValueError(f"{path}: no 'p sp n m' line")
    if len(us) != m:
        raise ValueError(f"{path}: header declares {m} arcs, file has {len(us)}")
    deg = array("i", bytes(4 * (n + 1)))
    for u in us:
        deg[u + 1] += 1
    off = array("i", accumulate(deg))
    pos = array("i", off)
    tgt, wt = array("i", bytes(4 * m)), array("q", bytes(8 * m))
    for u, v, w in zip(us, vs, ws):
        p = pos[u]
        tgt[p] = v
        wt[p] = w
        pos[u] = p + 1
    del us, vs, ws, deg, pos
    ids = list(range(n))
    wobj = {w: w for w in set(wt)}
    inter = [0] * (2 * m)
    inter[0::2] = map(ids.__getitem__, tgt)
    inter[1::2] = map(wobj.__getitem__, wt)
    del tgt, wt
    adj = [tuple(inter[2 * off[u]:2 * off[u + 1]]) for u in range(n)]
    return graph_cls(n, adj=adj)


def random_sources(n, count, seed):
    """Reproducible sources (0-indexed) for the benchmark."""
    return random.Random(seed).sample(range(n), count)


def sha256(path, bufsize=1 << 20):
    """SHA-256 hash of the file, read in chunks, to check the download."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def check(path):
    """Validate a DIMACS file in a single pass, without building the graph: check the number
    of arcs against the header, look for negative weights and collect statistics (min/max
    weight, self-loops, out-degree). Return a dict; ok=True if the file is usable."""
    t0 = time.perf_counter()
    n = m = None
    arcs = neg = loops = 0
    wmin, wmax = None, None
    outdeg = None
    with open_text(path) as f:
        for line in f:
            c = line[0]
            if c == "a":
                _, u, v, w = line.split()
                u, v, w = int(u), int(v), int(w)
                arcs += 1
                outdeg[u - 1] += 1
                if w < 0:
                    neg += 1
                if u == v:
                    loops += 1
                if wmin is None or w < wmin:
                    wmin = w
                if wmax is None or w > wmax:
                    wmax = w
                if not (1 <= u <= n and 1 <= v <= n):
                    raise ValueError(f"{path}: arc ({u},{v}) outside [1,{n}]")
            elif c == "p":
                _, _, n, m = line.split()
                n, m = int(n), int(m)
                outdeg = array("I", bytes(4 * n))
    maxdeg = max(outdeg)
    isolated = sum(1 for d in outdeg if d == 0)
    ok = arcs == m and neg == 0
    return dict(file=path, n=n, m=m, arcs_read=arcs, ok=ok, negative_weights=neg,
                min_weight=wmin, max_weight=wmax, self_loops=loops, max_out_degree=maxdeg,
                no_out_arcs=isolated, seconds=round(time.perf_counter() - t0, 1))


def main():
    """Command line of the validator (--check); exit with code 1 if any file fails."""
    ap = argparse.ArgumentParser(description="validate DIMACS .gr/.gr.gz files")
    ap.add_argument("--check", nargs="+", metavar="FILE", required=True)
    ap.add_argument("--sha256", action="store_true", help="also compute the sha256")
    a = ap.parse_args()
    bad = 0
    for p in a.check:
        r = check(p)
        if a.sha256:
            r["sha256"] = sha256(p)
        print(" ".join(f"{k}={v}" for k, v in r.items()))
        bad += not r["ok"]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
