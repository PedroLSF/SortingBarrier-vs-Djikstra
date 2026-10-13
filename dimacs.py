#!/usr/bin/env python3
"""Leitor e validador do formato DIMACS (9th Challenge) para grafos .gr / .gr.gz.

Formato: linhas `c` são comentários, `p sp <n> <m>` é o cabeçalho e cada
`a <u> <v> <w>` é um arco dirigido com vértices 1-indexados e peso inteiro.
O grafo é devolvido 0-indexado.

Uso como validador:
    python3 dimacs.py --check data/USA-road-d.NY.gr.gz [outros arquivos...]
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
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")


def read_header(path):
    with open_text(path) as f:
        for line in f:
            if line[0] == "p":
                _, _, n, m = line.split()
                return int(n), int(m)
    raise ValueError(f"{path}: sem linha 'p sp n m'")


def load_gr(path, graph_cls):
    """Lê os arcos em arrays, ordena por origem (contagem) e monta adj[u] como tupla
    intercalada (v0, w0, v1, w1, ...) com objetos int compartilhados. Sem listas
    por vértice no caminho: o pico de memória fica em ~1,6x o tamanho final."""
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
        raise ValueError(f"{path}: sem linha 'p sp n m'")
    if len(us) != m:
        raise ValueError(f"{path}: cabeçalho declara {m} arcos, arquivo tem {len(us)}")
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
    """Fontes reprodutíveis (0-indexadas) para o benchmark."""
    return random.Random(seed).sample(range(n), count)


def sha256(path, bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def check(path):
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
                    raise ValueError(f"{path}: arco ({u},{v}) fora de [1,{n}]")
            elif c == "p":
                _, _, n, m = line.split()
                n, m = int(n), int(m)
                outdeg = array("I", bytes(4 * n))
    maxdeg = max(outdeg)
    isolated = sum(1 for d in outdeg if d == 0)
    ok = arcs == m and neg == 0
    return dict(arquivo=path, n=n, m=m, arcos_lidos=arcs, ok=ok, pesos_negativos=neg,
                peso_min=wmin, peso_max=wmax, self_loops=loops, grau_saida_max=maxdeg,
                sem_saida=isolated, segundos=round(time.perf_counter() - t0, 1))


def main():
    ap = argparse.ArgumentParser(description="valida arquivos DIMACS .gr/.gr.gz")
    ap.add_argument("--check", nargs="+", metavar="ARQUIVO", required=True)
    ap.add_argument("--sha256", action="store_true", help="também calcula o sha256")
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
