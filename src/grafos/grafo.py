"""
Graph representation and graph generators shared by the algorithms, the CLI (bmssp.py),
the DIMACS loader (dimacs.py) and the benchmark (benchmark.py).
"""

import random

class Graph:
    """Directed graph with non-negative weights, vertices 0..n-1.

    adj[u] is a flat interleaved sequence (v0, w0, v1, w1, ...). With shared int objects
    this costs 16 bytes per arc, against ~160 bytes for a list of tuples; it is what lets
    the USA road graph (58M arcs) fit in memory. Iterate it as
    it = iter(adj[u]); zip(it, it), or through arcs(u).
    names, if given, holds a label per vertex for the logs.
    """

    def __init__(self, n, names=None, adj=None):
        self.n = n
        self.adj = adj if adj is not None else [[] for _ in range(n)]
        self.m = sum(len(a) for a in adj) // 2 if adj is not None else 0
        self.names = names

    def add_edge(self, u, v, w):
        """Append the arc u -> v with weight w >= 0."""
        if w < 0:
            raise ValueError("pesos devem ser não negativos")
        self.adj[u] += (v, w)
        self.m += 1

    def arcs(self, u):
        """Iterate over the (v, w) pairs of the arcs leaving u."""
        it = iter(self.adj[u])
        return zip(it, it)

    def name(self, v):
        """Label of v for the logs: names[v] if available, else its index."""
        return self.names[v] if self.names else str(v)


def to_constant_degree(g, source):
    """Constant-degree transformation of Section 2 ("Constant-Degree Graph").

    Each vertex v becomes a zero-weight cycle with one node x_vw per incident arc, and each
    arc (u, v) becomes x_uv -> x_vu, so every node has in- and out-degree at most 2.
    Vertices with no incident arc keep a single node.

    Returns (G', s', rep), where rep[v] is a node of v's cycle (all nodes of a cycle have
    the same distance) and s' = rep[source].
    """
    slots = [[] for _ in range(g.n)]
    owner = []
    edges = []

    def new_node(v):
        owner.append(v)
        slots[v].append(len(owner) - 1)
        return len(owner) - 1

    for u in range(g.n):
        for v, w in g.arcs(u):
            edges.append((new_node(u), new_node(v), w))
    for v in range(g.n):
        if not slots[v]:
            new_node(v)

    names = [f"{g.name(owner[x])}#{slots[owner[x]].index(x)}" for x in range(len(owner))]
    g2 = Graph(len(owner), names)
    for v in range(g.n):
        cyc = slots[v]
        if len(cyc) > 1:
            for i, x in enumerate(cyc):
                g2.add_edge(x, cyc[(i + 1) % len(cyc)], 0)
    for a, b, w in edges:
        g2.add_edge(a, b, w)
    rep = [slots[v][0] for v in range(g.n)]
    return g2, rep[source], rep


def example_graph():
    """Small fixed graph with 11 named vertices, for following the logs by hand.
    Returns (g, s)."""
    names = list("sabcdefghij")
    g = Graph(len(names), names)
    idx = {c: i for i, c in enumerate(names)}
    for u, v, w in [("s", "a", 4), ("s", "b", 1), ("b", "a", 2), ("a", "c", 5),
                    ("b", "d", 8), ("c", "d", 1), ("d", "e", 3), ("c", "f", 6),
                    ("e", "f", 1), ("f", "g", 2), ("e", "h", 7), ("g", "h", 1),
                    ("h", "i", 2), ("g", "j", 9), ("i", "j", 1), ("j", "s", 3),
                    ("a", "b", 0), ("d", "c", 2)]:
        g.add_edge(idx[u], idx[v], w)
    return g, idx["s"]


def random_graph(n, m, seed, wmin, wmax, real):
    """Random graph with n vertices and m arcs, reproducible by seed. Returns (g, 0).

    It starts from a random spanning tree with arcs in both directions, so every vertex
    reaches every other, then adds uniformly random arcs until there are m of them.
    Weights are uniform in [wmin, wmax]: reals if real is True, integers otherwise.
    """
    rng = random.Random(seed)
    g = Graph(n)
    order = list(range(1, n))
    rng.shuffle(order)
    seen = [0]
    draw = (lambda: rng.uniform(wmin, wmax)) if real else (lambda: rng.randint(wmin, wmax))
    for v in order:
        p, w = rng.choice(seen), draw()
        g.add_edge(p, v, w)
        g.add_edge(v, p, w)
        seen.append(v)
    while g.m < m:
        g.add_edge(rng.randrange(n), rng.randrange(n), draw())
    return g, 0
