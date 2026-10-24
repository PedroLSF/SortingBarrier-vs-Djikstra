"""
Reference Dijkstra (1959) for the comparison with DuanSSSP (duan.py).

Uses a binary heap with lazy deletion instead of DecreaseKey, which gives O(m log n) time.
With a Fibonacci heap [Fredman and Tarjan, 1987] the bound would be O(m + n log n), the
"sorting barrier" that Duan et al. (2025) break on sparse directed graphs; in practice the
binary heap is the usual choice and is what Python's heapq provides.
"""

from .common import INF, CountedKey, heap_ops


def dijkstra(g, s, count=False):
    """Reference Dijkstra from s, with a binary heap and lazy deletion.

    Returns (d, stats), where d is the distance list and stats counts relaxations, heap
    operations, examined arcs, additions and comparisons in the comparison-addition model
    (one per examined arc plus one per pop). With count=True the heap items are CountedKey
    and every `<` executed inside the heap is counted in heap_comparisons.
    """
    d = [INF] * g.n
    d[s] = 0
    adj = g.adj
    relax = pops = arcs = 0
    pushes = 1
    n0 = CountedKey.n
    push, pop = heap_ops(count)
    heap = []
    push(heap, (0, s))
    while heap:
        du, u = pop(heap)
        pops += 1
        if du > d[u]:
            continue
        adj_u = iter(adj[u])
        arcs += len(adj[u]) >> 1
        for v, w in zip(adj_u, adj_u):
            nd = du + w
            if nd < d[v]:
                d[v] = nd
                relax += 1
                push(heap, (nd, v))
                pushes += 1
    stats = dict(relaxations=relax, pops=pops, pushes=pushes, arcs_examined=arcs,
                 additions=arcs, explicit_comparisons=arcs + pops, heap_comparisons=CountedKey.n - n0)
    stats["total_comparisons"] = stats["explicit_comparisons"] + stats["heap_comparisons"]
    return d, stats
