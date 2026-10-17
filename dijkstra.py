"""
Reference Dijkstra (1959) for the comparison with DuanSSSP (duan.py).

Uses a binary heap with lazy deletion instead of DecreaseKey, which gives O(m log n) time.
With a Fibonacci heap [Fredman and Tarjan, 1987] the bound would be O(m + n log n), the
"sorting barrier" that Duan et al. (2025) break on sparse directed graphs; in practice the
binary heap is the usual choice and is what Python's heapq provides.
"""

from comum import INF, Chave, heap_ops


def dijkstra(g, s, contar=False):
    """Reference Dijkstra from s, with a binary heap and lazy deletion.

    Returns (d, stats), where d is the distance list and stats counts relaxations, heap
    operations, examined arcs, additions and comparisons in the comparison-addition model
    (one per examined arc plus one per pop). With contar=True the heap items are Chave and
    every `<` executed inside the heap is counted in comparacoes_heap.
    """
    d = [INF] * g.n
    d[s] = 0
    adj = g.adj
    relax = pops = arestas = 0
    pushes = 1
    n0 = Chave.n
    push, pop = heap_ops(contar)
    heap = []
    push(heap, (0, s))
    while heap:
        du, u = pop(heap)
        pops += 1
        if du > d[u]:
            continue
        adj_u = iter(adj[u])
        arestas += len(adj[u]) >> 1
        for v, w in zip(adj_u, adj_u):
            nd = du + w
            if nd < d[v]:
                d[v] = nd
                relax += 1
                push(heap, (nd, v))
                pushes += 1
    stats = dict(relaxacoes=relax, pops=pops, pushes=pushes, arestas_examinadas=arestas,
                 adicoes=arestas, comparacoes_explicitas=arestas + pops, comparacoes_heap=Chave.n - n0)
    stats["comparacoes_total"] = stats["comparacoes_explicitas"] + stats["comparacoes_heap"]
    return d, stats
