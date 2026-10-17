"""
Didactic, fully logged implementation of

    R. Duan, J. Mao, X. Mao, X. Shu, L. Yin.
    "Breaking the Sorting Barrier for Directed Single-Source Shortest Paths".
    arXiv:2504.17033v2, Jul 2025.  https://arxiv.org/abs/2504.17033

The paper gives a deterministic SSSP algorithm for directed graphs with non-negative real
weights that runs in O(m log^(2/3) n) time in the comparison-addition model.

Comparison with the literature:
    * Dijkstra (1959) with a Fibonacci heap [Fredman and Tarjan, 1987]: O(m + n log n).
      Because it settles vertices in sorted order it cannot beat sorting, the "sorting
      barrier"; with the binary heap used in dijkstra.py it is O(m log n).
    * Bellman-Ford: O(mn), but needs no ordering at all.
    * Thorup (1999): O(m) for undirected graphs with integer weights, outside the
      comparison-addition model.
    * Duan, Mao, Shu and Yin (2023): randomized O(m sqrt(log n · log log n)) for undirected
      real-weighted graphs.
    * This paper: first deterministic algorithm to break O(m + n log n) for directed graphs
      on sparse inputs (m = O(n)), by mixing Dijkstra with k rounds of Bellman-Ford
      (FindPivots) inside a recursive divide-and-conquer on distance ranges (BMSSP).

Code <-> paper map:
    DuanSSSP.find_pivots  -> Algorithm 1, FindPivots (Lemma 3.2)
    DuanSSSP.base_case    -> Algorithm 2, BaseCase
    DuanSSSP.bmssp        -> Algorithm 3, BMSSP (Lemma 3.1)
    estrutura_d.BlockDS   -> data structure of Lemma 3.3 (Insert / BatchPrepend / Pull)
    (d̂[v], h[v], v)       -> Assumption 2.1, total order on paths, built inline by the algorithms
    grafo.to_constant_degree -> constant-degree transformation (Section 2)

Notes:
    * The speed-up is asymptotic and holds in the comparison-addition model. In wall-clock
      time, in Python, Dijkstra with heapq is faster on every graph tested (see README).
    * comum.usar_heap("python") swaps the C-accelerated heapq for the pure-Python heapq.py,
      in both Dijkstra and BaseCase, to measure how much of that gap is due to the interpreter.
    * The structure D finds the k smallest items by quickselect (expected linear time),
      where the paper cites the deterministic selection of [BFP+73].
    * Logs are for following the execution; measurements live in benchmark.py.
"""

import logging
import math

from .comum import INF, KEY_INF, Chave, heap_ops, log
from .estrutura_d import BlockDS

class DuanSSSP:
    """Single-source shortest paths by Duan et al. (2025), with logs and operation counts.

    Parameters (Section 3.1): k = ⌊log^(1/3) n⌋ and t = ⌊log^(2/3) n⌋, unless forced.
    The top level is l = ⌈log n / t⌉, raised until k·2^(lt) > n so that the top call to
    BMSSP always completes successfully.

    State per vertex v: d[v] is d̂[v], h[v] is the number of arcs of the current path
    (tie-break of Assumption 2.1), pred[v] is Pred[v] and pred_w[v] the weight of the arc
    (Pred[v], v).

    Total order (Assumption 2.1), built inline by the three algorithms:
        * the value of d̂[v] is the tuple (d[v], h[v], v);
        * the relaxation test "d̂[u] + w_uv <= d̂[v]" (Remark 3.4) compares the candidate
          path ⟨d̂[u]+w, h[u]+1, v, u, ...⟩ with the current one ⟨d̂[v], h[v], v, Pred[v], ...⟩,
          that is, (d[u]+w, h[u]+1, u) <= (d[v], h[v], pred[v]); u == Pred[v] counts as
          equality, so re-relaxing the same arc is valid (counted in re_relaxacoes_iguais).

    Options:
        show_ties: logs show the tie-break ⟨h, v⟩ next to each distance;
        preview:   number of vertices printed per set in the logs;
        contar:    also count the comparisons made inside the heap and the structure D
                   (exact, via Chave). It slows the run down, so benchmark.py collects it
                   in a run separate from the timed one.

    After run(), `stats` holds the counters (see run()) and `niveis` maps each level l to
    the statistics of the BMSSP calls at that level, together with the bounds of Lemma 3.9
    on |S| (S_cota = 2^(lt)) and |U| (U_cota = 4k·2^(lt)).
    """

    def __init__(self, g, source, k=None, t=None, show_ties=False, preview=8, contar=False):
        self.g = g
        self.s = source
        n = g.n
        logn = math.log2(max(n, 2))
        self.k = k if k else max(1, math.floor(logn ** (1 / 3)))
        self.t = t if t else max(1, math.floor(logn ** (2 / 3)))
        self.top_level = max(1, math.ceil(logn / self.t))
        while self.k * 2 ** (self.top_level * self.t) <= n:
            self.top_level += 1

        self.d = [INF] * n
        self.h = [INF] * n
        self.pred = [-1] * n
        self.pred_w = [0] * n
        self.d[source] = 0
        self.h[source] = 0

        self.show_ties = show_ties
        self.preview = preview
        self.info = log.isEnabledFor(logging.INFO)
        self.debug = log.isEnabledFor(logging.DEBUG)
        self.contar = contar
        self.stats = dict(relaxacoes=0, re_relaxacoes_iguais=0, arestas_examinadas=0,
                          chamadas_bmssp=0, chamadas_base=0, chamadas_findpivots=0,
                          findpivots_desistencias=0, findpivots_S_total=0,
                          findpivots_W_total=0, findpivots_P_total=0,
                          pulls=0, inserts=0, prepends=0, heap_pushes=0, heap_pops=0,
                          comparacoes_cotas=0, adicoes=0, comparacoes_explicitas=0,
                          comparacoes_heap=0, comparacoes_D=0, comparacoes_total=0)
        self.niveis = {}

    def fmt(self, key):
        """Format a value (d, h, v) of the total order for the logs."""
        if key[0] == INF:
            return "∞"
        if self.show_ties:
            return f"{key[0]:g}⟨h={key[1]},{self.g.name(key[2])}⟩"
        return f"{key[0]:g}"

    def fmt_set(self, S):
        """Format a vertex set for the logs, sorted by d̂ and truncated to `preview` items."""
        d, h = self.d, self.h
        names = sorted(S, key=lambda v: (d[v], h[v], v))
        shown = ", ".join(f"{self.g.name(v)}:{self.fmt((d[v], h[v], v))}" for v in names[:self.preview])
        extra = f", … (+{len(names) - self.preview})" if len(names) > self.preview else ""
        return "{" + shown + extra + "}"

    def find_pivots(self, B, S, depth):
        """Algorithm 1, FindPivots(B, S) (Lemma 3.2).

        Relaxes the arcs out of S for k rounds, collecting in W every vertex reached with
        d̂ < B. If |W| exceeds k|S| it gives up early and returns P = S. Otherwise it builds
        the forest F = {(u, v) : u, v ∈ W, d̂[v] = d̂[u] + w_uv}, which under Assumption 2.1
        is the Pred[] forest restricted to W, and returns as pivots P the roots in S whose
        tree has at least k vertices.

        Returns (P, W). Counted comparisons: one bound test d̂[v] < B per relaxation, and one
        addition plus one comparison per v ∈ W for the tight-arc test.
        """
        st = self.stats
        st["chamadas_findpivots"] += 1
        st["findpivots_S_total"] += len(S)
        ind = "│   " * depth
        k = self.k
        adj = self.g.adj
        d, h, pred, pred_w = self.d, self.h, self.pred, self.pred_w
        if self.info:
            log.info("%s  FindPivots(B=%s, |S|=%d)", ind, self.fmt(B), len(S))
        W = set(S)
        prev = set(S)
        arestas = 0
        relax0 = st["relaxacoes"] + st["re_relaxacoes_iguais"]
        for i in range(1, k + 1):
            Wi = set()
            for u in prev:
                adj_u = iter(adj[u])
                arestas += len(adj[u]) >> 1
                for v, w in zip(adj_u, adj_u):
                    du, hu = d[u] + w, h[u] + 1
                    if (du, hu, u) <= (d[v], h[v], pred[v]):
                        same = pred[v] == u and d[v] == du and h[v] == hu
                        d[v], h[v], pred[v], pred_w[v] = du, hu, u, w
                        st["re_relaxacoes_iguais" if same else "relaxacoes"] += 1
                        if self.debug:
                            log.debug("%s    relaxa %s -> %s (w=%g): d̂[%s] = %s%s", ind,
                                      self.g.name(u), self.g.name(v), w, self.g.name(v),
                                      self.fmt((du, hu, v)),
                                      "  (igual: re-relaxação, Remark 3.4)" if same else "")
                        if (du, hu, v) < B:
                            Wi.add(v)
            W |= Wi
            if self.debug:
                log.debug("%s    passo %d/%d: |W_%d| = %d, |W| = %d", ind, i, k, i, len(Wi), len(W))
            if len(W) > k * len(S):
                st["arestas_examinadas"] += arestas
                st["comparacoes_cotas"] += st["relaxacoes"] + st["re_relaxacoes_iguais"] - relax0
                st["findpivots_desistencias"] += 1
                st["findpivots_W_total"] += len(W)
                st["findpivots_P_total"] += len(S)
                if self.info:
                    log.info("%s    |W| = %d > k|S| = %d -> P = S (sem redução)", ind, len(W), k * len(S))
                return set(S), W
            prev = Wi
        st["arestas_examinadas"] += arestas
        st["comparacoes_cotas"] += st["relaxacoes"] + st["re_relaxacoes_iguais"] - relax0

        children = {u: [] for u in W}
        has_parent = set()
        for v in W:
            u = pred[v]
            if u in children and d[v] == d[u] + pred_w[v] and h[v] == h[u] + 1:
                children[u].append(v)
                has_parent.add(v)

        P = set()
        for r in S:
            if r in has_parent:
                continue
            size, stack = 0, [r]
            while stack:
                x = stack.pop()
                size += 1
                stack.extend(children[x])
            if size >= k:
                P.add(r)
        st["findpivots_W_total"] += len(W)
        st["findpivots_P_total"] += len(P)
        st["comparacoes_cotas"] += len(W)
        st["adicoes"] += len(W)
        if self.info:
            log.info("%s    |W| = %d, pivôs P (árvores com >= k=%d vértices): %d de %d -> %s",
                     ind, len(W), k, len(P), len(S), self.fmt_set(P))
        return P, W

    def base_case(self, B, S, depth):
        """Algorithm 2, BaseCase(B, S), for S = {x}.

        Runs a small Dijkstra from x, with a binary heap and lazy deletion in place of
        DecreaseKey (stale entries are skipped when popped), relaxing only paths with
        d̂ < B, until k+1 vertices are settled or the heap empties.

        Returns (B', U):
            * if at most k vertices were settled: B' = B and U is all of them (success);
            * otherwise: B' is the largest d̂ among the k+1 settled vertices and U the
              vertices strictly below it (partial execution).

        With contar=True the heap items are Chave, so every `<` inside the heap is counted
        in comparacoes_heap. Counted bound tests: d̂[u]+w < B per valid relaxation and
        new value < queued value per re-queued vertex.
        """
        st = self.stats
        st["chamadas_base"] += 1
        ind = "│   " * depth
        d, h, pred, pred_w = self.d, self.h, self.pred, self.pred_w
        (x,) = S
        if self.info:
            log.info("%s  BaseCase(B=%s, S={%s}) -> mini-Dijkstra até k+1=%d vértices",
                     ind, self.fmt(B), self.g.name(x), self.k + 1)
        U0 = {x}
        push, pop = heap_ops(self.contar)
        heap = []
        kx = (d[x], h[x], x)
        push(heap, (kx, x))
        in_heap = {x: kx}
        done = set()
        adj = self.g.adj
        arestas = pushes = pops = cotas = 0
        n0 = Chave.n
        while heap and len(U0) < self.k + 1:
            kv, u = pop(heap)
            pops += 1
            if u in done or kv != (d[u], h[u], u):
                continue
            done.add(u)
            U0.add(u)
            if self.debug:
                log.debug("%s    extrai %s (d̂=%s)", ind, self.g.name(u), self.fmt(kv))
            adj_u = iter(adj[u])
            arestas += len(adj[u]) >> 1
            for v, w in zip(adj_u, adj_u):
                du, hu = d[u] + w, h[u] + 1
                if (du, hu, u) <= (d[v], h[v], pred[v]):
                    cotas += 1
                    if (du, hu, v) < B:
                        same = pred[v] == u and d[v] == du and h[v] == hu
                        d[v], h[v], pred[v], pred_w[v] = du, hu, u, w
                        st["re_relaxacoes_iguais" if same else "relaxacoes"] += 1
                        if self.debug:
                            log.debug("%s    relaxa %s -> %s (w=%g): d̂[%s] = %s%s", ind,
                                      self.g.name(u), self.g.name(v), w, self.g.name(v),
                                      self.fmt((du, hu, v)),
                                      "  (igual: re-relaxação, Remark 3.4)" if same else "")
                        kv2 = (du, hu, v)
                        old = in_heap.get(v)
                        if old is not None:
                            cotas += 1
                        if old is None or kv2 < old:
                            in_heap[v] = kv2
                            push(heap, (kv2, v))
                            pushes += 1
        st["arestas_examinadas"] += arestas
        st["comparacoes_cotas"] += cotas
        st["heap_pushes"] += pushes
        st["heap_pops"] += pops
        st["comparacoes_heap"] += Chave.n - n0
        if len(U0) <= self.k:
            if self.info:
                log.info("%s    -> sucesso: B' = B = %s, U = %s", ind, self.fmt(B), self.fmt_set(U0))
            return B, U0
        Bp = max((d[v], h[v], v) for v in U0)
        U = {v for v in U0 if (d[v], h[v], v) < Bp}
        st["comparacoes_cotas"] += 2 * len(U0)
        if self.info:
            log.info("%s    -> parcial: B' = %s, U = %s", ind, self.fmt(Bp), self.fmt_set(U))
        return Bp, U

    def bmssp(self, l, B, S, depth=0):
        """Algorithm 3, BMSSP(l, B, S) (Lemma 3.1): bounded multi-source shortest paths.

        At level 0 it delegates to base_case. Otherwise it calls find_pivots, inserts the
        pivots P into a structure D with M = 2^((l-1)t), and repeats while |U| < k·2^(lt)
        and D is not empty:
            1. Pull from D a set S_i with bound B_i;
            2. recurse BMSSP(l-1, B_i, S_i), getting (B'_i, U_i), and add U_i to U;
            3. relax the arcs out of U_i: values in [B_i, B) are Inserted into D, values in
               [B'_i, B_i) and the vertices of S_i in that range are BatchPrepended.
        Finally B' = min(B'_last, B) and U also takes the vertices of W with d̂ < B'.

        Returns (B', U); B' == B means a successful execution, B' < B a partial one caused by
        too large a workload. Each valid relaxation costs two bound comparisons (B_i <= val
        and either val < B or B'_i <= val), plus val < B_i for pairs that go to BatchPrepend.
        The per-level statistics in `niveis` are updated here.
        """
        st = self.stats
        st["chamadas_bmssp"] += 1
        lv = self.niveis.get(l)
        if lv is None:
            lv = self.niveis[l] = dict(chamadas=0, parciais=0, iteracoes=0, S_total=0, S_max=0,
                                       U_total=0, U_max=0, S_cota=2 ** (l * self.t),
                                       U_cota=4 * self.k * 2 ** (l * self.t))
        lv["chamadas"] += 1
        lv["S_total"] += len(S)
        if len(S) > lv["S_max"]:
            lv["S_max"] = len(S)
        ind = "│   " * depth
        d, h, pred, pred_w = self.d, self.h, self.pred, self.pred_w
        if self.info:
            log.info("%s┌ BMSSP(l=%d, B=%s, |S|=%d) S=%s", ind, l, self.fmt(B), len(S), self.fmt_set(S))
        if l == 0:
            Bp, U = self.base_case(B, S, depth)
            lv["parciais"] += Bp < B
            lv["U_total"] += len(U)
            if len(U) > lv["U_max"]:
                lv["U_max"] = len(U)
            if self.info:
                log.info("%s└ fim BMSSP(l=0): B'=%s, |U|=%d", ind, self.fmt(Bp), len(U))
            return Bp, U

        P, W = self.find_pivots(B, S, depth)
        M = 2 ** ((l - 1) * self.t)
        D = BlockDS(M, B, self.fmt, depth, self.contar)
        for x in P:
            D.insert(x, (d[x], h[x], x))
        Bp_last = min(((d[x], h[x], x) for x in P), default=B)
        U = set()
        limit = self.k * 2 ** (l * self.t)
        adj = self.g.adj
        arestas = 0
        st["comparacoes_cotas"] += len(P)
        if self.info:
            log.info("%s  D inicializada com M=2^((l-1)t)=%d; limite |U| < k·2^(lt)=%d; B'_0=%s",
                     ind, M, limit, self.fmt(Bp_last))

        i = 0
        while len(U) < limit and len(D) > 0:
            i += 1
            Bi, Si = D.pull()
            st["pulls"] += 1
            if self.info:
                log.info("%s  [iter %d] Pull -> B_%d=%s, |S_%d|=%d (restam %d em D)",
                         ind, i, i, self.fmt(Bi), i, len(Si), len(D))
            Bpi, Ui = self.bmssp(l - 1, Bi, set(Si), depth + 1)
            U |= Ui

            K = []
            n_ins = 0
            relax0 = st["relaxacoes"] + st["re_relaxacoes_iguais"]
            for u in Ui:
                adj_u = iter(adj[u])
                arestas += len(adj[u]) >> 1
                for v, w in zip(adj_u, adj_u):
                    du, hu = d[u] + w, h[u] + 1
                    if (du, hu, u) <= (d[v], h[v], pred[v]):
                        same = pred[v] == u and d[v] == du and h[v] == hu
                        d[v], h[v], pred[v], pred_w[v] = du, hu, u, w
                        st["re_relaxacoes_iguais" if same else "relaxacoes"] += 1
                        if self.debug:
                            log.debug("%s    relaxa %s -> %s (w=%g): d̂[%s] = %s%s", ind,
                                      self.g.name(u), self.g.name(v), w, self.g.name(v),
                                      self.fmt((du, hu, v)),
                                      "  (igual: re-relaxação, Remark 3.4)" if same else "")
                        val = (du, hu, v)
                        if Bi <= val < B:
                            n_ins += D.insert(v, val)
                        elif Bpi <= val < Bi:
                            K.append((v, val))
            st["comparacoes_cotas"] += 2 * (st["relaxacoes"] + st["re_relaxacoes_iguais"] - relax0) + len(K)
            K.extend((x, (d[x], h[x], x)) for x in Si if Bpi <= (d[x], h[x], x) < Bi)
            st["comparacoes_cotas"] += 2 * len(Si)
            n_pre = D.batch_prepend(K)
            if self.info:
                log.info("%s  [iter %d] B'_%d=%s, |U_%d|=%d, |U|=%d; Insert em D: %d, BatchPrepend: %d",
                         ind, i, i, self.fmt(Bpi), i, len(Ui), len(U), n_ins, n_pre)
            Bp_last = Bpi

        Bp = min(Bp_last, B)
        extra = {x for x in W if (d[x], h[x], x) < Bp} - U
        U |= extra
        st["arestas_examinadas"] += arestas
        st["comparacoes_cotas"] += len(W) + 1
        st["inserts"] += D.n_insert
        st["prepends"] += D.n_prepend
        lv["iteracoes"] += i
        lv["parciais"] += Bp < B
        lv["U_total"] += len(U)
        if len(U) > lv["U_max"]:
            lv["U_max"] = len(U)
        if self.info:
            tipo = "sucesso" if Bp == B else "parcial (carga grande)"
            log.info("%s└ fim BMSSP(l=%d): %s, B'=%s, |U|=%d (+%d de W)",
                     ind, l, tipo, self.fmt(Bp), len(U), len(extra))
        return Bp, U

    def run(self):
        """Call BMSSP(l, ∞, {s}) at the top level and return the distance list d.

        Afterwards `stats` holds the operations of the comparison-addition model (Section 2),
        counted exactly:
            * explicit: one addition and one comparison per examined arc (d̂[u]+w <= d̂[v]),
              one comparison per heap pop (stale-entry test), and the comparisons against
              the bounds B, B_i, B'_i plus the tight-arc test, gathered in comparacoes_cotas;
            * internal (only with contar=True): every `<` executed inside the heap
              (comparacoes_heap) and inside the structure D (comparacoes_D), through Chave,
              _bisect_contado and _particiona.
        comparacoes_total is the sum of the explicit and internal comparisons.
        """
        if self.info:
            log.info("Parâmetros: n=%d, m=%d, k=%d, t=%d, nível do topo l=%d",
                     self.g.n, self.g.m, self.k, self.t, self.top_level)
        n0 = Chave.n
        self.bmssp(self.top_level, KEY_INF, {self.s})
        st = self.stats
        st["adicoes"] += st["arestas_examinadas"]
        st["comparacoes_explicitas"] = st["arestas_examinadas"] + st["heap_pops"] + st["comparacoes_cotas"]
        st["comparacoes_D"] = Chave.n - n0 - st["comparacoes_heap"]
        st["comparacoes_total"] = st["comparacoes_explicitas"] + st["comparacoes_heap"] + st["comparacoes_D"]
        return self.d
